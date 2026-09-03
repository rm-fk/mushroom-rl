import torch

from mushroom_rl.core.spaces import Box

from .base import Go2Base


class Go2Walk(Go2Base):
    """
    Velocity tracking task for the Unitree Go2.

    Each episode samples a target base velocity (forward, lateral, yaw rate)
    and the policy is rewarded for tracking it while keeping the motion
    smooth and within joint limits. This is the standard legged locomotion
    formulation used by legged_gym and unitree_rl_gym, with the terms that
    require contact information (feet air time, collision penalties) left
    out until batched contact queries are available.

    Note that the weights below are per environment step, whereas legged_gym
    scales its reward terms by dt. Their published values are therefore not
    directly comparable; the ratios between terms are what carries over.

    """

    def __init__(
        self,
        num_envs,
        lin_vel_x_range=(-1.0, 1.0),
        lin_vel_y_range=(-0.5, 0.5),
        ang_vel_yaw_range=(-1.0, 1.0),
        command_deadband=0.2,
        tracking_sigma=0.25,
        tracking_lin_vel_weight=1.0,
        tracking_ang_vel_weight=0.5,
        lin_vel_z_weight=2.0,
        ang_vel_xy_weight=0.05,
        orientation_weight=0.0,
        torque_weight=2e-4,
        joint_acc_weight=2.5e-7,
        action_rate_weight=0.01,
        joint_limit_weight=10.0,
        **kwargs,
    ):
        """
        Constructor.

        Args:
            lin_vel_x_range (tuple): sampling range of the forward velocity
                command, in m/s;
            lin_vel_y_range (tuple): sampling range of the lateral velocity
                command, in m/s;
            ang_vel_yaw_range (tuple): sampling range of the yaw rate command,
                in rad/s;
            command_deadband (float): commands with norm below this value are
                set to zero, so the robot also learns to stand still;
            tracking_sigma (float): width of the exponential tracking kernel;
            *_weight (float): weights of the reward terms. Penalty weights are
                positive and subtracted.

        """
        self._lin_vel_x_range = lin_vel_x_range
        self._lin_vel_y_range = lin_vel_y_range
        self._ang_vel_yaw_range = ang_vel_yaw_range
        self._command_deadband = command_deadband
        self._tracking_sigma = tracking_sigma

        self._w_tracking_lin = tracking_lin_vel_weight
        self._w_tracking_ang = tracking_ang_vel_weight
        self._w_lin_vel_z = lin_vel_z_weight
        self._w_ang_vel_xy = ang_vel_xy_weight
        self._w_orientation = orientation_weight
        self._w_torque = torque_weight
        self._w_joint_acc = joint_acc_weight
        self._w_action_rate = action_rate_weight
        self._w_joint_limit = joint_limit_weight

        super().__init__(num_envs, **kwargs)

        # Per environment task state. Commands are resampled on reset; the
        # previous action and joint velocity are needed for rate penalties.
        self._commands = torch.zeros(num_envs, 3, device=self._device)
        self._prev_action = torch.zeros(num_envs, self._n_joints, device=self._device)
        self._prev_joint_vel = torch.zeros(
            num_envs, self._n_joints, device=self._device
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _modify_mdp_info(self, mdp_info):
        mdp_info = super()._modify_mdp_info(mdp_info)
        # Appended in _create_observation, in this order.
        self.obs_helper.add_obs("command", 3)
        self.obs_helper.add_obs("projected_gravity", 3, -1.0, 1.0)
        self.obs_helper.add_obs("prev_action", self._n_joints)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _create_observation(self, obs):
        obs = obs.clone()
        # Joint positions relative to the default pose, so that the standing
        # configuration reads as zero regardless of the absolute angles.
        obs[:, self._joint_pos_slice] -= self._default_joint_pos
        return torch.cat(
            [obs, self._commands, self._projected_gravity(), self._prev_action],
            dim=1,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def setup(self, env_indices, obs):
        super().setup(env_indices, obs)

        idx = (
            env_indices.to(self._device).long()
            if isinstance(env_indices, torch.Tensor)
            else torch.as_tensor(env_indices, device=self._device, dtype=torch.long)
        )
        n = len(idx)
        if n == 0:
            return

        self._commands[idx] = self._sample_commands(n)
        self._prev_action[idx] = 0.0
        self._prev_joint_vel[idx] = 0.0

    def _sample_commands(self, n):
        def uniform(lo, hi):
            return torch.rand(n, device=self._device) * (hi - lo) + lo

        cmd = torch.stack(
            [
                uniform(*self._lin_vel_x_range),
                uniform(*self._lin_vel_y_range),
                uniform(*self._ang_vel_yaw_range),
            ],
            dim=1,
        )
        # Small commands are zeroed so the policy also learns to stand still
        # rather than jittering in place chasing tiny targets.
        small = cmd[:, :2].norm(dim=1) < self._command_deadband
        cmd[small] = 0.0
        return cmd

    # ------------------------------------------------------------------
    # Step bookkeeping
    # ------------------------------------------------------------------

    def _step_init(self, obs, action):
        # Joint velocity before the step, for the acceleration penalty.
        self._prev_joint_vel = self._joint_vel().clone()

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _reward_terms(self, action):
        """
        Compute every reward component, unweighted, as a dict of (num_envs,)
        tensors. Kept separate from reward() so the same values can be
        reported through _create_info_dictionary.

        """
        lin_vel = self._base_lin_vel()
        ang_vel = self._base_ang_vel()
        gravity = self._projected_gravity()
        joint_pos = self._joint_pos()
        joint_vel = self._joint_vel()
        torque = self._joint_torque()

        cmd = self._commands

        lin_err = ((cmd[:, :2] - lin_vel[:, :2]) ** 2).sum(dim=1)
        ang_err = (cmd[:, 2] - ang_vel[:, 2]) ** 2

        joint_acc = (joint_vel - self._prev_joint_vel) / self.dt

        below = torch.clamp(self._joint_lower - joint_pos, min=0.0)
        above = torch.clamp(joint_pos - self._joint_upper, min=0.0)

        return {
            "tracking_lin_vel": torch.exp(-lin_err / self._tracking_sigma),
            "tracking_ang_vel": torch.exp(-ang_err / self._tracking_sigma),
            "lin_vel_z": lin_vel[:, 2] ** 2,
            "ang_vel_xy": (ang_vel[:, :2] ** 2).sum(dim=1),
            "orientation": (gravity[:, :2] ** 2).sum(dim=1),
            "torque": (torque**2).sum(dim=1),
            "joint_acc": (joint_acc**2).sum(dim=1),
            "action_rate": ((action - self._prev_action) ** 2).sum(dim=1),
            "joint_limit": (below + above).sum(dim=1),
        }

    def reward(self, obs, action, next_obs, absorbing):
        t = self._reward_terms(action)

        r = (
            self._w_tracking_lin * t["tracking_lin_vel"]
            + self._w_tracking_ang * t["tracking_ang_vel"]
            - self._w_lin_vel_z * t["lin_vel_z"]
            - self._w_ang_vel_xy * t["ang_vel_xy"]
            - self._w_orientation * t["orientation"]
            - self._w_torque * t["torque"]
            - self._w_joint_acc * t["joint_acc"]
            - self._w_action_rate * t["action_rate"]
            - self._w_joint_limit * t["joint_limit"]
        )

        self._prev_action = action.clone()
        return r

    def is_absorbing(self, obs):
        return self._terminate_when_unhealthy & ~self._is_healthy(obs)

    def _create_info_dictionary(self, obs):
        t = self._reward_terms(self._prev_action)
        t["command_x"] = self._commands[:, 0]
        t["command_y"] = self._commands[:, 1]
        t["command_yaw"] = self._commands[:, 2]
        return t
