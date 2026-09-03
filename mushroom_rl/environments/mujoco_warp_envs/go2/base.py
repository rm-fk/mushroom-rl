import os

import torch
import warp as wp

from mushroom_rl.core.spaces import Box
from mushroom_rl.environments.mujoco import ObservationType
from mushroom_rl.environments.mujoco_envs import __file__ as path_robots
from mushroom_rl.environments.mujoco_warp import MuJoCoWarp


class Go2Base(MuJoCoWarp):
    """
    Base class for the Unitree Go2 quadruped in MuJoCo Warp.

    Holds everything that is a property of the robot rather than of a task:
    the model, the joint and actuator specification, the mapping from policy
    actions to joint torques, and the health check used for termination.

    Task classes derive from this and supply reward, is_absorbing and the
    task specific parts of the observation. See Go2Walk.

    """

    # Joint order as declared in the menagerie MJCF. This is also the action
    # order. It does NOT necessarily match the ordering used by the Unitree
    # SDK; reconcile at deployment time, not here.
    LEG_JOINTS = [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ]

    ACTUATORS = [
        "FL_hip",
        "FL_thigh",
        "FL_calf",
        "FR_hip",
        "FR_thigh",
        "FR_calf",
        "RL_hip",
        "RL_thigh",
        "RL_calf",
        "RR_hip",
        "RR_thigh",
        "RR_calf",
    ]

    FEET = ["FL", "FR", "RL", "RR"]

    def __init__(
        self,
        num_envs,
        gamma=0.99,
        horizon=1000,
        healthy_z_range=(0.18, 0.45),
        healthy_gravity_z=-0.6,
        terminate_when_unhealthy=True,
        action_scale=0.25,
        kp=25.0,
        kd=0.5,
        soft_joint_limit=0.9,
        reset_noise_scale=0.05,
        n_substeps=1,
        n_intermediate_steps=10,
        use_graph_capture=False,
        nconmax=None,
        njmax=None,
        scene="scene_mjx.xml",
        **viewer_params,
    ):
        """
        Constructor.

        Args:
            num_envs (int): number of parallel environments;
            healthy_z_range (tuple): min and max base height, in metres, for
                which the robot is considered healthy. The nominal standing
                height is 0.27;
            healthy_gravity_z (float): upper bound on the z component of the
                gravity vector expressed in the base frame. Equals -1 when the
                robot is perfectly upright and 0 when it is on its side;
            action_scale (float): scaling from policy action to joint position
                offset relative to the default pose;
            kp (float): proportional gain of the joint position controller;
            kd (float): derivative gain of the joint position controller;
            soft_joint_limit (float): fraction of the joint range, centred on
                its midpoint, outside of which a limit penalty may apply;
            reset_noise_scale (float): scale of the uniform noise added to the
                default joint pose on reset;
            n_substeps (int): physics steps per intermediate step;
            n_intermediate_steps (int): intermediate steps per environment
                step. The PD controller is evaluated once per intermediate
                step, so this is what sets the control rate. Together with
                n_substeps and the 0.002 s model timestep, the defaults give a
                500 Hz control loop and a 50 Hz policy rate. Running the PD at
                the policy rate instead leaves the joints underdamped and the
                robot oscillates itself over;
            scene (str): scene file to load. The mjx variant uses a reduced
                solver iteration count and carries IMU sensors, which suits
                batched GPU simulation.

        """
        xml_path = os.path.join(
            os.path.dirname(os.path.abspath(path_robots)), "data", "go2", scene
        )

        observation_spec = [("base_pose", "base", ObservationType.JOINT_POS)]
        observation_spec += [
            (f"{j}_pos", j, ObservationType.JOINT_POS) for j in self.LEG_JOINTS
        ]
        observation_spec += [("base_vel", "base", ObservationType.JOINT_VEL)]
        observation_spec += [
            (f"{j}_vel", j, ObservationType.JOINT_VEL) for j in self.LEG_JOINTS
        ]

        additional_data_spec = [
            ("base_pos", "base", ObservationType.BODY_POS),
            ("base_rot", "base", ObservationType.BODY_ROT),
            ("base_vel_world", "base", ObservationType.BODY_VEL_WORLD),
            ("base_vel_local", "base", ObservationType.BODY_VEL),
        ]
        additional_data_spec += [
            (f"{f}_pos", f"{f}_calf", ObservationType.BODY_POS) for f in self.FEET
        ]

        self._healthy_z_range = healthy_z_range
        self._healthy_gravity_z = healthy_gravity_z
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._action_scale = action_scale
        self._kp = kp
        self._kd = kd
        self._soft_joint_limit = soft_joint_limit
        self._reset_noise_scale = reset_noise_scale

        # Must precede super().__init__(): the parent calls _modify_mdp_info
        # partway through its own constructor, and subclasses use this there.
        self._n_joints = len(self.LEG_JOINTS)

        super().__init__(
            num_envs=num_envs,
            xml_file=xml_path,
            gamma=gamma,
            horizon=horizon,
            observation_spec=observation_spec,
            actuation_spec=self.ACTUATORS,
            additional_data_spec=additional_data_spec,
            n_substeps=n_substeps,
            n_intermediate_steps=n_intermediate_steps,
            use_graph_capture=use_graph_capture,
            nconmax=nconmax,
            njmax=njmax,
            **viewer_params,
        )

        self._device = wp.to_torch(self._data_wp.qpos).device

        # Nominal standing pose, taken from the "home" keyframe of the MJCF.
        # qpos0 has the legs fully extended and the base at 0.445, which is
        # not a pose the robot can hold; the keyframe is the one to use.
        key_qpos = torch.as_tensor(
            self._model.key_qpos[0], dtype=torch.float32, device=self._device
        )
        self._default_qpos = key_qpos
        self._default_joint_pos = key_qpos[7:].clone()

        # Leg dofs in qvel, after the six free joint dofs.
        self._leg_dof_idx = torch.arange(6, 6 + self._n_joints, device=self._device)

        # Joint limits from the model, shrunk towards the midpoint by
        # soft_joint_limit. Joint 0 is the free joint and is skipped.
        rng = torch.as_tensor(
            self._model.jnt_range[1:], dtype=torch.float32, device=self._device
        )
        mid = 0.5 * (rng[:, 0] + rng[:, 1])
        half = 0.5 * (rng[:, 1] - rng[:, 0]) * soft_joint_limit
        self._joint_lower = mid - half
        self._joint_upper = mid + half

    # ------------------------------------------------------------------
    # MDP info / observation layout
    # ------------------------------------------------------------------

    def _modify_mdp_info(self, mdp_info):
        # The policy must not see absolute world position: x and y of the
        # free joint carry no task information and prevent generalisation.
        self.obs_helper.remove_obs("base_pose", 0)
        self.obs_helper.remove_obs("base_pose", 1)

        # Contiguous slices of the joint positions and velocities in the
        # observation, for use by subclasses. Computed after remove_obs so
        # the indices are final.
        first_pos = self.obs_helper.obs_idx_map[f"{self.LEG_JOINTS[0]}_pos"][0]
        last_pos = self.obs_helper.obs_idx_map[f"{self.LEG_JOINTS[-1]}_pos"][-1]
        first_vel = self.obs_helper.obs_idx_map[f"{self.LEG_JOINTS[0]}_vel"][0]
        last_vel = self.obs_helper.obs_idx_map[f"{self.LEG_JOINTS[-1]}_vel"][-1]
        self._joint_pos_slice = slice(first_pos, last_pos + 1)
        self._joint_vel_slice = slice(first_vel, last_vel + 1)

        mdp_info = super()._modify_mdp_info(mdp_info)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    # ------------------------------------------------------------------
    # Action mapping
    # ------------------------------------------------------------------

    def _preprocess_action(self, action):
        # Keep the policy action as is. The PD loop lives in _compute_action
        # so that reward() sees the raw action rather than the torques.
        return torch.as_tensor(action, dtype=torch.float32, device=self._device)

    def _compute_action(self, obs, action):
        """
        Map a policy action to joint torques through a PD controller.

        The action is a joint position offset relative to the default pose,
        as in the usual legged locomotion setup. The MJCF ships plain torque
        actuators, so the position loop is closed here rather than by MuJoCo.

        Overriding this method makes the base class recompute the torques
        once per intermediate step. Moving the PD loop into the MJCF as
        <position> actuators would run it inside the solver and inside the
        captured graph, which removes n_intermediate_steps Python calls per
        environment step. That is the recommended upgrade before scaling to
        large num_envs.

        """
        qpos = wp.to_torch(self._data_wp.qpos)
        qvel = wp.to_torch(self._data_wp.qvel)

        joint_pos = qpos[:, 7:]
        joint_vel = qvel[:, self._leg_dof_idx]

        target = self._default_joint_pos + self._action_scale * action
        return self._kp * (target - joint_pos) - self._kd * joint_vel

    # ------------------------------------------------------------------
    # Robot state helpers
    # ------------------------------------------------------------------

    def _projected_gravity(self):
        """
        Gravity direction expressed in the base frame.

        Returns a (num_envs, 3) tensor. The z component is -1 when the robot
        is upright and rises towards 0 as it tips onto its side, which makes
        it a convenient orientation health check and a standard policy input.

        """
        quat = self._read_data("base_rot")
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        # Third row of the transpose of the rotation matrix, applied to
        # (0, 0, -1). Equivalent to R^T @ g_world with |g| = 1.
        return torch.stack(
            [
                -2.0 * (x * z - w * y),
                -2.0 * (y * z + w * x),
                -(1.0 - 2.0 * (x * x + y * y)),
            ],
            dim=-1,
        )

    def _base_lin_vel(self):
        """Linear velocity of the base, in the base frame. (num_envs, 3)"""
        return self._read_data("base_vel_local")[:, 3:]

    def _base_ang_vel(self):
        """Angular velocity of the base, in the base frame. (num_envs, 3)"""
        return self._read_data("base_vel_local")[:, :3]

    def _joint_pos(self):
        return wp.to_torch(self._data_wp.qpos)[:, 7:]

    def _joint_vel(self):
        return wp.to_torch(self._data_wp.qvel)[:, self._leg_dof_idx]

    def _joint_torque(self):
        return wp.to_torch(self._data_wp.ctrl)

    # ------------------------------------------------------------------
    # Health / termination
    # ------------------------------------------------------------------

    def _is_finite(self, obs):
        qpos = wp.to_torch(self._data_wp.qpos)
        qvel = wp.to_torch(self._data_wp.qvel)
        return torch.isfinite(torch.cat([qpos, qvel], dim=1)).all(dim=1)

    def _is_within_z_range(self, obs):
        min_z, max_z = self._healthy_z_range
        z = self._read_data("base_pos")[:, 2]
        return (z >= min_z) & (z <= max_z)

    def _is_upright(self, obs):
        return self._projected_gravity()[:, 2] <= self._healthy_gravity_z

    def _is_healthy(self, obs):
        return (
            self._is_finite(obs) & self._is_within_z_range(obs) & self._is_upright(obs)
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def setup(self, env_indices, obs):
        """
        Reset the given environments to the default standing pose with a small
        amount of joint noise.

        """
        super().setup(env_indices, obs)

        qpos = wp.to_torch(self._data_wp.qpos)
        qvel = wp.to_torch(self._data_wp.qvel)

        idx = (
            env_indices.to(qpos.device).long()
            if isinstance(env_indices, torch.Tensor)
            else torch.as_tensor(env_indices, device=qpos.device, dtype=torch.long)
        )

        n = len(idx)
        if n == 0:
            self._mj_warp.forward(self._model_wp, self._data_wp)
            return

        qpos[idx] = self._default_qpos
        qvel[idx] = 0.0

        noise = (
            torch.rand(n, self._n_joints, device=qpos.device) * 2 - 1
        ) * self._reset_noise_scale
        qpos[idx, 7:] += noise

        self._mj_warp.forward(self._model_wp, self._data_wp)

    def get_states(self):
        qpos = wp.to_torch(self._data_wp.qpos)
        qvel = wp.to_torch(self._data_wp.qvel)
        return torch.cat([qpos, qvel], dim=1)
