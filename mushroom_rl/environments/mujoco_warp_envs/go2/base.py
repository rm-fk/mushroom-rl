import os

import numpy as np
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
    actions to joint torques, the health check used for termination, and the
    domain randomisation applied on reset and during episodes.

    Follows the structure of the IsaacSim A1Walking environment, which in
    turn resembles Rudin et al., "Learning to Walk in Minutes Using Massively
    Parallel Deep Reinforcement Learning". Task classes derive from this and
    supply reward, is_absorbing and the task specific observations.

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

    # Foot geoms, used for the contact proxy. Spheres of radius 0.022.
    FEET_GEOMS = ["FL", "FR", "RL", "RR"]

    def __init__(
        self,
        num_envs,
        gamma=0.99,
        horizon=1000,
        healthy_z_range=(0.15, 0.45),
        healthy_gravity_z=-0.6,
        terminate_when_unhealthy=True,
        action_scale=0.25,
        kp=20.0,
        kd=0.5,
        soft_joint_limit=0.9,
        domain_randomization=True,
        push_interval=750,
        push_max_vel=1.0,
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
            domain_randomization (bool): whether to randomise the initial pose
                and apply random pushes during episodes;
            push_interval (int): mean number of steps between random pushes;
            push_max_vel (float): magnitude bound of the velocity applied by a
                push, in m/s;
            n_substeps (int): physics steps per intermediate step;
            n_intermediate_steps (int): intermediate steps per environment
                step. The PD controller is evaluated once per intermediate
                step, so this sets the control rate. With the 0.002 s model
                timestep the defaults give a 500 Hz control loop and a 50 Hz
                policy rate. Running the PD at the policy rate instead leaves
                the joints underdamped and the robot oscillates itself over;
            scene (str): scene file to load. The mjx variant uses a reduced
                solver iteration count and carries IMU sensors, which suits
                batched GPU simulation.

        """
        xml_path = os.path.join(
            os.path.dirname(os.path.abspath(path_robots)), "data", "go2", scene
        )

        # Base velocity in the base frame, then joint positions and
        # velocities. World position and orientation are deliberately not
        # observed: the policy sees orientation only through the projected
        # gravity vector, which is what an IMU provides on the real robot.
        observation_spec = [("base_vel", "base", ObservationType.BODY_VEL)]
        observation_spec += [
            (f"{j}_pos", j, ObservationType.JOINT_POS) for j in self.LEG_JOINTS
        ]
        observation_spec += [
            (f"{j}_vel", j, ObservationType.JOINT_VEL) for j in self.LEG_JOINTS
        ]

        additional_data_spec = [
            ("base_pos", "base", ObservationType.BODY_POS),
            ("base_rot", "base", ObservationType.BODY_ROT),
        ]

        self._healthy_z_range = healthy_z_range
        self._healthy_gravity_z = healthy_gravity_z
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._action_scale = action_scale
        self._kp = kp
        self._kd = kd
        self._soft_joint_limit = soft_joint_limit
        self._domain_randomization = domain_randomization
        self._push_interval = push_interval
        self._push_max_vel = push_max_vel

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
        dev = self._device

        # Nominal standing pose, taken from the "home" keyframe of the MJCF.
        # qpos0 has the legs fully extended and the base at 0.445, which is
        # not a pose the robot can hold; the keyframe is the one to use.
        key_qpos = torch.as_tensor(
            self._model.key_qpos[0], dtype=torch.float32, device=dev
        )
        self._default_qpos = key_qpos
        self._default_joint_pos = key_qpos[7:].clone()

        # Leg dofs in qvel, after the six free joint dofs.
        self._leg_dof_idx = torch.arange(6, 6 + self._n_joints, device=dev)

        # Joint limits from the model. Joint 0 is the free joint and is
        # skipped. The soft limits are shrunk towards the midpoint.
        rng = torch.as_tensor(
            self._model.jnt_range[1:], dtype=torch.float32, device=dev
        )
        self._joint_lower = rng[:, 0]
        self._joint_upper = rng[:, 1]
        mid = 0.5 * (rng[:, 0] + rng[:, 1])
        half = 0.5 * (rng[:, 1] - rng[:, 0]) * soft_joint_limit
        self._soft_joint_lower = mid - half
        self._soft_joint_upper = mid + half

        # Torque limits from the actuator ctrlrange.
        self._effort_limit = torch.as_tensor(
            self._model.actuator_ctrlrange[:, 1], dtype=torch.float32, device=dev
        )

        self._feet_geom_ids = torch.as_tensor(
            [self._model.geom(g).id for g in self.FEET_GEOMS], device=dev
        )

        # The action is a joint position offset scaled by action_scale, so
        # the action space is the joint range expressed in those units.
        low = (self._joint_lower - self._default_joint_pos) / action_scale
        high = (self._joint_upper - self._default_joint_pos) / action_scale
        self.info.action_space = Box(low, high)

        self._actions = torch.zeros(num_envs, self._n_joints, device=dev)
        self._episode_length = torch.zeros(num_envs, dtype=torch.long, device=dev)

    # ------------------------------------------------------------------
    # MDP info / observation layout
    # ------------------------------------------------------------------

    def _modify_mdp_info(self, mdp_info):
        # Contiguous slices of the joint positions and velocities in the
        # observation, for use by subclasses.
        first_pos = self.obs_helper.obs_idx_map[f"{self.LEG_JOINTS[0]}_pos"][0]
        last_pos = self.obs_helper.obs_idx_map[f"{self.LEG_JOINTS[-1]}_pos"][-1]
        first_vel = self.obs_helper.obs_idx_map[f"{self.LEG_JOINTS[0]}_vel"][0]
        last_vel = self.obs_helper.obs_idx_map[f"{self.LEG_JOINTS[-1]}_vel"][-1]
        self._joint_pos_slice = slice(first_pos, last_pos + 1)
        self._joint_vel_slice = slice(first_vel, last_vel + 1)

        # BODY_VEL is ordered [angular(3), linear(3)].
        base_vel = self.obs_helper.obs_idx_map["base_vel"]
        self._ang_vel_slice = slice(base_vel[0], base_vel[0] + 3)
        self._lin_vel_slice = slice(base_vel[0] + 3, base_vel[0] + 6)

        mdp_info = super()._modify_mdp_info(mdp_info)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    # ------------------------------------------------------------------
    # Action mapping
    # ------------------------------------------------------------------

    def _preprocess_action(self, action):
        action = torch.as_tensor(action, dtype=torch.float32, device=self._device)
        action = torch.clamp(action, min=-100.0, max=100.0)
        self._actions[:] = action
        return action

    def _compute_action(self, obs, action):
        """
        Map a policy action to joint torques through a PD controller.

        The action is a joint position offset relative to the default pose.
        The MJCF ships plain torque actuators, so the position loop is closed
        here rather than by MuJoCo. Overriding this method makes the base
        class recompute the torques once per intermediate step.

        Moving the PD loop into the MJCF as <position> actuators would run it
        inside the solver and inside the captured graph, which removes
        n_intermediate_steps Python calls per environment step. That is the
        recommended upgrade before scaling to large num_envs.

        """
        joint_pos = self._joint_pos()
        joint_vel = self._joint_vel()

        target = self._default_joint_pos + self._action_scale * action
        torque = self._kp * (target - joint_pos) - self._kd * joint_vel
        return torch.clamp(torque, -self._effort_limit, self._effort_limit)

    # ------------------------------------------------------------------
    # Quaternion helpers (w, x, y, z), batched over environments
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_rotate(q, v):
        w, xyz = q[:, :1], q[:, 1:]
        c = torch.cross(xyz, v, dim=-1)
        return v + 2.0 * w * c + 2.0 * torch.cross(xyz, c, dim=-1)

    @staticmethod
    def _quat_rotate_inverse(q, v):
        w, xyz = q[:, :1], q[:, 1:]
        c = torch.cross(xyz, v, dim=-1)
        return v - 2.0 * w * c + 2.0 * torch.cross(xyz, c, dim=-1)

    @staticmethod
    def _wrap_to_pi(angles):
        return torch.atan2(torch.sin(angles), torch.cos(angles))

    # ------------------------------------------------------------------
    # Robot state helpers
    # ------------------------------------------------------------------

    def _projected_gravity(self):
        """
        Gravity direction expressed in the base frame. (num_envs, 3)

        The z component is -1 when the robot is upright and rises towards 0
        as it tips onto its side.

        """
        quat = self._read_data("base_rot")
        g = torch.tensor([0.0, 0.0, -1.0], device=self._device).expand(quat.shape[0], 3)
        return self._quat_rotate_inverse(quat, g)

    def _heading(self):
        """Yaw angle of the base forward axis, in radians. (num_envs,)"""
        quat = self._read_data("base_rot")
        fwd = torch.tensor([1.0, 0.0, 0.0], device=self._device).expand(
            quat.shape[0], 3
        )
        fwd = self._quat_rotate(quat, fwd)
        return torch.atan2(fwd[:, 1], fwd[:, 0])

    def _joint_pos(self):
        return wp.to_torch(self._data_wp.qpos)[:, 7:]

    def _joint_vel(self):
        return wp.to_torch(self._data_wp.qvel)[:, self._leg_dof_idx]

    def _joint_torque(self):
        return wp.to_torch(self._data_wp.ctrl)

    def _foot_height(self):
        """Height of each foot geom centre above the ground. (num_envs, 4)"""
        return wp.to_torch(self._data_wp.geom_xpos)[:, self._feet_geom_ids, 2]

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
    # Reset and domain randomisation
    # ------------------------------------------------------------------

    def setup(self, env_indices, obs):
        """
        Reset the given environments to the default standing pose. With domain
        randomisation the joint angles are scaled by a random factor in
        [0.5, 1.5] and the base is given a random initial velocity, as in the
        A1Walking reference.

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

        if self._domain_randomization:
            factors = torch.rand(n, self._n_joints, device=qpos.device) + 0.5
            qpos[idx, 7:] = self._default_joint_pos * factors
            qvel[idx, :6] = torch.rand(n, 6, device=qpos.device) - 0.5

        self._actions[idx] = 0.0
        self._episode_length[idx] = 0

        self._mj_warp.forward(self._model_wp, self._data_wp)

    def _step_finalize(self):
        self._episode_length += 1

        if self._domain_randomization:
            do_push = (
                torch.rand(self.number, device=self._device) < 1.0 / self._push_interval
            )
            do_push &= self._episode_length > 50
            self._push_robots(torch.nonzero(do_push, as_tuple=True)[0])

    def _push_robots(self, env_indices):
        """Overwrite the base planar velocity of the given environments."""
        if len(env_indices) == 0:
            return
        qvel = wp.to_torch(self._data_wp.qvel)
        vel = (
            torch.rand(len(env_indices), 2, device=self._device) * 2 - 1
        ) * self._push_max_vel
        qvel[env_indices, 0:2] = vel

    def get_states(self):
        qpos = wp.to_torch(self._data_wp.qpos)
        qvel = wp.to_torch(self._data_wp.qvel)
        return torch.cat([qpos, qvel], dim=1)
