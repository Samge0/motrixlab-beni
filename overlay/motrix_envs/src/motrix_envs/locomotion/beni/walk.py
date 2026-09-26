# Copyright (c) 2026 ported from Mondo-Robotics/open-beni (MIT)
# SPDX-License-Identifier: MIT
"""BENI wheel-legged velocity tracking environment (direct workflow).

Port of open-beni's mjlab manager-based velocity env to MotrixLab's
``DirectEnv``. The robot is driven through seven deployment channels in
upstream BENI_ACTION_ORDER ``(L1, R1, L2, R2, LW, RW, head_yaw)``:

- legs (L1/R1/L2/R2): position actuators, target = default + 0.25 * a
- wheels (LW/RW): velocity actuators, target = 20.0 * a (rad/s)
- head (head_yaw): position actuator, target = default + 0.5 * a

All other XML actuators (vel/tor mirrors, head_roll group) receive fixed
neutral targets every step: position/velocity mirrors at the default value
(zero force at default state) and torque mirrors at zero torque.

Compatibility: targets MotrixLab main-branch ArrayEnvState (reward_terms /
metrics fields; per-env custom state lives on the env object, not in an
info dict).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from motrix_env_core.array.env import NpObs
from motrix_env_core.base import ObsSpace
from motrix_env_core.direct.env import ArrayEnvState, DirectEnv
from motrix_env_core.math import quaternion
from motrix_env_core.sim import (
    BodyAngularVelocityWrite,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    GeomPairCollidingQuery,
    GeomSpecsQuery,
    JointPositionQuery,
    JointVelocityQuery,
    LinkAngularVelocityQuery,
    LinkLinearVelocityQuery,
    LinkPositionQuery,
    LinkQuaternionQuery,
)
try:  # MotrixLab >= 0.4.x moved ActuatorType/ActuatorSpec to sim.model
    from motrix_env_core.sim.model import ActuatorType
except ImportError:  # older layouts kept them in sim.backend
    from motrix_env_core.sim.backend import ActuatorType
from motrix_env_core.sim.write import CtrlTargetsWrite, JointPositionWrite, JointVelocityWrite
from motrix_envs.locomotion.beni.cfg import BeniVelocityTrackingEnvCfg
from motrix_envs.robot import Beni

# Ordered joint names behind the deployment action.
_ACTION_JOINTS = ("L1", "R1", "L2", "R2", "LW", "RW", "head_yaw_joint")
# XML actuator names for the deployment channels.
_POSITION_ACTUATORS = ("L1_pos", "R1_pos", "L2_pos", "R2_pos", "head_yaw_pos")
_VELOCITY_ACTUATORS = ("LW", "RW")
# Observation joint orders from upstream beni_robot.py.
_POS_OBS_JOINTS = ("L1", "R1", "head_yaw_joint", "L2", "R2")
_VEL_OBS_JOINTS = ("L1", "R1", "head_yaw_joint", "L2", "R2", "LW", "RW")
# Default pose (upstream BENI_DEFAULT_JOINT_POS over the action joints).
_DEFAULT_JOINT_POS = np.asarray([0.3, 0.3, 0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
_POS_OBS_DEFAULT = np.asarray([0.3, 0.3, 0.0, 0.5, 0.5], dtype=np.float32)
# Joint ranges from the MJCF, used for action bounds.
_LEG_RANGES = ((-2.5307, 6.8940), (-0.7853981633974483, 1.1344640137963142))
_HEAD_RANGE = (-5.2, 5.2)
# Upstream passive knee spring linearized on the standing range [0, 1.134]:
# tau(theta) = 0.22*sin(theta-1) + 0.15*sin(2*(theta-1))
#            ~= 0.3636 * (theta + 0.3848)
_KNEE_SPRING_K = 0.3636
_KNEE_SPRING_REF = -0.3848
_KNEE_ACTUATOR_KP = 0.5  # from the MJCF position actuator kp


@dataclass
class BeniQuantities:
    """Batched physical quantities used by observations and rewards."""

    base_pos: np.ndarray
    base_quat: np.ndarray
    base_lin_vel: np.ndarray
    base_ang_vel: np.ndarray
    projected_gravity: np.ndarray
    joint_pos: np.ndarray  # _ACTION_JOINTS order
    joint_vel: np.ndarray  # _ACTION_JOINTS order
    illegal_contact: np.ndarray  # (N,) bool


class BeniVelocityTrackingEnv(DirectEnv[BeniVelocityTrackingEnvCfg]):
    """Velocity tracking for the BENI two-wheel-legged robot."""

    def __init__(self, cfg: BeniVelocityTrackingEnvCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs, backend=backend)
        robot = cfg.scene.objs.robot
        if not isinstance(robot, Beni):
            raise TypeError(f"beni scene robot must be Beni, got {type(robot).__name__}")
        self._robot_cfg = robot
        self._base_link_name = robot.resolved_base_link_name
        asset = cfg.asset
        ground_geom = asset.ground_geom_name
        termination_geoms = tuple(name for name in asset.terminate_contact_geom_names if name != ground_geom)
        self.model = self.sim.compile_model({"geoms": GeomSpecsQuery(names=termination_geoms + (ground_geom,))})
        self._validate_model_contract()

        self._termination_geoms = termination_geoms
        self._num_termination_pairs = len(termination_geoms)
        queries = {
            "robot_joint_pos": JointPositionQuery(joints=_ACTION_JOINTS),
            "robot_joint_vel": JointVelocityQuery(joints=_ACTION_JOINTS),
            "pos_obs_joints": JointPositionQuery(joints=_POS_OBS_JOINTS),
            "vel_obs_joints": JointVelocityQuery(joints=_VEL_OBS_JOINTS),
            "base_pos": LinkPositionQuery(link=self._base_link_name),
            "base_quat": LinkQuaternionQuery(link=self._base_link_name),
            "base_lin_vel": LinkLinearVelocityQuery(link=self._base_link_name),
            "base_ang_vel": LinkAngularVelocityQuery(link=self._base_link_name),
            "termination_colliding": GeomPairCollidingQuery(
                pairs=tuple((name, ground_geom) for name in termination_geoms)
            ),
        }
        self._termination_query = queries["termination_colliding"]
        self.sim_data = self.sim.compile_reads(queries)

        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        self._reset_program = self.sim.write_compiler.compile(
            {
                "base_position": BodyPositionWrite((self._base_link_name,)),
                "base_rotation": BodyRotationWrite((self._base_link_name,)),
                "base_linear_velocity": BodyLinearVelocityWrite((self._base_link_name,)),
                "base_angular_velocity": BodyAngularVelocityWrite((self._base_link_name,)),
                "joints_position": JointPositionWrite(_ACTION_JOINTS),
                "joints_velocity": JointVelocityWrite(_ACTION_JOINTS),
            },
            reset=True,
        )
        self._reset_position = self._reset_program.buffer("base_position")[:, 0]
        self._reset_rotation = self._reset_program.buffer("base_rotation")[:, 0]
        self._reset_linear_velocity = self._reset_program.buffer("base_linear_velocity")[:, 0]
        self._reset_angular_velocity = self._reset_program.buffer("base_angular_velocity")[:, 0]
        self._reset_joint_position = self._reset_program.buffer("joints_position")
        self._reset_joint_velocity = self._reset_program.buffer("joints_velocity")

        self._num_action = len(_ACTION_JOINTS)
        self.gravity_vec = np.array([0, 0, -1], dtype=np.float32)
        self._init_base_pose = self.model.init_dof_pos[:7].copy()
        self._actuator_index = {act.name: i for i, act in enumerate(self.model.actuators)}
        self._default_ctrl = self._build_default_ctrl()
        self._scales = self._build_action_scales()
        self._init_obs_space()
        self._init_action_space()
        self._resample_steps = max(int(round(cfg.commands.resampling_time / cfg.ctrl_dt)), 1)
        self._ctrl_buffer = self._ctrl_writes.buffer("ctrl")
        self._ctrl_buffer[:] = self._default_ctrl
        # Per-env custom state lives on the env (ArrayEnvState has no info dict).
        self._commands = np.zeros((self._num_envs, 3), dtype=np.float32)
        self._current_actions = np.zeros((self._num_envs, self._num_action), dtype=np.float32)
        self._last_actions = np.zeros((self._num_envs, self._num_action), dtype=np.float32)

    # ------------------------------------------------------------------ setup

    def _validate_model_contract(self) -> None:
        by_name = {act.name: act for act in self.model.actuators}
        missing = sorted(set(_POSITION_ACTUATORS + _VELOCITY_ACTUATORS).difference(by_name))
        if missing:
            raise KeyError(f"beni model is missing actuators: {missing}")
        for name in _POSITION_ACTUATORS:
            if by_name[name].actuator_type is not ActuatorType.POSITION:
                raise TypeError(f"beni actuator {name!r} must be a position actuator")
        for name in _VELOCITY_ACTUATORS:
            if by_name[name].actuator_type is not ActuatorType.VELOCITY:
                raise TypeError(f"beni actuator {name!r} must be a velocity actuator")

    def _build_default_ctrl(self) -> np.ndarray:
        """Neutral ctrl targets for every actuator.

        Position channels get the default joint target; velocity mirrors get
        the default velocity (0); torque mirrors get 0 torque.
        """
        joint_default = dict(zip(_ACTION_JOINTS, _DEFAULT_JOINT_POS.tolist()))
        defaults: dict[str, float] = {}
        for act in self.model.actuators:
            if act.name in _POSITION_ACTUATORS:
                defaults[act.name] = joint_default[act.target_name]
            elif act.name == "head_roll_pos":
                defaults[act.name] = 0.0
            else:
                defaults[act.name] = 0.0
        return np.asarray([defaults[act.name] for act in self.model.actuators], dtype=np.float32)

    def _build_action_scales(self) -> np.ndarray:
        cfg = self.cfg.control_config
        return np.asarray(
            [cfg.leg_action_scale] * 4 + [cfg.wheel_action_scale] * 2 + [cfg.head_action_scale],
            dtype=np.float32,
        )

    def _init_obs_space(self) -> None:
        # actor: ang_vel(3) gravity(3) cmd(3) joint_pos(5) joint_vel(7) last_action(7)
        actor_dim = 3 + 3 + 3 + len(_POS_OBS_JOINTS) + len(_VEL_OBS_JOINTS) + self._num_action
        critic_dim = actor_dim + 3  # + base lin vel
        self._observation_space = ObsSpace(
            policy=gym.spaces.Box(-np.inf, np.inf, (actor_dim,), dtype=np.float32),
            value=gym.spaces.Box(-np.inf, np.inf, (critic_dim,), dtype=np.float32),
        )

    def _init_action_space(self) -> None:
        leg_lo, leg_hi = _LEG_RANGES
        leg_res = max(
            abs(leg_lo[0] - 0.3), abs(leg_hi[0] - 0.3), abs(leg_lo[1] - 0.5), abs(leg_hi[1] - 0.5)
        )
        bounds = np.asarray(
            [leg_res / 0.25] * 4
            + [self.cfg.wheel_max_speed / 20.0] * 2
            + [max(abs(_HEAD_RANGE[0]), _HEAD_RANGE[1]) / 0.5],
            dtype=np.float32,
        )
        self._action_space = gym.spaces.Box(-bounds, bounds, dtype=np.float32)

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> ObsSpace:
        return self._observation_space

    # ------------------------------------------------------------------ step

    def apply_action(self, actions: np.ndarray, state: ArrayEnvState) -> ArrayEnvState:
        steps = state.episode_steps
        due = (steps % self._resample_steps == 0) & (steps > 0)
        if np.any(due):
            self._commands[due] = self.resample_commands(int(due.sum()))
        self._last_actions = self._current_actions
        self._current_actions = np.asarray(actions, dtype=np.float32).copy()
        return state

    def physics_step(self) -> None:
        actions = self._current_actions
        ctrl = self._ctrl_buffer
        ctrl[:] = self._default_ctrl
        targets = _DEFAULT_JOINT_POS + actions * self._scales  # (N, 7)
        # Upstream passive knee spring, injected as a position-actuator ctrl
        # offset (MotrixSim ignores joint spring XML attributes).
        knee_pos = self.sim_data["robot_joint_pos"][:, 2:4]  # L2, R2
        tau_spring = _KNEE_SPRING_K * (knee_pos - _KNEE_SPRING_REF)
        targets[:, 2:4] += tau_spring / _KNEE_ACTUATOR_KP
        # ctrl columns follow the model's canonical actuator order.
        for column, actuator_name in enumerate((*_POSITION_ACTUATORS, *_VELOCITY_ACTUATORS)):
            ctrl[:, self._actuator_index[actuator_name]] = targets[:, column]
        self._ctrl_writes.execute()
        self.sim.step(self._cfg.sim_substeps)

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        state = self.update_terminated(state)
        q = self._state_quantities(state)
        self._wheel_overspeed = np.any(
            np.abs(q.joint_vel[:, 4:6]) > self.cfg.wheel_max_speed, axis=1
        )
        state = state.replace(terminated=state.terminated | self._wheel_overspeed)
        return self.update_reward(state, q)

    def update_terminated(self, state: ArrayEnvState) -> ArrayEnvState:
        terminated = np.zeros((self._num_envs,), dtype=bool)
        if self._num_termination_pairs:
            colliding = self.sim_data["termination_colliding"]
            terminated |= colliding.any(axis=1)
        return state.replace(terminated=terminated)

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        inputs = self.sim_data
        nrm = self.cfg.normalization
        rows = slice(None)
        base_quat = inputs["base_quat"][rows]
        ang_vel = quaternion.rotate_inverse(base_quat, inputs["base_ang_vel"][rows]) * nrm.base_ang_vel
        gravity = quaternion.rotate_inverse(base_quat, self.gravity_vec)
        lin_vel = quaternion.rotate_inverse(base_quat, inputs["base_lin_vel"][rows]) * 2.0
        cmd = self._commands
        pos_obs = (inputs["pos_obs_joints"][rows] - _POS_OBS_DEFAULT) * nrm.dof_pos
        vel_obs = inputs["vel_obs_joints"][rows] * nrm.dof_vel
        actions = self._current_actions
        pos_noisy = pos_obs + np.random.uniform(-1, 1, pos_obs.shape).astype(np.float32) * nrm.noise_dof_pos
        vel_noisy = vel_obs + np.random.uniform(-1, 1, vel_obs.shape).astype(np.float32) * nrm.noise_dof_vel
        actor = np.hstack(
            [ang_vel, gravity, cmd[:, :2], cmd[:, 2:3], pos_noisy, vel_noisy, actions]
        ).astype(np.float32)
        critic = np.hstack([lin_vel, actor]).astype(np.float32)
        return state.replace(obs=NpObs(policy=actor, value=critic))

    # ------------------------------------------------------------------ helpers

    def _state_quantities(self, state: ArrayEnvState) -> BeniQuantities:
        rows = slice(None)
        inputs = self.sim_data
        base_quat = inputs["base_quat"][rows]
        illegal = (
            inputs["termination_colliding"].any(axis=1)
            if self._num_termination_pairs
            else np.zeros((self._num_envs,), dtype=bool)
        )
        return BeniQuantities(
            base_pos=inputs["base_pos"][rows],
            base_quat=base_quat,
            base_lin_vel=quaternion.rotate_inverse(base_quat, inputs["base_lin_vel"][rows]),
            base_ang_vel=quaternion.rotate_inverse(base_quat, inputs["base_ang_vel"][rows]),
            projected_gravity=quaternion.rotate_inverse(base_quat, self.gravity_vec),
            joint_pos=inputs["robot_joint_pos"][rows],
            joint_vel=inputs["robot_joint_vel"][rows],
            illegal_contact=illegal,
        )

    def resample_commands(self, num_envs: int) -> np.ndarray:
        limits = np.asarray(self.cfg.commands.vel_limit, dtype=np.float32)
        commands = np.random.uniform(low=limits[0], high=limits[1], size=(num_envs, 3)).astype(np.float32)
        stand = np.random.uniform(size=(num_envs,)) < self.cfg.commands.stand_prob
        commands[stand] = 0.0
        return commands

    def reset(self, env_ids: np.ndarray) -> None:
        num_reset = len(env_ids)
        pose = np.broadcast_to(self._init_base_pose, (num_reset, 7)).copy()
        pose[:, :2] += np.random.uniform(-0.1, 0.1, size=(num_reset, 2)).astype(np.float32)
        self._reset_position[env_ids] = pose[:, :3]
        self._reset_rotation[env_ids] = pose[:, 3:7]
        self._reset_linear_velocity[env_ids] = 0.0
        self._reset_angular_velocity[env_ids] = 0.0
        joints = np.broadcast_to(_DEFAULT_JOINT_POS, (num_reset, self._num_action)).copy()
        joints[:, :4] += np.random.uniform(-0.05, 0.05, size=(num_reset, 4)).astype(np.float32)
        self._reset_joint_position[env_ids] = joints
        self._reset_joint_velocity[env_ids] = 0.0
        self._reset_program.execute(env_ids)
        self.sim_data.execute(np.asarray(env_ids, dtype=np.int64))
        self._commands[env_ids] = self.resample_commands(num_reset)
        self._current_actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0

    # ------------------------------------------------------------------ reward

    def update_reward(self, state: ArrayEnvState, q: BeniQuantities) -> ArrayEnvState:
        scales = self.cfg.reward_config.scales
        terms = self._get_reward(q, state)
        weighted = {
            name: (value * getattr(scales, name) * self.cfg.ctrl_dt).astype(np.float32)
            for name, value in terms.items()
        }
        state.reward_terms = weighted
        cmd = self._commands
        state.metrics = {
            "lin_vel_error": np.mean(np.linalg.norm(cmd[:, :2] - q.base_lin_vel[:, :2], axis=1)),
            "ang_vel_error": np.mean(np.abs(cmd[:, 2] - q.base_ang_vel[:, 2])),
            "base_height": np.mean(q.base_pos[:, 2]),
            "tilt": np.mean(
                np.arctan2(
                    np.linalg.norm(q.projected_gravity[:, :2], axis=1), -q.projected_gravity[:, 2]
                )
            ),
            "wheel_speed": np.mean(np.abs(q.joint_vel[:, 4:6])),
        }
        reward = np.asarray(sum(weighted.values()), dtype=np.float32)
        return state.replace(reward=reward)

    def _get_reward(self, q: BeniQuantities, state: ArrayEnvState) -> dict[str, np.ndarray]:
        cfg = self.cfg.reward_config
        cmd = self._commands
        sigma = cfg.tracking_sigma
        sharp = cfg.tracking_sigma_sharp

        lin_err = np.linalg.norm(cmd[:, :2] - q.base_lin_vel[:, :2], axis=1)
        ang_err = np.abs(cmd[:, 2] - q.base_ang_vel[:, 2])
        track_lin = np.exp(-(lin_err**2) / sigma**2)
        track_lin_sharp = np.exp(-(lin_err**2) / sharp**2)
        track_yaw = np.exp(-(ang_err**2) / sigma**2)

        # Posture: squared distance of the five position-obs joints to default.
        pos_obs = self.sim_data["pos_obs_joints"][slice(None)]
        pose_err = np.sum(np.square(pos_obs - _POS_OBS_DEFAULT), axis=1)
        # Leg symmetry L1/R1 and L2/R2 (columns 0,2 vs 1,3).
        sym = np.sum(np.square(q.joint_pos[:, 0:4:2] - q.joint_pos[:, 1:4:2]), axis=1)

        tilt = np.arctan2(
            np.linalg.norm(q.projected_gravity[:, :2], axis=1), -q.projected_gravity[:, 2]
        )
        # Gate positive rewards on upright, non-contacting support (upstream
        # standing_support uses wheel contact + height + tilt; tilt-only proxy).
        gate = (tilt < math.pi / 6).astype(np.float32) * (~q.illegal_contact).astype(np.float32)

        # Base-height shortfall below the wheel-supported pose.
        height_short = np.clip(self.cfg.asset.base_height_minimum - q.base_pos[:, 2], 0.0, None)

        actions = self._current_actions
        last_actions = self._last_actions
        leg_rate = np.sum(np.square(actions[:, 0:4] - last_actions[:, 0:4]), axis=1)
        wheel_rate = np.sum(np.square(actions[:, 4:6] - last_actions[:, 4:6]), axis=1)

        return {
            "termination_penalty": state.terminated.astype(np.float32),
            "track_linear_velocity": track_lin * gate,
            "track_linear_velocity_sharp": track_lin_sharp * gate,
            "track_yaw_velocity": track_yaw * gate,
            "base_height": height_short,
            "pose": pose_err,
            "leg_symmetry": sym,
            "base_tilt": tilt,
            "vertical_velocity": np.square(q.base_lin_vel[:, 2]),
            "roll_pitch_velocity": np.sum(np.square(q.base_ang_vel[:, :2]), axis=1),
            "leg_action_rate": leg_rate,
            "wheel_action_rate": wheel_rate,
            "undesired_contacts": q.illegal_contact.astype(np.float32),
            "alive": np.ones_like(leg_rate),
        }
