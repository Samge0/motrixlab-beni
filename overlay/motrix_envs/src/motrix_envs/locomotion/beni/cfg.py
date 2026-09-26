# Copyright (c) 2026 ported from Mondo-Robotics/open-beni (MIT)
# SPDX-License-Identifier: MIT
"""Shared configuration schema for BENI wheel-legged velocity tracking.

Ported from open-beni ``beni_rl/beni_velocity_flat_env_cfg.py`` (mjlab
manager-based) to the MotrixLab direct workflow. Reward semantics follow the
upstream mdp.py: exp-kernel velocity tracking gated by wheel support, posture,
leg symmetry, tilt/vertical-velocity penalties, wheel-overspeed termination.
"""

import math

from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.direct.env import DirectEnvCfg
from motrix_env_core.config.scene import SystemCameraCfg
from motrix_envs.config.scene import StandardSceneCfg, StandardSceneObjsCfg


@configclass
class ControlCfg:
    """Per-actuator action scales in BENI_ACTION_ORDER order."""

    leg_action_scale: float = 0.25
    wheel_action_scale: float = 20.0
    head_action_scale: float = 0.5


@configclass
class CommandsCfg:
    # Rows are min/max for [lin_vel_x, lin_vel_y, ang_vel_yaw].
    vel_limit: list[list[float]] = [
        [-1.0, 0.0, -2.0],
        [1.0, 0.0, 2.0],
    ]
    stand_prob: float = 0.25
    resampling_time: float = 5.0


@configclass
class NormalizationCfg:
    base_ang_vel: float = 0.25
    dof_pos: float = 1.0
    dof_vel: float = 0.05
    wheel_dof_vel: float = 0.01
    noise_dof_pos: float = 0.01
    noise_dof_vel: float = 0.1


@configclass
class AssetCfg:
    """Model element names needed by the BENI environment."""

    ground_geom_name: str = "floor"
    terminate_contact_geom_names: tuple[str, ...] = ()
    base_height_minimum: float = 0.11


@configclass
class RewardScales:
    termination_penalty: float = -10.0
    track_linear_velocity: float = 4.0
    track_linear_velocity_sharp: float = 2.0
    track_yaw_velocity: float = 4.0
    base_height: float = -20.0
    pose: float = -8.0
    leg_symmetry: float = -2.0
    base_tilt: float = -4.0
    vertical_velocity: float = -0.5
    roll_pitch_velocity: float = -0.02
    leg_action_rate: float = -0.05
    wheel_action_rate: float = -0.5
    undesired_contacts: float = -1.0
    alive: float = 2.0


@configclass
class RewardCfg:
    scales: RewardScales = RewardScales()
    tracking_sigma: float = 0.5
    tracking_sigma_sharp: float = math.sqrt(0.02)
    pose_weight: float = 0.05
    contact_threshold: float = 0.0  # GeomPairColliding is binary; unused.


@configclass
class BeniVelocityTrackingEnvCfg(DirectEnvCfg):
    """Robot-agnostic config consumed by ``BeniVelocityTrackingEnv``."""

    max_episode_seconds: float = 20.0
    scene: StandardSceneCfg = StandardSceneCfg(
        system_camera=SystemCameraCfg(distance=1.0, elevation=-15.0, azimuth=135.0),
        objs=StandardSceneObjsCfg(),
    )
    control_config: ControlCfg = ControlCfg()
    commands: CommandsCfg = CommandsCfg()
    normalization: NormalizationCfg = NormalizationCfg()
    asset: AssetCfg = AssetCfg()
    reward_config: RewardCfg = RewardCfg()
    wheel_max_speed: float = 350.0
    sim: SimCfg = SimCfg(dt=0.002, solver_iterations=3, solver_tolerance=1e-4)
    ctrl_dt: float = 0.01
    spawn_xy_range: float = 0.0
