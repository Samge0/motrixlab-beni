# Copyright (c) 2026 ported from Mondo-Robotics/open-beni (MIT)
# SPDX-License-Identifier: MIT

"""BENI wheel-legged velocity-tracking presets and registration."""

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_envs.config.scene import StandardSceneCfg, StandardSceneObjsCfg
from motrix_envs.locomotion.beni import cfg as beni_cfg
from motrix_envs.locomotion.beni.walk import BeniVelocityTrackingEnv
from motrix_envs.robot import Beni

# Bodies whose contact with the ground terminates the episode (upstream
# _ILLEGAL_BODY_PATTERNS = base_link, [LR][12]_Link, head). Named geoms from
# the collision group of the ported MJCF.
_BENI_TERMINATION_GEOMS = (
    "base_collision",
    "head_collision",
    "L1_collision",
    "R1_collision",
)


@registry.envcfg("beni-velocity-flat")
def make_beni_velocity_flat_cfg() -> beni_cfg.BeniVelocityTrackingEnvCfg:
    """Track twist commands with BENI on flat ground.

    zh_CN: 控制 BENI 轮腿机器人在平地上跟踪速度指令。
    """

    return beni_cfg.BeniVelocityTrackingEnvCfg(
        scene=StandardSceneCfg(objs=StandardSceneObjsCfg(robot=Beni())),
        commands=beni_cfg.CommandsCfg(
            vel_limit=[
                [-1.0, 0.0, -2.0],
                [1.0, 0.0, 2.0],
            ],
            stand_prob=0.25,
            resampling_time=5.0,
        ),
        asset=beni_cfg.AssetCfg(
            ground_geom_name="floor",
            terminate_contact_geom_names=_BENI_TERMINATION_GEOMS,
            base_height_minimum=0.11,
        ),
        sim=SimCfg(dt=0.002, solver_iterations=3, solver_tolerance=1e-4),
        ctrl_dt=0.01,
        max_episode_seconds=20.0,
    )


registry.env("beni-velocity-flat")(BeniVelocityTrackingEnv)
