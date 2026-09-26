# Copyright (c) 2026 ported from Mondo-Robotics/open-beni (MIT)
# SPDX-License-Identifier: MIT

"""BENI wheel-legged robot configuration, ported from open-beni.

Upstream: https://github.com/Mondo-Robotics/open-beni (MIT). The MJCF is
kept close to upstream with two classes of edits:

- meshdir/texturedir rewritten to be relative to the asset directory
  (upstream points at sibling ``../meshes`` directories);
- the actuator table is reduced to the seven deployment actuators in
  BENI_ACTION_ORDER (upstream also ships _vel/_tor mirrors of every joint,
  which mjlab deletes in code via ``get_beni_spec``).
"""

from pathlib import Path

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import KeyPoseCfg, MjcfFileCfg
from motrix_envs.robot.humanoid import HumanoidRobotCfg

BENI_ASSET_DIR = Path(__file__).parent / "assets" / "beni"
_BENI_MJCF = BENI_ASSET_DIR / "beni.xml"

# Deployment action order from upstream beni_rl/beni_robot.py.
BENI_ACTION_ORDER = ("L1", "R1", "L2", "R2", "LW", "RW", "head_yaw")
# Joint-position observation order from upstream (5 joints).
BENI_POSITION_OBS_JOINTS = ("L1", "R1", "head_yaw_joint", "L2", "R2")
# Joint-velocity observation order from upstream (all 7 actuated joints).
BENI_VELOCITY_OBS_JOINTS = ("L1", "R1", "head_yaw_joint", "L2", "R2", "LW", "RW")
BENI_LEG_JOINTS = ("L1", "R1", "L2", "R2")


@configclass(kw_only=True)
class Beni(HumanoidRobotCfg):
    """Mondo Robotics BENI two-wheel-legged robot (7 actuators)."""

    model: MjcfFileCfg = MjcfFileCfg(file=_BENI_MJCF)
    base_link_name: str = "base_link"
    # The wheels are the feet; foot-link semantics are used only for
    # reward/termination bookkeeping, never ground contact.
    left_foot_link_name: str = "LW_Link"
    right_foot_link_name: str = "RW_Link"
    key_pose: KeyPoseCfg = KeyPoseCfg(
        joint_names=[
            "L1",
            "R1",
            "L2",
            "R2",
            "LW",
            "RW",
            "head_yaw_joint",
            "head_roll_joint",
        ],
        poses={
            # Upstream BENI_DEFAULT_JOINT_POS: standing wheel-supported pose.
            "default": [0.3, 0.3, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        },
    )
