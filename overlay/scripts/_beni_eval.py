# Copyright (c) 2026 ported from Mondo-Robotics/open-beni (MIT)
# SPDX-License-Identifier: MIT
"""Headless eval for beni-velocity-flat: pinned command battery.

Measures per-command velocity-tracking error, episode survival, base height,
and tilt over fixed command grids (zero / +-x / +-yaw / combos), using the
deterministic policy from a checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import motrix_envs  # noqa: F401
from motrix_env_core import registry
from motrix_rl import checkpoints, runner, runs


def load_agent(run_dir: Path, ckpt: Path, motrixlab_root: Path):
    result = runs.find_metadata_for_policy(ckpt)
    if result is None:
        raise FileNotFoundError(f"no metadata for {ckpt}")
    meta_run_dir, meta = result
    # New MotrixLab schemas add MISSING-mandatory FastSAC fields; fill them
    # from the installed checkout's resolved algo_base (see _fastsac_compat).
    try:
        from _fastsac_compat import resolved_fastsac_async_options

        cfg_override = {
            "trainer": {"async_options": resolved_fastsac_async_options(motrixlab_root)}
        }
    except Exception:
        cfg_override = None  # older MotrixLab: snapshot alone is sufficient
    handle = runner.create_run_handle(
        runs.open_run_context(meta_run_dir, meta), cfg_override=cfg_override
    )
    trainer = handle.trainer
    from motrix_env_core.renderer import RenderConfig

    env = trainer._make_env(1, render=RenderConfig(), mode="play")
    agent = trainer._make_agent(env)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    agent.load_state_dict(state)
    return env, agent


def run_case(env, agent, cmd: np.ndarray, steps: int = 600):
    """Run one pinned-command episode; cmd = (vx, vy, wz)."""
    obs, _ = env.reset()
    inner = env._env
    # Command pinning works on both MotrixLab generations: new ArrayEnvState
    # has no info dict (commands live on env._commands); old ones use info.
    def _pin():
        if hasattr(inner, "_commands"):
            inner._commands[:] = cmd
        else:
            inner._state.info["commands"][:] = cmd

    _pin()
    errs_lin, errs_ang, heights, tilts = [], [], [], []
    steps_alive = 0
    for _ in range(steps):
        with torch.no_grad():
            action = agent.act(obs, deterministic=True)
        obs, _, _, terminated, truncated = env.step(action)
        _pin()  # Keep the command pinned across auto-resets.
        if bool(terminated[0]):
            break
        steps_alive += 1
        m = inner._state.metrics
        errs_lin.append(float(m["lin_vel_error"]))
        errs_ang.append(float(m["ang_vel_error"]))
        heights.append(float(m["base_height"]))
        tilts.append(float(m["tilt"]))
    return {
        "cmd": cmd.tolist(),
        "survival": steps_alive / steps,
        "lin_err": float(np.mean(errs_lin)) if errs_lin else float("nan"),
        "ang_err": float(np.mean(errs_ang)) if errs_ang else float("nan"),
        "height": float(np.mean(heights)) if heights else float("nan"),
        "tilt": float(np.mean(tilts)) if tilts else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run dir under runs/beni-velocity-flat/...")
    parser.add_argument("--ckpt", default="latest.pt")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--motrixlab", default=".", help="MotrixLab checkout root (for schema compat)")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = Path("runs/beni-velocity-flat/motrix/torch/fastsac") / run_dir
    ckpt = run_dir / "checkpoints" / args.ckpt

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    env, agent = load_agent(run_dir, ckpt, Path(args.motrixlab).resolve())
    print(f"loaded {ckpt}")

    cases = [
        ("stand  ", np.array([0.0, 0.0, 0.0])),
        ("fwd 0.5", np.array([0.5, 0.0, 0.0])),
        ("fwd 1.0", np.array([1.0, 0.0, 0.0])),
        ("back 0.5", np.array([-0.5, 0.0, 0.0])),
        ("yaw +1", np.array([0.0, 0.0, 1.0])),
        ("yaw -1", np.array([0.0, 0.0, -1.0])),
        ("fwd+turn", np.array([0.5, 0.0, 1.0])),
        ("fwd-turn", np.array([0.5, 0.0, -1.0])),
    ]
    print(f"{'case':10s} {'survive':>8s} {'lin_err':>8s} {'ang_err':>8s} {'height':>7s} {'tilt':>7s}")
    for name, cmd in cases:
        r = run_case(env, agent, cmd, args.steps)
        print(
            f"{name:10s} {r['survival']:8.2f} {r['lin_err']:8.3f} {r['ang_err']:8.3f} "
            f"{r['height']:7.3f} {r['tilt']:7.3f}"
        )


if __name__ == "__main__":
    main()
