# Copyright (c) 2026 ported from Mondo-Robotics/open-beni (MIT)
# SPDX-License-Identifier: MIT
"""Record a headless demo video of beni-velocity-flat with a checkpoint.

Follow camera: the system camera is re-aimed at the robot's live base
position every control step, so the robot stays framed while driving.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import motrix_envs  # noqa: F401
from motrix_env_core.renderer import RenderConfig
from motrix_rl import runner, runs

# Each phase: (seconds, vx, vy, wz)
PHASES = [
    (2.0, 0.0, 0.0, 0.0),
    (3.0, 0.6, 0.0, 0.0),
    (3.0, 0.6, 0.0, 1.0),
    (3.0, -0.4, 0.0, 0.0),
    (3.0, 0.0, 0.0, 1.5),
    (2.0, 0.0, 0.0, 0.0),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--ckpt", default="model_0008000.pt")
    parser.add_argument("--out", default="deliverables/beni-velocity-demo.mp4")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--motrixlab", default=".", help="MotrixLab checkout root (for schema compat)")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = Path("runs/beni-velocity-flat/motrix/torch/fastsac") / run_dir
    ckpt = run_dir / "checkpoints" / args.ckpt
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix(".raw.mp4")

    result = runs.find_metadata_for_policy(ckpt)
    meta_run_dir, meta = result
    # Schema compat: fill new mandatory FastSAC fields from the installed
    # checkout's resolved algo_base (no-op / skipped on older MotrixLab).
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _fastsac_compat import resolved_fastsac_async_options

        cfg_override = {
            "trainer": {
                "async_options": resolved_fastsac_async_options(
                    Path(args.motrixlab).resolve()
                )
            }
        }
    except Exception:
        cfg_override = None
    render = RenderConfig(
        headless=True, path=str(raw.resolve()), fps=60, num_frames=16 * 60, width=1280, height=720
    )
    handle = runner.create_run_handle(
        runs.open_run_context(meta_run_dir, meta), cfg_override=cfg_override, render=render
    )
    trainer = handle.trainer
    env = trainer._make_env(args.num_envs, render=render, mode="play")
    agent = trainer._make_agent(env)
    agent.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))

    obs, _ = env.reset()
    inner = env._env
    ctrl_dt = inner.cfg.ctrl_dt
    stand = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    renderer = env._renderer  # wrap-owned VideoRecorder
    sys_cam = renderer._frames._render.system_camera

    def follow_camera() -> None:
        base_pos = inner.sim_data["base_pos"][0]
        sys_cam.set_view(
            [float(base_pos[0]), float(base_pos[1]), 0.22],
            0.95,
            -15.0,
            120.0,
        )

    def _pin(cmd: np.ndarray) -> None:
        if hasattr(inner, "_commands"):
            inner._commands[:] = cmd
        else:
            inner._state.info["commands"][:] = cmd

    follow_camera()
    for seconds, vx, vy, wz in PHASES:
        steps = int(round(seconds / ctrl_dt))
        cmd = np.array([vx, vy, wz], dtype=np.float32)
        _pin(cmd)
        for _ in range(steps):
            with torch.no_grad():
                action = agent.act(obs, deterministic=True)
            obs, _, _, terminated, truncated = env.step(action)
            _pin(cmd)
            follow_camera()
            if env.render() is False:
                break
    print("recorded")
    ffmpeg = Path.home() / "scoop/apps/ffmpeg/current/bin/ffmpeg.exe"
    subprocess.run(
        [
            str(ffmpeg), "-y", "-i", str(raw), "-c:v", "libx264", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out),
        ],
        check=True,
        capture_output=True,
    )
    print(f"final -> {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
