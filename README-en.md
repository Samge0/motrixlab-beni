# BENI velocity-flat for MotrixLab

> [🇨🇳 中文文档](README.md) | 🇺🇸 English (this file)

A BENI two-wheel-legged robot velocity-tracking task for MotrixLab,
ported from Mondo-Robotics/open-beni (mjlab, MIT) to the MotrixLab
direct-workflow (FastSAC).

- Upstream project: https://github.com/Mondo-Robotics/open-beni (MIT)
- Target framework: https://github.com/Motphys/MotrixLab (Apache-2.0)
- Verified against: MotrixLab 0.4.0b0 and main @ 5bd0054, Python 3.10,
  CUDA 12.x, a single consumer GPU (8 GB, alongside other GPU jobs)

## Contents

```
overlay/          Files laid out with MotrixLab-relative paths. Copy into
                  your MotrixLab checkout root (or run install.py).
  motrix_envs/src/motrix_envs/robot/assets/beni/
                    MJCF + baked meshes + textures (MIT, from open-beni)
  motrix_envs/src/motrix_envs/robot/beni.py
                    RobotCfg (7 deployment actuators, official default pose)
  motrix_envs/src/motrix_envs/locomotion/beni/
                    cfg.py (reward/termination schema)
                    walk.py (DirectEnv: obs 28/31, act 7, knee spring)
                    beni_task.py (registry wiring, "beni-velocity-flat")
  configs/task/beni-velocity-flat/motrix.fastsac.yaml
  runs/beni-velocity-flat/.../      pretrained run (metadata-backed; best
                    checkpoint = model_0008000.pt)
  scripts/_beni_*.py                eval / record / plot / bake helper scripts
install.py        Copies overlay/ into a MotrixLab checkout and patches the
                  two registration __init__.py files (idempotent).
verify_install.py Post-install smoke check (env construct + 50 random steps).
LICENSE           MIT (port) with upstream notices.
NOTICE            Attribution details.
```

## About the pretrained runs/ folder

`overlay/runs/` ships a trained run so you can evaluate immediately, but the
checkpoints are ~16 MB each and do NOT belong in a git repository. When
publishing this port:

- keep `runs/` out of version control (add `runs/` to .gitignore; the
  upstream MotrixLab repo already ignores it since commit 5bd0054), and
- attach the `runs/beni-velocity-flat/` folder as a release asset
  (zip/tar.gz) on GitHub, then instruct users to unpack it into their
  checkout root. `verify_install.py` auto-detects any run timestamp;
  eval/record take `--run <RUN_DIR>`.

## Install (no clone of open-beni needed)

1. Prerequisites: a working MotrixLab checkout with its virtualenv built
   (see the MotrixLab README for environment setup).

2. From this folder:

       python install.py --motrixlab /path/to/MotrixLab

   This copies overlay/* into the checkout and appends the two registration
   lines (robot + task imports). Idempotent: safe to re-run; never
   overwrites files that are not part of this port.

3. Verify:

       # Linux / macOS
       /path/to/MotrixLab/.venv/bin/python verify_install.py --motrixlab /path/to/MotrixLab

       # Windows
       F:\path\to\MotrixLab\.venv\Scripts\python.exe verify_install.py --motrixlab F:\path\to\MotrixLab

## Train from scratch

From the MotrixLab checkout root:

    # Linux / macOS
    .venv/bin/python scripts/train.py task=beni-velocity-flat/motrix.fastsac

    # Windows: upstream fastsac needs a small sched shim (see Known
    # environment notes below)
    set TORCH_COMPILE_DISABLE=1
    set PYTHONPATH=overlay\scripts;%PYTHONPATH%
    .venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'overlay/scripts'); import _win_sched_compat; exec(open('scripts/train.py',encoding='utf-8').read())" -- --task=beni-velocity-flat/motrix.fastsac

Defaults: 2048 envs, 10000 iterations, checkpoint every 1000.
Measured on one RTX 2080 Ti-class GPU (learner ~300 MB VRAM, physics on CPU,
another GPU job running concurrently): ~4200 env-steps/s, ~1h30m wall
clock. The episode-length curve typically saturates the 20 s cap by
iteration ~4000. Pin the GPU with CUDA_VISIBLE_DEVICES if the machine has
other GPU jobs.

## Evaluate a pretrained checkpoint (pinned command battery)

From the MotrixLab checkout root (`RUN_DIR` is the timestamped folder under
runs/beni-velocity-flat/motrix/torch/fastsac/):

    .venv/bin/python scripts/_beni_eval.py \
        --run <RUN_DIR> --ckpt model_0008000.pt --steps 600

    # Windows: .venv\Scripts\python.exe scripts\_beni_eval.py ...

Plays 8 pinned commands (stand / fwd / back / yaw / combos), reports
survival, linear/yaw tracking error, base height, tilt.

Record a follow-camera demo video:

    .venv/bin/python scripts/_beni_record.py \
        --run <RUN_DIR> --ckpt model_0008000.pt \
        --out deliverables/demo.mp4

## Task summary

| Item | Value |
|---|---|
| Robot | BENI, 7 deployment actuators (L1 R1 L2 R2 LW RW head_yaw): legs = position targets (default + 0.25*a), wheels = velocity targets (20*a rad/s), head = position (0.5*a) |
| Observation | actor 28 (ang_vel 3, gravity 3, cmd 3, joint_pos 5, joint_vel 7, last_action 7); critic = actor + base lin vel (31) |
| Commands | vx [-1,1] m/s, wz [-2,2] rad/s, 25% stand, resample 5 s |
| Episode | 20 s; sim dt 0.002, ctrl dt 0.01 |
| Termination | illegal body-ground contact, wheel overspeed > 350 rad/s |
| Reward | exp-kernel lin/yaw tracking (sigma 0.5 + sharp sqrt(0.02)), posture-to-default (weight -8), leg symmetry, tilt, vertical velocity, action rates, contact penalties, alive bonus, termination penalty -10. All positive terms gated on upright (<30 deg) and no illegal contact |

Passive knee spring (ported from upstream BeniMotor, linearized on the
standing range): `tau = 0.3636 * (theta + 0.3848)`, injected as a position
actuator ctrl offset in walk.py. Note: MotrixSim IGNORES joint
stiffness/springref XML attributes, so the spring must stay in code.

## Reference results (single consumer GPU, seed 1, model_0008000.pt)

- 8/8 command battery 100% survival over 6 s episodes
- linear tracking error 0.025-0.106 m/s, height 0.134-0.137 m
- stance joints near official defaults (L1/R1 0.37/0.46, L2/R2 0.60/0.55
  vs official 0.3/0.5)
- known quirk: fwd+left-turn yaw tracking is poor (2.03 rad/s error) while
  fwd+right-turn is fine (0.13) -- see "improvement ideas"

## Improvement ideas

- stronger/direct yaw tracking reward or symmetric augmentation for the
  fwd+left-turn quirk
- exact nonlinear knee spring (per-substep compute) instead of the
  standing-range linearization
- port upstream fallen-reset curriculum (supine/prone/side resets)

## Known environment notes

- Verified against MotrixLab main @ 5bd0054 (2026-09-25) and 0.4.0b0-era
  checkouts. walk.py imports ActuatorType from `sim.model` with a fallback
  to the older `sim.backend` location, so both generations work.
- Upstream commit c58cf54 (NUMA binding) introduced `os.sched_getaffinity`
  calls without a win32 guard: fastsac training on Windows currently needs
  the tiny shim shipped as `overlay/scripts/_win_sched_compat.py`. Import it
  before motrix_rl when training on Windows, or run on Linux. A minimal
  patch is worth contributing upstream.
- MotrixSim ignores joint `stiffness/springref` XML attributes (the knee
  spring is therefore code-injected in walk.py) and multi-layer `<layer>`
  material PBR (rgb texture is applied via the classic attribute form).

## License

MIT for this port. BENI assets and reward design originate from
Mondo-Robotics/open-beni (MIT). MotrixLab itself is Apache-2.0; overlay
files only touch the extension points MotrixLab provides for new
robots/tasks and do not modify MotrixLab sources beyond the two
registration import lines.
