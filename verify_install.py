"""Post-install verification: construct beni-velocity-flat and run 50 random
steps in the caller's MotrixLab checkout. Run from anywhere:

    python verify_install.py --motrixlab <path-to-MotrixLab>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Pretrained run shipped in overlay/ (metadata-backed; see README).
PRETRAINED_RUN_GLOB = "runs/beni-velocity-flat/motrix/torch/fastsac/*/checkpoints"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motrixlab", required=True)
    args = parser.parse_args()
    root = Path(args.motrixlab).resolve()
    if root not in sys.path:
        sys.path.insert(0, str(root))

    import numpy as np  # noqa: E402

    import motrix_envs  # noqa: F401  (registers built-in tasks)
    from motrix_env_core import registry  # noqa: E402

    if not registry.contains("beni-velocity-flat"):
        print("FAIL: beni-velocity-flat not registered -- install.py patches missing?")
        return 1
    print("task registered OK")

    env = registry.make("beni-velocity-flat", num_envs=4)
    print(f"env constructed OK (action {env.action_space.shape}, "
          f"policy obs {env.observation_space.policy.shape})")

    rng = np.random.default_rng(0)
    env.init_state()
    for step in range(50):
        actions = rng.uniform(-1, 1, size=(4, env.action_space.shape[0])).astype(np.float32)
        state = env.step(actions)
    print(f"50 random steps OK (last reward {float(state.reward.mean()):.4f})")

    # pretrained checkpoint present? (any run dir, not a fixed timestamp)
    ckpts = sorted(root.glob(PRETRAINED_RUN_GLOB + "/model_*.pt"))
    if ckpts:
        print(f"pretrained checkpoint OK: {ckpts[0].relative_to(root)}")
    else:
        print("NOTE: pretrained checkpoint not installed; train from scratch per README")
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
