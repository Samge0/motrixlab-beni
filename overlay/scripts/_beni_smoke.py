# Copyright (c) 2026 ported from Mondo-Robotics/open-beni (MIT)
# SPDX-License-Identifier: MIT
"""Smoke test: construct beni-velocity-flat and run a few random steps."""

import numpy as np

import motrix_envs  # noqa: F401 registers tasks
from motrix_env_core import registry


def main() -> None:
    env = registry.make("beni-velocity-flat", num_envs=4)
    print("env constructed OK")
    print("action space:", env.action_space)
    print("obs space policy:", env.observation_space.policy.shape, "value:", env.observation_space.value.shape)

    rng = np.random.default_rng(0)
    state = env.init_state()
    print("init obs policy[0]:", state.obs.policy[0][:8])

    total_reward = 0.0
    for step in range(50):
        actions = rng.uniform(-1, 1, size=(4, env.action_space.shape[0])).astype(np.float32)
        state = env.step(actions)
        total_reward += float(state.reward.mean())
        if step % 10 == 0 or step == 49:
            print(
                f"step {step:3d} reward={state.reward.mean():8.4f} "
                f"terminated={state.terminated.sum()} steps={state.episode_steps[:4]} "
                f"metrics={ {k: round(float(v), 4) for k, v in state.metrics.items()} }"
            )
    print("smoke OK, mean reward:", total_reward / 50)


if __name__ == "__main__":
    main()
