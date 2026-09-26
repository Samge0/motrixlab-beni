# Copyright (c) 2026 ported from Mondo-Robotics/open-beni (MIT)
# SPDX-License-Identifier: MIT
"""Plot return / ep_len curves from a fastsac async training log."""

import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

log_path, out_png = sys.argv[1], sys.argv[2]
window = int(sys.argv[3]) if len(sys.argv) > 3 else 15

iter_re = re.compile(r"iter (\d+)/(\d+)")
return_re = re.compile(r"return\s+([-\d.]+|nan)")
eplen_re = re.compile(r"ep_len\s+([-\d.]+|nan)")

iters, returns, eplens = [], [], []
cur_it = None
with open(log_path, encoding="utf-8", errors="replace") as f:
    for line in f:
        m = iter_re.search(line)
        if m:
            cur_it = int(m.group(1))
            continue
        if cur_it is None:
            continue
        m = return_re.search(line)
        if m and cur_it is not None:
            iters.append(cur_it)
            returns.append(float(m.group(1)) if m.group(1) != "nan" else float("nan"))
        m = eplen_re.search(line)
        if m and cur_it is not None:
            eplens.append(float(m.group(1)) if m.group(1) != "nan" else float("nan"))

if not iters:
    sys.exit("No iterations parsed from log")


def rolling(xs, w):
    out, acc = [], []
    for x in xs:
        acc.append(x)
        if len(acc) > w:
            acc.pop(0)
        out.append(sum(acc) / len(acc) if all(a == a for a in acc) else float("nan"))
    return out


fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
axes[0].plot(iters, returns, alpha=0.3, label="return")
axes[0].plot(iters, rolling(returns, window), lw=2, label=f"rolling({window})")
axes[0].set_ylabel("rollout return")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[1].plot(iters, eplens, alpha=0.3, color="tab:orange", label="ep_len")
axes[1].plot(iters, rolling(eplens, window), lw=2, color="tab:orange", label=f"rolling({window})")
axes[1].axhline(2000, color="r", ls="--", alpha=0.5, label="20s cap")
axes[1].set_ylabel("episode length")
axes[1].set_xlabel("iteration")
axes[1].legend()
axes[1].grid(alpha=0.3)
fig.suptitle("beni-velocity-flat fastsac training")
fig.tight_layout()
fig.savefig(out_png, dpi=130)
print(f"saved {out_png} ({len(iters)} iters)")
