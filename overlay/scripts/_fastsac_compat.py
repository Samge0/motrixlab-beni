"""Compose the full motrix.fastsac algo_base defaults via hydra.

Eval/record scripts bypass train.py's hydra composition and read only a
run's task_config.yaml snapshot. Upstream evolves the FastSAC schema
(MISSING-mandatory fields appear over time), so old snapshots need the
current defaults filled in. This helper produces the resolved async_options
from the *installed* MotrixLab checkout, keeping old checkpoints playable.
"""

from __future__ import annotations

from pathlib import Path


def resolved_fastsac_async_options(motrixlab_root: Path) -> dict:
    """Return the fully-resolved ``trainer.async_options`` dict."""
    from hydra import compose, initialize_config_dir

    import motrix_rl  # noqa: F401  (registers the _motrix_fastsac_schema)

    config_dir = str((Path(motrixlab_root) / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="algo_base/motrix.fastsac")
    async_options = cfg.algo_base.trainer.async_options
    from omegaconf import OmegaConf

    return OmegaConf.to_container(async_options, resolve=True)
