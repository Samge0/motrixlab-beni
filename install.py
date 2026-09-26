"""Install the BENI velocity-flat task into a MotrixLab checkout.

Copies overlay/* into the checkout (MotrixLab-relative paths) and appends
the two registration import lines. Idempotent: safe to re-run.

The registration patches are anchor-based (not "last import line"), so they
survive upstream refactors of the __init__.py files. A trailing-fallback
append is used only when no anchor matches.

Usage:
    python install.py --motrixlab /path/to/MotrixLab
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ROBOT_INIT_REL = Path("motrix_envs/src/motrix_envs/robot/__init__.py")
LOCOMOTION_INIT_REL = Path("motrix_envs/src/motrix_envs/locomotion/__init__.py")


def _insert_after_imports(text: str, add_lines: list[str]) -> str:
    """Insert add_lines right before the first registry.* call or, failing
    that, after the import block (a run of top-level import/from statements,
    skipping multi-line parenthesized imports)."""
    lines = text.splitlines(keepends=True)

    # Strategy 1: before the first "registry." statement at top level.
    for i, line in enumerate(lines):
        if line.lstrip().startswith("registry."):
            for offset, extra in enumerate(add_lines):
                lines.insert(i + offset, extra)
            return "".join(lines)

    # Strategy 2: after the import block. Track paren depth so a multi-line
    # "from x import (a,\n b,\n c)" counts as ONE import statement.
    depth = 0
    last_import_end = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if depth > 0:
            depth += line.count("(") - line.count(")")
            if depth <= 0:
                depth = 0
                last_import_end = i
            continue
        if stripped.startswith(("import ", "from ")):
            depth = line.count("(") - line.count(")")
            if depth <= 0:
                depth = 0
                last_import_end = i
        elif stripped and not stripped.startswith("#"):
            break
    insert_at = (last_import_end + 1) if last_import_end is not None else len(lines)
    for offset, extra in enumerate(add_lines):
        lines.insert(insert_at + offset, extra)
    return "".join(lines)


def patch_robot_init(path: Path) -> str:
    """Returns: 'already' | 'patched' | error message."""
    text = path.read_text(encoding="utf-8")
    if "beni" in text and "Beni" in text:
        return "already"

    # Match the file's dominant line ending.
    eol = "\r\n" if "\r\n" in text[:2000] else "\n"
    import_line = f"from motrix_envs.robot.beni import Beni{eol}"
    reg_line = f'registry.robotcfg("beni")(Beni){eol}'

    if "from motrix_envs.robot.beni import Beni" not in text:
        text = _insert_after_imports(text, [import_line])
    if 'registry.robotcfg("beni")' not in text:
        # put the registration right before the __all__ block or at EOF
        if "__all__" in text:
            idx = text.index("__all__")
            text = text[:idx] + reg_line + eol + text[idx:]
        else:
            if not text.endswith(eol):
                text += eol
            text += reg_line
    path.write_text(text, encoding="utf-8", newline="")
    return "patched"


def patch_locomotion_init(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if re_search_word(text, "beni"):
        return "already"
    eol = "\r\n" if "\r\n" in text[:2000] else "\n"
    # Anchor: the canonical combined-import line in every known version.
    for old in (
        "from . import anymal_c, ball_balance, go1, humanoid, quadruped, wbt",
        "from . import anymal_c, ball_balance, go1, humanoid, quadruped",
    ):
        if old in text:
            new = old.replace("ball_balance,", "ball_balance, beni,", 1)
            text = text.replace(old, new, 1)
            path.write_text(text, encoding="utf-8", newline="")
            return "patched"
    # Fallback: any "from . import x, y" line -> append beni alphabetically.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("from . import ") and "beni" not in stripped:
            modules = [m.strip().rstrip("  # noqa") for m in stripped[len("from . import "):].split(",")]
            modules = [m for m in modules if m and not m.startswith("#")]
            modules.append("beni")
            modules = sorted(set(modules))
            new_line = f"from . import {', '.join(modules)}"
            text = text.replace(line, new_line, 1)
            path.write_text(text, encoding="utf-8", newline="")
            return "patched"
    return "ERROR: no 'from . import ...' anchor found"


def re_search_word(text: str, word: str) -> bool:
    import re

    return re.search(rf"\b{word}\b", text) is not None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motrixlab", required=True, help="Path to MotrixLab checkout root")
    args = parser.parse_args()

    root = Path(args.motrixlab).resolve()
    if not (root / "scripts" / "train.py").exists():
        print(f"ERROR: {root} does not look like a MotrixLab checkout (scripts/train.py missing)")
        return 1

    overlay = HERE / "overlay"
    if not overlay.exists():
        print("ERROR: overlay/ missing next to install.py")
        return 1

    copied = skipped = 0
    for src in overlay.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(overlay)
        dst = root / rel
        if dst.exists():
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(f"copied {copied} files, kept {skipped} existing")

    for label, rel, patcher in (
        ("robot", ROBOT_INIT_REL, patch_robot_init),
        ("locomotion", LOCOMOTION_INIT_REL, patch_locomotion_init),
    ):
        init_path = root / rel
        if not init_path.exists():
            print(f"ERROR: {init_path} missing")
            return 1
        result = patcher(init_path)
        if result == "ERROR: no 'from . import ...' anchor found":
            print(f"WARNING: {init_path}: no anchor; add manually:  from . import beni")
            return 1
        print(f"{label} registration {result}: {init_path}")

    print("\nInstall complete. Next:")
    print(f"  cd {root}")
    print(f"  .venv/Scripts/python.exe {HERE / 'verify_install.py'} --motrixlab {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
