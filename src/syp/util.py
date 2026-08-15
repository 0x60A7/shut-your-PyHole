"""Subprocess, filesystem and version helpers. Nothing here should ever raise."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Iterable, List, Optional, Sequence, Tuple

DEFAULT_TIMEOUT = 20


def which(name: str) -> Optional[str]:
    return shutil.which(name)


def run(
    cmd: Sequence[str],
    cwd: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[int, str]:
    """Run a command, returning (returncode, combined output).

    Returns (-1, reason) instead of raising for the three ways this normally
    fails: binary absent, timeout, or the OS refusing to exec it.
    """
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=cwd,
            capture_output=True,
            timeout=timeout,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return -1, "not found"
    except subprocess.TimeoutExpired:
        return -1, "timed out"
    except OSError as exc:  # permission denied, exec format error, ...
        return -1, str(exc)
    return proc.returncode, ((proc.stdout or "") + (proc.stderr or "")).strip()


def read_text(path: str, limit: int = 1_000_000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(limit)
    except OSError:
        return ""


PRUNE_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "node_modules", ".idea", ".vscode", ".tox", ".eggs",
    "site-packages", ".ipynb_checkpoints", ".cache", "dist", "build",
}
PRUNE_SUFFIXES = (".egg-info",)
VENV_MARKERS = ("pyvenv.cfg",)


def is_venv_dir(path: str) -> bool:
    return any(os.path.exists(os.path.join(path, m)) for m in VENV_MARKERS)


def walk_files(root: str, max_files: int = 40_000) -> List[str]:
    """Repo-relative paths with POSIX separators, skipping caches and venvs."""
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in PRUNE_DIRS
            and not d.endswith(PRUNE_SUFFIXES)
            and not is_venv_dir(os.path.join(dirpath, d))
        ]
        rel_dir = os.path.relpath(dirpath, root)
        prefix = "" if rel_dir == "." else rel_dir.replace(os.sep, "/") + "/"
        for name in filenames:
            out.append(prefix + name)
            if len(out) >= max_files:
                return out
    return out


def human_size(num_bytes: int) -> str:
    step = 1024.0
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < step:
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= step
    return f"{value:.1f}PB"


def path_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def dir_is_empty(path: str) -> bool:
    try:
        return not any(os.scandir(path))
    except OSError:
        return True


# --- version handling -------------------------------------------------------

_VER_PART = re.compile(r"\d+|[a-zA-Z]+")


def parse_version(text: str) -> Tuple:
    """Loose PEP 440-ish version key. Numeric parts sort before alphabetic ones."""
    parts: List[Tuple[int, object]] = []
    for token in re.split(r"[.\-_+]", text.strip()):
        for piece in _VER_PART.findall(token):
            if piece.isdigit():
                parts.append((1, int(piece)))
            else:
                parts.append((0, piece.lower()))
    return tuple(parts)


_SPEC = re.compile(r"(==|!=|>=|<=|~=|>|<|===)\s*([0-9][^,\s]*)")


def satisfies(version: str, specifier: str) -> Optional[bool]:
    """Check ``version`` against a PEP 440 specifier set.

    Returns None when we cannot decide (wildcards, epochs, empty specifier) —
    an honest 'unknown' beats a confident wrong answer.
    """
    if not specifier.strip():
        return True
    clauses = _SPEC.findall(specifier)
    if not clauses:
        return None
    have = parse_version(version)
    for op, raw in clauses:
        if "*" in raw or "!" in raw and op not in ("!=",):
            return None
        want = parse_version(raw)
        if op in ("==", "==="):
            # `==1.2` should accept 1.2.0: compare on the declared precision.
            if have[: len(want)] != want:
                return False
        elif op == "!=":
            if have[: len(want)] == want:
                return False
        elif op == ">=":
            if have < want:
                return False
        elif op == "<=":
            if have > want:
                return False
        elif op == ">":
            if have <= want:
                return False
        elif op == "<":
            if have >= want:
                return False
        elif op == "~=":
            if have < want:
                return False
            ceiling = want[:-1]
            if ceiling and have[: len(ceiling) - 1] != ceiling[: len(ceiling) - 1]:
                return False
    return True


def normalize_dist(name: str) -> str:
    """PEP 503 normalization, so ``opencv_python`` and ``opencv-python`` match."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
