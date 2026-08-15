"""Which files the thing you are running actually reaches.

A repository's requirements are not one set. WHAM's `demo.py` needs four
checkpoints and a body model; `train.py` additionally needs AMASS, 3DPW and a
stage-one checkpoint that no demo will ever open. Unioning them produces a
report where two thirds of the blockers are irrelevant to what you asked for.

This walks the local import graph from the entrypoint and returns the files it
can actually reach, so a requirement can be attributed to the run it belongs to.
"""

from __future__ import annotations

import ast
import os
from typing import List, Optional, Set

from .context import RepoContext

MAX_VISITED = 3000


def module_candidates(dotted: str) -> List[str]:
    """Repo-relative paths a dotted module name could resolve to."""
    parts = dotted.split(".")
    base = "/".join(parts)
    out = [f"{base}.py", f"{base}/__init__.py"]
    # src/ and lib/ layouts hide the importable root one level down.
    out += [f"src/{base}.py", f"src/{base}/__init__.py"]
    return out


def _resolve_relative(importer: str, level: int, module: Optional[str]) -> List[str]:
    parts = importer.split("/")[:-1]  # directory of the importing file
    for _ in range(level - 1):
        if parts:
            parts.pop()
    base = "/".join(parts + (module.split(".") if module else []))
    base = base.strip("/")
    if not base:
        return []
    return [f"{base}.py", f"{base}/__init__.py"]


def reachable(ctx: RepoContext, entry: str) -> Set[str]:
    """Repo files reachable from ``entry`` through local imports.

    Only local modules are followed: a third-party package's internals are not
    this repository's business. Includes the entry file itself.
    """
    if not ctx.exists(entry):
        return set()

    seen: Set[str] = {entry}
    queue = [entry]
    while queue and len(seen) < MAX_VISITED:
        current = queue.pop()
        try:
            tree = ast.parse(ctx.text(current), filename=current)
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            targets: List[str] = []
            if isinstance(node, ast.Import):
                for alias in node.names:
                    targets += module_candidates(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    targets += _resolve_relative(current, node.level, node.module)
                elif node.module:
                    targets += module_candidates(node.module)
                    # `from lib.models import wham` may name a module, not a symbol.
                    for alias in node.names:
                        targets += module_candidates(f"{node.module}.{alias.name}")
            for candidate in targets:
                if candidate in seen or not ctx.exists(candidate):
                    continue
                seen.add(candidate)
                queue.append(candidate)
    return seen


# Documentation whose subject is plainly not the demo.
_OFF_TOPIC_DOCS = ("dataset", "training", "train", "eval", "benchmark", "preprocess")


_TRAINING_ENTRY = ("train", "eval", "benchmark", "preprocess", "prepare")


def out_of_scope(
    ctx: RepoContext, source: str, reached: Set[str], entry: Optional[str] = None
) -> Optional[str]:
    """Why this source does not belong to the entrypoint's run, or None.

    Conservative by design: anything we cannot attribute stays in scope. A
    requirement wrongly dropped is worse than one wrongly kept, because the
    first is invisible.
    """
    rel_early = source.split(":")[0]
    if not reached:
        # No entrypoint could be identified, so no requirement can be attributed
        # to one. A library's source tree is a catalogue of things it *can*
        # fetch on demand — reporting all of it as missing turned pytorch_geometric
        # into 123 blockers and told nobody anything.
        return "no entrypoint identified, so nothing can be attributed to a run"
    # What counts as off-topic depends on the topic. DATASET.md is noise when
    # auditing a demo and the whole point when auditing training.
    entry_name = os.path.basename(entry or "").lower()
    auditing_training = any(word in entry_name for word in _TRAINING_ENTRY)
    rel = source.split(":")[0]
    if rel in reached:
        return None

    lowered = rel.lower()
    if lowered.endswith(".py"):
        head = lowered.split("/")[0]
        if head in ("train.py", "eval.py"):
            return f"used by {rel}"
        if "/eval" in lowered or lowered.startswith(("train", "tools/train", "scripts/train")):
            return f"used by {rel}, not by the entrypoint"
        return f"not reachable from the entrypoint ({rel})"

    if lowered.endswith((".md", ".rst", ".txt")):
        if auditing_training:
            return None  # training docs are the subject, not a distraction
        # A root README describes how to run the thing; docs/ is reference
        # material full of illustrative paths that were never yours to have.
        if lowered.startswith(("docs/", "doc/")) or "/docs/" in lowered:
            return f"illustrated in {rel}"
        if any(word in lowered for word in _OFF_TOPIC_DOCS):
            return f"documented in {rel}"
        return None
    return None
