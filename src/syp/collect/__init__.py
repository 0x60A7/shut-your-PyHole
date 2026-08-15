"""Collectors: each one turns a slice of the repository into Requirements.

Order matters only for presentation; collectors do not depend on each other.
"""

from __future__ import annotations

from typing import List

from ..context import RepoContext
from ..model import Report
from . import assets, container, entrypoint, git, pydeps, system

COLLECTORS = [
    ("system", system.collect),
    ("git", git.collect),
    ("container", container.collect),
    ("python", pydeps.collect),
    ("assets", assets.collect),
    ("entrypoint", entrypoint.collect),
]


def run_all(ctx: RepoContext, only: List[str] = None) -> Report:
    report = Report(root=ctx.root)
    for name, fn in COLLECTORS:
        if only and name not in only:
            continue
        try:
            fn(ctx, report)
        except Exception as exc:  # a broken collector must not sink the audit
            report.notes.append(f"collector '{name}' failed: {exc.__class__.__name__}: {exc}")
    return report
