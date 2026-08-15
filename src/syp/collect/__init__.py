"""Collectors: each one turns a slice of the repository into Requirements.

Order matters only for presentation; collectors do not depend on each other.
"""

from __future__ import annotations

from typing import List

from ..context import RepoContext
from ..model import Report, Status
from . import (
    assets, build, container, entrypoint, envvars, git, imports, observed, pydeps,
    resolve, system,
)

COLLECTORS = [
    ("system", system.collect),
    ("git", git.collect),
    ("container", container.collect),
    ("python", pydeps.collect),
    ("imports", imports.collect),
    ("resolve", resolve.collect),
    ("build", build.collect),
    ("env", envvars.collect),
    ("assets", assets.collect),
    ("entrypoint", entrypoint.collect),
    ("observed", observed.collect),
]

COLLECTOR_NAMES = [name for name, _ in COLLECTORS]


def run_all(ctx: RepoContext, only: List[str] = None) -> Report:
    report = Report(root=ctx.root, target=ctx.target.describe())
    for name, fn in COLLECTORS:
        if only and name not in only:
            continue
        try:
            fn(ctx, report)
        except Exception as exc:  # a broken collector must not sink the audit
            report.notes.append(f"collector '{name}' failed: {exc.__class__.__name__}: {exc}")
    _apply_config(ctx, report)
    if ctx.target.problem:
        report.notes.append(ctx.target.problem)
    if ctx.config.error:
        report.notes.append(f"{ctx.config.source}: {ctx.config.error}")
    return report


def _apply_config(ctx: RepoContext, report: Report) -> None:
    """Drop what `.syp.toml` says not to report, and say how much was dropped."""
    if not ctx.config.ignore_names:
        return
    kept = []
    for req in report.requirements:
        if req.status is not Status.OK and ctx.config.ignores_name(req.name):
            report.suppressed.append(req.name)
        else:
            kept.append(req)
    report.requirements = kept
    if report.suppressed:
        report.notes.append(
            f"{len(report.suppressed)} finding(s) suppressed by {ctx.config.source}: "
            + ", ".join(report.suppressed[:5])
        )
