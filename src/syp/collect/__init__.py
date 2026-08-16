"""Collectors: each one turns a slice of the repository into Requirements.

Order matters only for presentation; collectors do not depend on each other.
"""

from __future__ import annotations

import sys
from typing import List

from ..context import RepoContext
from ..model import Kind, Report, Requirement, Status
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
    _check_target(ctx, report)
    for name, fn in COLLECTORS:
        if only and name not in only:
            continue
        try:
            fn(ctx, report)
        except Exception as exc:  # a broken collector must not sink the audit
            report.notes.append(f"collector '{name}' failed: {exc.__class__.__name__}: {exc}")
    _report_parse_failures(ctx, report)
    _apply_config(ctx, report)
    if ctx.target.problem:
        report.notes.append(ctx.target.problem)
    if ctx.config.error:
        report.notes.append(f"{ctx.config.source}: {ctx.config.error}")
    return report


def _report_parse_failures(ctx: RepoContext, report: Report) -> None:
    """Say when the audit could not read some of the code it was given.

    An interpreter cannot parse syntax newer than itself, so auditing a repo
    that uses `match` from a Python 3.9 install loses every import, environment
    variable and path in those files. Reporting less has to look different from
    finding less.
    """
    failures = ctx.parse_failures
    if not failures:
        return
    # Material when the audit is meaningfully blinded: several files, or a file
    # the run being audited actually reaches. A ratio is a bad proxy — one dead
    # py2 script in an eight-file repo is 12% and matters not at all.
    from ..reach import reachable
    from .entrypoint import entry_file

    entry = entry_file(ctx)
    reached = reachable(ctx, entry) if entry else set()
    blinded = [f for f in failures if f.split(":")[0] in reached]
    material = len(failures) >= 3 or bool(blinded)
    version = ".".join(str(p) for p in sys.version_info[:3])
    report.add(
        Requirement(
            kind=Kind.PYTHON,
            name=f"unreadable source files ({len(failures)})",
            status=Status.MISMATCH if material else Status.INFO,
            detail=f"could not be parsed by the Python {version} running this audit",
            source=failures[0],
            manual=(
                "Imports, environment variables and paths in these files are missing "
                "from the report"
                + (f", including {blinded[0]}, which the entrypoint reaches" if blinded else "")
                + ". If the repo uses newer syntax, run syp on a newer Python."
            )
            if material
            else None,
            explain="Usually Python 2 files, templates, or syntax newer than this "
            "interpreter (`match`, PEP 695 type aliases).",
            meta={"packages": failures[:40], "verbose_list": True},
        )
    )


def _check_target(ctx: RepoContext, report: Report) -> None:
    """An audit that could not run is not an audit that found nothing.

    Without this, `--target image:x` on a machine with no docker reports zero
    blockers and a clean bill of health, because every probe merely returned
    UNKNOWN. Silence is not a pass.
    """
    if ctx.target.available:
        return
    report.add(
        Requirement(
            kind=Kind.SYSTEM,
            name=f"audit target: {ctx.target.label}",
            status=Status.MISSING,
            detail=ctx.target.problem or "the requested environment could not be inspected",
            manual="Nothing below was verified against the intended environment. "
            "Make the target reachable, or audit a different one with --target.",
            meta={"target_unavailable": True},
        )
    )


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
