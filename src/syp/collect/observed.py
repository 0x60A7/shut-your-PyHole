"""Findings that come from watching the program run, not from reading it.

`syp trace` records what the process asked for. Everything here is fact rather
than inference, so it is reported with more confidence than anything a scanner
produces — and it catches the whole class of dependencies that only exist at
run time: a subprocess call to ffmpeg, a host contacted for weights, a path
assembled from three config values.

Missing *files* are folded into the asset section instead, where they belong.
"""

from __future__ import annotations

import os
from typing import Set

from ..context import RepoContext
from ..model import Kind, Report, Requirement, Status
from ..util import which

# Spawned by the interpreter itself or by tooling, not by the project.
_IGNORED_EXECUTABLES = {
    "sh", "bash", "cmd", "cmd.exe", "conhost.exe", "env", "which", "where",
    "pip", "pip.exe", "uv", "uv.exe", "git", "git.exe", "ninja", "cl.exe",
}


def collect(ctx: RepoContext, report: Report) -> None:
    trace = ctx.trace
    if trace is None:
        return

    report.add(
        Requirement(
            kind=Kind.ENTRYPOINT,
            name="observed run",
            status=Status.INFO,
            detail=f"{trace.command or 'traced command'} — exit {trace.exit_code}"
            if trace.exit_code is not None
            else (trace.command or "traced command"),
            source=os.path.basename(trace.path or "trace"),
            explain=f"{len(trace.opened)} path(s) opened, {len(trace.imports)} module(s) imported, "
            f"{len(trace.missing)} path(s) missing.",
            meta={"observed": True},
        )
    )

    _report_executables(ctx, report, trace.executables)
    _report_network(ctx, report, trace.hosts, trace.urls)

    if trace.exit_code not in (None, 0):
        report.add(
            Requirement(
                kind=Kind.ENTRYPOINT,
                name="smoke test",
                status=Status.MISSING,
                detail=f"the traced run exited {trace.exit_code}",
                source=os.path.basename(trace.path or "trace"),
                manual="Clear the blockers above, then re-run `syp trace`.",
            )
        )
    elif trace.exit_code == 0:
        report.add(
            Requirement(
                kind=Kind.ENTRYPOINT,
                name="smoke test",
                status=Status.OK,
                detail="the traced run completed successfully",
                source=os.path.basename(trace.path or "trace"),
            )
        )


def _report_executables(ctx: RepoContext, report: Report, executables: Set[str]) -> None:
    for name in sorted(executables):
        stem = name[:-4] if name.endswith(".exe") else name
        if stem in _IGNORED_EXECUTABLES or not stem:
            continue
        found = which(stem)
        existing = next(
            (r for r in report.requirements if r.kind is Kind.SYSTEM and r.name == stem), None
        )
        if existing is not None:
            # Observation beats inference: "the image installs it" stops being a
            # good enough answer once we have watched the host program call it.
            existing.status = Status.OK if found else Status.MISSING
            existing.detail = (
                "spawned by the traced run" if found else "spawned by the traced run, not on PATH"
            )
            existing.source = "observed at runtime"
            existing.manual = None if found else f"Install {stem}; the program shells out to it."
            continue
        report.add(
            Requirement(
                kind=Kind.SYSTEM,
                name=stem,
                status=Status.OK if found else Status.MISSING,
                detail="spawned by the traced run" if found else "spawned by the traced run and not on PATH",
                source="observed at runtime",
                manual=None if found else f"Install {stem}; the program shells out to it.",
            )
        )


def _report_network(ctx: RepoContext, report: Report, hosts: Set[str], urls: Set[str]) -> None:
    external = sorted(h for h in hosts if h and not _is_local(h))
    if not external and not urls:
        return
    report.add(
        Requirement(
            kind=Kind.EXTERNAL,
            name=f"network access at runtime ({len(external) or len(urls)})",
            status=Status.INFO,
            detail=", ".join(external[:4]) or ", ".join(sorted(urls)[:2]),
            source="observed at runtime",
            explain="The run needed the network. On an air-gapped machine this is a hard failure, "
            "so pre-populate the relevant caches.",
            meta={"urls": sorted(urls)[:20], "verbose_urls": True},
        )
    )


def _is_local(host: str) -> bool:
    return host in ("127.0.0.1", "::1", "localhost") or host.startswith(("127.", "10.", "192.168."))
