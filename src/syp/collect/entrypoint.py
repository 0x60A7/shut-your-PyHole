"""The documented smoke test.

A repo is 'ready' only in relation to something you intend to run, so the audit
ends by finding the command the README tells you to run and checking that every
file that command names actually exists.
"""

from __future__ import annotations

import os
import re
import shlex
from typing import List, Optional, Tuple

from ..context import RepoContext
from ..model import Kind, Report, Requirement, Status

_FENCE = re.compile(r"```(?:bash|sh|shell|console|)\n(.*?)```", re.DOTALL)
_RUN_LINE = re.compile(r"^\s*(?:\$\s*)?((?:python\d?|python3|bash|sh|make|torchrun|accelerate)\s+.+)$")
_PLACEHOLDER = re.compile(r"[<>{}\[\]]|/path/to|YOUR_|\bxxx\b", re.IGNORECASE)

DEMO_NAMES = (
    "demo.py", "demo/demo.py", "inference.py", "infer.py", "run.py", "main.py",
    "predict.py", "test.py", "app.py", "scripts/demo.py", "tools/demo.py",
    "examples/demo.py", "demo.sh", "scripts/demo.sh", "run_demo.sh",
)


_DEMO_WORDS = re.compile(r"(demo|infer|predict|track|visuali[sz]e|run|main|eval|test)", re.IGNORECASE)
_SETUP_WORDS = re.compile(r"(fetch|download|prepare|install|setup|build|compile|train)", re.IGNORECASE)


def chosen_command(ctx: RepoContext) -> Optional[Tuple[str, str]]:
    """The command this repo is for, as (command, source).

    Shared with the asset scanner, which needs to know what is being run before
    it can say which requirements belong to that run.
    """
    # An explicit --entry answers the question outright: audit *that* run.
    if ctx.config.entry:
        entry = ctx.config.entry
        runner = "bash" if entry.endswith(".sh") else "python"
        matching = next(
            (c for c, _ in _documented_commands(ctx) if _target_file(c) == entry), None
        )
        return (matching or f"{runner} {entry}"), "--entry"
    if ctx.config.smoke_command:
        return ctx.config.smoke_command, ctx.config.source or ".syp.toml"

    commands = _documented_commands(ctx)
    # Document order alone picks the setup script, which is never the smoke test.
    ranked = sorted(
        enumerate(commands),
        key=lambda item: (
            1 if _SETUP_WORDS.search(item[1][0]) else 0,
            0 if _DEMO_WORDS.search(item[1][0]) else 1,
            item[0],
        ),
    )
    for _, (command, source) in ranked:
        target = _target_file(command)
        if target and ctx.exists(target):
            return command, source

    fallback = next((name for name in DEMO_NAMES if ctx.exists(name)), None)
    if fallback:
        runner = "bash" if fallback.endswith(".sh") else "python"
        return f"{runner} {fallback}", fallback
    return None


def entry_file(ctx: RepoContext) -> Optional[str]:
    """The Python file the documented command runs, if it is a Python file."""
    chosen = chosen_command(ctx)
    if not chosen:
        return None
    target = _target_file(chosen[0])
    return target if target and target.endswith(".py") and ctx.exists(target) else None


def collect(ctx: RepoContext, report: Report) -> None:
    chosen = chosen_command(ctx)

    if chosen is None:
        report.add(
            Requirement(
                kind=Kind.ENTRYPOINT,
                name="smoke test",
                status=Status.UNKNOWN,
                detail="no runnable demo command found in the docs and no demo.py in the tree",
                manual="Pick the command you actually intend to run and check its inputs by hand.",
            )
        )
        return

    command, source = chosen
    report.add(
        Requirement(
            kind=Kind.ENTRYPOINT,
            name=_target_file(command) or "entrypoint",
            status=Status.OK,
            detail=command if len(command) < 90 else command[:87] + "...",
            source=source,
            meta={"command": command, "smoke": True},
        )
    )

    for path in _referenced_files(ctx, command):
        exists = ctx.exists(path)
        report.add(
            Requirement(
                kind=Kind.ENTRYPOINT,
                name=path,
                status=Status.OK if exists else Status.MISSING,
                detail="input referenced by the demo command" if exists else "demo input is absent",
                source=source,
                manual=None if exists else "Supply your own input, or fetch the sample the README names.",
            )
        )


def _documented_commands(ctx: RepoContext) -> List[Tuple[str, str]]:
    """Runnable-looking commands from README/docs fenced blocks, in document order."""
    out: List[Tuple[str, str]] = []
    for rel in ctx.text_files((".md", ".rst")):
        if not (rel.lower().startswith(("readme", "docs/", "doc/")) or "install" in rel.lower()):
            continue
        text = ctx.text(rel)
        for block in _FENCE.findall(text):
            for line in block.splitlines():
                match = _RUN_LINE.match(line)
                if not match:
                    continue
                command = match.group(1).strip().rstrip("\\").strip()
                if _PLACEHOLDER.search(command) or "install" in command or "pip " in command:
                    continue
                out.append((command, ctx.source_ref(rel, line.strip()[:40])))
    return out


def _split(command: str) -> List[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return command.split()


def _target_file(command: str) -> Optional[str]:
    parts = _split(command)
    for token in parts[1:]:
        if token.startswith("-"):
            continue
        if token.endswith((".py", ".sh")):
            return token.replace("\\", "/").lstrip("./")
        break
    return None


def _referenced_files(ctx: RepoContext, command: str) -> List[str]:
    """Concrete input paths passed on the command line (not the script itself)."""
    target = _target_file(command)
    out: List[str] = []
    for token in _split(command)[1:]:
        token = token.replace("\\", "/").lstrip("./")
        if token.startswith("-") or "=" in token.split("/")[0] and "/" not in token:
            continue
        if token == target or not os.path.splitext(token)[1]:
            continue
        if _PLACEHOLDER.search(token) or token.endswith((".py", ".sh")):
            continue
        if "/" in token or token.endswith((".mp4", ".mov", ".yaml", ".yml", ".jpg", ".png", ".json")):
            out.append(token)
    return out
