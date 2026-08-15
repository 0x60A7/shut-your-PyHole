"""Git-layer requirements: submodules and LFS payloads.

Submodules are the single most common reason a research repo fails on a fresh
clone, and the failure mode is silent: the directory exists, it is just empty.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List

from ..context import RepoContext
from ..model import Kind, Report, Requirement, Status
from ..util import dir_is_empty, path_size, run, which

_SECTION = re.compile(r'^\s*\[submodule\s+"([^"]+)"\]\s*$', re.MULTILINE)
_KEY = re.compile(r"^\s*(\w+)\s*=\s*(.+?)\s*$", re.MULTILINE)

LFS_POINTER_PREFIX = "version https://git-lfs"


def parse_gitmodules(text: str) -> List[Dict[str, str]]:
    """Parse .gitmodules into [{name, path, url, branch}] without configparser.

    Git's config format allows duplicate section names, which configparser rejects.
    """
    mods: List[Dict[str, str]] = []
    matches = list(_SECTION.finditer(text))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end() : end]
        entry = {"name": match.group(1)}
        for key, value in _KEY.findall(body):
            entry[key.lower()] = value
        mods.append(entry)
    return mods


def _submodule_states(root: str) -> Dict[str, str]:
    """Map submodule path -> state char from `git submodule status`.

    ' ' in sync, '-' not initialized, '+' checked out at a different commit,
    'U' merge conflict.
    """
    code, out = run(["git", "-C", root, "submodule", "status", "--recursive"])
    if code != 0:
        return {}
    states: Dict[str, str] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        marker = line[0] if line[0] in "-+U " else " "
        rest = line[1:] if line[0] in "-+U " else line
        parts = rest.split()
        if len(parts) >= 2:
            states[parts[1].replace(os.sep, "/")] = marker
    return states


def collect(ctx: RepoContext, report: Report) -> None:
    is_repo = os.path.isdir(os.path.join(ctx.root, ".git")) or os.path.isfile(
        os.path.join(ctx.root, ".git")
    )
    has_modules = ctx.exists(".gitmodules")

    if not is_repo:
        if has_modules:
            report.add(
                Requirement(
                    kind=Kind.GIT,
                    name="git repository",
                    status=Status.MISSING,
                    detail="no .git — this looks like a source archive, so submodules cannot be fetched",
                    source=".gitmodules",
                    manual="Clone the project with git instead of downloading a zip.",
                )
            )
        return

    report.add(
        Requirement(kind=Kind.GIT, name="git repository", status=Status.OK, detail=_describe_head(ctx.root))
    )

    if has_modules:
        _collect_submodules(ctx, report)
    _collect_lfs(ctx, report, is_repo)


def _describe_head(root: str) -> str:
    code, out = run(["git", "-C", root, "rev-parse", "--short", "HEAD"])
    sha = out.strip() if code == 0 else "?"
    code, out = run(["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"])
    branch = out.strip() if code == 0 else "?"
    return f"{branch} @ {sha}"


def _collect_submodules(ctx: RepoContext, report: Report) -> None:
    mods = parse_gitmodules(ctx.text(".gitmodules"))
    if not mods:
        return
    states = _submodule_states(ctx.root)
    unready = [
        m
        for m in mods
        if states.get(m.get("path", m["name"]).replace(os.sep, "/")) in ("-", "+")
        or not os.path.exists(ctx.abspath(m.get("path", m["name"])))
        or dir_is_empty(ctx.abspath(m.get("path", m["name"])))
    ]
    # One `--init --recursive` handles the lot; per-path commands would just be
    # three ways of typing the same thing.
    bulk_fix = "git submodule update --init --recursive" if len(unready) > 1 else None
    report.add(
        Requirement(
            kind=Kind.GIT,
            name=f"submodules declared ({len(mods)})",
            status=Status.INFO,
            detail=", ".join(m.get("path", m["name"]) for m in mods),
            source=".gitmodules",
        )
    )
    for mod in mods:
        path = mod.get("path", mod["name"]).replace(os.sep, "/")
        url = mod.get("url", "")
        abs_path = ctx.abspath(path)
        state = states.get(path)
        fix = bulk_fix or f"git submodule update --init --recursive -- {path}"
        source = ctx.source_ref(".gitmodules", f'"{mod["name"]}"')

        if state == "-" or (not os.path.exists(abs_path)) or dir_is_empty(abs_path):
            report.add(
                Requirement(
                    kind=Kind.GIT,
                    name=path,
                    status=Status.MISSING,
                    detail=f"submodule not initialized ({url})" if url else "submodule not initialized",
                    source=source,
                    fix=fix,
                    meta={"url": url, "submodule": True},
                )
            )
        elif state == "+":
            report.add(
                Requirement(
                    kind=Kind.GIT,
                    name=path,
                    status=Status.STALE,
                    detail="submodule checked out at a different commit than the superproject pins",
                    source=source,
                    fix=fix,
                    meta={"url": url, "submodule": True},
                )
            )
        elif state == "U":
            report.add(
                Requirement(
                    kind=Kind.GIT,
                    name=path,
                    status=Status.MISMATCH,
                    detail="submodule has merge conflicts",
                    source=source,
                    manual="Resolve the conflict in the submodule by hand.",
                )
            )
        else:
            report.add(
                Requirement(
                    kind=Kind.GIT,
                    name=path,
                    status=Status.OK,
                    detail="submodule initialized",
                    source=source,
                )
            )


def _collect_lfs(ctx: RepoContext, report: Report, is_repo: bool) -> None:
    attrs = ctx.text(".gitattributes") if ctx.exists(".gitattributes") else ""
    if "filter=lfs" not in attrs:
        return

    if not which("git-lfs") and run(["git", "lfs", "version"])[0] != 0:
        report.add(
            Requirement(
                kind=Kind.GIT,
                name="git-lfs",
                status=Status.MISSING,
                detail="repo tracks files with LFS but git-lfs is not installed",
                source=".gitattributes",
                manual="Install git-lfs, then run `git lfs pull`.",
            )
        )
        return

    # A file left as a pointer is ~130 bytes of text starting with a version line.
    pointers = []
    for rel in ctx.files:
        abs_path = ctx.abspath(rel)
        if 0 < path_size(abs_path) < 1024:
            head = ctx.text(rel)[:64] if ctx.is_textish(rel) else _peek(abs_path)
            if head.startswith(LFS_POINTER_PREFIX):
                pointers.append(rel)
    if pointers:
        shown = ", ".join(pointers[:3]) + (f" (+{len(pointers) - 3} more)" if len(pointers) > 3 else "")
        report.add(
            Requirement(
                kind=Kind.GIT,
                name="git-lfs objects",
                status=Status.MISSING,
                detail=f"{len(pointers)} file(s) are still LFS pointers: {shown}",
                source=".gitattributes",
                fix="git lfs pull",
            )
        )
    else:
        report.add(
            Requirement(
                kind=Kind.GIT, name="git-lfs objects", status=Status.OK, detail="no unresolved pointers"
            )
        )


def _peek(abs_path: str, n: int = 64) -> str:
    try:
        with open(abs_path, "rb") as fh:
            return fh.read(n).decode("utf-8", "replace")
    except OSError:
        return ""
