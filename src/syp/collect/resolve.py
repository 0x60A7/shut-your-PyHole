"""Does this set of requirements even hold together?

Three checks, in increasing cost:
  - the installed environment's own metadata (`pip check`) — offline, exact;
  - declared pins that contradict each other across files — offline, textual;
  - ecosystem rules that no manifest encodes (torch/torchvision lockstep,
    numpy 2's ABI break) — offline, heuristic;
  - and with --network, an actual resolution attempt via uv.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..context import RepoContext
from ..knowledge import CONFLICT_RULES, PAIR_RULES
from ..model import Kind, Report, Requirement, Status
from ..util import normalize_dist, parse_version, run, satisfies, which
from .pydeps import Declarations, gather


def collect(ctx: RepoContext, report: Report) -> None:
    decl = gather(ctx)
    if not decl.packages:
        return
    installed = _installed(ctx)

    _pip_check(ctx, report)
    _contradictory_pins(ctx, decl, report)
    _pair_rules(ctx, decl, installed, report)
    _conflict_rules(ctx, decl, installed, report)
    if ctx.network:
        _resolution_attempt(ctx, report)


def _installed(ctx: RepoContext) -> Dict[str, str]:
    script = (
        "import json;import importlib.metadata as m;"
        "print(json.dumps({(d.metadata['Name'] or '').lower():(d.version or '') "
        "for d in m.distributions() if d.metadata['Name']}))"
    )
    code, out = ctx.target.python(["-c", script], timeout=90)
    if code != 0:
        return {}
    try:
        import json

        return {normalize_dist(k): v for k, v in json.loads(out.strip().splitlines()[-1]).items()}
    except (ValueError, IndexError):
        return {}


def _pip_check(ctx: RepoContext, report: Report) -> None:
    """The environment's own opinion of itself. Offline and authoritative."""
    code, out = ctx.target.python(["-m", "pip", "check"], timeout=180)
    if code == -1 or "No module named pip" in (out or ""):
        return
    lines = [l.strip() for l in (out or "").splitlines() if l.strip() and "requires" in l]
    if code == 0 and not lines:
        report.add(
            Requirement(
                kind=Kind.PYTHON,
                name="environment consistency",
                status=Status.OK,
                detail="pip check reports no broken requirements",
            )
        )
        return
    for line in lines[:6]:
        report.add(
            Requirement(
                kind=Kind.PYTHON,
                name="broken requirement",
                status=Status.MISMATCH,
                detail=line[:160],
                source=f"pip check ({ctx.target.describe()})",
                manual="Resolve the version conflict before trusting anything else in this report.",
            )
        )


def _contradictory_pins(ctx: RepoContext, decl: Declarations, report: Report) -> None:
    """The same package pinned to disjoint versions in two different files."""
    for name, pins in decl.duplicate_specs.items():
        exacts = {}
        for spec, source in pins:
            match = re.match(r"^\s*==\s*([\w.]+)", spec or "")
            if match:
                exacts.setdefault(match.group(1), source)
        if len(exacts) > 1:
            listed = ", ".join(f"{v} ({s})" for v, s in list(exacts.items())[:3])
            report.add(
                Requirement(
                    kind=Kind.PYTHON,
                    name=f"{name} pinned twice",
                    status=Status.MISMATCH,
                    detail=f"conflicting pins: {listed}",
                    source=list(exacts.values())[0],
                    manual="Decide which file is authoritative; installs will differ by invocation order.",
                )
            )


def _effective(decl: Declarations, installed: Dict[str, str], name: str) -> Optional[Tuple[str, str]]:
    """Best-known version of a package: what is installed, else an exact pin."""
    key = normalize_dist(name)
    if key in installed and installed[key]:
        return installed[key], "installed"
    dec = decl.packages.get(key)
    if dec and dec.spec:
        match = re.match(r"^\s*(?:==|~=|>=)\s*([\w.]+)", dec.spec)
        if match:
            return match.group(1), dec.source
    return None


def _pair_rules(ctx: RepoContext, decl: Declarations, installed: Dict[str, str], report: Report) -> None:
    for rule in PAIR_RULES:
        left = _effective(decl, installed, rule.left)
        right = _effective(decl, installed, rule.right)

        if rule.forbid_together:
            if normalize_dist(rule.left) in installed and normalize_dist(rule.right) in installed:
                report.add(
                    Requirement(
                        kind=Kind.PYTHON,
                        name=f"{rule.left} + {rule.right}",
                        status=Status.MISMATCH,
                        detail="both installed in the same environment",
                        manual=rule.note,
                        meta={"rule": rule.key},
                    )
                )
            continue

        if not (left and right and rule.table):
            continue
        left_minor = ".".join(left[0].split(".")[:2])
        want = rule.table.get(left_minor)
        if not want:
            continue
        right_minor = ".".join(right[0].split(".")[:2])
        if right_minor != want:
            report.add(
                Requirement(
                    kind=Kind.PYTHON,
                    name=f"{rule.left} / {rule.right} mismatch",
                    status=Status.MISMATCH,
                    detail=f"{rule.left} {left[0]} expects {rule.right} {want}.x, found {right[0]}",
                    source=right[1] if right[1] != "installed" else left[1],
                    manual=rule.note,
                    meta={"rule": rule.key},
                )
            )


def _conflict_rules(ctx: RepoContext, decl: Declarations, installed: Dict[str, str], report: Report) -> None:
    for rule in CONFLICT_RULES:
        version = _effective(decl, installed, rule.package)
        if not version:
            continue
        if satisfies(version[0], rule.boundary) is not True:
            continue
        affected = [
            name
            for name in rule.breaks
            if normalize_dist(name) in installed or normalize_dist(name) in decl.packages
        ]
        if not affected:
            continue
        report.add(
            Requirement(
                kind=Kind.PYTHON,
                name=f"{rule.package} {version[0]} vs {', '.join(affected[:3])}",
                status=Status.MISMATCH,
                detail=f"{rule.package}{rule.boundary} is known to break {', '.join(affected[:4])}",
                source=version[1] if version[1] != "installed" else None,
                manual=rule.note,
                meta={"rule": rule.key},
            )
        )


def _resolution_attempt(ctx: RepoContext, report: Report) -> None:
    """Ask a real resolver whether the declared set is satisfiable at all."""
    if not which("uv"):
        return
    targets = [rel for rel in ctx.glob("requirements*.txt")][:1]
    if not targets:
        return
    code, out = run(
        ["uv", "pip", "compile", targets[0], "--quiet", "--no-header", "-o", "-"],
        cwd=ctx.root,
        timeout=300,
    )
    if code == 0:
        report.add(
            Requirement(
                kind=Kind.PYTHON,
                name="declared set resolves",
                status=Status.OK,
                detail=f"uv resolved {targets[0]} without conflict",
                source=targets[0],
            )
        )
        return
    reason = next(
        (l.strip() for l in (out or "").splitlines() if "conflict" in l.lower() or "because" in l.lower()),
        (out or "").strip().splitlines()[-1] if (out or "").strip() else "no detail",
    )
    report.add(
        Requirement(
            kind=Kind.PYTHON,
            name="declared set does not resolve",
            status=Status.MISMATCH,
            detail=reason[:200],
            source=targets[0],
            manual="No environment satisfies this file as written; the pins need loosening.",
        )
    )
