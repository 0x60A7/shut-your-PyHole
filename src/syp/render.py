"""Terminal rendering. Degrades to ASCII and no-colour without complaint."""

from __future__ import annotations

import os
import sys
from typing import List, Optional

from .model import SECTION_ORDER, SECTION_TITLES, FixKind, Report, Requirement, Status

BAR_WIDTH = 24
NAME_WIDTH = 46

_COLORS = {
    Status.OK: "\033[32m",
    Status.MISSING: "\033[31m",
    Status.STALE: "\033[31m",
    Status.MISMATCH: "\033[33m",
    Status.BLOCKED: "\033[33m",
    Status.UNKNOWN: "\033[36m",
    Status.INFO: "\033[90m",
}
_RESET = "\033[0m"
_DIM = "\033[90m"
_BOLD = "\033[1m"


class Style:
    def __init__(self, color: Optional[bool] = None, unicode_ok: Optional[bool] = None):
        self.color = _want_color() if color is None else color
        self.unicode = _want_unicode() if unicode_ok is None else unicode_ok

    def paint(self, text: str, code: str) -> str:
        return f"{code}{text}{_RESET}" if self.color else text

    def status(self, status: Status) -> str:
        symbol = status.symbol if self.unicode else status.ascii_symbol
        return self.paint(symbol, _COLORS[status])

    def dim(self, text: str) -> str:
        return self.paint(text, _DIM)

    def bold(self, text: str) -> str:
        return self.paint(text, _BOLD)

    @property
    def rule_char(self) -> str:
        return "─" if self.unicode else "-"

    @property
    def bar_chars(self):
        return ("█", "░") if self.unicode else ("#", ".")

    def fit(self, text: str) -> str:
        """Make text safe for the target stream. No-op when unicode is available."""
        if self.unicode:
            return text
        for src, dst in _TRANSLITERATE.items():
            text = text.replace(src, dst)
        return text.encode("ascii", "replace").decode("ascii")


_TRANSLITERATE = {
    "—": "-", "–": "-", "→": "->", "←": "<-", "·": "-", "…": "...",
    "“": '"', "”": '"', "‘": "'", "’": "'", "✓": "+", "✗": "x", "⚠": "!",
    "█": "#", "░": ".", "─": "-",
}


def _want_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        # VT processing is on by default in Windows Terminal; conhost sets neither.
        return bool(os.environ.get("WT_SESSION") or os.environ.get("TERM"))
    return True


def _want_unicode() -> bool:
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower()
    if "utf" in encoding:
        return True
    try:  # Python 3.7+: ask the stream to switch rather than mangling output
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        return True
    except Exception:
        return False


class AsciiStream:
    """Transliterating stdout proxy, for consoles that cannot encode box drawing.

    Cheaper than threading ``style.fit`` through every print in the CLI, and it
    also catches text we did not author (subprocess output is unaffected — that
    goes straight to the real file descriptor).
    """

    def __init__(self, stream, style: "Style"):
        self._stream = stream
        self._style = style

    def write(self, text):
        return self._stream.write(self._style.fit(text))

    def __getattr__(self, name):
        return getattr(self._stream, name)


def render(report: Report, style: Style, verbose: bool = False, width: int = 78) -> str:
    lines: List[str] = []
    rule = style.rule_char * width

    lines.append(style.bold("SHUT-YOUR-PYHOLE"))
    lines.append(style.dim(f"repository readiness audit  {report.root}"))
    lines.append(style.dim(f"target: {report.target}"))
    lines.append(style.dim(rule))

    for kind in SECTION_ORDER:
        reqs = report.by_kind(kind)
        if not reqs:
            continue
        if not verbose:
            reqs = [r for r in reqs if r.status is not Status.INFO or r.detail]
        lines.append(style.bold(SECTION_TITLES[kind]))
        for req in reqs:
            lines.extend(_render_requirement(req, style, verbose))
        lines.append("")

    if report.notes:
        lines.append(style.bold("Notes"))
        for note in report.notes:
            lines.append(f"  {style.dim(note)}")
        lines.append("")

    lines.append(style.dim(rule))
    lines.append(_render_score(report, style))
    lines.extend(_render_blockers(report, style))
    lines.extend(_render_next(report, style))
    return style.fit("\n".join(lines))


def _render_requirement(req: Requirement, style: Style, verbose: bool) -> List[str]:
    name = req.name
    if len(name) > NAME_WIDTH:
        # Paths are identified by their tail, prose by its head.
        if "/" in name:
            name = "..." + name[-(NAME_WIDTH - 3) :]
        else:
            name = name[: NAME_WIDTH - 3] + "..."
    line = f"  {style.status(req.status)} {name.ljust(NAME_WIDTH)}"
    detail = req.detail or ""
    if req.source and verbose:
        detail = f"{detail}  [{req.source}]" if detail else f"[{req.source}]"
    out = [line + ("  " + style.dim(detail) if detail else "")]

    if req.manual and req.status.is_blocker:
        out.append(f"      {style.dim('manual:')} {req.manual}")
    if verbose:
        if req.fix:
            out.append(f"      {style.dim('fix:')} {req.fix}")
        if req.explain:
            out.append(f"      {style.dim('note:')} {req.explain}")
        for item in req.meta.get("packages", []) if req.meta.get("verbose_list") else []:
            out.append(f"      {style.dim('- ' + str(item))}")
        for url in req.meta.get("urls", []) if req.meta.get("verbose_urls") else []:
            out.append(f"      {style.dim('- ' + url)}")
    return out


def _render_score(report: Report, style: Style) -> str:
    """Lead with the count that means something.

    The bar is a progress indicator whose denominator moves as detection
    improves, so the blocker count is the headline and the percentage is
    explicitly a ratio of checks, not a probability of working.
    """
    ratio = report.readiness
    full, empty = style.bar_chars
    filled = int(round(ratio * BAR_WIDTH))
    bar = full * filled + empty * (BAR_WIDTH - filled)
    blocking = len(report.blockers)
    color = (
        _COLORS[Status.OK]
        if blocking == 0
        else (_COLORS[Status.MISMATCH] if blocking <= 3 else _COLORS[Status.MISSING])
    )
    headline = style.paint(f"{blocking} blocker(s)", color)
    return (
        f"{headline}  {style.dim(bar)}  "
        f"{style.dim(f'{report.satisfied}/{len(report.scored)} checks satisfied')}"
    )


def group_blockers(blockers: List[Requirement]):
    """Collapse blockers that share one action into a single step.

    Three checkpoints fetched by the same script is one thing to do, not three,
    and a list of actions is more useful than a list of symptoms. Local fixes
    sort first, then downloads, then repo scripts — cheapest and safest first.
    """
    groups = []  # [(action, is_command, [requirement, ...])]
    index = {}
    for req in blockers:
        action = req.fix or req.manual or f"resolve {req.name}"
        key = (bool(req.fix), action)
        if key not in index:
            index[key] = len(groups)
            groups.append((action, bool(req.fix), []))
        groups[index[key]][2].append(req)
    order = {FixKind.LOCAL: 0, FixKind.NETWORK: 1, FixKind.SCRIPT: 2}
    groups.sort(key=lambda g: (not g[1], order.get(g[2][0].fix_kind, 3), -len(g[2])))
    return groups


def fix_label(req: Requirement, style: "Style") -> str:
    if req.fix_kind is FixKind.SCRIPT:
        return style.paint(" [runs repo code]", _COLORS[Status.MISMATCH])
    return ""


def _render_blockers(report: Report, style: Style, limit: int = 12) -> List[str]:
    blockers = report.blockers
    if not blockers:
        return ["", style.paint("Nothing is blocking a run. Try the smoke test: syp smoke", _COLORS[Status.OK])]

    auto = [b for b in blockers if b.fix]
    manual = [b for b in blockers if not b.fix]
    summary = f"{len(blockers)} blocker(s)"
    if auto:
        summary += f" · {len(auto)} fixable automatically"
    if manual:
        summary += f" · {len(manual)} need a human"

    lines = ["", style.bold("BLOCKERS") + "  " + style.dim(summary)]
    groups = group_blockers(blockers)
    for i, (action, is_command, reqs) in enumerate(groups[:limit], start=1):
        prefix = f"  {i}."
        if is_command:
            lines.append(f"{prefix} {action}{fix_label(reqs[0], style)}")
        else:
            lines.append(f"{prefix} {style.dim('[manual]')} {action}")
        names = ", ".join(r.name for r in reqs[:4])
        if len(reqs) > 4:
            names += f" (+{len(reqs) - 4} more)"
        lines.append(f"     {style.dim(names)}")
    if len(groups) > limit:
        lines.append(style.dim(f"  ... and {len(groups) - limit} more; see syp audit -v"))
    return lines


def _render_next(report: Report, style: Style) -> List[str]:
    commands = [g for g in group_blockers(report.blockers) if g[1]]
    gated = [r for r in report.requirements if r.status is Status.BLOCKED]
    lines = ["", style.bold("NEXT")]
    if commands:
        lines.append(f"  syp fix        {style.dim(f'run {len(commands)} command(s) that resolve blockers')}")
    if gated:
        term = gated[0].meta.get("gated") or os.path.basename(gated[0].name).split()[0]
        lines.append(f"  syp explain {term[:20]:<20}{style.dim('what this gate actually requires')}")
    lines.append(f"  syp audit -v   {style.dim('show sources, fixes and full lists')}")
    if not any(r.meta.get("observed") for r in report.requirements):
        lines.append(
            f"  syp trace      {style.dim('run the demo and record what it actually opens')}"
        )
    return lines
