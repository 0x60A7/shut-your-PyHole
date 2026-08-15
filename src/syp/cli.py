"""Command line entry point: audit, fix, explain, smoke."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import List, Optional

from . import __version__
from .collect import run_all
from .context import RepoContext
from .knowledge import AWKWARD_PACKAGES, GATED_ASSETS, HOST_HINTS
from .model import Report, Requirement, Status
from .render import AsciiStream, Style, group_blockers, render

EXIT_OK = 0
EXIT_BLOCKED = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="syp",
        description="Repository readiness audit. Because `pip install -r requirements.txt` "
        "was never the whole story.",
    )
    parser.add_argument("--version", action="version", version=f"shut-your-pyhole {__version__}")
    sub = parser.add_subparsers(dest="command")

    def common(p, with_path=True):
        if with_path:
            p.add_argument("path", nargs="?", default=".", help="repository to inspect (default: .)")
        p.add_argument("-v", "--verbose", action="store_true", help="show sources, fixes and full lists")
        p.add_argument("--ascii", action="store_true", help="ASCII symbols only")
        p.add_argument("--no-color", action="store_true")
        return p

    audit = common(sub.add_parser("audit", help="report what the repo needs and what is missing"))
    audit.add_argument("--json", action="store_true", help="machine-readable output")
    audit.add_argument("--network", action="store_true", help="also check that download URLs resolve")
    audit.add_argument("--only", action="append", metavar="COLLECTOR",
                       help="restrict to one of: system git container python assets entrypoint")
    audit.add_argument("--python", metavar="EXE", help="interpreter to check packages against")
    audit.add_argument("--exit-zero", action="store_true", help="always exit 0")

    fix = common(sub.add_parser("fix", help="apply the fixes that are safe to automate"))
    fix.add_argument("--yes", action="store_true", help="actually run the commands (default: dry run)")
    fix.add_argument("--network", action="store_true")
    fix.add_argument("--python", metavar="EXE")

    # `term` comes first here, so `syp explain SMPL_NEUTRAL.pkl` reads naturally.
    explain = sub.add_parser("explain", help="explain one requirement in full")
    explain.add_argument("term", help="name or fragment, e.g. SMPL_NEUTRAL.pkl")
    common(explain)
    explain.add_argument("--network", action="store_true")
    explain.add_argument("--python", metavar="EXE")

    smoke = common(sub.add_parser("smoke", help="show (or run) the documented demo command"))
    smoke.add_argument("--run", action="store_true", help="execute it")
    smoke.add_argument("--timeout", type=int, default=1800, help="seconds before giving up (default 1800)")
    smoke.add_argument("--python", metavar="EXE")
    smoke.add_argument("--network", action="store_true")
    return parser


COMMANDS = ("audit", "fix", "explain", "smoke")


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    # `syp` and `syp <path>` both mean `syp audit <path>`.
    if not raw or (raw[0] not in COMMANDS and raw[0] not in ("-h", "--help", "--version")):
        raw = ["audit"] + raw
    args = parser.parse_args(raw)

    root = os.path.abspath(args.path)
    if not os.path.isdir(root):
        print(f"syp: not a directory: {root}", file=sys.stderr)
        return EXIT_ERROR

    if getattr(args, "python", None):
        os.environ["SYP_PYTHON"] = args.python

    style = Style(color=False if args.no_color else None, unicode_ok=False if args.ascii else None)
    ctx = RepoContext.load(root, network=getattr(args, "network", False))
    report = run_all(ctx, only=getattr(args, "only", None))

    original_stdout = sys.stdout
    if not style.unicode:
        sys.stdout = AsciiStream(original_stdout, style)
    try:
        if args.command == "audit":
            return _cmd_audit(args, report, style)
        if args.command == "fix":
            return _cmd_fix(args, report, style, root)
        if args.command == "explain":
            return _cmd_explain(args, report, style)
        if args.command == "smoke":
            return _cmd_smoke(args, report, style, root)
        parser.print_help()
        return EXIT_ERROR
    finally:
        sys.stdout = original_stdout


def _cmd_audit(args, report: Report, style: Style) -> int:
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render(report, style, verbose=args.verbose))
    if args.exit_zero:
        return EXIT_OK
    return EXIT_BLOCKED if report.blockers else EXIT_OK


def _cmd_fix(args, report: Report, style: Style, root: str) -> int:
    manual = [r for r in report.blockers if not r.fix]
    # Group first: one script that supplies four checkpoints is one command.
    commands = [
        (action, reqs) for action, is_command, reqs in group_blockers(report.blockers) if is_command
    ]

    if not commands:
        print("Nothing to fix automatically.")
        _print_manual(manual, style)
        return EXIT_BLOCKED if manual else EXIT_OK

    print(style.bold(f"{len(commands)} command(s) would resolve {sum(len(r) for _, r in commands)} blocker(s):"))
    for action, reqs in commands:
        print(f"  {action}")
        print(f"      {style.dim(', '.join(r.name for r in reqs[:4]))}")

    if not args.yes:
        print()
        print(style.dim("Dry run. Re-run with --yes to execute."))
        print(style.dim("These commands clone, download and install. Read them first."))
        _print_manual(manual, style)
        return EXIT_BLOCKED

    failures = 0
    for action, _ in commands:
        print()
        print(style.bold(f"$ {action}"))
        code = subprocess.call(action, shell=True, cwd=root)
        if code != 0:
            failures += 1
            print(style.paint(f"  failed (exit {code})", "\033[31m"))
    print()
    print(f"{len(commands) - failures}/{len(commands)} command(s) succeeded.")
    _print_manual(manual, style)
    return EXIT_BLOCKED if failures or manual else EXIT_OK


def _print_manual(manual: List[Requirement], style: Style) -> None:
    if not manual:
        return
    print()
    print(style.bold(f"{len(manual)} thing(s) no command can do:"))
    for req in manual:
        print(f"  - {req.name}: {req.manual or req.detail}")


def _cmd_explain(args, report: Report, style: Style) -> int:
    term = args.term.lower()
    hits = [r for r in report.requirements if term in r.name.lower() or term in (r.detail or "").lower()]
    if not hits:
        hits = [r for r in report.requirements if term in json.dumps(r.meta, default=str).lower()]

    if not hits:
        if not _explain_knowledge(term, style):
            print(f"Nothing in this repo matches '{args.term}'.")
            return EXIT_ERROR
        return EXIT_OK

    for req in hits[:5]:
        print(style.bold(f"{style.status(req.status)} {req.name}"))
        print(f"   status   {req.status.value}")
        if req.detail:
            print(f"   detail   {req.detail}")
        if req.source:
            print(f"   declared {req.source}")
        if req.fix:
            print(f"   fix      {req.fix}")
        if req.manual:
            print(f"   manual   {req.manual}")
        if req.explain:
            print(f"   note     {req.explain}")
        for key in ("url", "script", "gated"):
            if req.meta.get(key):
                print(f"   {key:<8} {req.meta[key]}")
        print()
    _explain_knowledge(term, style)
    return EXIT_OK


def _explain_knowledge(term: str, style: Style) -> bool:
    found = False
    for entry in GATED_ASSETS:
        # `SMPL_NEUTRAL.pkl`, `SMPL_NEUTRAL` and `smpl` should all land here.
        if (
            entry.key in term
            or term in entry.key
            or any(re.search(p, term, re.IGNORECASE) for p in entry.patterns)
        ):
            print(style.bold(f"Licence gate: {entry.provider}"))
            print(f"   url      {entry.url}")
            print(f"   requires {entry.requires}")
            if entry.note:
                print(f"   note     {entry.note}")
            print()
            found = True
    for entry in HOST_HINTS:
        if term == entry.key or term in entry.label.lower():
            print(style.bold(f"Host: {entry.label}"))
            print(f"   note     {entry.note}")
            print()
            found = True
    for name, pkg in AWKWARD_PACKAGES.items():
        if term == name or term in name:
            print(style.bold(f"Package: {name}"))
            print(f"   note     {pkg.note}")
            if pkg.hint:
                print(f"   hint     {pkg.hint}")
            print()
            found = True
    return found


def _cmd_smoke(args, report: Report, style: Style, root: str) -> int:
    entry = next((r for r in report.requirements if r.meta.get("smoke")), None)
    if entry is None:
        print("No documented demo command found. Run `syp audit -v` to see what was inspected.")
        return EXIT_ERROR

    command = entry.meta["command"]
    blockers = report.blockers
    print(style.bold("smoke test") + f"  {command}")
    print(style.dim(f"  documented in {entry.source}"))
    if blockers:
        print()
        print(style.paint(f"  {len(blockers)} unresolved blocker(s); this will probably fail:", "\033[33m"))
        for req in blockers[:5]:
            print(f"    - {req.name}")

    if not args.run:
        print()
        print(style.dim("Re-run with --run to execute it."))
        return EXIT_BLOCKED if blockers else EXIT_OK

    print()
    print(style.bold(f"$ {command}"))
    try:
        code = subprocess.call(command, shell=True, cwd=root, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        print(style.paint(f"  timed out after {args.timeout}s", "\033[33m"))
        return EXIT_BLOCKED
    print()
    if code == 0:
        print(style.paint("  smoke test passed", "\033[32m"))
        return EXIT_OK
    print(style.paint(f"  smoke test failed (exit {code})", "\033[31m"))
    return EXIT_BLOCKED


if __name__ == "__main__":
    sys.exit(main())
