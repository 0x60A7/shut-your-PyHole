"""Command line entry point: audit, fix, explain, smoke, trace."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import List, Optional

from . import __version__, trace as trace_mod
from .collect import COLLECTOR_NAMES, run_all
from .context import RepoContext
from .knowledge import AWKWARD_PACKAGES, GATED_ASSETS, HOST_HINTS
from .model import FixKind, Report, Requirement, Status
from .render import AsciiStream, Style, group_blockers, render

EXIT_OK = 0
EXIT_BLOCKED = 1
EXIT_ERROR = 2

COMMANDS = ("audit", "fix", "explain", "smoke", "trace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="syp",
        description="Repository readiness audit. Because `pip install -r requirements.txt` "
        "was never the whole story.",
    )
    parser.add_argument("--version", action="version", version=f"shut-your-pyhole {__version__}")
    # dest is `cmd`, not `command`: `syp trace --command ...` would collide.
    sub = parser.add_subparsers(dest="cmd")

    def common(p, with_path=True):
        if with_path:
            p.add_argument("path", nargs="?", default=".", help="repository to inspect (default: .)")
        p.add_argument("-v", "--verbose", action="store_true", help="show sources, fixes and full lists")
        p.add_argument("--ascii", action="store_true", help="ASCII symbols only")
        p.add_argument("--no-color", action="store_true")
        p.add_argument("--network", action="store_true", help="also make network checks")
        p.add_argument(
            "--target",
            metavar="T",
            help="environment to inspect: host, venv, image, or image:NAME (default: venv)",
        )
        p.add_argument("--trace-file", metavar="FILE", help="fold in a recorded run (default: newest in .syp/)")
        p.add_argument("--no-trace", action="store_true", help="ignore any recorded run")
        return p

    audit = common(sub.add_parser("audit", help="report what the repo needs and what is missing"))
    audit.add_argument("--json", action="store_true", help="machine-readable output")
    audit.add_argument("--only", action="append", metavar="COLLECTOR",
                       help="restrict to: " + " ".join(COLLECTOR_NAMES))
    audit.add_argument("--exit-zero", action="store_true", help="always exit 0")

    fix = common(sub.add_parser("fix", help="apply the fixes that are safe to automate"))
    fix.add_argument("--yes", action="store_true", help="actually run the commands (default: dry run)")
    fix.add_argument("--allow-scripts", action="store_true",
                     help="also run scripts belonging to the audited repo")

    explain = sub.add_parser("explain", help="explain one requirement in full")
    explain.add_argument("term", help="name or fragment, e.g. SMPL_NEUTRAL.pkl")
    common(explain)

    smoke = common(sub.add_parser("smoke", help="show (or run) the documented demo command"))
    smoke.add_argument("--run", action="store_true", help="execute it")
    smoke.add_argument("--timeout", type=int, default=1800, help="seconds before giving up (default 1800)")

    traced = common(sub.add_parser(
        "trace", help="run the demo under an audit hook and record what it really needs"))
    traced.add_argument("--command", metavar="CMD", help="command to run (default: the documented one)")
    traced.add_argument("--timeout", type=int, default=1800)
    traced.add_argument("--out", metavar="FILE", help="where to write the trace (default: .syp/trace.jsonl)")
    traced.add_argument("--keep", action="store_true", help="append to an existing trace instead of replacing it")
    return parser


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

    style = Style(color=False if args.no_color else None, unicode_ok=False if args.ascii else None)
    original_stdout = sys.stdout
    if not style.unicode:
        sys.stdout = AsciiStream(original_stdout, style)
    try:
        if args.cmd == "trace":
            return _cmd_trace(args, root, style)

        ctx = RepoContext.load(root, network=args.network, target_spec=args.target)
        ctx.trace = _load_trace(args, root, style)
        report = run_all(ctx, only=getattr(args, "only", None))

        if args.cmd == "audit":
            return _cmd_audit(args, report, style)
        if args.cmd == "fix":
            return _cmd_fix(args, report, style, root)
        if args.cmd == "explain":
            return _cmd_explain(args, report, style)
        if args.cmd == "smoke":
            return _cmd_smoke(args, report, style, root)
        parser.print_help()
        return EXIT_ERROR
    finally:
        sys.stdout = original_stdout


def _load_trace(args, root: str, style: Style):
    if getattr(args, "no_trace", False):
        return None
    path = getattr(args, "trace_file", None) or trace_mod.latest(root)
    if not path or not os.path.exists(path):
        return None
    recorded = trace_mod.load(path, root)
    return None if recorded.empty else recorded


# --- audit ------------------------------------------------------------------


def _cmd_audit(args, report: Report, style: Style) -> int:
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render(report, style, verbose=args.verbose))
    if args.exit_zero:
        return EXIT_OK
    return EXIT_BLOCKED if report.blockers else EXIT_OK


# --- fix --------------------------------------------------------------------


def _cmd_fix(args, report: Report, style: Style, root: str) -> int:
    manual = [r for r in report.blockers if not r.fix]
    groups = [(action, reqs) for action, is_cmd, reqs in group_blockers(report.blockers) if is_cmd]
    runnable = [(a, r) for a, r in groups if r[0].fix_kind is not FixKind.SCRIPT or args.allow_scripts]
    withheld = [(a, r) for a, r in groups if (a, r) not in runnable]

    if not groups:
        print("Nothing to fix automatically.")
        _print_manual(manual, style)
        return EXIT_BLOCKED if manual else EXIT_OK

    print(style.bold(f"{len(groups)} command(s) would resolve {sum(len(r) for _, r in groups)} blocker(s):"))
    for action, reqs in groups:
        tag = "" if reqs[0].fix_kind is not FixKind.SCRIPT else style.dim("  [repo script]")
        print(f"  {action}{tag}")
        print(f"      {style.dim(', '.join(r.name for r in reqs[:4]))}")

    if withheld:
        print()
        print(style.bold(f"{len(withheld)} withheld:") + style.dim(
            " these execute scripts from the audited repository, which is arbitrary code."))
        for action, _ in withheld:
            print(f"  {style.dim(action)}")
        print(style.dim("  Read them, then re-run with --allow-scripts."))

    if not args.yes:
        print()
        print(style.dim("Dry run. Re-run with --yes to execute."))
        _print_manual(manual, style)
        return EXIT_BLOCKED

    failures = 0
    for action, _ in runnable:
        print()
        print(style.bold(f"$ {action}"))
        code = subprocess.call(action, shell=True, cwd=root)
        if code != 0:
            failures += 1
            print(style.paint(f"  failed (exit {code})", "\033[31m"))
    print()
    print(f"{len(runnable) - failures}/{len(runnable)} command(s) succeeded.")
    _print_manual(manual, style)
    return EXIT_BLOCKED if failures or manual or withheld else EXIT_OK


def _print_manual(manual: List[Requirement], style: Style) -> None:
    if not manual:
        return
    print()
    print(style.bold(f"{len(manual)} thing(s) no command can do:"))
    for req in manual:
        print(f"  - {req.name}: {req.manual or req.detail}")


# --- explain ----------------------------------------------------------------


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
            kind = req.fix_kind.value if req.fix_kind else "?"
            print(f"   fix      {req.fix}  ({kind})")
        if req.manual:
            print(f"   manual   {req.manual}")
        if req.explain:
            print(f"   note     {req.explain}")
        for key in ("url", "script", "gated", "module", "distribution", "rule"):
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


# --- smoke / trace ----------------------------------------------------------


def _smoke_command(report: Report, ctx_config=None) -> Optional[Requirement]:
    return next((r for r in report.requirements if r.meta.get("smoke")), None)


def _cmd_smoke(args, report: Report, style: Style, root: str) -> int:
    entry = _smoke_command(report)
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
        print(style.dim("Re-run with --run to execute it, or `syp trace` to record what it needs."))
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


def _cmd_trace(args, root: str, style: Style) -> int:
    """Run the demo with an audit hook installed and fold the result back in."""
    ctx = RepoContext.load(root, network=args.network, target_spec=args.target)
    command = args.command or ctx.config.smoke_command
    if not command:
        report = run_all(ctx, only=["entrypoint"])
        entry = _smoke_command(report)
        if entry is None:
            print("No command to trace. Pass --command, or set smoke.command in .syp.toml.")
            return EXIT_ERROR
        command = entry.meta["command"]

    out = args.out or trace_mod.default_path(root)
    if not args.keep and os.path.exists(out):
        os.remove(out)

    print(style.bold("tracing") + f"  {command}")
    print(style.dim(f"  recording to {os.path.relpath(out, root)}"))
    print(style.dim("  the run may fail — that is the point; the trace records how far it got."))
    print()

    code = trace_mod.run_traced(command, cwd=root, trace_path=out, timeout=args.timeout)
    trace_mod.record_exit(out, code)
    recorded = trace_mod.load(out, root)

    print()
    if recorded.unsupported_interpreter:
        print(style.paint(
            f"  the traced interpreter is Python {recorded.python_version or '?'}, which predates "
            "sys.addaudithook (3.8+).", "\033[33m"))
        print(style.dim("  Nothing could be observed; the static audit is all you get here."))
        print()
    elif recorded.hook_active is None and code != 0:
        print(style.dim("  no hook events recorded — the command may not have started a Python process."))
        print()
    print(style.bold("observed"))
    print(f"  exit code       {code}")
    print(f"  paths opened    {len(recorded.opened)}")
    print(f"  paths missing   {len(recorded.missing)}")
    print(f"  modules         {len(recorded.imports)}")
    print(f"  subprocesses    {', '.join(sorted(recorded.executables)) or 'none'}")
    print(f"  network         {', '.join(sorted(recorded.hosts)) or 'none'}")
    if recorded.missing:
        print()
        print(style.bold("opened but absent"))
        for path in recorded.missing[:15]:
            print(f"  {style.status(Status.MISSING)} {path}")
        if len(recorded.missing) > 15:
            print(style.dim(f"  ... and {len(recorded.missing) - 15} more"))
    print()
    print(style.dim("Folded into the next `syp audit` automatically."))
    return EXIT_OK if code == 0 else EXIT_BLOCKED


if __name__ == "__main__":
    sys.exit(main())
