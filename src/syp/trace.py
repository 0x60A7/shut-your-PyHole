"""Runtime observation.

Static scanning guesses what a program will need. This watches what it actually
asked for: every path opened, module imported, binary spawned and host dialled,
recorded up to the moment it crashed. A path the program opened and did not
find is not a heuristic — it is the failure, named.

The hook is installed in the *child* process via a generated `sitecustomize.py`
on PYTHONPATH, so it applies to `python demo.py`, to a shell script that calls
python, and to subprocesses of either.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .util import read_text

# Written verbatim into the child's sitecustomize.py. Deliberately dependency-free
# and defensive: a crash in here must never be mistaken for a crash in the repo.
HOOK_SOURCE = r'''
import os, sys

_TRACE_FD = None
_SEEN = set()
_BUSY = [False]
_LIMIT = 20000


def _emit(kind, value, extra=None):
    # Writing triggers `open`/`write` audit events; guard against recursion.
    if _BUSY[0] or _TRACE_FD is None:
        return
    key = (kind, value)
    if key in _SEEN or len(_SEEN) > _LIMIT:
        return
    _SEEN.add(key)
    _BUSY[0] = True
    try:
        line = '{"kind": %s, "value": %s, "extra": %s}\n' % (
            _json(kind), _json(value), _json(extra or ""))
        os.write(_TRACE_FD, line.encode("utf-8", "replace"))
    except Exception:
        pass
    finally:
        _BUSY[0] = False


def _json(value):
    text = str(value)
    out = ['"']
    for ch in text:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch in "\n\r\t":
            out.append(" ")
        elif ord(ch) < 32:
            out.append(" ")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _hook(event, args):
    try:
        if event == "open":
            path, mode = args[0], args[1]
            if isinstance(path, (str, bytes)) and (mode is None or "r" in str(mode or "r")):
                _emit("open", os.fsdecode(path))
        elif event == "import":
            _emit("import", args[0])
        elif event == "subprocess.Popen":
            # Signature is (executable, args, cwd, env); executable is None
            # whenever the caller passed a list, which is almost always.
            exe = args[0]
            if not exe and len(args) > 1:
                argv = args[1]
                if isinstance(argv, (list, tuple)) and argv:
                    exe = argv[0]
                elif isinstance(argv, (str, bytes)):
                    exe = os.fsdecode(argv).split()[0] if argv else None
            if exe:
                _emit("exec", os.fsdecode(exe))
        elif event == "os.system":
            _emit("exec", os.fsdecode(args[0]))
        elif event == "socket.connect":
            address = args[1]
            if isinstance(address, tuple) and address:
                _emit("connect", address[0])
        elif event == "urllib.Request":
            _emit("url", args[0])
    except Exception:
        pass


def _install():
    global _TRACE_FD
    path = os.environ.get("SYP_TRACE_FILE")
    if not path:
        return
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
    try:
        _TRACE_FD = os.open(path, flags, 0o644)
    except Exception:
        return
    _emit("python", sys.version.split()[0])
    # sys.addaudithook needs 3.8. Older interpreters are common in research code,
    # so record whether the hook took: an empty trace must not look like a run
    # that opened nothing.
    try:
        sys.addaudithook(_hook)
        _emit("hook", "active")
    except Exception:
        _emit("hook", "unavailable")


_install()
'''


@dataclass
class Trace:
    """Parsed trace events, resolved against the repository."""

    opened: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    imports: Set[str] = field(default_factory=set)
    executables: Set[str] = field(default_factory=set)
    hosts: Set[str] = field(default_factory=set)
    urls: Set[str] = field(default_factory=set)
    command: str = ""
    exit_code: Optional[int] = None
    path: Optional[str] = None
    python_version: Optional[str] = None
    hook_active: Optional[bool] = None

    @property
    def empty(self) -> bool:
        return not (self.opened or self.imports or self.executables)

    @property
    def unsupported_interpreter(self) -> bool:
        """The child ran, but was too old to be observed."""
        return self.hook_active is False


def hook_dir() -> str:
    """Create a throwaway directory containing sitecustomize.py with the hook."""
    directory = tempfile.mkdtemp(prefix="syp-hook-")
    with open(os.path.join(directory, "sitecustomize.py"), "w", encoding="utf-8") as fh:
        fh.write(HOOK_SOURCE)
    return directory


def run_traced(command: str, cwd: str, trace_path: str, timeout: int = 1800) -> int:
    """Run ``command`` with the audit hook installed, appending events to a file."""
    directory = hook_dir()
    env = dict(os.environ)
    env["SYP_TRACE_FILE"] = os.path.abspath(trace_path)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = directory + (os.pathsep + existing if existing else "")
    # Some repos set PYTHONDONTWRITEBYTECODE/-S; sitecustomize needs site enabled.
    env.pop("PYTHONNOUSERSITE", None)
    with open(trace_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "command", "value": command, "extra": cwd}) + "\n")
    try:
        return subprocess.call(command, shell=True, cwd=cwd, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return -9


def record_exit(trace_path: str, code: int) -> None:
    with open(trace_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "exit", "value": str(code), "extra": ""}) + "\n")


# --- reading ----------------------------------------------------------------

_NOISE_PREFIXES = ("<", "/proc/", "/sys/", "/dev/", "/etc/")
# Note: `.pth` is deliberately absent. It means "setuptools path file" inside
# site-packages and "PyTorch checkpoint" everywhere else, and the site-packages
# case is already covered by the interpreter-path hints below.
_NOISE_SUFFIXES = (".pyc", ".pyi", ".dist-info", ".egg-link")
_STDLIB_HINTS = ("site-packages", "dist-packages", "lib/python", "Lib\\", "python3.", "conda")


def load(path: str, root: str) -> Trace:
    trace = Trace(path=path)
    root_abs = os.path.abspath(root)
    for line in read_text(path, 8_000_000).splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        kind, value = event.get("kind"), event.get("value") or ""
        if kind == "command":
            trace.command = value
        elif kind == "exit":
            try:
                trace.exit_code = int(value)
            except ValueError:
                pass
        elif kind == "open":
            rel = _relevant_path(value, root_abs)
            if rel:
                trace.opened.append(rel)
                if not os.path.exists(os.path.join(root_abs, rel.replace("/", os.sep))):
                    trace.missing.append(rel)
        elif kind == "import":
            top = value.split(".")[0]
            if top:
                trace.imports.add(top)
        elif kind == "exec":
            name = os.path.basename(value).lower()
            if name and not name.startswith("python"):
                trace.executables.add(name)
        elif kind == "connect":
            trace.hosts.add(value)
        elif kind == "url":
            trace.urls.add(value)
        elif kind == "python":
            trace.python_version = value
        elif kind == "hook":
            trace.hook_active = value == "active"
    trace.opened = list(dict.fromkeys(trace.opened))
    trace.missing = list(dict.fromkeys(trace.missing))
    return trace


def _relevant_path(value: str, root_abs: str) -> Optional[str]:
    """Keep only repo-relative paths; drop the interpreter's own churn."""
    if not value or value.startswith(_NOISE_PREFIXES) or value.endswith(_NOISE_SUFFIXES):
        return None
    if any(hint in value for hint in _STDLIB_HINTS):
        return None
    path = value.replace("\\", "/")
    if os.path.isabs(value):
        try:
            rel = os.path.relpath(value, root_abs)
        except ValueError:  # different drive on Windows
            return None
        if rel.startswith(".."):
            return None
        path = rel.replace(os.sep, "/")
    path = path.lstrip("./")
    if not path or path.startswith(".git/"):
        return None
    return path


def latest(root: str) -> Optional[str]:
    """The most recent trace recorded for this repo, if any."""
    directory = os.path.join(root, ".syp")
    if not os.path.isdir(directory):
        return None
    traces = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.startswith("trace") and name.endswith(".jsonl")
    ]
    if not traces:
        return None
    return max(traces, key=lambda p: os.path.getmtime(p))


def default_path(root: str) -> str:
    directory = os.path.join(root, ".syp")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, "trace.jsonl")
