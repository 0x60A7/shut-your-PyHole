"""What the code actually imports, and what those imports drag in with them.

`pydeps` checks declarations against an environment. This checks the *code*
against both: a module imported by the source but named in no manifest is the
classic research-repo failure, and it is invisible to every manifest parser.

Three findings come out of one AST pass:
  - imports that nothing declares and the target cannot satisfy;
  - shared libraries those imports need (libGL and friends);
  - weights the code downloads at run time, which look like nothing in the source.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

from ..context import RepoContext
from ..knowledge import HUB_DOWNLOADS, SYSTEM_LIBS, match_system_libs
from ..model import Kind, Report, Requirement, Status
from ..util import normalize_dist, run
from ..reach import reachable
from .assets import is_test_file
from .entrypoint import entry_file
from .pydeps import gather

# Modules that are never a dependency: the interpreter's own, and the repo's.
_FALLBACK_STDLIB = {
    "abc", "argparse", "ast", "asyncio", "base64", "collections", "contextlib", "copy",
    "csv", "ctypes", "dataclasses", "datetime", "decimal", "difflib", "enum", "errno",
    "functools", "gc", "glob", "gzip", "hashlib", "heapq", "html", "http", "importlib",
    "inspect", "io", "itertools", "json", "logging", "math", "multiprocessing", "operator",
    "os", "pathlib", "pickle", "platform", "pprint", "queue", "random", "re", "shutil",
    "signal", "socket", "sqlite3", "string", "struct", "subprocess", "sys", "tarfile",
    "tempfile", "textwrap", "threading", "time", "traceback", "types", "typing", "unittest",
    "urllib", "uuid", "warnings", "weakref", "xml", "zipfile", "zlib", "__future__",
    # site machinery, including the hook `syp trace` installs itself
    "site", "sitecustomize", "usercustomize", "_distutils_hack",
}

# import name -> distribution name, where they differ and it matters.
IMPORT_TO_DIST = {
    "cv2": "opencv-python", "PIL": "pillow", "sklearn": "scikit-learn",
    "skimage": "scikit-image", "yaml": "pyyaml", "OpenGL": "pyopengl",
    "torch": "torch", "torchvision": "torchvision", "np": "numpy",
    "mmcv": "mmcv", "pycocotools": "pycocotools", "Cython": "cython",
    "google": "protobuf", "pkg_resources": "setuptools", "dateutil": "python-dateutil",
    "smplx": "smplx", "chumpy": "chumpy", "trimesh": "trimesh", "joblib": "joblib",
    "tensorboardX": "tensorboardx", "wandb": "wandb", "einops": "einops",
    "hydra": "hydra-core", "omegaconf": "omegaconf", "yacs": "yacs", "gdown": "gdown",
    "ultralytics": "ultralytics", "timm": "timm", "transformers": "transformers",
}


def collect(ctx: RepoContext, report: Report) -> None:
    # Scope imports the same way assets are scoped: to the run being audited.
    entry = entry_file(ctx)
    reached = reachable(ctx, entry) if entry else set()
    imported = _scan_imports(ctx, reached)
    if not imported:
        return

    local = _local_modules(ctx)
    stdlib = _stdlib_names()
    third_party = {
        name: sources
        for name, sources in imported.items()
        if name not in stdlib and name not in local and not name.startswith("_")
    }
    if ctx.trace is not None:
        # Observed imports outrank inferred ones: they definitely happened.
        for name in getattr(ctx.trace, "imports", ()):  # type: ignore[attr-defined]
            if name not in stdlib and name not in local and name not in third_party:
                third_party[name] = ["observed at runtime"]

    declared = {normalize_dist(n) for n in gather(ctx).packages}
    installed = _installed_modules(ctx)

    _report_undeclared(ctx, report, third_party, declared, installed)
    _report_system_libs(ctx, report, third_party)
    _report_hub_downloads(ctx, report)


# --- scanning ---------------------------------------------------------------


# Maintainer tooling and illustrations: their imports are not the project's.
_ILLUSTRATIVE = re.compile(
    r"^(docs?|examples?|notebooks?|benchmarks?|scripts|tools|dev)/", re.IGNORECASE
)


def _subproject_dirs(ctx: RepoContext) -> Set[str]:
    """Top-level directories that are their own project, with their own manifest.

    peft ships `method_comparison/`, a Gradio app with its own requirements. Its
    imports are that app's dependencies, not peft's.
    """
    out: Set[str] = set()
    for rel in ctx.files:
        parts = rel.split("/")
        if len(parts) < 2 or parts[0] in ("src", "lib", "tests", "test"):
            continue
        base = parts[-1].lower()
        # `requirements-app.txt` counts too: peft's nested app is declared that way.
        if base in ("pyproject.toml", "setup.py", "environment.yml") or (
            base.startswith("requirements") and base.endswith((".txt", ".in"))
        ):
            out.add(parts[0])
    return out


def _guarded_lines(tree: ast.AST) -> set:
    """Line numbers of imports the code already treats as optional.

    `try: import bitsandbytes / except ImportError:` is how every library
    declares an optional backend. Reporting those as missing dependencies turned
    peft into 45 blockers, all of them things peft works fine without.
    """
    guarded = set()
    for node in ast.walk(tree):
        bodies = []
        if isinstance(node, ast.Try):
            bodies = [node.body]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bodies = [node.body]  # a deferred import is an optional one
        for body in bodies:
            for inner in body:
                for sub in ast.walk(inner):
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        guarded.add(sub.lineno)
    return guarded


def _scan_imports(ctx: RepoContext, reached: Set[str]) -> Dict[str, List[str]]:
    """Top-level module names imported by the code this run reaches."""
    found: Dict[str, List[str]] = {}
    subprojects = _subproject_dirs(ctx)
    for rel in ctx.text_files((".py",)):
        if is_test_file(rel) or _is_vendored(rel):
            continue
        # Illustrative code, maintainer tooling and nested projects import
        # whatever they need; none of it is a requirement of this project.
        if _ILLUSTRATIVE.match(rel) and rel not in reached:
            continue
        if rel.split("/")[0] in subprojects and rel not in reached:
            continue
        if reached and rel not in reached:
            continue
        tree = ctx.parse(rel)
        if tree is None:
            continue  # recorded centrally; reported rather than silently dropped
        guarded = _guarded_lines(tree)
        for node in ast.walk(tree):
            if node.lineno in guarded if hasattr(node, "lineno") else False:
                continue
            if isinstance(node, ast.Import):
                for alias in node.names:
                    _record(found, alias.name.split(".")[0], rel, node.lineno)
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import: local by definition
                    continue
                if node.module:
                    _record(found, node.module.split(".")[0], rel, node.lineno)
    return found


def _record(found: Dict[str, List[str]], name: str, rel: str, lineno: int) -> None:
    if not name:
        return
    entries = found.setdefault(name, [])
    if len(entries) < 3:
        entries.append(f"{rel}:{lineno}")


def _is_vendored(rel: str) -> bool:
    lowered = rel.lower()
    return lowered.startswith(("third-party/", "third_party/", "external/", "vendor/", "submodules/"))


def _local_modules(ctx: RepoContext) -> Set[str]:
    """Top-level names that resolve inside the repo, so are never dependencies.

    Any directory containing Python anywhere below it counts: research repos are
    full of namespace packages with no `__init__.py`, and calling `import lib` a
    missing dependency would be worse than occasionally shadowing a real one.
    """
    local: Set[str] = set()
    for rel in ctx.files:
        if not rel.endswith(".py"):
            continue
        parts = rel.split("/")
        # A script imports its siblings by bare name, wherever it lives:
        # `app.py` next to `utils.py` does `import utils`.
        local.add(parts[-1][:-3])
        if len(parts) == 1:
            local.add(parts[0][:-3])
        else:
            local.add(parts[0])
            # A src/ layout hides the importable name one level down.
            if parts[0] in ("src", "lib") and len(parts) >= 3:
                local.add(parts[1])
    return local


def _stdlib_names() -> Set[str]:
    names = set(getattr(sys, "stdlib_module_names", ()) or ())
    return (names | _FALLBACK_STDLIB) if names else _FALLBACK_STDLIB


# packages_distributions() only exists from Python 3.10, and research code runs
# on 3.7-3.9 constantly. Without the fallbacks this returns nothing there, and
# every import in the repo looks uninstalled.
_MODULES_PROBE = r"""
import json, os, sys
names = set()
try:
    import importlib.metadata as m
except ImportError:
    import importlib_metadata as m
mapping = getattr(m, "packages_distributions", None)
if mapping:
    names.update(mapping().keys())
else:
    for dist in m.distributions():
        try:
            top = dist.read_text("top_level.txt")
        except Exception:
            top = None
        if top:
            names.update(line.strip() for line in top.splitlines() if line.strip())
        name = (dist.metadata["Name"] or "").replace("-", "_")
        if name:
            names.add(name)
# Editable installs and hand-dropped packages have no metadata at all.
for entry in sys.path:
    if not entry or not os.path.isdir(entry):
        continue
    if "site-packages" not in entry and "dist-packages" not in entry:
        continue
    try:
        listing = os.listdir(entry)
    except OSError:
        continue
    for item in listing:
        if item.endswith(".py"):
            names.add(item[:-3])
        elif "." not in item and not item.startswith("_"):
            names.add(item)
print(json.dumps(sorted(n for n in names if n)))
"""


def _installed_modules(ctx: RepoContext) -> Optional[Set[str]]:
    """Top-level importable names in the target environment."""
    code, out = ctx.target.python(["-c", _MODULES_PROBE], timeout=120)
    if code != 0:
        return None
    try:
        return set(json.loads(out.strip().splitlines()[-1]))
    except (ValueError, IndexError):
        return None


# --- reporting --------------------------------------------------------------


def _report_undeclared(
    ctx: RepoContext,
    report: Report,
    third_party: Dict[str, List[str]],
    declared: Set[str],
    installed: Optional[Set[str]],
) -> None:
    assumed = set(ctx.config.assume_installed)
    undeclared: List[Tuple[str, List[str]]] = []
    for name, sources in sorted(third_party.items()):
        dist = normalize_dist(IMPORT_TO_DIST.get(name, name))
        if dist in declared or normalize_dist(name) in declared or dist in assumed:
            continue
        if installed is not None and name in installed:
            # Present in the environment but declared nowhere: it will not survive
            # a fresh install on another machine.
            undeclared.append((name, sources))
            continue
        undeclared.append((name, sources))

    for name, sources in undeclared:
        dist = IMPORT_TO_DIST.get(name, name)
        present = installed is not None and name in installed
        report.add(
            Requirement(
                kind=Kind.PYTHON,
                name=f"import {name}",
                status=Status.MISMATCH if present else Status.MISSING,
                detail="imported by the code but declared in no manifest"
                + (" (installed here, so a fresh environment would break)" if present else ""),
                source=sources[0],
                fix=None if present else ctx.target.pip_command(dist),
                explain=f"Distribution is probably `{dist}`." if dist != name else None,
                meta={"module": name, "distribution": dist},
            )
        )


def _report_system_libs(ctx: RepoContext, report: Report, third_party: Dict[str, List[str]]) -> None:
    libs = match_system_libs(third_party)
    if not libs:
        return
    probe = _ldconfig(ctx)
    for lib in libs:
        source = third_party[lib.module][0]
        missing = [name for name in lib.libs if probe is not None and not _has_lib(probe, name)]
        if probe is None:
            report.add(
                Requirement(
                    kind=Kind.SYSTEM,
                    name=f"{lib.libs[0]} (for {lib.module})",
                    status=Status.UNKNOWN,
                    detail="cannot check shared libraries on this platform"
                    if not ctx.target.is_container
                    else "could not run ldconfig in the image",
                    source=source,
                    manual=f"On Linux: apt-get install {' '.join(lib.apt)}",
                    explain=lib.note or None,
                )
            )
        elif missing:
            # apt-get is only a *fix* on a Linux host we are actually auditing.
            # Inside an image it would not persist, and elsewhere it is fiction.
            apt = " ".join(lib.apt)
            on_linux_host = sys.platform.startswith("linux") and not ctx.target.is_container
            report.add(
                Requirement(
                    kind=Kind.SYSTEM,
                    name=f"{missing[0]} (for {lib.module})",
                    status=Status.MISSING,
                    detail=f"import {lib.module} will fail with a missing shared object",
                    source=source,
                    fix=f"sudo apt-get install -y {apt}" if on_linux_host else None,
                    manual=None
                    if on_linux_host
                    else (
                        f"The image is missing it: rebuild with `apt-get install -y {apt}`, "
                        "or use an image that has it."
                        if ctx.target.is_container
                        else f"On a Debian-family host: apt-get install -y {apt}"
                    ),
                    explain=lib.note or None,
                )
            )
        else:
            report.add(
                Requirement(
                    kind=Kind.SYSTEM,
                    name=f"{lib.libs[0]} (for {lib.module})",
                    status=Status.OK,
                    detail="shared library present",
                    source=source,
                )
            )


def _ldconfig(ctx: RepoContext) -> Optional[str]:
    """`ldconfig -p` output from the target, or None where the concept does not apply."""
    if ctx.target.is_container:
        code, out = ctx.target.run(["sh", "-lc", "ldconfig -p 2>/dev/null || true"], timeout=120)
        return out if code == 0 and out.strip() else None
    if not sys.platform.startswith("linux"):
        return None
    code, out = run(["ldconfig", "-p"], timeout=30)
    return out if code == 0 else None


def _has_lib(probe: str, name: str) -> bool:
    return name.split(".so")[0] + ".so" in probe


def _report_hub_downloads(ctx: RepoContext, report: Report) -> None:
    """Weights fetched at run time — invisible to every manifest, fatal offline."""
    for entry in HUB_DOWNLOADS:
        pattern = re.compile(entry.pattern)
        hit = None
        for rel in ctx.text_files((".py", ".ipynb")):
            if is_test_file(rel) or _is_vendored(rel):
                continue
            match = pattern.search(ctx.text(rel))
            if match:
                hit = ctx.source_ref(rel, match.group(0)[:40])
                break
        if not hit:
            continue
        report.add(
            Requirement(
                kind=Kind.EXTERNAL,
                name=entry.label,
                status=Status.INFO,
                detail=entry.note,
                source=hit,
                explain=(
                    f"Cache location follows ${entry.cache_env}; pre-populate it for offline runs."
                    if entry.cache_env
                    else "Requires network access on first run."
                ),
                meta={"hub": entry.key, "cache_env": entry.cache_env},
            )
        )
