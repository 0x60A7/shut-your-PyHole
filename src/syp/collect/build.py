"""Compilation steps hiding inside the repository.

`pip install -r requirements.txt` succeeding says nothing about the CUDA
extension in `lib/ops/` that has to be built against the installed torch, with
a matching nvcc and a compiler the toolkit will accept. That step is usually
one line in a README and never in a manifest.
"""

from __future__ import annotations

import os
import re
import sys
from typing import List, Optional, Tuple

from .. import makefile
from ..context import RepoContext
from ..model import Kind, Report, Requirement, Status
from ..util import run, which

CUDA_SOURCES = (".cu", ".cuh")
NATIVE_SOURCES = (".c", ".cc", ".cpp", ".cxx", ".pyx", ".pxd")

_EXT_MARKERS = (
    "CUDAExtension", "CppExtension", "BuildExtension", "cythonize",
    "Extension(", "build_ext", "cpp_extension.load", "load_inline",
)
_EDITABLE_INSTALL = re.compile(r"(?:pip|python -m pip)\s+install\s+(?:-v\s+)?-e\s+(\S+)")
_SETUP_BUILD = re.compile(r"python\s+setup\.py\s+(build\w*|install|develop)")


def collect(ctx: RepoContext, report: Report) -> None:
    _collect_makefiles(ctx, report)
    cuda_files = [f for f in ctx.files if f.lower().endswith(CUDA_SOURCES)]
    native_files = [f for f in ctx.files if f.lower().endswith(NATIVE_SOURCES)]
    markers = _extension_markers(ctx)
    cmake = [f for f in ctx.files if f.split("/")[-1] == "CMakeLists.txt"]
    steps = _documented_build_steps(ctx)

    if not (cuda_files or markers or cmake or steps):
        return

    if markers:
        rel, marker = markers[0]
        report.add(
            Requirement(
                kind=Kind.BUILD,
                name="native extension",
                status=Status.INFO,
                detail=f"{marker} in {rel}"
                + (f"; {len(cuda_files)} CUDA source(s)" if cuda_files else ""),
                source=rel,
                explain="This must be compiled in the same environment as the torch it links against.",
            )
        )
    elif cuda_files:
        report.add(
            Requirement(
                kind=Kind.BUILD,
                name="CUDA sources",
                status=Status.INFO,
                detail=f"{len(cuda_files)} file(s), e.g. {cuda_files[0]}",
                source=cuda_files[0],
            )
        )

    for command, source in steps[:3]:
        report.add(
            Requirement(
                kind=Kind.BUILD,
                name="build step",
                status=Status.INFO,
                detail=command if len(command) < 80 else command[:77] + "...",
                source=source,
                explain="Documented in the project's own instructions; easy to skip and fatal to omit.",
            )
        )

    _check_compiler(ctx, report, bool(native_files or cuda_files or markers))
    if cuda_files or any("CUDA" in m for _, m in markers):
        _check_nvcc_matches_torch(ctx, report)
    if any("cpp_extension.load" in m or "load_inline" in m for _, m in markers):
        _check_tool(ctx, report, "ninja", "torch's JIT extension loader shells out to ninja")
    if cmake:
        # A CMakeLists under tools/ or examples/ builds an optional extra, not
        # the package: detectron2's only one is for its C++ deploy sample.
        core = [c for c in cmake if not _PERIPHERAL_BUILD.match(c)]
        _check_tool(
            ctx, report, "cmake",
            f"{(core or cmake)[0]} needs cmake to configure the build",
            required=bool(core),
        )


def repo_makefiles(ctx: RepoContext):
    """Parsed Makefiles that belong to the project.

    Not its docs build, and not its test fixtures: `requests` generates TLS
    certificates from eight Makefiles under `tests/certs/`, none of which is a
    step in installing requests.
    """
    from .assets import is_test_file

    out = []
    for rel in ctx.files:
        if not makefile.is_makefile(rel) or _PERIPHERAL_BUILD.match(rel) or is_test_file(rel):
            continue
        out.append(makefile.parse(ctx.text(rel), rel))
    return out


def _collect_makefiles(ctx: RepoContext, report: Report) -> None:
    """A Makefile is a build manifest that no dependency tool reads.

    Only targets that build, fetch, install or run count: `make style` and
    `make publish` are the maintainers' business, and reporting them would be
    the same mistake as auditing a repo's CI configuration.
    """
    files = repo_makefiles(ctx)
    interesting = [(mk, t) for mk in files for t in mk.interesting()]
    if not files:
        return

    documented = _documented_make_targets(ctx)
    # `make` is only a requirement when the project compiles with it or the docs
    # tell you to use it. A convenience `init: pip install -r ...` target does
    # not make GNU make a dependency of `requests`.
    compiles = any(t.builds for _, t in interesting)
    named_in_docs = any(t.name in documented for _, t in interesting)
    if interesting:
        _check_tool(
            ctx, report, "make",
            f"{files[0].path} defines {len(interesting)} build/setup target(s)",
            required=compiles or named_in_docs,
        )

    shown = [pair for pair in interesting if pair[1].builds or pair[1].fetches
             or pair[1].name in documented]
    for mk, target in (shown or interesting)[:4]:
        kinds = [
            label
            for label, flag in (
                ("builds", target.builds), ("fetches", target.fetches),
                ("installs", target.installs), ("runs", target.runs),
            )
            if flag
        ]
        report.add(
            Requirement(
                kind=Kind.BUILD,
                name=f"make {target.name}",
                status=Status.INFO,
                detail=f"{', '.join(kinds)}: {_signal_line(mk, target)[:60]}"
                if target.recipe
                else ", ".join(kinds),
                source=f"{mk.path}:{target.lineno}",
                explain="Documented in the project's own Makefile."
                + (" Named in the README, so it is part of the instructions."
                   if target.name in documented else ""),
                meta={"make_target": target.name},
            )
        )


def _signal_line(mk, target) -> str:
    """The recipe line that earned the classification, not just the first one."""
    for line in target.recipe:
        expanded = mk.expand(line)
        if makefile.BUILD_TOOLS.search(expanded) or makefile.FETCH_TOOLS.search(expanded):
            return expanded
    return mk.expand(target.recipe[0]) if target.recipe else ""


def _documented_make_targets(ctx: RepoContext) -> set:
    """`make <target>` as the docs tell you to run it."""
    found = set()
    for rel in ctx.text_files((".md", ".rst", ".txt")):
        for match in re.finditer(r"make\s+([a-zA-Z][\w.-]*)", ctx.text(rel)):
            found.add(match.group(1))
    return found


def _extension_markers(ctx: RepoContext) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for rel in ctx.text_files((".py",)):
        base = rel.split("/")[-1]
        if base not in ("setup.py", "build.py") and "/ops/" not in rel and "extension" not in rel.lower():
            # cpp_extension.load can be anywhere, so still scan cheaply for it.
            text = ctx.text(rel)
            if "cpp_extension" not in text and "load_inline" not in text:
                continue
        text = ctx.text(rel)
        for marker in _EXT_MARKERS:
            if marker in text:
                out.append((rel, marker.rstrip("(")))
                break
    return out


def _documented_build_steps(ctx: RepoContext) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for rel in ctx.text_files((".md", ".rst", ".sh", ".bash", ".txt")):
        text = ctx.text(rel)
        for pattern in (_EDITABLE_INSTALL, _SETUP_BUILD):
            for match in pattern.finditer(text):
                out.append((match.group(0).strip(), ctx.source_ref(rel, match.group(0)[:30])))
    seen = set()
    unique = []
    for command, source in out:
        if command not in seen:
            seen.add(command)
            unique.append((command, source))
    return unique


def _check_compiler(ctx: RepoContext, report: Report, needed: bool) -> None:
    if not needed:
        return
    candidates = ("cl",) if sys.platform == "win32" and not ctx.target.is_container else ("cc", "gcc", "clang")
    found = next((c for c in candidates if ctx.target.which(c)), None)
    if found:
        code, out = ctx.target.run([found, "--version"], timeout=20)
        version = re.search(r"\d+\.\d+(?:\.\d+)?", out or "")
        report.add(
            Requirement(
                kind=Kind.BUILD,
                name="C/C++ compiler",
                status=Status.OK,
                detail=f"{found} {version.group(0)}" if version else found,
            )
        )
        return
    report.add(
        Requirement(
            kind=Kind.BUILD,
            name="C/C++ compiler",
            status=Status.MISSING,
            detail=f"none of {', '.join(candidates)} found in {ctx.target.describe()}",
            manual="Install build-essential (Linux) or the MSVC build tools (Windows).",
        )
    )


_PERIPHERAL_BUILD = re.compile(r"^(tools|examples?|deploy|docs?|benchmarks?|third[-_]party)/",
                               re.IGNORECASE)


def _check_tool(
    ctx: RepoContext, report: Report, tool: str, why: str, required: bool = True
) -> None:
    # The system collector may already have reported this binary; one fact, one
    # finding, and this one carries the better explanation.
    for existing in list(report.requirements):
        if existing.kind is Kind.SYSTEM and existing.name == tool:
            report.requirements.remove(existing)
    if ctx.target.which(tool):
        report.add(Requirement(kind=Kind.BUILD, name=tool, status=Status.OK, detail="present"))
    else:
        report.add(
            Requirement(
                kind=Kind.BUILD,
                name=tool,
                status=Status.MISSING if required else Status.INFO,
                detail=why if required else f"{why} (an optional extra, not the package)",
                fix=f"pip install {tool}" if tool == "ninja" and required else None,
                manual=(None if tool == "ninja" else f"Install {tool}.") if required else None,
            )
        )


def _check_nvcc_matches_torch(ctx: RepoContext, report: Report) -> None:
    """The specific failure: nvcc exists, but not the version torch was built for."""
    nvcc_version = None
    if ctx.target.which("nvcc"):
        code, out = ctx.target.run(["nvcc", "--version"], timeout=20)
        match = re.search(r"release ([\d.]+)", out or "")
        nvcc_version = match.group(1) if match else None

    code, out = ctx.target.python(
        ["-c", "import torch;print(torch.version.cuda or '')"], timeout=120
    )
    torch_cuda = out.strip().splitlines()[-1].strip() if code == 0 and out.strip() else None

    if not nvcc_version:
        report.add(
            Requirement(
                kind=Kind.BUILD,
                name="nvcc",
                status=Status.MISSING,
                detail="CUDA sources present but no nvcc in " + ctx.target.describe(),
                manual="Install the CUDA toolkit"
                + (f" matching torch's {torch_cuda}" if torch_cuda else "")
                + ", and set CUDA_HOME.",
            )
        )
        return

    if torch_cuda and nvcc_version.split(".")[:2] != torch_cuda.split(".")[:2]:
        report.add(
            Requirement(
                kind=Kind.BUILD,
                name="nvcc / torch CUDA",
                status=Status.MISMATCH,
                detail=f"nvcc is {nvcc_version} but torch was built for CUDA {torch_cuda}",
                manual="Extensions built with a mismatched nvcc link but crash at run time. "
                "Install the matching toolkit, or a torch built for this one.",
            )
        )
    else:
        report.add(
            Requirement(
                kind=Kind.BUILD,
                name="nvcc",
                status=Status.OK,
                detail=f"release {nvcc_version}"
                + (f", matches torch CUDA {torch_cuda}" if torch_cuda else ""),
            )
        )
