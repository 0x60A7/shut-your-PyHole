"""System-layer requirements.

Only checks tools the repository actually asks for. Reporting a missing CUDA
runtime for a pure-CPU project is noise, and noise is what makes people stop
reading audit output.
"""

from __future__ import annotations

import re
from typing import List, Optional

from ..context import RepoContext
from ..model import Kind, Report, Requirement, Status
from ..util import run, which
from .container import apt_packages

# tool -> (probe args, why the repo might need it, detectors)
BINARY_HINTS = [
    ("ffmpeg", ["-version"], ("ffmpeg", "libx264", "video")),
    ("cmake", ["--version"], ("cmake", "CMakeLists.txt")),
    ("conda", ["--version"], ("environment.yml", "conda install", "conda env")),
    ("node", ["--version"], ("package.json",)),
    ("colmap", ["-h"], ("colmap",)),
    ("blender", ["--version"], ("blender",)),
]


def collect(ctx: RepoContext, report: Report) -> None:
    wants_docker = _wants_docker(ctx)
    wants_gpu = _wants_gpu(ctx)

    _check_binary(report, "git", ["--version"], required=ctx.exists(".gitmodules"))

    if wants_docker:
        _check_docker(ctx, report, wants_gpu)
    if wants_gpu:
        _check_gpu(report)

    apt_provided = _apt_provided(ctx) if wants_docker else set()
    lowered_files = {f.lower() for f in ctx.files}
    for tool, args, needles in BINARY_HINTS:
        reason = _relevance(ctx, lowered_files, tool, needles, apt_provided)
        if reason is None:
            continue
        _check_binary(report, tool, args, required=True, from_image=(reason == "image"))


def _relevance(ctx: RepoContext, lowered_files, tool: str, needles, apt_provided) -> Optional[str]:
    """Why this repo might need ``tool`` — or None.

    A bare mention is not evidence: plenty of READMEs name Blender once in a
    citation. We want a marker file, an apt line, or something that looks like
    an actual invocation.
    """
    for needle in needles:
        if needle.lower() in lowered_files or any(
            f.lower().endswith("/" + needle.lower()) for f in lowered_files
        ):
            return "files"
    if tool in apt_provided:
        return "image"
    if _looks_invoked(ctx, tool):
        return "docs"
    return None


_SCRIPT_SUFFIXES = (".sh", ".bash", ".md", ".rst", ".txt", ".yml", ".yaml")


def _looks_invoked(ctx: RepoContext, tool: str) -> bool:
    pattern = re.compile(r"(?:^|[\s|&;(`$])" + re.escape(tool) + r"\s+[-\w./]", re.MULTILINE)
    for rel in ctx.text_files(_SCRIPT_SUFFIXES):
        if tool in ctx.text(rel).lower() and pattern.search(ctx.text(rel)):
            return True
    return False


def _apt_provided(ctx: RepoContext) -> set:
    """Tools the Docker image installs — needed on the host only for bare-metal runs."""
    provided = set()
    for rel in ctx.files:
        base = rel.split("/")[-1].lower()
        if base == "dockerfile" or base.startswith("dockerfile.") or base.endswith(".dockerfile"):
            provided.update(apt_packages(ctx.text(rel)))
    return provided


def _wants_docker(ctx: RepoContext) -> bool:
    if any(f.lower().startswith("dockerfile") or "docker-compose" in f.lower() for f in ctx.files):
        return True
    return "docker run" in ctx.docs_text().lower() or "docker pull" in ctx.docs_text().lower()


def _wants_gpu(ctx: RepoContext) -> bool:
    docs = ctx.docs_text().lower()
    if any(token in docs for token in ("cuda", "nvidia", "gpu")):
        return True
    return bool(ctx.grep("torch.cuda", (".py",)) or ctx.grep("cuda", (".txt", ".yml", ".yaml")))


def _check_binary(
    report: Report, tool: str, args: List[str], required: bool, from_image: bool = False
) -> None:
    if not required:
        return
    path = which(tool)
    if not path:
        report.add(
            Requirement(
                kind=Kind.SYSTEM,
                name=tool,
                status=Status.INFO if from_image else Status.MISSING,
                detail="not on PATH — the Docker image installs it"
                if from_image
                else "not on PATH",
                manual=None if from_image else f"Install {tool}.",
                explain="Only needed on the host if you run outside the container."
                if from_image
                else None,
            )
        )
        return
    code, out = run([tool] + args, timeout=15)
    version = _first_version(out) if code == 0 else None
    report.add(
        Requirement(
            kind=Kind.SYSTEM,
            name=tool,
            status=Status.OK,
            detail=version or path,
        )
    )


def _first_version(text: str) -> Optional[str]:
    match = re.search(r"\d+\.\d+(?:\.\d+)?", text or "")
    return match.group(0) if match else None


def _check_docker(ctx: RepoContext, report: Report, wants_gpu: bool) -> None:
    if not which("docker"):
        report.add(
            Requirement(
                kind=Kind.SYSTEM,
                name="docker",
                status=Status.MISSING,
                detail="not on PATH, but the project documents a Docker workflow",
                manual="Install Docker (or Docker Desktop) and start the daemon.",
            )
        )
        return

    code, out = run(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=30)
    if code != 0:
        report.add(
            Requirement(
                kind=Kind.SYSTEM,
                name="docker daemon",
                status=Status.MISSING,
                detail="client installed but the daemon is not reachable",
                manual="Start Docker Desktop / the docker service.",
                explain=(out or "").strip().splitlines()[-1] if out else None,
            )
        )
        return

    report.add(
        Requirement(kind=Kind.SYSTEM, name="docker", status=Status.OK, detail=f"daemon {out.strip()}")
    )

    if not wants_gpu:
        return
    code, out = run(["docker", "info", "--format", "{{json .Runtimes}}"], timeout=30)
    if code == 0 and "nvidia" in (out or "").lower():
        report.add(
            Requirement(
                kind=Kind.SYSTEM, name="nvidia container runtime", status=Status.OK, detail="registered with docker"
            )
        )
    else:
        report.add(
            Requirement(
                kind=Kind.SYSTEM,
                name="nvidia container runtime",
                status=Status.MISMATCH,
                detail="not registered with docker; --gpus all will fail",
                manual="Install nvidia-container-toolkit (Linux) or enable WSL2 GPU support.",
            )
        )


def _check_gpu(report: Report) -> None:
    if not which("nvidia-smi"):
        report.add(
            Requirement(
                kind=Kind.SYSTEM,
                name="nvidia gpu",
                status=Status.MISMATCH,
                detail="nvidia-smi not found; the project expects CUDA",
                manual="A GPU host or a CPU-only code path is needed.",
            )
        )
        return

    code, out = run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], timeout=30
    )
    if code != 0 or not out.strip():
        report.add(
            Requirement(
                kind=Kind.SYSTEM,
                name="nvidia gpu",
                status=Status.MISMATCH,
                detail="nvidia-smi present but returned no devices",
                explain=(out or "").strip()[:200] or None,
            )
        )
        return

    first = out.strip().splitlines()[0]
    report.add(Requirement(kind=Kind.SYSTEM, name="nvidia gpu", status=Status.OK, detail=first.strip()))

    code, header = run(["nvidia-smi"], timeout=30)
    match = re.search(r"CUDA Version:\s*([\d.]+)", header or "")
    if match:
        report.add(
            Requirement(
                kind=Kind.SYSTEM,
                name="cuda (driver)",
                status=Status.OK,
                detail=f"driver supports up to CUDA {match.group(1)}",
            )
        )

    if which("nvcc"):
        code, out = run(["nvcc", "--version"], timeout=15)
        match = re.search(r"release ([\d.]+)", out or "")
        report.add(
            Requirement(
                kind=Kind.SYSTEM,
                name="cuda toolkit (nvcc)",
                status=Status.OK,
                detail=f"release {match.group(1)}" if match else "installed",
            )
        )
    else:
        report.add(
            Requirement(
                kind=Kind.SYSTEM,
                name="cuda toolkit (nvcc)",
                status=Status.INFO,
                detail="not on PATH — only needed to compile CUDA extensions from source",
            )
        )
