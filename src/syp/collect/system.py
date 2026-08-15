"""System-layer requirements.

Only checks tools the repository actually asks for. Reporting a missing CUDA
runtime for a pure-CPU project is noise, and noise is what makes people stop
reading audit output.
"""

from __future__ import annotations

import platform
import re
import sys
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
        _check_accelerator(ctx, report)

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
    """Does the repo *call* this tool, as opposed to merely mentioning it?

    Any whitespace before the name matches prose as happily as commands ("see
    ffmpeg docs"), so an invocation must either begin a line or a shell clause,
    or be followed immediately by a flag.
    """
    name = re.escape(tool)
    pattern = re.compile(
        r"(?:^\s*(?:\$\s*)?|[|&;(]\s*|`\s*)" + name + r"\s+\S"      # start of a command
        r"|(?<![\w./-])" + name + r"\s+-\w",                          # ... or takes a flag
        re.MULTILINE,
    )
    for rel in ctx.text_files(_SCRIPT_SUFFIXES):
        text = ctx.text(rel)
        if tool in text.lower() and pattern.search(text):
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
    engine = "docker" if which("docker") else ("podman" if which("podman") else "")
    if not engine:
        report.add(
            Requirement(
                kind=Kind.SYSTEM,
                name="docker",
                status=Status.MISSING,
                detail="neither docker nor podman is on PATH, but the project documents a container workflow",
                manual="Install Docker (or Docker Desktop) and start the daemon.",
            )
        )
        return
    if engine == "podman":
        report.add(
            Requirement(
                kind=Kind.SYSTEM,
                name="podman",
                status=Status.OK,
                detail="standing in for docker",
                explain="GPU passthrough uses --device nvidia.com/gpu=all under podman, not --gpus all.",
            )
        )

    code, out = run([engine, "info", "--format", "{{.ServerVersion}}"], timeout=30)
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
        Requirement(kind=Kind.SYSTEM, name=engine, status=Status.OK, detail=f"daemon {out.strip()}")
    )

    if not wants_gpu:
        return

    # If we are auditing an image, we have already started a container with
    # --gpus all and watched it succeed or fail. That beats every inference,
    # so it is checked before the runtime registry is consulted at all.
    if ctx.target.is_container and ctx.target.gpu_verified is not None:
        verified = ctx.target.gpu_verified
        report.add(
            Requirement(
                kind=Kind.SYSTEM,
                name="gpu passthrough",
                status=Status.OK if verified else Status.MISMATCH,
                detail=f"`{engine} run --gpus all` "
                + ("works with this image" if verified else "fails with this image"),
                source="verified by running a container",
                manual=None if verified else "Install nvidia-container-toolkit, or enable GPU support "
                "in Docker Desktop.",
            )
        )
        return

    code, out = run([engine, "info", "--format", "{{json .Runtimes}}"], timeout=30)
    if code == 0 and "nvidia" in (out or "").lower():
        report.add(
            Requirement(
                kind=Kind.SYSTEM, name="nvidia container runtime", status=Status.OK,
                detail=f"registered with {engine}",
            )
        )
        return

    # Docker Desktop on Windows/WSL2 does GPU passthrough without advertising an
    # nvidia runtime, so the absence of one proves nothing there.
    if _is_wsl_or_windows_docker():
        report.add(
            Requirement(
                kind=Kind.SYSTEM,
                name="nvidia container runtime",
                status=Status.UNKNOWN,
                detail="not listed, which is normal for Docker Desktop / WSL2",
                explain="Verify for real with: docker run --rm --gpus all <image> nvidia-smi",
            )
        )
        return

    report.add(
        Requirement(
            kind=Kind.SYSTEM,
            name="nvidia container runtime",
            status=Status.MISMATCH,
            detail=f"not registered with {engine}; --gpus all will fail",
            manual="Install nvidia-container-toolkit (Linux) or enable WSL2 GPU support.",
        )
    )


def _is_wsl_or_windows_docker() -> bool:
    if sys.platform == "win32":
        return True
    release = platform.uname().release.lower()
    return "microsoft" in release or "wsl" in release


def _check_accelerator(ctx: RepoContext, report: Report) -> None:
    """Find whatever accelerator this machine has, not just an NVIDIA one."""
    if which("nvidia-smi"):
        _check_nvidia(report)
        return

    if which("rocm-smi") or which("rocminfo"):
        tool = "rocm-smi" if which("rocm-smi") else "rocminfo"
        code, out = run([tool, "--showproductname" if tool == "rocm-smi" else "-l"], timeout=30)
        report.add(
            Requirement(
                kind=Kind.SYSTEM,
                name="amd gpu (rocm)",
                status=Status.OK if code == 0 else Status.UNKNOWN,
                detail=_first_line(out) or "ROCm stack present",
                explain="The repo asks for CUDA; ROCm works only if torch was built for it "
                "and the code does not call CUDA-only APIs.",
            )
        )
        return

    if sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64"):
        report.add(
            Requirement(
                kind=Kind.SYSTEM,
                name="apple silicon gpu",
                status=Status.MISMATCH,
                detail="Metal (MPS) is available; the project asks for CUDA",
                manual="Code that hardcodes .cuda() needs porting to device='mps', or run on CUDA hardware.",
            )
        )
        return

    report.add(
        Requirement(
            kind=Kind.SYSTEM,
            name="gpu",
            status=Status.MISMATCH,
            detail="no NVIDIA, ROCm or Metal accelerator detected; the project expects one",
            manual="A GPU host, or a CPU-only code path, is needed.",
        )
    )


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()[:80]
    return ""


def _check_nvidia(report: Report) -> None:

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
