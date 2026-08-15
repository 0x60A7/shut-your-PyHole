"""Container-layer requirements.

The interesting case is not the Dockerfile — it is the repo whose canonical
environment is an image mentioned only in the README, which no manifest parser
would ever find.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

from ..context import RepoContext
from ..model import Kind, Report, Requirement, Status
from ..util import run, which

_FROM = re.compile(r"^\s*FROM\s+(?:--platform=\S+\s+)?(\S+)", re.MULTILINE | re.IGNORECASE)
_ARG = re.compile(r"^\s*ARG\s+(\w+)\s*=\s*(\S+)", re.MULTILINE | re.IGNORECASE)
_APT = re.compile(r"apt-get\s+(?:-y\s+)?install\s+((?:[^\n&|]|\\\n)+)", re.IGNORECASE)
_COMPOSE_IMAGE = re.compile(r"^\s*image:\s*['\"]?([^'\"\s]+)", re.MULTILINE)
_DOCKER_CMD = re.compile(
    r"docker\s+(?:pull|run)\s+(?:(?:-{1,2}[\w-]+(?:[= ]\S+)?|--\S+)\s+)*"
    r"([a-z0-9][\w./-]*(?::[\w.-]+)?)",
    re.IGNORECASE,
)
# A bare word is not an image; require a registry path, a tag, or a known org form.
_IMAGE_SHAPE = re.compile(r"^[a-z0-9][a-z0-9._-]*(?:/[a-z0-9._-]+)+(?::[\w.-]+)?$|^[a-z0-9._-]+:[\w.-]+$")

PY_VERSION = re.compile(r"python[:/-]?(\d\.\d+)", re.IGNORECASE)
CUDA_VERSION = re.compile(r"cuda[:/-]?(\d+\.\d+)", re.IGNORECASE)


def documented_images(ctx: RepoContext) -> List[str]:
    """Every container image this repo names, wherever it names it."""
    images = [img for img, _ in _compose_images(ctx) + _images_from_docs(ctx)]
    return [i for i in dict.fromkeys(images) if _IMAGE_SHAPE.match(i)]


def _compose_images(ctx: RepoContext) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for rel in ctx.files:
        if rel.split("/")[-1] in (
            "docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"
        ):
            for match in _COMPOSE_IMAGE.finditer(ctx.text(rel)):
                out.append((match.group(1), ctx.source_ref(rel, match.group(1))))
    return out


def collect(ctx: RepoContext, report: Report) -> None:
    dockerfiles = [f for f in ctx.files if _is_dockerfile(f)]
    images: List[Tuple[str, str]] = []  # (image, source)

    for rel in dockerfiles:
        _report_dockerfile(ctx, rel, report)
    images.extend(_compose_images(ctx))
    images.extend(_images_from_docs(ctx))
    _report_run_flags(ctx, report)

    if not dockerfiles and not images:
        return

    seen: Set[str] = set()
    for image, source in images:
        if image in seen or not _IMAGE_SHAPE.match(image):
            continue
        seen.add(image)
        _report_image(ctx, image, source, report)


def _is_dockerfile(rel: str) -> bool:
    base = rel.split("/")[-1].lower()
    return base == "dockerfile" or base.startswith("dockerfile.") or base.endswith(".dockerfile")


def _report_dockerfile(ctx: RepoContext, rel: str, report: Report) -> None:
    text = ctx.text(rel)
    args = dict(_ARG.findall(text))
    bases = [_expand_args(b, args) for b in _FROM.findall(text)]
    stages = [b for b in bases if not b.lower().startswith(("scratch",))]
    detail = " <- ".join(stages[-2:]) if stages else "no FROM found"

    facts = []
    for base in stages:
        match = PY_VERSION.search(base)
        if match:
            facts.append(f"python {match.group(1)}")
        match = CUDA_VERSION.search(base)
        if match:
            facts.append(f"cuda {match.group(1)}")
    match = re.search(r"python(\d\.\d+)", text)
    if match and f"python {match.group(1)}" not in facts:
        facts.append(f"python {match.group(1)}")

    report.add(
        Requirement(
            kind=Kind.CONTAINER,
            name=rel,
            status=Status.OK,
            detail=detail + (f" ({', '.join(dict.fromkeys(facts))})" if facts else ""),
            source=rel,
            meta={"bases": stages},
        )
    )

    apt = apt_packages(text)
    if apt:
        report.add(
            Requirement(
                kind=Kind.CONTAINER,
                name=f"apt packages ({len(apt)})",
                status=Status.INFO,
                detail=", ".join(apt[:8]) + ("..." if len(apt) > 8 else ""),
                source=rel,
                explain="These are system libraries the image installs. Running outside Docker means installing them yourself.",
                meta={"packages": apt, "verbose_list": True},
            )
        )


def _expand_args(value: str, args: dict) -> str:
    def repl(match):
        return args.get(match.group(1) or match.group(2), match.group(0))

    return re.sub(r"\$\{(\w+)\}|\$(\w+)", repl, value)


def apt_packages(text: str) -> List[str]:
    """System packages installed by `apt-get install` lines in a Dockerfile."""
    packages: List[str] = []
    for match in _APT.finditer(text):
        chunk = match.group(1).replace("\\\n", " ")
        for token in chunk.split():
            if token.startswith("-") or token in ("&&", "\\"):
                continue
            if re.match(r"^[a-z0-9][a-z0-9+._-]*$", token):
                packages.append(token)
    return list(dict.fromkeys(packages))


def _images_from_docs(ctx: RepoContext) -> List[Tuple[str, str]]:
    """Find `docker pull`/`docker run` invocations in prose and shell scripts."""
    out: List[Tuple[str, str]] = []
    for rel in ctx.text_files((".md", ".rst", ".txt", ".sh", ".bash")):
        text = ctx.text(rel)
        if "docker" not in text.lower():
            continue
        for match in _DOCKER_CMD.finditer(text):
            image = match.group(1)
            if image in ("run", "pull", "-it", "--rm", "."):
                continue
            out.append((image, ctx.source_ref(rel, match.group(0)[:40])))
    return out


_RUN_FLAGS = {
    "--shm-size": "dataloader workers crash with a bus error on the 64MB default",
    "--ipc=host": "shares the host IPC namespace; same purpose as --shm-size",
    "--gpus": "without it the container sees no GPU at all",
    "--runtime=nvidia": "older syntax for GPU passthrough",
    "--network=host": "the container expects host networking",
    "--privileged": "the container expects elevated privileges",
}


def _report_run_flags(ctx: RepoContext, report: Report) -> None:
    """Flags the docs put on `docker run` — the part everyone copies wrong."""
    found = {}
    for rel in ctx.text_files((".md", ".rst", ".txt", ".sh", ".bash")):
        text = ctx.text(rel)
        if "docker run" not in text and "podman run" not in text:
            continue
        for line in text.splitlines():
            if "docker run" not in line and "podman run" not in line:
                continue
            for flag, why in _RUN_FLAGS.items():
                if flag in line and flag not in found:
                    value = re.search(re.escape(flag) + r"[= ]([^\s\\]+)", line)
                    found[flag] = (
                        f"{flag}={value.group(1)}" if value and flag != "--ipc=host" else flag,
                        why,
                        ctx.source_ref(rel, line.strip()[:40]),
                    )
    for flag, (label, why, source) in found.items():
        report.add(
            Requirement(
                kind=Kind.CONTAINER,
                name=f"run flag {label}",
                status=Status.INFO,
                detail=why,
                source=source,
                explain="Documented on the project's own `docker run` line; it is part of the "
                        "environment even though no manifest mentions it.",
            )
        )


def _report_image(ctx: RepoContext, image: str, source: str, report: Report) -> None:
    if not which("docker"):
        report.add(
            Requirement(
                kind=Kind.CONTAINER,
                name=image,
                status=Status.UNKNOWN,
                detail="docker not installed, cannot check whether the image is available",
                source=source,
            )
        )
        return

    code, _ = run(["docker", "image", "inspect", image], timeout=30)
    if code == 0:
        report.add(
            Requirement(
                kind=Kind.CONTAINER, name=image, status=Status.OK, detail="image present locally", source=source
            )
        )
        return

    if ctx.network:
        code, out = run(["docker", "manifest", "inspect", image], timeout=60)
        if code == 0:
            report.add(
                Requirement(
                    kind=Kind.CONTAINER,
                    name=image,
                    status=Status.MISSING,
                    detail="not pulled, but available in the registry",
                    source=source,
                    fix=f"docker pull {image}",
                )
            )
            return
        report.add(
            Requirement(
                kind=Kind.CONTAINER,
                name=image,
                status=Status.BLOCKED,
                detail="registry lookup failed — the image may be private or gone",
                source=source,
                manual="Log in to the registry, or find a replacement image.",
                explain=out.strip().splitlines()[-1] if out.strip() else None,
            )
        )
        return

    report.add(
        Requirement(
            kind=Kind.CONTAINER,
            name=image,
            status=Status.MISSING,
            detail="not pulled locally",
            source=source,
            fix=f"docker pull {image}",
        )
    )
