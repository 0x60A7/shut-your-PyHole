"""Environment variables the code reads.

A missing `HF_TOKEN` is the same class of problem as a missing SMPL licence —
a credential the machine cannot obtain for you — and nothing in a manifest
records it. Variables with a default in the code are optional; variables read
without one are requirements.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from ..context import RepoContext
from ..knowledge import HEADLESS_MODULES
from ..model import Kind, Report, Requirement, Status
from .assets import is_test_file

SECRET_HINT = re.compile(r"(TOKEN|SECRET|PASSWORD|APIKEY|API_KEY|_KEY|CREDENTIAL)$", re.IGNORECASE)

# Variables the interpreter or the shell always sets; not requirements.
IGNORED = {
    "PATH", "HOME", "USER", "USERNAME", "PWD", "SHELL", "TERM", "TMPDIR", "TEMP", "TMP",
    "LANG", "LC_ALL", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX",
    "HOSTNAME", "COMPUTERNAME", "APPDATA", "LOCALAPPDATA", "SYSTEMROOT", "OS",
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "SLURM_PROCID", "SLURM_JOB_ID",
}

KNOWN: Dict[str, str] = {
    "HF_TOKEN": "Hugging Face access token; required for gated model repos.",
    "HUGGING_FACE_HUB_TOKEN": "Hugging Face access token; required for gated model repos.",
    "WANDB_API_KEY": "Weights & Biases key; runs fail or go offline without it.",
    "OPENAI_API_KEY": "OpenAI credential.",
    "ANTHROPIC_API_KEY": "Anthropic credential.",
    "CUDA_HOME": "Points at the CUDA toolkit; needed to compile extensions.",
    "CUDA_VISIBLE_DEVICES": "Selects GPUs; absent means all of them.",
    "PYOPENGL_PLATFORM": "Selects the GL backend (osmesa/egl) for headless rendering.",
    "MUJOCO_GL": "Selects the MuJoCo rendering backend (egl/osmesa/glfw).",
    "EGL_DEVICE_ID": "Selects the EGL device for headless rendering.",
    "TORCH_HOME": "Cache directory for torch.hub and torchvision weights.",
    "HF_HOME": "Cache directory for Hugging Face downloads.",
    "TRANSFORMERS_CACHE": "Cache directory for transformers downloads.",
    "LD_LIBRARY_PATH": "Shared-library search path; often needed for CUDA or OSMesa.",
    "OMP_NUM_THREADS": "Thread count; unset can mean severe oversubscription.",
    "DISPLAY": "X11 display; absent on headless machines.",
}

_GETENV_DEFAULTED = re.compile(r"(?:os\.getenv|os\.environ\.get)\s*\(\s*['\"]([A-Z][A-Z0-9_]{2,})['\"]\s*,")
_GETENV_BARE = re.compile(r"(?:os\.getenv|os\.environ\.get)\s*\(\s*['\"]([A-Z][A-Z0-9_]{2,})['\"]\s*\)")
_ENVIRON_INDEX = re.compile(r"os\.environ\s*\[\s*['\"]([A-Z][A-Z0-9_]{2,})['\"]\s*\]")
_SHELL_EXPORT = re.compile(r"^\s*export\s+([A-Z][A-Z0-9_]{2,})\s*=", re.MULTILINE)
_DOCKER_ENV = re.compile(r"^\s*ENV\s+([A-Z][A-Z0-9_]{2,})[\s=]", re.MULTILINE)


@dataclass
class Usage:
    """How the code reads a variable, which decides how bad its absence is.

    Only ``os.environ["X"]`` raises. ``os.getenv("X")`` quietly returns None,
    which may still be wrong but is not a crash — reporting it as a blocker is
    how a checker teaches people to ignore it.
    """

    name: str
    source: str
    required: bool          # indexed access: absence raises KeyError
    defaulted: bool = False  # read with an explicit fallback
    assigned: bool = False   # the project sets it for itself


def collect(ctx: RepoContext, report: Report) -> None:
    usages = _scan(ctx)
    _add_headless_hint(ctx, usages)
    if not usages:
        return

    assumed = {n.upper() for n in ctx.config.assume_env}
    environment = _target_environment(ctx)

    for usage in sorted(usages.values(), key=lambda u: (not u.required, u.name)):
        name = usage.name
        if name in IGNORED or name in assumed:
            continue
        value = environment.get(name) if environment is not None else os.environ.get(name)
        known = KNOWN.get(name, "")
        secret = bool(SECRET_HINT.search(name))

        if value:
            report.add(
                Requirement(
                    kind=Kind.ENV,
                    name=name,
                    status=Status.OK,
                    detail="set" + (" (value hidden)" if secret else f"={_trim(value)}"),
                    source=usage.source,
                    explain=known or None,
                )
            )
        elif usage.assigned:
            report.add(
                Requirement(
                    kind=Kind.ENV,
                    name=name,
                    status=Status.INFO,
                    detail="set by the project's own scripts, not by you",
                    source=usage.source,
                    explain=known or None,
                )
            )
        elif secret:
            report.add(
                Requirement(
                    kind=Kind.ENV,
                    name=name,
                    status=Status.BLOCKED,
                    detail="credential is not set",
                    source=usage.source,
                    manual=known or f"Obtain a value for {name} and export it.",
                    meta={"credential": True},
                )
            )
        elif usage.required:
            report.add(
                Requirement(
                    kind=Kind.ENV,
                    name=name,
                    status=Status.MISSING,
                    detail="indexed directly; absence raises KeyError",
                    source=usage.source,
                    manual=known or f"export {name}=...",
                )
            )
        else:
            report.add(
                Requirement(
                    kind=Kind.ENV,
                    name=name,
                    status=Status.INFO,
                    detail="optional (the code supplies a default)"
                    if usage.defaulted
                    else "read with no default; the code receives None",
                    source=usage.source,
                    explain=known or None,
                )
            )


def _scan(ctx: RepoContext) -> Dict[str, Usage]:
    usages: Dict[str, Usage] = {}

    def note(
        name: str, source: str, required: bool, defaulted: bool = False, assigned: bool = False
    ) -> None:
        existing = usages.get(name)
        if existing is None:
            usages[name] = Usage(
                name=name, source=source, required=required, defaulted=defaulted, assigned=assigned
            )
            return
        existing.required = existing.required or required
        existing.defaulted = existing.defaulted or defaulted
        existing.assigned = existing.assigned or assigned

    for rel in ctx.text_files((".py",)):
        if is_test_file(rel):
            continue
        text = ctx.text(rel)
        if "environ" not in text and "getenv" not in text:
            continue
        for match in _ENVIRON_INDEX.finditer(text):
            note(match.group(1), ctx.source_ref(rel, match.group(0)), required=True)
        for match in _GETENV_BARE.finditer(text):
            note(match.group(1), ctx.source_ref(rel, match.group(0)), required=False)
        for match in _GETENV_DEFAULTED.finditer(text):
            note(match.group(1), ctx.source_ref(rel, match.group(0)), required=False, defaulted=True)
        # os.environ["X"] = "y" is the project setting it for itself.
        for match in re.finditer(r"os\.environ\s*\[\s*['\"]([A-Z][A-Z0-9_]{2,})['\"]\s*\]\s*=", text):
            note(match.group(1), ctx.source_ref(rel, match.group(0)), required=False, assigned=True)

    for rel in ctx.text_files((".sh", ".bash")):
        for match in _SHELL_EXPORT.finditer(ctx.text(rel)):
            note(match.group(1), ctx.source_ref(rel, match.group(0)), required=False, assigned=True)

    for rel in ctx.files:
        base = rel.split("/")[-1].lower()
        if base == "dockerfile" or base.startswith("dockerfile."):
            for match in _DOCKER_ENV.finditer(ctx.text(rel)):
                note(match.group(1), ctx.source_ref(rel, match.group(0)), required=False, assigned=True)
        if base in (".env.example", "env.example", ".env.sample"):
            for line in ctx.text(rel).splitlines():
                match = re.match(r"\s*([A-Z][A-Z0-9_]{2,})\s*=", line)
                if match:
                    note(match.group(1), rel, required=True)
    return usages


def _add_headless_hint(ctx: RepoContext, usages: Dict[str, Usage]) -> None:
    """Offscreen renderers need a backend chosen explicitly; nothing declares that."""
    for module in HEADLESS_MODULES:
        hits = ctx.grep(f"import {module}", (".py",))
        if not hits:
            continue
        name = "MUJOCO_GL" if module == "mujoco" else "PYOPENGL_PLATFORM"
        if name not in usages:
            usages[name] = Usage(
                name=name,
                source=hits[0],
                required=False,
            )
        return


def _target_environment(ctx: RepoContext) -> Optional[Dict[str, str]]:
    """Env of the target. For a container this is the image's own ENV block."""
    if not ctx.target.is_container:
        return None  # fall back to os.environ, which is this shell's truth
    code, out = ctx.target.run(["sh", "-lc", "env"], timeout=90)
    if code != 0:
        return {}
    result: Dict[str, str] = {}
    for line in (out or "").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def _trim(value: str, width: int = 24) -> str:
    return value if len(value) <= width else value[: width - 3] + "..."
