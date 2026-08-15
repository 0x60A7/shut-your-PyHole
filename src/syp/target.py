"""Where the audit's questions get asked.

Auditing the host when the README says "use this image" answers the wrong
question: the host does not need ffmpeg if the container has it. A Target is
the environment probes execute in — the host, a virtualenv, or a container —
so `syp audit --target image` inspects the environment you will actually run in.

Filesystem questions (does this checkpoint exist?) stay on the host: the repo is
bind-mounted into the container, so it is the same tree either way.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .util import run, which


@dataclass
class Target:
    kind: str = "host"   # "host" | "venv" | "image"
    label: str = "host interpreter"
    python_exe: str = sys.executable
    image: Optional[str] = None
    root: Optional[str] = None
    engine: str = "docker"
    available: bool = True
    problem: Optional[str] = None
    gpu_flags: List[str] = field(default_factory=list)
    gpu_verified: Optional[bool] = None

    @property
    def is_container(self) -> bool:
        return self.kind == "image"

    def describe(self) -> str:
        if self.kind == "image":
            return f"{self.engine} image {self.image}"
        return self.label

    # --- probing ------------------------------------------------------------

    def run(self, argv: List[str], timeout: int = 60) -> Tuple[int, str]:
        """Run a command inside the target and return (returncode, output)."""
        if not self.available:
            return -1, self.problem or "target unavailable"
        if self.kind != "image":
            return run(argv, timeout=timeout)
        mount = []
        if self.root:
            mount = ["-v", f"{self.root}:/repo", "-w", "/repo"]
        return run(
            [self.engine, "run", "--rm", "--entrypoint", ""]
            + self.gpu_flags
            + mount
            + [self.image]
            + argv,
            timeout=max(timeout, 120),
        )

    def which(self, tool: str) -> Optional[str]:
        if self.kind != "image":
            return which(tool)
        code, out = self.run(["sh", "-lc", f"command -v {tool} || true"], timeout=90)
        path = (out or "").strip().splitlines()[-1] if out.strip() else ""
        return path if code == 0 and path.startswith("/") else None

    def python(self, args: List[str], timeout: int = 60) -> Tuple[int, str]:
        if self.kind == "image":
            return self.run([self.python_exe] + args, timeout=timeout)
        return run([self.python_exe] + args, timeout=timeout)

    def pip_command(self, spec: str) -> str:
        """A `pip install` that lands in *this* target, not whatever is active."""
        if self.kind == "image":
            return (
                f"{self.engine} run --rm {self.image} "
                f"python -m pip install {self.quote(spec)}"
            )
        return f"{self.quote(self.python_exe)} -m pip install {self.quote(spec)}"

    def quote(self, argument: str) -> str:
        """Quote for the shell that will actually run the command.

        cmd.exe does not treat single quotes as quoting at all, so
        `pip install 'numpy>=1.21'` there redirects stdout into a file called
        `=1.21'` and installs the wrong thing.
        """
        if self.shell_is_posix:
            return f"'{argument}'"
        return f'"{argument}"'

    @property
    def shell_is_posix(self) -> bool:
        # Container commands always go through the container's /bin/sh.
        return self.kind == "image" or sys.platform != "win32"


def resolve(root: str, spec: Optional[str], images: Optional[List[str]] = None) -> Target:
    """Turn a --target string into a Target.

    ``image`` with no name picks the single image the repo documents; ambiguity
    is reported rather than guessed at.
    """
    spec = (spec or "host").strip()

    if spec in ("host", ""):
        return Target(kind="host", label="host interpreter", python_exe=sys.executable)

    if spec == "venv":
        found = _find_venv(root)
        if found:
            return Target(kind="venv", label=f"{found[1]} (venv)", python_exe=found[0])
        return Target(
            kind="host",
            label="host interpreter",
            python_exe=sys.executable,
            problem="no virtualenv found in the repo; fell back to the host interpreter",
        )

    if spec.startswith("image"):
        name = spec.split(":", 1)[1].strip() if ":" in spec else ""
        if not name:
            candidates = images or []
            if len(candidates) == 1:
                name = candidates[0]
            elif not candidates:
                return Target(kind="image", label="image", available=False,
                              problem="no image documented in the repo; pass --target image:NAME")
            else:
                return Target(kind="image", label="image", available=False,
                              problem=f"several images documented ({', '.join(candidates[:3])}); "
                                      "pass --target image:NAME")
        return _container_target(root, name)

    # A bare image name is a reasonable thing for someone to type.
    if "/" in spec or ":" in spec:
        return _container_target(root, spec)

    return Target(kind="host", label="host interpreter", python_exe=sys.executable,
                  problem=f"unknown target {spec!r}; using the host")


def _container_target(root: str, image: str) -> Target:
    engine = "docker" if which("docker") else ("podman" if which("podman") else "")
    if not engine:
        return Target(kind="image", label=image, image=image, available=False,
                      problem="neither docker nor podman is installed")
    code, _ = run([engine, "image", "inspect", image], timeout=60)
    if code != 0:
        return Target(
            kind="image", label=image, image=image, engine=engine, available=False,
            problem=f"image not present locally — run `{engine} pull {image}` first",
        )
    target = Target(kind="image", label=image, image=image, engine=engine, root=root,
                    python_exe="python")
    _probe_gpu(target)
    # Some research images only ship `python3`, others put conda's python first.
    code, _ = target.run(["python", "-c", "pass"], timeout=90)
    if code != 0:
        target.python_exe = "python3"
    return target


def _probe_gpu(target: Target) -> None:
    """Actually start a container with --gpus all and see whether it works.

    This settles the question the host-side runtime check can only guess at
    (Docker Desktop and WSL2 do not advertise an nvidia runtime), and it stops
    every later probe from reporting a CPU-only torch by accident.
    """
    if not which("nvidia-smi"):
        return
    flag = ["--gpus", "all"] if target.engine == "docker" else ["--device", "nvidia.com/gpu=all"]
    code, _ = run(
        [target.engine, "run", "--rm", "--entrypoint", ""] + flag + [target.image, "true"],
        timeout=180,
    )
    target.gpu_verified = code == 0
    if code == 0:
        target.gpu_flags = flag


def _find_venv(root: str) -> Optional[Tuple[str, str]]:
    for candidate in (".venv", "venv", "env", ".env"):
        for sub in ("bin/python", "Scripts/python.exe"):
            path = os.path.join(root, candidate.replace("/", os.sep), *sub.split("/"))
            if os.path.exists(path):
                return path, candidate
    # A conda env activated in this shell is the interpreter we are running under.
    if os.environ.get("CONDA_PREFIX") and sys.executable.startswith(os.environ["CONDA_PREFIX"]):
        return sys.executable, os.path.basename(os.environ["CONDA_PREFIX"])
    return None


def default_spec(config_target: Optional[str], cli_target: Optional[str]) -> str:
    """CLI wins over config; both default to the venv-if-present behaviour."""
    return cli_target or config_target or "venv"
