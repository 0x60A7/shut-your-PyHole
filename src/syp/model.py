"""Core data model: a requirement, its status, and the report that holds them."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


class Status(enum.Enum):
    """Outcome of checking a single requirement.

    The distinction that matters is *who can fix it*: OK needs nobody, MISSING
    and STALE are machine-resolvable, BLOCKED needs a human with an account,
    and UNKNOWN means we could not tell (which is honest, not a failure).
    """

    OK = "ok"
    MISSING = "missing"
    STALE = "stale"
    MISMATCH = "mismatch"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"
    INFO = "info"

    @property
    def symbol(self) -> str:
        return _SYMBOLS[self][0]

    @property
    def ascii_symbol(self) -> str:
        return _SYMBOLS[self][1]

    @property
    def is_blocker(self) -> bool:
        return self in (Status.MISSING, Status.STALE, Status.BLOCKED)

    @property
    def counts_toward_readiness(self) -> bool:
        return self is not Status.INFO


_SYMBOLS = {
    Status.OK: ("✓", "+"),
    Status.MISSING: ("✗", "x"),
    Status.STALE: ("✗", "x"),
    Status.MISMATCH: ("⚠", "!"),
    Status.BLOCKED: ("⚠", "!"),
    Status.UNKNOWN: ("?", "?"),
    Status.INFO: ("·", "."),
}


class Kind(enum.Enum):
    """Which layer of the stack a requirement lives in. Drives report grouping."""

    GIT = "git"
    PYTHON = "python"
    CONTAINER = "container"
    SYSTEM = "system"
    BUILD = "build"
    ENV = "env"
    ASSET = "asset"
    EXTERNAL = "external"
    ENTRYPOINT = "entrypoint"


SECTION_ORDER = [
    Kind.SYSTEM,
    Kind.GIT,
    Kind.CONTAINER,
    Kind.PYTHON,
    Kind.BUILD,
    Kind.ENV,
    Kind.ASSET,
    Kind.EXTERNAL,
    Kind.ENTRYPOINT,
]

SECTION_TITLES = {
    Kind.SYSTEM: "System",
    Kind.GIT: "Git",
    Kind.CONTAINER: "Container",
    Kind.PYTHON: "Python",
    Kind.BUILD: "Build",
    Kind.ENV: "Environment",
    Kind.ASSET: "Runtime assets",
    Kind.EXTERNAL: "External access",
    Kind.ENTRYPOINT: "Execution",
}


class FixKind(enum.Enum):
    """How much trust running a fix requires.

    ``SCRIPT`` is the dangerous one: a shell script from the audited repository
    is arbitrary code execution, so it never runs without an explicit opt-in.
    """

    LOCAL = "local"      # touches only this checkout (git submodule init, mkdir)
    NETWORK = "network"  # downloads from a known package/registry (pip, docker pull)
    SCRIPT = "script"    # runs a script belonging to the audited repository


_SCRIPT_START = ("bash ", "sh ", "python ", "make ", "./")
_NETWORK_VERBS = ("pip ", "pip3 ", "uv ", "docker pull", "conda ", "wget ", "curl ", "gdown",
                  "huggingface-cli", "apt-get", "npm ", "git lfs")


def classify_fix(command: str) -> FixKind:
    stripped = command.strip()
    if stripped.startswith("git submodule") or stripped.startswith("git lfs pull"):
        return FixKind.LOCAL
    # `python -m pip install` is a package install however it is spelled, not a
    # script belonging to the repository.
    if "-m pip install" in stripped or "-m uv " in stripped:
        return FixKind.NETWORK
    if any(verb in stripped for verb in _NETWORK_VERBS) and not stripped.startswith(_SCRIPT_START):
        return FixKind.NETWORK
    if stripped.startswith(_SCRIPT_START):
        return FixKind.SCRIPT
    return FixKind.NETWORK


@dataclass
class Requirement:
    """One thing the repository needs in order to run.

    ``source`` is where we learned about it (``requirements.txt:14``), which is
    what makes a finding auditable instead of an assertion. ``fix`` is a shell
    command we believe would resolve it; ``manual`` marks the ones no command can.
    """

    kind: Kind
    name: str
    status: Status
    detail: str = ""
    source: Optional[str] = None
    fix: Optional[str] = None
    manual: Optional[str] = None
    explain: Optional[str] = None
    fix_kind: Optional[FixKind] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.fix and self.fix_kind is None:
            self.fix_kind = classify_fix(self.fix)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["status"] = self.status.value
        d["fix_kind"] = self.fix_kind.value if self.fix_kind else None
        return d


@dataclass
class Report:
    root: str
    requirements: List[Requirement] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    suppressed: List[str] = field(default_factory=list)
    target: str = "host"

    def add(self, req: Requirement) -> Requirement:
        self.requirements.append(req)
        return req

    def extend(self, reqs) -> None:
        self.requirements.extend(reqs)

    def by_kind(self, kind: Kind) -> List[Requirement]:
        return [r for r in self.requirements if r.kind is kind]

    @property
    def scored(self) -> List[Requirement]:
        return [r for r in self.requirements if r.status.counts_toward_readiness]

    @property
    def blockers(self) -> List[Requirement]:
        return [r for r in self.requirements if r.status.is_blocker]

    @property
    def readiness(self) -> float:
        """Fraction of scored requirements that are satisfied. 1.0 if nothing was found."""
        scored = self.scored
        if not scored:
            return 1.0
        ok = sum(1 for r in scored if r.status is Status.OK)
        # A soft mismatch is 'probably fine', so it counts as half a point rather
        # than dragging the score down as hard as an outright missing file.
        soft = sum(1 for r in scored if r.status in (Status.MISMATCH, Status.UNKNOWN))
        return (ok + 0.5 * soft) / len(scored)

    @property
    def satisfied(self) -> int:
        return sum(1 for r in self.scored if r.status is Status.OK)

    @property
    def inconclusive(self) -> bool:
        """True when the environment under audit could not be inspected at all."""
        return any(r.meta.get("target_unavailable") for r in self.requirements)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root,
            "target": self.target,
            # `blocking` is the number to gate on. `readiness` is a progress bar:
            # its denominator moves as detection improves, so it is advisory only.
            "blocking": len(self.blockers),
            "inconclusive": self.inconclusive,
            "satisfied": self.satisfied,
            "checked": len(self.scored),
            "readiness": round(self.readiness, 4),
            "readiness_is_advisory": True,
            "counts": {
                s.value: sum(1 for r in self.requirements if r.status is s)
                for s in Status
            },
            "requirements": [r.to_dict() for r in self.requirements],
            "notes": self.notes,
            "suppressed": self.suppressed,
        }
