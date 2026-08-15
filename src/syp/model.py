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
    ASSET = "asset"
    EXTERNAL = "external"
    ENTRYPOINT = "entrypoint"


SECTION_ORDER = [
    Kind.SYSTEM,
    Kind.GIT,
    Kind.CONTAINER,
    Kind.PYTHON,
    Kind.ASSET,
    Kind.EXTERNAL,
    Kind.ENTRYPOINT,
]

SECTION_TITLES = {
    Kind.SYSTEM: "System",
    Kind.GIT: "Git",
    Kind.CONTAINER: "Container",
    Kind.PYTHON: "Python",
    Kind.ASSET: "Runtime assets",
    Kind.EXTERNAL: "External access",
    Kind.ENTRYPOINT: "Execution",
}


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
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["status"] = self.status.value
        return d


@dataclass
class Report:
    root: str
    requirements: List[Requirement] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root,
            "readiness": round(self.readiness, 4),
            "counts": {
                s.value: sum(1 for r in self.requirements if r.status is s)
                for s in Status
            },
            "requirements": [r.to_dict() for r in self.requirements],
            "notes": self.notes,
        }
