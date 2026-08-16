"""Makefiles: the build instructions that are neither a manifest nor a script.

A repo with a Makefile has already written down how to build it, fetch its data
and run its demo — in a file no dependency tool reads. `make` itself is also a
requirement, and one that is routinely absent on Windows.

The same scoping discipline applies as everywhere else: a target named `style`
or `publish` is the maintainers' business, not a requirement of running the
project, so only targets that build, fetch or run are treated as requirements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

MAKEFILE_NAMES = ("makefile", "gnumakefile", "bsdmakefile")

# Targets that exist for the people who maintain the project.
MAINTENANCE = {
    "quality", "style", "lint", "format", "fmt", "check", "checks", "pretty",
    "publish", "release", "upload", "dist", "bump", "tag", "sign",
    "clean", "distclean", "coverage", "ci", "docs", "doc", "html", "man",
    "changelog", "typecheck", "mypy", "precommit", "pre-commit", "watch",
}
TEST_TARGETS = {"test", "tests", "pytest", "unittest"}

_ASSIGN = re.compile(r"^\s*([A-Za-z_][\w.-]*)\s*(:?\+?\??=)\s*(.*)$")
_RULE = re.compile(r"^([^\s:=#][^:=]*?)\s*:(?!=)\s*(.*)$")
_VAR_REF = re.compile(r"\$[({]([A-Za-z_][\w.-]*)[)}]")

# What a recipe line tells us it needs.
BUILD_TOOLS = re.compile(
    r"\b(nvcc|cmake|gcc|g\+\+|clang|clang\+\+|cc|make\s+-C|ninja|cargo|go\s+build|"
    # `python setup.py check` is a lint, not a build; only the building
    # subcommands count.
    r"python\s+setup\.py\s+(?:build|install|develop|bdist|sdist)|"
    r"maturin|meson|bazel|protoc|swig|cython)\b"
)
FETCH_TOOLS = re.compile(
    r"\b(wget|curl|gdown|aria2c|git\s+clone|huggingface-cli\s+download|hf\s+download|"
    r"aws\s+s3\s+cp|rsync|azcopy|unzip|tar\s+-x)\b"
)
INSTALL_TOOLS = re.compile(r"\b(pip\s+install|python\s+-m\s+pip\s+install|conda\s+install|uv\s+pip)\b")
RUN_TOOLS = re.compile(r"\b(python\d?|python3|torchrun|accelerate\s+launch|\./\w+)\b")


@dataclass
class Target:
    name: str
    prerequisites: List[str] = field(default_factory=list)
    recipe: List[str] = field(default_factory=list)
    lineno: int = 0
    phony: bool = False

    @property
    def body(self) -> str:
        return "\n".join(self.recipe)

    @property
    def is_maintenance(self) -> bool:
        # Running the test suite is not a prerequisite for running the project.
        name = self.name.lower()
        return name in MAINTENANCE or name in TEST_TARGETS or name.startswith("test")

    @property
    def builds(self) -> bool:
        return bool(BUILD_TOOLS.search(self.body))

    @property
    def fetches(self) -> bool:
        return bool(FETCH_TOOLS.search(self.body))

    @property
    def installs(self) -> bool:
        return bool(INSTALL_TOOLS.search(self.body))

    @property
    def runs(self) -> bool:
        return bool(RUN_TOOLS.search(self.body)) and not self.is_maintenance


@dataclass
class Makefile:
    path: str
    variables: Dict[str, str] = field(default_factory=dict)
    targets: Dict[str, Target] = field(default_factory=dict)

    def expand(self, text: str, depth: int = 0) -> str:
        """Substitute `$(VAR)` / `${VAR}`; unknown names are left alone."""
        if depth > 5:
            return text

        def replace(match):
            value = self.variables.get(match.group(1))
            return self.expand(value, depth + 1) if value is not None else match.group(0)

        return _VAR_REF.sub(replace, text)

    def interesting(self) -> List[Target]:
        """Targets that build, fetch, install or run — not the tidying ones."""
        return [
            t
            for t in self.targets.values()
            if not t.is_maintenance and (t.builds or t.fetches or t.installs or t.runs)
        ]

    def recipe_text(self) -> str:
        """Every non-maintenance recipe, expanded. Used to look for downloads."""
        return "\n".join(
            self.expand(t.body) for t in self.targets.values() if not t.is_maintenance
        )


def is_makefile(rel: str) -> bool:
    base = rel.split("/")[-1].lower()
    return base in MAKEFILE_NAMES or base.endswith(".mk")


def parse(text: str, path: str = "Makefile") -> Makefile:
    """A tolerant Makefile reader: rules, recipes, variables and .PHONY."""
    result = Makefile(path=path)
    phony: List[str] = []
    current: Optional[Target] = None
    pending = ""

    for lineno, raw in enumerate(text.splitlines(), start=1):
        if raw.rstrip().endswith("\\"):
            pending += raw.rstrip()[:-1] + " "
            continue
        line = pending + raw
        pending = ""

        if not line.strip() or line.lstrip().startswith("#"):
            continue

        # A tab (or a continuation of one) means we are inside a recipe.
        if line.startswith("\t") and current is not None:
            recipe_line = line.lstrip("\t").strip()
            if recipe_line.startswith(("@", "-", "+")):
                recipe_line = recipe_line[1:].strip()
            if recipe_line:
                current.recipe.append(recipe_line)
            continue

        assignment = _ASSIGN.match(line)
        if assignment and not line.startswith("\t"):
            result.variables[assignment.group(1)] = assignment.group(3).strip()
            current = None
            continue

        rule = _RULE.match(line)
        if rule:
            names = rule.group(1).split()
            prerequisites = rule.group(2).split()
            if names and names[0] == ".PHONY":
                phony.extend(prerequisites)
                current = None
                continue
            current = None
            for name in names:
                if name.startswith(".") or "%" in name:
                    continue  # special targets and pattern rules
                target = Target(name=name, prerequisites=prerequisites, lineno=lineno)
                result.targets[name] = target
                current = target
            continue
        current = None

    for name in phony:
        if name in result.targets:
            result.targets[name].phony = True
    return result
