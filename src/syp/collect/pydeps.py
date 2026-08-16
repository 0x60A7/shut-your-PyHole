"""Python-layer requirements: declarations, and whether an interpreter satisfies them.

Declarations are gathered from every convention a repo might use at once —
research repos routinely ship three that disagree — and each package keeps a
pointer back to the file that declared it.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..context import RepoContext
from ..knowledge import AWKWARD_PACKAGES, GPU_PACKAGES
from ..model import Kind, Report, Requirement, Status
from ..util import normalize_dist, run, satisfies

REQ_FILE_PATTERNS = ("requirements*.txt", "requirements/*.txt", "requirements*.in")

_REQ_LINE = re.compile(
    r"""^\s*
    (?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)
    (?:\[(?P<extras>[^\]]*)\])?
    \s*(?P<spec>(?:(?:==|!=|>=|<=|~=|>|<|===)\s*[^,;\s]+\s*,?\s*)*)
    (?:;\s*(?P<marker>.+))?
    \s*$""",
    re.VERBOSE,
)
_VCS_LINE = re.compile(r"^\s*(?:(?P<name>[A-Za-z0-9._-]+)\s*@\s*)?(?P<url>(?:git\+|hg\+|https?://)\S+)")


@dataclass
class Declared:
    name: str
    spec: str = ""
    source: str = ""
    marker: str = ""
    url: str = ""
    extras: str = ""
    # Declared in a test/dev/docs extra: needed to develop the project, not to
    # run it. `requests` failing an audit over pytest-cov helps nobody.
    optional: bool = False
    group: str = ""


OPTIONAL_FILE = re.compile(r"(dev|test|doc|lint|ci|build|optional|extra)", re.IGNORECASE)
RUNTIME_GROUPS = {"", "all", "full", "runtime", "default"}


@dataclass
class Declarations:
    packages: Dict[str, Declared] = field(default_factory=dict)
    python_specs: List[Tuple[str, str]] = field(default_factory=list)  # (spec, source)
    index_urls: List[Tuple[str, str]] = field(default_factory=list)
    conda_packages: List[Tuple[str, str]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    # Every declaration of every package, so contradictions stay visible.
    all_specs: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)

    def add(self, dec: Declared) -> None:
        key = normalize_dist(dec.name)
        if not key or key in ("python",):
            return
        self.all_specs.setdefault(key, []).append((dec.spec, dec.source))
        existing = self.packages.get(key)
        # First declaration wins for the spec; keep a record of every source file.
        if existing is None:
            self.packages[key] = dec
        elif existing.optional and not dec.optional:
            dec.source = f"{existing.source}, {dec.source}"
            self.packages[key] = dec
        elif dec.source not in existing.source:
            existing.source = f"{existing.source}, {dec.source}"

    @property
    def duplicate_specs(self) -> Dict[str, List[Tuple[str, str]]]:
        return {
            name: pins
            for name, pins in self.all_specs.items()
            if len({spec for spec, _ in pins if spec}) > 1
        }


def collect(ctx: RepoContext, report: Report) -> None:
    decl = gather(ctx)
    if not decl.sources:
        report.add(
            Requirement(
                kind=Kind.PYTHON,
                name="dependency declaration",
                status=Status.UNKNOWN,
                detail="no requirements.txt, pyproject.toml, setup.py or environment.yml found",
                manual="Dependencies are probably documented in prose — check the README.",
            )
        )
        return

    report.add(
        Requirement(
            kind=Kind.PYTHON,
            name="declarations",
            status=Status.INFO,
            detail=f"{len(decl.packages)} package(s) from {', '.join(decl.sources)}",
        )
    )

    target = ctx.target
    installed = _installed_distributions(target)
    py_version = _interpreter_version(target)

    _check_python_version(decl, py_version, target, report)
    _check_packages(ctx, decl, installed, report)
    _check_indexes(decl, report)
    _check_accelerator(ctx, installed, report)


# --- gathering --------------------------------------------------------------


def gather(ctx: RepoContext) -> Declarations:
    decl = Declarations()
    for pattern in REQ_FILE_PATTERNS:
        for rel in ctx.rglob(pattern) if "/" in pattern else ctx.glob(pattern):
            if rel.endswith(".txt") or rel.endswith(".in"):
                _parse_requirements(ctx, rel, decl, set())
    if ctx.exists("pyproject.toml"):
        _parse_pyproject(ctx, decl)
    if ctx.exists("setup.py"):
        _parse_setup_py(ctx, decl)
    if ctx.exists("setup.cfg"):
        _parse_setup_cfg(ctx, decl)
    for name in ("environment.yml", "environment.yaml", "conda.yml", "conda.yaml", "env.yml"):
        if ctx.exists(name):
            _parse_conda(ctx, name, decl)
    _parse_dockerfiles(ctx, decl)
    return decl


_DOCKER_PY = re.compile(
    r"(?:FROM\s+\S*python:(\d\.\d+)"          # FROM python:3.9
    r"|conda\s+create[^\n]*python\s*=\s*(\d\.\d+)"
    r"|apt-get[^\n]*\bpython(\d\.\d+)\b"
    r"|ENV\s+PYTHON_VERSION[= ]\s*(\d\.\d+))",
    re.IGNORECASE,
)


def _parse_dockerfiles(ctx: RepoContext, decl: Declarations) -> None:
    """A Dockerfile often carries the only honest statement of the Python version.

    Recorded as a declaration so it can be compared against, and disagreed with,
    like any other.
    """
    for rel in ctx.files:
        base = rel.split("/")[-1].lower()
        if not (base == "dockerfile" or base.startswith("dockerfile.") or base.endswith(".dockerfile")):
            continue
        text = ctx.text(rel)
        match = _DOCKER_PY.search(text)
        if match:
            version = next(g for g in match.groups() if g)
            decl.python_specs.append((f"=={version}", ctx.source_ref(rel, match.group(0)[:30])))


def _parse_requirements(ctx: RepoContext, rel: str, decl: Declarations, seen: set) -> None:
    if rel in seen or not ctx.exists(rel):
        return
    seen.add(rel)
    decl.sources.append(rel)
    # requirements-dev.txt / requirements/test.txt describe developing the
    # project, not running it.
    file_optional = bool(OPTIONAL_FILE.search(os.path.basename(rel).replace("requirements", "")))
    for lineno, raw in enumerate(ctx.text(rel).splitlines(), start=1):
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        source = f"{rel}:{lineno}"
        if line.startswith(("-r ", "--requirement ")):
            target = line.split(None, 1)[1].strip()
            nested = os.path.normpath(os.path.join(os.path.dirname(rel), target)).replace(os.sep, "/")
            _parse_requirements(ctx, nested, decl, seen)
            continue
        if line.startswith(("-i ", "--index-url", "--extra-index-url", "-f ", "--find-links")):
            url = line.split(None, 1)[1].strip() if " " in line else line
            decl.index_urls.append((url, source))
            continue
        if line.startswith("-e") or line.startswith("--editable"):
            continue
        if line.startswith("-"):
            continue
        vcs = _VCS_LINE.match(line)
        if vcs and vcs.group("url"):
            name = vcs.group("name") or _name_from_url(vcs.group("url"))
            decl.add(Declared(name=name, source=source, url=vcs.group("url")))
            continue
        match = _REQ_LINE.match(line)
        if match:
            decl.add(
                Declared(
                    name=match.group("name"),
                    spec=(match.group("spec") or "").strip().rstrip(","),
                    source=source,
                    marker=(match.group("marker") or "").strip(),
                    extras=(match.group("extras") or "").strip(),
                    optional=file_optional,
                    group="dev" if file_optional else "",
                )
            )


def _name_from_url(url: str) -> str:
    tail = url.split("#egg=")[-1] if "#egg=" in url else url.rstrip("/").split("/")[-1]
    return re.sub(r"\.git$", "", tail)


def _load_toml(text: str) -> Optional[dict]:
    try:
        import tomllib  # type: ignore[import-not-found]

        return tomllib.loads(text)
    except ImportError:
        pass
    try:
        import tomli  # type: ignore[import-not-found]

        return tomli.loads(text)
    except Exception:  # not installed, or the file is malformed
        return None


def _parse_pyproject(ctx: RepoContext, decl: Declarations) -> None:
    rel = "pyproject.toml"
    text = ctx.text(rel)
    decl.sources.append(rel)
    data = _load_toml(text)
    if data is None:
        # No TOML parser available: fall back to scraping the dependency arrays.
        for match in re.finditer(r'"([A-Za-z0-9][A-Za-z0-9._-]*)\s*([^"]*)"', text):
            decl.add(Declared(name=match.group(1), spec=match.group(2).strip(), source=rel))
        match = re.search(r'requires-python\s*=\s*"([^"]+)"', text)
        if match:
            decl.python_specs.append((match.group(1), rel))
        return

    project = data.get("project") or {}
    for item in project.get("dependencies") or []:
        _add_pep508(decl, item, ctx.source_ref(rel, item[:20]))
    for name, group in (project.get("optional-dependencies") or {}).items():
        for item in group:
            _add_pep508(decl, item, rel, optional=name.lower() not in RUNTIME_GROUPS, group=name)
    if project.get("requires-python"):
        decl.python_specs.append((project["requires-python"], rel))

    poetry = ((data.get("tool") or {}).get("poetry") or {}).get("dependencies") or {}
    for name, value in poetry.items():
        if name.lower() == "python":
            spec = value if isinstance(value, str) else str(value.get("version", ""))
            decl.python_specs.append((_poetry_spec(spec), rel))
            continue
        spec = value if isinstance(value, str) else str((value or {}).get("version", ""))
        decl.add(Declared(name=name, spec=_poetry_spec(spec), source=rel))


def _poetry_spec(spec: str) -> str:
    """Translate poetry's caret/tilde shorthand into something comparable."""
    spec = (spec or "").strip()
    if spec.startswith("^"):
        return f">={spec[1:]}"
    if spec.startswith("~"):
        return f">={spec[1:]}"
    if spec and spec[0].isdigit():
        return f"=={spec}"
    return spec


def _add_pep508(
    decl: Declarations, item: str, source: str, optional: bool = False, group: str = ""
) -> None:
    match = _REQ_LINE.match(item.strip())
    if match:
        decl.add(
            Declared(
                name=match.group("name"),
                spec=(match.group("spec") or "").strip().rstrip(","),
                source=source,
                marker=(match.group("marker") or "").strip(),
                optional=optional,
                group=group,
            )
        )


def _parse_setup_py(ctx: RepoContext, decl: Declarations) -> None:
    """Read setup.py with the AST, not with regexes.

    Pairing quotes by hand fails on the first apostrophe in a comment: one
    "preferrably by OS's package manager" in detectron2's setup.py desynchronised
    every string after it, so 4 of its 14 dependencies were parsed and the other
    10 were then reported as undeclared imports.
    """
    rel = "setup.py"
    text = ctx.text(rel)
    decl.sources.append(rel)
    try:
        tree = ast.parse(text, filename=rel)
    except (SyntaxError, ValueError):
        _parse_setup_py_regex(text, rel, decl)
        return

    # `deps = [...]` followed by `install_requires=deps` is common enough to follow.
    assigned: Dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned[target.id] = node.value

    def resolve(value):
        if isinstance(value, ast.Name):
            return assigned.get(value.id)
        return value

    def strings(value) -> List[str]:
        value = resolve(value)
        out: List[str] = []
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            for element in value.elts:
                element = resolve(element)
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    out.append(element.value)
        return out

    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name != "setup":
            continue
        for keyword in node.keywords:
            if keyword.arg == "install_requires":
                found = True
                for item in strings(keyword.value):
                    _add_pep508(decl, item, rel)
            elif keyword.arg == "extras_require":
                mapping = resolve(keyword.value)
                if isinstance(mapping, ast.Dict):
                    for key, value in zip(mapping.keys, mapping.values):
                        group = key.value if isinstance(key, ast.Constant) else ""
                        optional = str(group).lower() not in RUNTIME_GROUPS
                        for item in strings(value):
                            _add_pep508(decl, item, rel, optional=optional, group=str(group))
            elif keyword.arg == "python_requires":
                value = resolve(keyword.value)
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    decl.python_specs.append((value.value, rel))
    if not found:
        _parse_setup_py_regex(text, rel, decl)


def _parse_setup_py_regex(text: str, rel: str, decl: Declarations) -> None:
    match = re.search(r"install_requires\s*=\s*\[(.*?)\]", text, re.DOTALL)
    if match:
        for item in re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)):
            _add_pep508(decl, item, rel)
    match = re.search(r"python_requires\s*=\s*['\"]([^'\"]+)['\"]", text)
    if match:
        decl.python_specs.append((match.group(1), rel))


def _parse_setup_cfg(ctx: RepoContext, decl: Declarations) -> None:
    rel = "setup.cfg"
    text = ctx.text(rel)
    match = re.search(r"^\s*install_requires\s*=\s*\n((?:\s+\S.*\n)+)", text, re.MULTILINE)
    if match:
        decl.sources.append(rel)
        for line in match.group(1).splitlines():
            _add_pep508(decl, line.strip(), rel)
    match = re.search(r"^\s*python_requires\s*=\s*(.+)$", text, re.MULTILINE)
    if match:
        decl.python_specs.append((match.group(1).strip(), rel))


def _parse_conda(ctx: RepoContext, rel: str, decl: Declarations) -> None:
    decl.sources.append(rel)
    in_deps = False
    in_pip = False
    for lineno, raw in enumerate(ctx.text(rel).splitlines(), start=1):
        line = raw.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if stripped.startswith("dependencies:"):
            in_deps, in_pip = True, False
            continue
        if in_deps and indent == 0 and not stripped.startswith("-"):
            in_deps = False
            continue
        if not in_deps:
            continue
        if re.match(r"^-\s*pip\s*:", stripped):
            in_pip = True
            continue
        if not stripped.startswith("-"):
            continue
        item = stripped.lstrip("- ").strip()
        if in_pip and indent <= 2:
            in_pip = False
        source = f"{rel}:{lineno}"
        if item.lower().startswith("python") and re.match(r"python\s*[=<>]", item):
            decl.python_specs.append((_conda_spec(item.split("python", 1)[1]), source))
            continue
        if in_pip:
            _add_pep508(decl, item, source)
        else:
            decl.conda_packages.append((item, source))


def _conda_spec(spec: str) -> str:
    """conda writes `python=3.9`; PEP 440 wants `==3.9`."""
    spec = spec.strip()
    if spec.startswith("=") and not spec.startswith("=="):
        return "==" + spec.lstrip("=")
    return spec


# --- verification -----------------------------------------------------------


def _interpreter_version(target) -> Optional[str]:
    code, out = target.python(["-c", "import sys;print('%d.%d.%d'%sys.version_info[:3])"])
    return out.strip().splitlines()[-1].strip() if code == 0 and out.strip() else None


def _installed_distributions(target) -> Optional[Dict[str, str]]:
    script = (
        "import json;"
        "import importlib.metadata as m;"
        "print(json.dumps({(d.metadata['Name'] or '').lower():(d.version or '') "
        "for d in m.distributions() if d.metadata['Name']}))"
    )
    code, out = target.python(["-c", script], timeout=120)
    if code != 0:
        return None
    try:
        raw = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None
    return {normalize_dist(k): v for k, v in raw.items()}


def _check_python_version(
    decl: Declarations,
    py_version: Optional[str],
    target,
    report: Report,
) -> None:
    interpreter = target.python_exe
    label = f"{py_version or '?'} ({target.describe()})"
    if not decl.python_specs:
        report.add(
            Requirement(
                kind=Kind.PYTHON,
                name="python version",
                status=Status.INFO,
                detail=f"nothing declared; checking against {label}",
                meta={"interpreter": interpreter},
            )
        )
        return

    spec, source = decl.python_specs[0]
    others = {s for s, _ in decl.python_specs[1:] if s != spec}
    if not py_version:
        report.add(
            Requirement(
                kind=Kind.PYTHON,
                name=f"python {spec}",
                status=Status.UNKNOWN,
                detail="could not run the target interpreter",
                source=source,
            )
        )
        return

    verdict = satisfies(py_version, spec)
    status = Status.OK if verdict is not False else Status.MISMATCH
    detail = f"declared {spec} by {source}, found {label}"
    if others:
        detail += f"; other files declare {', '.join(sorted(others))}"
        status = Status.MISMATCH if status is Status.OK else status
    report.add(
        Requirement(
            kind=Kind.PYTHON,
            name="python version",
            status=status,
            detail=detail,
            source=source,
            manual=None if status is Status.OK else f"Use an interpreter matching {spec}.",
            meta={"interpreter": interpreter},
        )
    )


def _check_packages(
    ctx: RepoContext,
    decl: Declarations,
    installed: Optional[Dict[str, str]],
    report: Report,
) -> None:
    target = ctx.target
    if installed is None:
        report.add(
            Requirement(
                kind=Kind.PYTHON,
                name="installed packages",
                status=Status.UNKNOWN,
                detail=target.problem or f"could not query {target.describe()}",
                manual="Point --target at an environment that exists, or create one.",
            )
        )
        return

    assumed = set(ctx.config.assume_installed)
    ok: List[str] = []
    problems: List[Requirement] = []
    for key, dec in sorted(decl.packages.items()):
        version = installed.get(key) or ("assumed" if key in assumed else None)
        awkward = AWKWARD_PACKAGES.get(key)
        label = dec.name + (f" {dec.spec}" if dec.spec else "")
        if version is None:
            problems.append(
                Requirement(
                    kind=Kind.PYTHON,
                    name=label,
                    status=Status.MISSING,
                    detail="not installed" + (" (not a plain pip install)" if awkward else ""),
                    source=dec.source,
                    fix=None if awkward else target.pip_command(f"{dec.name}{dec.spec}"),
                    manual=awkward.hint or awkward.note if awkward else None,
                    explain=awkward.note if awkward else None,
                    meta={"package": dec.name, "optional": dec.optional, "awkward": bool(awkward)},
                )
            )
            continue
        verdict = satisfies(version, dec.spec) if dec.spec and version != "assumed" else True
        if verdict is False:
            problems.append(
                Requirement(
                    kind=Kind.PYTHON,
                    name=label,
                    status=Status.MISMATCH,
                    detail=f"installed {version}",
                    source=dec.source,
                    fix=target.pip_command(f"{dec.name}{dec.spec}"),
                    explain=awkward.note if awkward else None,
                    meta={"package": dec.name, "installed": version, "optional": dec.optional},
                )
            )
        else:
            ok.append(f"{dec.name}=={version}")

    if ok:
        report.add(
            Requirement(
                kind=Kind.PYTHON,
                name=f"declared packages installed ({len(ok)}/{len(decl.packages)})",
                status=Status.OK,
                detail=f"checked against {target.describe()}",
                meta={"packages": ok, "verbose_list": True},
            )
        )
    _report_missing(ctx, decl, problems, ok, report)

    if decl.conda_packages:
        report.add(
            Requirement(
                kind=Kind.PYTHON,
                name=f"conda-only packages ({len(decl.conda_packages)})",
                status=Status.INFO,
                detail=", ".join(p for p, _ in decl.conda_packages[:6])
                + ("..." if len(decl.conda_packages) > 6 else ""),
                source=decl.conda_packages[0][1].split(":")[0],
                meta={"packages": [p for p, _ in decl.conda_packages], "verbose_list": True},
            )
        )


COLLAPSE_MIN = 4
COLLAPSE_RATIO = 0.75


def _report_missing(
    ctx: RepoContext,
    decl: Declarations,
    problems: List[Requirement],
    ok: List[str],
    report: Report,
) -> None:
    """One unprovisioned environment is one problem, not sixty.

    Listing every declared package separately turned `requests` — a pure-Python
    library — into sixteen blockers, none of which told you anything beyond
    "you have not installed this project yet".
    """
    optional = [p for p in problems if p.meta.get("optional")]
    required = [p for p in problems if not p.meta.get("optional")]
    missing = [p for p in required if p.status is Status.MISSING]
    required_total = len(missing) + len(ok)

    if optional:
        report.add(
            Requirement(
                kind=Kind.PYTHON,
                name=f"dev/test extras absent ({len(optional)})",
                status=Status.INFO,
                detail="declared for developing the project, not for running it",
                meta={"packages": [p.name for p in optional], "verbose_list": True},
            )
        )

    unprovisioned = len(missing) >= COLLAPSE_MIN and (
        not ok or len(missing) >= COLLAPSE_RATIO * max(required_total, 1)
    )
    if not unprovisioned:
        report.extend(required)
        return

    # Keep the detail, drop the noise: individual findings stay for `explain`
    # and -v, but they are symptoms of the single fact reported as the blocker.
    for req in required:
        req.status = Status.INFO if req.status is Status.MISSING else req.status
        req.fix = None
    awkward = [p.name for p in missing if p.meta.get("awkward")]
    requirements_file = next(
        (s for s in decl.sources if s.startswith("requirements") and s.endswith(".txt")), None
    )
    report.add(
        Requirement(
            kind=Kind.PYTHON,
            name=f"environment not provisioned ({len(missing)} of {required_total} packages absent)",
            status=Status.MISSING,
            detail=f"nothing to check against in {ctx.target.describe()}",
            fix=(
                f"{ctx.target.quote(ctx.target.python_exe)} -m pip install -r {requirements_file}"
                if requirements_file and not ctx.target.is_container
                else None
            ),
            manual=None
            if requirements_file
            else "Create an environment for this project before trusting anything below.",
            explain=(
                f"{len(awkward)} of these will not install from a plain requirements file: "
                + ", ".join(awkward[:5])
            )
            if awkward
            else None,
            meta={
                "packages": [p.name for p in missing],
                "verbose_list": True,
                "unprovisioned": True,
            },
        )
    )
    report.extend(required)


def _check_indexes(decl: Declarations, report: Report) -> None:
    for url, source in decl.index_urls[:4]:
        report.add(
            Requirement(
                kind=Kind.PYTHON,
                name="custom package index",
                status=Status.INFO,
                detail=url,
                source=source,
                explain=(
                    "Installing without this index will silently pull a different build "
                    "(the CPU-only torch wheel is the usual casualty)."
                ),
            )
        )


_TORCH_PROBE = (
    "import json,torch\n"
    "info={'v':torch.__version__,'cuda':torch.version.cuda,"
    "'hip':getattr(torch.version,'hip',None),'avail':torch.cuda.is_available(),"
    "'n':torch.cuda.device_count() if torch.cuda.is_available() else 0,"
    "'mps':bool(getattr(getattr(torch.backends,'mps',None),'is_available',lambda:False)())}\n"
    "print(json.dumps(info))"
)


def _check_accelerator(ctx: RepoContext, installed: Optional[Dict[str, str]], report: Report) -> None:
    """Ask torch itself, rather than inferring from the driver.

    torch is the only component that knows whether it was built for CUDA, ROCm,
    Metal or nothing at all — and 'installed correctly but CPU-only' is the most
    common silent failure in this whole ecosystem.
    """
    if not installed or "torch" not in installed:
        return
    code, out = ctx.target.python(["-c", _TORCH_PROBE], timeout=180)
    if code != 0:
        report.add(
            Requirement(
                kind=Kind.PYTHON,
                name="torch import",
                status=Status.MISMATCH,
                detail=f"torch {installed['torch']} is installed but fails to import",
                explain=(out or "").strip().splitlines()[-1] if (out or "").strip() else None,
                manual="A broken torch invalidates everything below it; fix this first.",
            )
        )
        return
    try:
        info = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return

    backend = "ROCm " + str(info["hip"]) if info.get("hip") else (
        "CUDA " + str(info["cuda"]) if info.get("cuda") else "CPU-only"
    )
    if info.get("avail"):
        report.add(
            Requirement(
                kind=Kind.PYTHON,
                name="torch accelerator",
                status=Status.OK,
                detail=f"torch {info['v']} ({backend}), {info['n']} device(s) visible",
            )
        )
    elif info.get("mps"):
        report.add(
            Requirement(
                kind=Kind.PYTHON,
                name="torch accelerator",
                status=Status.OK,
                detail=f"torch {info['v']} on Apple Metal (MPS)",
                explain="Code that hardcodes .cuda() will still fail; MPS needs device='mps'.",
            )
        )
    else:
        report.add(
            Requirement(
                kind=Kind.PYTHON,
                name="torch accelerator",
                status=Status.MISMATCH,
                detail=f"torch {info['v']} sees no usable GPU ({backend})",
                manual="Reinstall torch from the index matching your driver"
                if backend == "CPU-only"
                else "The build expects a GPU that is not visible here.",
            )
        )


def declared_gpu_packages(decl: Declarations) -> List[str]:
    return sorted(name for name in decl.packages if name in GPU_PACKAGES)
