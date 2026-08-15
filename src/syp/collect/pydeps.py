"""Python-layer requirements: declarations, and whether an interpreter satisfies them.

Declarations are gathered from every convention a repo might use at once —
research repos routinely ship three that disagree — and each package keeps a
pointer back to the file that declared it.
"""

from __future__ import annotations

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


@dataclass
class Declarations:
    packages: Dict[str, Declared] = field(default_factory=dict)
    python_specs: List[Tuple[str, str]] = field(default_factory=list)  # (spec, source)
    index_urls: List[Tuple[str, str]] = field(default_factory=list)
    conda_packages: List[Tuple[str, str]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)

    def add(self, dec: Declared) -> None:
        key = normalize_dist(dec.name)
        if not key or key in ("python",):
            return
        existing = self.packages.get(key)
        # First declaration wins for the spec; keep a record of every source file.
        if existing is None:
            self.packages[key] = dec
        elif dec.source not in existing.source:
            existing.source = f"{existing.source}, {dec.source}"


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

    interpreter, why = _target_interpreter(ctx)
    installed = _installed_distributions(interpreter)
    py_version = _interpreter_version(interpreter)

    _check_python_version(decl, py_version, interpreter, why, report)
    _check_packages(decl, installed, interpreter, report)
    _check_indexes(decl, report)
    _check_torch(interpreter, installed, report)


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
    for group in (project.get("optional-dependencies") or {}).values():
        for item in group:
            _add_pep508(decl, item, rel)
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


def _add_pep508(decl: Declarations, item: str, source: str) -> None:
    match = _REQ_LINE.match(item.strip())
    if match:
        decl.add(
            Declared(
                name=match.group("name"),
                spec=(match.group("spec") or "").strip().rstrip(","),
                source=source,
                marker=(match.group("marker") or "").strip(),
            )
        )


def _parse_setup_py(ctx: RepoContext, decl: Declarations) -> None:
    rel = "setup.py"
    text = ctx.text(rel)
    decl.sources.append(rel)
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


def _target_interpreter(ctx: RepoContext) -> Tuple[str, str]:
    override = os.environ.get("SYP_PYTHON")
    if override:
        return override, "SYP_PYTHON"
    for candidate in (".venv", "venv", "env", ".env"):
        for sub in ("bin/python", "Scripts/python.exe"):
            path = ctx.abspath(f"{candidate}/{sub}")
            if os.path.exists(path):
                return path, f"{candidate}/"
    return sys.executable, "active interpreter"


def _interpreter_version(interpreter: str) -> Optional[str]:
    code, out = run([interpreter, "-c", "import sys;print('%d.%d.%d'%sys.version_info[:3])"])
    return out.strip() if code == 0 else None


def _installed_distributions(interpreter: str) -> Optional[Dict[str, str]]:
    script = (
        "import json;"
        "import importlib.metadata as m;"
        "print(json.dumps({(d.metadata['Name'] or '').lower():(d.version or '') "
        "for d in m.distributions() if d.metadata['Name']}))"
    )
    code, out = run([interpreter, "-c", script], timeout=60)
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
    interpreter: str,
    why: str,
    report: Report,
) -> None:
    label = f"{py_version or '?'} ({why})"
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
    decl: Declarations,
    installed: Optional[Dict[str, str]],
    interpreter: str,
    report: Report,
) -> None:
    if installed is None:
        report.add(
            Requirement(
                kind=Kind.PYTHON,
                name="installed packages",
                status=Status.UNKNOWN,
                detail=f"could not query {interpreter}",
            )
        )
        return

    ok: List[str] = []
    problems: List[Requirement] = []
    for key, dec in sorted(decl.packages.items()):
        version = installed.get(key)
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
                    fix=None if awkward else f"pip install '{dec.name}{dec.spec}'",
                    manual=awkward.hint or awkward.note if awkward else None,
                    explain=awkward.note if awkward else None,
                    meta={"package": dec.name},
                )
            )
            continue
        verdict = satisfies(version, dec.spec) if dec.spec else True
        if verdict is False:
            problems.append(
                Requirement(
                    kind=Kind.PYTHON,
                    name=label,
                    status=Status.MISMATCH,
                    detail=f"installed {version}",
                    source=dec.source,
                    fix=f"pip install '{dec.name}{dec.spec}'",
                    explain=awkward.note if awkward else None,
                    meta={"package": dec.name, "installed": version},
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
                detail=f"checked against {os.path.basename(interpreter)}",
                meta={"packages": ok, "verbose_list": True},
            )
        )
    report.extend(problems)

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


def _check_torch(interpreter: str, installed: Optional[Dict[str, str]], report: Report) -> None:
    if not installed or "torch" not in installed:
        return
    code, out = run(
        [
            interpreter,
            "-c",
            "import torch,json;print(json.dumps({'v':torch.__version__,"
            "'cuda':torch.version.cuda,'avail':torch.cuda.is_available(),"
            "'n':torch.cuda.device_count()}))",
        ],
        timeout=120,
    )
    if code != 0:
        report.add(
            Requirement(
                kind=Kind.PYTHON,
                name="torch import",
                status=Status.MISMATCH,
                detail=f"torch {installed['torch']} is installed but fails to import",
                explain=out.strip().splitlines()[-1] if out.strip() else None,
            )
        )
        return
    try:
        info = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return
    if info.get("avail"):
        report.add(
            Requirement(
                kind=Kind.PYTHON,
                name="torch CUDA",
                status=Status.OK,
                detail=f"torch {info['v']} (cuda {info['cuda']}), {info['n']} device(s) visible",
            )
        )
    else:
        report.add(
            Requirement(
                kind=Kind.PYTHON,
                name="torch CUDA",
                status=Status.MISMATCH,
                detail=f"torch {info['v']} reports no usable GPU"
                + (" — this is a CPU-only build" if not info.get("cuda") else f" (built for cuda {info['cuda']})"),
                manual="Reinstall torch from the index matching your driver's CUDA version.",
            )
        )


def declared_gpu_packages(decl: Declarations) -> List[str]:
    return sorted(name for name in decl.packages if name in GPU_PACKAGES)
