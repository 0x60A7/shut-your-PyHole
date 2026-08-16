"""Runtime assets: the checkpoints, body models and demo inputs the code opens
at run time but nothing declares.

This is where the cross-referencing happens. Three independent scans —
paths the code opens, URLs the setup scripts fetch, and the curated licence
registry — are joined so that a missing file can be reported as *fetchable by
this script* or *needs an account here* rather than merely absent.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..context import RepoContext
from ..knowledge import (
    ARCHIVE_EXTENSIONS,
    MEDIA_EXTENSIONS,
    MODEL_EXTENSIONS,
    match_gated,
    match_host,
)
from ..integrity import declared_checksums, inspect as inspect_file, verify_checksum
from ..model import Kind, Report, Requirement, Status
from .. import pyscan
from ..reach import out_of_scope, reachable
from ..util import dedupe, human_size, path_size, which
from .build import repo_makefiles
from .entrypoint import entry_file

SCAN_SUFFIXES = (".py", ".sh", ".bash", ".yaml", ".yml", ".cfg", ".json", ".ipynb", ".md", ".txt", ".toml")
CODE_SUFFIXES = (".py", ".sh", ".bash", ".yaml", ".yml", ".cfg", ".ipynb")

# Longest first, so `.pth.tar` wins over `.pth` and `.pth` over `.pt`.
_ASSET_EXT_RE = "|".join(
    re.escape(e.lstrip("."))
    for e in sorted(MODEL_EXTENSIONS + MEDIA_EXTENSIONS, key=len, reverse=True)
)
_QUOTED = re.compile(r"""['"]([^'"\n]{3,200}?\.(?:%s))['"]""" % _ASSET_EXT_RE, re.IGNORECASE)
_BARE = re.compile(r"(?<![\w/'\"])((?:[\w.-]+/)+[\w.-]+\.(?:%s))\b" % _ASSET_EXT_RE, re.IGNORECASE)
# Any scheme, not just http: `detectron2://COCO-Detection/.../model.pkl` is a
# model-zoo URI whose tail looks exactly like a relative path.
_URL = re.compile(r"\w+://[^\s'\"<>)\]}\\]+")

# Lines that write rather than read; their paths are outputs, not requirements.
_OUTPUT_HINT = re.compile(
    r"\b(save|dump|write|to_csv|savefig|export|output_path|out_path|out_file|"
    r"savez|imwrite|VideoWriter|makedirs)\b",
    re.IGNORECASE,
)
_REJECT_CHARS = re.compile(r"[{}\[\]$*<>%\\|?]")

_DOWNLOAD_CMDS = re.compile(
    r"\b(wget|curl|gdown|aria2c|azcopy|rsync|huggingface-cli\s+download|hf\s+download|"
    r"aws\s+s3\s+cp|git\s+clone|git\s+lfs\s+clone)\b",
    re.IGNORECASE,
)
_GDRIVE_ID = re.compile(r"(?:--id\s+|[?&]id=|/d/)([A-Za-z0-9_-]{20,})")


@dataclass
class Downloader:
    """A script (or README block) that fetches things from the network."""

    source: str
    text: str
    urls: List[str] = field(default_factory=list)

    def mentions(self, *needles: str) -> bool:
        return any(n and n in self.text for n in needles)


@dataclass
class Candidate:
    path: str
    source: str
    is_demo: bool = False
    observed: bool = False


def collect(ctx: RepoContext, report: Report) -> None:
    downloaders = _find_downloaders(ctx)
    candidates = _find_candidates(ctx)
    candidates.extend(_observed_candidates(ctx, candidates))
    ignored = [c.path for c in candidates if ctx.config.ignores_path(c.path)]
    candidates = [c for c in candidates if not ctx.config.ignores_path(c.path)]
    if ignored:
        # Silent filtering is how a report starts lying; say what was dropped.
        report.suppressed.extend(ignored)
        report.notes.append(
            f"{len(ignored)} asset path(s) ignored by {ctx.config.source}: "
            + ", ".join(ignored[:4])
            + (f" (+{len(ignored) - 4} more)" if len(ignored) > 4 else "")
        )
    if not candidates and not downloaders:
        return

    present: List[Tuple[Candidate, int]] = []
    missing: List[Requirement] = []
    providers: Dict[str, Requirement] = {}
    checksums = _declared_checksums(ctx)

    # Requirements belong to a run, not to a repository. Anything the entrypoint
    # cannot reach is reported separately instead of blocking it.
    entry = entry_file(ctx)
    reached = reachable(ctx, entry) if entry else set()
    elsewhere_needed: List[Tuple[str, str]] = []

    for cand in candidates:
        abs_path = ctx.abspath(cand.path)
        if os.path.exists(abs_path):
            broken = _integrity_problem(ctx, cand, abs_path, checksums, downloaders)
            if broken is not None:
                report.add(broken)
                continue
            present.append((cand, path_size(abs_path)))
            continue

        scope = out_of_scope(ctx, cand.source, reached, entry) if not cand.observed else None
        if scope:
            elsewhere_needed.append((cand.path, scope))
            continue

        elsewhere = ctx.find_basename(os.path.basename(cand.path))
        if elsewhere:
            report.add(
                Requirement(
                    kind=Kind.ASSET,
                    name=cand.path,
                    status=Status.MISMATCH,
                    detail=f"not at the expected path, but present at {elsewhere[0]}",
                    source=cand.source,
                    manual=f"Move or symlink it to {cand.path}.",
                )
            )
            continue

        missing.append(_classify_missing(ctx, cand, downloaders, providers))

    _report_present(ctx, report, present)
    report.extend(missing)
    _report_out_of_scope(ctx, report, elsewhere_needed, entry)
    _report_external(ctx, report, downloaders, providers)


def _report_out_of_scope(
    ctx: RepoContext, report: Report, items: List[Tuple[str, str]], entry: Optional[str]
) -> None:
    """Absent, but for a different job — training, evaluation, another script.

    Reported rather than hidden: if you came here to train, these are exactly
    the requirements you want, and `syp audit --entry train.py` will promote
    them.
    """
    if not items:
        return
    shown = ", ".join(path for path, _ in items[:4])
    unattributed = entry is None
    report.add(
        Requirement(
            kind=Kind.ASSET,
            name=f"referenced files not on disk ({len(items)})"
            if unattributed
            else f"assets for other entrypoints ({len(items)})",
            status=Status.INFO,
            detail=(
                f"no entrypoint identified, so these are an inventory, not requirements: {shown}"
                if unattributed
                else f"absent, and not reachable from {entry}: {shown}"
            )
            + ("..." if len(items) > 4 else ""),
            explain=(
                "This looks like a library rather than an application: nothing here is "
                "runnable, so these paths are things the code can fetch on demand. "
                "Pass --entry <script> to audit a particular run."
                if unattributed
                else "Training and evaluation data, or files another script uses. "
                "Re-run with --entry <script> to audit that run instead."
            ),
            meta={
                "packages": [f"{path}  ({why})" for path, why in items],
                "verbose_list": True,
                "out_of_scope": True,
            },
        )
    )


# --- discovery --------------------------------------------------------------


_TEST_PATH = re.compile(r"(^|/)(tests?|testing)/|(^|/)(test_[^/]*|conftest|fixtures)\.py$", re.IGNORECASE)
# CI declares what the *project's* pipeline downloads across every matrix entry.
# It is never a statement about what your checkout needs to run.
_CI_PATH = re.compile(
    r"^\.github/|^\.gitlab-ci\.ya?ml$|^\.circleci/|^\.buildkite/|^azure-pipelines|^\.travis\.ya?ml$|"
    r"^\.pre-commit-config\.ya?ml$",
    re.IGNORECASE,
)


_GENERATED_PATH = re.compile(r"(^|/)(autogen|_generated|generated|gen)/", re.IGNORECASE)
_GENERATED_HEADER = re.compile(r"(auto[- ]?generated|automatically generated|generated by|do not edit)",
                               re.IGNORECASE)


def is_test_file(rel: str) -> bool:
    """Test fixtures reference paths the demo never needs. Ignore them."""
    return bool(_TEST_PATH.search(rel)) or bool(_CI_PATH.match(rel))


def is_generated(ctx: RepoContext, rel: str) -> bool:
    """Generated bindings enumerate everything a driver could ever load.

    tinygrad's autogenerated AMD firmware table alone produced 104 findings —
    a catalogue of hardware blobs, not a list of things your checkout is missing.
    """
    if _GENERATED_PATH.search(rel):
        return True
    return bool(_GENERATED_HEADER.search(ctx.text(rel)[:2048]))


_OUTPUT_VAR = re.compile(
    r"(?:os\.path\.join|osp\.join|Path)\s*\(\s*[\w.]*(out|output|save|result|dst|target|log)[\w.]*\s*,",
    re.IGNORECASE,
)


def _written_here(text: str) -> Set[str]:
    """Basenames this file writes somewhere, so reads of them are not inputs.

    The pattern that matters is cache-and-reuse: demo.py saves
    `tracking_results.pth` on the first run and loads it on the next. Looking at
    one line in isolation sees only the load and calls it a missing dependency.
    """
    written: Set[str] = set()
    for line in text.splitlines():
        if not (_OUTPUT_HINT.search(line) or _OUTPUT_VAR.search(line)):
            continue
        for match in _QUOTED.finditer(line):
            written.add(os.path.basename(match.group(1)))
        for match in _BARE.finditer(line):
            written.add(os.path.basename(match.group(1)))
    return written


def _find_candidates(ctx: RepoContext) -> List[Candidate]:
    seen: Set[str] = set()
    out: List[Candidate] = []
    for rel in ctx.text_files(SCAN_SUFFIXES):
        if is_test_file(rel) or is_generated(ctx, rel):
            continue
        # Python is parsed, not pattern-matched: see syp.pyscan.
        if rel.lower().endswith(".py"):
            out.extend(_python_candidates(ctx, rel, seen))
            continue
        is_code = rel.lower().endswith(CODE_SUFFIXES)
        text = ctx.text(rel)
        written = _written_here(text) if is_code else set()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if len(line) > 2000:
                continue
            if _OUTPUT_HINT.search(line):
                continue
            # A URL's tail looks exactly like a relative path; blank them out first.
            line = _URL.sub(" ", line)
            found = [m.group(1) for m in _QUOTED.finditer(line)]
            if is_code or rel.lower().endswith((".md", ".txt")):
                found += [m.group(1) for m in _BARE.finditer(line)]
            for raw in found:
                path = _normalize(raw)
                if not path or path in seen or not _plausible(path):
                    continue
                if os.path.basename(path) in written:
                    continue
                seen.add(path)
                out.append(
                    Candidate(
                        path=path,
                        source=f"{rel}:{lineno}",
                        is_demo=bool(re.search(r"(demo|example|sample)", path, re.IGNORECASE)),
                    )
                )
    out.extend(_makefile_candidates(ctx, seen))
    out.extend(_gated_directories(ctx, seen))
    return _drop_bare_duplicates(out)


def _makefile_candidates(ctx: RepoContext, seen: Set[str]) -> List[Candidate]:
    """Files a Makefile recipe names, with its variables expanded.

    `wget $(URL) -O $(CKPT_DIR)/model.pth` only names a path once CKPT_DIR is
    substituted, which is why the recipe text is expanded before scanning.
    """
    out: List[Candidate] = []
    for mk in repo_makefiles(ctx):
        for target in mk.targets.values():
            if target.is_maintenance:
                continue
            text = _URL.sub(" ", mk.expand(target.body))
            for match in list(_QUOTED.finditer(text)) + list(_BARE.finditer(text)):
                path = _normalize(match.group(1))
                if not path or path in seen or not _plausible(path):
                    continue
                seen.add(path)
                out.append(Candidate(path=path, source=f"{mk.path}:{target.lineno}"))
    return out


def _observed_candidates(ctx: RepoContext, known: List[Candidate]) -> List[Candidate]:
    """Paths a traced run actually tried to open. No inference involved."""
    trace = ctx.trace
    if trace is None:
        return []
    seen = {c.path for c in known}
    out = []
    for path in getattr(trace, "missing", []):
        if path in seen or not _plausible(path):
            continue
        seen.add(path)
        out.append(Candidate(path=path, source="observed at runtime", observed=True))
    return out


def _declared_checksums(ctx: RepoContext):
    texts = {
        rel: ctx.text(rel)
        for rel in ctx.text_files((".txt", ".md", ".sh", ".sha256", ".md5"))
        if "sum" in rel.lower() or "checksum" in ctx.text(rel).lower()[:4000]
    }
    return declared_checksums(texts) if texts else {}


def _integrity_problem(
    ctx: RepoContext,
    cand: Candidate,
    abs_path: str,
    checksums,
    downloaders: List[Downloader],
) -> Optional[Requirement]:
    """A present file that is not what it claims to be is worse than a missing one."""
    if not cand.path.lower().endswith(MODEL_EXTENSIONS + ARCHIVE_EXTENSIONS):
        return None
    verdict = inspect_file(abs_path, cand.path)
    basename = os.path.basename(cand.path)
    expected = checksums.get(basename)

    if verdict.ok and expected:
        matched = verify_checksum(abs_path, expected[0])
        if matched is False:
            verdict = type(verdict)(
                False,
                "checksum does not match the value the repo publishes",
                f"Expected {expected[0][:16]}... per {expected[1]}.",
            )
    if verdict.ok:
        return None

    script = _fetching_script(downloaders, basename, os.path.dirname(cand.path), cand.path)
    runnable = script is not None and script.source.endswith((".sh", ".bash", ".py"))
    command = _invocation(script.source) if runnable else None
    return Requirement(
        kind=Kind.ASSET,
        name=cand.path,
        status=Status.STALE,
        detail=f"present but {verdict.problem}",
        source=cand.source,
        fix=command,
        manual=None if command else "Delete the file and fetch it again from a working source.",
        explain=verdict.explain or None,
        meta={"corrupt": True},
    )


def _python_candidates(ctx: RepoContext, rel: str, seen: Set[str]) -> List[Candidate]:
    """Paths named by a Python file, taken from its syntax tree."""
    tree = ctx.parse(rel)
    if tree is None:
        return _regex_candidates(ctx, rel, seen)  # py2, or newer syntax than us
    found = pyscan.scan_tree(tree)

    # A file that writes a name is not asking for it, wherever else it reads it:
    # the cache-and-reuse pattern reads back what it saved on an earlier run.
    written = {os.path.basename(f.path) for f in found if f.is_output}
    out: List[Candidate] = []
    for item in found:
        if item.is_output:
            continue
        path = _normalize(item.path)
        if not path or path in seen or not _plausible(path):
            continue
        if os.path.basename(path) in written:
            continue
        seen.add(path)
        out.append(
            Candidate(
                path=path,
                source=f"{rel}:{item.lineno}",
                is_demo=bool(re.search(r"(demo|example|sample)", path, re.IGNORECASE)),
            )
        )
    return out


def _regex_candidates(ctx: RepoContext, rel: str, seen: Set[str]) -> List[Candidate]:
    """Fallback for Python that will not parse (py2 syntax, templates)."""
    out: List[Candidate] = []
    text = ctx.text(rel)
    written = _written_here(text)
    for lineno, line in enumerate(text.splitlines(), start=1):
        if len(line) > 2000 or _OUTPUT_HINT.search(line):
            continue
        line = _URL.sub(" ", line)
        for match in list(_QUOTED.finditer(line)) + list(_BARE.finditer(line)):
            path = _normalize(match.group(1))
            if not path or path in seen or not _plausible(path):
                continue
            if os.path.basename(path) in written:
                continue
            seen.add(path)
            out.append(Candidate(path=path, source=f"{rel}:{lineno}"))
    return out


def _drop_bare_duplicates(candidates: List[Candidate]) -> List[Candidate]:
    """`model.pth` and `checkpoints/model.pth` are one file. Keep the located one."""
    located = {os.path.basename(c.path) for c in candidates if "/" in c.path}
    return [c for c in candidates if "/" in c.path or c.path not in located]


_SYSTEM_ROOT = re.compile(r"^(sys|dev|proc|etc|opt|usr|var|tmp|bin|sbin|run|home|Users|mnt|media)/")


def _normalize(raw: str) -> Optional[str]:
    raw = raw.strip()
    if raw.startswith("-"):
        return None  # `--run_smplify` is a command-line flag, not a file
    # An absolute path is the machine's, not the repository's. Stripping the
    # leading slash used to turn /dev/kfd into a missing file called dev/kfd.
    if raw.startswith(("/", "\\\\")) or re.match(r"^[A-Za-z]:[\/]", raw):
        return None
    path = raw.replace("\\", "/").lstrip("./")
    if _SYSTEM_ROOT.match(path):
        return None
    if not path or len(path) > 200:
        return None
    return path


# Documentation writes `path/to/model.pt` to mean "your file here".
_PLACEHOLDER_PATH = re.compile(
    r"(^|/)(path|paths?_to|your|my|some|foo|bar|xxx|dir|folder)[/_]|/(to)/|^\.{3}|<|\{",
    re.IGNORECASE,
)


def _plausible(path: str) -> bool:
    if _REJECT_CHARS.search(path):
        return False
    # Prose sentences that happen to end in a filename are not paths.
    if re.search(r"[\s,;]", path):
        return False
    if _PLACEHOLDER_PATH.search(path):
        return False
    if "://" in path:
        return False
    # `MODEL.WEIGHTS` is a config key, not a file: real filenames are not
    # SHOUTED, and `.WEIGHTS` only matched because the scan is case-insensitive.
    extension = os.path.splitext(path)[1]
    if extension and extension != extension.lower():
        return False
    if path.startswith(("http", "ftp", "s3:", "~")):
        return False
    if any(part in ("..",) for part in path.split("/")):
        return False
    lowered = path.lower()
    # Archives are usually intermediates a setup script deletes after unpacking —
    # but `.pth.tar` is a checkpoint, so model extensions win the tie.
    if lowered.endswith(ARCHIVE_EXTENSIONS) and not lowered.endswith(MODEL_EXTENSIONS):
        return False
    if re.search(r"^(output|outputs|results?|logs?|runs?|tmp|temp|cache|wandb)/", lowered):
        return False
    # A path must name a file of a kind we understand. The one exception is a
    # model *directory* the licence registry recognises, and it has to look like
    # a directory: bare words such as `3dpw`, `AMASS` and `--run_smplify` match
    # those patterns too, and none of them is a path.
    if not lowered.endswith(MODEL_EXTENSIONS + MEDIA_EXTENSIONS):
        return "/" in path and bool(match_gated(path + "/"))
    return True


def _gated_directories(ctx: RepoContext, seen: Set[str]) -> List[Candidate]:
    """Catch model *directories* (e.g. `dataset/body_models/smpl`) that the
    licence registry recognises even though no filename is spelled out."""
    out: List[Candidate] = []
    quoted_dir = re.compile(r"""['"]((?:[\w.-]+/){1,6}[\w.-]+)/?['"]""")
    for rel in ctx.text_files(CODE_SUFFIXES):
        if is_test_file(rel):
            continue
        for lineno, line in enumerate(ctx.text(rel).splitlines(), start=1):
            if _OUTPUT_HINT.search(line):
                continue
            for match in quoted_dir.finditer(line):
                path = _normalize(match.group(1))
                if not path or path in seen or "." in os.path.basename(path):
                    continue
                if not match_gated(path + "/"):
                    continue
                seen.add(path)
                out.append(Candidate(path=path, source=f"{rel}:{lineno}"))
    return out


def _find_downloaders(ctx: RepoContext) -> List[Downloader]:
    out: List[Downloader] = []
    for mk in repo_makefiles(ctx):
        fetchers = [t for t in mk.targets.values() if t.fetches and not t.is_maintenance]
        for target in fetchers:
            out.append(
                Downloader(
                    source=f"{mk.path}::{target.name}",
                    text=mk.expand(target.body),
                    urls=dedupe(_URL.findall(mk.expand(target.body))),
                )
            )
    for rel in ctx.text_files((".sh", ".bash", ".md", ".rst", ".txt", ".py", ".yml", ".yaml")):
        if is_test_file(rel):
            continue
        text = ctx.text(rel)
        if not _DOWNLOAD_CMDS.search(text) and "http" not in text:
            continue
        urls = dedupe(
            u.rstrip(".,;)")
            for u in _URL.findall(text)
            if not re.search(r"(shields\.io|badge|arxiv\.org/abs|youtube|\.png|\.jpg|\.svg|license)", u, re.I)
        )
        if not urls and not _DOWNLOAD_CMDS.search(text):
            continue
        out.append(Downloader(source=rel, text=text, urls=urls))
    return out


# --- classification ---------------------------------------------------------


def _classify_missing(
    ctx: RepoContext,
    cand: Candidate,
    downloaders: List[Downloader],
    providers: Dict[str, Requirement],
) -> Requirement:
    basename = os.path.basename(cand.path)
    parent = os.path.dirname(cand.path)
    gated = match_gated(cand.path)

    if gated:
        providers.setdefault(
            gated.key,
            Requirement(
                kind=Kind.EXTERNAL,
                name=f"{gated.key.upper()} account — {gated.provider}",
                status=Status.BLOCKED,
                detail=gated.requires,
                source=gated.url,
                manual=f"Register at {gated.url}, accept the licence, download manually.",
                explain=gated.note or None,
                meta={"provider": gated.key},
            ),
        )
        return Requirement(
            kind=Kind.ASSET,
            name=cand.path,
            status=Status.BLOCKED,
            detail=f"licence-gated ({gated.provider})",
            source=cand.source,
            # The path belongs in `detail`, not here: one licence gate covering
            # six body-model files is one action, and embedding the path in the
            # instruction would split it into six.
            manual=f"Download from {gated.url} after accepting the licence, "
            "then place the files where the repo expects them.",
            explain=gated.note or gated.requires,
            meta={"gated": gated.key, "url": gated.url},
        )

    script = _fetching_script(downloaders, basename, parent, cand.path)
    if script is not None:
        # Prose is not a command: never hand `syp fix` something to run that is
        # actually a paragraph of instructions.
        runnable = script.source.endswith((".sh", ".bash", ".py")) or "::" in script.source
        command = _invocation(script.source) if runnable else None
        return Requirement(
            kind=Kind.ASSET,
            name=cand.path,
            status=Status.MISSING,
            detail=f"fetched by {_describe_source(script.source)}",
            source=cand.source,
            fix=command,
            manual=None
            if command
            else (
                f"Run {_describe_source(script.source)} — it is a POSIX shell script, so it needs bash "
                "(Git Bash or WSL on Windows)."
                if runnable
                else f"Follow the download instructions in {script.source}."
            ),
            meta={"script": script.source},
        )

    return Requirement(
        kind=Kind.ASSET,
        name=cand.path,
        status=Status.MISSING,
        detail="the program opened this at runtime and it was not there"
        if cand.observed
        else "referenced by the code, not present, and no setup script fetches it",
        source=cand.source,
        manual="Find out where this file comes from — the README or the paper is the usual answer.",
    )


def _fetching_script(
    downloaders: List[Downloader], basename: str, parent: str, full: str
) -> Optional[Downloader]:
    """The script that fetches this exact file, or None.

    Matching on the containing directory alone is too generous — every
    checkpoint lives in ``checkpoints/`` — so a directory only counts when it is
    specific (nested) and the script does not merely create it in passing.
    """
    stem = os.path.splitext(basename)[0]
    specific_parent = parent if parent.count("/") >= 1 else ""
    for dl in downloaders:
        if "::" not in dl.source and (
            not dl.source.endswith((".sh", ".bash", ".py")) or not _is_setup_script(dl)
        ):
            continue
        if dl.mentions(full, basename, stem) or (specific_parent and dl.mentions(specific_parent)):
            return dl
    for dl in downloaders:  # second pass: prose counts, but only as a weaker signal
        if dl.mentions(full, basename):
            return dl
    return None


_SETUP_NAME = re.compile(r"(fetch|download|prepare|setup|install|get_|bootstrap)", re.IGNORECASE)


def _is_setup_script(dl: Downloader) -> bool:
    """Is this something you can run, or a module that merely names a path?

    `lib/eval/eval_utils.py` mentions a checkpoint and imports urllib; running it
    to obtain that checkpoint would be nonsense. A shell script is fair game; a
    Python file has to look like a script and be named like one.
    """
    if dl.source.endswith((".sh", ".bash")):
        return True
    if not dl.source.endswith(".py"):
        return False
    runnable = "__main__" in dl.text or dl.source.split("/")[0] in ("scripts", "tools")
    return runnable and bool(_SETUP_NAME.search(dl.source))


def _describe_source(source: str) -> str:
    """`Makefile::data` is internal bookkeeping; say `make data`."""
    if "::" in source:
        path, _, target = source.partition("::")
        return f"`make {target}`" + (f" in {path}" if path.lower() != "makefile" else "")
    return source


def _invocation(script: str) -> Optional[str]:
    if "::" in script:  # a Makefile target
        path, _, target = script.partition("::")
        return f"make -f {path} {target}" if path.lower() != "makefile" else f"make {target}"
    """A command, or None when this platform cannot run the script.

    Offering `bash fetch_demo_data.sh` on a Windows box with no bash produces a
    confusing failure instead of a useful instruction.
    """
    if script.endswith((".sh", ".bash")):
        return f"bash {script}" if which("bash") else None
    if script.endswith(".py"):
        return f"python {script}"
    return None


# --- reporting --------------------------------------------------------------


def _report_present(ctx: RepoContext, report: Report, present: List[Tuple[Candidate, int]]) -> None:
    if not present:
        return
    if len(present) <= 8:
        for cand, size in present:
            report.add(
                Requirement(
                    kind=Kind.ASSET,
                    name=cand.path,
                    status=Status.OK,
                    detail=human_size(size),
                    source=cand.source,
                )
            )
        return
    total = sum(size for _, size in present)
    report.add(
        Requirement(
            kind=Kind.ASSET,
            name=f"referenced assets present ({len(present)})",
            status=Status.OK,
            detail=f"{human_size(total)} on disk",
            meta={"packages": [c.path for c, _ in present], "verbose_list": True},
        )
    )


def _report_external(
    ctx: RepoContext,
    report: Report,
    downloaders: List[Downloader],
    providers: Dict[str, Requirement],
) -> None:
    for req in providers.values():
        report.add(req)

    hosts: Dict[str, Tuple[str, List[str]]] = {}
    for dl in downloaders:
        for url in dl.urls:
            hint = match_host(url)
            if not hint:
                continue
            label, urls = hosts.setdefault(hint.key, (dl.source, []))
            urls.append(url)

    for key, (source, urls) in hosts.items():
        hint = next(h for h in _all_hints() if h.key == key)
        report.add(
            Requirement(
                kind=Kind.EXTERNAL,
                name=f"{hint.label} ({len(dedupe(urls))} url(s))",
                status=Status.INFO if hint.reliable else Status.MISMATCH,
                detail=hint.note,
                source=source,
                meta={"urls": dedupe(urls)[:20], "verbose_urls": True},
            )
        )

    if ctx.network:
        _check_urls(report, downloaders)


def _all_hints():
    from ..knowledge import HOST_HINTS

    return HOST_HINTS


def _check_urls(report: Report, downloaders: List[Downloader]) -> None:
    """HEAD every discovered URL. This is what catches the dead-link case that
    no amount of manifest parsing can."""
    import urllib.error
    import urllib.request

    checked: Set[str] = set()
    for dl in downloaders:
        for url in dl.urls:
            if url in checked or len(checked) >= 40:
                continue
            checked.add(url)
            status, note = _head(urllib, url)
            if status is Status.OK:
                continue
            report.add(
                Requirement(
                    kind=Kind.EXTERNAL,
                    name=url if len(url) < 80 else url[:77] + "...",
                    status=status,
                    detail=note,
                    source=dl.source,
                    manual="Find a current mirror; the repo's issue tracker usually has one."
                    if status is Status.MISSING
                    else None,
                )
            )


def _head(urllib_mod, url: str) -> Tuple[Status, str]:
    request = urllib_mod.request.Request(url, method="HEAD", headers={"User-Agent": "shut-your-pyhole/0.1"})
    try:
        with urllib_mod.request.urlopen(request, timeout=12) as resp:
            return Status.OK, f"HTTP {resp.status}"
    except urllib_mod.error.HTTPError as exc:
        if exc.code in (401, 403):
            return Status.BLOCKED, f"HTTP {exc.code} — needs authentication"
        if exc.code in (404, 410):
            return Status.MISSING, f"HTTP {exc.code} — dead link"
        if exc.code in (405, 501):
            return Status.OK, "HEAD not allowed (host is up)"
        return Status.UNKNOWN, f"HTTP {exc.code}"
    except Exception as exc:  # DNS, TLS, timeouts
        return Status.UNKNOWN, f"unreachable: {exc.__class__.__name__}"
