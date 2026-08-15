"""The shared view of a repository that every collector reads from.

Built once, so we walk the tree and read each file at most once no matter how
many collectors want to look at it.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import config as config_mod
from . import target as target_mod
from .util import read_text, walk_files

TEXT_SUFFIXES = (
    ".py", ".sh", ".bash", ".zsh", ".txt", ".md", ".rst", ".cfg", ".toml",
    ".yaml", ".yml", ".json", ".ini", ".ipynb", ".bat", ".ps1", ".mk",
)
TEXT_NAMES = ("Makefile", "makefile", "Dockerfile", "dockerfile", "Justfile")

# Files we read in full for cross-referencing, whatever their extension.
MAX_TEXT_BYTES = 512_000


@dataclass
class RepoContext:
    root: str
    files: List[str] = field(default_factory=list)
    _text_cache: Dict[str, str] = field(default_factory=dict, repr=False)
    network: bool = False
    config: "config_mod.Config" = field(default_factory=config_mod.Config)
    target: "target_mod.Target" = field(default_factory=target_mod.Target)
    trace: Optional["object"] = None
    depth: int = 0

    @classmethod
    def load(
        cls,
        root: str,
        network: bool = False,
        target_spec: Optional[str] = None,
        images: Optional[List[str]] = None,
        depth: int = 0,
    ) -> "RepoContext":
        root = os.path.abspath(root)
        cfg = config_mod.load(root)
        ctx = cls(root=root, files=walk_files(root), network=network, config=cfg, depth=depth)
        spec = target_mod.default_spec(cfg.target, target_spec)
        if images is None and spec.startswith("image") and ":" not in spec:
            # `--target image` means "the image this repo documents"; find it first.
            from .collect.container import documented_images

            images = documented_images(ctx)
        ctx.target = target_mod.resolve(root, spec, images)
        return ctx

    # --- path helpers -------------------------------------------------------

    def abspath(self, rel: str) -> str:
        return os.path.join(self.root, rel.replace("/", os.sep))

    def exists(self, rel: str) -> bool:
        return os.path.exists(self.abspath(rel))

    def isdir(self, rel: str) -> bool:
        return os.path.isdir(self.abspath(rel))

    def glob(self, pattern: str) -> List[str]:
        """Match repo-relative paths. ``*`` does not cross directory boundaries."""
        if "/" in pattern:
            return sorted(f for f in self.files if fnmatch.fnmatch(f, pattern))
        return sorted(
            f for f in self.files if "/" not in f and fnmatch.fnmatch(f, pattern)
        )

    def rglob(self, pattern: str) -> List[str]:
        return sorted(
            f
            for f in self.files
            if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(os.path.basename(f), pattern)
        )

    def find_basename(self, name: str) -> List[str]:
        lowered = name.lower()
        return [f for f in self.files if os.path.basename(f).lower() == lowered]

    # --- content helpers ----------------------------------------------------

    def text(self, rel: str) -> str:
        if rel not in self._text_cache:
            self._text_cache[rel] = read_text(self.abspath(rel), MAX_TEXT_BYTES)
        return self._text_cache[rel]

    def is_textish(self, rel: str) -> bool:
        base = os.path.basename(rel)
        return base.startswith(TEXT_NAMES) or rel.lower().endswith(TEXT_SUFFIXES)

    def text_files(self, suffixes=None) -> List[str]:
        out = []
        for rel in self.files:
            if suffixes is not None:
                if not rel.lower().endswith(tuple(suffixes)):
                    continue
            elif not self.is_textish(rel):
                continue
            out.append(rel)
        return out

    def grep(self, needle: str, suffixes=None) -> List[str]:
        """Case-insensitive substring search over text files. Returns paths."""
        needle = needle.lower()
        hits = []
        for rel in self.text_files(suffixes):
            if needle in self.text(rel).lower():
                hits.append(rel)
        return hits

    def mentions(self, *needles: str) -> bool:
        for rel in self.text_files():
            lowered = self.text(rel).lower()
            if any(n.lower() in lowered for n in needles):
                return True
        return False

    # --- convenience --------------------------------------------------------

    @property
    def readme(self) -> Optional[str]:
        for rel in self.files:
            if "/" not in rel and rel.lower().startswith("readme"):
                return rel
        return None

    def docs_text(self) -> str:
        """README plus docs/ and install notes, concatenated. Used for hints only."""
        parts = []
        for rel in self.files:
            low = rel.lower()
            if (
                low.startswith("docs/")
                or low.startswith("doc/")
                or os.path.basename(low).startswith(("readme", "install", "setup.md", "getting"))
            ) and low.endswith((".md", ".rst", ".txt")):
                parts.append(self.text(rel))
        return "\n".join(parts)

    def line_of(self, rel: str, needle: str) -> Optional[int]:
        """1-indexed line number of the first occurrence of ``needle``."""
        text = self.text(rel)
        idx = text.find(needle)
        if idx < 0:
            return None
        return text.count("\n", 0, idx) + 1

    def source_ref(self, rel: str, needle: Optional[str] = None) -> str:
        if needle:
            line = self.line_of(rel, needle)
            if line:
                return f"{rel}:{line}"
        return rel
