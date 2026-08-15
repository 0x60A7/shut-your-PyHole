"""Per-repository configuration: `.syp.toml`.

Any checker without a suppression mechanism gets switched off wholesale the
first time it is wrong. This is that mechanism, plus the small set of facts a
repo owner knows and the scanner cannot infer.

    [audit]
    target = "venv"                     # host | venv | image or image:NAME

    [ignore]
    names = ["blender", "colmap"]       # requirement names (glob) to drop
    paths = ["assets/optional/**"]      # asset paths (glob) that are not required

    [assume]
    installed = ["mmcv-full"]           # present despite not being importable metadata
    env = ["WANDB_API_KEY"]             # set elsewhere (CI secret, module system)

    [smoke]
    command = "python demo.py --video examples/clip.mov"
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from typing import List, Optional

from .util import read_text

CONFIG_NAMES = (".syp.toml", "syp.toml", ".config/syp.toml")


@dataclass
class Config:
    target: Optional[str] = None
    ignore_names: List[str] = field(default_factory=list)
    ignore_paths: List[str] = field(default_factory=list)
    assume_installed: List[str] = field(default_factory=list)
    assume_env: List[str] = field(default_factory=list)
    smoke_command: Optional[str] = None
    entry: Optional[str] = None
    source: Optional[str] = None
    error: Optional[str] = None

    def ignores_name(self, name: str) -> bool:
        lowered = name.lower()
        return any(fnmatch.fnmatch(lowered, p.lower()) for p in self.ignore_names)

    def ignores_path(self, path: str) -> bool:
        return any(_path_match(path, p) for p in self.ignore_paths)


def _path_match(path: str, pattern: str) -> bool:
    if fnmatch.fnmatch(path, pattern):
        return True
    # `dir/**` should also match `dir/a/b.txt`, which fnmatch alone does not do.
    if pattern.endswith("/**") and path.startswith(pattern[:-3] + "/"):
        return True
    return False


def load(root: str) -> Config:
    for name in CONFIG_NAMES:
        path = os.path.join(root, name.replace("/", os.sep))
        if os.path.exists(path):
            return _parse(read_text(path), name)
    return Config()


def _parse(text: str, source: str) -> Config:
    data = _load_toml(text)
    if data is None:
        return Config(source=source, error="could not parse (install tomli on Python < 3.11)")
    audit = data.get("audit") or {}
    ignore = data.get("ignore") or {}
    assume = data.get("assume") or {}
    smoke = data.get("smoke") or {}
    return Config(
        target=audit.get("target"),
        ignore_names=list(ignore.get("names") or []),
        ignore_paths=list(ignore.get("paths") or []),
        assume_installed=[str(x).lower() for x in (assume.get("installed") or [])],
        assume_env=list(assume.get("env") or []),
        smoke_command=smoke.get("command"),
        source=source,
    )


def _load_toml(text: str):
    try:
        import tomllib  # type: ignore[import-not-found]

        return tomllib.loads(text)
    except ImportError:
        pass
    except Exception:
        return None
    try:
        import tomli  # type: ignore[import-not-found]

        return tomli.loads(text)
    except Exception:
        return None
