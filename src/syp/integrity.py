"""Is the file that is present actually the file that was wanted?

`knowledge.py` warns that Google Drive serves an HTML interstitial once a
quota is hit. That page saves to disk as `model.pth`, is non-zero, and passes
every existence check — so existence checks are not enough. This reads the
first bytes and says what the file really is.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .util import read_text

# Smallest plausible size for a real model file. Interstitials and error pages
# land far below this; genuinely tiny configs do not use these extensions.
MIN_MODEL_BYTES = 8192

_MAGIC = {
    ".pth": (b"PK\x03\x04", b"\x80"),          # torch.save is a zip (new) or pickle (legacy)
    ".pt": (b"PK\x03\x04", b"\x80"),
    ".pth.tar": (b"PK\x03\x04", b"\x80", b"\x1f\x8b"),
    ".ckpt": (b"PK\x03\x04", b"\x80"),
    ".pkl": (b"\x80", b"(", b"]", b"}"),
    ".npy": (b"\x93NUMPY",),
    ".npz": (b"PK\x03\x04",),
    ".h5": (b"\x89HDF",),
    ".zip": (b"PK\x03\x04",),
    ".safetensors": None,                        # 8-byte little-endian length, then '{'
    ".gguf": (b"GGUF",),
}

_HTML_SNIFF = (b"<!doctype", b"<html", b"<?xml", b"<!DOCTYPE")
_LFS_POINTER = b"version https://git-lfs"
_CHECKSUM_LINE = re.compile(r"^([0-9a-fA-F]{32,64})\s+\*?(\S+)\s*$", re.MULTILINE)


@dataclass
class Verdict:
    ok: bool
    problem: str = ""
    explain: str = ""


def inspect(abs_path: str, rel_path: str) -> Verdict:
    """Cheap structural check: size, HTML sniff, LFS pointer, magic bytes."""
    try:
        size = os.path.getsize(abs_path)
        with open(abs_path, "rb") as fh:
            head = fh.read(64)
    except OSError as exc:
        return Verdict(False, f"unreadable ({exc.__class__.__name__})")

    lowered = head.lower()
    if any(lowered.startswith(marker.lower()) for marker in _HTML_SNIFF):
        return Verdict(
            False,
            f"contains an HTML page, not a model ({size} bytes)",
            "This is the Google Drive quota interstitial (or a login page) saved under the "
            "model's name. Delete it and fetch again from a different mirror.",
        )
    if head.startswith(_LFS_POINTER):
        return Verdict(False, "is an unresolved git-lfs pointer", "Run `git lfs pull`.")

    extension = _extension(rel_path)
    if extension in _MAGIC and size < MIN_MODEL_BYTES:
        return Verdict(
            False,
            f"is only {size} bytes",
            "Far too small for a model file; the download almost certainly failed silently.",
        )

    if extension == ".safetensors":
        if size > 8 and not head[8:9] == b"{":
            return Verdict(False, "does not have a safetensors header")
        return Verdict(True)

    magic = _MAGIC.get(extension)
    if magic and not any(head.startswith(m) for m in magic):
        return Verdict(
            False,
            f"does not look like a {extension} file",
            f"First bytes are {head[:8]!r}, which matches nothing this format starts with.",
        )
    return Verdict(True)


def _extension(path: str) -> str:
    lowered = path.lower()
    if lowered.endswith(".pth.tar"):
        return ".pth.tar"
    return os.path.splitext(lowered)[1]


# --- checksums the repo already published ----------------------------------


def declared_checksums(texts: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    """Parse `<hex>  filename` lines out of the repo's own files.

    Projects that publish checksums have already told us what correct looks
    like; not using that would be silly.
    """
    out: Dict[str, Tuple[str, str]] = {}
    for source, text in texts.items():
        for digest, name in _CHECKSUM_LINE.findall(text):
            base = os.path.basename(name.replace("\\", "/"))
            if base and base not in out:
                out[base] = (digest.lower(), source)
    return out


def verify_checksum(abs_path: str, expected: str) -> Optional[bool]:
    import hashlib

    algorithm = {32: hashlib.md5, 40: hashlib.sha1, 64: hashlib.sha256}.get(len(expected))
    if algorithm is None:
        return None
    digest = algorithm()
    try:
        with open(abs_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest().lower() == expected.lower()
