"""Curated knowledge about assets a scanner can find but cannot interpret.

This is the deliberately small, deliberately human-maintained part of the tool.
Everything else is derived from the repository; these entries encode the facts
that live outside it — that a file is licence-gated, that a host rate-limits,
that a package will not install from a plain requirements.txt.

Entries are hints, not gospel: URLs and registration flows change. Each one is
surfaced to the user with its source so they can check it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class GatedAsset:
    key: str
    patterns: List[str]          # regexes matched against the asset path
    provider: str
    url: str
    requires: str                # what the human has to do
    note: str = ""


GATED_ASSETS: List[GatedAsset] = [
    GatedAsset(
        key="smpl",
        patterns=[
            r"SMPL_(NEUTRAL|MALE|FEMALE)\.pkl",
            r"basicmodel_[mfn].*\.pkl",
            r"basicModel_[mfn].*\.pkl",
            r"body_models?/smpl(/|$)",
            r"SMPL_python_v",
        ],
        provider="Max Planck Institute for Intelligent Systems",
        url="https://smpl.is.tue.mpg.de/",
        requires="Free account + acceptance of a non-commercial licence, then a manual download.",
        note=(
            "The body model is distributed per-user; there is no unauthenticated URL. "
            "Many repos also expect the files renamed (e.g. basicmodel_neutral_*.pkl -> "
            "SMPL_NEUTRAL.pkl) — check the repo's own instructions for the layout."
        ),
    ),
    GatedAsset(
        key="smplx",
        patterns=[r"SMPLX_(NEUTRAL|MALE|FEMALE)\.(npz|pkl)", r"body_models?/smplx(/|$)", r"smplx_npz"],
        provider="Max Planck Institute for Intelligent Systems",
        url="https://smpl-x.is.tue.mpg.de/",
        requires="Free account + licence acceptance, then a manual download.",
    ),
    GatedAsset(
        key="mano",
        patterns=[r"MANO_(LEFT|RIGHT)\.pkl", r"body_models?/mano(/|$)"],
        provider="Max Planck Institute for Intelligent Systems",
        url="https://mano.is.tue.mpg.de/",
        requires="Free account + licence acceptance, then a manual download.",
    ),
    GatedAsset(
        key="flame",
        patterns=[r"FLAME[_-].*\.(pkl|npz)", r"body_models?/flame(/|$)"],
        provider="Max Planck Institute for Intelligent Systems",
        url="https://flame.is.tue.mpg.de/",
        requires="Free account + licence acceptance, then a manual download.",
    ),
    GatedAsset(
        key="smplify",
        patterns=[r"smplify", r"J_regressor_extra", r"neutral_smpl_mean_params", r"smpl_mean_params"],
        provider="SMPLify / SMPL ecosystem",
        url="https://smplify.is.tue.mpg.de/",
        requires="Account on the SMPLify site; some auxiliary files ship only inside its archive.",
    ),
    GatedAsset(
        key="amass",
        # Dataset gates are anchored to directory components: a checkpoint merely
        # *trained on* 3DPW is not itself a gated download.
        patterns=[r"(^|/)amass(/|$)"],
        provider="AMASS (MPI)",
        url="https://amass.is.tue.mpg.de/",
        requires="Account + per-dataset licence acceptance; downloads are per sub-dataset.",
    ),
    GatedAsset(
        key="h36m",
        patterns=[r"(^|/)(human3\.?6m|h36m)(/|$)"],
        provider="Human3.6M",
        url="http://vision.imar.ro/human3.6m/",
        requires="Academic account request, manually approved; redistribution is prohibited.",
        note="Approval is not instant and the site is intermittently offline.",
    ),
    GatedAsset(
        key="3dpw",
        patterns=[r"(^|/)(3dpw|threedpw)(/|$)"],
        provider="3D Poses in the Wild (MPI)",
        url="https://virtualhumans.mpi-inf.mpg.de/3DPW/",
        requires="Registration form; download link is emailed.",
    ),
    GatedAsset(
        key="agora",
        patterns=[r"(^|/)agora(/|$)"],
        provider="AGORA (MPI)",
        url="https://agora.is.tue.mpg.de/",
        requires="Account + licence acceptance.",
    ),
]


@dataclass(frozen=True)
class HostHint:
    key: str
    patterns: List[str]          # regexes matched against a URL
    label: str
    note: str
    reliable: bool = True


HOST_HINTS: List[HostHint] = [
    HostHint(
        key="gdrive",
        patterns=[r"drive\.google\.com", r"docs\.google\.com/uc", r"\bgdown\b"],
        label="Google Drive",
        note=(
            "Google Drive throttles popular files and serves an HTML interstitial "
            "instead of the payload once a quota is hit. A 'downloaded' file that is "
            "a few KB of HTML is the classic symptom."
        ),
        reliable=False,
    ),
    HostHint(
        key="hf",
        patterns=[r"huggingface\.co"],
        label="Hugging Face",
        note="Some repos are gated: you must accept terms on the model page and use a token.",
    ),
    HostHint(
        key="mpg",
        patterns=[r"\.is\.tue\.mpg\.de", r"mpi-inf\.mpg\.de"],
        label="MPI download portal",
        note="Requires an authenticated session; plain wget/curl will fetch a login page.",
        reliable=False,
    ),
    HostHint(
        key="onedrive",
        patterns=[r"1drv\.ms", r"onedrive\.live\.com", r"sharepoint\.com"],
        label="OneDrive",
        note="Share links expire and often need a browser round-trip to resolve.",
        reliable=False,
    ),
    HostHint(
        key="dropbox",
        patterns=[r"dropbox\.com"],
        label="Dropbox",
        note="Links break when the owner's account changes; add ?dl=1 for direct download.",
        reliable=False,
    ),
    HostHint(
        key="gh_release",
        patterns=[r"github\.com/[^/]+/[^/]+/releases", r"objects\.githubusercontent\.com"],
        label="GitHub release",
        note="Stable and unauthenticated unless the repo is private.",
    ),
    HostHint(
        key="university",
        patterns=[r"\.edu/", r"\.ac\.uk/", r"\.ethz\.ch/", r"\.mpg\.de/"],
        label="University web host",
        note="Lab pages rot; if this 404s the file has usually moved to a mirror.",
        reliable=False,
    ),
]


@dataclass(frozen=True)
class AwkwardPackage:
    name: str
    note: str
    hint: str = ""


# Packages that a plain `pip install -r requirements.txt` routinely fails on.
AWKWARD_PACKAGES = {
    p.name: p
    for p in [
        AwkwardPackage(
            "mmcv-full",
            "Needs a wheel built against your exact torch + CUDA pair.",
            "Install from the openmmlab index matching torch/CUDA, not from PyPI.",
        ),
        AwkwardPackage("mmcv", "Version must match the mmengine/mmdet/mmpose generation in use."),
        AwkwardPackage("mmpose", "Pinned to a specific mmcv major version; mixing generations fails at import."),
        AwkwardPackage("mmdet", "Pinned to a specific mmcv major version."),
        AwkwardPackage("detectron2", "Not on PyPI; installs from a git URL and compiles against torch."),
        AwkwardPackage("pytorch3d", "Compiles CUDA extensions; prebuilt wheels exist only for some torch/CUDA pairs."),
        AwkwardPackage("torch", "The PyPI default may be CPU-only or a different CUDA build than the repo assumes."),
        AwkwardPackage("torchvision", "Version is locked to the torch version; a mismatch fails at import."),
        AwkwardPackage("flash-attn", "Compiles from source; needs nvcc and a lot of RAM."),
        AwkwardPackage("chumpy", "Unmaintained; breaks on numpy>=1.24 (uses removed aliases)."),
        AwkwardPackage("dpvo", "Built from source with CUDA extensions; usually a submodule, not a PyPI package."),
        AwkwardPackage("apex", "NVIDIA Apex builds from source and is version-sensitive."),
        AwkwardPackage("xformers", "Wheel must match the torch build exactly."),
        AwkwardPackage("opencv-python", "Conflicts with opencv-python-headless if both land in one env."),
    ]
}

# Packages whose presence tells us the project wants a GPU.
GPU_PACKAGES = {
    "torch", "torchvision", "torchaudio", "tensorflow", "tensorflow-gpu", "jax",
    "cupy", "onnxruntime-gpu", "mmcv-full", "detectron2", "pytorch3d",
    "flash-attn", "xformers", "bitsandbytes", "vllm", "triton",
}

MODEL_EXTENSIONS = (
    ".pth.tar", ".pth", ".ckpt", ".pt", ".bin", ".safetensors", ".onnx",
    ".pkl", ".npz", ".npy", ".h5", ".pb", ".tflite", ".gguf", ".caffemodel",
    ".weights", ".t7", ".pdparams", ".msgpack",
)

MEDIA_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".wav", ".flac", ".obj", ".ply", ".glb")

ARCHIVE_EXTENSIONS = (".tar.gz", ".tgz", ".zip", ".tar", ".7z", ".tar.xz")


def match_gated(path: str) -> Optional[GatedAsset]:
    for entry in GATED_ASSETS:
        for pattern in entry.patterns:
            if re.search(pattern, path, re.IGNORECASE):
                return entry
    return None


def match_host(url: str) -> Optional[HostHint]:
    for entry in HOST_HINTS:
        for pattern in entry.patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return entry
    return None
