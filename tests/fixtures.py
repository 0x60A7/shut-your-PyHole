"""A synthetic repository shaped like the ones this tool exists for.

Modelled on WHAM: a submodule, a Docker image documented only in the README,
a fetch script that pulls checkpoints off Google Drive, and a licence-gated body
model that no script can ever fetch.
"""

from __future__ import annotations

import os
import subprocess
from typing import Optional

FILES = {
    ".gitmodules": """\
[submodule "third-party/ViTPose"]
\tpath = third-party/ViTPose
\turl = https://github.com/ViTAE-Transformer/ViTPose.git
[submodule "third-party/DPVO"]
\tpath = third-party/DPVO
\turl = https://github.com/princeton-vl/DPVO.git
""",
    "requirements.txt": """\
# core
numpy>=1.21
torch==1.11.0
torchvision==0.12.0
mmcv-full==1.3.9
chumpy
pytest
--extra-index-url https://download.pytorch.org/whl/cu113
""",
    "Dockerfile": """\
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ARG PY=3.9
RUN apt-get update && apt-get install -y ffmpeg libgl1-mesa-glx git wget
RUN conda create -n wham python=3.9
COPY . /app
""",
    "README.md": """\
# WHAM-ish

## Installation

We recommend the prebuilt image:

```bash
docker pull example/wham-vitpose-dpvo-cuda11.3-python3.9:latest
```

Then fetch the demo data:

```bash
bash fetch_demo_data.sh
```

You also need the SMPL body model, which requires an account.

Download checkpoints/pretrain.pth by hand from https://example.org/files/pretrain.pth
and drop it in place.

## Demo

```bash
python demo.py --video examples/IMG_9732.mov --visualize
```
""",
    "fetch_demo_data.sh": """\
#!/bin/bash
mkdir -p checkpoints
gdown "https://drive.google.com/uc?id=1AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" -O checkpoints/wham_vit_bedlam_w_3dpw.pth.tar
gdown --id 1BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB -O checkpoints/hmr2a.ckpt
wget https://github.com/example/wham/releases/download/v1.0/dpvo.pth -O checkpoints/dpvo.pth
wget https://example.edu/~lab/yolov8x.pt -O checkpoints/yolov8x.pt
""",
    "demo.py": """\
import torch
from lib.models import build_network

CHECKPOINT = 'checkpoints/wham_vit_bedlam_w_3dpw.pth.tar'
DETECTOR = "checkpoints/yolov8x.pt"
POSE = "checkpoints/vitpose-h-multi-coco.pth"


def main(video):
    smpl = SMPL(model_path="dataset/body_models/smpl")
    regressor = torch.load('dataset/body_models/J_regressor_h36m.npy')
    net = build_network(CHECKPOINT)
    out = net(video)
    torch.save(out, 'output/results.pkl')
    return out


if __name__ == '__main__':
    main('examples/IMG_9732.mov')
""",
    "configs/yamls/demo.yaml": """\
MODEL:
  CHECKPOINT: checkpoints/wham_vit_bedlam_w_3dpw.pth.tar
  BACKBONE: checkpoints/hmr2a.ckpt
""",
    "lib/models/__init__.py": "def build_network(path):\n    return path\n",
    # Test data is not a runtime requirement; the scanner must ignore this file.
    "tests/test_models.py": "FIXTURE = 'checkpoints/only_in_tests.pth'\n",
    # Present on disk: one checkpoint and the demo clip.
    "checkpoints/dpvo.pth": "x" * 2048,
    "examples/IMG_9732.mov": "x" * 4096,
}

EMPTY_DIRS = ["third-party/ViTPose", "third-party/DPVO", "output"]


def build(root: str, git: bool = False) -> str:
    for rel, content in FILES.items():
        path = os.path.join(root, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
    for rel in EMPTY_DIRS:
        os.makedirs(os.path.join(root, rel.replace("/", os.sep)), exist_ok=True)
    if git:
        _git_init(root)
    return root


def _git_init(root: str) -> None:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    for cmd in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
    ):
        subprocess.run(cmd, cwd=root, env=env, capture_output=True)


def have_git() -> bool:
    try:
        return subprocess.run(["git", "--version"], capture_output=True).returncode == 0
    except OSError:
        return False


if __name__ == "__main__":  # `python tests/fixtures.py <dir>` for manual poking
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "fixture-repo"
    os.makedirs(target, exist_ok=True)
    print(build(target, git="--git" in sys.argv))
