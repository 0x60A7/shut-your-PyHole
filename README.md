# shut-your-pyhole

Repository readiness audit for research code. Because `pip install -r requirements.txt` was never the whole story.

Point it at a repo. It tells you everything that repo needs in order to run its
advertised demo, what is already satisfied, what it can fix for you, and what
requires a human with an account.

```bash
syp audit .
```

```
SHUT-YOUR-PYHOLE
repository readiness audit  /home/you/WHAM
──────────────────────────────────────────────────────────────────────────────
System
  ✓ docker                     daemon 27.1.1
  ✓ nvidia container runtime   registered with docker
  ✓ nvidia gpu                 NVIDIA GeForce RTX 3060, 12288 MiB, 596.36
Git
  ✓ git repository             main @ 1f4c2ab
  ✗ third-party/ViTPose        submodule not initialized
Container
  ✓ Dockerfile                 nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 (cuda 11.3)
  ✗ yusun9/wham-...:latest     not pulled locally
Python
  ✓ python version             declared ==3.9 by Dockerfile:6, found 3.9.18 (.venv/)
  ✓ declared packages          38/41 installed
  ✗ mmcv-full ==1.3.9          not installed (not a plain pip install)
Runtime assets
  ✓ checkpoints/dpvo.pth       119.4MB
  ✗ checkpoints/hmr2a.ckpt     fetched by fetch_demo_data.sh
  ⚠ dataset/body_models/smpl   licence-gated (Max Planck Institute)
External access
  ⚠ SMPL account               Free account + licence acceptance, then a manual download
  ⚠ Google Drive (4 url(s))    throttles popular files and serves HTML instead of the payload
Execution
  ✓ demo.py                    python demo.py --video examples/IMG_9732.mov --visualize

──────────────────────────────────────────────────────────────────────────────
READY  ███████████████████░░░░░  78%

BLOCKERS  6 blocker(s) · 4 fixable automatically · 2 need a human
  1. bash fetch_demo_data.sh
     checkpoints/hmr2a.ckpt, checkpoints/yolov8x.pt, checkpoints/vitpose-h.pth
  2. git submodule update --init --recursive
     third-party/ViTPose
  3. [manual] Register at https://smpl.is.tue.mpg.de/, accept the licence, download manually
     dataset/body_models/smpl
```

Exit code is 1 while blockers remain, so it drops straight into CI.

## Why

Software projects converge on declared dependencies. Research repositories do
not: the real specification is scattered across a README, a Dockerfile, a
`.gitmodules`, a shell script full of `gdown` calls, a paper, and a licence
agreement on a university web server. You discover it one traceback at a time.

The information is already there. It just needs to be collected, normalised,
cross-referenced and verified — which is a janitorial problem, not an AI one.
This tool is the janitor.

## Commands

| Command | What it does |
| --- | --- |
| `syp audit [path]` | Inventory and verify. Never modifies anything. |
| `syp fix [path]` | Run the commands that resolve blockers. Dry run unless `--yes`. |
| `syp explain <term>` | Everything known about one requirement, including where it was declared. |
| `syp smoke [path]` | Show (or `--run`) the demo command the docs document. |

Useful flags: `--json` (machine-readable), `-v` (sources, fixes, full lists),
`--network` (also check that download URLs still resolve), `--python EXE`
(check packages against a specific interpreter), `--ascii` / `--no-color`.

## What it inspects

**Declared** — `.gitmodules`, `requirements*.txt` (following `-r` includes),
`pyproject.toml` (PEP 621 and poetry), `setup.py`, `setup.cfg`,
`environment.yml`, `Dockerfile*`, `docker-compose.yml`, `.gitattributes`.

**Inferred** — asset paths opened by the code and by config files; download
commands (`wget`, `curl`, `gdown`, `huggingface-cli`, `git clone`) in shell
scripts and README blocks; Docker images named only in prose; the Python
version implied by a base image or a `conda create` line; the demo command in
a fenced README block.

**Verified** — submodules initialised and at the pinned commit; LFS objects
resolved rather than left as pointers; packages installed in the target
interpreter at compatible versions; `torch` actually seeing a GPU; Docker
daemon reachable and the NVIDIA runtime registered; images pulled; every
referenced asset present on disk; with `--network`, every download URL still
resolving.

**Cross-referenced** — this is the part that earns its keep. A missing file is
reported as *fetched by `fetch_demo_data.sh`*, or *licence-gated at
smpl.is.tue.mpg.de*, or *referenced by the code and fetched by nothing* —
three findings that demand completely different responses.

## Design

90% deterministic, by construction. Three tiers, and the tool is honest about
which one it is in:

1. **Trivial** — manifests map to packages, submodules, base images.
2. **Inferable** — code and setup scripts imply files and their provenance.
3. **Semantic** — that `SMPL_NEUTRAL.pkl` needs an account and a signed licence
   is not derivable from the repository. It lives in
   [`knowledge.py`](src/syp/knowledge.py): a small, hand-maintained registry of
   licence gates, flaky hosts, and packages that a plain `pip install` cannot
   satisfy. Every entry surfaces with its source so you can check it.

The goal is not omniscience. It is to reduce a repository to the smallest
possible set of human interventions, *before* you start executing things.

```
              Repository
                  │
     ┌────────────┼────────────┐
     ↓            ↓            ↓
  manifests     source     docs/scripts
     └────────────┼────────────┘
                  ↓
          dependency graph
                  ↓
            verification
                  ↓
       ┌──────────┴──────────┐
       ↓                     ↓
   resolvable            human blockers
       ↓                     ↓
    syp fix              syp explain
       └──────────┬──────────┘
                  ↓
              syp smoke
```

## Install

```bash
uv tool install shut-your-pyhole
```

Or from a checkout:

```bash
uv pip install -e .
```

Standard library only. Python 3.9+. `tomli` is used for `pyproject.toml` on
Python 3.10 and older; without it that one parser falls back to a regex.

## Status

Alpha. The collectors are tested against a synthetic repository modelled on
WHAM (`tests/fixtures.py`) — submodules, a README-only Docker image, a `gdown`
fetch script, and a licence-gated body model. Run `python tests/fixtures.py
/tmp/fixture --git` to generate it and audit it yourself.

Known limits: the asset scanner reads string literals, so paths assembled at
runtime (`os.path.join(cfg.root, name)`) are invisible; the licence registry
covers the 3D human-pose ecosystem well and everything else not at all;
version comparison is PEP 440-ish rather than exact.

## Licence

MIT.
