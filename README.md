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
target: docker image yusun9/wham-vitpose-dpvo-cuda11.3-python3.9:latest
──────────────────────────────────────────────────────────────────────────────
System
  ✓ gpu passthrough            `docker run --gpus all` works with this image
  ✓ libGL.so.1 (for cv2)       shared library present
  ✗ ffmpeg                     spawned by the traced run, not on PATH
Git
  ✗ third-party/ViTPose        submodule not initialized
Container
  ✓ Dockerfile                 nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 (cuda 11.3)
  · run flag --shm-size=8g     dataloader workers crash with a bus error on the 64MB default
Python
  ✓ python version             declared ==3.9 by Dockerfile:6, found 3.9.18 (image)
  ✓ torch accelerator          torch 1.11.0 (CUDA 11.3), 1 device(s) visible
  ✗ import yacs                imported by the code but declared in no manifest
  ⚠ torch / torchvision        torch 2.1.0 expects torchvision 0.16.x, found 0.12.0
Build
  ✗ nvcc / torch CUDA          nvcc is 12.9 but torch was built for CUDA 11.3
Environment
  ⚠ HF_TOKEN                   credential is not set
  · PYOPENGL_PLATFORM          read with no default; the code receives None
Runtime assets
  ✗ checkpoints/hmr2a.ckpt     present but contains an HTML page, not a model (155 bytes)
  ⚠ dataset/body_models/smpl   licence-gated (Max Planck Institute)
External access
  ⚠ SMPL account               Free account + licence acceptance, then a manual download
  · Hugging Face hub           fetches the model at run time; gated repos also need HF_TOKEN
Execution
  ✓ demo.py                    python demo.py --video examples/IMG_9732.mov
  · observed run               python demo.py — exit 1

──────────────────────────────────────────────────────────────────────────────
6 blocker(s)  ███████████████████░░░░░  31/40 checks satisfied

BLOCKERS  6 blocker(s) · 2 fixable automatically · 4 need a human
  1. git submodule update --init --recursive
     third-party/ViTPose
  2. bash fetch_demo_data.sh  [runs repo code]
     checkpoints/hmr2a.ckpt, checkpoints/yolov8x.pt
  3. [manual] Register at https://smpl.is.tue.mpg.de/, accept the licence
     dataset/body_models/smpl
```

The blocker count is the number to gate on; exit status is 1 while any remain.
The percentage is a progress bar whose denominator moves as detection improves,
so it is advisory and labelled as such in `--json`.

## Why

Software projects converge on declared dependencies. Research repositories do
not: the real specification is scattered across a README, a Dockerfile, a
`.gitmodules`, a shell script full of `gdown` calls, a paper, and a licence
agreement on a university web server. You discover it one traceback at a time.

The information is already there. It just needs to be collected, normalised,
cross-referenced and verified — which is a janitorial problem.
This tool is the janitor.

## Commands

| Command | What it does |
| --- | --- |
| `syp audit [path]` | Inventory and verify. Never modifies anything. |
| `syp trace [path]` | Run the demo under an audit hook and record what it *actually* needs. |
| `syp fix [path]` | Run the commands that resolve blockers. Dry run unless `--yes`. |
| `syp explain <term>` | Everything known about one requirement, including where it was declared. |
| `syp smoke [path]` | Show (or `--run`) the demo command the docs document. |

Flags: `--target` (see below), `--json`, `-v`, `--network`, `--trace-file`,
`--no-trace`, `--allow-scripts`, `--only <collector>`, `--ascii`, `--no-color`.

## Observation beats inference

Static scanning guesses what a program will open. `syp trace` watches it.

```bash
syp trace .
```

It installs a `sys.addaudithook` hook in the child process via a generated
`sitecustomize.py`, so it covers `python demo.py`, a shell script that calls
python, and subprocesses of either. Every path opened, module imported, binary
spawned and host contacted is recorded to `.syp/trace.jsonl` — right up to the
traceback.

```
observed
  exit code       1
  paths opened    247
  paths missing   3
  modules         180
  subprocesses    ffmpeg
  network         huggingface.co
```

The next `syp audit` folds that in automatically. A path the program opened and
did not find is not a heuristic — it is the failure, named, before you have
fixed anything else. Runtime findings override static ones: "the Docker image
installs ffmpeg" stops being a good enough answer once we have watched the host
program shell out to it.

## Audit the machine you will actually run on

If the README says "use this image", checking your laptop for ffmpeg answers
the wrong question.

```bash
syp audit . --target image           # the image the repo documents
syp audit . --target image:org/name  # a specific one
syp audit . --target venv            # the repo's virtualenv (default)
syp audit . --target host
```

With an image target, probes run *inside* the container: the interpreter, the
installed distributions, `ldconfig` for shared libraries, the image's own `ENV`
block. Filesystem checks stay on the host, since the repo is bind-mounted and
it is the same tree either way.

Choosing an image target also starts a throwaway container with `--gpus all` and
reports whether it worked — which settles the question the host-side runtime
check can only guess at, since Docker Desktop and WSL2 do GPU passthrough
without advertising an nvidia runtime.

## What it inspects

**Declared** — `.gitmodules`, `requirements*.txt` (following `-r` includes),
`pyproject.toml` (PEP 621 and poetry), `setup.py`, `setup.cfg`,
`environment.yml`, `Dockerfile*`, `docker-compose.yml`, `.gitattributes`.

**Inferred** — asset paths opened by code and config; download commands
(`wget`, `curl`, `gdown`, `huggingface-cli`, `git clone`) in scripts and README
blocks; Docker images named only in prose, and the `--shm-size` / `--gpus`
flags on the documented `docker run` line; the Python version implied by a base
image or a `conda create`; every `import` in the source, and the shared
libraries and runtime downloads those imports imply; environment variables the
code reads; compiled extensions and the build steps for them.

**Verified** — submodules initialised and at the pinned commit (and audited
recursively, one level, for their own requirements and build steps); LFS
objects resolved; packages installed in the target at compatible versions;
`pip check` consistency; torch's own view of its accelerator (CUDA, ROCm or
Metal); nvcc matching the CUDA torch was built for; Docker daemon, engine
(docker or podman) and GPU passthrough; every referenced asset present *and
structurally valid*; with `--network`, that download URLs still resolve and
that the declared set resolves at all.

**Cross-referenced** — the part that earns its keep. A missing file is reported
as *fetched by `fetch_demo_data.sh`*, or *licence-gated at
smpl.is.tue.mpg.de*, or *referenced by the code and fetched by nothing* — three
findings that demand completely different responses.

## Present is not the same as correct

A Google Drive quota interstitial saves to disk as `model.pth`, is non-zero,
and passes every existence check. So existence is not the test: files are
checked for size, HTML sniffing, unresolved LFS pointers, and format magic
bytes, plus any checksum the repo itself publishes.

```
✗ checkpoints/hmr2a.ckpt   present but contains an HTML page, not a model (155 bytes)
    note: This is the Google Drive quota interstitial saved under the model's name.
```

## Fixing things is a trust decision

`syp fix` groups blockers by action, so one script that supplies four
checkpoints is one step rather than four. Each command is classified:

- **local** — touches only this checkout (`git submodule update --init`)
- **network** — installs from a known registry (`pip install`, `docker pull`)
- **script** — runs a script belonging to the audited repository

Scripts are withheld by default, because `bash fetch_demo_data.sh` from a repo
you have not read is arbitrary code execution. `--allow-scripts` opts in, and
nothing runs at all without `--yes`.

Package fixes always name the interpreter being audited
(`/path/to/.venv/bin/python -m pip install ...`), never a bare `pip`.

## Configuration

`.syp.toml` in the repo root. Every checker without a suppression mechanism
gets switched off wholesale the first time it is wrong.

```toml
[audit]
target = "image:org/name"

[ignore]
names = ["blender", "colmap"]        # requirement names (glob)
paths = ["assets/optional/**"]       # asset paths that are not really required

[assume]
installed = ["mmcv-full"]            # present despite unhelpful metadata
env = ["WANDB_API_KEY"]              # supplied by CI, not by you

[smoke]
command = "python demo.py --video examples/clip.mov"
```

Suppressed findings are counted and named in the report's notes — silent
filtering is how a report starts lying. This repo ships its own `.syp.toml`,
since a pattern scanner inevitably matches its own patterns.

## Install

```bash
uv tool install shut-your-pyhole
```

Or from a checkout:

```bash
uv pip install -e .
```

Standard library only. Python 3.9+. `tomli` is used for TOML on Python 3.10 and
older; without it those parsers fall back to a regex.

## Status

Alpha. 65 tests run against a synthetic repository modelled on WHAM
(`tests/fixtures.py`): submodules, a README-only Docker image, a `gdown` fetch
script, undeclared imports, a licence-gated body model, a required env var, a
credential, and a checkpoint that is secretly an HTML error page. Generate it
with `python tests/fixtures.py /tmp/fixture --git` and audit it yourself.

Known limits:

- The static asset scanner reads string literals, so paths assembled at runtime
  are invisible to it. That is what `syp trace` is for — but tracing only sees
  the code paths a given run reaches.
- The licence registry covers the 3D human-pose ecosystem well and everything
  else not at all. Contributions to `knowledge.py` are the point of that file.
- Version comparison is PEP 440-ish rather than exact, and the pair rules
  (torch/torchvision, numpy 2 ABI, mmcv generations) are curated heuristics.
- Import → distribution mapping is a lookup table plus a guess.
- `--target image` requires the image to be present locally; it will not pull
  one for you.

## Licence

MIT.
