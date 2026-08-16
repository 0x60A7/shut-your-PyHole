# shut-your-pyhole

Repository readiness audit for research code. Because `pip install -r requirements.txt` was never the whole story.

Point it at a repo. It tells you everything that repo needs in order to run, what is already satisfied, what it can fix for you, and what
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
python, and subprocesses of either. Add `--target image:org/name` to trace
*inside the container*, which for most ML repos is the only environment that
gets far enough to be worth watching — the hook directory and the repository
are bind mounted in and the trace is written back out. Every path opened, module imported, binary
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

## Docker is optional

Nothing requires it. On a repo that never mentions containers there is no
Container section at all, `--target venv` picks up the repo's virtualenv, and
package fixes name that interpreter. Docker checks only appear when the repo
documents a Docker workflow — and if it does but you have neither docker nor
podman, that is reported as a missing requirement of *the project*, not an
error in the tool.

The suite runs green with no container engine present (it is run that way
inside the WHAM image on every check).

What does need it is `--target image`, which is the one feature that
inspects a container. If the engine is missing or the image is not pulled, the
audit says so and stops claiming anything:

```
NOT VERIFIED  docker image org/name:tag could not be inspected

INCONCLUSIVE  the environment under audit could not be inspected; findings below are partial
  image not present locally — run `docker pull org/name:tag` first
```

Exit status is 1. An audit that could not run is never reported as an audit
that found nothing.

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

## Requirements belong to a run, not to a repository

WHAM's `demo.py` needs four checkpoints and a body model. `train.py`
additionally needs AMASS, 3DPW and a stage-one checkpoint no demo will ever
open. Unioning them produced a report where two thirds of the blockers were
irrelevant to what you asked for.

The asset scanner now walks the local import graph from the entrypoint and
attributes each requirement to the run that reaches it. Anything else is
reported, not hidden, in one non-blocking line:

```
· assets for other entrypoints (7)   absent, and not reachable from demo.py:
                                     dataset/parsed_data/amass.pth, shape.npz...
```

`--entry train.py` audits that run instead and promotes them. Files a script
writes and reads back later — the cache-and-reuse pattern — stop being counted
as inputs at all.

On real WHAM this took the report from 33 blockers in 11 groups to 21 in 8.

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

## Measured on a corpus, not on one repo

Nineteen public repositories — mmpose, detectron2, pytorch_geometric,
ultralytics, tinygrad, peft, litgpt, LLaVA, GroundingDINO, whisper, CLIP,
nanoGPT, mamba, DPVO, 4D-Humans, InstantSplat, segment-anything, requests,
flask — audited headless. No crashes, no collector failures, slowest run 9 s.

The first pass reported **1,315 blockers**. Almost all of it was noise from
seven systematic faults, each of which reported one fact many times:

| fault | example |
| --- | --- |
| every declared package listed separately against an empty environment | `requests`: 16 blockers meaning "not installed yet" |
| dev/test extras treated as runtime requirements | `pytest-cov`, `httpbin` |
| no entrypoint meant no scoping, so a library's whole source tree counted | `pytorch_geometric`: 123 asset blockers |
| generated bindings enumerated as requirements | `tinygrad`: 104 AMD firmware blobs from one autogen file |
| CI matrices read as runtime needs | `ultralytics`: every model variant it ships |
| documentation placeholders taken literally | `path/to/model.pt` |
| a docs builder chosen as the entrypoint | `ultralytics` scoped everything to `docs/build_docs.py` |

Reading detectron2's and peft's remaining findings line by line then exposed
four more, including two parsers that read prose as if it were code:

| fault | example |
| --- | --- |
| setup.py parsed by pairing quotes | one apostrophe in a comment (`OS's package manager`) desynchronised every string after it: 4 of detectron2's 14 dependencies parsed, the other 10 reported as undeclared imports |
| env vars pattern-matched over source text | a docstring showing `os.environ["FOO"]  # raises KeyError` became a requirement |
| custom URL schemes | `detectron2://COCO-Detection/.../model.pkl` — only `http://` was being stripped |
| maintainer tooling and nested projects | peft's `scripts/` wanted a `SLACK_API_TOKEN`; its `method_comparison/` app wanted gradio |

Both parsers now use the AST. After all of it: **178 blockers, an 86%
reduction, median 10 per repo, worst case 16**, with no loss of real signal —
WHAM is unchanged at 21, DPVO still reports its Pangolin and DBoW2 submodules
and CUDA build step, 4D-Humans still reports the SMPL licence gate, and peft
still reports the C++ compiler its BOFT kernel is JIT-compiled with.

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

Alpha. 86 tests run against a synthetic repository modelled on WHAM
(`tests/fixtures.py`): submodules, a README-only Docker image, a `gdown` fetch
script, undeclared imports, a licence-gated body model, a required env var, a
credential, and a checkpoint that is secretly an HTML error page. Generate it
with `python tests/fixtures.py /tmp/fixture --git` and audit it yourself.

## Portability

Verified: Windows 11 / Python 3.11 (host) and Linux / Python 3.9 (inside the
WHAM image) — the same 86 tests, green on both — the container run also covers the no-docker case. macOS is reasoned about, not
tested.

| Concern | Where it stands |
| --- | --- |
| Accelerators | NVIDIA via nvidia-smi, ROCm via rocm-smi/rocminfo, Apple Metal via platform + torch's own MPS check. torch is asked what it was built for rather than inferred from the driver. |
| Container engines | docker and podman, with podman's different GPU flag noted. Windows containers are not handled: probes use `sh -lc`. |
| Fix commands | Quoted for the shell that will run them — cmd.exe reads `'numpy>=1.21'` as a redirect. `apt-get` is only offered as a runnable fix on a Linux host; inside an image it would not persist, and elsewhere it is fiction. POSIX fetch scripts become manual instructions when bash is absent. |
| Tracing | Needs Python 3.8+ *in the traced child*. Older interpreters are recorded as such, so an empty trace is never mistaken for a run that opened nothing. |
| Paths | Repo-relative with forward slashes internally; cross-drive and case-insensitive filesystems handled. |
| Shared libraries | `ldconfig` is Linux-only, so elsewhere the finding is UNKNOWN with the apt package named, rather than a false pass. |

Known limits:

- The static asset scanner reads string literals, so paths assembled at runtime
  are invisible to it. That is what `syp trace` is for — but tracing only sees
  the code paths a given run reaches, and `os.path.exists` raises no audit
  event, so a library that *checks* for a file without opening it stays
  invisible. The two methods are complementary: on WHAM the static pass named
  the SMPL directory that the traced run then died on, and the trace could not
  have found it.
- Reachability follows local imports only. A path named in a config file the
  entrypoint loads dynamically is still attributed to whatever file mentions
  it.
- The licence registry covers the 3D human-pose ecosystem well and everything
  else not at all. Contributions to `knowledge.py` are the point of that file.
- Version comparison is PEP 440-ish rather than exact, and the pair rules
  (torch/torchvision, numpy 2 ABI, mmcv generations) are curated heuristics.
  They are evaluated separately over declared pins and installed versions, so a
  coherent environment cannot mask an incoherent manifest.
- Import → distribution mapping is a lookup table plus a guess.
- `--target image` requires the image to be present locally; it will not pull
  one for you.

## Licence

MIT.
