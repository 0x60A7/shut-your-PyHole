"""Tests for the second layer: tracing, imports, env, build, integrity, config,
targets and fix safety."""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import fixtures  # noqa: E402

from syp import integrity, trace as trace_mod  # noqa: E402
from syp.cli import main  # noqa: E402
from syp.collect import run_all  # noqa: E402
from syp.config import Config  # noqa: E402
from syp.context import RepoContext  # noqa: E402
from syp.model import FixKind, Kind, Status, classify_fix  # noqa: E402
from syp.target import resolve as resolve_target  # noqa: E402


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("stack")
    fixtures.build(str(root), git=fixtures.have_git())
    return str(root)


@pytest.fixture(scope="module")
def report(repo):
    return run_all(RepoContext.load(repo, target_spec="host"))


def named(report, needle):
    return [r for r in report.requirements if needle.lower() in r.name.lower()]


def one(report, needle):
    hits = named(report, needle)
    assert hits, f"no requirement matching {needle!r}"
    return hits[0]


# --- imports ----------------------------------------------------------------


def test_undeclared_import_is_reported(report):
    req = one(report, "import cv2")
    assert req.status in (Status.MISSING, Status.MISMATCH)
    assert "declared in no manifest" in req.detail
    assert req.meta["distribution"] == "opencv-python"


def test_local_namespace_package_is_not_a_dependency(report):
    # lib/models/__init__.py exists but lib/__init__.py does not; `import lib`
    # must still resolve locally.
    assert not named(report, "import lib")


def test_declared_import_is_not_reported_twice(report):
    assert not named(report, "import torch")


def test_system_library_for_cv2_is_surfaced(report):
    req = one(report, "for cv2")
    assert "libGL" in req.name
    # Status depends on the platform; the finding itself must exist either way.
    assert req.status in (Status.OK, Status.MISSING, Status.UNKNOWN)


# --- environment ------------------------------------------------------------


def test_required_env_var_is_a_blocker(report):
    req = one(report, "WHAM_CACHE")
    assert req.status is Status.MISSING
    assert "KeyError" in req.detail


def test_credential_env_var_is_blocked_not_missing(report):
    req = one(report, "HF_TOKEN")
    assert req.status is Status.BLOCKED
    assert req.meta.get("credential") is True


def test_defaulted_env_var_is_only_informational(report):
    req = one(report, "DEVICE")
    assert req.status is Status.INFO


# --- integrity --------------------------------------------------------------


def test_html_interstitial_is_caught(report):
    req = one(report, "interstitial.pth")
    assert req.status is Status.STALE
    assert "HTML" in req.detail
    assert "Drive" in (req.explain or "")


def test_valid_checkpoint_passes(report):
    req = one(report, "checkpoints/dpvo.pth")
    assert req.status is Status.OK


def test_integrity_rejects_undersized_model(tmp_path):
    path = tmp_path / "tiny.pth"
    path.write_bytes(b"PK\x03\x04" + b"\x00" * 10)
    verdict = integrity.inspect(str(path), "tiny.pth")
    assert not verdict.ok and "bytes" in verdict.problem


def test_integrity_rejects_wrong_magic(tmp_path):
    path = tmp_path / "wrong.npy"
    path.write_bytes(b"not a numpy file" * 1000)
    verdict = integrity.inspect(str(path), "wrong.npy")
    assert not verdict.ok


def test_integrity_accepts_a_real_looking_checkpoint(tmp_path):
    path = tmp_path / "ok.pth"
    path.write_bytes(b"PK\x03\x04" + b"\x00" * 20000)
    assert integrity.inspect(str(path), "ok.pth").ok


def test_checksum_parsing():
    found = integrity.declared_checksums(
        {"SHA256SUMS": "a" * 64 + "  models/net.pth\n" + "b" * 32 + "  other.bin\n"}
    )
    assert found["net.pth"][0] == "a" * 64
    assert found["other.bin"][0] == "b" * 32


# --- build ------------------------------------------------------------------


def test_cuda_sources_trigger_build_requirements(tmp_path):
    fixtures.build(str(tmp_path))
    (tmp_path / "lib" / "ops").mkdir(parents=True, exist_ok=True)
    (tmp_path / "lib" / "ops" / "kernel.cu").write_text("__global__ void k() {}\n")
    (tmp_path / "setup.py").write_text(
        "from torch.utils.cpp_extension import CUDAExtension, BuildExtension\n"
        "ext_modules=[CUDAExtension('ops', ['lib/ops/kernel.cu'])]\n"
    )
    report = run_all(RepoContext.load(str(tmp_path), target_spec="host"), only=["build"])
    assert [r for r in report.by_kind(Kind.BUILD)]
    assert any("extension" in r.name or "CUDA" in r.name for r in report.by_kind(Kind.BUILD))


def test_documented_editable_install_is_a_build_step(tmp_path):
    fixtures.build(str(tmp_path))
    (tmp_path / "INSTALL.md").write_text("Then run:\n\n```bash\npip install -v -e third-party/DPVO\n```\n")
    report = run_all(RepoContext.load(str(tmp_path), target_spec="host"), only=["build"])
    steps = [r for r in report.by_kind(Kind.BUILD) if r.name == "build step"]
    assert steps and "-e" in steps[0].detail


# --- resolve ----------------------------------------------------------------


def test_torch_torchvision_pair_mismatch(tmp_path):
    fixtures.build(str(tmp_path))
    (tmp_path / "requirements.txt").write_text("torch==2.1.0\ntorchvision==0.12.0\n")
    report = run_all(RepoContext.load(str(tmp_path), target_spec="host"), only=["resolve"])
    hits = named(report, "torchvision mismatch")
    assert hits, [r.name for r in report.requirements]
    assert "0.16" in hits[0].detail


def test_contradictory_pins_across_files(tmp_path):
    fixtures.build(str(tmp_path))
    (tmp_path / "requirements.txt").write_text("numpy==1.21.0\n")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\nversion="1"\ndependencies=["numpy==1.26.0"]\n'
    )
    report = run_all(RepoContext.load(str(tmp_path), target_spec="host"), only=["resolve"])
    hits = named(report, "numpy pinned twice")
    assert hits and "1.21.0" in hits[0].detail


def test_numpy2_conflict_rule(tmp_path):
    fixtures.build(str(tmp_path))
    (tmp_path / "requirements.txt").write_text("numpy==2.1.0\nchumpy\n")
    report = run_all(RepoContext.load(str(tmp_path), target_spec="host"), only=["resolve"])
    hits = named(report, "numpy 2.1.0")
    assert hits and "chumpy" in hits[0].detail


# --- trace ------------------------------------------------------------------


def test_trace_records_opens_imports_and_execs(tmp_path):
    fixtures.build(str(tmp_path))
    probe = tmp_path / "probe_run.py"
    probe.write_text(
        "import json\n"
        "try:\n"
        "    open('checkpoints/absent_model.pth', 'rb')\n"
        "except OSError:\n"
        "    pass\n"
        "open('checkpoints/dpvo.pth', 'rb').close()\n"
    )
    out = str(tmp_path / "trace.jsonl")
    code = trace_mod.run_traced(f'"{sys.executable}" probe_run.py', str(tmp_path), out, timeout=120)
    trace_mod.record_exit(out, code)
    recorded = trace_mod.load(out, str(tmp_path))

    assert "checkpoints/absent_model.pth" in recorded.missing
    assert "checkpoints/dpvo.pth" in recorded.opened
    assert "checkpoints/dpvo.pth" not in recorded.missing
    assert "json" in recorded.imports
    assert recorded.exit_code == 0


def test_checkpoint_paths_survive_the_noise_filter():
    # `.pth` is both a torch checkpoint and a setuptools path file.
    assert trace_mod._relevant_path("checkpoints/model.pth", "/repo") == "checkpoints/model.pth"
    assert trace_mod._relevant_path("/usr/lib/python3.11/site-packages/x.pth", "/repo") is None
    assert trace_mod._relevant_path("/repo/lib/thing.pyc", "/repo") is None


def test_observed_missing_path_becomes_a_requirement(tmp_path):
    fixtures.build(str(tmp_path))
    out = os.path.join(str(tmp_path), "trace.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for event in [
            {"kind": "command", "value": "python demo.py", "extra": ""},
            {"kind": "open", "value": "weights/observed_only.pth", "extra": ""},
            {"kind": "exec", "value": "ffmpeg", "extra": ""},
            {"kind": "exit", "value": "1", "extra": ""},
        ]:
            fh.write(json.dumps(event) + "\n")

    ctx = RepoContext.load(str(tmp_path), target_spec="host")
    ctx.trace = trace_mod.load(out, str(tmp_path))
    report = run_all(ctx)

    asset = one(report, "weights/observed_only.pth")
    assert asset.status is Status.MISSING
    assert "opened this at runtime" in asset.detail
    smoke = [r for r in report.requirements if r.name == "smoke test"]
    assert smoke and smoke[0].status is Status.MISSING


def test_observation_overrides_inference_for_binaries(tmp_path):
    fixtures.build(str(tmp_path))
    out = os.path.join(str(tmp_path), "trace.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "exec", "value": "ffmpeg"}) + "\n")
    ctx = RepoContext.load(str(tmp_path), target_spec="host")
    ctx.trace = trace_mod.load(out, str(tmp_path))
    report = run_all(ctx)
    entries = named(report, "ffmpeg")
    # The Dockerfile-provided INFO entry must be replaced, not duplicated.
    assert len(entries) == 1
    assert entries[0].source == "observed at runtime"


# --- config -----------------------------------------------------------------


def test_config_suppresses_named_findings(tmp_path):
    fixtures.build(str(tmp_path))
    (tmp_path / ".syp.toml").write_text(
        '[ignore]\nnames = ["WHAM_CACHE", "import cv2"]\npaths = ["checkpoints/yolov8x.pt"]\n'
    )
    report = run_all(RepoContext.load(str(tmp_path), target_spec="host"))
    assert not named(report, "WHAM_CACHE")
    assert not named(report, "import cv2")
    assert not named(report, "yolov8x.pt")
    assert len(report.suppressed) >= 2


def test_config_assume_installed(tmp_path):
    fixtures.build(str(tmp_path))
    (tmp_path / ".syp.toml").write_text('[assume]\ninstalled = ["mmcv-full"]\n')
    report = run_all(RepoContext.load(str(tmp_path), target_spec="host"))
    assert not [r for r in named(report, "mmcv-full") if r.status is Status.MISSING]


def test_config_path_glob():
    cfg = Config(ignore_paths=["data/**"])
    assert cfg.ignores_path("data/nested/file.pth")
    assert not cfg.ignores_path("checkpoints/file.pth")


# --- targets ----------------------------------------------------------------


def test_target_venv_falls_back_with_an_explanation(tmp_path):
    target = resolve_target(str(tmp_path), "venv")
    assert target.kind == "host"
    assert "no virtualenv" in (target.problem or "")


def test_target_image_without_a_name_is_reported_not_guessed(tmp_path):
    target = resolve_target(str(tmp_path), "image", images=["a/one:latest", "b/two:latest"])
    assert not target.available
    assert "several images" in (target.problem or "")


def test_pip_fix_targets_the_inspected_interpreter(report):
    hits = [r for r in named(report, "numpy") if r.fix]
    assert hits, "numpy should be missing with a pip fix"
    # Never a bare `pip install`: that installs into whatever happens to be active.
    assert "-m pip install" in hits[0].fix


def test_target_is_recorded_in_the_report(report):
    assert report.target


# --- fix safety -------------------------------------------------------------


def test_fix_classification():
    assert classify_fix("git submodule update --init --recursive") is FixKind.LOCAL
    assert classify_fix("docker pull x/y:latest") is FixKind.NETWORK
    assert classify_fix("/usr/bin/python -m pip install 'numpy>=1.21'") is FixKind.NETWORK
    assert classify_fix('"C:\\envs\\p\\python.exe" -m pip install numpy') is FixKind.NETWORK
    assert classify_fix("python -m pip install numpy") is FixKind.NETWORK
    assert classify_fix("bash fetch_demo_data.sh") is FixKind.SCRIPT
    assert classify_fix("python tools/download.py") is FixKind.SCRIPT


def test_repo_scripts_are_withheld_from_fix(tmp_path, capsys):
    fixtures.build(str(tmp_path))
    main(["fix", str(tmp_path), "--no-color"])
    out = capsys.readouterr().out
    assert "withheld" in out
    assert "--allow-scripts" in out
    assert "fetch_demo_data.sh" in out


# --- report -----------------------------------------------------------------


def test_json_leads_with_the_blocking_count(tmp_path, capsys):
    fixtures.build(str(tmp_path))
    main(["audit", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["blocking"] >= 1
    assert payload["readiness_is_advisory"] is True
    assert payload["checked"] >= payload["satisfied"]


def test_score_line_leads_with_blockers(tmp_path, capsys):
    fixtures.build(str(tmp_path))
    main(["audit", str(tmp_path), "--no-color", "--ascii"])
    out = capsys.readouterr().out
    assert "blocker(s)" in out
    assert "checks satisfied" in out
