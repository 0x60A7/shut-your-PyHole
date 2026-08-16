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
    # Must be found in the *declared* pins even when a coherent torch pair
    # happens to be installed in the environment doing the auditing.
    declared = [h for h in hits if h.meta.get("view") == "declared"]
    assert declared and "0.16" in declared[0].detail


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
    # The declared numpy must be judged on its own terms; an older numpy being
    # installed here says nothing about what this manifest asks for.
    assert hits and "chumpy" in hits[0].detail
    assert hits[0].meta.get("view") == "declared"


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


# --- entrypoint scoping -----------------------------------------------------


def test_reachability_follows_local_imports(repo):
    from syp.reach import reachable

    ctx = RepoContext.load(repo, target_spec="host")
    reached = reachable(ctx, "demo.py")
    assert "demo.py" in reached
    assert "lib/models/__init__.py" in reached
    assert "train.py" not in reached
    assert "lib/data/loaders.py" not in reached


def test_training_assets_do_not_block_the_demo(report):
    """The failure this fixes: two thirds of WHAM's blockers belonged to
    training, which nobody running the demo needs."""
    assert not [r for r in named(report, "amass.pth") if r.status.is_blocker]
    scoped = one(report, "assets for other entrypoints")
    assert scoped.status is Status.INFO
    assert any("amass.pth" in item for item in scoped.meta["packages"])


def test_entry_flag_promotes_the_other_entrypoint(repo):
    ctx = RepoContext.load(repo, target_spec="host")
    ctx.config.entry = "train.py"
    report = run_all(ctx, only=["assets", "entrypoint"])
    hits = [r for r in named(report, "amass.pth") if r.status.is_blocker]
    assert hits, "auditing train.py must demand the training data"
    # ... and the demo's own inputs drop out of scope in exchange.
    scoped = named(report, "assets for other entrypoints")
    assert scoped


def test_cache_written_then_read_is_not_a_requirement(report):
    # demo.py saves tracking_results.pth and loads it on a later run.
    assert not named(report, "tracking_results.pth")


# --- signal-to-noise (measured on a 19-repo corpus) -------------------------


def test_unprovisioned_environment_is_one_blocker_not_sixty(tmp_path):
    """`requests` reported 16 blockers, all saying 'you have not installed it'."""
    # Fabricated names, so the test means the same thing on a bare host and
    # inside an image that happens to have half the real ones installed.
    (tmp_path / "requirements.txt").write_text(
        "syp-absent-alpha>=1\nsyp-absent-beta>=2\nsyp-absent-gamma\n"
        "syp-absent-delta==3\nsyp-absent-epsilon\n"
    )
    (tmp_path / "run.py").write_text("import os\n")
    report = run_all(RepoContext.load(str(tmp_path), target_spec="host"), only=["python"])
    blockers = [r for r in report.blockers]
    assert len(blockers) == 1, [b.name for b in blockers]
    assert "not provisioned" in blockers[0].name
    assert "requirements.txt" in (blockers[0].fix or "")
    # The detail is not lost, just demoted.
    assert len(blockers[0].meta["packages"]) == 5


def test_dev_extras_are_not_runtime_requirements(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\nversion="1"\ndependencies=["certifi"]\n'
        '[project.optional-dependencies]\ntest=["pytest-cov","httpbin"]\ndocs=["sphinx"]\n'
    )
    (tmp_path / "run.py").write_text("import certifi\n")
    report = run_all(RepoContext.load(str(tmp_path), target_spec="host"), only=["python"])
    assert not [r for r in report.blockers if "pytest-cov" in r.name or "sphinx" in r.name]
    extras = one(report, "dev/test extras absent")
    assert extras.status is Status.INFO


def test_requirements_dev_file_is_optional(tmp_path):
    (tmp_path / "requirements.txt").write_text("certifi\n")
    (tmp_path / "requirements-dev.txt").write_text("black\nruff\n")
    (tmp_path / "run.py").write_text("import certifi\n")
    report = run_all(RepoContext.load(str(tmp_path), target_spec="host"), only=["python"])
    assert not [r for r in report.blockers if "black" in r.name or "ruff" in r.name]


def test_library_without_an_entrypoint_reports_an_inventory(tmp_path):
    """pytorch_geometric produced 123 asset blockers: its own download catalogue."""
    pkg = tmp_path / "mylib"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "datasets.py").write_text(
        "URLS = {'a': 'data/cora.npz', 'b': 'data/citeseer.npz', 'c': 'weights/gcn.pt'}\n"
    )
    (tmp_path / "pyproject.toml").write_text('[project]\nname="mylib"\nversion="1"\n')
    report = run_all(RepoContext.load(str(tmp_path), target_spec="host"), only=["assets", "entrypoint"])
    assert not [r for r in report.blockers if r.kind is Kind.ASSET]
    inventory = one(report, "referenced files not on disk")
    assert inventory.status is Status.INFO
    assert "library" in (inventory.explain or "")


def test_ci_workflows_never_produce_requirements(tmp_path):
    """ultralytics' CI matrix names every model variant it has ever shipped."""
    fixtures.build(str(tmp_path))
    ci = tmp_path / ".github" / "workflows"
    ci.mkdir(parents=True)
    (ci / "ci.yml").write_text("jobs:\n  t:\n    run: yolo predict model=yolo26n-seg.pt\n")
    report = run_all(RepoContext.load(str(tmp_path), target_spec="host"), only=["assets"])
    assert not named(report, "yolo26n-seg.pt")


def test_generated_bindings_are_not_requirements(tmp_path):
    """tinygrad's autogenerated AMD firmware table alone produced 104 findings."""
    fixtures.build(str(tmp_path))
    gen = tmp_path / "mylib" / "autogen"
    gen.mkdir(parents=True)
    (gen / "fw.py").write_text("BLOBS = ['amdgpu/psp_13_0_0_ta.bin', 'amdgpu/vcn_4_0_0.bin']\n")
    plain = tmp_path / "mylib" / "hand_written.py"
    plain.write_text('"""auto-generated by ctypesgen"""\nB = "amdgpu/other_blob.bin"\n')
    report = run_all(RepoContext.load(str(tmp_path), target_spec="host"), only=["assets"])
    assert not named(report, "psp_13_0_0_ta.bin"), "autogen/ directory"
    assert not named(report, "other_blob.bin"), "auto-generated header"


def test_guarded_imports_are_optional_backends(tmp_path):
    """peft reported 45 blockers for backends it works fine without."""
    (tmp_path / "requirements.txt").write_text("certifi\n")
    (tmp_path / "run.py").write_text(
        "import certifi\n"
        "try:\n    import bitsandbytes\nexcept ImportError:\n    bitsandbytes = None\n"
        "def later():\n    import aqlm\n    return aqlm\n"
        "import scipy\n"
    )
    (tmp_path / "README.md").write_text("```bash\npython run.py\n```\n")
    report = run_all(RepoContext.load(str(tmp_path), target_spec="host"), only=["imports"])
    found = [r.name for r in report.requirements]
    assert not any("bitsandbytes" in n for n in found)
    assert not any("aqlm" in n for n in found)
    assert any("scipy" in n for n in found), "a plain top-level import is still reported"


def test_placeholder_paths_are_never_requirements(tmp_path):
    fixtures.build(str(tmp_path))
    (tmp_path / "README.md").write_text(
        "```bash\npython demo.py\n```\nSet model to path/to/model.pt or your_weights.pth\n"
    )
    report = run_all(RepoContext.load(str(tmp_path), target_spec="host"), only=["assets"])
    assert not named(report, "path/to/model.pt")
    assert not named(report, "your_weights.pth")


def test_docs_builder_is_never_the_entrypoint(tmp_path):
    """ultralytics anchored all its scoping to docs/build_docs.py."""
    from syp.collect.entrypoint import entry_file

    fixtures.build(str(tmp_path))
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "build_docs.py").write_text("print('building')\n")
    (tmp_path / "README.md").write_text(
        "```bash\npython docs/build_docs.py\n```\n```bash\npython demo.py --video x.mov\n```\n"
    )
    assert entry_file(RepoContext.load(str(tmp_path), target_spec="host")) == "demo.py"


# --- container tracing ------------------------------------------------------


def test_container_trace_carries_hook_and_trace_across_the_boundary():
    from syp.target import Target
    from syp.trace import container_argv

    target = Target(kind="image", label="img", image="org/img:tag", engine="docker",
                    gpu_flags=["--gpus", "all"])
    argv = container_argv(target, "python demo.py", "/repo/src", "/repo/src/.syp/trace.jsonl", "/tmp/hook")
    joined = " ".join(argv)
    assert "--gpus all" in joined
    assert "/tmp/hook:/syp-hook:ro" in joined
    assert "PYTHONPATH=/syp-hook" in joined
    assert "SYP_TRACE_FILE=/repo/.syp/trace.jsonl" in joined
    assert argv[-3:] == ["sh", "-lc", "python demo.py"]


def test_container_paths_map_back_to_the_host():
    assert trace_mod._relevant_path("/repo/checkpoints/m.pth", "/anything") == "checkpoints/m.pth"


def test_bytecode_writes_are_never_requirements():
    # CPython writes `x.pyc.<id>` then renames; it is not a missing dependency.
    assert trace_mod._relevant_path("lib/__pycache__/m.cpython-39.pyc.13260", "/repo") is None


def test_trace_records_reads_but_not_writes(tmp_path):
    fixtures.build(str(tmp_path))
    probe = tmp_path / "probe_rw.py"
    probe.write_text(
        "open('checkpoints/dpvo.pth', 'rb').close()\n"
        "open('written_output.bin', 'wb').close()\n"
    )
    out = str(tmp_path / "rw.jsonl")
    trace_mod.run_traced(f'"{sys.executable}" probe_rw.py', str(tmp_path), out, timeout=120)
    recorded = trace_mod.load(out, str(tmp_path))
    assert "checkpoints/dpvo.pth" in recorded.opened
    assert "written_output.bin" not in recorded.opened


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


def test_pip_fix_targets_the_inspected_interpreter(tmp_path):
    fixtures.build(str(tmp_path))
    # A name no environment will have, so the assertion holds on any machine.
    (tmp_path / "requirements.txt").write_text("syp-definitely-not-installed==1.0\n")
    report = run_all(RepoContext.load(str(tmp_path), target_spec="host"), only=["python"])
    hits = [r for r in named(report, "syp-definitely-not-installed") if r.fix]
    assert hits, [r.name for r in report.requirements]
    # Never a bare `pip install`: that installs into whatever happens to be active.
    assert "-m pip install" in hits[0].fix
    assert sys.executable.split(os.sep)[-1] in hits[0].fix


def test_pip_fix_quoting_suits_the_shell_that_runs_it():
    from syp.target import Target

    posix = Target(kind="venv", label="v", python_exe="/env/bin/python")
    windows = Target(kind="venv", label="v", python_exe=r"C:\env\Scripts\python.exe")
    quoted = posix.pip_command("numpy>=1.21") if posix.shell_is_posix else None
    if quoted:
        assert "'numpy>=1.21'" in quoted
    # cmd.exe treats a single-quoted `>` as redirection, so it must be double
    # quotes there — otherwise `syp fix --yes` writes a junk file and installs
    # the wrong package.
    if not windows.shell_is_posix:
        assert '"numpy>=1.21"' in windows.pip_command("numpy>=1.21")


def test_target_is_recorded_in_the_report(report):
    assert report.target


def test_unreachable_target_is_inconclusive_not_clean(tmp_path):
    """The failure mode this guards against: every probe returns UNKNOWN because
    nothing could be inspected, and the report calls that zero blockers."""
    fixtures.build(str(tmp_path))
    ctx = RepoContext.load(str(tmp_path), target_spec="image:syp-definitely/not-here:v0")
    report = run_all(ctx, only=["python"])
    assert report.inconclusive
    assert report.blockers, "an un-inspectable target is itself a blocker"
    assert report.to_dict()["inconclusive"] is True


def test_cli_says_not_verified_and_exits_nonzero(tmp_path, capsys):
    fixtures.build(str(tmp_path))
    code = main([
        "audit", str(tmp_path), "--no-color", "--only", "python",
        "--target", "image:syp-definitely/not-here:v0",
    ])
    out = capsys.readouterr().out
    assert code == 1
    assert "NOT VERIFIED" in out
    assert "Nothing is blocking" not in out


def test_repo_without_docker_reports_no_container_section(tmp_path):
    (tmp_path / "requirements.txt").write_text("requests\n")
    (tmp_path / "run.py").write_text("import requests\n")
    (tmp_path / "README.md").write_text("# plain\n\n```bash\npython run.py\n```\n")
    report = run_all(RepoContext.load(str(tmp_path), target_spec="host"))
    assert not report.by_kind(Kind.CONTAINER)
    assert not [r for r in report.requirements if r.name in ("docker", "podman")]
    assert not report.inconclusive


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


# --- parsing python instead of pattern-matching it --------------------------


def _scan(code):
    from syp import pyscan

    return {(f.path, f.is_output) for f in pyscan.scan(code)}


def test_composed_paths_are_resolved():
    """The limitation this removes: `os.path.join(ROOT, ...)` was invisible."""
    found = _scan(
        'import os\n'
        'ROOT = "dataset/body_models"\n'
        'CKPT = os.path.join(ROOT, "smpl", "SMPL_NEUTRAL.pkl")\n'
    )
    assert ("dataset/body_models/smpl/SMPL_NEUTRAL.pkl", False) in found
    # ...and the pieces are not also reported as three separate requirements.
    assert not any(p == "smpl" for p, _ in found)
    assert not any(p == "SMPL_NEUTRAL.pkl" for p, _ in found)


def test_docstrings_are_not_requirements():
    found = _scan('def f():\n    """Load it with torch.load("docs/example.pth")."""\n    return 1\n')
    assert not any("example.pth" in p for p, _ in found)


def test_write_context_is_understood():
    found = _scan(
        'import os, torch\n'
        'def main(out_dir):\n'
        '    torch.save(1, os.path.join(out_dir, "ckpt.pt"))\n'
        '    torch.load("checkpoints/model.pth")\n'
        '    open("logs/run.log", "w").close()\n'
    )
    assert ("checkpoints/model.pth", False) in found
    assert ("out_dir/ckpt.pt", True) in found or any(
        p.endswith("ckpt.pt") and is_out for p, is_out in found
    )
    assert any(p == "logs/run.log" and is_out for p, is_out in found)


def test_absolute_and_system_paths_are_not_repo_files(tmp_path):
    fixtures.build(str(tmp_path))
    (tmp_path / "probe_sys.py").write_text(
        'A = "/dev/kfd"\nB = "/sys/fs/cgroup/cpu.max"\nC = "C:/Windows/x.pkl"\n'
    )
    report = run_all(RepoContext.load(str(tmp_path), target_spec="host"), only=["assets"])
    assert not named(report, "dev/kfd")
    assert not named(report, "sys/fs")
    assert not named(report, "Windows/x.pkl")


def test_extensionless_strings_are_not_paths(tmp_path):
    """`train/loss`, `application/json` and `org/model-name` all look like paths."""
    fixtures.build(str(tmp_path))
    (tmp_path / "probe_ids.py").write_text(
        'M = "liuhaotian/llava-v1.5-13b"\nK = "train/grad_norm"\nT = "application/json"\n'
    )
    report = run_all(RepoContext.load(str(tmp_path), target_spec="host"), only=["assets"])
    for noise in ("llava-v1.5-13b", "grad_norm", "application/json"):
        assert not named(report, noise)


def test_unparseable_python_falls_back_to_regex(tmp_path):
    fixtures.build(str(tmp_path))
    # Python 2 syntax: the AST pass cannot read it, the scan must still work.
    (tmp_path / "legacy.py").write_text('print "hi"\nW = "checkpoints/legacy_model.pth"\n')
    report = run_all(RepoContext.load(str(tmp_path), target_spec="host"), only=["assets"])
    # legacy.py is not reachable from demo.py, so it lands in the out-of-scope
    # inventory rather than the blocker list — but it must be found at all.
    inventory = [
        item
        for r in report.requirements
        for item in r.meta.get("packages", [])
    ]
    assert named(report, "legacy_model.pth") or any("legacy_model.pth" in i for i in inventory)


# --- makefiles --------------------------------------------------------------


def test_makefile_parsing():
    from syp import makefile

    mk = makefile.parse(fixtures.FILES["Makefile"], "Makefile")
    assert mk.variables["CKPT_DIR"] == "checkpoints"
    assert mk.targets["style"].is_maintenance and mk.targets["style"].phony
    assert mk.targets["build"].builds
    assert mk.targets["data"].fetches
    assert mk.targets["demo"].runs
    assert sorted(t.name for t in mk.interesting()) == ["build", "data", "demo"]
    # Variables must be expanded before a recipe means anything.
    assert "checkpoints/make_fetched.pth" in mk.expand(mk.targets["data"].body)


def test_maintenance_targets_are_not_requirements(report):
    assert not named(report, "make style")


def test_makefile_fetch_target_is_credited_for_the_asset(report):
    req = one(report, "make_fetched.pth")
    assert req.status is Status.MISSING
    assert req.fix == "make data"
    assert "make data" in req.detail  # not the internal Makefile::data notation
    assert "::" not in req.detail


def test_make_is_required_when_the_project_compiles_with_it(report):
    req = one(report, "make")
    assert req.kind is Kind.BUILD
    # This fixture has an nvcc target, so make is genuinely needed.
    assert req.status in (Status.OK, Status.MISSING)


def test_make_is_not_required_for_a_convenience_target(tmp_path):
    """requests' Makefile has `init: pip install -r requirements-dev.txt`; that
    does not make GNU make a dependency of requests."""
    (tmp_path / "requirements.txt").write_text("certifi\n")
    (tmp_path / "run.py").write_text("import certifi\n")
    (tmp_path / "Makefile").write_text(
        ".PHONY: init test\ninit:\n\tpython -m pip install -r requirements.txt\ntest:\n\tpytest tests\n"
    )
    report = run_all(RepoContext.load(str(tmp_path), target_spec="host"), only=["build"])
    blocking = [r for r in report.blockers if r.name == "make"]
    assert not blocking, "a convenience target must not make `make` a blocker"


def test_documented_make_target_can_be_the_entrypoint(tmp_path):
    from syp.collect.entrypoint import chosen_command, entry_file

    fixtures.build(str(tmp_path))
    (tmp_path / "README.md").write_text("Run it:\n\n```bash\nmake demo\n```\n")
    ctx = RepoContext.load(str(tmp_path), target_spec="host")
    command, _ = chosen_command(ctx)
    assert command == "make demo"
    # ...and scoping still resolves to the script the recipe actually runs.
    assert entry_file(ctx) == "demo.py"


def test_docs_makefile_is_not_the_projects_build(tmp_path):
    (tmp_path / "requirements.txt").write_text("certifi\n")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "Makefile").write_text("html:\n\tsphinx-build -b html . _build\n")
    report = run_all(RepoContext.load(str(tmp_path), target_spec="host"), only=["build"])
    assert not named(report, "make")
