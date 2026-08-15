from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import fixtures  # noqa: E402

from syp.cli import main  # noqa: E402
from syp.collect import run_all  # noqa: E402
from syp.collect.git import parse_gitmodules  # noqa: E402
from syp.context import RepoContext  # noqa: E402
from syp.model import Kind, Status  # noqa: E402
from syp.util import satisfies  # noqa: E402


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    root = tmp_path_factory.mktemp("wham_like")
    fixtures.build(str(root), git=fixtures.have_git())
    ctx = RepoContext.load(str(root))
    return run_all(ctx)


def named(report, needle):
    return [r for r in report.requirements if needle.lower() in r.name.lower()]


def one(report, needle):
    hits = named(report, needle)
    assert hits, f"no requirement matching {needle!r}; got {[r.name for r in report.requirements]}"
    return hits[0]


# --- git --------------------------------------------------------------------


def test_parse_gitmodules_handles_duplicate_sections():
    mods = parse_gitmodules(fixtures.FILES[".gitmodules"])
    assert [m["path"] for m in mods] == ["third-party/ViTPose", "third-party/DPVO"]
    assert mods[0]["url"].endswith("ViTPose.git")


def test_uninitialized_submodule_is_a_blocker(report):
    req = one(report, "third-party/ViTPose")
    assert req.status is Status.MISSING
    assert "not initialized" in req.detail
    assert req.fix and req.fix.startswith("git submodule update --init")


# --- python -----------------------------------------------------------------


def test_declared_packages_are_parsed(report):
    decl = one(report, "declarations")
    assert "requirements.txt" in decl.detail
    assert "package(s)" in decl.detail


def test_missing_package_carries_the_awkwardness_note(report):
    req = one(report, "mmcv-full")
    assert req.status in (Status.MISSING, Status.MISMATCH)
    if req.status is Status.MISSING:
        # Known-awkward packages must not get a naive `pip install` suggestion.
        assert req.fix is None
        assert "wheel" in (req.manual or "").lower() or "torch" in (req.manual or "").lower()


def test_extra_index_url_is_surfaced(report):
    req = one(report, "custom package index")
    assert "download.pytorch.org" in req.detail
    assert req.status is Status.INFO


def test_python_version_comparison():
    assert satisfies("3.9.7", ">=3.8") is True
    assert satisfies("3.9.7", "==3.9") is True
    assert satisfies("3.7.0", ">=3.8") is False
    assert satisfies("1.11.0", "==1.11.0") is True
    assert satisfies("2.0.1", "==1.11.0") is False
    assert satisfies("1.0", "") is True


# --- container --------------------------------------------------------------


def test_dockerfile_base_and_versions(report):
    req = one(report, "Dockerfile")
    assert "nvidia/cuda:11.3.1" in req.detail
    assert "cuda 11.3" in req.detail


def test_apt_packages_from_dockerfile(report):
    req = one(report, "apt packages")
    assert "ffmpeg" in req.detail


def test_image_documented_only_in_readme_is_found(report):
    reqs = named(report, "wham-vitpose-dpvo")
    assert reqs, "image mentioned in a README docker pull line should be discovered"
    assert reqs[0].kind is Kind.CONTAINER


# --- assets -----------------------------------------------------------------


def test_present_asset_is_ok(report):
    req = one(report, "checkpoints/dpvo.pth")
    assert req.status is Status.OK


def test_missing_checkpoint_points_at_the_fetch_script(report):
    req = one(report, "wham_vit_bedlam_w_3dpw.pth.tar")
    assert req.status is Status.MISSING
    assert req.fix == "bash fetch_demo_data.sh"


def test_gated_body_model_is_blocked_not_missing(report):
    req = one(report, "dataset/body_models/smpl")
    assert req.status is Status.BLOCKED
    assert "smpl.is.tue.mpg.de" in (req.manual or "")


def test_licence_gate_appears_once_in_external_section(report):
    gates = [r for r in report.by_kind(Kind.EXTERNAL) if r.meta.get("provider") == "smpl"]
    assert len(gates) == 1
    assert gates[0].status is Status.BLOCKED


def test_google_drive_hosts_are_flagged(report):
    req = one(report, "Google Drive")
    assert req.status is Status.MISMATCH
    assert "quota" in req.detail.lower() or "throttle" in req.detail.lower()


def test_output_paths_are_not_treated_as_inputs(report):
    assert not named(report, "output/results.pkl")


def test_test_fixtures_are_not_runtime_requirements(report):
    assert not named(report, "only_in_tests.pth")


def test_prose_instructions_never_become_a_command(report):
    req = one(report, "checkpoints/pretrain.pth")
    assert req.status is Status.MISSING
    assert req.fix is None, "`syp fix --yes` must never try to execute a README"
    assert "README.md" in (req.manual or "")


def test_bare_filename_duplicates_are_dropped(report):
    # demo.yaml names the checkpoint with a directory; nothing should report the
    # bare basename as a second, separate requirement.
    assert not [r for r in report.requirements if r.name == "wham_vit_bedlam_w_3dpw.pth.tar"]


def test_tool_installed_by_the_image_is_not_a_blocker(report):
    req = one(report, "ffmpeg")
    assert req.status is Status.INFO
    assert req not in report.blockers


def test_unfetchable_unknown_asset_says_so(report):
    req = one(report, "vitpose-h-multi-coco.pth")
    assert req.status is Status.MISSING
    assert req.fix is None
    assert "no setup script" in req.detail


# --- entrypoint -------------------------------------------------------------


def test_documented_demo_command_is_found(report):
    entry = next(r for r in report.requirements if r.meta.get("smoke"))
    assert entry.meta["command"].startswith("python demo.py")
    assert "README.md" in (entry.source or "")


def test_demo_input_existence_is_checked(report):
    req = one(report, "examples/IMG_9732.mov")
    assert req.status is Status.OK


# --- report / cli -----------------------------------------------------------


def test_readiness_is_between_zero_and_one(report):
    assert 0.0 < report.readiness < 1.0


def test_blockers_are_ordered_and_actionable(report):
    assert report.blockers
    assert all(b.fix or b.manual for b in report.blockers), "every blocker needs a next step"


def test_no_collector_crashed(report):
    assert report.notes == []


def test_cli_json_output(tmp_path, capsys):
    fixtures.build(str(tmp_path))
    code = main(["audit", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1  # blockers present
    assert payload["counts"]["blocked"] >= 1
    assert payload["readiness"] < 1.0


def test_cli_text_output_is_ascii_safe(tmp_path, capsys):
    fixtures.build(str(tmp_path))
    main(["audit", str(tmp_path), "--ascii", "--no-color"])
    out = capsys.readouterr().out
    assert "SHUT-YOUR-PYHOLE" in out
    assert "BLOCKERS" in out
    out.encode("ascii")  # must not raise


def test_fix_is_a_dry_run_by_default(tmp_path, capsys):
    fixtures.build(str(tmp_path))
    code = main(["fix", str(tmp_path), "--no-color"])
    out = capsys.readouterr().out
    assert "Dry run" in out
    assert code == 1


@pytest.mark.parametrize("term", ["SMPL", "SMPL_NEUTRAL.pkl", "smpl_neutral"])
def test_explain_reaches_the_knowledge_base(tmp_path, capsys, term):
    fixtures.build(str(tmp_path))
    main(["explain", term, str(tmp_path), "--no-color"])
    out = capsys.readouterr().out
    assert "smpl.is.tue.mpg.de" in out


def test_empty_directory_does_not_crash(tmp_path, capsys):
    code = main(["audit", str(tmp_path), "--no-color"])
    assert code in (0, 1)
    assert "SHUT-YOUR-PYHOLE" in capsys.readouterr().out
