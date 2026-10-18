import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import config_loader
from experiments.distill import config_runner_impl, run_distill


def _student_data():
    return SimpleNamespace(
        x=np.arange(12, dtype=np.float32).reshape(4, 1, 3),
        y=np.asarray([0, 1, 1, 0], dtype=np.int64),
        subjects=np.asarray([1, 1, 0, 0], dtype=np.int64),
        sample_uid=np.asarray(["u0", "u1", "u2", "u3"]),
        source_manifest_sha256="source-sha",
        input_manifest_sha256="student-input-sha",
        identity={"input_policy_sha256": "student-policy"},
        source_path=Path("/tmp/loso-source.npy"),
        input_manifest_path=None,
    )


def _targets():
    return {
        "logits": np.asarray([[0.2, 0.8], [0.7, 0.3]]),
        "feats": None,
        "_manifest_path": "/tmp/targets/manifest.json",
        "_manifest_sha256": "target-manifest-sha",
        "_manifest": {"teacher_checkpoint_sha256": "teacher-checkpoint-sha"},
        "_artifact_path": "/tmp/targets/teacher.npz",
        "_input_manifest_sha256": "teacher-input-sha",
        "_input_policy_sha256": "teacher-policy",
    }


def _spec(seed=666, params=None, protocol="loso", methods=None):
    return {
        "type": "distill",
        "method": "kd",
        "runtime_method": "KD_all",
        "config_name": "teacher_seed_test",
        "run_id": "teacher_seed_test__run",
        "grid_index": 0,
        "pair_index": 0,
        "variant": None,
        "dataset": "BNCI2014001",
        "protocol": protocol,
        "teacher": "cbramod",
        "student": "ifnet",
        "methods": methods or ["KD_all"],
        "params": dict(params or {}),
        "training": {
            "epochs": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "batch_size": 2,
        },
        "subjects": [0],
        "seeds": [seed],
        "val_split": 0.2,
        "artifact_root": "/data1/llx/BigSmallcollab/cache/artifact-loso",
        "output": "/tmp/teacher_seed_test.csv",
        "log_file": "/tmp/teacher_seed_test.log",
        "fail_fast": False,
    }


def _patch_config_cell(monkeypatch, target_calls, training_calls):
    student_data = _student_data()
    monkeypatch.setattr(
        config_runner_impl.legacy.config, "load_dataset_config",
        lambda _dataset: {
            "num_subjects": 2, "num_classes": 2, "seeds": [666],
            "val_split": 0.2,
        })
    monkeypatch.setattr(
        config_runner_impl.loso_source, "load_dataset",
        lambda *_args, **_kwargs: student_data)

    def load_targets(dataset, teacher, teacher_seed, fold, **kwargs):
        target_calls.append((dataset, teacher, teacher_seed, fold))
        return _targets()

    monkeypatch.setattr(
        config_runner_impl.legacy, "_load_or_ensure_loso_teacher_targets",
        load_targets)
    monkeypatch.setattr(
        config_runner_impl.legacy, "_check_teacher", lambda *_args, **_kwargs: {
            "labels_match": True, "uid_set_match": True,
        })
    monkeypatch.setattr(
        config_runner_impl.legacy, "_artifact_info",
        lambda path: {"path": str(path), "sha256": "teacher-artifact-sha"})

    def run_method(args, dataset, protocol, teacher, student, artifact, fold,
                   student_seed, method, X_tr, y_tr, X_te, y_te, subj_tr,
                   targets, num_classes, device, **kwargs):
        training_calls.append((student_seed, method))
        return ({
            "seed": int(student_seed), "method": method,
            "acc": 50.0, "kappa": 0.0, "test_balanced_accuracy": 50.0,
            "failure_status": "complete",
            "teacher_target_manifest_path": targets.get("_manifest_path", ""),
            "teacher_target_manifest_sha256": targets.get("_manifest_sha256", ""),
            "teacher_checkpoint_sha256": targets.get("_manifest", {}).get(
                "teacher_checkpoint_sha256", ""),
            "teacher_input_manifest_sha256": targets.get(
                "_input_manifest_sha256", ""),
            "teacher_input_policy_sha256": targets.get("_input_policy_sha256", ""),
        }, {})

    monkeypatch.setattr(config_runner_impl.legacy, "_run_method", run_method)


@pytest.mark.parametrize(
    "parser, option",
    [
        (run_distill.parse_args, "--teacher-seed"),
        (run_distill.parse_args, "--teacher_seed"),
        (config_runner_impl.parse_args, "--teacher-seed"),
        (config_runner_impl.parse_args, "--teacher_seed"),
    ],
)
def test_both_parsers_accept_teacher_seed_spellings(parser, option):
    if parser is config_runner_impl.parse_args:
        args = parser(["--config", "unused.yaml", option, "0"])
    else:
        args = parser([option, "0"])
    assert args.teacher_seed == 0


def test_teacher_seed_defaults_to_none_in_both_parsers():
    assert run_distill.parse_args([]).teacher_seed is None
    assert config_runner_impl.parse_args(["--config", "unused.yaml"]).teacher_seed is None


def test_config_main_selects_loso_student_seed_and_cli_teacher_seed(
        tmp_path, monkeypatch):
    config_path = tmp_path / "teacher-seed.yaml"
    config_path.write_text(f"""
name: teacher_seed_main_test
type: distill
method: kd
datasets: [BNCI2014001]
protocols: [loso]
seeds: null
subjects: [0]
include_baseline: true
artifact_root: /data1/llx/BigSmallcollab/cache/artifact-loso
output_dir: {tmp_path}
pairs:
  - teacher: cbramod
    student: ifnet
    params:
      temperature: 1.0
      lam_kd: 0.5
      teacher_seed: 17
    grid: {{}}
""")
    original = config_loader.load_experiment(config_path)["runs"][0]
    original_run_id = original["run_id"]
    baseline_id = config_loader._stable_run_id(
        original["config_name"], original["pair_index"], original["dataset"],
        original["protocol"], original["teacher"], original["student"],
        original["grid_index"],
        {key: value for key, value in original["params"].items()
         if key != "teacher_seed"},
        original["training"], variant=original.get("variant"))
    captured = {}

    def no_training_run(args, loaded):
        captured["args"] = args
        captured["loaded"] = loaded
        return 0

    monkeypatch.setattr(config_runner_impl, "run", no_training_run)
    result = config_runner_impl.main([
        "--config", str(config_path), "--datasets", "BNCI2014001",
        "--seeds", "666", "--teachers", "cbramod", "--students", "ifnet",
        "--teacher-seed", "0",
    ])
    assert result == 0
    spec = captured["loaded"]["runs"][0]
    assert spec["seeds"] == [666]
    assert spec["params"]["teacher_seed"] == 0
    assert captured["loaded"]["config"]["runtime_selection"][
        "teacher_seed_policy"] == "cli_override"
    assert spec["run_id"] != original_run_id
    assert config_runner_impl._condition_run_id(spec, "Base") == baseline_id
    manifest = config_runner_impl._build_manifest([spec])
    assert [row["teacher_seed"] for row in manifest if row["condition"] == "Base"] == [""]
    assert [row["teacher_seed"] for row in manifest if row["condition"] == "KD_all"] == [0]


@pytest.mark.parametrize(
    "params, expected_teacher_seed",
    [({"teacher_seed": 0}, 0), ({}, 666)],
)
def test_config_cache_seed_is_independent_from_student_training_seed(
        monkeypatch, params, expected_teacher_seed):
    target_calls = []
    training_calls = []
    _patch_config_cell(monkeypatch, target_calls, training_calls)
    rows, errors = config_runner_impl._run_spec(
        _spec(params=params), "cpu", set(), False, gpu=None)
    assert errors == []
    assert target_calls == [("BNCI2014001", "cbramod", expected_teacher_seed, 0)]
    assert training_calls == [(666, "KD_all")]
    assert rows[0]["seed"] == 666
    assert rows[0]["teacher_seed"] == expected_teacher_seed


def test_resume_rejects_a_completed_row_from_another_teacher_seed(monkeypatch):
    spec = _spec(params={"teacher_seed": 0})
    row = {
        "run_id": spec["run_id"], "dataset": spec["dataset"],
        "teacher": spec["teacher"], "student": spec["student"],
        "key": 0, "seed": 666, "method": "KD_all",
        "teacher_seed": "1", "failure_status": "complete",
    }
    with pytest.raises(ValueError, match="teacher_seed recorded=1, current=0"):
        config_runner_impl._verify_resume_data_identity(
            [spec], [row], {config_runner_impl._row_key(row)})


def test_legacy_same_seed_resume_infers_missing_teacher_seed(monkeypatch):
    spec = _spec()
    student_data = _student_data()
    monkeypatch.setattr(
        config_runner_impl.loso_source, "load_dataset",
        lambda *_args, **_kwargs: student_data)
    cache_calls = []

    def load_targets(dataset, teacher, teacher_seed, fold, **kwargs):
        cache_calls.append(teacher_seed)
        return _targets()

    monkeypatch.setattr(
        config_runner_impl.legacy, "_load_or_ensure_loso_teacher_targets",
        load_targets)
    row = {
        "run_id": spec["run_id"], "dataset": spec["dataset"],
        "teacher": spec["teacher"], "student": spec["student"],
        "key": 0, "seed": 666, "method": "KD_all",
        "failure_status": "complete",
        "source_manifest_sha256": "source-sha",
        "student_input_manifest_sha256": "student-input-sha",
        "student_input_policy_sha256": "student-policy",
        "teacher_input_manifest_sha256": "teacher-input-sha",
        "teacher_input_policy_sha256": "teacher-policy",
        "teacher_target_manifest_sha256": "target-manifest-sha",
        "teacher_checkpoint_sha256": "teacher-checkpoint-sha",
    }
    config_runner_impl._verify_resume_data_identity(
        [spec], [row], {config_runner_impl._row_key(row)})
    assert cache_calls == [666]


def test_explicit_teacher_seed_is_rejected_for_fewshot():
    with pytest.raises(ValueError, match="only supported with protocol=loso"):
        run_distill._validate_teacher_seed_protocol("fewshot", 0)
    loaded = {"runs": [_spec(protocol="fewshot", params={"teacher_seed": 0})]}
    with pytest.raises(ValueError, match="only supported with protocol=loso"):
        config_runner_impl._validate_teacher_seed_scope(loaded)


def test_direct_loso_cache_uses_teacher_seed_but_trains_with_student_seed(
        tmp_path, monkeypatch):
    target_calls = []
    training_calls = []
    student_data = _student_data()
    monkeypatch.setattr(run_distill, "_device", lambda _gpu: "cpu")
    monkeypatch.setattr(run_distill, "_set_thread_defaults", lambda: None)
    monkeypatch.setattr(
        run_distill.config, "load_dataset_config",
        lambda _dataset: {
            "num_subjects": 2, "num_classes": 2, "seeds": [666],
            "loso": {"seeds": [0, 1, 2]}, "val_split": 0.2,
        })
    monkeypatch.setattr(
        run_distill.loso_source, "load_dataset",
        lambda *_args, **_kwargs: student_data)

    def load_targets(dataset, teacher, teacher_seed, fold, **kwargs):
        target_calls.append((dataset, teacher, teacher_seed, fold))
        return _targets()

    monkeypatch.setattr(run_distill, "_load_or_ensure_loso_teacher_targets", load_targets)
    monkeypatch.setattr(
        run_distill, "_check_teacher", lambda *_args, **_kwargs: {
            "labels_match": True, "uid_set_match": True,
        })
    monkeypatch.setattr(
        run_distill, "_artifact_info",
        lambda path: {"path": str(path), "sha256": "teacher-artifact-sha"})

    def run_method(args, dataset, protocol, teacher, student, artifact, fold,
                   student_seed, method, *rest, **kwargs):
        training_calls.append((student_seed, method))
        return {
            "dataset": dataset, "teacher": teacher, "student": student,
            "seed": student_seed, "method": method,
            "acc": 50.0, "kappa": 0.0,
        }

    monkeypatch.setattr(run_distill, "_run_method", run_method)
    output = tmp_path / "direct.csv"
    args = run_distill.parse_args([
        "--protocol", "loso", "--dataset", "BNCI2014001",
        "--teacher", "cbramod", "--student", "ifnet",
        "--keys", "0", "--seeds", "666", "--teacher-seed", "0",
        "--methods", "kd", "--out_csv", str(output), "--fail_fast",
    ])
    assert run_distill._run(args) == 0
    assert target_calls == [("BNCI2014001", "cbramod", 0, 0)]
    assert training_calls == [(666, "KD_all")]
    with output.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["seed"] == "666"
    assert row["teacher_seed"] == "0"


def test_teacher_seed_must_be_nonnegative():
    with pytest.raises(ValueError, match="non-negative"):
        run_distill._validate_teacher_seed_protocol("loso", -1)
    with pytest.raises(config_loader.ExperimentConfigError, match="integer"):
        config_loader._validate_param_value("teacher_seed", 0.5, "test")
    with pytest.raises(config_loader.ExperimentConfigError, match=">= 0"):
        config_loader._validate_param_value("teacher_seed", -1, "test")


def test_manifest_records_teacher_seed_and_selected_target_source(monkeypatch):
    spec = _spec(params={"teacher_seed": 0}, methods=["Base", "KD_all"])
    manifest = config_runner_impl._build_manifest([spec])
    kd_item = next(item for item in manifest if item["condition"] == "KD_all")
    row = {
        **kd_item,
        "method": "KD_all",
        "artifact_run_id": kd_item["artifact_run_id"],
        "teacher_seed": 0,
        "teacher_artifact_path": "/tmp/targets/teacher.npz",
        "teacher_artifact_sha256": "teacher-artifact-sha",
        "teacher_target_manifest_path": "/tmp/targets/manifest.json",
        "teacher_target_manifest_sha256": "target-manifest-sha",
        "teacher_checkpoint_sha256": "teacher-checkpoint-sha",
        "teacher_input_manifest_sha256": "teacher-input-sha",
        "teacher_input_policy_sha256": "teacher-policy",
    }
    config_runner_impl._update_manifest(manifest, row, "complete")
    updated = next(item for item in manifest if item["condition"] == "KD_all")
    assert updated["teacher_seed"] == 0
    assert updated["teacher_target_manifest_sha256"] == "target-manifest-sha"
    assert updated["teacher_checkpoint_sha256"] == "teacher-checkpoint-sha"
    fields = {}
    monkeypatch.setattr(
        config_runner_impl, "_atomic_write_csv",
        lambda _path, _rows, fieldnames: fields.setdefault("names", fieldnames))
    config_runner_impl._write_manifest("/tmp/unused-manifest.csv", manifest)
    assert "teacher_seed" in fields["names"]
    assert "teacher_target_manifest_path" in fields["names"]


def test_config_params_teacher_seed_changes_kd_id_but_not_base_control_id():
    resolved_specs = []
    for teacher_seed in (0, 1):
        spec = _spec(params={"temperature": 1.0, "lam_kd": 0.5,
                             "teacher_seed": teacher_seed},
                     methods=["Base", "KD_all"])
        spec["run_id"] = config_loader._stable_run_id(
            spec["config_name"], spec["pair_index"], spec["dataset"],
            spec["protocol"], spec["teacher"], spec["student"],
            spec["grid_index"], spec["params"], spec["training"], variant=None)
        resolved = config_runner_impl._apply_teacher_seed_override(
            {"config": {}, "runs": [spec]}, None)
        resolved_specs.append(resolved["runs"][0])

    assert resolved_specs[0]["run_id"] != resolved_specs[1]["run_id"]
    assert (config_runner_impl._condition_run_id(resolved_specs[0], "Base")
            == config_runner_impl._condition_run_id(resolved_specs[1], "Base"))
    assert (config_runner_impl._condition_run_id(resolved_specs[0], "KD_all")
            != config_runner_impl._condition_run_id(resolved_specs[1], "KD_all"))

