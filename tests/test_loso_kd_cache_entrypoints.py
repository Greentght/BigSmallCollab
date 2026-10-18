"""CPU-only checks for LOSO KD cache entrypoint behavior.

All cache producers and training functions are mocked; these tests never read
or write project caches and never start a training subprocess.
"""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from data import loso_source
from experiments.distill import config_runner_impl, run_distill


def test_legacy_cli_cache_flag_and_gpu_are_passed_to_ensure(monkeypatch):
    args = run_distill.parse_args([
        "--protocol", "loso", "--no_auto_teacher_cache",
    ])
    assert args.no_auto_teacher_cache
    assert args.gpu is None
    assert args.methods == ["kd"]
    assert run_distill._parse_methods(args.methods) == ["KD_all"]

    calls = []
    producer = ModuleType("experiments.distill.loso_target_cache")
    producer.ensure_teacher_targets = lambda *args, **kwargs: (
        calls.append((args, kwargs)) or {"strict": True})
    monkeypatch.setitem(sys.modules, producer.__name__, producer)
    student_data = object()
    result = run_distill._load_or_ensure_loso_teacher_targets(
        "BNCI2015001", "cbramod", 666, 2, student_data, "/cache",
        gpu=args.gpu)

    assert result == {"strict": True}
    assert calls == [(("BNCI2015001", "cbramod", 666, 2, student_data),
                      {"cache_root": "/cache", "gpu": None})]


def test_read_only_flag_uses_strict_loader_without_importing_producer(monkeypatch):
    calls = []
    monkeypatch.setattr(
        loso_source, "load_teacher_targets",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"cached": True})
    student_data = object()

    result = run_distill._load_or_ensure_loso_teacher_targets(
        "BNCI2015001", "cbramod", 666, 0, student_data, "/cache",
        gpu=3, no_auto_teacher_cache=True)

    assert result == {"cached": True}
    assert calls == [(("BNCI2015001", "cbramod", 666, 0),
                      {"student_data": student_data, "cache_root": "/cache"})]


def test_config_cli_passes_gpu_and_no_auto_flag_into_runner_namespace():
    args = config_runner_impl.parse_args([
        "--config", "unused.yaml", "--teacher", "cbramod",
        "--student", "ifnet", "--gpu", "2", "--no-auto-teacher-cache",
    ])
    assert args.teachers == ["cbramod"]
    assert args.students == ["ifnet"]
    assert args.no_auto_teacher_cache

    namespace = config_runner_impl._namespace({
        "dataset": "BNCI2015001", "protocol": "loso",
        "teacher": "cbramod", "student": "ifnet", "methods": ["KD_all"],
        "params": {}, "training": {
            "epochs": 1, "lr": 0.001, "weight_decay": 0.0,
            "batch_size": 2,
        },
    }, "/cache", no_auto_teacher_cache=args.no_auto_teacher_cache,
       gpu=args.gpu)
    assert namespace.no_auto_teacher_cache
    assert namespace.gpu == 2


def test_config_spec_forwards_cache_policy_and_selected_gpu(monkeypatch):
    student_data = SimpleNamespace(
        x=np.ones((4, 1, 8), dtype=np.float32),
        y=np.asarray([0, 1, 0, 1]),
        subjects=np.asarray([0, 0, 1, 1]),
        sample_uid=np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]]),
        source_manifest_sha256="source-sha",
        input_manifest_sha256="student-input-sha",
        identity={"input_policy_sha256": "student-policy-sha"},
        source_path=Path("fixture-source"), input_manifest_path=None,
    )
    monkeypatch.setattr(
        config_runner_impl.legacy.config, "load_dataset_config",
        lambda _dataset: {"num_subjects": 2, "num_classes": 2,
                          "seeds": [666], "val_split": 0.2})
    monkeypatch.setattr(loso_source, "load_dataset",
                        lambda *_args, **_kwargs: student_data)
    monkeypatch.setattr(config_runner_impl.legacy, "_expected_split_policy",
                        lambda _protocol: "fixture-policy")
    calls = []
    monkeypatch.setattr(
        config_runner_impl.legacy, "_load_or_ensure_loso_teacher_targets",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {
            "_artifact_path": "fixture-teacher.npz"})
    monkeypatch.setattr(config_runner_impl.legacy, "_check_teacher",
                        lambda *_args, **_kwargs: {"uid_set_match": True})
    monkeypatch.setattr(config_runner_impl.legacy, "_artifact_info",
                        lambda path: {"path": path, "sha256": "teacher-sha"})
    monkeypatch.setattr(
        config_runner_impl.legacy, "_run_method",
        lambda *_args, **_kwargs: ({
            "method": "KD_all", "failure_status": "complete",
            "acc": 50.0, "test_balanced_accuracy": 50.0, "kappa": 0.0,
        }, {}))

    spec = {
        "config_name": "fixture", "run_id": "fixture-run", "grid_index": 0,
        "dataset": "BNCI2015001", "protocol": "loso", "teacher": "cbramod",
        "student": "eegnet", "subjects": [0], "seeds": [666],
        "method": "kd", "methods": ["KD_all"], "params": {},
        "training": {"epochs": 1, "lr": 0.001, "weight_decay": 0.0,
                     "batch_size": 2},
        "artifact_root": "/cache",
    }
    rows, errors = config_runner_impl._run_spec(
        spec, "cpu", set(), True, no_auto_teacher_cache=True, gpu=2)

    assert not errors
    assert len(rows) == 1
    assert calls[0][1]["no_auto_teacher_cache"] is True
    assert calls[0][1]["gpu"] == 2


def test_model_filter_keeps_requested_pair_and_one_variant_control():
    loaded = {
        "config": {
            "method": "kd", "variants": {"kd": {}, "mmd": {}},
            "type": "distill", "include_baseline": True,
        },
        "runs": [
            {"teacher": "mirepnet", "student": "ifnet", "pair_index": 0,
             "variant": "kd", "runtime_method": "KD_all", "methods": ["Base"],
             "dataset": "BNCI2015001", "protocol": "loso",
             "training": {"epochs": 1}},
            {"teacher": "cbramod", "student": "ifnet", "pair_index": 1,
             "variant": "kd", "runtime_method": "KD_all", "methods": ["Base"],
             "dataset": "BNCI2015001", "protocol": "loso",
             "training": {"epochs": 1}},
            {"teacher": "cbramod", "student": "ifnet", "pair_index": 1,
             "variant": "mmd", "runtime_method": "MMD", "methods": ["Base"],
             "dataset": "BNCI2015001", "protocol": "loso",
             "training": {"epochs": 1}},
            {"teacher": "cbramod", "student": "eegnet", "pair_index": 2,
             "variant": "kd", "runtime_method": "KD_all", "methods": ["Base"],
             "dataset": "BNCI2015001", "protocol": "loso",
             "training": {"epochs": 1}},
        ],
    }
    selected = config_runner_impl._select_model_runs(
        loaded, teachers=["cbramod"], students=["ifnet"])

    assert [(run["teacher"], run["student"]) for run in selected["runs"]] == [
        ("cbramod", "ifnet"), ("cbramod", "ifnet")]
    assert [run["methods"] for run in selected["runs"]] == [
        ["Base", "KD_all"], ["MMD"]]
    assert selected["config"]["runtime_selection"] == {
        "teachers": ["cbramod"], "students": ["ifnet"]}

    with pytest.raises(ValueError, match="pass both --teachers and --students"):
        config_runner_impl._select_model_runs(loaded, teachers=["cbramod"])
    with pytest.raises(ValueError, match="pairs outside this config"):
        config_runner_impl._select_model_runs(
            loaded, teachers=["mirepnet"], students=["eegnet"])


def test_resume_rejects_regenerated_target_with_different_manifest(monkeypatch):
    student_data = SimpleNamespace(
        source_manifest_sha256="source-sha",
        input_manifest_sha256="student-input-sha",
        identity={"input_policy_sha256": "student-policy-sha"},
    )
    targets = {
        "_input_manifest_sha256": "teacher-input-sha",
        "_input_policy_sha256": "teacher-policy-sha",
        "_manifest_sha256": "new-target-sha",
        "_manifest": {"teacher_checkpoint_sha256": "checkpoint-sha"},
    }
    monkeypatch.setattr(
        loso_source, "load_dataset", lambda *_args, **_kwargs: student_data)
    monkeypatch.setattr(
        config_runner_impl.legacy.config, "load_dataset_config",
        lambda _dataset: {"num_subjects": 1, "seeds": [666],
                          "loso": {"seeds": [666]}})
    ensure_calls = []
    monkeypatch.setattr(
        config_runner_impl.legacy, "_load_or_ensure_loso_teacher_targets",
        lambda *args, **kwargs: ensure_calls.append((args, kwargs)) or targets)

    spec = {
        "dataset": "BNCI2015001", "protocol": "loso", "subjects": [0],
        "seeds": [666], "methods": ["KD_all"], "teacher": "cbramod",
        "student": "eegnet", "artifact_root": "/cache", "run_id": "run",
    }
    row = {
        "run_id": "run", "dataset": "BNCI2015001", "teacher": "cbramod",
        "student": "eegnet", "key": 0, "seed": 666, "method": "KD_all",
        "failure_status": "complete", "source_manifest_sha256": "source-sha",
        "student_input_manifest_sha256": "student-input-sha",
        "student_input_policy_sha256": "student-policy-sha",
        "teacher_input_manifest_sha256": "teacher-input-sha",
        "teacher_input_policy_sha256": "teacher-policy-sha",
        "teacher_target_manifest_sha256": "old-target-sha",
        "teacher_checkpoint_sha256": "checkpoint-sha",
    }
    row_key = config_runner_impl._row_key(row)

    with pytest.raises(ValueError, match="teacher_target_manifest_sha256"):
        config_runner_impl._verify_resume_data_identity(
            [spec], [row], {row_key}, gpu=2)
    assert ensure_calls[0][1]["gpu"] == 2


def test_config_runner_applies_default_variants_without_runtime_filters(
        monkeypatch, tmp_path):
    source = {
        "source_path": str(tmp_path / "config.yaml"),
        "config": {"type": "distill", "method": "kd",
                   "variants": {"kd": {}, "mmd": {}},
                   "default_variants": ["kd"]},
        "runs": [{"output": str(tmp_path / "report.csv"),
                  "config_name": "fixture"}],
    }
    monkeypatch.setattr(config_runner_impl.config_loader, "load_experiment",
                        lambda _path: source)
    selections = []

    def select(loaded, datasets=None, seeds=None, variants=None):
        selections.append((datasets, seeds, variants))
        selected = dict(loaded)
        selected["runs"] = list(loaded["runs"])
        return selected

    monkeypatch.setattr(config_runner_impl.config_loader,
                        "select_experiment_runs", select)
    captured = []
    monkeypatch.setattr(config_runner_impl, "run",
                        lambda args, loaded: captured.append(loaded) or 0)

    assert config_runner_impl.main(["--config", "unused.yaml"]) == 0
    assert selections == [(None, None, None)]
    assert captured == [source]


def test_config_runner_does_not_apply_default_variants_to_formal_mi(
        monkeypatch, tmp_path):
    source = {
        "source_path": str(tmp_path / "config.yaml"),
        "config": {"type": "distill", "method": "mi",
                   "variants": None, "default_variants": None},
        "runs": [{"output": str(tmp_path / "report.csv"),
                  "config_name": "fixture"}],
    }
    monkeypatch.setattr(config_runner_impl.config_loader, "load_experiment",
                        lambda _path: source)
    monkeypatch.setattr(
        config_runner_impl.config_loader, "select_experiment_runs",
        lambda *_args, **_kwargs: pytest.fail("formal MI must keep fixed scope"))
    captured = []
    monkeypatch.setattr(config_runner_impl, "run",
                        lambda args, loaded: captured.append(loaded) or 0)

    assert config_runner_impl.main(["--config", "unused.yaml"]) == 0
    assert captured == [source]
