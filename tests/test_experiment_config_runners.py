import pytest

from experiments.config_loader import ExperimentConfigError
from experiments.distill import config_runner_impl as distill_runner
from experiments.fusion import config_runner_impl as fusion_runner


def _loaded(kind):
    return {
        "source_path": f"{kind}.yaml",
        "config": {"type": kind, "name": kind},
        "runs": [],
    }


@pytest.mark.parametrize("runner", [distill_runner, fusion_runner])
def test_runner_force_and_resume_are_mutually_exclusive(runner):
    with pytest.raises(SystemExit):
        runner.parse_args([
            "--config", "config.yaml", "--force", "--resume",
        ])


def test_runner_row_keys_normalize_csv_and_runtime_values():
    distill_row = {
        "run_id": "r", "dataset": "d", "teacher": "big",
        "student": "small", "key": 2, "seed": 666, "method": "KD_all",
    }
    fusion_row = {
        "run_id": "r", "dataset": "d", "big": "big", "small": "small",
        "key": 2, "seed": 666, "method": "avg_prob",
    }
    assert distill_runner._row_key(distill_row) == (
        "r", "d", "big", "small", "2", "666", "KD_all",
    )
    assert fusion_runner._row_key(fusion_row) == (
        "r", "d", "big", "small", "2", "666", "avg_prob",
    )


def test_wrong_config_type_is_rejected_before_runner_side_effects(monkeypatch):
    monkeypatch.setattr(
        distill_runner.config_loader, "load_experiment",
        lambda _path: _loaded("fusion"),
    )
    with pytest.raises(ExperimentConfigError, match="distill runner"):
        distill_runner.main(["--config", "unused.yaml"])

    monkeypatch.setattr(
        fusion_runner.config_loader, "load_experiment",
        lambda _path: _loaded("distill"),
    )
    with pytest.raises(ExperimentConfigError, match="fusion runner"):
        fusion_runner.main(["--config", "unused.yaml"])


def test_fusion_resume_executes_only_missing_methods(monkeypatch):
    calls = []
    monkeypatch.setattr(
        fusion_runner.legacy.config, "load_dataset_config",
        lambda _dataset: {"num_subjects": 1, "seeds": [666]},
    )
    monkeypatch.setattr(
        fusion_runner.legacy, "_run_cell",
        lambda _args, _protocol, methods, _device, _big_artifact,
               _small_artifact, _policy, _key, _seed: calls.append(methods) or [
                   {
                       "run_id": "ignored", "dataset": "BNCI2014004",
                       "big": "mirepnet", "small": "ifnet", "key": 0,
                       "seed": 666, "method": method, "acc": 50.0,
                       "kappa": 0.0,
                   }
                   for method in methods
               ],
    )
    spec = {
        "config_name": "fusion_avg_prob",
        "run_id": "run-1",
        "grid_index": 0,
        "dataset": "BNCI2014004",
        "protocol": "fewshot",
        "teacher": "mirepnet",
        "student": "ifnet",
        "big_model": "mirepnet",
        "small_model": "ifnet",
        "subjects": [0],
        "seeds": [666],
        "methods": ["big_only", "small_only", "avg_prob"],
        "runtime_method": "avg_prob",
        "params": {
            "big_temperature": 1.0,
            "small_temperature": 1.0,
            "big_weight": 0.5,
        },
        "training": {
            "epochs": 1,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 8,
        },
        "artifact_root": "results/artifacts",
    }
    existing = {
        ("run-1", "BNCI2014004", "mirepnet", "ifnet", "0", "666", "big_only")
    }
    rows, errors = fusion_runner._run_spec(
        spec, "cpu", existing, fail_fast=True)
    assert not errors
    assert calls == [["small_only", "avg_prob"]]
    assert [row["method"] for row in rows] == ["small_only", "avg_prob"]
