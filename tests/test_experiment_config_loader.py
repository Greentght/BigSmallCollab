from itertools import product
from pathlib import Path

import pytest
import yaml

from experiments.config_loader import (
    ExperimentConfigError,
    load_experiment,
    resolved_path,
)
ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIG_DIR = ROOT / "configs" / "experiments"
DISTILL_PAIRS = set(product(
    ("mirepnet", "cbramod"), ("ifnet", "eegnet", "adfcnn")))
FUSION_PAIRS = {
    ("mirepnet", "ifnet"), ("mirepnet", "eegnet"),
    ("mirepnet", "adfcnn"), ("cbramod", "ifnet"),
    ("cbramod", "eegnet"),
}


def test_distill_pair_grid_and_controls_expand():
    loaded = load_experiment(ROOT / "configs/experiments/distill_kd.yaml")
    # Five datasets x two protocols x six pairs; two pair grids have 9 and 3 cells.
    assert len(loaded["runs"]) == 5 * 2 * (9 + 1 + 1 + 3 + 1 + 1)
    assert {run["config_name"] for run in loaded["runs"]} == {"distill_kd"}
    assert len({run["run_id"] for run in loaded["runs"]}) == len(loaded["runs"])
    assert sum("Base" in run["methods"] for run in loaded["runs"]) == 5 * 2 * 6
    ifnet = next(run for run in loaded["runs"]
                 if run["dataset"] == "BNCI2014001"
                 and run["protocol"] == "fewshot"
                 and run["teacher"] == "mirepnet"
                 and run["student"] == "ifnet"
                 and run["grid_index"] == 0)
    assert ifnet["methods"] == ["Base", "KD_all"]
    assert ifnet["training"] == {
        "epochs": 100, "lr": 0.001,
        "weight_decay": 0.01, "batch_size": 16,
    }
    assert ifnet["params"]["temperature"] == 1.0
    assert ifnet["params"]["lam_kd"] == 0.25


def test_fusion_pair_grid_and_controls_expand():
    loaded = load_experiment(ROOT / "configs/experiments/fusion_concat_mlp.yaml")
    # Four datasets x two protocols x (9 + four non-grid pairs).
    assert len(loaded["runs"]) == 4 * 2 * (9 + 4)
    assert sum("big_only" in run["methods"] for run in loaded["runs"]) == 4 * 2 * 5
    run = next(item for item in loaded["runs"]
               if item["dataset"] == "BNCI2014001-4"
               and item["protocol"] == "fewshot"
               and item["big_model"] == "mirepnet"
               and item["small_model"] == "ifnet"
               and item["grid_index"] == 0)
    assert run["methods"] == ["big_only", "small_only", "concat_mlp"]
    assert run["training"] == {
        "epochs": 100, "lr": 0.001,
        "weight_decay": 0.0001, "batch_size": 32,
    }
    assert run["params"]["hidden"] == 0
    assert run["params"]["dropout"] == 0.0


def test_dataset_override_has_priority_over_pair_and_grid(tmp_path):
    config = tmp_path / "override.yaml"
    config.write_text(
        """
name: override
type: distill
method: kd
datasets: [BNCI2014004]
protocols: [fewshot]
seeds: [666]
subjects: [0]
include_baseline: false
artifact_root: results/artifacts
output_dir: results/distill
pairs:
  - teacher: cbramod
    student: ifnet
    params:
      temperature: 2.0
      lam_kd: 0.25
      epochs: 100
      lr: 0.001
      weight_decay: 0.01
      batch_size: 16
    grid:
      lam_kd: [0.1, 0.2]
    datasets:
      BNCI2014004:
        params:
          temperature: 3.0
          epochs: 50
        grid:
          lam_kd: [0.15, 0.25]
""")
    loaded = load_experiment(config)
    assert len(loaded["runs"]) == 2
    for run in loaded["runs"]:
        assert run["methods"] == ["KD_all"]
        assert run["params"]["temperature"] == 3.0
        assert run["training"]["epochs"] == 50
    assert {run["params"]["lam_kd"] for run in loaded["runs"]} == {0.15, 0.25}


def test_unknown_field_and_illegal_value_are_rejected(tmp_path):
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(
        """
name: bad
type: distill
method: kd
datasets: [BNCI2014004]
protocols: [fewshot]
extra: true
pairs: []
""")
    with pytest.raises(ExperimentConfigError, match="unknown field"):
        load_experiment(unknown)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        """
name: bad
type: fusion
method: avg_prob
datasets: [BNCI2014004]
protocols: [fewshot]
output_dir: results/fusion
pairs:
  - big: mirepnet
    small: ifnet
    params: {big_weight: 2.0}
    grid: {}
""")
    with pytest.raises(ExperimentConfigError, match="big_weight"):
        load_experiment(invalid)


def test_controls_can_be_disabled_and_resolved_output_is_sibling_yaml(tmp_path):
    config = tmp_path / "controls.yaml"
    config.write_text(
        """
name: controls
type: fusion
method: avg_prob
datasets: [BNCI2014004]
protocols: [fewshot]
include_controls: false
output_dir: results/fusion
pairs:
  - big: mirepnet
    small: ifnet
    params: {}
    grid: {}
""")
    loaded = load_experiment(config)
    assert loaded["runs"][0]["methods"] == ["avg_prob"]
    assert resolved_path("results/fusion/fusion_avg_prob.csv").name == "fusion_avg_prob.resolved.yaml"


def test_shipped_configs_are_one_method_with_every_selected_pair_unique():
    paths = sorted(EXPERIMENT_CONFIG_DIR.glob("*.yaml"))
    assert {path.stem for path in paths} == {
        "distill_kd", "distill_kd_masked", "distill_mmd", "distill_kd_mmd", "distill_mi",
        "fusion_avg_prob", "fusion_concat_mlp", "fusion_gate_conf_acc",
    }
    methods = set()
    for path in paths:
        raw = yaml.safe_load(path.read_text())
        method_key = (raw["type"], raw["method"])
        assert method_key not in methods
        methods.add(method_key)
        left, right = (("teacher", "student") if raw["type"] == "distill"
                       else ("big", "small"))
        expected_pairs = ({("mirepnet", "ifnet")} if raw["method"] == "mi"
                          else DISTILL_PAIRS if raw["type"] == "distill"
                          else FUSION_PAIRS)
        assert {(pair[left], pair[right]) for pair in raw["pairs"]} == expected_pairs
        assert len(raw["pairs"]) == len(expected_pairs)
        assert all("params" in pair and "grid" in pair for pair in raw["pairs"])
        assert all("training" not in pair for pair in raw["pairs"])


def test_probability_mi_config_is_subjectwise_and_has_exact_three_conditions():
    loaded = load_experiment(ROOT / "configs/experiments/distill_mi.yaml")
    assert len(loaded["runs"]) == 5  # one pair x five datasets x fewshot
    assert {run["protocol"] for run in loaded["runs"]} == {"fewshot"}
    assert all(run["subjects"] is None for run in loaded["runs"])
    assert all(run["methods"] == ["Base", "KD_all", "CE_MI"]
               for run in loaded["runs"])
    assert all(run["params"] == {"lam_mi": 0.1} for run in loaded["runs"])


def test_probability_mi_rejects_loso_or_subject_subset(tmp_path):
    base = """
name: bad_mi
type: distill
method: mi
datasets: [BNCI2015001]
protocols: {protocols}
subjects: {subjects}
include_baseline: true
pairs:
  - teacher: mirepnet
    student: ifnet
    params: {{lam_mi: 0.1}}
    grid: {{}}
"""
    cases = {
        "loso": ("[fewshot, loso]", "null"),
        "subset": ("[fewshot]", "[0]"),
    }
    for name, (protocols, subjects) in cases.items():
        path = tmp_path / f"bad_{name}.yaml"
        path.write_text(base.format(protocols=protocols, subjects=subjects))
        with pytest.raises(ExperimentConfigError):
            load_experiment(path)


def test_training_values_live_in_params_and_controls_dedupe_correctly(tmp_path):
    fusion_config = tmp_path / "fusion_training_grid.yaml"
    fusion_config.write_text(
        """
name: fusion_training_grid
type: fusion
method: concat_mlp
datasets: [BNCI2014004]
protocols: [fewshot]
pairs:
  - big: mirepnet
    small: ifnet
    params: {hidden: 32, epochs: 1}
    grid: {epochs: [1, 2]}
""")
    fusion_runs = load_experiment(fusion_config)["runs"]
    assert [run["training"]["epochs"] for run in fusion_runs] == [1, 2]
    assert sum("big_only" in run["methods"] for run in fusion_runs) == 1
    assert sum("small_only" in run["methods"] for run in fusion_runs) == 1

    distill_config = tmp_path / "distill_training_grid.yaml"
    distill_config.write_text(
        """
name: distill_training_grid
type: distill
method: kd
datasets: [BNCI2014004]
protocols: [fewshot]
pairs:
  - teacher: mirepnet
    student: ifnet
    params: {epochs: 1}
    grid: {epochs: [1, 2]}
""")
    distill_runs = load_experiment(distill_config)["runs"]
    assert sum("Base" in run["methods"] for run in distill_runs) == 2


def test_duplicate_pair_duplicate_training_and_irrelevant_params_are_rejected(tmp_path):
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        """
name: duplicate
type: fusion
method: avg_prob
datasets: [BNCI2014004]
protocols: [fewshot]
pairs:
  - {big: mirepnet, small: ifnet, params: {}, grid: {}}
  - {big: mirepnet, small: ifnet, params: {}, grid: {}}
""")
    with pytest.raises(ExperimentConfigError, match="duplicates"):
        load_experiment(duplicate)

    duplicate_training = tmp_path / "duplicate_training.yaml"
    duplicate_training.write_text(
        """
name: duplicate_training
type: distill
method: kd
datasets: [BNCI2014004]
protocols: [fewshot]
pairs:
  - teacher: mirepnet
    student: ifnet
    params: {epochs: 2}
    training: {epochs: 1}
    grid: {}
""")
    with pytest.raises(ExperimentConfigError, match="specified twice"):
        load_experiment(duplicate_training)

    irrelevant = tmp_path / "irrelevant.yaml"
    irrelevant.write_text(
        """
name: irrelevant
type: distill
method: mmd
datasets: [BNCI2014004]
protocols: [fewshot]
pairs:
  - teacher: mirepnet
    student: ifnet
    params: {lam_kd: 0.5}
    grid: {}
""")
    with pytest.raises(ExperimentConfigError, match="unknown field"):
        load_experiment(irrelevant)
