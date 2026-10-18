"""Keep legacy 001 teacher reuse strict about its data and training recipe."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data import loso_source
from experiments.finetune import run_loso_config_alignment as reference


@pytest.fixture
def teacher_identity(monkeypatch, tmp_path):
    recipe = {"epochs": 20, "lr": 0.001, "optimizer": "adamw"}
    current = {"lr": 0.001, "feature_head": "flatten", "in_channels": 22}
    files = {name: {"bytes": 10, "sha256": name} for name in
             ("X.npy", "y.npy", "subjects.npy", "trials.csv")}
    data = SimpleNamespace(
        x=np.zeros((4, 22, 4, 200), dtype=np.float32),
        input_manifest_sha256="input-sha", input_manifest={"files": files})
    manifest = {
        "profile": "wideband_npy_v3", "input_shape": [22, 4, 200],
        "input_manifest_sha256": "input-sha", "input_files": dict(files),
        "resolved_training_config": dict(recipe),
    }
    spec = tmp_path / "protocol.yaml"
    spec.write_text("{}\n")
    monkeypatch.setattr(reference, "SPEC", spec)
    monkeypatch.setattr(reference, "profile_config", lambda *_args: dict(recipe))
    monkeypatch.setattr(loso_source.config, "load_model_config",
                        lambda *_args: dict(current))
    monkeypatch.setattr(loso_source, "_validate_baseline_manifest", lambda *_a, **_k: None)
    monkeypatch.setattr(loso_source, "_validate_cached_input_policy", lambda *_a: None)
    return manifest, data, current


def validate(manifest, data, teacher="cbramod"):
    return loso_source.validated_teacher_model_config(
        manifest, Path("result.json"), data, "BNCI2014001", teacher, 0, 0)


def test_known_001_recipe_is_reconstructed_before_reuse(teacher_identity):
    manifest, data, current = teacher_identity
    assert validate(manifest, data) == current


@pytest.mark.parametrize("changed", ["recipe", "input_files", "input_shape", "profile"])
def test_stale_001_recipe_or_data_is_rejected(teacher_identity, changed):
    manifest, data, _ = teacher_identity
    if changed == "recipe":
        manifest["resolved_training_config"]["lr"] = 0.1
    elif changed == "input_files":
        manifest["input_files"]["y.npy"] = {"bytes": 10, "sha256": "wrong-labels"}
    elif changed == "input_shape":
        manifest["input_shape"] = [3, 4, 200]
    else:
        manifest["profile"] = "legacy_source"
    with pytest.raises(ValueError):
        validate(manifest, data)


def test_001_compatibility_does_not_accept_a_changed_teacher_architecture(teacher_identity):
    manifest, data, current = teacher_identity
    current.update(depth=7, emb_size=256)
    with pytest.raises(ValueError, match="architecture differs"):
        validate(manifest, data, teacher="mirepnet")


def test_full_model_config_still_requires_an_exact_current_match(teacher_identity):
    manifest, data, current = teacher_identity
    manifest["model_config"] = dict(current, lr=0.1)
    with pytest.raises(ValueError, match="model config differs"):
        validate(manifest, data)
