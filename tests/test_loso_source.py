"""Focused tests for current-wide LOSO identity and read-only consumers."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from data import loso_source
from experiments.config_loader import (
    ExperimentConfigError,
    load_experiment,
    select_experiment_runs,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _record(path):
    return {"bytes": Path(path).stat().st_size, "sha256": _sha(path)}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True))
    return _sha(path)


def _fixture(monkeypatch, tmp_path, cache_models=("cbramod", "eegnet"), order=(2, 0, 3, 1)):
    source_dir = tmp_path / "BNCI2015001" / "wide_fre_band"
    source_dir.mkdir(parents=True)
    classes = {"left_hand": 2, "feet": 0, "right_hand": 1}
    rows = []
    values = []
    for subject in (1, 2):
        for session in ("0A", "0B"):
            for name in ("feet", "right_hand"):
                rows.append({
                    "subject": subject, "session": session,
                    "class_name": name,
                    "trial_uid": f"fixture|s{subject}|{session}|{name}",
                    "label_id": classes[name], "source_row": len(rows),
                })
                values.append(np.full((1, 1024), len(rows), dtype=np.float32))
    trials_all = pd.DataFrame(rows)
    X = np.stack(values)
    labels = trials_all["class_name"].to_numpy(dtype=str)
    np.save(source_dir / "X.npy", X)
    np.save(source_dir / "labels.npy", labels)
    trials_all.to_csv(source_dir / "trials.csv", index=False)
    source_manifest = {
        "dataset": "BNCI2015001", "class_map": classes,
        "files": {name: _record(source_dir / name)
                  for name in ("X.npy", "labels.npy", "trials.csv")},
    }
    _write_json(source_dir / "manifest.json", source_manifest)
    result_root = tmp_path / "results"

    def load_loso(dataset):
        assert dataset == "BNCI2015001"
        return {
            "source_directory": str(source_dir),
            "native_sampling_rate_hz": 250,
            "selected_session": "session_A",
            "class_mapping": {"feet": 0, "right_hand": 1},
            "result_root": str(result_root),
        }

    monkeypatch.setattr(loso_source.config, "load_loso_dataset_config", load_loso)
    monkeypatch.setattr(loso_source.config, "load_dataset_config",
                        lambda _dataset: {"num_subjects": 2, "seeds": [666],
                                          "loso": {"seeds": [666]}})
    monkeypatch.setattr(loso_source.config, "load_model_config",
                        lambda model, *_args: {"in_channels": 1, "sample_rate": 250,
                                               "l_freq": 8.0, "h_freq": 32.0})
    monkeypatch.setattr(loso_source, "_input_policy_identity",
                        lambda _dataset, model: f"fixture-policy:{model}")
    monkeypatch.setattr(loso_source, "_validate_cached_input_policy",
                        lambda *_args: None)
    source_sha = _sha(source_dir / "manifest.json")
    selected = trials_all.query("session == '0A'").reset_index(drop=True)
    expected_y = selected["class_name"].map({"feet": 0, "right_hand": 1}).to_numpy()
    expected_subjects = selected["subject"].to_numpy() - 1
    cache_root = tmp_path / "cache"
    for model in cache_models:
        model_dir = cache_root / "model_inputs" / "BNCI2015001" / model
        model_dir.mkdir(parents=True)
        perm = np.asarray(order, dtype=np.int64)
        cache_trials = selected.iloc[perm][["trial_uid"]].copy()
        cache_trials["label_id"] = expected_y[perm]
        x_cache = np.stack([np.full((1, 1000), int(i + 10), dtype=np.float32)
                            for i in perm])
        np.save(model_dir / "X.npy", x_cache)
        np.save(model_dir / "y.npy", expected_y[perm].astype(np.int64))
        np.save(model_dir / "subjects.npy", expected_subjects[perm].astype(np.int64))
        cache_trials.to_csv(model_dir / "trials.csv", index=False)
        input_manifest = {
            "dataset": "BNCI2015001", "model": model,
            "source_manifest_sha256": source_sha,
            "class_mapping": {"feet": 0, "right_hand": 1},
            "input_shape": list(x_cache.shape[1:]), "trial_count": len(x_cache),
            "files": {name: _record(model_dir / name)
                      for name in ("X.npy", "y.npy", "subjects.npy", "trials.csv")},
        }
        _write_json(model_dir / "manifest.json", input_manifest)
    return source_dir, cache_root, result_root, selected, source_sha


def test_5001_session_classmap_and_cache_uid_order_are_canonical(monkeypatch, tmp_path):
    _, cache_root, _, selected, _ = _fixture(monkeypatch, tmp_path)
    data = loso_source.load_dataset("BNCI2015001", "eegnet", cache_root)
    assert data.y.tolist() == selected["class_name"].map(
        {"feet": 0, "right_hand": 1}).to_numpy()[[2, 0, 3, 1]].tolist()
    assert set(data.y.tolist()) == {0, 1}
    assert set(data.trial_uid.tolist()) == set(selected["trial_uid"])
    assert {uid: tuple(value) for uid, value in zip(data.trial_uid, data.sample_uid)} == {
        uid: (int(subject), int(index))
        for subject in (0, 1)
        for index, uid in enumerate(selected.loc[selected.subject == subject + 1,
                                                  "trial_uid"].tolist())
    }
    assert data.x.shape == (4, 1, 1000)
    assert data.class_mapping == {"feet": 0, "right_hand": 1}


def test_model_cache_duplicate_uid_and_tampered_hash_are_rejected(monkeypatch, tmp_path):
    _, cache_root, _, _, _ = _fixture(monkeypatch, tmp_path)
    data = loso_source.load_dataset("BNCI2015001", "eegnet", cache_root)
    model_dir = cache_root / "model_inputs" / "BNCI2015001" / "eegnet"
    x = np.load(model_dir / "X.npy")
    x[0, 0, 0] += 1
    np.save(model_dir / "X.npy", x)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        loso_source.load_dataset("BNCI2015001", "eegnet", cache_root)

    # Rebuild a clean fixture and make the manifest honestly describe duplicate IDs.
    _, cache_root, _, _, _ = _fixture(monkeypatch, tmp_path / "duplicate")
    model_dir = cache_root / "model_inputs" / "BNCI2015001" / "eegnet"
    trials = pd.read_csv(model_dir / "trials.csv")
    trials.loc[1, "trial_uid"] = trials.loc[0, "trial_uid"]
    trials.to_csv(model_dir / "trials.csv", index=False)
    manifest = json.loads((model_dir / "manifest.json").read_text())
    manifest["files"]["trials.csv"] = _record(model_dir / "trials.csv")
    _write_json(model_dir / "manifest.json", manifest)
    with pytest.raises(ValueError, match="duplicate trial_uid"):
        loso_source.load_dataset("BNCI2015001", "eegnet", cache_root)


def test_prediction_rows_align_by_physical_uid_and_validate_local_uid(monkeypatch, tmp_path):
    _, cache_root, _, _, _ = _fixture(monkeypatch, tmp_path)
    data = loso_source.load_dataset("BNCI2015001", "eegnet", cache_root)
    mask = data.subjects == 0
    uids = data.trial_uid[mask][::-1]
    y = data.y[mask][::-1]
    local = data.sample_uid[mask][::-1]
    logits = np.asarray([[20.0, 21.0], [10.0, 11.0]])
    aligned = loso_source._align_prediction_arrays(
        {"trial_uid": uids, "y": y, "local_sample_uid": local,
         "logits": logits}, data, 0, "test", Path("fixture.npz"))
    assert aligned["sample_uid"].tolist() == [[0, 0], [0, 1]]
    by_uid = dict(zip(uids.tolist(), logits.tolist()))
    assert aligned["logits"].tolist() == [by_uid[uid] for uid in aligned["trial_uid"]]
    with pytest.raises(ValueError, match="local_sample_uid disagrees"):
        broken = local.copy()
        broken[0, 1] += 50
        loso_source._align_prediction_arrays(
            {"trial_uid": uids, "y": y, "local_sample_uid": broken,
             "logits": logits}, data, 0, "test", Path("fixture.npz"))


def test_teacher_targets_align_to_student_uid_and_baseline_reader_is_strict(
        monkeypatch, tmp_path):
    _, cache_root, result_root, selected, source_sha = _fixture(monkeypatch, tmp_path)
    teacher_data = loso_source.load_dataset("BNCI2015001", "cbramod", cache_root)
    student_data = loso_source.load_dataset("BNCI2015001", None, cache_root)
    input_sha = teacher_data.input_manifest_sha256
    input_x_sha = teacher_data.input_manifest["files"]["X.npy"]["sha256"]
    current_cfg = {"in_channels": 1, "sample_rate": 250}
    monkeypatch.setattr(loso_source.config, "load_model_config",
                        lambda *_args: dict(current_cfg))

    baseline_dir = (result_root / "BNCI2015001" / "cbramod" /
                    "source_refreshed_loso_v1" / "seed_666" / "subject_01")
    baseline_dir.mkdir(parents=True)
    baseline_manifest = {
        "dataset": "BNCI2015001", "model": "cbramod", "seed": 666,
        "held_out_subject": 1, "source_manifest_sha256": source_sha,
        "class_mapping": {"feet": 0, "right_hand": 1}, "status": "complete",
        "final_model_sha256": "checkpoint-sha", "model_config": current_cfg,
    }
    baseline_sha = _write_json(baseline_dir / "manifest.json", baseline_manifest)
    target_dir = (cache_root / "teacher_targets" / "BNCI2015001" / "cbramod" /
                  "seed_666" / "subject_01")
    target_dir.mkdir(parents=True)
    train_mask = student_data.subjects != 0
    target_uid = student_data.trial_uid[train_mask][::-1]
    target_y = student_data.y[train_mask][::-1]
    logits = np.asarray([[3.0, 1.0], [1.0, 3.0]])
    feats = np.arange(6, dtype=np.float32).reshape(2, 3)
    target_path = target_dir / "targets.npz"
    np.savez(target_path, logits=logits, feats=feats, y=target_y,
             trial_uid=target_uid.astype(object))
    target_manifest = {
        "dataset": "BNCI2015001", "teacher": "cbramod", "seed": 666,
        "fold_zero_based": 0, "test_subject": 1,
        "source_manifest_sha256": source_sha,
        "input_manifest_sha256": input_sha,
        "class_mapping": {"feet": 0, "right_hand": 1},
        "teacher_manifest_sha256": baseline_sha,
        "teacher_checkpoint_sha256": "checkpoint-sha",
        "input_X_sha256": input_x_sha,
        "cache_bytes": target_path.stat().st_size,
        "cache_file_sha256": _sha(target_path),
        "train_uids": target_uid.tolist(), "train_subjects": [2],
    }
    _write_json(target_dir / "manifest.json", target_manifest)
    targets = loso_source.load_teacher_targets(
        "BNCI2015001", "cbramod", 666, 0, student_data, cache_root)
    expected_sample = {
        uid: sample for uid, sample in zip(student_data.trial_uid, student_data.sample_uid)
    }
    assert targets["sample_uid"].tolist() == [expected_sample[uid].tolist()
                                               for uid in target_uid]
    from experiments.distill.run_distill import _check_teacher
    train_mask = student_data.subjects != 0
    train_uid = student_data.sample_uid[train_mask]
    _check_teacher(targets, student_data.y[train_mask], train_uid,
                   "loso", "fixture teacher")
    assert targets["sample_uid"].tolist() == train_uid.tolist()

    # A current held-out result may be read for Fusion; training predictions
    # must come from the canonical same-fold cache and are absent here.
    prediction_dir = (result_root / "BNCI2015001" / "eegnet" /
                      "source_refreshed_loso_v1" / "seed_666" / "subject_01")
    prediction_dir.mkdir(parents=True)
    test_mask = teacher_data.subjects == 0
    test_uids = teacher_data.trial_uid[test_mask][::-1]
    test_y = teacher_data.y[test_mask][::-1]
    test_local = teacher_data.sample_uid[test_mask][::-1]
    prediction_path = prediction_dir / "result.npz"
    prediction_logits = np.asarray([[5.0, 1.0], [1.0, 5.0]])
    np.savez(prediction_path, y=test_y, trial_uid=test_uids.astype(object),
             logits=prediction_logits, features=np.ones((2, 3)),
             local_sample_uid=test_local)
    student_manifest = {
        "dataset": "BNCI2015001", "model": "eegnet", "seed": 666,
        "held_out_subject": 1, "source_manifest_sha256": source_sha,
        "class_mapping": {"feet": 0, "right_hand": 1}, "status": "complete",
        "result_sha256": _sha(prediction_path), "result_bytes": prediction_path.stat().st_size,
    }
    _write_json(prediction_dir / "manifest.json", student_manifest)
    predictions = loso_source.load_baseline_prediction(
        "BNCI2015001", "eegnet", 666, 0, "test", teacher_data)
    assert predictions["sample_uid"].tolist() == [[0, 0], [0, 1]]
    with pytest.raises(FileNotFoundError, match="missing current-model source-train"):
        loso_source.load_baseline_prediction(
            "BNCI2015001", "eegnet", 666, 0, "train", teacher_data)


def test_fusion_test_only_methods_do_not_require_training_arrays():
    from experiments.fusion import run_fusion

    args = SimpleNamespace(dataset="BNCI2015001", big="cbramod", small="eegnet",
                           big_temperature=1.0, small_temperature=1.0,
                           big_weight=0.5, epochs=1, lr=1e-3,
                           weight_decay=0.0, batch_size=2, hidden=0,
                           dropout=0.0, alpha=1.0, beta=1.0)
    test = {"y": np.asarray([0, 1]), "trial_uid": np.asarray(["a", "b"]),
            "logits": np.asarray([[2.0, 0.0], [0.0, 2.0]])}
    rows = run_fusion._run_cell_from_arrays(
        args, "loso", ["avg_prob"], "cpu", "big", "small", 0, 666,
        None, None, np.asarray([0, 1]), test, dict(test), np.asarray([0, 1]))
    assert rows[0]["method"] == "avg_prob"
    assert rows[0]["train_acc_big_pct"] == ""
    assert rows[0]["train_acc_small_pct"] == ""
    with pytest.raises(ValueError, match="require current same-fold train predictions"):
        run_fusion._run_cell_from_arrays(
            args, "loso", ["concat_mlp"], "cpu", "big", "small", 0, 666,
            None, None, np.asarray([0, 1]), test, dict(test), np.asarray([0, 1]))


def test_base_distillation_runs_without_teacher_targets(monkeypatch):
    from experiments.distill import run_distill

    args = SimpleNamespace(dataset="BNCI2015001", temperature=2.0,
                           lam_kd=0.5, lam_mi=0.1,
                           lam_mmd=0.5, mmd_sigmas=[1.0], mmd_normalize=True,
                           mmd_class_conditional=False, epochs=1, lr=0.001,
                           weight_decay=0.0, batch_size=2, no_mmd_normalize=False)
    monkeypatch.setattr(run_distill.config, "load_model_config",
                        lambda *_args: {"epochs": 1, "lr": 0.001,
                                        "weight_decay": 0.0, "batch_size": 2})
    monkeypatch.setattr(run_distill, "get_adapter", lambda *_args, **_kwargs: object())
    called = {}

    def fake_train(_adapter, _classes, _xtr, _ytr, teacher_feats, teacher_logits,
                   _xte, return_training_details=False, **_kwargs):
        called["teacher_feats"] = teacher_feats
        called["teacher_logits"] = teacher_logits
        assert return_training_details
        return np.asarray([0, 1]), {
            "full_train_mi_history": [], "train_logits": np.zeros((2, 2)),
            "train_preds": np.asarray([0, 1]), "training_history": [],
            "batch_order_hashes": [],
        }

    monkeypatch.setattr(run_distill, "distill_student", fake_train)
    row, _ = run_distill._run_method(
        args, "BNCI2015001", "loso", "cbramod", "eegnet", "cbramod_loso",
        0, 666, "Base", np.ones((2, 1, 4), dtype=np.float32),
        np.asarray([0, 1]), np.ones((2, 1, 4), dtype=np.float32),
        np.asarray([0, 1]), np.asarray([1, 1]), {}, 2, "cpu",
        uid_tr=np.asarray([[1, 0], [1, 1]]), uid_te=np.asarray([[0, 0], [0, 1]]),
        return_details=True)
    assert called == {"teacher_feats": None, "teacher_logits": None}
    assert row["failure_status"] == "complete"
    assert row["teacher_train_acc_pct"] == ""


def test_kd_resume_does_not_route_fewshot_rows_through_loso_reader(monkeypatch):
    from experiments.distill import config_runner_impl

    spec = {
        "dataset": "BNCI2015001", "protocol": "fewshot", "subjects": [0],
        "seeds": [666], "methods": ["Base"], "teacher": "cbramod",
        "student": "eegnet", "artifact_root": "/unused",
        "run_id": "fewshot-run",
    }
    row = {
        "run_id": "fewshot-run", "dataset": "BNCI2015001",
        "teacher": "cbramod", "student": "eegnet", "key": 0,
        "seed": 666, "method": "Base", "failure_status": "complete",
    }
    monkeypatch.setattr(
        loso_source, "load_dataset",
        lambda *_args, **_kwargs: pytest.fail("fewshot resume invoked current LOSO reader"))
    config_runner_impl._verify_resume_data_identity(
        [spec], [row], {config_runner_impl._row_key(row)})


def test_policy_identity_tracks_filter_settings(monkeypatch):
    current = {"l_freq": 8.0, "h_freq": 32.0, "sample_rate": 250}
    monkeypatch.setattr(loso_source.config, "load_model_config",
                        lambda *_args: dict(current))
    first = loso_source._input_policy_identity("BNCI2014001", "eegnet")
    current["l_freq"] = 10.0
    second = loso_source._input_policy_identity("BNCI2014001", "eegnet")
    assert first != second


def test_512_hz_fallback_resamples_full_epoch_before_crop():
    from scipy.signal import resample

    epoch = np.arange(1 * 1 * 2048, dtype=np.float64).reshape(1, 1, 2048)
    actual = loso_source._canonical_window(epoch, 512)
    expected = resample(epoch, 1000, axis=-1)[..., :1000].astype(np.float32)
    np.testing.assert_array_equal(actual, expected)


def test_runtime_seed_selection_is_scoped_and_rejects_unsupported_values():
    repo_root = Path(__file__).resolve().parents[1]
    loaded = load_experiment(repo_root / "configs/experiments/loso/distill_kd.yaml")
    selected = select_experiment_runs(
        loaded, datasets=["BNCI2014001", "BNCI2014004"], seeds=[0, 666])
    seeds_by_dataset = {}
    for run in selected["runs"]:
        seeds_by_dataset.setdefault(run["dataset"], set()).update(run["seeds"])
    assert seeds_by_dataset == {"BNCI2014001": {0}, "BNCI2014004": {666}}

    with pytest.raises(ExperimentConfigError, match="unsupported by every selected"):
        select_experiment_runs(
            loaded, datasets=["BNCI2014001", "BNCI2014004"], seeds=[666, 999])


def test_alexmi_readers_delegate_to_the_verified_legacy_adapter(monkeypatch):
    from data import loso_alexmi

    source = SimpleNamespace(
        source_manifest_sha256="source-sha",
        class_mapping={"feet": 0, "right_hand": 1},
        trial_uid=np.asarray(["trial-0"]), y=np.asarray([0]),
        sample_uid=np.asarray([[0, 0]]),
        identity={"input_policy_sha256": "legacy_alexmi"},
        input_manifest_sha256=None,
    )
    teacher_data = SimpleNamespace(
        source_manifest_sha256="source-sha",
        class_mapping={"feet": 0, "right_hand": 1},
        trial_uid=np.asarray(["trial-0"]), y=np.asarray([0]),
        sample_uid=np.asarray([[0, 0]]),
        identity={"input_policy_sha256": "legacy_alexmi"},
        input_manifest_sha256="teacher-input-sha",
    )
    calls = []
    monkeypatch.setattr(loso_source, "load_dataset",
                        lambda dataset, model, cache_root=None: teacher_data)
    monkeypatch.setattr(
        loso_alexmi, "load_teacher_targets",
        lambda teacher, seed, fold, data_obj: calls.append(
            ("teacher", teacher, seed, fold, data_obj)) or {"kind": "target"})
    monkeypatch.setattr(
        loso_alexmi, "load_baseline_prediction",
        lambda model, seed, fold, split, data_obj: calls.append(
            ("prediction", model, seed, fold, split, data_obj)) or {"kind": split})

    assert loso_source.load_teacher_targets(
        "AlexMI", "cbramod", 666, 0, source)["kind"] == "target"
    assert loso_source.load_baseline_prediction(
        "AlexMI", "eegnet", 666, 0, "test", source)["kind"] == "test"
    assert loso_source.load_cached_prediction(
        "AlexMI", "eegnet", 666, 0, "train", source)["kind"] == "train"
    assert calls == [
        ("teacher", "cbramod", 666, 0, source),
        ("prediction", "eegnet", 666, 0, "test", source),
        ("prediction", "eegnet", 666, 0, "train", source),
    ]


def test_alexmi_teacher_target_rejects_student_teacher_uid_label_mismatch(monkeypatch):
    source = SimpleNamespace(
        source_manifest_sha256="source-sha",
        class_mapping={"feet": 0, "right_hand": 1},
        trial_uid=np.asarray(["trial-0"]), y=np.asarray([0]),
        sample_uid=np.asarray([[0, 0]]),
        identity={"input_policy_sha256": "legacy_alexmi"},
        input_manifest_sha256=None,
    )
    teacher_data = SimpleNamespace(
        source_manifest_sha256="source-sha",
        class_mapping={"feet": 0, "right_hand": 1},
        trial_uid=np.asarray(["trial-1"]), y=np.asarray([0]),
        sample_uid=np.asarray([[0, 0]]),
        identity={"input_policy_sha256": "legacy_alexmi"},
        input_manifest_sha256=None,
    )
    monkeypatch.setattr(loso_source, "load_dataset",
                        lambda dataset, model, cache_root=None: teacher_data)
    with pytest.raises(ValueError, match="trial_uid→label/sample_uid mappings differ"):
        loso_source.load_teacher_targets("AlexMI", "cbramod", 666, 0, source)
