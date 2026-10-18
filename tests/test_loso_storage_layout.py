"""Focused behavior tests for the Phase 1 LOSO storage moves."""
from pathlib import Path

import pytest

from experiments import storage


@pytest.fixture
def relocated_layout(tmp_path, monkeypatch):
    """Install a small external layout without touching the real cache tree."""
    shared = tmp_path / "shared"
    project_data = shared / "BigSmallcollab"
    cache = project_data / "cache"
    inputs = cache / "artifact-loso" / "model_inputs"

    monkeypatch.setattr(storage, "SHARED_DATA_ROOT", shared)
    monkeypatch.setattr(storage, "PROJECT_DATA_ROOT", project_data)
    monkeypatch.setattr(storage, "DATA_CACHE_ROOT", cache)
    monkeypatch.setattr(storage, "BNCI14001_SOURCE_ROOT",
                        shared / "BNCI2014001" / "wide_fre_band")
    monkeypatch.setattr(storage, "BNCI14004_SOURCE_ROOT",
                        shared / "BNCI2014004" / "wide_fre_band")
    monkeypatch.setattr(storage, "BNCI15001_SOURCE_ROOT",
                        shared / "BNCI2015001" / "wide_fre_band")
    monkeypatch.setattr(storage, "BNCI14001_OLD_ROOT",
                        shared / "BNCI2014001" / "old")
    monkeypatch.setattr(storage, "BNCI14004_OLD_ROOT",
                        shared / "BNCI2014004" / "old")
    monkeypatch.setattr(storage, "BNCI15001_OLD_ROOT",
                        shared / "BNCI2015001" / "old")
    monkeypatch.setattr(storage, "LOSO_INPUT_ROOT", inputs)
    monkeypatch.setattr(storage, "LOSO_TEACHER_TARGET_ROOT",
                        cache / "artifact-loso" / "teacher_targets")

    source_aliases = (
        (shared / "BNCI2014001" / "broadband_0p1_75hz",
         storage.BNCI14001_SOURCE_ROOT),
        (shared / "BNCI2014004" / "broadband_0_120hz",
         storage.BNCI14004_SOURCE_ROOT),
        (shared / "BNCI2015001" / "broadband_0p1_75hz",
         storage.BNCI15001_SOURCE_ROOT),
    )
    monkeypatch.setattr(storage, "_BNCI_SOURCE_DIRECTORY_ALIASES", source_aliases)

    old_input_root = cache / "eegfm_alignment_v2" / "model_inputs" / "wideband_npy_v3"
    old_task_dir = (old_input_root / "BNCI2014001" / "cbramod"
                    / "all_sessions_source_train_session")
    new_model_dir = inputs / "BNCI2014001" / "cbramod"
    input_aliases = ((old_task_dir, new_model_dir),)
    monkeypatch.setattr(storage, "_LOSO_CACHE_ALIASES", input_aliases)
    return {
        "shared": shared,
        "cache": cache,
        "inputs": inputs,
        "old_input_root": old_input_root,
        "old_task_dir": old_task_dir,
        "new_model_dir": new_model_dir,
    }


@pytest.mark.parametrize(
    ("dataset", "old_variant", "new_variant"),
    [
        ("BNCI2014001", "broadband_0p1_75hz", "wide_fre_band"),
        ("BNCI2014004", "broadband_0_120hz", "wide_fre_band"),
        ("BNCI2015001", "broadband_0p1_75hz", "wide_fre_band"),
    ],
)
def test_historical_broadband_paths_resolve_to_the_moved_source(
        relocated_layout, dataset, old_variant, new_variant):
    shared = relocated_layout["shared"]
    historical = shared / dataset / old_variant / "manifest.json"
    current = shared / dataset / new_variant / "manifest.json"

    assert storage.external_path(historical) == current
    current.parent.mkdir(parents=True)
    current.write_text('{"source": "moved"}', encoding="utf-8")
    assert storage.resolve_local_file(historical) == current
    assert storage.resolve_local_file(historical).read_text(encoding="utf-8") == (
        '{"source": "moved"}')


@pytest.mark.parametrize(
    ("dataset", "filename"),
    [
        ("BNCI2014001", "X.npy"),
        ("BNCI2014001", "meta.csv"),
        ("BNCI2014004", "X.npy"),
        ("BNCI2014004", "meta004.csv"),
        ("BNCI2015001", "X.npy"),
        ("BNCI2015001", "meta.csv"),
    ],
)
def test_historical_flat_bnci_files_resolve_inside_old_snapshot(
        relocated_layout, dataset, filename):
    shared = relocated_layout["shared"]
    historical = shared / dataset / filename
    current = shared / dataset / "old" / filename

    assert storage.external_path(historical) == current
    current.parent.mkdir(parents=True)
    current.write_bytes(b"old-source-snapshot")
    assert storage.resolve_local_file(historical) == current
    assert storage.resolve_local_file(historical).read_bytes() == b"old-source-snapshot"


def test_001_task_cache_alias_flattens_variant(relocated_layout):
    historical = relocated_layout["old_task_dir"] / "manifest.json"
    current = relocated_layout["new_model_dir"] / "manifest.json"

    assert storage.external_path(historical) == current
    current.parent.mkdir(parents=True)
    current.write_text('{"dataset": "BNCI2014001"}', encoding="utf-8")
    assert storage.resolve_local_file(historical) == current
    assert storage.resolve_local_file(historical).read_text(encoding="utf-8") == (
        '{"dataset": "BNCI2014001"}')


def test_production_001_alias_uses_the_most_specific_variant_prefix():
    historical = (
        storage.DATA_CACHE_ROOT / "eegfm_alignment_v2" / "model_inputs"
        / "wideband_npy_v3" / "BNCI2014001" / "cbramod"
        / "all_sessions_source_train_session" / "manifest.json"
    )
    current = (storage.LOSO_INPUT_ROOT / "BNCI2014001" / "cbramod"
               / "manifest.json")

    assert storage.external_path(historical) == current


def test_deleted_source_refresh_plan_resolves_to_its_retired_archive():
    plan_name = "loso_source_refresh_distillation_004_5001_v1.yaml"
    historical = storage.PROJECT_ROOT / "configs" / "reproductions" / plan_name
    archive = (storage.PROJECT_DATA_ROOT / "migrations" / "retired_loso_configs"
               / plan_name)

    assert storage.external_path(historical) == archive


def test_unrelated_dataset_roots_are_not_reinterpreted(relocated_layout):
    shared = relocated_layout["shared"]
    bnci_root = shared / "BNCI2014001"
    unknown_source = bnci_root / "another_source" / "X.npy"
    alexmi_path = shared / "AlexMI" / "broadband_0p1_75hz" / "X.npy"

    assert storage.external_path(bnci_root) == bnci_root
    assert storage.external_path(unknown_source) == unknown_source
    assert storage.external_path(alexmi_path) == alexmi_path
