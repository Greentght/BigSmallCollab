import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.distill import loso_target_cache as cache


def _paths(monkeypatch, tmp_path, dataset='BNCI2014001'):
    root = tmp_path / 'artifact-loso'
    monkeypatch.setattr(cache, 'LOSO_CACHE_ROOT', root)
    monkeypatch.setattr(cache, 'require_loso_cache_output', lambda value: Path(value))
    monkeypatch.setattr(cache, 'require_external_output', lambda value: Path(value))
    return root, cache._target_paths(dataset, 'cbramod', 0, 0, root)


def _loader_for(paths, bundle):
    cell, artifact, manifest, _ = paths

    def load(*args, **kwargs):
        if artifact.is_file() and manifest.is_file():
            return bundle
        raise FileNotFoundError(f'missing targets under {cell}')

    return load


def test_ensure_returns_valid_cache_without_spawning(monkeypatch, tmp_path):
    root, paths = _paths(monkeypatch, tmp_path)
    _, artifact, manifest, _ = paths
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'valid target placeholder')
    manifest.write_text('{}')
    bundle = {'logits': np.ones((2, 2), dtype=np.float32)}
    monkeypatch.setattr(cache.loso_source, 'load_teacher_targets', _loader_for(paths, bundle))
    monkeypatch.setattr(cache.subprocess, 'run', lambda *args, **kwargs: pytest.fail('unexpected export'))

    py_state = random.getstate()
    np_state = np.random.get_state()
    result = cache.ensure_teacher_targets(
        'BNCI2014001', 'cbramod', 0, 0, object(), cache_root=root)

    assert result is bundle
    assert random.getstate() == py_state
    current = np.random.get_state()
    assert current[0] == np_state[0]
    np.testing.assert_array_equal(current[1], np_state[1])
    assert current[2:] == np_state[2:]


def test_missing_pair_runs_one_teacher_env_export_then_strictly_reloads(
        monkeypatch, tmp_path):
    root, paths = _paths(monkeypatch, tmp_path)
    cell, artifact, manifest, _ = paths
    bundle = {'trial_uid': np.asarray(['u0', 'u1'])}
    monkeypatch.setattr(cache.loso_source, 'load_teacher_targets', _loader_for(paths, bundle))
    monkeypatch.setattr(cache, '_teacher_python', lambda teacher, dataset: Path('/envs/cbramod/bin/python'))
    captured = {}

    def fake_run(command, cwd, check):
        captured.update(command=command, cwd=cwd, check=check)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b'produced target')
        manifest.write_text('{"status":"complete"}')

    monkeypatch.setattr(cache.subprocess, 'run', fake_run)
    result = cache.ensure_teacher_targets(
        'BNCI2014001', 'cbramod', 0, 0, object(), cache_root=root, gpu=None)

    assert result is bundle
    assert captured['command'][0] == '/envs/cbramod/bin/python'
    assert '--export-one' in captured['command']
    assert captured['command'][captured['command'].index('--device') + 1] == 'cpu'
    assert '--lock-held' in captured['command']
    assert captured['check'] is True
    assert artifact.is_file() and manifest.is_file()
    assert cell.joinpath('.lock').is_file()


def test_existing_invalid_or_partial_cache_is_never_overwritten(monkeypatch, tmp_path):
    root, paths = _paths(monkeypatch, tmp_path)
    _, artifact, manifest, _ = paths
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'preserve me')
    manifest.write_text('{"wrong":"identity"}')
    (root / 'model_inputs' / 'BNCI2014001' / 'cbramod').mkdir(parents=True)

    def invalid(*args, **kwargs):
        raise ValueError('teacher source identity mismatch')

    monkeypatch.setattr(cache.loso_source, 'load_teacher_targets', invalid)
    monkeypatch.setattr(cache.subprocess, 'run', lambda *args, **kwargs: pytest.fail('must not export'))
    with pytest.raises(ValueError, match='identity mismatch'):
        cache.ensure_teacher_targets(
            'BNCI2014001', 'cbramod', 0, 0, object(), cache_root=root)
    assert artifact.read_bytes() == b'preserve me'
    assert manifest.read_text() == '{"wrong":"identity"}'

    manifest.unlink()

    def missing(*args, **kwargs):
        raise FileNotFoundError('target manifest missing')

    monkeypatch.setattr(cache.loso_source, 'load_teacher_targets', missing)
    with pytest.raises(RuntimeError, match='incomplete teacher-target cache'):
        cache.ensure_teacher_targets(
            'BNCI2014001', 'cbramod', 0, 0, object(), cache_root=root)
    assert artifact.read_bytes() == b'preserve me'


@pytest.mark.parametrize('lock_held', [False, True])
def test_export_one_strictly_reuses_complete_cache(monkeypatch, tmp_path, lock_held):
    root, paths = _paths(monkeypatch, tmp_path)
    cell, artifact, manifest, _ = paths
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'valid target placeholder')
    manifest.write_text('{"status":"complete"}')
    # Keep the model-input directory present so the strict loader remains the
    # only authority for cache validity in this test.
    (root / 'model_inputs' / 'BNCI2014001' / 'cbramod').mkdir(parents=True)
    student_data = object()
    bundle = {'validated': True}
    monkeypatch.setattr(cache.loso_source, 'load_dataset',
                        lambda dataset, model=None, cache_root=None: student_data)
    monkeypatch.setattr(cache.loso_source, 'load_teacher_targets',
                        lambda *args, **kwargs: bundle)
    monkeypatch.setattr(cache, '_export_dispatch',
                        lambda *args, **kwargs: pytest.fail('valid cache must be reused'))

    result = cache._export_one(
        'BNCI2014001', 'cbramod', 0, 0, root, 'cpu', lock_held)

    assert result is bundle
    assert artifact.read_bytes() == b'valid target placeholder'


def test_atomic_target_pair_records_exact_npz_hash(monkeypatch, tmp_path):
    _, paths = _paths(monkeypatch, tmp_path)
    cell, artifact, manifest_path, lock_path = paths
    cell.mkdir(parents=True)
    arrays = {
        'logits': np.asarray([[0.2, 0.8]], dtype=np.float32),
        'feats': np.asarray([[1.0, 2.0]], dtype=np.float32),
        'y': np.asarray([1], dtype=np.int64),
        'trial_uid': np.asarray(['uid-a']),
    }
    saved = cache._atomic_write_targets(
        cell, artifact, manifest_path, lock_path, arrays, {'status': 'complete'})

    assert saved['cache_bytes'] == artifact.stat().st_size
    assert saved['cache_file_sha256'] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert json.loads(manifest_path.read_text()) == saved
    with np.load(artifact, allow_pickle=False) as loaded:
        assert set(loaded.files) == set(arrays)
        np.testing.assert_array_equal(loaded['trial_uid'], arrays['trial_uid'])


def test_001_mirepnet_build_uses_profile_dropout_when_current_config_omits_it(monkeypatch):
    from experiments.finetune import run_loso_config_alignment as reference

    calls = {}

    class FakeModel:
        def to(self, device):
            calls['device'] = device
            return self

    model = FakeModel()

    def make_model(teacher, dataset, n_classes, input_shape, dropout, profile):
        calls.update(teacher=teacher, dataset=dataset, n_classes=n_classes,
                     input_shape=input_shape, dropout=dropout, profile=profile)
        return model

    monkeypatch.setattr(reference, 'make_model', make_model)
    built, adapter = cache._build_teacher(
        'BNCI2014001', 'mirepnet', {'skip_preprocess': True}, (45, 1000), 2, 'cpu')

    assert built is model
    assert adapter is None
    assert calls['dropout'] == 0.5
    assert calls['profile'] == reference.WIDEBAND_PROFILE
    assert calls['input_shape'] == (45, 1000)


def test_export_standard_forwards_only_training_uids_in_student_order(monkeypatch, tmp_path):
    import torch

    student = SimpleNamespace(
        x=np.zeros((4, 1), dtype=np.float32),
        y=np.asarray([0, 1, 0, 1], dtype=np.int64),
        subjects=np.asarray([0, 0, 1, 1], dtype=np.int64),
        trial_uid=np.asarray(['s0_a', 's0_b', 's1_a', 's1_b']),
        class_mapping={'left': 0, 'right': 1},
        source_manifest_sha256='source-sha',
    )
    # Teacher input order is reversed within each subject. X encodes its UID,
    # so the fake model can prove both UID alignment and held-out exclusion.
    teacher = SimpleNamespace(
        x=np.asarray([[20.0], [10.0], [40.0], [30.0]], dtype=np.float32),
        y=np.asarray([1, 0, 1, 0], dtype=np.int64),
        subjects=np.asarray([0, 0, 1, 1], dtype=np.int64),
        trial_uid=np.asarray(['s0_b', 's0_a', 's1_b', 's1_a']),
        class_mapping={'left': 0, 'right': 1},
        source_manifest_sha256='source-sha',
        input_manifest_sha256='input-manifest-sha',
        input_manifest={'files': {'X.npy': {'sha256': 'input-x-sha'}}},
    )
    monkeypatch.setattr(cache.loso_source, 'load_dataset',
                        lambda dataset, model=None, cache_root=None: student)
    monkeypatch.setattr(cache, '_load_teacher_inputs', lambda dataset, model, root: teacher)
    cfg = {'skip_preprocess': True, 'batch_size': 1}
    monkeypatch.setattr(
        cache, '_load_checkpoint',
        lambda *args: (tmp_path / 'baseline.json', {}, tmp_path / 'final_model.pt', 'checkpoint-sha', cfg))
    monkeypatch.setattr(cache.loso_source, 'sha256_file', lambda path: 'teacher-manifest-sha')
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: {'state': 'checkpoint'})
    observed = {'batches': []}

    class FakeModel:
        def load_state_dict(self, state, strict):
            assert state == {'state': 'checkpoint'}
            assert strict is True

        def eval(self):
            return self

        def requires_grad_(self, enabled):
            assert enabled is False
            return self

        def __call__(self, x):
            seen = x.detach().cpu().numpy().copy()
            observed['batches'].append(seen)
            feat = x.reshape(len(x), -1)
            logits = torch.stack((feat[:, 0], -feat[:, 0]), dim=1)
            return feat, logits

    monkeypatch.setattr(cache, '_build_teacher',
                        lambda *args: (FakeModel(), None))
    saved = {}

    def save_pair(cell, artifact, manifest_path, lock_path, arrays, metadata):
        saved.update(arrays=arrays, metadata=metadata)
        return metadata

    monkeypatch.setattr(cache, '_atomic_write_targets', save_pair)
    expected_bundle = {'validated': True}
    monkeypatch.setattr(cache.loso_source, 'load_teacher_targets',
                        lambda *args, **kwargs: expected_bundle)

    bundle = cache._export_standard(
        'BNCI2014001', 'mirepnet', 123, 1, tmp_path,
        'cpu', tmp_path / 'cell', tmp_path / 'targets.npz',
        tmp_path / 'manifest.json', tmp_path / '.lock')

    seen_values = np.concatenate(observed['batches'], axis=0).reshape(-1)
    np.testing.assert_array_equal(seen_values, [10.0, 20.0])
    np.testing.assert_array_equal(saved['arrays']['trial_uid'], ['s0_a', 's0_b'])
    np.testing.assert_array_equal(saved['arrays']['y'], [0, 1])
    assert saved['metadata']['train_uids'] == ['s0_a', 's0_b']
    assert saved['metadata']['train_subjects'] == [0]
    assert 1 not in saved['metadata']['train_subjects']
    assert bundle is expected_bundle
