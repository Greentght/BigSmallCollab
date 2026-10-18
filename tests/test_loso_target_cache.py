import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
from types import ModuleType

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
    monkeypatch.setattr(cache.subprocess, 'Popen',
                        lambda *args, **kwargs: pytest.fail('unexpected export'))

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

    def fake_popen(command, cwd, stdout, stderr, text, bufsize):
        captured.setdefault('commands', []).append(command)
        captured['cwd'] = cwd
        captured['streaming'] = (stdout, stderr, text, bufsize)
        if '--export-one' in command:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b'produced target')
            manifest.write_text('{"status":"complete"}')
        return SimpleNamespace(stdout=io.StringIO('[child] complete\n'), wait=lambda: 0)

    monkeypatch.setattr(cache.subprocess, 'Popen', fake_popen)
    result = cache.ensure_teacher_targets(
        'BNCI2014001', 'cbramod', 0, 0, object(), cache_root=root, gpu=None)

    assert result is bundle
    export_command = next(cmd for cmd in captured['commands'] if '--export-one' in cmd)
    assert export_command[0] == '/envs/cbramod/bin/python'
    assert export_command[export_command.index('--device') + 1] == 'cpu'
    assert '--lock-held' in export_command
    assert '--rebuild-invalid' in export_command
    assert captured['streaming'] == (subprocess.PIPE, subprocess.STDOUT, True, 1)
    assert artifact.is_file() and manifest.is_file()
    assert cell.joinpath('.lock').is_file()


def test_valid_cache_created_by_waiting_writer_is_reused_after_lock(monkeypatch, tmp_path):
    root, _ = _paths(monkeypatch, tmp_path)
    bundle = {'created_by_other_writer': True}
    calls = {'count': 0}

    def strict_load(*args, **kwargs):
        calls['count'] += 1
        if calls['count'] == 1:
            raise FileNotFoundError('cache is not present before lock')
        return bundle

    monkeypatch.setattr(cache.loso_source, 'load_teacher_targets', strict_load)
    monkeypatch.setattr(cache.subprocess, 'Popen',
                        lambda *args, **kwargs: pytest.fail('valid cache must be reused'))
    assert cache.ensure_teacher_targets(
        'BNCI2014001', 'cbramod', 0, 0, object(), cache_root=root) is bundle
    assert calls['count'] == 2


@pytest.mark.parametrize('failure', [
    ValueError('teacher target identity mismatch'),
    __import__('zipfile').BadZipFile('truncated npz'),
    EOFError('truncated archive'),
    OSError('target read failed'),
])
def test_invalid_complete_cache_is_rebuilt_after_lock(monkeypatch, tmp_path, failure):
    root, paths = _paths(monkeypatch, tmp_path)
    _, artifact, manifest, _ = paths
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'corrupt target')
    manifest.write_text('{"wrong":"identity"}')
    (root / 'model_inputs' / 'BNCI2014001' / 'cbramod').mkdir(parents=True)
    bundle = {'rebuilt': True}

    def strict_load(*args, **kwargs):
        if artifact.read_bytes() == b'rebuilt target':
            return bundle
        raise failure

    monkeypatch.setattr(cache.loso_source, 'load_teacher_targets', strict_load)
    monkeypatch.setattr(cache, '_teacher_python', lambda teacher, dataset: Path('/envs/cbramod/bin/python'))
    exported = {'count': 0}

    def fake_popen(command, **kwargs):
        if '--export-one' in command:
            exported['count'] += 1
            artifact.write_bytes(b'rebuilt target')
            manifest.write_text('{"status":"complete"}')
        return SimpleNamespace(stdout=io.StringIO('rebuild line\n'), wait=lambda: 0)

    monkeypatch.setattr(cache.subprocess, 'Popen', fake_popen)
    result = cache.ensure_teacher_targets(
        'BNCI2014001', 'cbramod', 0, 0, object(), cache_root=root)

    assert result is bundle
    assert exported['count'] == 1
    assert artifact.read_bytes() == b'rebuilt target'
    assert json.loads(manifest.read_text()) == {'status': 'complete'}


@pytest.mark.parametrize(('payload', 'shape_error'), [
    ('[]', TypeError('manifest must be a mapping')),
    ('null', AttributeError("'NoneType' object has no attribute 'get'")),
])
def test_json_non_object_target_manifest_is_rebuilt(monkeypatch, tmp_path,
                                                    payload, shape_error):
    root, paths = _paths(monkeypatch, tmp_path)
    _, artifact, manifest, _ = paths
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'old target')
    manifest.write_text(payload)
    bundle = {'rebuilt': True}

    def strict_load(*_args, **_kwargs):
        value = json.loads(manifest.read_text())
        if not isinstance(value, dict):
            raise shape_error
        if artifact.read_bytes() == b'rebuilt target':
            return bundle
        raise ValueError('target pair is not complete')

    monkeypatch.setattr(cache.loso_source, 'load_teacher_targets', strict_load)
    monkeypatch.setattr(cache, '_run_teacher_input_prepare', lambda *_args: None)
    monkeypatch.setattr(cache, '_teacher_python', lambda *_args: Path('/envs/cbramod/bin/python'))

    def fake_popen(command, **kwargs):
        if '--export-one' in command:
            artifact.write_bytes(b'rebuilt target')
            manifest.write_text('{"status":"complete"}')
        return SimpleNamespace(stdout=io.StringIO(''), wait=lambda: 0)

    monkeypatch.setattr(cache.subprocess, 'Popen', fake_popen)
    assert cache.ensure_teacher_targets(
        'BNCI2014001', 'cbramod', 0, 0, object(), cache_root=root) is bundle
    assert json.loads(manifest.read_text()) == {'status': 'complete'}


def test_partial_target_pair_is_rebuilt(monkeypatch, tmp_path):
    root, paths = _paths(monkeypatch, tmp_path)
    _, artifact, manifest, _ = paths
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'old partial target')
    bundle = {'rebuilt': True}

    def strict_load(*args, **kwargs):
        if artifact.read_bytes() == b'rebuilt target' and manifest.exists():
            return bundle
        raise FileNotFoundError('target manifest missing')

    monkeypatch.setattr(cache.loso_source, 'load_teacher_targets', strict_load)
    monkeypatch.setattr(cache, '_teacher_python', lambda teacher, dataset: Path('/envs/cbramod/bin/python'))

    def fake_popen(command, **kwargs):
        if '--export-one' in command:
            artifact.write_bytes(b'rebuilt target')
            manifest.write_text('{"status":"complete"}')
        return SimpleNamespace(stdout=io.StringIO(''), wait=lambda: 0)

    monkeypatch.setattr(cache.subprocess, 'Popen', fake_popen)
    assert cache.ensure_teacher_targets(
        'BNCI2014001', 'cbramod', 0, 0, object(), cache_root=root) is bundle


def test_failed_export_leaves_old_complete_target_pair(monkeypatch, tmp_path):
    root, paths = _paths(monkeypatch, tmp_path)
    _, artifact, manifest, _ = paths
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'old complete cache')
    manifest.write_text('{"old":"manifest"}')
    monkeypatch.setattr(cache.loso_source, 'load_teacher_targets',
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            ValueError('invalid cache hash')))
    monkeypatch.setattr(cache, '_teacher_python', lambda teacher, dataset: Path('/envs/cbramod/bin/python'))

    def fake_popen(command, **kwargs):
        code = 3 if '--export-one' in command else 0
        return SimpleNamespace(stdout=io.StringIO('child failure\n'), wait=lambda: code)

    monkeypatch.setattr(cache.subprocess, 'Popen', fake_popen)
    with pytest.raises(subprocess.CalledProcessError):
        cache.ensure_teacher_targets(
            'BNCI2014001', 'cbramod', 0, 0, object(), cache_root=root)
    assert artifact.read_bytes() == b'old complete cache'
    assert manifest.read_text() == '{"old":"manifest"}'


@pytest.mark.parametrize('returncode', [0, 7])
def test_streaming_subprocess_forwards_child_output_and_checks_status(
        monkeypatch, capsys, returncode):
    monkeypatch.setattr(
        cache.subprocess, 'Popen',
        lambda *args, **kwargs: SimpleNamespace(
            stdout=io.StringIO('[child] visible\n'), wait=lambda: returncode))

    if returncode:
        with pytest.raises(subprocess.CalledProcessError) as raised:
            cache._run_streaming_subprocess(['/env/python', '-m', 'producer'], Path('/repo'))
        assert raised.value.returncode == returncode
    else:
        cache._run_streaming_subprocess(['/env/python', '-m', 'producer'], Path('/repo'))
    assert '[child] visible\n' in capsys.readouterr().out


def test_checkpoint_prepare_runs_before_strict_sha_validation(monkeypatch, tmp_path):
    baseline_path = tmp_path / 'manifest.json'
    checkpoint = tmp_path / 'final_model.pt'
    checkpoint.write_bytes(b'checkpoint')
    baseline_path.write_text('{}')
    baseline = {'final_model_sha256': hashlib.sha256(b'checkpoint').hexdigest()}
    teacher_data = object()
    calls = []
    prepare = ModuleType('experiments.distill.loso_teacher_prepare')
    prepare.ensure_teacher_checkpoint = lambda *args: calls.append(args)
    monkeypatch.setitem(sys.modules, prepare.__name__, prepare)
    monkeypatch.setattr(cache.loso_source, '_baseline_teacher_manifest',
                        lambda *args: baseline_path)
    monkeypatch.setattr(cache.loso_source, '_json', lambda path: baseline)
    config = {'validated': True}
    monkeypatch.setattr(cache.loso_source, 'validated_teacher_model_config',
                        lambda *args: config)

    result = cache._load_checkpoint(
        'BNCI2014004', 'mirepnet', 666, 1, teacher_data, 'cpu')

    assert calls == [('BNCI2014004', 'mirepnet', 666, 1, teacher_data, 'cpu')]
    assert result == (baseline_path, baseline, checkpoint,
                      hashlib.sha256(b'checkpoint').hexdigest(), config)

    baseline['final_model_sha256'] = 'wrong-sha'
    with pytest.raises(ValueError, match='SHA-256 differs'):
        cache._load_checkpoint(
            'BNCI2014004', 'mirepnet', 666, 1, teacher_data, 'cpu')
    assert len(calls) == 2


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
    monkeypatch.setattr(cache, '_load_teacher_inputs',
                        lambda dataset, model, root: SimpleNamespace(
                            input_manifest={'files': {}}, input_manifest_sha256='input'))
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

    def save_pair(cell, artifact, manifest_path, lock_path, arrays, metadata,
                  **kwargs):
        saved.update(arrays=arrays, metadata=metadata)
        return metadata, {}

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
