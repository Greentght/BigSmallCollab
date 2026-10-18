import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

from data import loso_source
from experiments.distill import loso_target_cache as cache


def test_student_input_cache_helper_preserves_caller_rng(monkeypatch):
    random.seed(391)
    python_state = random.getstate()
    expected_python = random.random()
    random.setstate(python_state)
    np.random.seed(427)
    numpy_state = np.random.get_state()
    expected_numpy = np.random.random()
    np.random.set_state(numpy_state)

    def simulated_prepare(*_args, **_kwargs):
        random.seed(2)
        np.random.seed(3)
        return 'loaded'

    monkeypatch.setattr(cache, '_load_or_prepare_loso_dataset_impl', simulated_prepare)
    assert cache.load_or_prepare_loso_dataset('BNCI2014004', 'mirepnet') == 'loaded'
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy


def _setup_root(monkeypatch, tmp_path):
    root = tmp_path / 'artifact-loso'
    monkeypatch.setattr(cache, 'LOSO_CACHE_ROOT', root)
    monkeypatch.setattr(cache, 'require_loso_cache_output', lambda value: Path(value))
    monkeypatch.setattr(cache, 'require_external_output', lambda value: Path(value))
    monkeypatch.setattr(cache, '_canonical_input_root', lambda value: True)
    return root


def _write_input_files(model_dir, *, corrupt_x=False, class_mapping=None):
    model_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        'X.npy': b'corrupt X' if corrupt_x else b'rebuilt X',
        'y.npy': b'y array',
        'subjects.npy': b'subject array',
        'trials.csv': b'trial_uid,label_id\nu0,0\n',
    }
    files = {}
    for name, content in payloads.items():
        path = model_dir / name
        path.write_bytes(content)
        files[name] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
    manifest = {
        'dataset': 'BNCI2014004', 'model': 'mirepnet',
        'source_manifest_sha256': 'source-sha', 'recipe': 'fixed-current',
        'class_mapping': class_mapping or {'left': 0, 'right': 1},
        'files': files,
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode() + b'\n'
    (model_dir / 'manifest.json').write_bytes(manifest_bytes)
    return manifest, manifest_bytes


def _install_rebuild_mocks(monkeypatch, root, model_dir, *, old_bytes,
                           old_identity_matches=True, reject_old_identity=False):
    stale = object()
    fresh = object()
    source = object()
    key = ('BNCI2014004', 'mirepnet', 'fixture')
    loso_source._DATA_CACHE[key] = stale

    def load_dataset(dataset, model=None, cache_root=None):
        if model is None:
            return source
        manifest_path = model_dir / 'manifest.json'
        if model_dir.is_dir() and manifest_path.is_file():
            current_manifest = json.loads(manifest_path.read_text())
            if (current_manifest.get('class_mapping') == {'left': 0, 'right': 1}
                    and (model_dir / 'X.npy').read_bytes() == b'rebuilt X'):
                return fresh
        raise ValueError('model-input cache hash or manifest mismatch')

    def prepare(dataset, model, cache_root):
        _write_input_files(model_dir)

    monkeypatch.setattr(loso_source, 'load_dataset', load_dataset)
    monkeypatch.setattr(cache, '_has_registered_input_recipe', lambda *_args: True)
    monkeypatch.setattr(cache, '_prepare_registered_input', prepare)
    if reject_old_identity:
        def identity(_dataset, _model, manifest):
            return (old_identity_matches
                    and manifest.get('class_mapping') == {'left': 0, 'right': 1})
        monkeypatch.setattr(cache, '_existing_input_identity_matches', identity)
    else:
        monkeypatch.setattr(cache, '_existing_input_identity_matches',
                            lambda *_args: old_identity_matches)
    return stale, fresh, old_bytes, key


def test_registered_input_rebuild_preserves_manifest_and_reloads_memmap(
        monkeypatch, tmp_path):
    root = _setup_root(monkeypatch, tmp_path)
    model_dir = root / 'model_inputs' / 'BNCI2014004' / 'mirepnet'
    _, expected_bytes = _write_input_files(model_dir)
    _write_input_files(model_dir, corrupt_x=True)
    old_manifest = json.loads(expected_bytes)
    old_manifest['created_utc'] = 'historical-timestamp'
    old_bytes = json.dumps(old_manifest, sort_keys=True).encode() + b'\n'
    (model_dir / 'manifest.json').write_bytes(old_bytes)
    stale, fresh, _, key = _install_rebuild_mocks(
        monkeypatch, root, model_dir, old_bytes=old_bytes)

    result = cache.load_or_prepare_loso_dataset(
        'BNCI2014004', 'mirepnet', root, auto_prepare=True)

    assert result is fresh
    assert result is not stale
    assert loso_source._DATA_CACHE.get(key) is None
    assert (model_dir / 'manifest.json').read_bytes() == old_bytes
    assert not list(model_dir.parent.glob('.mirepnet.input-backup-*'))


def test_missing_registered_student_input_is_prepared_and_strictly_reloaded(
        monkeypatch, tmp_path):
    root = _setup_root(monkeypatch, tmp_path)
    model_dir = root / 'model_inputs' / 'BNCI2014004' / 'mirepnet'
    source = object()
    fresh = object()
    load_calls = []

    def load_dataset(dataset, model=None, cache_root=None):
        load_calls.append(model)
        if model is None or not model_dir.exists():
            return source
        return fresh

    def prepare(dataset, model, cache_root):
        _write_input_files(model_dir)

    monkeypatch.setattr(loso_source, 'load_dataset', load_dataset)
    monkeypatch.setattr(cache, '_has_registered_input_recipe', lambda *_args: True)
    monkeypatch.setattr(cache, '_prepare_registered_input', prepare)

    result = cache.load_or_prepare_loso_dataset(
        'BNCI2014004', 'mirepnet', root, auto_prepare=True)

    assert result is fresh
    assert model_dir.joinpath('manifest.json').is_file()
    assert load_calls.count('mirepnet') >= 2


def test_semantically_invalid_old_manifest_is_not_preserved(monkeypatch, tmp_path):
    root = _setup_root(monkeypatch, tmp_path)
    model_dir = root / 'model_inputs' / 'BNCI2014004' / 'mirepnet'
    _, expected_bytes = _write_input_files(model_dir)
    old_manifest = json.loads(expected_bytes)
    old_manifest['class_mapping'] = {'changed': 0, 'mapping': 1}
    old_bytes = json.dumps(old_manifest, sort_keys=True).encode() + b'\n'
    _write_input_files(model_dir, corrupt_x=True)
    (model_dir / 'manifest.json').write_bytes(old_bytes)
    _, fresh, _, _ = _install_rebuild_mocks(
        monkeypatch, root, model_dir, old_bytes=old_bytes,
        old_identity_matches=True, reject_old_identity=True)

    result = cache.load_or_prepare_loso_dataset(
        'BNCI2014004', 'mirepnet', root, auto_prepare=True)

    assert result is fresh
    rebuilt_manifest = json.loads((model_dir / 'manifest.json').read_text())
    assert rebuilt_manifest['class_mapping'] == {'left': 0, 'right': 1}
    assert (model_dir / 'manifest.json').read_bytes() != old_bytes


@pytest.mark.parametrize(('payload', 'shape_error'), [
    ('[]', TypeError('manifest must be a mapping')),
    ('null', AttributeError("'NoneType' object has no attribute 'get'")),
])
def test_json_non_object_input_manifest_is_rebuilt(monkeypatch, tmp_path,
                                                   payload, shape_error):
    root = _setup_root(monkeypatch, tmp_path)
    model_dir = root / 'model_inputs' / 'BNCI2014004' / 'mirepnet'
    model_dir.mkdir(parents=True)
    (model_dir / 'manifest.json').write_text(payload)
    fresh = object()

    def load_dataset(dataset, model=None, cache_root=None):
        if model is None:
            return object()
        manifest_path = model_dir / 'manifest.json'
        if not model_dir.exists():
            return fresh
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict):
            raise shape_error
        return fresh

    monkeypatch.setattr(loso_source, 'load_dataset', load_dataset)
    monkeypatch.setattr(cache, '_has_registered_input_recipe', lambda *_args: True)
    monkeypatch.setattr(cache, '_prepare_registered_input',
                        lambda *_args: _write_input_files(model_dir))

    assert cache.load_or_prepare_loso_dataset(
        'BNCI2014004', 'mirepnet', root, auto_prepare=True) is fresh
    assert isinstance(json.loads((model_dir / 'manifest.json').read_text()), dict)
    assert not list(model_dir.parent.glob('.mirepnet.input-backup-*'))


def test_broken_optional_001_small_model_cache_archives_and_uses_source_fallback(
        monkeypatch, tmp_path):
    root = _setup_root(monkeypatch, tmp_path)
    model_dir = root / 'model_inputs' / 'BNCI2014001' / 'ifnet'
    model_dir.mkdir(parents=True)
    (model_dir / 'broken.bin').write_bytes(b'bad')
    fallback = object()
    source = object()
    calls = []

    def load_dataset(dataset, model=None, cache_root=None):
        calls.append(model)
        if model is None:
            return source
        if model_dir.exists():
            raise ValueError('bad optional cache')
        return fallback

    monkeypatch.setattr(loso_source, 'load_dataset', load_dataset)
    result = cache.load_or_prepare_loso_dataset(
        'BNCI2014001', 'ifnet', root, auto_prepare=True)

    assert result is fallback
    assert calls.count(None) >= 1
    assert not model_dir.exists()
    assert not list(model_dir.parent.glob('.ifnet.input-backup-*'))


def test_raw_source_error_does_not_archive_input_cache(monkeypatch, tmp_path):
    root = _setup_root(monkeypatch, tmp_path)
    model_dir = root / 'model_inputs' / 'BNCI2014001' / 'ifnet'
    model_dir.mkdir(parents=True)
    sentinel = model_dir / 'keep.bin'
    sentinel.write_bytes(b'preserve')

    def load_dataset(dataset, model=None, cache_root=None):
        if model is None:
            raise FileNotFoundError('raw source is unavailable')
        raise ValueError('derived input cache is invalid')

    monkeypatch.setattr(loso_source, 'load_dataset', load_dataset)
    try:
        cache.load_or_prepare_loso_dataset(
            'BNCI2014001', 'ifnet', root, auto_prepare=True)
    except FileNotFoundError as error:
        assert 'raw source is unavailable' in str(error)
    else:
        raise AssertionError('raw source errors must propagate')
    assert sentinel.read_bytes() == b'preserve'
    assert model_dir.is_dir()


def test_read_only_input_load_does_not_create_lock_or_rebuild(monkeypatch, tmp_path):
    root = _setup_root(monkeypatch, tmp_path)
    model_dir = root / 'model_inputs' / 'BNCI2014001' / 'ifnet'
    model_dir.mkdir(parents=True)
    sentinel = model_dir / 'keep.bin'
    sentinel.write_bytes(b'preserve')

    def load_dataset(dataset, model=None, cache_root=None):
        if model is None:
            return object()
        raise ValueError('input cache remains read-only invalid')

    monkeypatch.setattr(loso_source, 'load_dataset', load_dataset)
    try:
        cache.load_or_prepare_loso_dataset(
            'BNCI2014001', 'ifnet', root, auto_prepare=False)
    except ValueError as error:
        assert 'read-only invalid' in str(error)
    else:
        raise AssertionError('read-only load should preserve the validation error')
    assert sentinel.read_bytes() == b'preserve'
    assert not (model_dir.parent / '.ifnet.prepare.lock').exists()
