"""Completed-fold persistence, locking and provenance without GPU training."""
import json
from pathlib import Path

import pytest

from experiments.distill import progress


@pytest.fixture
def ledger_path(tmp_path, monkeypatch):
    # Exercise storage-policy invocation while keeping this test's files local.
    def report_path(value):
        path = Path(value)
        assert path.parent == tmp_path
        assert path.suffix == '.json'
        return path
    monkeypatch.setattr(progress, 'require_report_output', report_path)
    return tmp_path / 'progress.json'


SPEC = {'dataset': 'BNCI2014004', 'seeds': [666, 667, 668]}
IDENTITY = {'source_sha256': 'wideband-source', 'config': {'epochs': 100}}
ROW = {'acc': 0.75, 'kappa': 0.5, 'seed': 666, 'failure_status': 'complete'}


def test_completed_fold_survives_reopen_and_reuses_without_mutation(ledger_path):
    cell = ['BNCI2014004', 'cbramod', 'ifnet', 666, 0, 'KD_all']
    with progress.ProgressStore(ledger_path, SPEC) as store:
        store.set_status('running', total_cells=1)
        assert store.get(cell, IDENTITY) is None
        store.put(cell, IDENTITY, ROW)
        # Identical writes are safe; returned values cannot mutate the ledger.
        store.put(cell, IDENTITY, ROW)
        returned = store.get(cell, IDENTITY)
        returned['acc'] = -1
        assert store.get(cell, IDENTITY) == ROW
        records = store.all_records()
        assert records[0]['cell'] == cell
        records[0]['identity']['config']['epochs'] = -1
        records[0]['row']['acc'] = -1
        records[0]['cell'].append('changed-copy')
        assert store.get(cell, IDENTITY) == ROW
        assert store.all_records()[0]['cell'] == cell
        store.set_status('completed')
    saved = json.loads(ledger_path.read_text())
    assert saved['status'] == 'completed'
    assert saved['total_cells'] == len(saved['records']) == 1
    with progress.ProgressStore(ledger_path, dict(reversed(list(SPEC.items())))) as store:
        assert store.get(cell, IDENTITY) == ROW
        assert store.all_rows() == [ROW]


def test_running_lock_rejects_second_writer_and_releases_on_exit(ledger_path):
    with progress.ProgressStore(ledger_path, SPEC):
        with pytest.raises(progress.ProgressBusyError, match='progress lock'):
            with progress.ProgressStore(ledger_path, SPEC):
                pytest.fail('second writer acquired the lock')
    with progress.ProgressStore(ledger_path, SPEC) as store:
        store.set_status('paused', total_cells=1)
    assert json.loads(ledger_path.read_text())['status'] == 'paused'


def test_spec_mismatch_preserves_file_and_releases_lock(ledger_path):
    with progress.ProgressStore(ledger_path, SPEC) as store:
        store.put('fold0', IDENTITY, ROW)
    before = ledger_path.read_bytes()
    with pytest.raises(ValueError, match='run_spec identity mismatch'):
        with progress.ProgressStore(ledger_path, {**SPEC, 'seeds': [666]}):
            pass
    assert ledger_path.read_bytes() == before
    with progress.ProgressStore(ledger_path, SPEC) as store:
        assert store.all_rows() == [ROW]


def test_cell_identity_or_changed_result_cannot_replace_completed_fold(ledger_path):
    with progress.ProgressStore(ledger_path, SPEC) as store:
        store.put('fold0', IDENTITY, ROW)
        before = ledger_path.read_bytes()
        changed = {**IDENTITY, 'source_sha256': 'other-source'}
        with pytest.raises(ValueError, match='cell identity mismatch'):
            store.get('fold0', changed)
        with pytest.raises(ValueError, match='cell identity mismatch'):
            store.put('fold0', changed, ROW)
        with pytest.raises(ValueError, match='replace a completed'):
            store.put('fold0', IDENTITY, {**ROW, 'acc': 0.9})
        assert ledger_path.read_bytes() == before


@pytest.mark.parametrize('field,value', [('acc', float('nan')), ('kappa', float('inf')),
                                         ('acc', 'NaN'), ('kappa', None)])
def test_invalid_metrics_do_not_save_completed_records(ledger_path, field, value):
    with progress.ProgressStore(ledger_path, SPEC) as store:
        with pytest.raises(ValueError, match=field):
            store.put('fold0', IDENTITY, {**ROW, field: value})
        with pytest.raises(ValueError, match='failure_status'):
            store.put('fold0', IDENTITY, {**ROW, 'failure_status': 'invalid_metrics'})
        assert not store.all_rows()


def test_atomic_replace_failure_keeps_previous_file_and_memory(ledger_path, monkeypatch):
    with progress.ProgressStore(ledger_path, SPEC) as store:
        store.put('fold0', IDENTITY, ROW)
        before = ledger_path.read_bytes()
        def fail_replace(_source, _destination):
            raise OSError('simulated interrupted rename')
        with monkeypatch.context() as patch:
            patch.setattr(progress.os, 'replace', fail_replace)
            with pytest.raises(OSError, match='interrupted rename'):
                store.put('fold1', IDENTITY, {**ROW, 'seed': 667})
        assert ledger_path.read_bytes() == before
        assert store.get('fold1', IDENTITY) is None
        assert not list(ledger_path.parent.glob('*.tmp'))
        store.put('fold1', IDENTITY, {**ROW, 'seed': 667})
    assert len(json.loads(ledger_path.read_text())['records']) == 2


def test_file_sync_failure_keeps_previous_completed_fold(ledger_path, monkeypatch):
    with progress.ProgressStore(ledger_path, SPEC) as store:
        store.put('fold0', IDENTITY, ROW)
        before = ledger_path.read_bytes()
        def fail_sync(_fd):
            raise OSError('simulated full disk')
        with monkeypatch.context() as patch:
            patch.setattr(progress.os, 'fsync', fail_sync)
            with pytest.raises(OSError, match='full disk'):
                store.put('fold1', IDENTITY, {**ROW, 'seed': 667})
        assert ledger_path.read_bytes() == before
        assert store.all_rows() == [ROW]
        assert not list(ledger_path.parent.glob('*.tmp'))


def test_schema_and_corrupt_identity_are_rejected(ledger_path):
    with progress.ProgressStore(ledger_path, SPEC) as store:
        store.put('fold0', IDENTITY, ROW)
    saved = json.loads(ledger_path.read_text())
    for field, value, expected in [('schema_version', 900, 'schema_version'),
                                    ('run_spec_hash', 'wrong-hash', 'run_spec')]:
        ledger_path.write_text(json.dumps({**saved, field: value}))
        before = ledger_path.read_bytes()
        with pytest.raises(ValueError, match=expected):
            with progress.ProgressStore(ledger_path, SPEC):
                pass
        assert ledger_path.read_bytes() == before
    saved['records']['"fold0"']['identity_hash'] = 'bad'
    ledger_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='cell identity'):
        with progress.ProgressStore(ledger_path, SPEC):
            pass


def test_status_count_and_context_usage_are_checked(ledger_path):
    store = progress.ProgressStore(ledger_path, SPEC)
    with pytest.raises(RuntimeError, match='with block'):
        store.all_rows()
    with store:
        store.set_status('running', total_cells=1)
        with pytest.raises(ValueError, match='incomplete'):
            store.set_status('completed')
        store.put('fold0', IDENTITY, ROW)
        with pytest.raises(ValueError, match='exceed'):
            store.put('fold1', IDENTITY, ROW)
        with pytest.raises(ValueError, match='change an established'):
            store.set_status('running', total_cells=2)
        store.set_status('failed')
    assert json.loads(ledger_path.read_text())['status'] == 'failed'
