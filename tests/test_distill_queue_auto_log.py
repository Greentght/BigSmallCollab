"""Completion callbacks are separate from training success; no GPU work."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.distill import queue


@pytest.fixture
def completed(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, 'validated_root', lambda value=None: Path(value or tmp_path))
    store = queue.QueueStore(tmp_path)
    job_id = queue.enqueue_snapshot({'args': {}}, root=tmp_path)
    state = store.load(job_id)
    state['status'] = 'completed'
    store.save(state)
    return store, job_id


def stub_writer(monkeypatch, function):
    monkeypatch.setitem(sys.modules, 'experiments.distill.experiment_log',
                        SimpleNamespace(register_completed_job=function))


def test_success_is_recorded_once_and_explicit_register_can_recheck(completed, monkeypatch):
    store, job = completed
    calls = []
    stub_writer(monkeypatch, lambda path: calls.append(path) or
                {'status': 'written', 'reason': 'one result registered'})
    assert queue.register_job_log(store, job)['status'] == 'written'
    assert queue.register_job_log(store, job)['status'] == 'written'
    assert len(calls) == 1
    assert queue.register_job_log(store, job, force=True)['status'] == 'written'
    assert len(calls) == 2
    assert store.load(job)['status'] == 'completed'


def test_writer_failure_keeps_training_complete_and_retries_after_backoff(completed, monkeypatch):
    store, job = completed
    calls = []
    clock = [1000.0]
    monkeypatch.setattr(queue.time, 'time', lambda: clock[0])

    def writer(path):
        calls.append(path)
        if len(calls) == 1:
            raise PermissionError('CSV locked by filesystem')
        return {'status': 'unchanged'}

    stub_writer(monkeypatch, writer)
    result = queue.register_job_log(store, job)
    assert result['status'] == 'error'
    assert 'PermissionError' in result['reason']
    assert store.load(job)['status'] == 'completed'
    queue.register_job_log(store, job)
    assert len(calls) == 1
    clock[0] += 61
    assert queue.register_job_log(store, job)['status'] == 'unchanged'
    assert len(calls) == 2


def test_pending_failed_and_paused_tasks_are_never_written(completed, monkeypatch):
    store, job = completed
    stub_writer(monkeypatch, lambda _: pytest.fail('incomplete task registered'))
    for status in ('pending', 'running', 'paused', 'failed'):
        state = store.load(job)
        state['status'] = status
        store.save(state)
        assert queue.register_job_log(store, job) is None
        assert queue.register_completed_logs(store) == {}


def test_recovered_completion_is_discovered_without_running_training(completed, monkeypatch):
    store, job = completed
    state = store.load(job)
    state.update(status='running', worker=None, child=None)
    store.save(state)
    queue.atomic_json(store.job_dir(job) / 'progress.json',
                      {'status': 'completed', 'records': {'fold': {}}, 'total_cells': 1})
    calls = []
    stub_writer(monkeypatch, lambda path: calls.append(path) or {'status': 'written'})
    with store.lock():
        queue.reconcile(store)
    assert queue.register_completed_logs(store)[job]['status'] == 'written'
    assert calls == [store.job_dir(job)]
    assert store.load(job)['status'] == 'completed'


def test_status_shows_registration_outcome(completed, monkeypatch, capsys):
    store, job = completed
    stub_writer(monkeypatch, lambda _: {'status': 'skipped', 'reason': 'requires three seeds'})
    queue.register_job_log(store, job)
    queue.show_status(store)
    assert 'exp_log=skipped requires three seeds' in capsys.readouterr().out
    queue.show_status(store, as_json=True)
    assert json.loads(capsys.readouterr().out)['jobs'][0]['log_registration']['status'] == 'skipped'


def test_registration_cli_does_not_dispatch_jobs(completed, monkeypatch):
    store, job = completed
    stub_writer(monkeypatch, lambda _: {'status': 'written'})
    monkeypatch.setattr(queue, 'run_manager', lambda _: pytest.fail('manager launched'))
    assert queue.main(['register', '--root', str(store.root), job]) == 0
    assert store.load(job)['log_registration']['status'] == 'written'


def test_registration_cli_reports_unknown_job(completed):
    store, _job = completed
    assert queue.main(['register', '--root', str(store.root), 'unknown']) == 2


@pytest.mark.parametrize('interval', ['0', '-1', 'nan', 'inf'])
def test_watch_rejects_invalid_poll_interval(interval):
    with pytest.raises(SystemExit):
        queue.parse_args(['register', '--watch', '--poll-seconds', interval])
