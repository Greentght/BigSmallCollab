"""CPU-only lifecycle tests: no dataset/cache/model loading and no real GPU calls."""
from argparse import Namespace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from experiments.distill import queue


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, 'validated_root', lambda value=None: Path(value or tmp_path))
    return queue.QueueStore(tmp_path)


def snapshot():
    return {'schema_version': 1, 'args': {
        'dataset': 'BNCI2014004', 'teacher': 'cbramod', 'student': 'ifnet',
        'seeds': [666, 667, 668], 'epochs': 50, 'lr': 0.001,
        'batch_size': 16, 'methods': ['KD_all'], 'lam_kd': 0.25,
        'temperature': 1.0, 'keys': list(range(9))}, 'dataset_configs': {'BNCI2014004': {'source': 'wide_fre_band', 'num_subjects': 9}},
        'student_configs': {'BNCI2014004/ifnet': {'epochs': 50}}}


def submit(store, **updates):
    spec = snapshot()
    spec['args'].update(updates)
    job_id = queue.enqueue_snapshot(spec, root=store.root)
    return store.load(job_id)


def test_submission_freezes_settings_and_deduplicates(store):
    original = snapshot()
    job_id = queue.enqueue_snapshot(original, root=store.root, argv=['--epochs', '50'])
    original['args']['epochs'] = 999
    saved = queue.read_json(store.job_dir(job_id) / 'spec.json')
    assert saved['snapshot']['args']['epochs'] == 50
    assert saved['argv'] == ['--epochs', '50']
    assert queue.enqueue_snapshot(snapshot(), root=store.root) == job_id
    assert len(store.states()) == 1
    assert queue.enqueue_snapshot(original, root=store.root) != job_id


def test_enqueue_args_success_is_zero_exit_status(store, monkeypatch):
    monkeypatch.setitem(sys.modules, 'experiments.distill.run_distill',
                        SimpleNamespace(freeze_run=lambda args: snapshot()))
    assert queue.enqueue_args(Namespace(queue_root=str(store.root))) == 0


def test_reconcile_preserves_live_orphan_and_rejects_pid_reuse(store):
    state = submit(store)
    current = queue.process_identity()
    state.update(status='running', gpu=2, worker=None, child=current)
    store.save(state)
    queue.reconcile(store)
    assert store.load(state['id'])['status'] == 'running'
    reused = dict(current, start_ticks=str(int(current['start_ticks']) + 1))
    state['child'] = reused
    store.save(state)
    assert not queue.process_alive(reused)
    queue.reconcile(store)
    recovered = store.load(state['id'])
    assert recovered['status'] == 'pending'
    assert 'resume' in recovered['message']


def test_reconcile_recovers_completed_fold_state(store):
    state = submit(store)
    state.update(status='running', worker=None, child=None)
    store.save(state)
    queue.atomic_json(store.job_dir(state['id']) / 'progress.json',
                      {'status': 'completed', 'records': {'fold': {}}, 'total_cells': 1})
    queue.reconcile(store)
    assert store.load(state['id'])['status'] == 'completed'


def test_reconcile_retains_failure_until_explicit_retry(store):
    state = submit(store)
    state.update(status='running', worker=None, child=None)
    store.save(state)
    queue.atomic_json(store.job_dir(state['id']) / 'progress.json',
                      {'status': 'failed', 'records': {}, 'total_cells': 27})
    queue.reconcile(store)
    assert store.load(state['id'])['status'] == 'failed'
    queue.mutate_job(store, 'retry', [state['id']])
    assert store.load(state['id'])['status'] == 'pending'


def test_starting_state_waits_for_live_launcher(store):
    import time
    state = submit(store)
    state.update(status='starting', launcher=queue.process_identity(),
                 launch_time=time.time(), child=None, worker=None)
    store.save(state)
    queue.reconcile(store)
    assert store.load(state['id'])['status'] == 'starting'


def test_inherited_lease_closes_spawn_before_pid_save_crash_window(store):
    state = submit(store)
    state.update(status='running', gpu=2, worker=None, child=None)
    store.save(state)
    folder = store.job_dir(state['id'])
    finish = folder / 'finish_orphan'
    code = (
        'import pathlib,time; '
        f'p=pathlib.Path({str(finish)!r}); '
        '\nwhile not p.exists(): time.sleep(.01)\n'
    )
    with queue.file_lock(folder / 'worker.lock', blocking=False) as lease:
        child = subprocess.Popen([sys.executable, '-c', code],
                                 pass_fds=(lease.fileno(),))
    try:
        # The parent no longer owns its FD, and no child PID was saved, but
        # the orphan's inherited lease still reserves the trial and GPU slot.
        assert queue.worker_lock_held(store, state['id'])
        queue.reconcile(store)
        assert store.load(state['id'])['status'] == 'running'
    finally:
        finish.touch()
        child.wait(timeout=5)
    assert not queue.worker_lock_held(store, state['id'])
    queue.reconcile(store)
    assert store.load(state['id'])['status'] == 'pending'


def test_delayed_worker_cannot_claim_new_launch_reservation(store, monkeypatch):
    state = submit(store)
    state.update(status='starting', gpu=2, gpu_uuid='GPU-new', launch_token='new-launch')
    store.save(state)
    monkeypatch.setattr(queue.subprocess, 'Popen', lambda *a, **k: pytest.fail('stale worker launched training'))
    args = Namespace(root=str(store.root), job=state['id'], gpu=2, launch_token='old-launch')
    assert queue.run_worker(args) == 0
    assert store.load(state['id'])['status'] == 'starting'
    assert store.load(state['id'])['launch_token'] == 'new-launch'


def gpu(used=0, utilization=0, pids=()):
    return {'used_mb': used, 'total_mb': 24000, 'utilization': utilization,
            'uuid': 'GPU-test', 'pids': set(pids)}


def test_gpu_requires_no_external_compute_and_low_memory_and_utilization():
    assert queue.choose_gpu({2: gpu()}, [], [2]) == 2
    assert queue.choose_gpu({2: gpu(pids=[123456])}, [], [2]) is None
    assert queue.choose_gpu({2: gpu(used=1000)}, [], [2]) is None
    assert queue.choose_gpu({2: gpu(utilization=11)}, [], [2]) is None
    assert queue.choose_gpu({}, [], [2]) is None


def test_gpu_explicit_parallelism_allows_only_own_processes():
    state = {'status': 'running', 'gpu': 2, 'worker': queue.process_identity(), 'child': None}
    available = {2: gpu(used=4000, utilization=85, pids=[os.getpid()])}
    assert queue.choose_gpu(available, [state], [2], per_gpu=1) is None
    assert queue.choose_gpu(available, [state], [2], per_gpu=2) == 2
    available[2]['pids'].add(123456)
    assert queue.choose_gpu(available, [state], [2], per_gpu=2) is None


@pytest.mark.parametrize('failure', ['exception', 'empty', 'malformed', 'process-query'])
def test_gpu_query_failure_is_not_treated_as_idle(monkeypatch, failure):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if failure == 'exception' or (failure == 'process-query' and len(calls) == 2):
            raise subprocess.CalledProcessError(1, command)
        if len(calls) == 1:
            output = '2, GPU-test, 0, 24000, 0\n'
            if failure == 'empty':
                output = ''
            if failure == 'malformed':
                output = '2, GPU-test, [N/A], 24000, 0\n'
        else:
            output = ''
        return Namespace(stdout=output)

    monkeypatch.setattr(queue.subprocess, 'run', fake_run)
    with pytest.raises(RuntimeError, match='cannot verify GPU availability'):
        queue.query_gpus()


def test_manager_lock_excludes_second_manager(store):
    with queue.file_lock(store.root / 'manager.lock', blocking=False):
        with pytest.raises(RuntimeError, match='already held'):
            with queue.file_lock(store.root / 'manager.lock', blocking=False):
                pytest.fail('second manager acquired the lock')


def test_pause_resume_stop_retry_and_reorder_preserve_records(store):
    first = submit(store)
    second = submit(store, epochs=100)
    queue.mutate_job(store, 'move', [second['id']])
    assert store.states()[0]['id'] == second['id']
    queue.mutate_job(store, 'pause', [])
    assert store.control()['paused']
    queue.mutate_job(store, 'resume', [])
    assert not store.control()['paused']
    progress = {'status': 'running', 'records': {'seed666fold0': {'acc': 0.5}}, 'total_cells': 27}
    queue.atomic_json(store.job_dir(first['id']) / 'progress.json', progress)
    queue.mutate_job(store, 'stop', [first['id']])
    assert store.load(first['id'])['status'] == 'paused'
    queue.mutate_job(store, 'resume', [first['id']])
    assert store.load(first['id'])['status'] == 'pending'
    assert not (store.job_dir(first['id']) / 'stop').exists()
    assert queue.read_json(store.job_dir(first['id']) / 'progress.json') == progress


def test_worker_runs_cpu_child_and_reuses_progress_with_unique_attempt_logs(store, monkeypatch):
    state = submit(store)
    state.update(status='starting', gpu=2, gpu_uuid='GPU-test')
    store.save(state)
    folder = store.job_dir(state['id'])
    initial = {'status': 'paused', 'records': {'already-finished': {'acc': 0.6}}, 'total_cells': 2}
    queue.atomic_json(folder / 'progress.json', initial)
    real_popen = subprocess.Popen
    commands = []
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '7')

    def cpu_child(command, **kwargs):
        commands.append(command)
        assert '--resume' in command
        assert command[command.index('--snapshot') + 1] == str(folder / 'spec.json')
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == 'GPU-test'
        assert command[command.index('--gpu') + 1] == '0'
        script = (
            'import json,pathlib; '
            f'p=pathlib.Path({str(folder / "progress.json")!r}); '
            'x=json.loads(p.read_text()); '
            'assert "already-finished" in x["records"]; '
            'x["records"]["new-fold"]={"acc":0.7}; x["status"]="completed"; '
            'p.write_text(json.dumps(x))'
        )
        return real_popen([sys.executable, '-c', script], **kwargs)

    monkeypatch.setattr(queue.subprocess, 'Popen', cpu_child)
    args = Namespace(root=str(store.root), job=state['id'], gpu=2)
    assert queue.run_worker(args) == 0
    assert store.load(state['id'])['status'] == 'completed'
    assert len(queue.read_json(folder / 'progress.json')['records']) == 2
    assert (folder / 'attempt_001.json').exists()
    state = store.load(state['id'])
    state['status'] = 'starting'
    store.save(state)
    assert queue.run_worker(args) == 0
    assert (folder / 'attempt_002.json').exists()
    assert commands[0][commands[0].index('--log_file') + 1] != commands[1][commands[1].index('--log_file') + 1]


def test_manager_keeps_pending_when_gpu_query_fails(store, monkeypatch):
    state = submit(store)
    def query_failed():
        raise RuntimeError('nvidia-smi unavailable')
    monkeypatch.setattr(queue, 'query_gpus', query_failed)
    monkeypatch.setattr(queue, 'dispatch_one', lambda *args: pytest.fail('launched without GPU verification'))
    args = queue.parse_args(['run', '--root', str(store.root), '--gpus', '2', '--once'])
    assert queue.run_manager(args) == 0
    assert store.load(state['id'])['status'] == 'pending'
    assert 'unavailable' in queue.read_json(store.root / 'manager_message.json')['message']


def test_manager_dispatches_pending_once_and_does_not_touch_external_processes(store, monkeypatch):
    state = submit(store)
    monkeypatch.setattr(queue, 'query_gpus', lambda: {2: gpu()})
    launched = []
    def dispatch(store_, gpu_, gpu_uuid=None):
        launched.append((store_.root, gpu_))
        assert gpu_uuid == 'GPU-test'
        return state['id']
    monkeypatch.setattr(queue, 'dispatch_one', dispatch)
    args = queue.parse_args(['run', '--root', str(store.root), '--gpus', '2', '--once'])
    assert queue.run_manager(args) == 0
    assert launched == [(store.root, 2)]
    queue.mutate_job(store, 'pause', [])
    launched.clear()
    assert queue.run_manager(args) == 0
    assert not launched


def test_detached_cpu_worker_survives_manager_reconstruction_without_duplicate(store, monkeypatch):
    """A real CPU child outlives dispatch and keeps its slot after reconstruction."""
    state = submit(store)
    folder = store.job_dir(state['id'])
    started = folder / 'fake_started'
    finish = folder / 'fake_finish'
    script = f'''
import time
from pathlib import Path
from experiments.distill.queue import QueueStore, process_identity, atomic_json
store = QueueStore({str(store.root)!r})
with store.lock():
    state = store.load({state['id']!r})
    state.update(status='running', worker=process_identity(), child=None)
    store.save(state)
Path({str(started)!r}).write_text('one CPU worker')
while not Path({str(finish)!r}).exists():
    time.sleep(.01)
atomic_json(Path({str(folder / 'progress.json')!r}),
            {{'status':'completed','records':{{'cpu-fold':{{}}}},'total_cells':1}})
'''
    monkeypatch.setattr(queue, 'worker_command', lambda *_: [sys.executable, '-c', script])
    with store.lock():
        assert queue.dispatch_one(store, 2) == state['id']
    identity = None
    try:
        deadline = time.monotonic() + 5
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert started.exists()
        reconstructed = queue.QueueStore(store.root)
        with reconstructed.lock():
            queue.reconcile(reconstructed)
            saved = reconstructed.load(state['id'])
            identity = saved['worker']
            assert saved['status'] == 'running'
            assert queue.process_alive(identity)
            # A new manager must retain the reservation even when the fake
            # GPU reports no compute processes yet.
            assert queue.choose_gpu({2: gpu()}, reconstructed.states(), [2]) is None
            assert queue.dispatch_one(reconstructed, 2) is None
        assert started.read_text() == 'one CPU worker'
    finally:
        finish.touch()
        if identity:
            deadline = time.monotonic() + 5
            while queue.process_alive(identity) and time.monotonic() < deadline:
                time.sleep(.01)
            assert not queue.process_alive(identity)
    with store.lock():
        queue.reconcile(store)
    assert store.load(state['id'])['status'] == 'completed'


def test_status_reports_actual_progress(store, capsys):
    state = submit(store)
    queue.atomic_json(store.job_dir(state['id']) / 'progress.json',
                      {'status': 'running', 'records': {'a': {}, 'b': {}}, 'total_cells': 27})
    capsys.readouterr()
    queue.show_status(store, as_json=True)
    result = json.loads(capsys.readouterr().out)
    assert result['jobs'][0]['completed_cells'] == 2
    assert result['jobs'][0]['total_cells'] == 27
    assert result['manager_running'] is False


def test_status_pending_shows_resolved_expected_fold_count(store, capsys):
    submit(store)
    capsys.readouterr()
    queue.show_status(store, as_json=True)
    result = json.loads(capsys.readouterr().out)
    assert result['jobs'][0]['total_cells'] == 27
    assert result['jobs'][0]['completed_cells'] == 0
