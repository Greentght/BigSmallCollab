"""Persistent, user-controlled parallel queue for explicit distillation trials.

This module imports no training framework until a trial is submitted. The
manager may be stopped and restarted while its independent workers continue.
Only the worker's own child process is managed; external GPU jobs are never
signalled. Queue files contain reports/configuration, never model tensors.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from experiments.storage import PROJECT_ROOT, REPORTS_ROOT, require_report_output

DEFAULT_ROOT = REPORTS_ROOT / 'loso/distill/queue'
ACTIVE = {'starting', 'running', 'stopping'}
TERMINAL = {'completed', 'failed', 'paused', 'cancelled'}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path, default=None):
    try:
        with Path(path).open(encoding='utf-8') as stream:
            return json.load(stream)
    except FileNotFoundError:
        return default


def resolve_training_python(preferred=None):
    """Resolve the project's CUDA-capable interpreter for this V1 job."""
    candidates = []
    override = os.environ.get('BIGSMALLCOLLAB_TRAIN_PYTHON')
    if override:
        candidates.append(override)
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix and Path(conda_prefix).parent.name == 'envs':
        candidates.append(str(Path(conda_prefix) / 'bin' / 'python'))
    executable = Path(sys.executable).resolve()
    for ancestor in executable.parents:
        candidates.append(str(ancestor / 'envs' / 'mirepnet' / 'bin' / 'python'))
    candidates.extend([sys.executable, str(preferred) if preferred else None])
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        candidate = str(Path(candidate).expanduser())
        if candidate in seen:
            continue
        seen.add(candidate)
        if Path(candidate).is_file():
            return candidate
    raise RuntimeError('no training Python found; activate mirepnet or set BIGSMALLCOLLAB_TRAIN_PYTHON')


@contextmanager
def file_lock(path, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError(f'lock is already held: {path}') from exc
        # Closing releases the lock after the final inherited descriptor closes.
        # An explicit LOCK_UN here would release a runner child's inherited lock.
        yield stream


def process_identity(pid=None):
    """Linux boot ID + /proc start ticks prevent accepting a reused PID."""
    pid = os.getpid() if pid is None else int(pid)
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        if fields[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': fields[19],
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (OSError, IndexError, ValueError):
        return None


def process_alive(identity):
    return bool(identity and process_identity(identity['pid']) == identity)


def worker_lock_held(store, job_id):
    try:
        with file_lock(store.job_dir(job_id) / 'worker.lock', blocking=False):
            return False
    except RuntimeError:
        return True


def descendant_of(pid, identity):
    if not process_alive(identity):
        return False
    target = int(identity['pid'])
    seen = set()
    try:
        while int(pid) > 1 and int(pid) not in seen:
            pid = int(pid)
            if pid == target:
                return True
            seen.add(pid)
            raw = Path(f'/proc/{pid}/stat').read_text()
            pid = int(raw[raw.rfind(')') + 2:].split()[1])
    except (OSError, IndexError, ValueError):
        pass
    return False


class QueueStore:
    def __init__(self, root=DEFAULT_ROOT):
        self.root = Path(root)
        self.jobs = self.root / 'jobs'

    def job_dir(self, job_id):
        if not job_id or Path(job_id).name != job_id or job_id.startswith('.'):
            raise ValueError(f'invalid job ID: {job_id!r}')
        return self.jobs / job_id

    def states(self):
        states = []
        for path in self.jobs.glob('*/state.json'):
            state = read_json(path)
            if state is not None:
                states.append(state)
        return sorted(states, key=lambda state: (state.get('priority', 0),
                                                 state['created_at'], state['id']))

    def load(self, job_id):
        state = read_json(self.job_dir(job_id) / 'state.json')
        if state is None:
            raise ValueError(f'unknown job: {job_id}')
        return state

    def save(self, state):
        state['updated_at'] = utc_now()
        atomic_json(self.job_dir(state['id']) / 'state.json', state)

    def lock(self):
        return file_lock(self.root / 'state.lock')

    def control(self):
        return read_json(self.root / 'control.json', {'paused': False})


def validated_root(value=None):
    # require_report_output checks the real checkout results tree and symlinks.
    return require_report_output(Path(value or DEFAULT_ROOT) / 'queue.json').parent


def enqueue_snapshot(snapshot, *, root=DEFAULT_ROOT, argv=None, label=None):
    """Save an immutable resolved trial without loading tensors or starting GPU work."""
    store = QueueStore(validated_root(root))
    canonical = json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                           separators=(',', ':'))
    identity = hashlib.sha256(canonical.encode()).hexdigest()
    args = snapshot.get('args', {})
    with store.lock():
        for state in store.states():
            if state.get('snapshot_sha256') == identity and state['status'] != 'cancelled':
                print(f"[queue] already registered {state['id']} status={state['status']}")
                return state['id']
        job_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
        folder = store.job_dir(job_id)
        folder.mkdir(parents=True)
        atomic_json(folder / 'spec.json', {
            'schema_version': 1, 'snapshot': snapshot,
            'snapshot_sha256': identity, 'argv': list(argv or []),
            'submitted_at': utc_now(), 'python': sys.executable,
        })
        state = {'id': job_id, 'status': 'pending', 'created_at': utc_now(),
                 'priority': 0, 'attempt': 0, 'snapshot_sha256': identity,
                 'label': label, 'args': args, 'gpu': None,
                 'worker': None, 'child': None, 'launcher': None,
                 'message': 'waiting for an idle GPU'}
        store.save(state)
    print(f'[queue] submitted {job_id}: {folder}')
    return job_id


def enqueue_args(args):
    from experiments.distill.run_distill import freeze_run
    enqueue_snapshot(freeze_run(args), root=getattr(args, 'queue_root', None) or DEFAULT_ROOT,
                     argv=sys.argv[1:])
    return 0


def query_gpus():
    """Fail closed: query failures are errors, never a list of empty GPUs."""
    gpu_command = ['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
                   '--format=csv,noheader,nounits']
    process_command = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                       '--format=csv,noheader,nounits']
    try:
        output = subprocess.run(gpu_command, check=True, text=True, capture_output=True,
                                timeout=15).stdout
        processes = subprocess.run(process_command, check=True, text=True,
                                   capture_output=True, timeout=15).stdout
        result = {}
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(',')]
            if len(parts) != 5:
                raise ValueError(f'bad GPU query line: {line}')
            index, gpu_uuid, used, total, utilization = parts
            result[int(index)] = {'uuid': gpu_uuid, 'used_mb': int(used),
                                  'total_mb': int(total), 'utilization': int(utilization),
                                  'pids': set()}
        by_uuid = {info['uuid']: info for info in result.values()}
        for line in processes.splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(',')]
            if len(parts) != 2 or parts[0] not in by_uuid:
                raise ValueError(f'bad GPU process query line: {line}')
            by_uuid[parts[0]]['pids'].add(int(parts[1]))
        if not result:
            raise ValueError('nvidia-smi returned no GPUs')
        return result
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f'cannot verify GPU availability: {exc}') from exc


def choose_gpu(gpus, states, requested, per_gpu=1, max_used_mb=512,
               max_utilization=10, min_free_mb=2048):
    for index in requested:
        info = gpus.get(index)
        if info is None:
            continue
        active = [state for state in states
                  if state['status'] in ACTIVE and state.get('gpu') == index]
        if len(active) >= per_gpu:
            continue
        identities = [identity for state in active
                      for identity in (state.get('worker'), state.get('child')) if identity]
        external = [pid for pid in info['pids']
                    if not any(descendant_of(pid, identity) for identity in identities)]
        if external or info['total_mb'] - info['used_mb'] < min_free_mb:
            continue
        if not active and (info['used_mb'] > max_used_mb or
                           info['utilization'] > max_utilization):
            continue
        return index
    return None


def progress_for(store, state):
    progress = read_json(store.job_dir(state['id']) / 'progress.json', {})
    records = progress.get('records', [])
    return progress, len(records) if isinstance(records, (dict, list)) else 0


def register_job_log(store, job_id, force=False):
    """Register completed metrics without changing the training outcome."""
    with store.lock():
        state = store.load(job_id)
        if state['status'] != 'completed':
            return None
        previous = state.get('log_registration', {})
        if not force:
            if previous.get('status') in {'written', 'unchanged', 'skipped'}:
                return previous
            if time.time() < previous.get('retry_after', 0):
                return previous
    try:
        from experiments.distill.experiment_log import register_completed_job
        result = dict(register_completed_job(store.job_dir(job_id)))
    except Exception as exc:
        result = {'status': 'error', 'reason': f'{type(exc).__name__}: {exc}'}
    result['checked_at'] = utc_now()
    if result.get('status') == 'error':
        result['retry_after'] = time.time() + 60
    with store.lock():
        state = store.load(job_id)
        state['log_registration'] = result
        store.save(state)
    print(f"[experiment-log] {job_id} {result.get('status')}: "
          f"{result.get('reason', '')}", flush=True)
    return result


def register_completed_logs(store, job_ids=None, force=False):
    with store.lock():
        for job_id in job_ids or []:
            store.load(job_id)  # An unknown requested ID must not look successful.
        candidates = [state['id'] for state in store.states()
                      if state['status'] == 'completed' and
                      (not job_ids or state['id'] in job_ids)]
    return {job_id: register_job_log(store, job_id, force) for job_id in candidates}


def watch_registration(args):
    """Cover already-running workers without restarting their GPU processes."""
    store = QueueStore(validated_root(args.root))
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    with file_lock(store.root / 'registration_watcher.lock', blocking=False):
        atomic_json(store.root / 'registration_watcher.json', {
            'process': process_identity(), 'status': 'running',
            'started_at': utc_now(), 'poll_seconds': args.poll_seconds})
        print(f'[experiment-log] watching completed jobs in {store.root}', flush=True)
        try:
            while not stopping:
                try:
                    with store.lock():
                        reconcile(store)
                    register_completed_logs(store)
                except Exception as exc:
                    print(f'[experiment-log] scan error: {type(exc).__name__}: {exc}', flush=True)
                deadline = time.monotonic() + args.poll_seconds
                while not stopping and time.monotonic() < deadline:
                    time.sleep(min(.25, max(0, deadline - time.monotonic())))
        finally:
            atomic_json(store.root / 'registration_watcher.json', {
                'process': process_identity(), 'status': 'stopped',
                'stopped_at': utc_now()})
    return 0


def reconcile(store):
    """Called under the state lock; orphan live workers/children keep their slot."""
    for state in store.states():
        if state['status'] not in ACTIVE:
            continue
        if process_alive(state.get('worker')) or process_alive(state.get('child')):
            continue
        # The runner inherits this lock before exec. It closes the SIGKILL
        # window between Popen and persisting the child's PID in state.json.
        if worker_lock_held(store, state['id']):
            continue
        if state['status'] == 'starting' and process_alive(state.get('launcher')):
            # Parent may have spawned a worker which has not registered yet.
            if time.time() - state.get('launch_time', 0) < 60:
                continue
        progress, _ = progress_for(store, state)
        if progress.get('status') == 'completed':
            state['status'] = 'completed'
            state['message'] = 'completed result recovered after manager/worker exit'
        elif progress.get('status') == 'failed':
            state['status'] = 'failed'
            state['message'] = 'runner failure recovered; inspect logs and explicitly retry'
        elif (store.job_dir(state['id']) / 'stop').exists():
            state['status'] = 'paused'
            state['message'] = 'paused; completed folds retained'
        else:
            state['status'] = 'pending'
            state['message'] = 'interrupted process exited; queued to resume completed folds'
        state['worker'] = state['child'] = state['launcher'] = None
        store.save(state)


def worker_command(store, state):
    spec = read_json(store.job_dir(state['id']) / 'spec.json')
    return [spec.get('python', sys.executable), '-B', '-m',
            'experiments.distill.queue', '_worker', '--root', str(store.root),
            '--job', state['id'], '--gpu', str(state['gpu']),
            '--launch-token', state['launch_token']]


def dispatch_one(store, gpu, gpu_uuid=None):
    """Reserve and spawn while holding the state lock to close duplicate races."""
    candidates = [state for state in store.states() if state['status'] == 'pending']
    if not candidates:
        return None
    state = candidates[0]
    state.update(status='starting', gpu=gpu, gpu_uuid=gpu_uuid,
                 launch_token=uuid.uuid4().hex, worker=None, child=None,
                 launcher=process_identity(), launch_time=time.time(),
                 message=f'launching on GPU {gpu}')
    store.save(state)
    folder = store.job_dir(state['id'])
    try:
        with (folder / 'worker.log').open('a', encoding='utf-8') as stream:
            process = subprocess.Popen(worker_command(store, state), cwd=PROJECT_ROOT,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
        # The worker also registers itself. Do not overwrite its newer state.
        state = store.load(state['id'])
        if state['status'] == 'starting':
            state['worker'] = process_identity(process.pid)
            store.save(state)
    except Exception as exc:
        state.update(status='failed', message=f'worker launch failed: {exc}')
        store.save(state)
        raise
    return state['id']


def run_manager(args):
    store = QueueStore(validated_root(args.root))
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    with file_lock(store.root / 'manager.lock', blocking=False):
        atomic_json(store.root / 'manager.json', {
            'process': process_identity(), 'gpus': args.gpus, 'per_gpu': args.per_gpu,
            'started_at': utc_now(), 'status': 'running'})
        last_message = None
        print(f'[queue] manager active, GPUs={args.gpus}, per_gpu={args.per_gpu}', flush=True)
        try:
            while not stopping:
                try:
                    register_completed_logs(store)
                except Exception as exc:
                    print(f'[experiment-log] registration scan failed: {exc}', flush=True)
                with store.lock():
                    reconcile(store)
                    states = store.states()
                    pending = any(state['status'] == 'pending' for state in states)
                    paused = store.control().get('paused', False)
                message = None
                dispatched = False
                if pending and not paused:
                    try:
                        gpus = query_gpus()
                    except RuntimeError as exc:
                        message = str(exc)
                    else:
                        with store.lock():
                            reconcile(store)
                            if not store.control().get('paused', False):
                                gpu = choose_gpu(gpus, store.states(), args.gpus,
                                                 args.per_gpu, args.max_used_mb,
                                                 args.max_utilization, args.min_free_mb)
                                if gpu is not None:
                                    job_id = dispatch_one(store, gpu, gpu_uuid=gpus[gpu]['uuid'])
                                    dispatched = job_id is not None
                                    message = f'started {job_id} on GPU {gpu}' if job_id else None
                                else:
                                    message = 'waiting: no verified idle GPU slot'
                elif paused:
                    message = 'scheduling paused; existing workers continue'
                if message and message != last_message:
                    print(f'[queue] {message}', flush=True)
                    last_message = message
                    atomic_json(store.root / 'manager_message.json',
                                {'message': message, 'updated_at': utc_now()})
                if args.once:
                    break
                if dispatched:
                    continue  # Fill other verified GPU slots without an artificial poll delay.
                # Short increments let SIGTERM/keyboard interruption stop promptly.
                deadline = time.monotonic() + args.poll_seconds
                while not stopping and time.monotonic() < deadline:
                    time.sleep(min(0.25, max(0, deadline - time.monotonic())))
        finally:
            atomic_json(store.root / 'manager.json', {
                'process': process_identity(), 'status': 'stopped', 'stopped_at': utc_now()})
            print('[queue] manager stopped; active workers continue independently', flush=True)
    return 0


def run_worker(args):
    store = QueueStore(validated_root(args.root))
    folder = store.job_dir(args.job)
    with file_lock(folder / 'worker.lock', blocking=False) as run_lock:
        state = store.load(args.job)
        if (getattr(args, 'launch_token', None) != state.get('launch_token') or
                state.get('gpu') != args.gpu):
            return 0  # A delayed old worker must not claim a newer reservation.
        gpu_uuid = state.get('gpu_uuid')
        if not gpu_uuid:
            try:
                gpu_uuid = query_gpus()[args.gpu]['uuid']
            except (RuntimeError, KeyError) as exc:
                with store.lock():
                    state = store.load(args.job)
                    state.update(status='failed', message=f'cannot resolve physical GPU UUID: {exc}')
                    store.save(state)
                return 1
        with store.lock():
            state = store.load(args.job)
            if (state['status'] not in ACTIVE or
                    getattr(args, 'launch_token', None) != state.get('launch_token')):
                return 0
            state.update(status='running', gpu=args.gpu, gpu_uuid=gpu_uuid,
                         worker=process_identity(),
                         child=None, launcher=None, attempt=state.get('attempt', 0) + 1,
                         message=f'running on GPU {args.gpu}')
            store.save(state)
        attempt = state['attempt']
        log = folder / f'attempt_{attempt:03d}.log'
        console = folder / f'attempt_{attempt:03d}.console.log'
        spec = read_json(folder / 'spec.json')
        snapshot = spec.get('snapshot', {})
        training_python = spec.get('python', sys.executable)
        if snapshot.get('kind') == 'cbramod_channel_feature_alignment':
            command = [training_python, '-B', '-m',
                       'experiments.distill.run_cbramod_channel_feature',
                       '--snapshot', str(folder / 'spec.json'),
                       '--progress', str(folder / 'progress.json'),
                       '--gpu', '0', '--stop-file', str(folder / 'stop')]
        elif snapshot.get('kind') == 'student_aware_diffusion_prior_v1':
            v1_python = resolve_training_python(spec.get('python'))
            command = [v1_python, '-B', '-m',
                       'experiments.distill.run_student_aware_diffusion_loso',
                       '--config', snapshot['config'], '--run-id', snapshot['run_id'],
                       '--expected-config-sha256', snapshot['config_sha256'],
                       '--gpu', '0', '--progress', str(folder / 'progress.json'),
                       '--stop-file', str(folder / 'stop')]
            if snapshot.get('stage') == 'smoke':
                command.append('--smoke')
            elif snapshot.get('stage') == 'evaluate':
                command.append('--evaluate')
            else:
                command.extend(['--fold', str(snapshot['fold'])])
        elif snapshot.get('kind') == 'mcc_diffkd_loso_v1':
            command = [training_python, '-B', '-m',
                       'experiments.distill.run_mcc_diffkd_loso',
                       '--config', snapshot['config'], '--run-id', snapshot['run_id'],
                       '--expected-config-sha256', snapshot['config_sha256'],
                       '--gpu', '0', '--progress', str(folder / 'progress.json'),
                       '--stop-file', str(folder / 'stop'), '--folds',
                       *[str(fold) for fold in snapshot['folds']]]
            if snapshot.get('stage') == 'smoke':
                command.append('--smoke')
        else:
            command = [training_python, '-B', '-m',
                       'experiments.distill.run_distill', '--snapshot', str(folder / 'spec.json'),
                       '--progress', str(folder / 'progress.json'),
                       '--out_csv', str(folder / 'results.csv'), '--log_file', str(log),
                       '--gpu', '0', '--stop-file', str(folder / 'stop'), '--resume']
        child_environment = os.environ.copy()
        if snapshot.get('kind') in {
                'cbramod_channel_feature_alignment', 'student_aware_diffusion_prior_v1',
                'mcc_diffkd_loso_v1'}:
            for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                         'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                         'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                         'BIGSMALLCOLLAB_CPU_THREADS'):
                child_environment[name] = '1'
            child_environment['TORCH_NUM_THREADS'] = '1'
        # Pin the physical GPU by UUID. CUDA ordinal zero then refers to that
        # same device in both the student and its teacher-cache subprocesses.
        child_environment['CUDA_VISIBLE_DEVICES'] = gpu_uuid
        child_environment['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        atomic_json(folder / f'attempt_{attempt:03d}.json', {
            'command': command, 'started_at': utc_now(), 'gpu': args.gpu,
            'gpu_uuid': gpu_uuid,
            'snapshot_sha256': state['snapshot_sha256']})

        def request_stop(signum, frame):
            # Never relay signals to a GPU process: request a clean fold boundary.
            (folder / 'stop').touch()

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, request_stop)
        child = None
        try:
            with console.open('a', encoding='utf-8') as stream:
                child = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True,
                                         env=child_environment,
                                         pass_fds=(run_lock.fileno(),))
                with store.lock():
                    state = store.load(args.job)
                    state['child'] = process_identity(child.pid)
                    store.save(state)
                returncode = child.wait()
            progress, _ = progress_for(store, state)
            if progress.get('status') == 'completed' and returncode == 0:
                status, message = 'completed', 'all requested folds completed'
            elif returncode == 75 or progress.get('status') == 'paused':
                status, message = 'paused', 'paused after current fold; resume preserves completed folds'
            else:
                status = 'failed'
                message = f'runner exited {returncode}; see {console.name} and {log.name}'
        except Exception as exc:
            if child is not None and child.poll() is None:
                # Leave the independently running child visible. The inherited
                # job lock also protects it if even this state write fails.
                with store.lock():
                    state = store.load(args.job)
                    state.update(status='running', worker=None,
                                 child=process_identity(child.pid),
                                 message=f'worker exited while child continues: {exc}')
                    store.save(state)
                return 1
            returncode = -1
            status, message = 'failed', f'worker error: {type(exc).__name__}: {exc}'
        with store.lock():
            state = store.load(args.job)
            state.update(status=status, message=message, returncode=returncode,
                         child=None, worker=None, launcher=None)
            if status == 'completed' and snapshot.get('kind') in {
                    'cbramod_channel_feature_alignment', 'student_aware_diffusion_prior_v1',
                    'mcc_diffkd_loso_v1'}:
                state['log_registration'] = {
                    'status': 'skipped',
                    'reason': ('single-seed MCC-DiffKD LOSO is summarized separately, not registered in the three-seed tuning log'
                               if snapshot.get('kind') == 'mcc_diffkd_loso_v1' else
                               'student-aware V1 uses its dedicated post-training evaluator'
                               if snapshot.get('kind') == 'student_aware_diffusion_prior_v1' else
                               'CBraMod channel-feature results are not tuning-log entries')}
            store.save(state)
        if (status == 'completed' and snapshot.get('kind') not in {
                'cbramod_channel_feature_alignment', 'student_aware_diffusion_prior_v1',
                'mcc_diffkd_loso_v1'}):
            try:
                register_job_log(store, args.job)
            except Exception as exc:
                print(f'[experiment-log] registration failed; training completed: {exc}', flush=True)
        return 0 if status in {'completed', 'paused'} else 1


def mutate_job(store, action, job_ids):
    with store.lock():
        reconcile(store)
        if action in {'pause', 'resume'} and not job_ids:
            atomic_json(store.root / 'control.json', {'paused': action == 'pause',
                                                     'updated_at': utc_now()})
            return
        if not job_ids:
            raise ValueError(f'{action} requires at least one job ID')
        for job_id in job_ids:
            state = store.load(job_id)
            folder = store.job_dir(job_id)
            if action in {'retry', 'resume'}:
                if state['status'] not in {'failed', 'paused'}:
                    raise ValueError(f"cannot {action} {job_id}: status={state['status']}")
                (folder / 'stop').unlink(missing_ok=True)
                state.update(status='pending', message='queued to resume retained completed folds')
            elif action == 'stop':
                (folder / 'stop').touch()
                if state['status'] in ACTIVE:
                    state.update(status='stopping', message='stop requested; waiting for fold boundary')
                elif state['status'] == 'pending':
                    state.update(status='paused', message='paused before starting')
            elif action == 'cancel':
                if state['status'] in ACTIVE:
                    raise ValueError('stop the running job first; cancel does not kill processes')
                state.update(status='cancelled', message='cancelled by user; artifacts preserved')
            elif action == 'move':
                if state['status'] != 'pending':
                    raise ValueError('only pending jobs can be moved to the front')
                state['priority'] = min([s.get('priority', 0) for s in store.states()] or [0]) - 1
            store.save(state)


def show_status(store, as_json=False):
    with store.lock():
        reconcile(store)
        states = store.states()
        control = store.control()
    entries = []
    for state in states:
        progress, done = progress_for(store, state)
        entry = dict(state)
        entry['completed_cells'] = done
        expected = progress.get('total_cells')
        if expected is None:
            spec = read_json(store.job_dir(state['id']) / 'spec.json', {})
            snapshot = spec.get('snapshot', {})
            args = snapshot.get('args', {})
            keys = args.get('keys')
            if keys is None:
                datasets = snapshot.get('dataset_configs', {})
                fold_count = sum(int(cfg.get('num_subjects', 0)) for cfg in datasets.values())
            else:
                fold_count = len(keys)
            if fold_count and args.get('seeds') and args.get('methods'):
                expected = fold_count * len(args['seeds']) * len(args['methods'])
        entry['total_cells'] = expected
        entries.append(entry)
    manager = read_json(store.root / 'manager.json', {})
    manager_live = manager.get('status') == 'running' and process_alive(manager.get('process'))
    if as_json:
        print(json.dumps({'manager_running': manager_live, 'scheduling_paused': control['paused'],
                          'jobs': entries}, ensure_ascii=False, indent=2))
        return
    print(f"manager={'running' if manager_live else 'stopped'} scheduling={'paused' if control['paused'] else 'enabled'} root={store.root}")
    for state in entries:
        a = state.get('args', {})
        dataset = a.get('datasets') or a.get('dataset')
        teacher = a.get('teachers') or a.get('teacher')
        student = a.get('students') or a.get('student')
        parameters = ' '.join(f'{key}={a.get(key)}' for key in
                              ('seeds', 'methods', 'epochs', 'batch_size', 'lr',
                               'weight_decay', 'lam_kd', 'temperature', 'lam_mmd'))
        print(f"{state['id']} {state['status']:9s} GPU={state.get('gpu')} "
              f"UUID={state.get('gpu_uuid') or '-'} "
              f"folds={state['completed_cells']}/{state['total_cells'] or '?'} "
              f"{dataset} {teacher}->{student} {parameters}")
        if state['status'] == 'completed':
            registration = state.get('log_registration', {})
            print(f"  exp_log={registration.get('status', 'pending')} "
                  f"{registration.get('reason', '')}")
        if state.get('message'):
            print('  ' + state['message'])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    run = subparsers.add_parser('run', help='dispatch manual trials when verified GPU slots are idle')
    run.add_argument('--root', '--queue-root', default=str(DEFAULT_ROOT))
    run.add_argument('--gpus', nargs='+', required=True, type=int)
    run.add_argument('--per-gpu', type=int, default=1)
    run.add_argument('--poll-seconds', type=float, default=15)
    run.add_argument('--max-used-mb', type=int, default=512)
    run.add_argument('--max-utilization', type=int, default=10)
    run.add_argument('--min-free-mb', type=int, default=2048)
    run.add_argument('--once', action='store_true', help=argparse.SUPPRESS)
    status = subparsers.add_parser('status')
    status.add_argument('--root', '--queue-root', default=str(DEFAULT_ROOT))
    status.add_argument('--json', action='store_true')
    register = subparsers.add_parser('register', help='register complete results in docs/exp_log.csv')
    register.add_argument('jobs', nargs='*')
    register.add_argument('--root', '--queue-root', default=str(DEFAULT_ROOT))
    register.add_argument('--watch', action='store_true',
                          help='watch completed jobs without restarting the training manager')
    register.add_argument('--poll-seconds', type=float, default=15)
    for name in ('pause', 'resume', 'retry', 'stop', 'cancel', 'move'):
        command = subparsers.add_parser(name)
        command.add_argument('jobs', nargs='*')
        command.add_argument('--root', '--queue-root', default=str(DEFAULT_ROOT))
    worker = subparsers.add_parser('_worker', help=argparse.SUPPRESS)
    worker.add_argument('--root', required=True)
    worker.add_argument('--job', required=True)
    worker.add_argument('--gpu', type=int, required=True)
    worker.add_argument('--launch-token', required=True)
    args = parser.parse_args(argv)
    if args.command == 'register' and (not math.isfinite(args.poll_seconds) or
                                       args.poll_seconds <= 0):
        parser.error('poll-seconds must be positive and finite')
    if args.command == 'run' and (args.per_gpu < 1 or args.poll_seconds <= 0 or
                                  args.min_free_mb < 0 or args.max_used_mb < 0 or
                                  not 0 <= args.max_utilization <= 100 or
                                  any(gpu < 0 for gpu in args.gpus) or
                                  len(set(args.gpus)) != len(args.gpus)):
        parser.error('invalid concurrency, polling interval or GPU thresholds')
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.command == 'run':
            return run_manager(args)
        if args.command == '_worker':
            return run_worker(args)
        if args.command == 'register' and args.watch:
            if args.jobs:
                raise ValueError('register --watch scans the entire queue; omit job IDs')
            return watch_registration(args)
        store = QueueStore(validated_root(args.root))
        if args.command == 'status':
            show_status(store, args.json)
        elif args.command == 'register':
            results = register_completed_logs(store, args.jobs, force=True)
            if not results:
                print('[experiment-log] no completed jobs selected', flush=True)
            return 1 if any(r and r.get('status') == 'error' for r in results.values()) else 0
        else:
            mutate_job(store, args.command, args.jobs)
            show_status(store)
        return 0
    except (ValueError, RuntimeError) as exc:
        print(f'[queue] error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
