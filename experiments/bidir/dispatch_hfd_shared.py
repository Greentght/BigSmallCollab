"""Dispatch this HFD experiment to GPUs with fewer than three processes.

This companion uses the existing queue's lock and worker protocol. It only
dispatches jobs listed by the frozen HFD controller; the general queue manager
and all other jobs retain their existing scheduling settings.
"""

import argparse
from collections import Counter
import signal
import subprocess
import time
import uuid
from pathlib import Path

from experiments.distill.queue import (
    ACTIVE, DEFAULT_ROOT, PROJECT_ROOT, QueueStore, atomic_json, descendant_of,
    file_lock, process_alive, process_identity, progress_for, query_gpus,
    read_json, utc_now, validated_root, worker_command, worker_lock_held,
)
from experiments.storage import require_report_output


CONTROLLER = PROJECT_ROOT / 'results/hfd_loso_004_v1/queue_controller.json'
STATUS = PROJECT_ROOT / 'results/hfd_loso_004_v1/shared_dispatcher.json'


def effective_occupancy(gpu, info, states):
    """Count actual CUDA processes plus queued workers not visible on the GPU yet."""
    active = [state for state in states if state['status'] in ACTIVE and
              (state.get('gpu_uuid') == info['uuid'] or
               (not state.get('gpu_uuid') and state.get('gpu') == gpu))]
    reserved = 0
    for state in active:
        identities = [identity for identity in (state.get('worker'), state.get('child')) if identity]
        represented = any(descendant_of(pid, identity)
                          for pid in info['pids'] for identity in identities)
        if not represented:
            reserved += 1
    return len(info['pids']) + reserved, reserved


def suitable_gpus(gpus, states, process_limit, min_free_mb):
    candidates = []
    for gpu, info in gpus.items():
        effective, reserved = effective_occupancy(gpu, info, states)
        free = info['total_mb'] - info['used_mb']
        if effective < process_limit and free >= min_free_mb:
            candidates.append((effective, -free, gpu, info['uuid']))
    return sorted(candidates)


def reconcile_hfd(store, states, allowed):
    """Recover only this experiment's orphaned reservations, under queue lock."""
    for state in states:
        if state['id'] not in allowed or state['status'] not in ACTIVE:
            continue
        if process_alive(state.get('worker')) or process_alive(state.get('child')):
            continue
        if worker_lock_held(store, state['id']):
            continue
        if (state['status'] == 'starting' and process_alive(state.get('launcher')) and
                time.time() - state.get('launch_time', 0) < 60):
            continue
        progress, _ = progress_for(store, state)
        if progress.get('status') == 'completed':
            state.update(status='completed', message='completed HFD result recovered')
        elif progress.get('status') == 'failed':
            state.update(status='failed', message='HFD runner failed; inspect its logs')
        elif (store.job_dir(state['id']) / 'stop').exists():
            state.update(status='paused', message='HFD job paused')
        else:
            state.update(status='pending', message='interrupted HFD job queued to resume')
        state['worker'] = state['child'] = state['launcher'] = None
        store.save(state)


def dispatch_hfd(store, state, gpu, gpu_uuid):
    """Reserve a specific HFD job and spawn its standard queue worker."""
    state.update(status='starting', gpu=gpu, gpu_uuid=gpu_uuid,
                 launch_token=uuid.uuid4().hex, worker=None, child=None,
                 launcher=process_identity(), launch_time=time.time(),
                 message=f'launching HFD on shared GPU {gpu}')
    store.save(state)
    folder = store.job_dir(state['id'])
    try:
        with (folder / 'worker.log').open('a', encoding='utf-8') as stream:
            process = subprocess.Popen(worker_command(store, state), cwd=PROJECT_ROOT,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
        updated = store.load(state['id'])
        if updated['status'] == 'starting':
            updated['worker'] = process_identity(process.pid)
            store.save(updated)
    except Exception as exc:
        state.update(status='failed', message=f'HFD worker launch failed: {exc}')
        store.save(state)
        raise


def tick(store, controller_path, process_limit, min_free_mb):
    controller = read_json(controller_path)
    if not controller:
        return {'stage': 'waiting_for_controller'}, False
    if controller.get('stage') in {'completed', 'preflight_failed', 'formal_failed', 'error'}:
        return {'stage': 'controller_' + controller['stage']}, True
    allowed = set(controller.get('preflight_jobs', {}).values()) | set(
        controller.get('formal_jobs', {}).values())
    with store.lock():
        states = store.states()
        reconcile_hfd(store, states, allowed)
        states = store.states()
        hfd_states = [state for state in states if state['id'] in allowed]
        summary = {'stage': controller.get('stage'), 'queue_counts': dict(
            Counter(state['status'] for state in hfd_states))}
        if store.control().get('paused', False):
            summary['scheduling'] = 'paused_by_shared_queue'
            return summary, False
        pending = [state for state in hfd_states if state['status'] == 'pending']
        if not pending:
            summary['scheduling'] = 'no_pending_hfd_jobs'
            return summary, False
        # Verify the job's frozen experiment identity before dispatching it.
        frozen = controller['frozen']
        eligible = []
        for state in pending:
            spec = read_json(store.job_dir(state['id']) / 'spec.json', {})
            snapshot = spec.get('snapshot', {})
            pair = snapshot.get('pair')
            if (snapshot.get('kind') != 'hfd_loso_004' or
                    snapshot.get('config_sha256') != frozen['config_sha256'] or
                    snapshot.get('implementation_sha256') != frozen['implementation_sha256'].get(pair) or
                    snapshot.get('verifier_sha256') != frozen['verifier_sha256']):
                raise ValueError(f"Unexpected queued HFD job identity: {state['id']}")
            eligible.append(state)
        gpus = query_gpus()
        summary['gpus'] = {
            str(gpu): {'processes': len(info['pids']),
                       'reserved': effective_occupancy(gpu, info, states)[1],
                       'free_mb': info['total_mb'] - info['used_mb']}
            for gpu, info in gpus.items()}
        candidates = suitable_gpus(gpus, states, process_limit, min_free_mb)
        if not candidates:
            summary['scheduling'] = 'waiting_for_gpu_below_process_limit'
            return summary, False
        effective, negative_free, gpu, gpu_uuid = candidates[0]
        state = eligible[0]
        dispatch_hfd(store, state, gpu, gpu_uuid)
        summary.update(scheduling='dispatched', job=state['id'], gpu=gpu,
                       previous_effective_processes=effective, previous_free_mb=-negative_free)
        print(f"[hfd-gpu] dispatched {state['id']} on GPU {gpu}; "
              f'{effective} processes/reservations and {-negative_free} MiB free before launch', flush=True)
        return summary, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--controller', type=Path, default=CONTROLLER)
    parser.add_argument('--process-limit', type=int, default=3)
    parser.add_argument('--min-free-mb', type=int, default=2048)
    parser.add_argument('--poll-seconds', type=float, default=10)
    args = parser.parse_args()
    if args.process_limit < 1 or args.min_free_mb < 0 or args.poll_seconds < 1:
        parser.error('invalid GPU process, memory or polling limit')
    status = require_report_output(STATUS)
    status.parent.mkdir(parents=True, exist_ok=True)
    store = QueueStore(validated_root(args.root))
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    with file_lock(status.parent / 'shared_dispatcher.lock', blocking=False):
        while not stopping:
            try:
                state, done = tick(store, args.controller, args.process_limit, args.min_free_mb)
            except RuntimeError as exc:
                # nvidia-smi failures must never be interpreted as idle GPUs.
                state, done = {'stage': 'waiting', 'scheduling': str(exc)}, False
            except Exception as exc:
                atomic_json(status, {'stage': 'error', 'error': f'{type(exc).__name__}: {exc}',
                                     'updated_at': utc_now()})
                raise
            state.update(updated_at=utc_now(), process_limit=args.process_limit,
                         min_free_mb=args.min_free_mb, process=process_identity())
            atomic_json(status, state)
            if done:
                break
            deadline = time.monotonic() + (0.5 if state.get('scheduling') == 'dispatched'
                                           else args.poll_seconds)
            while not stopping and time.monotonic() < deadline:
                time.sleep(min(0.25, max(0, deadline - time.monotonic())))


if __name__ == '__main__':
    main()
