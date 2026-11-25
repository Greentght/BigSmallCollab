"""Gate a fixed HFD LOSO experiment on two GPU preflights, then report it."""

from collections import Counter
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import time

import yaml

from data.loso_source import sha256_file
from experiments.bidir.run_hfd_loso import CONFIG, implementation_hashes
from experiments.bidir.summarize_hfd_loso import summarize
from experiments.distill.queue import (DEFAULT_ROOT, QueueStore, atomic_json,
                                       enqueue_snapshot, file_lock, validated_root)
from experiments.storage import require_report_output


def frozen_experiment(config_path):
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    if (cfg['dataset'] != 'BNCI2014004' or cfg['selected_session'] != 'session_3' or
            set(cfg['pairs']) != {'mirepnet_ifnet', 'cbramod_eegnet'} or
            cfg['conditions'] != ['ce', 'ce_hfd', 'dml_kl', 'dml_kl_hfd'] or
            cfg['folds'] != list(range(9)) or cfg['seeds'] != [666, 667, 668]):
        raise ValueError('The HFD controller requires the agreed two-pair 216-run protocol')
    implementation = {}
    for pair in cfg['pairs']:
        files = implementation_hashes(pair)
        implementation[pair] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    verifier = Path(__file__).with_name('verify_hfd_preflight.py')
    return cfg, {'config': str(config_path), 'config_sha256': sha256_file(config_path),
                 'implementation_sha256': implementation, 'verifier_sha256': sha256_file(verifier)}


def snapshot(frozen, pair, condition, fold, seed, stage):
    return {'kind': 'hfd_loso_004', 'stage': stage, 'config': frozen['config'],
            'config_sha256': frozen['config_sha256'],
            'implementation_sha256': frozen['implementation_sha256'][pair],
            'verifier_sha256': frozen['verifier_sha256'],
            'pair': pair, 'condition': condition, 'fold': fold, 'seed': seed,
            'args': {'dataset': 'BNCI2014004', 'folds': [fold],
                     'seeds': [seed], 'methods': [condition], 'pair': pair}}


def submit(frozen, pair, condition, fold, seed, stage, root):
    return enqueue_snapshot(snapshot(frozen, pair, condition, fold, seed, stage), root=root,
                            label=f'HFD {stage} {pair} {condition} S{fold + 1:02d} seed={seed}')


def state_counts(store, jobs):
    with store.lock():
        states = [store.load(job_id) for job_id in jobs]
    return dict(Counter(state['status'] for state in states)), states


def verify_preflights(store, ids, frozen):
    for pair, job_id in ids.items():
        progress = json.loads((store.job_dir(job_id) / 'progress.json').read_text())
        if (progress.get('status') != 'completed' or progress.get('pair') != pair or
                progress.get('config_sha256') != frozen['config_sha256'] or
                progress.get('implementation_sha256') != frozen['implementation_sha256'][pair] or
                progress.get('verifier_sha256') != frozen['verifier_sha256'] or
                not progress.get('validation_only') or not progress.get('resume_exactly_matches') or
                not progress.get('hfd_gradient_isolation_checked')):
            raise ValueError(f'Invalid GPU preflight confirmation: {job_id}')


def supervise(config_path, root, poll_seconds):
    cfg, frozen = frozen_experiment(config_path)
    report = require_report_output(Path(cfg['report_root']) / 'queue_controller.json')
    report.parent.mkdir(parents=True, exist_ok=True)
    store = QueueStore(validated_root(root))
    lock = report.parent / 'queue_controller.lock'
    with file_lock(lock, blocking=False):
        old = json.loads(report.read_text()) if report.exists() else None
        if old and old.get('frozen') != frozen:
            raise ValueError('Existing controller has different frozen configuration or code')
        status = old or {'frozen': frozen, 'preflight_jobs': {}, 'formal_jobs': {}}

        def save(stage, counts=None, **extra):
            status.update(stage=stage, counts=counts or {}, **extra)
            atomic_json(report, status)

        try:
            for pair in cfg['pairs']:
                status['preflight_jobs'][pair] = submit(frozen, pair, 'dml_kl_hfd', 0, 666, 'smoke', root)
                save('preflight_queued')
            while True:
                counts, states = state_counts(store, status['preflight_jobs'].values())
                save('preflight', counts)
                if any(item['status'] in ('failed', 'cancelled') for item in states):
                    save('preflight_failed', counts, failures=[item for item in states
                         if item['status'] in ('failed', 'cancelled')])
                    return 1
                if all(item['status'] == 'completed' for item in states):
                    break
                time.sleep(poll_seconds)
            verify_preflights(store, status['preflight_jobs'], frozen)
            _, current = frozen_experiment(config_path)
            if current != frozen:
                raise ValueError('Configuration or implementation changed after preflight')
            # Interleave pairs and conditions rather than enqueueing an entire
            # model family before any of the other comparisons can start.
            for fold, seed, condition, pair in product(
                    cfg['folds'], cfg['seeds'], cfg['conditions'], cfg['pairs']):
                key = f'{pair}/{condition}/fold_{fold}/seed_{seed}'
                status['formal_jobs'][key] = submit(frozen, pair, condition, fold, seed, 'formal', root)
                save('submitting_formal', submitted=len(status['formal_jobs']))
            if len(status['formal_jobs']) != 216 or len(set(status['formal_jobs'].values())) != 216:
                raise ValueError('Formal queue does not contain 216 unique paired trainings')
            while True:
                counts, states = state_counts(store, status['formal_jobs'].values())
                failures = [dict(id=item['id'], status=item['status'], message=item['message'])
                            for item in states if item['status'] in ('failed', 'cancelled')]
                save('formal', counts, failures=failures)
                if all(item['status'] in ('completed', 'failed', 'cancelled') for item in states):
                    if failures:
                        save('formal_failed', counts, failures=failures)
                        return 1
                    break
                time.sleep(poll_seconds)
            _, current = frozen_experiment(config_path)
            if current != frozen:
                raise ValueError('Configuration or implementation changed before report generation')
            summarize(cfg, Path(frozen['config']))
            save('completed', counts)
            return 0
        except Exception as exc:
            save('error', error=f'{type(exc).__name__}: {exc}')
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 10:
        parser.error('poll interval must be at least 10 seconds')
    return supervise(args.config, args.root, args.poll_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
