"""Run only DML-DS+HFD, then compare against completed DML-DS baselines."""

import argparse
from collections import Counter
import hashlib
from itertools import product
import json
from pathlib import Path
import time

import yaml

from data.loso_source import sha256_file
from experiments.bidir.run_dml_ds_hfd_fewshot import CONFIG, ROOT, implementation_hashes
from experiments.bidir.summarize_dml_ds_hfd_fewshot import summarize
from experiments.distill.queue import (DEFAULT_ROOT, QueueStore, atomic_json,
                                       enqueue_snapshot, file_lock, validated_root)
from experiments.storage import require_report_output


def frozen_experiment(config_path):
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    base_path = ROOT / cfg['baseline_config']
    base = yaml.safe_load(base_path.read_text())
    if (cfg['dataset'] != 'BNCI2014004' or cfg['pair'] != 'cbramod_eegnet' or
            cfg['condition'] != 'dml_ds_hfd' or cfg['subjects'] != list(range(9)) or
            cfg['seeds'] != [666, 667, 668] or base['calibration_fraction'] != 0.3 or
            base['test_fraction'] != 0.7 or base['teacher_training']['cbramod']['epochs'] != 20):
        raise ValueError('Unexpected DML-DS+HFD few-shot protocol')
    digest = hashlib.sha256(json.dumps(implementation_hashes(), sort_keys=True).encode()).hexdigest()
    verifier = Path(__file__).with_name('verify_dml_ds_hfd_preflight.py')
    return cfg, {'config': str(config_path), 'config_sha256': sha256_file(config_path),
                 'baseline_config_sha256': sha256_file(base_path),
                 'implementation_sha256': {cfg['pair']: digest},
                 'verifier_sha256': sha256_file(verifier)}


def snapshot(frozen, pair, subject, seed, stage):
    return {'kind': 'dml_ds_hfd_004', 'stage': stage, 'config': frozen['config'],
            'config_sha256': frozen['config_sha256'],
            'implementation_sha256': frozen['implementation_sha256'][pair],
            'verifier_sha256': frozen['verifier_sha256'],
            'pair': pair, 'subject': subject, 'seed': seed,
            'args': {'dataset': 'BNCI2014004', 'pair': pair, 'subjects': [subject],
                     'seeds': [seed], 'methods': ['dml_ds_hfd']}}


def submit(frozen, pair, subject, seed, stage, root):
    return enqueue_snapshot(snapshot(frozen, pair, subject, seed, stage), root=root,
        label=f'DML-DS+HFD {stage} {pair} S{subject + 1:02d} seed={seed}')


def state_counts(store, jobs):
    with store.lock():
        states = [store.load(job_id) for job_id in jobs]
    return dict(Counter(state['status'] for state in states)), states


def verify_preflight(store, job_id, pair, frozen):
    progress = json.loads((store.job_dir(job_id) / 'progress.json').read_text())
    if (progress.get('status') != 'completed' or progress.get('pair') != pair or
            progress.get('config_sha256') != frozen['config_sha256'] or
            progress.get('implementation_sha256') != frozen['implementation_sha256'][pair] or
            progress.get('verifier_sha256') != frozen['verifier_sha256'] or
            not progress.get('calibration_only') or not progress.get('resume_exactly_matches') or
            not progress.get('hfd_gradient_isolation_checked')):
        raise ValueError(f'Invalid GPU DML-DS+HFD preflight: {job_id}')


def supervise(config_path, root, poll_seconds):
    cfg, frozen = frozen_experiment(config_path)
    report = require_report_output(Path(cfg['report_root']) / 'queue_controller.json')
    report.parent.mkdir(parents=True, exist_ok=True)
    store = QueueStore(validated_root(root))
    with file_lock(report.parent / 'queue_controller.lock', blocking=False):
        old = json.loads(report.read_text()) if report.exists() else None
        if old and old.get('frozen') != frozen:
            raise ValueError('Existing controller has a different frozen configuration or code')
        status = old or {'frozen': frozen, 'preflight_jobs': {}, 'formal_jobs': {}}

        def save(stage, counts=None, **extra):
            status.update(stage=stage, counts=counts or {}, **extra)
            atomic_json(report, status)

        try:
            pair = cfg['pair']
            status['preflight_jobs'][pair] = submit(frozen, pair, 0, 666, 'smoke', root)
            save('preflight_queued')
            while True:
                counts, states = state_counts(store, status['preflight_jobs'].values())
                save('preflight', counts)
                if any(item['status'] in ('failed', 'cancelled') for item in states):
                    save('preflight_failed', counts, failures=[item for item in states
                         if item['status'] in ('failed', 'cancelled')])
                    return 1
                if all(item['status'] == 'completed' for item in states): break
                time.sleep(poll_seconds)
            verify_preflight(store, status['preflight_jobs'][pair], pair, frozen)
            if frozen_experiment(config_path)[1] != frozen:
                raise ValueError('Configuration or implementation changed after preflight')
            for subject, seed in product(cfg['subjects'], cfg['seeds']):
                key = f'{pair}/subject_{subject}/seed_{seed}'
                status['formal_jobs'][key] = submit(frozen, pair, subject, seed, 'formal', root)
                save('submitting_formal', submitted=len(status['formal_jobs']))
            if len(status['formal_jobs']) != 27 or len(set(status['formal_jobs'].values())) != 27:
                raise ValueError('Expected 27 unique add-on trainings')
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
            while True:
                missing = [f'{subject}/{seed}' for subject, seed in product(cfg['subjects'], cfg['seeds'])
                           if not (Path(cfg['baseline_root']) / f'subject_{subject}' /
                                   f'seed_{seed}' / 'evaluation.json').is_file()]
                if not missing: break
                save('waiting_for_baseline', counts, missing_baselines=missing)
                time.sleep(poll_seconds)
            if frozen_experiment(config_path)[1] != frozen:
                raise ValueError('Configuration or implementation changed before reporting')
            summarize(cfg, Path(frozen['config']))
            save('completed', counts, missing_baselines=[])
            return 0
        except Exception as exc:
            save('error', error=f'{type(exc).__name__}: {exc}')
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--poll-seconds', type=int, default=15)
    args = parser.parse_args()
    if args.poll_seconds < 5: parser.error('poll interval must be at least 5 seconds')
    return supervise(args.config, args.root, args.poll_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
