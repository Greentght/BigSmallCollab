"""Preflight gates exactly the 27 new runs, with no baseline retraining."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from experiments.bidir import queue_dml_ds_hfd_fewshot as queue


def test_failed_preflight_blocks_and_success_submits_addons_only():
    with TemporaryDirectory(prefix='dml-ds-hfd-queue-') as directory:
        root = Path(directory)
        cfg = {'pair': 'cbramod_eegnet', 'subjects': list(range(9)), 'seeds': [666, 667, 668],
               'report_root': 'unused', 'baseline_root': str(root/'baseline')}
        frozen = {'config': '/tmp/mock-config', 'config_sha256': 'test',
                  'implementation_sha256': {}, 'verifier_sha256': 'test'}
        for subject in cfg['subjects']:
            for seed in cfg['seeds']:
                path = Path(cfg['baseline_root'])/f'subject_{subject}'/f'seed_{seed}'/'evaluation.json'
                path.parent.mkdir(parents=True); path.write_text('{}')
        for success in (False, True):
            verified, submitted = [], []
            def submit(_frozen, pair, subject, seed, stage, _root):
                if stage == 'formal': assert verified == [True]
                job = f'{stage}-{pair}-{subject}-{seed}'; submitted.append(job); return job
            def states(_store, ids):
                ids = list(ids)
                status = 'completed' if success or ids[0].startswith('formal-') else 'failed'
                values = [{'id': job, 'status': status, 'message': 'synthetic'} for job in ids]
                return {status: len(values)}, values
            path = root/('ok' if success else 'fail')/'queue_controller.json'
            with patch.object(queue, 'frozen_experiment', return_value=(cfg, frozen)), \
                    patch.object(queue, 'require_report_output', return_value=path), \
                    patch.object(queue, 'validated_root', return_value=root), \
                    patch.object(queue, 'QueueStore', return_value=object()), \
                    patch.object(queue, 'submit', side_effect=submit), \
                    patch.object(queue, 'state_counts', side_effect=states), \
                    patch.object(queue, 'verify_preflight', side_effect=lambda *_: verified.append(True)), \
                    patch.object(queue, 'summarize') as summarize:
                code = queue.supervise('/tmp/mock-config', root, 5)
            state = json.loads(path.read_text())
            if success:
                assert code == 0 and state['stage'] == 'completed'
                assert len(state['formal_jobs']) == len(set(state['formal_jobs'].values())) == 27
                assert len(submitted) == 28 and all(x.startswith(('smoke-', 'formal-')) for x in submitted)
                summarize.assert_called_once()
            else:
                assert code == 1 and state['stage'] == 'preflight_failed' and not state['formal_jobs']
                summarize.assert_not_called()
