"""The few-shot controller must gate the full experiment on both preflights."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json

from experiments.bidir import queue_hfd_fewshot as queue


def test_failed_preflight_blocks_formal_and_success_submits_all_subjects():
    cfg = {'pairs': {'mirepnet_ifnet': {}, 'cbramod_eegnet': {}},
           'conditions': ['ce', 'ce_hfd', 'dml_kl', 'dml_kl_hfd'],
           'subjects': list(range(9)), 'seeds': [666, 667, 668], 'report_root': 'unused'}
    frozen = {'config': '/tmp/mock-config', 'config_sha256': 'test',
              'implementation_sha256': {}, 'verifier_sha256': 'test'}
    with TemporaryDirectory(prefix='hfd-fewshot-queue-') as directory:
        root = Path(directory)
        for success in (False, True):
            verified = []

            def submit(_frozen, pair, condition, subject, seed, stage, _root):
                if stage == 'formal':
                    assert verified == [True]
                return f'{stage}-{pair}-{condition}-{subject}-{seed}'

            def states(_store, ids):
                ids = list(ids)
                status = 'completed' if success or ids[0].startswith('formal-') else 'failed'
                values = [{'id': job, 'status': status, 'message': 'synthetic'} for job in ids]
                return {status: len(values)}, values

            path = root / ('ok' if success else 'fail') / 'queue_controller.json'
            with patch.object(queue, 'frozen_experiment', return_value=(cfg, frozen)), \
                    patch.object(queue, 'require_report_output', return_value=path), \
                    patch.object(queue, 'validated_root', return_value=root), \
                    patch.object(queue, 'QueueStore', return_value=object()), \
                    patch.object(queue, 'submit', side_effect=submit), \
                    patch.object(queue, 'state_counts', side_effect=states), \
                    patch.object(queue, 'verify_preflights', side_effect=lambda *_: verified.append(True)), \
                    patch.object(queue, 'summarize') as summarize:
                code = queue.supervise('/tmp/mock-config', root, 10)
            report = json.loads(path.read_text())
            if success:
                assert code == 0 and report['stage'] == 'completed'
                assert len(report['formal_jobs']) == len(set(report['formal_jobs'].values())) == 216
                summarize.assert_called_once()
            else:
                assert code == 1 and report['stage'] == 'preflight_failed'
                assert not report['formal_jobs']
                summarize.assert_not_called()
