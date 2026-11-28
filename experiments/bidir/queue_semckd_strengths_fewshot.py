"""Run both published SemCKD strengths, keeping every condition and result."""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import json
from pathlib import Path
import subprocess
import sys

from experiments.bidir.run_dml_ds_cross_hfd_fewshot import atomic_json
from experiments.distill.queue import file_lock, process_identity
from experiments.storage import PROJECT_ROOT, REPORTS_ROOT, require_external_output, require_report_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    external = require_external_output('/data1/llx/BigSmallcollab/results/bidir/semckd_strengths_fewshot_v1')
    external.mkdir(parents=True, exist_ok=True)
    report = require_report_output(REPORTS_ROOT / 'semckd_strengths_fewshot_v1/controller.json')
    report.parent.mkdir(parents=True, exist_ok=True)
    phases = ['fewshot_semckd_weight50_v1.yaml', 'fewshot_semckd_weight400_v1.yaml']
    completed = []
    with file_lock(external / 'controller.lock', blocking=False):
        for phase in phases:
            atomic_json(report, {'stage': 'training', 'current_config': phase, 'phases_completed': completed,
                                 'phases': phases, 'expected_new_units': 216, 'controller': process_identity()})
            command = [sys.executable, '-u', '-B', '-m', 'experiments.bidir.queue_cross_domain_features_fewshot',
                       '--config', str(PROJECT_ROOT / 'configs/experiments' / phase), '--workers', str(args.workers)]
            with (external / (Path(phase).stem + '.log')).open('a') as stream:
                code = subprocess.call(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if code:
                atomic_json(report, {'stage': 'error', 'current_config': phase, 'exit_code': code,
                                     'phases_completed': completed, 'phases': phases})
                return code
            completed.append(phase)
        subprocess.run([sys.executable, '-B', '-m', 'experiments.bidir.summarize_cross_domain_strengths'],
                       cwd=PROJECT_ROOT, check=True)
        atomic_json(report, {'stage': 'completed', 'phases_completed': completed,
                             'expected_new_units': 216, 'controller': process_identity()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
