#!/usr/bin/env python
"""Detached end-to-end source-build, preprocessing and LOSO dispatcher."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = ROOT / 'results/reproductions/loso_config_alignment_v2/execution_logs'
SOURCE_ROOT = ROOT / 'data_cache/eegfm_alignment_v2/rebuilt'
MNE_ROOT = ROOT / 'data_cache/eegfm_alignment_v2/mne_data'
CONDA = '/home/lixinli/anaconda3/bin/conda'
MOABB_OVERLAY = '/tmp/loso_alignment_deps_moabb'
DATASETS = ('BNCI2014001-4', 'BNCI2014004', 'BNCI2015001')
BUILDER = ROOT / 'experiments/finetune/build_eegfm_reference_source.py'
PREPARER = ROOT / 'experiments/finetune/prepare_loso_alignment_inputs.py'
DISPATCHER = ROOT / 'experiments/finetune/dispatch_loso_config_alignment.py'


def log(message: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}'
    print(line, flush=True)
    with (LOG_ROOT / 'workflow.log').open('a') as f:
        f.write(line + '\n')


def save_status(phase: str, **fields) -> None:
    status = {'phase': phase, 'updated_at_epoch_s': time.time(), **fields}
    path = LOG_ROOT / 'workflow_status.json'
    tmp = path.with_suffix('.json.partial')
    tmp.write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def source_manifest_path(dataset: str) -> Path:
    return SOURCE_ROOT / dataset / 'manifest.json'


def valid_source(dataset: str) -> bool:
    path = source_manifest_path(dataset)
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text())
        expected = {'BNCI2014001-4': 2592, 'BNCI2014004': 1400,
                    'BNCI2015001': 2400}[dataset]
        pair = data['pairing_report']
        files_ok = all((path.parent / name).is_file() for name in
                       ('X.npy', 'y.npy', 'trials.csv', 'legacy_row_mapping.csv'))
        return (data.get('selected_trials') == expected and files_ok
                and pair.get('one_to_one') is True
                and pair.get('mismatch_count') == 0
                and pair.get('mapped_trials') == expected)
    except Exception:
        return False


def process_state(pid: int) -> str:
    stat = Path(f'/proc/{pid}/stat')
    if not stat.exists():
        return 'dead'
    try:
        # The command name may contain spaces, so state follows the final ')'.
        value = stat.read_text().rsplit(')', 1)[1].strip().split()[0]
        return value
    except Exception:
        return 'unknown'


def pid_alive(pid: int) -> bool:
    return process_state(pid) not in ('dead', 'Z')


def run_logged(command: list[str], log_path: Path, env: dict | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('ab', buffering=0) as out:
        proc = subprocess.Popen(command, cwd=ROOT, env=env or os.environ.copy(),
                                stdin=subprocess.DEVNULL, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True,
                                close_fds=True)
        log(f'started command pid={proc.pid}: {" ".join(command)}')
        save_status('running_command', pid=proc.pid, command=command,
                    log=str(log_path))
        code = proc.wait()
    log(f'command exited code={code}: {command[-1]}')
    return code


def ensure_sources() -> None:
    env = os.environ.copy()
    env['PYTHONPATH'] = MOABB_OVERLAY + os.pathsep + str(ROOT)
    env['MNE_DATA'] = str(MNE_ROOT)
    env['MNE_DATASETS_BNCI_PATH'] = str(MNE_ROOT)
    first_retry = True
    download_attempts = 0
    while True:
        bad = [d for d in DATASETS if not valid_source(d)]
        if not bad:
            log('all rebuilt sources have expected counts and one-to-one trial maps')
            return
        if first_retry:
            current = LOG_ROOT / 'rebuild_remaining.pid'
            if current.is_file():
                pid = int(current.read_text().strip())
                while pid_alive(pid):
                    save_status('waiting_for_active_source_builder', missing_datasets=bad,
                                builder_pid=pid)
                    time.sleep(60)
                    bad = [d for d in DATASETS if not valid_source(d)]
                    if not bad:
                        log('active source builder completed all missing datasets')
                        return
            first_retry = False
        log(f'rebuilding missing/invalid source datasets with retries: {bad}')
        save_status('rebuilding_sources', missing_datasets=bad)
        code = run_logged(
            [CONDA, 'run', '--no-capture-output', '-n', 'cbramod', 'python', '-u',
             str(BUILDER), '--datasets', *bad, '--download-retries', '5'],
            LOG_ROOT / 'rebuild_retry.log', env)
        if code != 0:
            failure_log = (LOG_ROOT / 'rebuild_retry.log').read_text(errors='replace')
            structural_markers = (
                'legacy/rebuilt trial mapping failed', 'expected 2400 selected trials',
                'expected 1400 selected trials', 'expected 2592 selected trials',
                'cannot map to one raw', 'unexpected shape',
            )
            if any(marker in failure_log for marker in structural_markers):
                save_status('source_validation_failed', datasets=bad,
                            log=str(LOG_ROOT / 'rebuild_retry.log'))
                raise RuntimeError('rebuilt source failed a structural/session/trial identity check')
            download_attempts += 1
            if download_attempts >= 4:
                save_status('source_download_failed', datasets=bad,
                            attempts=download_attempts,
                            log=str(LOG_ROOT / 'rebuild_retry.log'))
                raise RuntimeError('source download failed after four bounded attempts')
            log(f'source builder exit={code}; retry {download_attempts}/4 after 60 seconds')
            time.sleep(60)
        else:
            time.sleep(3)


def prepare_inputs() -> None:
    env = os.environ.copy()
    env['PYTHONPATH'] = MOABB_OVERLAY + os.pathsep + str(ROOT)
    env['MNE_DATA'] = str(MNE_ROOT)
    env['MNE_DATASETS_BNCI_PATH'] = str(MNE_ROOT)
    while True:
        save_status('preparing_model_inputs')
        code = run_logged(
            [CONDA, 'run', '--no-capture-output', '-n', 'cbramod', 'python', '-u',
             str(PREPARER), '--include-bridge'],
            LOG_ROOT / 'prepare_all_inputs.log', env)
        if code == 0:
            log('all reference-aligned and paired-bridge model inputs prepared')
            return
        log(f'input-preparation exit={code}; valid completed arrays will be reused; retry in 60 seconds')
        time.sleep(60)


def dispatch() -> None:
    save_status('dispatching_gpu_jobs')
    code = run_logged(
        [CONDA, 'run', '--no-capture-output', '-n', 'cbramod', 'python', '-u',
         str(DISPATCHER), '--gpus', '1,2,3,4,5,6,7,8,9',
         '--max-concurrent', '4', '--max-util', '40',
         '--min-free-mib', '10000', '--poll-seconds', '30'],
        LOG_ROOT / 'dispatcher_process.log')
    if code != 0:
        save_status('dispatcher_failed', return_code=code,
                    log=str(LOG_ROOT / 'dispatcher_process.log'))
        raise RuntimeError(f'GPU dispatcher exited with code {code}')
    save_status('complete')
    log('workflow finished all planned LOSO training cells')


def main() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log('workflow coordinator started; GPU 0 is excluded by the dispatcher')
    ensure_sources()
    prepare_inputs()
    dispatch()


if __name__ == '__main__':
    main()
