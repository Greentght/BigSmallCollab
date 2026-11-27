"""Run a DS-only ablation after matching the completed MIRepNet DML-DS run."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import fcntl
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

from experiments.bidir.queue_dcm_loso import gpu_processes, job_is_running
from experiments.bidir.run_mirepnet_ds_only_fewshot import CONFIG, ROOT, code_hashes
from experiments.storage import REPORTS_ROOT, require_external_output, require_report_output

MODULE = 'experiments.bidir.run_mirepnet_ds_only_fewshot'
REPORT = REPORTS_ROOT / 'ds_only_mirepnet_fewshot_8_30_v1'
BASE_CONFIG = ROOT / 'configs/experiments/fewshot_dml_ds_mirepnet_8_30.yaml'


def write_status(name, value):
    path = require_report_output(REPORT / name)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n')
    os.replace(temp, path)


def active_workers():
    workers = {}
    for line in subprocess.check_output(['ps', '-eo', 'pid,args'], text=True).splitlines():
        parts = line.split()
        if MODULE in parts and '--gpu' in parts:
            workers[int(parts[0])] = int(parts[parts.index('--gpu') + 1])
    return workers


def available_devices(max_workers=6):
    active = active_workers()
    if len(active) >= max_workers: return []
    result = []
    for gpu, registered in sorted(gpu_processes().items()):
        reserved = sum(assigned == gpu and pid not in registered for pid, assigned in active.items())
        result.extend([gpu] * max(0, 3 - len(registered) - reserved))
    return result[:max(0, max_workers - len(active))]


def jobs(cfg):
    for dataset, count in cfg['datasets'].items():
        for subject, seed, student in itertools.product(range(count), cfg['seeds'], cfg['students']):
            yield dataset, subject, seed, student


def folder_for(root, job, smoke_tag=None):
    dataset, subject, seed, student = job
    base = root / ('smoke' if smoke_tag else 'runs')
    if smoke_tag: base /= smoke_tag
    return base / dataset / 'ds_only' / student / f'subject_{subject}' / f'seed_{seed}'


def base_folder(root, job, kind):
    dataset, subject, seed, student = job
    model = 'mirepnet' if kind == 'teacher' else student
    return root/'runs'/dataset/kind/model/f'subject_{subject}'/f'seed_{seed}'


def start_one(root, job, gpu, smoke_tag=None, stop_after=None):
    dataset, subject, seed, student = job
    folder = folder_for(root, job, smoke_tag); folder.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-u', '-B', '-m', MODULE, '--dataset', dataset, '--student', student,
           '--subject', str(subject), '--seed', str(seed), '--gpu', str(gpu)]
    if smoke_tag: cmd += ['--smoke', '--smoke-tag', smoke_tag]
    if stop_after: cmd += ['--stop-after', str(stop_after)]
    log = (folder/'worker.log').open('a')
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
    except Exception:
        log.close(); raise
    return proc, log, folder


def assert_matched(ds_identity, teacher, student, dml, student_name):
    for key in ('dataset','subject','seed','source_manifest_sha256','train_uid_sha256','test_uid_sha256'):
        if any(other[key] != ds_identity[key] for other in (teacher,student,dml)):
            raise ValueError(f'Unmatched {key} for {student_name}')
    for key in ('epochs','batch_size','mirepnet_preprocessing','mirepnet_ea_sha256','pretrained_sha256'):
        if ds_identity[key] != dml[key] or ds_identity[key] != teacher[key]:
            raise ValueError(f'Unmatched MIRepNet setting {key} for {student_name}')
    if (ds_identity['initial_hashes']['mirepnet'] != teacher['initial_hashes']['mirepnet'] or
        ds_identity['initial_hashes']['mirepnet'] != dml['initial_hashes']['mirepnet'] or
        ds_identity['initial_hashes'][student_name] != student['initial_hashes'][student_name] or
        ds_identity['initial_hashes'][student_name] != dml['initial_hashes'][student_name]):
        raise ValueError(f'Unmatched model initialization for {student_name}')


def validate_config(cfg, base):
    fields = ('datasets','seeds','students','calibration_fraction','test_fraction','epochs',
              'batch_size','temperature','alpha','lambda','teacher_training','student_training')
    for field in fields:
        if cfg[field] != base[field]: raise ValueError(f'Unmatched configuration: {field}')
    if cfg['output_root'] == base['output_root']: raise ValueError('Must preserve previous results')


def preflight(root, base_root):
    base_job = ('BNCI2014001-4',0,666)
    units = [(*base_job, s) for s in ('ifnet','eegnet','adfcnn')]
    phases = [(job,'normal',None) for job in units] + [(units[0],'resume',1),(units[0],'resume',None)]
    for job, tag, stop in phases:
        folder = folder_for(root, job, tag)
        if not stop and (folder/'evaluation.json').is_file(): continue
        if stop and ((folder/'training_state.pt').exists() or (folder/'evaluation.json').exists()): continue
        while True:
            gpus = available_devices()
            if gpus: break
            write_status('queue_status.json', {'stage':'preflight_waiting_for_gpu','job':job})
            time.sleep(10)
        proc, log, folder = start_one(root, job, gpus[0], tag, stop)
        write_status('queue_status.json', {'stage':'preflight','job':job,'gpu':gpus[0],'pid':proc.pid})
        code = proc.wait(); log.close()
        if code: raise RuntimeError(f'Preflight failed: {job}, {tag}; see {folder/"worker.log"}')
    for job in units:
        ds = json.loads((folder_for(root,job,'normal')/'evaluation.json').read_text())['identity']
        # Compare to formal identities; the smoke run has two epochs, so only
        # match its identity to the previous two-epoch preflight identities.
        prior = base_root/'smoke'/'normal'/'BNCI2014001-4'
        teacher = json.loads((prior/'teacher'/'mirepnet'/'subject_0'/'seed_666'/'evaluation.json').read_text())['identity']
        student = json.loads((prior/'student'/job[3]/'subject_0'/'seed_666'/'evaluation.json').read_text())['identity']
        dml = json.loads((prior/'dml_ds'/job[3]/'subject_0'/'seed_666'/'evaluation.json').read_text())['identity']
        assert_matched(ds, teacher, student, dml, job[3])
    normal=folder_for(root,units[0],'normal'); resumed=folder_for(root,units[0],'resume')
    if json.loads((normal/'evaluation.json').read_text()) != json.loads((resumed/'evaluation.json').read_text()):
        raise RuntimeError('Interrupted resume result mismatch')
    with np.load(normal/'predictions.npz',allow_pickle=False) as a, np.load(resumed/'predictions.npz',allow_pickle=False) as b:
        if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):
            raise RuntimeError('Interrupted resume predictions mismatch')
    write_status('preflight.json', {'status':'passed','dataset':base_job[0],'subject':0,'seed':666,
        'students':[x[3] for x in units], 'checks':['8–30 Hz inputs','main and auxiliary logits',
        'gradients','same initialization and trial split as baseline','interrupted resume']})


def main():
    cfg = yaml.safe_load(CONFIG.read_text()); base = yaml.safe_load(BASE_CONFIG.read_text())
    validate_config(cfg,base)
    root=require_external_output(cfg['output_root']); root.mkdir(parents=True,exist_ok=True)
    base_root=Path(base['output_root']); REPORT.mkdir(parents=True,exist_ok=True)
    all_jobs=list(jobs(cfg)); assert len(all_jobs)==len(set(all_jobs))==423
    with (root/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            preflight(root,base_root)
            frozen={dataset:code_hashes(dataset,['mirepnet']+cfg['students']) for dataset in cfg['datasets']}
            pending=[j for j in all_jobs if not (folder_for(root,j)/'evaluation.json').is_file()]
            workers={}; failure=None
            while pending or workers:
                for pid,(proc,log,job) in list(workers.items()):
                    result=proc.poll()
                    if result is not None:
                        log.close(); del workers[pid]
                        if result: failure=f'Worker exited {result}: {job}; see {folder_for(root,job)/"worker.log"}'
                        else: print(f'Completed {job}',flush=True)
                if any(code_hashes(ds,['mirepnet']+cfg['students'])!=hashes for ds,hashes in frozen.items()):
                    failure='Frozen source or configuration changed'
                if failure: pending.clear()
                if not failure:
                    for gpu in available_devices():
                        index=next((i for i,j in enumerate(pending) if not job_is_running(folder_for(root,j))),None)
                        if index is None: break
                        job=pending.pop(index); proc,log,_=start_one(root,job,gpu)
                        workers[proc.pid]=(proc,log,job)
                        print(f'Started {job} gpu={gpu} pid={proc.pid}',flush=True)
                completed=sum((folder_for(root,j)/'evaluation.json').is_file() for j in all_jobs)
                write_status('queue_status.json', {'stage':'failed' if failure else 'running',
                    'completed':completed,'total':len(all_jobs),
                    'pending':len(all_jobs)-completed-len(active_workers()),
                    'running':active_workers(),'error':failure})
                if failure and not workers: raise RuntimeError(failure)
                if pending or workers: time.sleep(5)
            missing=[j for j in all_jobs if not (folder_for(root,j)/'evaluation.json').is_file()]
            if missing: raise RuntimeError(f'Missing {len(missing)} results after queue completion')
            from experiments.bidir.report_mirepnet_ds_only_fewshot import create_report
            create_report(cfg,base)
            write_status('queue_status.json', {'stage':'completed','completed':len(all_jobs),'total':len(all_jobs)})
        except Exception as exc:
            write_status('queue_status.json', {'stage':'failed','error':f'{type(exc).__name__}: {exc}'})
            raise


if __name__ == '__main__': main()
