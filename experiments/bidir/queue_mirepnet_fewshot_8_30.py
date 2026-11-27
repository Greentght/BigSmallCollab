"""Preflight and persistent, process-count-limited MIRepNet few-shot queue."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import csv
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
from experiments.bidir.run_mirepnet_fewshot_8_30 import CONFIG, ROOT, code_hashes, uid_hash
from experiments.storage import REPORTS_ROOT, require_external_output, require_report_output

MODULE = 'experiments.bidir.run_mirepnet_fewshot_8_30'
REPORT = REPORTS_ROOT / 'dml_ds_mirepnet_fewshot_8_30_v1'


def atomic_json(path, obj):
    path = require_report_output(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + '\n')
    os.replace(tmp, path)


def running_workers():
    active = {}
    for line in subprocess.check_output(['ps', '-eo', 'pid,args'], text=True).splitlines():
        parts = line.split()
        if MODULE in parts and '--gpu' in parts:
            active[int(parts[0])] = int(parts[parts.index('--gpu') + 1])
    return active


def available_devices(max_workers=6):
    active = running_workers()
    if len(active) >= max_workers: return []
    devices = gpu_processes()
    result = []
    for gpu, registered in sorted(devices.items()):
        reserved = sum(g == gpu and pid not in registered for pid, g in active.items())
        slots = max(0, 3 - len(registered) - reserved)
        result.extend([gpu] * slots)
    return result[:max(0, max_workers-len(active))]


def folder_for(root, job, smoke_tag=None):
    dataset, subject, seed, kind, student = job
    base = root / ('smoke' if smoke_tag else 'runs')
    if smoke_tag: base /= smoke_tag
    return base / dataset / kind / (student or 'mirepnet') / f'subject_{subject}' / f'seed_{seed}'


def command(job, gpu, smoke_tag=None, stop_after=None):
    dataset, subject, seed, kind, student = job
    cmd = [sys.executable, '-u', '-B', '-m', MODULE, '--dataset', dataset, '--kind', kind,
           '--subject', str(subject), '--seed', str(seed), '--gpu', str(gpu)]
    if student: cmd += ['--student', student]
    if smoke_tag: cmd += ['--smoke', '--smoke-tag', smoke_tag]
    if stop_after: cmd += ['--stop-after', str(stop_after)]
    return cmd


def start_one(root, job, gpu, smoke_tag=None, stop_after=None):
    folder = folder_for(root, job, smoke_tag); folder.mkdir(parents=True, exist_ok=True)
    log = (folder / 'worker.log').open('a')
    try:
        process = subprocess.Popen(command(job, gpu, smoke_tag, stop_after), cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, close_fds=True)
    except Exception:
        log.close(); raise
    return process, log, folder


def preflight(root):
    dataset, subject, seed = 'BNCI2014001-4', 0, 666
    units = [(dataset, subject, seed, 'teacher', None)]
    units += [(dataset, subject, seed, 'student', s) for s in ('ifnet','eegnet','adfcnn')]
    units += [(dataset, subject, seed, 'dml_ds', s) for s in ('ifnet','eegnet','adfcnn')]
    phases = [(unit, 'normal', None) for unit in units]
    phases += [(units[0], 'resume', 1), (units[0], 'resume', None),
               (units[4], 'resume', 1), (units[4], 'resume', None)]
    for job, tag, stop in phases:
        folder = folder_for(root, job, tag)
        if not stop and (folder / 'evaluation.json').is_file(): continue
        if stop and ((folder / 'training_state.pt').exists() or (folder / 'evaluation.json').exists()): continue
        while True:
            gpus = available_devices()
            if gpus: break
            atomic_json(REPORT / 'queue_status.json', {'stage':'preflight_waiting_for_gpu', 'job':job})
            time.sleep(10)
        process, log, folder = start_one(root, job, gpus[0], tag, stop)
        atomic_json(REPORT / 'queue_status.json', {'stage':'preflight', 'job':job, 'gpu':gpus[0], 'pid':process.pid})
        code = process.wait(); log.close()
        if code: raise RuntimeError(f'Preflight failed: {job}, tag={tag}; see {folder / "worker.log"}')
    for job in units:
        normal = json.loads((folder_for(root, job, 'normal') / 'evaluation.json').read_text())
        if normal['identity']['kind'] != job[3] or not normal['identity']['smoke']:
            raise RuntimeError(f'Unexpected preflight result: {job}')
    for job in (units[0], units[4]):
        normal = folder_for(root, job, 'normal'); resumed = folder_for(root, job, 'resume')
        a = json.loads((normal / 'evaluation.json').read_text())
        b = json.loads((resumed / 'evaluation.json').read_text())
        if a != b: raise RuntimeError(f'Interrupted resume result mismatch: {job}')
        with np.load(normal/'predictions.npz', allow_pickle=False) as x, np.load(resumed/'predictions.npz', allow_pickle=False) as y:
            if set(x.files) != set(y.files) or any(not np.array_equal(x[k], y[k]) for k in x.files):
                raise RuntimeError(f'Interrupted resume prediction mismatch: {job}')
    teacher = json.loads((folder_for(root, units[0], 'normal')/'evaluation.json').read_text())['identity']
    for i, student in enumerate(('ifnet','eegnet','adfcnn')):
        single = json.loads((folder_for(root, units[1+i], 'normal')/'evaluation.json').read_text())['identity']
        pair = json.loads((folder_for(root, units[4+i], 'normal')/'evaluation.json').read_text())['identity']
        if (teacher['initial_hashes']['mirepnet'] != pair['initial_hashes']['mirepnet'] or
            single['initial_hashes'][student] != pair['initial_hashes'][student] or
            teacher['train_uid_sha256'] != pair['train_uid_sha256'] or
            teacher['test_uid_sha256'] != pair['test_uid_sha256']):
            raise RuntimeError(f'Preflight init or trial alignment mismatch: {student}')
    atomic_json(REPORT/'preflight.json', {'status':'passed', 'units':len(units),
        'dataset':dataset, 'subject':subject, 'seed':seed,
        'checks':['8-30Hz teacher path', 'main and auxiliary logits', 'gradients',
                  'trial alignment', 'initial state hashes', 'teacher and DML resume']})


def jobs(cfg):
    for dataset, count in cfg['datasets'].items():
        for subject, seed in itertools.product(range(count), cfg['seeds']):
            yield (dataset, subject, seed, 'teacher', None)
            for student in cfg['students']: yield (dataset, subject, seed, 'student', student)
            for student in cfg['students']: yield (dataset, subject, seed, 'dml_ds', student)


def validate_and_report(root, cfg, all_jobs):
    rows, units = [], {}
    for job in all_jobs:
        dataset, subject, seed, kind, student = job
        folder = folder_for(root, job); path = folder/'evaluation.json'
        result = json.loads(path.read_text()); ident = result['identity']
        if (ident['dataset'], ident['subject'], ident['seed'], ident['kind'], ident['student'], ident['smoke']) != (dataset, subject, seed, kind, student, False):
            raise RuntimeError(f'Invalid identity: {path}')
        names = ['mirepnet'] if kind == 'teacher' else [student] if kind == 'student' else ['mirepnet', student]
        if ident['code_sha256'] != code_hashes(dataset, names): raise RuntimeError(f'Code identity changed: {path}')
        with np.load(folder/'predictions.npz', allow_pickle=False) as z:
            if uid_hash(z['uids'].tolist()) != ident['test_uid_sha256'] or len(z['y']) != result['test_count']:
                raise RuntimeError(f'Test UID or label count mismatch: {path}')
            for name in names:
                score = float(np.mean(z[name] == z['y']) * 100)
                if abs(score - result['accuracy_percent'][name]) > 1e-9: raise RuntimeError(f'Accuracy mismatch: {path}')
                rows.append([dataset, subject, seed, kind, student or '', name, score, str(path)])
        key=(dataset, subject, seed); units.setdefault(key,{})[(kind,student)] = ident
    for key, group in units.items():
        teacher = group['teacher',None]
        if teacher['mirepnet_preprocessing'] != 'broadband_250Hz_8_30Hz_order4_filtfilt_calibration_EA_pad45':
            raise RuntimeError(f'Incorrect teacher preprocessing: {key}')
        for student in cfg['students']:
            single, pair = group['student',student], group['dml_ds',student]
            for identity in (single,pair):
                if any(identity[k] != teacher[k] for k in ('source_manifest_sha256','train_uid_sha256','test_uid_sha256')):
                    raise RuntimeError(f'Split mismatch: {key}/{student}')
            if (pair['initial_hashes']['mirepnet'] != teacher['initial_hashes']['mirepnet'] or
                pair['initial_hashes'][student] != single['initial_hashes'][student] or
                pair['mirepnet_ea_sha256'] != teacher['mirepnet_ea_sha256']):
                raise RuntimeError(f'Init or teacher preprocessing mismatch: {key}/{student}')
    target=require_report_output(REPORT/'subject_accuracy.csv'); tmp=target.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        writer=csv.writer(f); writer.writerow(['dataset','subject_zero_based','seed','kind','student','model','accuracy_percent','source']); writer.writerows(rows)
    os.replace(tmp,target)
    summary=[]
    for dataset,count in cfg['datasets'].items():
        for kind, student, model in ([('teacher',None,'mirepnet')] +
              [('student',s,s) for s in cfg['students']] +
              [( 'dml_ds',s,m) for s in cfg['students'] for m in ('mirepnet',s)]):
            chosen=[r for r in rows if (r[0],r[3],r[4],r[5])==(dataset,kind,student or '',model)]
            if len(chosen)!=count*len(cfg['seeds']): raise RuntimeError('Summary coverage mismatch')
            value=sum(r[6] for r in chosen)/len(chosen)
            summary.append([dataset,kind,student or '',model,value])
    target=require_report_output(REPORT/'summary.csv'); tmp=target.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        writer=csv.writer(f); writer.writerow(['dataset','kind','student','model','accuracy_percent']); writer.writerows(summary)
    os.replace(tmp,target)


def main():
    cfg=yaml.safe_load(CONFIG.read_text()); root=require_external_output(cfg['output_root']); root.mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    all_jobs=list(jobs(cfg)); assert len(all_jobs)==987
    with (root/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            preflight(root)
            frozen={dataset:code_hashes(dataset,['mirepnet']+cfg['students']) for dataset in cfg['datasets']}
            pending=[j for j in all_jobs if not (folder_for(root,j)/'evaluation.json').is_file()]
            workers={}; failure=None
            while pending or workers or any(job_is_running(folder_for(root,j)) for j in pending):
                for pid,(process,log,job) in list(workers.items()):
                    code=process.poll()
                    if code is not None:
                        log.close(); del workers[pid]
                        if code: failure=f'Worker exited {code}: {job}; see {folder_for(root,job)/"worker.log"}'
                        else: print(f'Completed {job}',flush=True)
                if any(code_hashes(ds,['mirepnet']+cfg['students']) != hashes for ds,hashes in frozen.items()):
                    failure='Frozen code or config changed; refusing further dispatch.'
                if failure: pending.clear()
                if not failure:
                    for gpu in available_devices():
                        index=next((i for i,j in enumerate(pending) if not job_is_running(folder_for(root,j))),None)
                        if index is None: break
                        job=pending.pop(index); process,log,_=start_one(root,job,gpu)
                        workers[process.pid]=(process,log,job)
                        print(f'Started {job} gpu={gpu} pid={process.pid}',flush=True)
                completed=sum((folder_for(root,j)/'evaluation.json').is_file() for j in all_jobs)
                atomic_json(REPORT/'queue_status.json',{'stage':'failed' if failure else 'running', 'completed':completed,
                    'total':len(all_jobs),'pending':len(all_jobs)-completed-len(running_workers()),
                    'running':running_workers(), 'error':failure})
                if failure and not workers: raise RuntimeError(failure)
                if pending or workers: time.sleep(5)
            # Existing workers can survive controller restart; wait and recover their output.
            missing=[j for j in all_jobs if not (folder_for(root,j)/'evaluation.json').is_file()]
            if missing: raise RuntimeError(f'Missing {len(missing)} results after queue completion')
            validate_and_report(root,cfg,all_jobs)
            atomic_json(REPORT/'queue_status.json',{'stage':'completed','completed':len(all_jobs),'total':len(all_jobs)})
        except Exception as exc:
            atomic_json(REPORT/'queue_status.json',{'stage':'failed','error':f'{type(exc).__name__}: {exc}'})
            raise


if __name__ == '__main__': main()
