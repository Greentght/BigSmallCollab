"""Frozen tasks, validation-only selection and paths for few-shot tuning."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import yaml
from experiments.storage import REPORTS_ROOT, require_external_output, require_report_output

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'configs/experiments/fewshot_dml_ds_tuning_v1.yaml'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cfg_load():
    return yaml.safe_load(CONFIG.read_text())


def roots(cfg):
    return require_external_output(cfg['output_root']), REPORTS_ROOT/cfg['report_subdirectory']


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp'); tmp.write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
    tmp.replace(path)


def report_json(report, name, value):
    write_json(require_report_output(report/name), value)


def fingerprint():
    paths=[CONFIG, ROOT/'config.py', ROOT/'models/base.py', ROOT/'collab/dcm.py', ROOT/'collab/dml_ds_heads.py',
           ROOT/'experiments/distill/run_fewshot_structure_kd.py', ROOT/'experiments/finetune/labram_alexmi_broadband.py',
           ROOT/'experiments/bidir/run_mirepnet_fewshot_8_30.py', ROOT/'experiments/finetune/run_neurogpt_loso.py',
           ROOT/'configs/experiments/loso/neurogpt.yaml']
    for pattern in ('experiments/bidir/tune_fewshot_*.py','data/*.py','configs/datasets/*.yaml','configs/models/*.yaml'):
        paths.extend(ROOT.glob(pattern))
    for model in ('mirepnet','cbramod','neurogpt','ifnet','eegnet','adfcnn'):
        paths.extend((ROOT/'models'/model).rglob('*.py'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def training(cfg, teacher, student, candidate):
    t=dict(cfg['teacher_training'][teacher]); a=cfg['stage_a'][candidate]
    weights=cfg['student_weight_decay_alternative'] if a['wd']=='alternative' else cfg['student_weight_decay']
    s={'optimizer':'adamw','lr':a['lr'],'weight_decay':weights[student],
       'batch_size':t['batch_size']//a['batch_divisor']}
    return t,s


def make_task(cfg, frozen_hash, teacher, dataset, subject, seed, target, method,
              student=None, a='A0', b='B0', phase='search', selection_hash=None, smoke=False):
    t=dict(cfg['teacher_training'][teacher]); s=None
    if student: t,s=training(cfg,teacher,student,a)
    spec={'experiment':cfg['experiment_id'],'frozen_hash':frozen_hash,'teacher':teacher,'dataset':dataset,
          'subject':subject,'seed':seed,'target':target,'method':method,'student':student,
          'teacher_training':t,'student_training':s,'alpha':cfg['alpha'],
          'micro_batch':cfg['micro_batch'],'phase':phase,'smoke':smoke,
          'selection_hash':selection_hash}
    if method=='dml_ds':spec.update(cfg['stage_b'][b])
    if target=='teacher':spec['student_training']=None
    return spec


def task_path(root, spec, tag=None):
    base=root/('smoke' if spec['smoke'] else 'tasks')
    if tag:base/=tag
    return base/digest(spec)


def install_task(root,spec,tag=None):
    folder=task_path(root,spec,tag); folder.mkdir(parents=True,exist_ok=True)
    path=folder/'task.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('Task identity collision')
    if not path.exists():write_json(path,spec)
    return folder


def load_result(root,spec):
    folder=task_path(root,spec); result=json.loads((folder/'evaluation.json').read_text())
    if result['task']!=spec or not (folder/'predictions.npz').is_file():raise ValueError(f'Invalid completion: {folder}')
    expected='validation' if spec['phase']=='search' else 'test'
    if result['evaluation_scope']!=expected:raise ValueError('Evaluation scope mismatch')
    return result


def group_specs(cfg, frozen_hash, teacher, dataset, student, method, a, b='B0', phase='search',selection_hash=None):
    target='pair' if method=='dml_ds' else 'student'
    return [make_task(cfg,frozen_hash,teacher,dataset,subject,seed,target,method,student,a,b,phase,selection_hash)
            for seed in cfg['seeds'] for subject in range(cfg['datasets'][dataset])]


def mean_score(cfg,root,specs,model):
    by_seed={seed:[] for seed in cfg['seeds']}
    for spec in specs:
        r=load_result(root,spec); n=r['evaluation_count']
        by_seed[spec['seed']].append(Fraction(r['correct'][model]*100,n))
    count=cfg['datasets'][specs[0]['dataset']]
    if any(len(v)!=count for v in by_seed.values()):raise ValueError('Incomplete subjects or seeds')
    return sum(sum(v)/len(v) for v in by_seed.values())/len(by_seed)


def choose(candidates, scores):
    # Preserve configuration order on exact ties; never use test scores.
    return max(candidates,key=lambda candidate:scores[candidate])
