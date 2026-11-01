"""Five-dataset MIRepNet teacher-to-student feature/DiffKD LOSO experiment."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
             'TORCH_NUM_THREADS','TORCH_NUM_INTEROP_THREADS'):
    os.environ[_key] = '1'

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
torch.set_num_threads(1)
if torch.get_num_interop_threads()!=1:
    torch.set_num_interop_threads(1)
import yaml

import config as project_config
from collab import mirepnet_t2s as core
from collab import t2s_training as train
from experiments.distill import mirepnet_t2s_data as source
from experiments.storage import (DATA_CACHE_ROOT, RESULTS_ROOT, WEIGHTS_ROOT,
    REPORTS_ROOT, require_external_output, require_report_output, resolve_local_file)
from models import get_adapter

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs/experiments/loso/mirepnet_t2s_allsets_v1.yaml'
DATASETS = ('BNCI2014004','BNCI2014001','BNCI2014001-4','BNCI2015001','AlexMI')
SUBJECTS = dict(zip(DATASETS,(9,9,9,12,8)))
STUDENTS = ('ifnet','eegnet','adfcnn')
SEEDS = (666,667,668)
METHODS = ('none','CE_T2S_FEATURE','MCC_T2S_FEATURE','CE_T2S_DIFFKD','MCC_T2S_DIFFKD')


def sha(path):
    digest=hashlib.sha256()
    with Path(resolve_local_file(path)).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()


def read(path):
    return json.loads(Path(resolve_local_file(path)).read_text())


def report_json(path,value):
    path=require_report_output(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,default=str)+'\n')
    os.replace(temp,path)


def npz(path,**arrays):
    path=require_external_output(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as stream:np.savez_compressed(stream,**arrays)
    os.replace(temp,path)


def implementation():
    files=list((
        Path(__file__),ROOT/'collab/mirepnet_t2s.py',ROOT/'collab/t2s_training.py',
        ROOT/'experiments/distill/mirepnet_t2s_data.py',ROOT/'collab/mcc_diffkd.py',
        ROOT/'config.py',ROOT/'data/loso_source.py',ROOT/'data/loso_alexmi.py',
        ROOT/'experiments/distill/loso_target_cache.py',ROOT/'models/base.py'))
    for model in ('mirepnet',*STUDENTS):
        files.extend(sorted((ROOT/'models'/model).glob('*.py')))
        files.append(ROOT/'configs/models'/f'{model}.yaml')
    files.extend(ROOT/'configs/datasets'/f'{ds}.yaml' for ds in DATASETS)
    return identity({str(p.relative_to(ROOT)):sha(p) for p in files})


def load_config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if (cfg['experiment_id']!='MIREPNET_T2S_ALLSETS_LOSO_V1'
        or cfg['datasets']!=list(DATASETS) or cfg['students']!=list(STUDENTS)
        or cfg['seeds']!=list(SEEDS) or cfg['conditions']!=list(METHODS)):
        raise ValueError('fixed experiment scope differs from registered V1')
    return cfg


def report_root(cfg):
    return require_report_output(REPORTS_ROOT/cfg['report']['directory']/'dispatch.json').parent


def roots(cfg,dataset,seed,fold,smoke=False):
    run=cfg['storage']['run_id']+('_smoke' if smoke else '')
    suffix=Path(dataset)/f'seed_{seed}'/f'subject_{fold+1:02d}'
    return {name:require_external_output(base/run/suffix) for name,base in (
        ('result',RESULTS_ROOT/'distill'),('weights',WEIGHTS_ROOT),('cache',DATA_CACHE_ROOT))}


def progress(path,**values):
    if path is None:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(values,default=str)+'\n')
    os.replace(tmp,path)


def freeze(config_path,cfg):
    """Resolve recipes and verify retained source checkpoints before submission."""
    bundle={'experiment_id':cfg['experiment_id'],'config_sha256':sha(config_path),
            'implementation_sha256':implementation(),'students':{},'inputs':{},'teachers':{}}
    for dataset in DATASETS:
        bundle['students'][dataset]={};bundle['inputs'][dataset]={}
        common_uids=None
        for student in STUDENTS:
            teacher_data,student_data=source.load_inputs(dataset,student,auto_prepare=True)
            if common_uids is None:common_uids=student_data.trial_uid
            elif not np.array_equal(common_uids,student_data.trial_uid):
                raise RuntimeError('student input order differs across architectures')
            scfg=project_config.load_model_config(student,dataset,'loso')
            scfg.update(dataset_name=dataset,in_channels=int(student_data.x.shape[1]),
                        samples=int(student_data.x.shape[-1]))
            if int(scfg.get('epochs',0))!=100:
                raise ValueError(f'{dataset}/{student}: fixed final-epoch protocol requires100epochs')
            bundle['students'][dataset][student]=scfg
            bundle['inputs'][dataset][student]={
                'teacher_source':teacher_data.source_manifest_sha256,
                'student_source':student_data.source_manifest_sha256,
                'teacher_input':teacher_data.input_manifest_sha256,
                'student_input':student_data.input_manifest_sha256,
                'teacher_policy':identity(teacher_data.identity),
                'student_policy':identity(student_data.identity),
                'uid_sha256':identity(student_data.trial_uid.tolist()),
                'labels_sha256':identity(student_data.y.tolist()),
                'student_shape':list(student_data.x.shape),
                'teacher_shape':list(teacher_data.x.shape),
                'class_mapping':student_data.class_mapping}
            if student==STUDENTS[0]:
                for seed in SEEDS:
                    for fold in range(SUBJECTS[dataset]):
                        record=source.teacher_record(dataset,seed,fold,teacher_data)
                        bundle['teachers'][f'{dataset}/{seed}/{fold}']={
                            'checkpoint':str(record['checkpoint']),
                            'checkpoint_sha256':record['checkpoint_sha256'],
                            'model_config':record['model_config'],
                            'manifest_path':str(record['manifest_path'])}
        print(f'[audit] {dataset}: all source teachers and three student inputs verified',flush=True)
    bundle['identity']=identity(bundle)
    path=report_root(cfg)/'resolved.json'
    if path.exists():
        old=read(path)
        if old['identity']!=bundle['identity']:
            raise RuntimeError('existing V1 resolved identity differs; create a new experiment version')
    else:report_json(path,bundle)
    return path,bundle


def fold_context(cfg,bundle,dataset,seed,fold,device,smoke):
    teacher_data,_=source.load_inputs(dataset,'ifnet',auto_prepare=True)
    expected_input=bundle['inputs'][dataset]['ifnet']
    if (teacher_data.source_manifest_sha256!=expected_input['teacher_source']
        or teacher_data.input_manifest_sha256!=expected_input['teacher_input']
        or identity(teacher_data.identity)!=expected_input['teacher_policy']):
        raise RuntimeError('teacher input identity changed after dispatch')
    record=source.teacher_record(dataset,seed,fold,teacher_data)
    expected=bundle['teachers'][f'{dataset}/{seed}/{fold}']
    if (record['checkpoint_sha256']!=expected['checkpoint_sha256']
        or identity(record['model_config'])!=identity(expected['model_config'])):
        raise RuntimeError('retained teacher recipe/checkpoint changed since dispatch')
    train_ids=np.flatnonzero(teacher_data.subjects!=fold)
    test_ids=np.flatnonzero(teacher_data.subjects==fold)
    if set(teacher_data.subjects[train_ids]) & set(teacher_data.subjects[test_ids]):
        raise RuntimeError('subject leakage in LOSO split')
    fp=identity({'resolved':bundle['identity'],'dataset':dataset,'seed':seed,'fold':fold,
        'smoke':smoke,'source_teacher':record['checkpoint_sha256'],
        'train_uid':teacher_data.trial_uid[train_ids].tolist(),
        'test_uid':teacher_data.trial_uid[test_ids].tolist()})
    adapter,teacher=source.build_teacher(record,teacher_data,device)
    return roots(cfg,dataset,seed,fold,smoke),teacher_data,record,train_ids,test_ids,fp,adapter,teacher


def stages(r,tag):
    return {'results_dir':r['result']/tag,'weights_dir':r['weights']/tag}


def student_adapter(cfg,data,student,device):
    cpu_adapter=get_adapter(student,device='cpu',**cfg)
    values=cpu_adapter.preprocess(np.asarray(data.x,dtype=np.float32))
    values=values.detach().cpu() if torch.is_tensor(values) else np.asarray(values,dtype=np.float32)
    adapter=get_adapter(student,device=device,**cfg)
    return adapter,values


def train_fold(cfg,bundle,dataset,seed,fold,device,smoke=False,progress_path=None):
    r,tdata,record,tr_idx,te_idx,fp,ta,t0=fold_context(cfg,bundle,dataset,seed,fold,device,smoke)
    completion=r['result']/'complete.json'
    if completion.exists():
        done=read(completion)
        if done.get('fingerprint')!=fp:raise RuntimeError('fold completion fingerprint differs')
        return done
    for path in r.values():path.mkdir(parents=True,exist_ok=True)
    if smoke:tr_idx=tr_idx[:16]
    record_summary={key:record[key] for key in ('dataset','seed','fold','checkpoint_sha256',
        'model_config','manifest_sha256','train_uid_sha256','test_uid_sha256','training_protocol')}
    record_summary.update(checkpoint=str(record['checkpoint']),manifest_path=str(record['manifest_path']))
    train.atomic_json(r['result']/'resolved.json',{'fingerprint':fp,'source_teacher':record_summary,
        'bundle_identity':bundle['identity'],'protocol':cfg['protocol'],
        'source_teacher_sha256':record['checkpoint_sha256'],'smoke':smoke,
        'train_uids':tdata.trial_uid[tr_idx].tolist(),'test_uid_sha256':identity(tdata.trial_uid[te_idx].tolist())})
    x_teacher=np.asarray(tdata.x[tr_idx],dtype=np.float32);y=tdata.y[tr_idx]
    initial={k:v.detach().cpu().clone() for k,v in t0.state_dict().items()}
    teacher_cfg={'epochs':cfg['teacher']['adaptation_epochs'],'batch_size':cfg['teacher']['batch_size'],
        'lr':cfg['teacher']['lr'],'weight_decay':cfg['teacher']['weight_decay'],
        'gradient_clip_norm':cfg['teacher']['gradient_clip_norm'],
        'mcc':{'temperature':cfg['teacher']['mcc_temperature']}}
    if smoke:teacher_cfg['batch_size']=2
    teachers={'T0':t0};teacher_meta={};targets={}
    for name in ('T_CE','T_MCC'):
        progress(progress_path,status='running',dataset=dataset,seed=seed,fold=fold+1,stage=name)
        teacher,manifest=train.adapt_teacher(ta,copy.deepcopy(t0),x_teacher,y,teacher_cfg,
            **stages(r,name),fingerprint=identity([fp,name]),seed=core.derive_seed('teacher',seed,fold,dataset),
            initial_state=initial,mcc_weight=0. if name=='T_CE' else cfg['teacher']['mcc_weight'],
            epochs=1 if smoke else None,schedule_seed=core.derive_seed('teacher_order',seed,fold,dataset),
            max_batches=2 if smoke else None,forward_fn=core.forward_native)
        teachers[name]=teacher;teacher_meta[name]=manifest
    for name,model in teachers.items():
        targets[name]=train.extract(ta,model,x_teacher,batch_size=64,device=device,
            forward_fn=core.forward_native,cache_path=r['cache']/f'{name}_train.npz',
            fingerprint=identity([fp,name,'train_targets']),uids=tdata.trial_uid[tr_idx],labels=y)
        if targets[name]['features'].shape[1]!=256:raise RuntimeError('MIRepNet classifier input is not256D')
    student_meta={};projection_meta={}
    for student in STUDENTS:
        teacher_data,sdata=source.load_inputs(dataset,student,auto_prepare=True)
        if not np.array_equal(tdata.trial_uid,sdata.trial_uid) or not np.array_equal(tdata.y,sdata.y):
            raise RuntimeError('teacher/student trial alignment changed')
        expected_input=bundle['inputs'][dataset][student]
        if (sdata.source_manifest_sha256!=expected_input['student_source']
            or sdata.input_manifest_sha256!=expected_input['student_input']
            or identity(sdata.identity)!=expected_input['student_policy']
            or identity(sdata.y.tolist())!=expected_input['labels_sha256']):
            raise RuntimeError('student input/labels/preprocessing changed after dispatch')
        scfg=dict(bundle['students'][dataset][student]);adapter,values=student_adapter(scfg,sdata,student,device)
        if smoke:scfg['batch_size']=2
        init_seed=core.derive_seed('student_initial',seed,fold,dataset,student)
        train.seed_all(init_seed,device);model=adapter.build(len(sdata.class_mapping))
        initial_student={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        sample=train.extract(adapter,model,values[tr_idx[:2]],device=device,forward_fn=core.forward_native)
        dim=int(sample['features'].shape[1]);pseed=core.derive_seed('projection',seed,fold,dataset,student)
        ppath=r['cache']/student/'P.npz';pmeta_path=r['result']/student/'projection.json'
        if ppath.exists() or pmeta_path.exists():
            if not(ppath.exists() and pmeta_path.exists()):raise RuntimeError('partial projection record')
            pmeta=read(pmeta_path)
            if pmeta['fingerprint']!=fp or pmeta['seed']!=pseed or pmeta['sha256']!=sha(ppath):
                raise RuntimeError('fixed projection identity/hash changed')
            with np.load(ppath,allow_pickle=False) as saved:P=saved['P_float64'].copy()
        else:
            P=core.fixed_teacher_projection(256,dim,pseed)
            npz(ppath,P_float64=P,P_float32=P.astype(np.float32))
            pmeta={'fingerprint':fp,'seed':pseed,'sha256':sha(ppath),**core.projection_properties(P)}
            train.atomic_json(pmeta_path,pmeta)
        projection_meta[student]=pmeta
        projected={};priors={}
        for name in ('T_CE','T_MCC'):
            _,projected[name]=core.project_teacher_features(targets[name]['features'],P,device)
            weight,bias=core.classifier_readout(teachers[name],'mirepnet')
            diagnostic=core.projection_readout_diagnostic(targets[name]['features'],targets[name]['logits'],
                weight.detach().cpu().numpy(),bias.detach().cpu().numpy(),P,y)
            diagnostic.pop('projected_logits')
            train.atomic_json(r['result']/student/f'{name}_projection_diagnostic.json',diagnostic)
            prior,manifest=train.fit_prior(projected[name],cfg['diffusion'],
                **stages(r,f'{student}/prior_{name}'),fingerprint=identity([fp,student,name,'prior']),
                seed=core.derive_seed('prior',seed,fold,dataset,student),device=device,
                updates=4 if smoke else cfg['diffusion']['training_updates'])
            priors[name]=prior
        for method in METHODS:
            progress(progress_path,status='running',dataset=dataset,seed=seed,fold=fold+1,
                stage=f'{student}/{method}')
            target_name='T_MCC' if method.startswith('MCC') else 'T_CE'
            use_diff=method.endswith('DIFFKD')
            manifest=train.train_student(adapter,model,values[tr_idx],y,scfg,
                **stages(r,f'{student}/{method}'),fingerprint=identity([fp,student,method]),
                seed=core.derive_seed('student_dropout',seed,fold,dataset,student),
                initial_state=initial_student,teacher_targets=None if method=='none' else projected[target_name],
                prior=priors[target_name] if use_diff else None,
                feature_weight=0. if method=='none' else cfg['student_training']['feature_weight'],
                epochs=1 if smoke else cfg['student_training']['epochs'],
                schedule_seed=core.derive_seed('student_order',seed,fold,dataset,student),
                noise_seed=core.derive_seed('student_noise',seed,fold,dataset,student),
                max_batches=2 if smoke else None,forward_fn=core.forward_native,diffusion_cfg=cfg['diffusion'])
            student_meta[f'{student}/{method}']=manifest
        del model,adapter,values,priors
        if device.type=='cuda':torch.cuda.empty_cache()
    done={'status':'complete','fingerprint':fp,'dataset':dataset,'seed':seed,'subject':fold+1,
        'smoke':smoke,'target_evaluated':False,'source_teacher_sha256':record['checkpoint_sha256'],
        'source_training_trials_used':int(len(tr_idx)),
        'teacher_adaptation':teacher_meta,'student_conditions':student_meta,
        'projections':projection_meta,'student_count':3,'conditions_per_student':5}
    train.atomic_json(completion,done)
    return done


def metrics(y,logits):
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,cohen_kappa_score
    pred=logits.argmax(axis=1)
    return {'accuracy':float(accuracy_score(y,pred)),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'kappa':float(cohen_kappa_score(y,pred)),
        'cross_entropy':float(torch.nn.functional.cross_entropy(
            torch.as_tensor(logits,dtype=torch.float64),torch.as_tensor(y,dtype=torch.long)))}


def load_stage(model,r,tag,expected_fingerprint):
    metadata=read(r['result']/tag/'complete.json');path=Path(metadata['checkpoint'])
    if metadata.get('fingerprint')!=expected_fingerprint:raise RuntimeError('trained stage fingerprint mismatch')
    if not path.exists():raise FileNotFoundError(f'completed checkpoint missing; refusing retrain: {path}')
    if sha(path)!=metadata['checkpoint_sha256']:raise RuntimeError('trained checkpoint hash mismatch')
    payload=torch.load(resolve_local_file(path),map_location='cpu',weights_only=False)
    model.load_state_dict(payload['model'],strict=True)
    if train.state_hash(model)!=metadata['model_state_sha256']:raise RuntimeError('trained model state mismatch')
    return model


def evaluate_dataset(cfg,bundle,dataset,device,progress_path=None):
    # No target evaluation starts until the full predeclared grid has terminal training records.
    for ds in DATASETS:
        for seed in SEEDS:
            for fold in range(SUBJECTS[ds]):
                if not (roots(cfg,ds,seed,fold)['result']/'complete.json').is_file():
                    raise RuntimeError(f'target evaluation locked: missing training {ds}/{seed}/{fold}')
    records=[]
    for seed in SEEDS:
        for fold in range(SUBJECTS[dataset]):
            r,tdata,record,tr_idx,te_idx,fp,ta,t0=fold_context(cfg,bundle,dataset,seed,fold,device,False)
            complete=read(r['result']/'complete.json')
            if complete['fingerprint']!=fp or complete['smoke']:raise RuntimeError('training identity mismatch')
            eval_path=r['result']/'evaluation.json'
            if eval_path.exists():
                saved=read(eval_path)
                if saved['fingerprint']!=fp:raise RuntimeError('saved evaluation identity mismatch')
                records.extend(saved['records']);continue
            fold_records=[];y=tdata.y[te_idx];uids=tdata.trial_uid[te_idx]
            def predict(student,method,adapter,model,x):
                arrays=train.extract(adapter,model,x,device=device,forward_fn=core.forward_native)
                path=r['result']/'predictions'/student/f'{method}.npz'
                npz(path,trial_uid=uids,y=y,logits=arrays['logits'])
                row={'dataset':dataset,'student':student,'method':method,'seed':seed,'subject':fold+1,
                    'n_trials':len(y),'trial_uid_sha256':identity(uids.tolist()),
                    'prediction_path':str(path),'prediction_sha256':sha(path),**metrics(y,arrays['logits'])}
                fold_records.append(row)
            for name in ('T0','T_CE','T_MCC'):
                model=t0 if name=='T0' else load_stage(copy.deepcopy(t0),r,name,identity([fp,name]))
                predict('MIRepNet',name,ta,model,np.asarray(tdata.x[te_idx],dtype=np.float32))
            for student in STUDENTS:
                _,data=source.load_inputs(dataset,student,auto_prepare=False)
                expected_input=bundle['inputs'][dataset][student]
                if (not np.array_equal(data.trial_uid,tdata.trial_uid)
                    or not np.array_equal(data.y,tdata.y)
                    or data.source_manifest_sha256!=expected_input['student_source']
                    or data.input_manifest_sha256!=expected_input['student_input']
                    or identity(data.identity)!=expected_input['student_policy']
                    or identity(data.trial_uid.tolist())!=expected_input['uid_sha256']):
                    raise RuntimeError('target student input identity/order changed after training')
                scfg=bundle['students'][dataset][student];adapter,values=student_adapter(scfg,data,student,device)
                for method in METHODS:
                    progress(progress_path,status='running',stage='evaluate',dataset=dataset,
                        seed=seed,fold=fold+1,student=student,method=method)
                    model=load_stage(adapter.build(len(data.class_mapping)),r,f'{student}/{method}',
                                     identity([fp,student,method]))
                    predict(student,method,adapter,model,values[te_idx])
                    del model
            train.atomic_json(eval_path,{'status':'complete','fingerprint':fp,'records':fold_records})
            records.extend(fold_records)
    path=report_root(cfg)/f'{dataset}_records.json'
    report_json(path,records)
    return records


def enqueue(cfg,config_path,resolved_path,bundle,mode,datasets=None):
    from experiments.distill.queue import DEFAULT_ROOT,enqueue_snapshot
    jobs=[]
    for dataset in datasets or DATASETS:
        cells=[(666,0)] if mode=='evaluate' else [(s,f) for s in SEEDS for f in range(SUBJECTS[dataset])]
        for seed,fold in cells:
            snapshot={'kind':'mirepnet_t2s_allsets_v1','stage':mode,
                'config':str(config_path),'config_sha256':sha(config_path),
                'resolved':str(resolved_path),'resolved_sha256':sha(resolved_path),
                'implementation_sha256':bundle['implementation_sha256'],
                'dataset':dataset,'seed':seed,'fold':fold,
                'args':{'dataset':dataset,'teacher':'mirepnet','students':list(STUDENTS),
                    'seeds':[seed],'keys':[fold],'methods':list(METHODS),'protocol':'standard_loso'}}
            jobs.append(enqueue_snapshot(snapshot,root=DEFAULT_ROOT,
                label=f'MIRepNet T2S {mode} {dataset} seed={seed} S{fold+1:02d}, 3students'))
    return jobs


def watch(cfg,config_path,resolved_path,bundle,job_ids):
    from experiments.distill.queue import QueueStore,DEFAULT_ROOT
    store=QueueStore(DEFAULT_ROOT);path=report_root(cfg)/'dispatch.json'
    eval_ids=[]
    while True:
        ids=eval_ids or job_ids
        states=[store.load(jid) for jid in ids]
        failed=[s for s in states if s['status'] in ('failed','cancelled')]
        if failed:
            report_json(path,{'status':'failed','training_jobs':job_ids,'evaluation_jobs':eval_ids,
                'failed':[{'id':s['id'],'message':s.get('message')} for s in failed]})
            return 1
        report_json(path,{'status':'evaluating' if eval_ids else 'training',
            'training_jobs':job_ids,'evaluation_jobs':eval_ids,
            'completed_current_stage':sum(s['status']=='completed' for s in states),
            'total_current_stage':len(states),'watcher_pid':os.getpid(),
            'resolved':str(resolved_path),'seeds':list(SEEDS),'datasets':list(DATASETS),
            'students':list(STUDENTS)})
        if all(s['status']=='completed' for s in states):
            if not eval_ids:
                eval_ids=enqueue(cfg,config_path,resolved_path,bundle,'evaluate');continue
            from experiments.distill.mirepnet_t2s_report import report
            records=[]
            for dataset in DATASETS:records.extend(read(report_root(cfg)/f'{dataset}_records.json'))
            result=report(cfg,records,report_root(cfg))
            report_json(path,{'status':'complete','training_jobs':job_ids,
                'evaluation_jobs':eval_ids,'summary':result})
            return 0
        time.sleep(30)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--resolved',type=Path)
    p.add_argument('--mode',choices=('prepare','smoke','train','evaluate','enqueue','watch'),default='enqueue')
    p.add_argument('--dataset',choices=DATASETS);p.add_argument('--seed',type=int,default=666)
    p.add_argument('--fold',type=int,default=0);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--progress',type=Path);p.add_argument('--stop-file',type=Path)
    p.add_argument('--expected-config-sha256');p.add_argument('--expected-resolved-sha256')
    p.add_argument('--expected-implementation-sha256');p.add_argument('--job-ids',nargs='*',default=[])
    args=p.parse_args(argv)
    try:
        config_path=args.config.resolve();cfg=load_config(config_path)
        if args.mode in ('prepare','enqueue'):
            resolved_path,bundle=freeze(config_path,cfg)
            if args.mode=='prepare':return 0
        else:
            resolved_path=args.resolved or report_root(cfg)/'resolved.json';bundle=read(resolved_path)
        for expected,actual,label in (
            (args.expected_config_sha256,sha(config_path),'config'),
            (args.expected_resolved_sha256,sha(resolved_path),'resolved'),
            (args.expected_implementation_sha256,implementation(),'implementation')):
            if expected and expected!=actual:raise RuntimeError(f'{label} changed after submission')
        if bundle['config_sha256']!=sha(config_path) or bundle['implementation_sha256']!=implementation():
            raise RuntimeError('frozen run source/config identity changed')
        if args.mode=='enqueue':
            # A short source-only CPU smoke covers all datasets and all native head shapes.
            for dataset in DATASETS:
                print(f'[smoke] {dataset}',flush=True)
                train_fold(cfg,bundle,dataset,666,0,torch.device('cpu'),True)
            jobs=enqueue(cfg,config_path,resolved_path,bundle,'train')
            log=report_root(cfg)/'watcher.log'
            command=[sys.executable,'-B','-m','experiments.distill.run_mirepnet_t2s_loso',
                '--config',str(config_path),'--resolved',str(resolved_path),'--mode','watch','--job-ids',*jobs]
            with log.open('a') as stream:
                proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,
                    stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            report_json(report_root(cfg)/'dispatch.json',{'status':'submitted','training_jobs':jobs,
                'watcher_pid':proc.pid,'resolved':str(resolved_path)})
            print(json.dumps({'training_jobs':len(jobs),'watcher_pid':proc.pid}),flush=True)
            return 0
        if args.mode=='watch':return watch(cfg,config_path,resolved_path,bundle,args.job_ids)
        if args.device=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable for queued worker')
            torch.cuda.set_device(args.gpu)
        device=torch.device('cpu' if args.device=='cpu' else f'cuda:{args.gpu}')
        if args.stop_file and args.stop_file.exists():
            progress(args.progress,status='paused',dataset=args.dataset,seed=args.seed,fold=args.fold+1)
            return 75
        if args.mode=='evaluate':evaluate_dataset(cfg,bundle,args.dataset,device,args.progress)
        else:train_fold(cfg,bundle,args.dataset,args.seed,args.fold,device,args.mode=='smoke',args.progress)
        progress(args.progress,status='completed',dataset=args.dataset,seed=args.seed,fold=args.fold+1)
        return 0
    except Exception:
        traceback.print_exc();return 1


if __name__=='__main__':raise SystemExit(main())
