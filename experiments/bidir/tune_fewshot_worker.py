"""Validation-only few-shot tuning with independently accumulated model gradients."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','BIGSMALLCOLLAB_CPU_THREADS'):
    os.environ[key]='1'

import argparse
import fcntl
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import config
from collab.dcm import mutual_losses
from collab.dml_ds_heads import build_heads
from data.preproc import bandpass
from experiments.bidir.run_mirepnet_fewshot_8_30 import (
    ModelRNG, STUDENT_OFFSET, atomic_torch, data_for, state_hash, uid_hash)
from experiments.bidir.tune_fewshot_protocol import cfg_load, digest, fingerprint, roots, write_json
from experiments.distill.run_fewshot_structure_kd import (
    _prepare_mirepnet_input, _prepare_cbramod_input, _fit_ea)
from experiments.finetune.run_neurogpt_loso import _adapter_for, _load_spec, DEFAULT_CONFIG
from experiments.storage import require_external_output
from models import get_adapter


def nested_split(split,seed,cfg,phase):
    if phase=='search':
        train,validation=train_test_split(np.arange(len(split['y_cal'])),
            test_size=cfg['inner_validation_fraction'], random_state=seed+cfg['inner_seed_offset'],
            stratify=split['y_cal'])
        train=np.asarray(train);validation=np.asarray(validation)
        if set(train)&set(validation):raise ValueError('Overlapping inner split')
        return (split['x_cal_raw'][train],split['y_cal'][train],split['trial_uid_cal'][train],
                split['x_cal_raw'][validation],split['y_cal'][validation],split['trial_uid_cal'][validation])
    return (split['x_cal_raw'],split['y_cal'],split['trial_uid_cal'],
            split['x_test_raw'],split['y_test'],split['trial_uid_test'])


def create_adapter(name,dataset,channels,device):
    if name=='neurogpt':
        exp,spec,_=_load_spec(DEFAULT_CONFIG,dataset)
        return _adapter_for(dataset,channels,device,exp,spec)[0]
    settings=config.load_model_config(name,dataset,'fewshot')
    settings.update(in_channels=45 if name=='mirepnet' else channels,
                    samples=1000,skip_preprocess=name=='mirepnet')
    return get_adapter(name,device=device,dataset_name=dataset,**settings)


def prepare(name,dataset,adapter,train,evaluation):
    provenance={'model':name}
    if name=='mirepnet':
        train=bandpass(np.asarray(train,dtype=np.float64),250,8,30).astype(np.float32)
        evaluation=bandpass(np.asarray(evaluation,dtype=np.float64),250,8,30).astype(np.float32)
        train,evaluation,ea=_prepare_mirepnet_input(dataset,train,evaluation,adapter)
        provenance.update(policy='8_30Hz_order4_filtfilt_train_only_EA_pad45',ea_sha256=ea)
        return adapter.preprocess(train),adapter.preprocess(evaluation),provenance
    if name=='cbramod':
        settings=adapter.cfg;ea=None
        if settings.get('apply_EA',False):
            pre=adapter._resample_to_target(train[:,:,:1000],adapter._target_fs())
            ea=_fit_ea(adapter._apply_frequency_filters(pre,adapter._target_fs()))
        a=_prepare_cbramod_input(train,settings,adapter,ea)
        b=_prepare_cbramod_input(evaluation,settings,adapter,ea)
        provenance.update(policy='configured_filter_resample_normalize_patchify',
                          ea_sha256=digest(ea.tolist()) if ea is not None else None)
        return torch.as_tensor(a),torch.as_tensor(b),provenance
    provenance['policy']='trial_channel_zscore_two_500_chunks' if name=='neurogpt' else 'configured_student_adapter'
    return adapter.preprocess(train),adapter.preprocess(evaluation),provenance


def objectives(outputs,labels,method,alpha,weight=0.25,temperature=2):
    if method=='dml_ds':return mutual_losses(outputs,labels,'dml_ds',temperature,alpha,weight)
    return [F.cross_entropy(side[0],labels)+(float(alpha)*torch.stack(
        [F.cross_entropy(z,labels) for z in side[1:]]).mean() if method=='ds_only' else 0)
        for side in outputs]


def step_accumulated(optimizer,count):
    if count<1:raise ValueError('Empty optimizer update')
    parameters=optimizer.param_groups[0]['params']
    for parameter in parameters:
        if parameter.grad is not None:parameter.grad.div_(count)
    norm=torch.nn.utils.clip_grad_norm_(parameters,1.0)
    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient norm')
    optimizer.step();optimizer.zero_grad(set_to_none=True)


def run(folder,gpu,stop_after=None):
    cfg=cfg_load();root,_=roots(cfg);spec=json.loads((folder/'task.json').read_text())
    frozen=json.loads((root/'frozen.json').read_text())
    if spec['frozen_hash']!=digest(frozen):raise ValueError('Frozen experiment mismatch')
    if fingerprint()!=frozen['code']:raise ValueError('Source changed after experiment was frozen')
    if spec['phase']=='final':
        manifest=json.loads((root/f"final_manifest_{spec['teacher']}.json").read_text())
        if (spec['selection_hash']!=digest(manifest['selection']) or
                spec not in manifest['aliases'].values()):
            raise ValueError('Final test evaluation requires a locked validation selection')
    for name,weight in frozen['weights'].items():
        stat=Path(weight['path']).stat()
        if stat.st_size!=weight['bytes'] or stat.st_mtime_ns!=weight['mtime_ns']:
            raise ValueError(f'Pretrained file changed: {name}')
    result_path=folder/'evaluation.json';state_path=folder/'training_state.pt'
    if result_path.exists():
        done=json.loads(result_path.read_text())
        if done['task']!=spec or not (folder/'predictions.npz').is_file():raise ValueError('Invalid completed result')
        state_path.unlink(missing_ok=True);return
    # The imported NeuroGPT factory already initializes both pools once.
    # PyTorch aborts if inter-op configuration is set a second time.
    if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1:
        raise RuntimeError('Training requires one intra-op and inter-op CPU thread')
    device=torch.device(f'cuda:{gpu}');torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    seed=spec['seed'];random.seed(seed);np.random.seed(seed)
    source,outer,cal_uids,test_uids=data_for(spec['dataset'],spec['subject'],seed,cfg)
    raw_train,y_train,train_uid,raw_eval,y_eval,eval_uid=nested_split(outer,seed,cfg,spec['phase'])
    if set(train_uid)&set(eval_uid):raise ValueError('Training and evaluation overlap')
    if spec['phase']=='search' and set(eval_uid)&set(test_uids):raise ValueError('Test trials in validation')
    names=([spec['teacher'],spec['student']] if spec['target']=='pair' else
           [spec['teacher']] if spec['target']=='teacher' else [spec['student']])
    channels=raw_train.shape[1];classes=int(config.load_dataset_config(spec['dataset'])['num_classes'])
    adapters=[];models=[];heads=[];model_rng=[];head_rng=[];inputs=[];evaluation=[];policies={};initial={};resolved={}
    for name in names:
        model_seed=seed if name==spec['teacher'] else seed+STUDENT_OFFSET[name]
        torch.manual_seed(model_seed);torch.cuda.manual_seed_all(model_seed)
        adapter=create_adapter(name,spec['dataset'],channels,device);model=adapter.build(classes)
        resolved[name]=json.loads(json.dumps(adapter.cfg,default=str))
        adapters.append(adapter);models.append(model);initial[name]=state_hash(model)
        model_rng.append(ModelRNG(model_seed+40001,device))
        if spec['method']!='none':
            torch.manual_seed(model_seed+50021);torch.cuda.manual_seed_all(model_seed+50021)
            heads.append(build_heads(model,name,classes).to(device));head_rng.append(ModelRNG(model_seed+60013,device))
        train,ev,policy=prepare(name,spec['dataset'],adapter,raw_train,raw_eval)
        if len(train)!=len(y_train) or len(ev)!=len(y_eval):raise ValueError('Input alignment mismatch')
        inputs.append(train);evaluation.append(ev);policies[name]=policy
    identity={'source_manifest_sha256':source.source_manifest_sha256,'calibration_uid_sha256':uid_hash(cal_uids),
              'outer_test_uid_sha256':uid_hash(test_uids),'train_uid_sha256':uid_hash(train_uid.tolist()),
              'evaluation_uid_sha256':uid_hash(eval_uid.tolist()),'initial_hashes':initial,'preprocessing':policies,
              'train_count':len(y_train),'evaluation_count':len(y_eval),'test_data_used_for_selection':False,
              'resolved_model_config':resolved,'cpu_threads':1,'dataloader_workers':0}
    opts=[];settings=[]
    for i,name in enumerate(names):
        training=spec['teacher_training'] if name==spec['teacher'] else spec['student_training']
        settings.append(training);parameters=list(models[i].parameters())+(list(heads[i].parameters()) if heads else [])
        opt={'adam':torch.optim.Adam,'adamw':torch.optim.AdamW}[training['optimizer']]
        opts.append(opt(parameters,lr=float(training['lr']),weight_decay=float(training['weight_decay'])))
    epochs=2 if spec['smoke'] else int(spec['teacher_training']['epochs'])
    generator=torch.Generator().manual_seed(seed+100003);start=0;updates=[0]*len(names);history=[]
    if state_path.exists():
        saved=torch.load(state_path,map_location='cpu',weights_only=False)
        if saved['task']!=spec or saved['identity']!=identity:raise ValueError('Resume identity mismatch')
        for objects,key in ((models,'models'),(heads,'heads'),(opts,'optimizers')):
            for obj,value in zip(objects,saved[key]):obj.load_state_dict(value)
        for objects,key in ((model_rng,'model_rng'),(head_rng,'head_rng')):
            for obj,value in zip(objects,saved[key]):obj.load(value)
        generator.set_state(saved['generator']);start=saved['epoch'];updates=saved['updates'];history=saved['history']
    labels=torch.as_tensor(y_train,dtype=torch.long)
    for epoch in range(start,epochs):
        for model in models:model.train()
        for head in heads:head.train()
        for opt in opts:opt.zero_grad(set_to_none=True)
        counts=[0]*len(models);order=torch.randperm(len(labels),generator=generator);before=updates.copy()
        for pos in range(0,len(order),spec['micro_batch']):
            rows=order[pos:pos+spec['micro_batch']];y=labels[rows].to(device);outputs=[]
            for i,(adapter,model,train) in enumerate(zip(adapters,models,inputs)):
                with model_rng[i].use():_,logit=adapter.forward(model,train[rows].to(device))
                if logit.shape!=(len(y),classes):raise ValueError('Main logits shape mismatch')
                side=[logit]
                if heads:
                    with head_rng[i].use():side.extend(heads[i](len(y)))
                    if len(side)!=3 or any(z.shape!=logit.shape for z in side):raise ValueError('Auxiliary shape mismatch')
                outputs.append(side)
            losses=objectives(outputs,y,spec['method'],spec['alpha'],spec.get('lambda',0.25),spec.get('temperature',2))
            loss=sum(losses)*len(y)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward()
            if epoch==0 and pos==0:
                for obj in models+heads:
                    if not any(p.grad is not None and p.grad.abs().sum()>0 for p in obj.parameters()):
                        raise RuntimeError('Missing backbone/head gradient')
            for i,opt in enumerate(opts):
                counts[i]+=len(y)
                if counts[i]>=settings[i]['batch_size'] or pos+len(rows)==len(order):
                    step_accumulated(opt,counts[i]);counts[i]=0;updates[i]+=1
        expected=[int(np.ceil(len(labels)/s['batch_size'])) for s in settings]
        if [x-y for x,y in zip(updates,before)]!=expected:raise RuntimeError('Effective batch update count mismatch')
        for opt,s in zip(opts,settings):
            for group in opt.param_groups:group['lr']=float(s['lr'])*0.5*(1+np.cos(np.pi*(epoch+1)/epochs))
        history.append({'epoch':epoch+1,'loss':float(loss.item()/len(y)),'optimizer_updates':updates.copy()})
        if ((epoch+1)%cfg['checkpoint_interval_epochs']==0 or epoch+1==epochs or stop_after==epoch+1):
            atomic_torch(state_path,{'task':spec,'identity':identity,'epoch':epoch+1,
                'models':[m.state_dict() for m in models],'heads':[h.state_dict() for h in heads],
                'optimizers':[o.state_dict() for o in opts],'model_rng':[r.save() for r in model_rng],
                'head_rng':[r.save() for r in head_rng],'generator':generator.get_state(),
                'updates':updates,'history':history})
        print(f"{spec['teacher']} {spec['dataset']} {spec['student']} {spec['method']} {spec['phase']} "
              f"subject={spec['subject']} seed={seed} epoch={epoch+1}/{epochs} updates={updates}",flush=True)
        if stop_after and epoch+1>=stop_after:return
    for head in heads:head.close()
    predictions={};correct={}
    for name,adapter,model,ev in zip(names,adapters,models,evaluation):
        model.eval();parts=[]
        with torch.no_grad():
            for pos in range(0,len(ev),spec['micro_batch']):
                parts.append(adapter.forward(model,ev[pos:pos+spec['micro_batch']].to(device))[1].argmax(1).cpu())
        predictions[name]=torch.cat(parts).numpy();correct[name]=int(np.count_nonzero(predictions[name]==y_eval))
    tmp=folder/'predictions.tmp.npz';np.savez_compressed(tmp,uids=np.asarray(eval_uid,dtype=str),y=y_eval,**predictions)
    tmp.replace(folder/'predictions.npz')
    write_json(result_path,{'task':spec,'identity':identity,'evaluation_scope':'validation' if spec['phase']=='search' else 'test',
        'evaluation_count':len(y_eval),'correct':correct,'accuracy_percent':{n:v*100/len(y_eval) for n,v in correct.items()},
        'optimizer_updates':dict(zip(names,updates)),'history':history,
        'final_state_hashes':{name:state_hash(model) for name,model in zip(names,models)}})
    state_path.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--stop-after',type=int);args=p.parse_args();folder=require_external_output(args.folder)
    if args.stop_after and not json.loads((folder/'task.json').read_text())['smoke']:p.error('Smoke only')
    with (folder/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(folder,args.gpu,args.stop_after)


if __name__=='__main__':main()
