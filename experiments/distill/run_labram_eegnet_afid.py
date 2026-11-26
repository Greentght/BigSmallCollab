"""Train LaBraM→EEGNet AFID with verified broadband 30/70 splits."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

import argparse
import copy
import csv
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, TensorDataset

import config
from collab.seed import set_seed
from collab.afid_labram import LaBraMEEGNetAFID
from data import loso_source
from experiments.distill.run_fewshot_structure_kd import _split_subject
from experiments.finetune.labram_alexmi_broadband import load_alexmi_broadband
from experiments.finetune.run_labram_broadband_fewshot import BroadbandLaBraMAdapter, preprocess, build_optimizer, scheduler_for, digest_json, file_hash, write_json
from experiments.storage import PROJECT_ROOT, RESULTS_ROOT, REPORTS_ROOT, require_external_output, require_report_output
from models.eegnet.adapter import EEGNetAdapter

BASELINE = 'labram_benchmark_aligned_20261124_v1'
METHODS = ('none', 'afid_self_gate', 'afid')
DATASETS = {'BNCI2014001': 9, 'BNCI2014001-4': 9, 'BNCI2014004': 9, 'BNCI2015001': 12, 'AlexMI': 8}


def kl(student, target):
    return F.kl_div(F.log_softmax(student / 2., dim=1), F.softmax(target.detach() / 2., dim=1), reduction='batchmean') * 4.


def student_input(x, dataset):
    if dataset == 'AlexMI':
        # Existing framework student convention: 3 s resampled to 250 Hz,
        # followed by a copy of the first second to create a 4 s input.
        x = resample(np.asarray(x)[..., :1536], 750, axis=-1)
        x = np.concatenate((x, x[..., :250]), axis=-1)
    return torch.as_tensor(np.asarray(x, dtype=np.float32))


@torch.no_grad()
def predict(adapter, model, x, device):
    model.eval()
    return np.concatenate([adapter.forward(model, x[i:i+16].to(device))[1].argmax(1).cpu().numpy()
                           for i in range(0, len(x), 16)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--subject', type=int, required=True)
    parser.add_argument('--seed', type=int, choices=(666, 667, 668), required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.subject < DATASETS[args.dataset] or Path(args.run_id).name != args.run_id:
        raise ValueError('invalid subject or run ID')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    device = torch.device(f'cuda:{args.gpu}')
    source = load_alexmi_broadband() if args.dataset == 'AlexMI' else loso_source.load_dataset(args.dataset, model=None)
    raw = _split_subject(source, args.dataset, args.subject, args.seed, .3, .7)
    train_uids = raw['trial_uid_cal'].tolist()
    test_uids = raw['trial_uid_test'].tolist()
    if len(set(train_uids)) != len(train_uids) or len(set(test_uids)) != len(test_uids) or set(train_uids) & set(test_uids):
        raise ValueError('training and test UIDs must be unique and disjoint')
    baseline_record = RESULTS_ROOT / 'labram_broadband_fewshot' / BASELINE / args.dataset / f'subject{args.subject+1:02d}' / f'seed{args.seed}' / 'completed.json'
    ref = json.loads(baseline_record.read_text())
    ref_identity = ref['identity']
    if (ref['train_trial_uids'] != train_uids or ref['test_trial_uids'] != test_uids or
        ref_identity['source_manifest_sha256'] != source.source_manifest_sha256 or
        ref_identity['pretrained_sha256'] != file_hash(config.weight_path('labram'))):
        raise ValueError('baseline teacher identity or trial split does not match')
    tcfg = config.load_model_config('labram', args.dataset, 'fewshot')
    if ref_identity['model_config'] != tcfg:
        raise ValueError('baseline teacher settings do not match current LaBraM config')
    classes = int(config.load_dataset_config(args.dataset)['num_classes'])
    scfg = config.load_model_config('eegnet', args.dataset, 'fewshot')
    patches = 3 if args.dataset == 'AlexMI' else 4
    ta = BroadbandLaBraMAdapter(device=str(device), **dict(tcfg, dataset_name=args.dataset,
                                  in_channels=raw['x_cal_raw'].shape[1], patches=patches,
                                  patch_num=patches, samples=1536 if args.dataset == 'AlexMI' else 1000))
    sa = EEGNetAdapter(device=str(device), **dict(scfg, dataset_name=args.dataset,
                        in_channels=raw['x_cal_raw'].shape[1], samples=1000))
    xt = preprocess(ta, raw['x_cal_raw'], args.dataset)
    xte = preprocess(ta, raw['x_test_raw'], args.dataset)
    xs = student_input(raw['x_cal_raw'], args.dataset)
    xse = student_input(raw['x_test_raw'], args.dataset)
    y = torch.as_tensor(raw['y_cal'], dtype=torch.long)
    yt = np.asarray(raw['y_test'], dtype=np.int64)
    if not (len(xt) == len(xs) == len(y) == len(train_uids)):
        raise ValueError('training input order/length mismatch')
    identity = dict(dataset=args.dataset, subject=args.subject, seed=args.seed,
                    protocol='within_subject_30_70', train_uids_sha256=digest_json(train_uids),
                    test_uids_sha256=digest_json(test_uids), source_manifest_sha256=source.source_manifest_sha256,
                    baseline_record_sha256=file_hash(baseline_record), teacher_config=tcfg, student_config=scfg,
                    teacher_pretrained_sha256=ref_identity['pretrained_sha256'], methods=list(METHODS),
                    student_epochs=6 if args.smoke else 100, batch_size=min(16, len(y)),
                    teacher_online_lr=.0001, student_lr=.001, temperature=2., kd_weight=.5,
                    code_sha256={name: file_hash(PROJECT_ROOT / name) for name in (
                        'experiments/distill/run_labram_eegnet_afid.py', 'collab/afid_labram.py')},
                    alexmi_student_window='3s_250hz_plus_first_second' if args.dataset == 'AlexMI' else None,
                    smoke=args.smoke)
    root = require_external_output(RESULTS_ROOT / 'fewshot/afid_labram_eegnet' / args.run_id / args.dataset / f'subject{args.subject+1:02d}' / f'seed{args.seed}')
    report = REPORTS_ROOT / 'fewshot/afid_labram_eegnet' / args.run_id / args.dataset / f'subject{args.subject+1:02d}' / f'seed{args.seed}'
    result_path = root / 'completed.json'
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved['identity'] != identity or any(file_hash(Path(v['path'])) != v['sha256'] for v in saved['methods'].values()):
            raise ValueError('completed AFID cell identity or predictions changed')
        print('[skip] verified completed cell', flush=True)
        return
    set_seed(args.seed)
    teacher = ta.build(classes)
    optimizer = build_optimizer(teacher, tcfg['lr'], tcfg['weight_decay'], tcfg['layer_decay'])
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(xt, y), batch_size=tcfg['batch_size'], shuffle=True, generator=gen, num_workers=0)
    schedule = scheduler_for(tcfg['epochs'], tcfg['warmup_epochs'], tcfg['lr'], tcfg['min_lr'], len(loader))
    step = 0
    for epoch in range(tcfg['epochs']):
        teacher.train()
        for xb, by in loader:
            xb, by = xb.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(ta.forward(teacher, xb)[1], by, label_smoothing=tcfg['label_smoothing'])
            loss.backward()
            optimizer.step()
            step += 1
        for group in optimizer.param_groups:
            group['lr'] = float(schedule[min(step, len(schedule)-1)])
    teacher_pred = predict(ta, teacher, xte, device)
    ref_report = Path(ref['predictions_path'])
    if file_hash(ref_report) != ref['predictions_sha256']:
        raise ValueError('baseline prediction digest mismatch')
    with ref_report.open(newline='') as f:
        ref_rows = list(csv.DictReader(f))
    if ([r['trial_uid'] for r in ref_rows] != test_uids or
        [int(r['y_true']) for r in ref_rows] != yt.tolist() or
        [int(r['y_pred']) for r in ref_rows] != teacher_pred.tolist()):
        raise ValueError('retrained teacher predictions differ from recorded LaBraM baseline')
    print('[teacher] reconstructed predictions exactly match baseline', flush=True)
    teacher_initial = copy.deepcopy(teacher.state_dict())
    set_seed(args.seed)
    student = sa.build(classes)
    student_initial = copy.deepcopy(student.state_dict())
    results = {}
    for method in METHODS:
        set_seed(args.seed)
        teacher.load_state_dict(teacher_initial)
        for p in teacher.parameters():
            p.requires_grad_(False)
        for module in (teacher.blocks[-1], teacher.head):
            for p in module.parameters():
                p.requires_grad_(True)
        student = sa.build(classes)
        student.load_state_dict(student_initial)
        fusion = None
        if method != 'none':
            fusion = LaBraMEEGNetAFID(teacher, student, ta.input_chans, classes, patches,
                                      mode='cross' if method == 'afid' else 'self').to(device)
        groups = [{'params': list(student.parameters()), 'lr': .001, 'weight_decay': float(scfg['weight_decay'])}]
        if fusion:
            groups.append({'params': [p for p in teacher.parameters() if p.requires_grad], 'lr': .0001, 'weight_decay': float(tcfg['weight_decay'])})
            excluded = {id(p) for p in teacher.parameters()} | {id(p) for p in student.parameters()}
            groups.append({'params': [p for p in fusion.parameters() if id(p) not in excluded], 'lr': .001, 'weight_decay': .0001})
        opt = torch.optim.AdamW(groups)
        epochs = identity['student_epochs']
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        gen = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(TensorDataset(xt, xs, y), batch_size=identity['batch_size'], shuffle=True, generator=gen, num_workers=0)
        set_seed(args.seed)
        checks = []
        for epoch in range(epochs):
            student.train()
            teacher.eval()
            if fusion:
                fusion.train()
                teacher.eval()
                teacher.blocks[-1].train()
                teacher.head.train()
            for bt, bs, by in loader:
                bt, bs, by = bt.to(device), bs.to(device), by.to(device)
                opt.zero_grad(set_to_none=True)
                if fusion:
                    lt, ls, lf = fusion(bt, bs)
                else:
                    ls = sa.forward(student, bs)[1]
                loss = F.cross_entropy(ls, by)
                if fusion:
                    loss = loss + F.cross_entropy(lt, by)
                    ramp = min(1., max(0., (epoch - 4) / 5.))
                    loss = loss + ramp * (F.cross_entropy(lf, by) + .5 * (kl(lf, (lt + ls)/2.) + kl(lt, lf) + kl(ls, lf)))
                loss.backward()
                if fusion and ramp > 0 and not checks:
                    checks = [bool(g.weight.grad is not None and torch.isfinite(g.weight.grad).all() and torch.count_nonzero(g.weight.grad)) for g in (*fusion.teacher_gates, *fusion.student_gates)]
                    if not all(checks):
                        raise RuntimeError(f'gate gradient check failed: {checks}')
                opt.step()
            sched.step()
        if fusion:
            if not checks:
                raise RuntimeError('no cross-gate gradient validation')
            fusion.close()
        pred = predict(sa, student, xse, device)
        report_file = require_report_output(report / f'{method}_predictions.csv')
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with report_file.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['trial_uid', 'y_true', 'y_pred', 'teacher_y_pred'])
            writer.writerows(zip(test_uids, yt.tolist(), pred.tolist(), teacher_pred.tolist()))
        results[method] = {'path': str(report_file), 'sha256': file_hash(report_file),
                           'accuracy_pct': float((pred == yt).mean() * 100),
                           'balanced_accuracy_pct': float(balanced_accuracy_score(yt, pred)*100),
                           'gate_gradient_checks': checks}
        print(json.dumps({'method': method, 'accuracy_pct': results[method]['accuracy_pct']}), flush=True)
        del fusion, student, opt
    write_json(result_path, {'identity': identity, 'methods': results, 'train_trial_uids': train_uids,
                              'test_trial_uids': test_uids, 'teacher_prediction_match': True,
                              'teacher_accuracy_pct': float((teacher_pred == yt).mean()*100)})


if __name__ == '__main__':
    main()
