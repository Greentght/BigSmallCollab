"""Run paired AFID-style few-shot experiments on BNCI2014001.

Each process handles one teacher/student/subject/seed/shot cell. Completed
methods have separate atomic result files, allowing failed cells to be resumed.
"""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile

script_dir = str(Path(__file__).resolve().parent)
sys.path = [path for path in sys.path if str(Path(path).resolve()) != script_dir]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import config
from collab.seed import set_seed
from data import loso_source
from experiments.distill import run_fewshot_structure_kd as fewshot
from experiments.storage import RESULTS_ROOT, REPORTS_ROOT, require_external_output, require_report_output
from models import get_adapter

DATASET = 'BNCI2014001'
METHODS = ('none', 'kd', 'dml', 'afid_self_gate', 'afid')


def select_support(raw, shot, seed):
    """Draw nested class-balanced supports from the fixed calibration pool."""
    rng = np.random.RandomState(seed)
    y = raw['y_cal']
    selected = []
    for label in (0, 1):
        candidates = np.flatnonzero(y == label)
        if len(candidates) < shot:
            raise ValueError('insufficient calibration samples for shot count')
        selected.extend(rng.permutation(candidates)[:shot])
    selected = np.asarray(sorted(selected), dtype=np.int64)
    return selected


def prepare_inputs(raw, indices, teacher, student):
    x_support = raw['x_cal_raw'][indices]
    x_test = raw['x_test_raw']
    tc = config.load_model_config(teacher, DATASET, 'fewshot')
    tc.update(dataset_name=DATASET, in_channels=45 if teacher == 'mirepnet' else 22,
              samples=1000)
    ta = get_adapter(teacher, device='cpu', **tc)
    if teacher == 'mirepnet':
        train_t, test_t, ea_hash = fewshot._prepare_mirepnet_input(DATASET, x_support, x_test, ta)
    else:
        train_t = fewshot._prepare_cbramod_input(x_support, tc, ta)
        test_t = fewshot._prepare_cbramod_input(x_test, tc, ta)
        ea_hash = None
    sc = config.load_model_config(student, DATASET, 'fewshot')
    sc.update(dataset_name=DATASET, in_channels=22, samples=1000)
    sa = get_adapter(student, device='cpu', **sc)
    train_s = sa.preprocess(x_support).numpy()
    test_s = sa.preprocess(x_test).numpy()
    return tc, sc, train_t, test_t, train_s, test_s, ea_hash


def teacher_trainable(model, name):
    for param in model.parameters():
        param.requires_grad_(False)
    modules = ([model.transformer[-1], model.clshead] if name == 'mirepnet'
               else [model.backbone.encoder.layers[-1], model.classifier])
    for module in modules:
        for param in module.parameters():
            param.requires_grad_(True)
    return modules


def kl(student, target, temperature=2.0):
    return F.kl_div(F.log_softmax(student / temperature, dim=1),
                    F.softmax(target.detach() / temperature, dim=1),
                    reduction='batchmean') * temperature ** 2


@torch.no_grad()
def predict(adapter, model, values, device):
    model.eval()
    out = []
    for start in range(0, len(values), 16):
        x = torch.as_tensor(values[start:start + 16], dtype=torch.float32, device=device)
        out.append(adapter.forward(model, x)[1].argmax(1).cpu().numpy())
    return np.concatenate(out)


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, encoding='utf-8', delete=False) as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write('\n')
        temporary = f.name
    os.replace(temporary, path)


def atomic_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, encoding='utf-8', newline='', delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = f.name
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.gpu is not None and not torch.cuda.is_available():
        raise RuntimeError('GPU requested but CUDA is unavailable')
    device = torch.device('cpu' if args.gpu is None else f'cuda:{args.gpu}')
    source = loso_source.load_dataset(DATASET, model=None)
    raw = fewshot._split_subject(source, DATASET, args.subject, args.seed, .3, .7)
    indices = select_support(raw, args.shots, args.seed)
    train_uids = raw['trial_uid_cal'][indices].tolist()
    test_uids = raw['trial_uid_test'].tolist()
    if len(set(train_uids)) != len(train_uids) or set(train_uids) & set(test_uids):
        raise RuntimeError('duplicate or overlapping support and test trial UIDs')
    tc, sc, xt, xte, xs, xse, ea_hash = prepare_inputs(raw, indices, args.teacher, args.student)
    y = np.asarray(raw['y_cal'][indices], dtype=np.int64)
    yt = np.asarray(raw['y_test'], dtype=np.int64)
    project = Path(__file__).resolve().parents[2]
    code_files = [
        'experiments/distill/run_afid_fewshot.py', 'collab/afid_fewshot.py',
        'experiments/distill/run_fewshot_structure_kd.py', 'data/loso_source.py',
        'data/split.py', 'data/preproc.py', 'data/channels.py', 'collab/seed.py',
        'models/base.py', 'models/__init__.py', 'config.py',
        f'models/{args.teacher}/adapter.py', f'models/{args.student}/adapter.py',
        'models/mirepnet/mlm.py', 'models/cbramod/cbramod.py',
        'models/cbramod/criss_cross_transformer.py', 'models/ifnet/ifnet.py',
        'models/eegnet/residual_eegnet.py', 'models/adfcnn/adfcnn.py',
        f'configs/models/{args.teacher}.yaml', f'configs/models/{args.student}.yaml',
        'configs/datasets/BNCI2014001.yaml',
    ]
    identity = dict(dataset=DATASET, subject=args.subject, seed=args.seed, shot=args.shots,
                    teacher=args.teacher, student=args.student, epochs=args.epochs,
                    teacher_epochs=args.teacher_epochs or tc['epochs'],
                    source_manifest_sha256=source.source_manifest_sha256,
                    pretrained_sha256=file_hash(config.weight_path(args.teacher)),
                    code_sha256={name: file_hash(project / name) for name in code_files},
                    support_uid_sha256=digest(train_uids), test_uid_sha256=digest(test_uids),
                    ea_sha256=ea_hash, temperature=2.0, lambda_kd=.5,
                    batch_size=min(16, len(y)), teacher_config=copy.deepcopy(tc),
                    student_config=copy.deepcopy(sc))
    cell_id = digest(identity)
    root = require_external_output(RESULTS_ROOT / 'fewshot/afid' / args.run_id /
                                   args.teacher / args.student / f'shot{args.shots}' /
                                   f'subject{args.subject + 1:02d}' / f'seed{args.seed}')
    report_root = REPORTS_ROOT / 'fewshot/afid' / args.run_id / args.teacher / args.student / f'shot{args.shots}' / f'subject{args.subject + 1:02d}' / f'seed{args.seed}'
    pending = []
    for method in args.methods:
        record = root / method / 'completed.json'
        if record.exists():
            with record.open(encoding='utf-8') as f:
                saved = json.load(f)
            if saved['identity_sha256'] != cell_id:
                raise ValueError(f'existing result has different identity: {record}')
            report = require_report_output(report_root / f'{method}_predictions.csv')
            if not report.exists():
                raise FileNotFoundError(f'completed result is missing predictions: {report}')
            print(f'[skip] {record}', flush=True)
        else:
            pending.append(method)
    if not pending:
        return

    tc['skip_preprocess'] = True
    if args.teacher_epochs:
        tc['epochs'] = args.teacher_epochs
    ta = get_adapter(args.teacher, device=str(device), **tc)
    sa = get_adapter(args.student, device=str(device), **sc)
    set_seed(args.seed)
    teacher = ta.build(2)
    teacher = ta.finetune(teacher, xt, y, 2)
    initial_teacher = copy.deepcopy(teacher.state_dict())
    teacher_preds = predict(ta, teacher, xte, device)
    set_seed(args.seed)
    student = sa.build(2)
    initial_student = copy.deepcopy(student.state_dict())
    del teacher, student

    for method in pending:
        set_seed(args.seed)
        student = sa.build(2)
        student.load_state_dict(initial_student)
        teacher = ta.build(2)
        teacher.load_state_dict(initial_teacher)
        train_modules = teacher_trainable(teacher, args.teacher)
        online = method in ('dml', 'afid_self_gate', 'afid')
        fusion = None
        if method in ('afid_self_gate', 'afid'):
            from collab.afid_fewshot import AFIDFeatureFusion
            fusion = AFIDFeatureFusion(teacher, args.teacher, student, args.student,
                                       mode='self' if method == 'afid_self_gate' else 'cross',
                                       bins=4, dim=32).to(device)
        groups = [{'params': list(student.parameters()), 'lr': .001,
                   'weight_decay': float(sc['weight_decay'])}]
        if online:
            groups.append({'params': [p for p in teacher.parameters() if p.requires_grad],
                           'lr': .0001, 'weight_decay': float(tc['weight_decay'])})
        if fusion is not None:
            excluded = {id(p) for p in teacher.parameters()} | {id(p) for p in student.parameters()}
            auxiliary = [p for p in fusion.parameters() if id(p) not in excluded]
            groups.append({'params': auxiliary, 'lr': .001, 'weight_decay': .0001})
        optimizer = torch.optim.AdamW(groups)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        generator = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(TensorDataset(torch.as_tensor(xt), torch.as_tensor(xs),
                                         torch.as_tensor(y)), batch_size=min(16, len(y)),
                            shuffle=True, generator=generator, num_workers=0)
        set_seed(args.seed)
        gate_gradient_checks = []
        for epoch in range(args.epochs):
            student.train()
            teacher.eval()
            if online:
                for module in train_modules:
                    module.train()
            if fusion is not None:
                fusion.train()
                teacher.eval()
                for module in train_modules:
                    module.train()
            for bt, bs, by in loader:
                bt, bs, by = bt.to(device), bs.to(device), by.to(device)
                optimizer.zero_grad()
                if fusion is not None:
                    lt, ls, lf = fusion(bt, bs)
                else:
                    lt = ta.forward(teacher, bt)[1]
                    ls = sa.forward(student, bs)[1]
                loss = F.cross_entropy(ls, by)
                if online:
                    loss = loss + F.cross_entropy(lt, by)
                ramp = min(1.0, max(0.0, (epoch - 4) / 5.0))
                if method == 'kd':
                    loss = loss + .5 * ramp * kl(ls, lt)
                elif method == 'dml':
                    loss = loss + .5 * ramp * (kl(ls, lt) + kl(lt, ls))
                elif fusion is not None:
                    ensemble = (lt + ls) / 2
                    loss = loss + ramp * (F.cross_entropy(lf, by) + .5 *
                                          (kl(lf, ensemble) + kl(lt, lf) + kl(ls, lf)))
                loss.backward()
                if fusion is not None and ramp > 0 and not gate_gradient_checks:
                    for gate in (*fusion.teacher_gates, *fusion.student_gates):
                        grad = gate.weight.grad
                        gate_gradient_checks.append(bool(grad is not None and
                                                         torch.isfinite(grad).all() and
                                                         torch.count_nonzero(grad)))
                    if not all(gate_gradient_checks):
                        raise RuntimeError(f'AFID gate gradient check failed: {gate_gradient_checks}')
                optimizer.step()
            schedule.step()
        if fusion is not None:
            if args.smoke and not gate_gradient_checks:
                raise RuntimeError('smoke requires at least six epochs to verify gate gradients')
            fusion.close()
        predictions = predict(sa, student, xse, device)
        accuracy = float(100 * np.mean(predictions == yt))
        teacher_accuracy = float(100 * np.mean(teacher_preds == yt))
        rows = [dict(trial_uid=uid, y_true=int(label), y_pred=int(pred),
                     teacher_y_pred=int(tpred))
                for uid, label, pred, tpred in zip(test_uids, yt, predictions, teacher_preds)]
        report = require_report_output(report_root / f'{method}_predictions.csv')
        atomic_csv(report, rows)
        result = dict(identity=identity, identity_sha256=cell_id, method=method,
                      accuracy_pct=accuracy, teacher_accuracy_pct=teacher_accuracy,
                      n_test=len(yt), predictions_path=str(report),
                      support_trial_uids=train_uids, test_trial_uids=test_uids,
                      gate_gradient_checks=gate_gradient_checks,
                      student_inference='standalone_original_classifier')
        atomic_json(root / method / 'completed.json', result)
        print(json.dumps(dict(method=method, accuracy_pct=accuracy,
                              teacher_accuracy_pct=teacher_accuracy)), flush=True)
        del student, teacher, fusion, optimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', choices=('mirepnet', 'cbramod'), required=True)
    parser.add_argument('--student', choices=('ifnet', 'eegnet', 'adfcnn'), required=True)
    parser.add_argument('--subject', type=int, choices=range(9), required=True,
                        help='zero-based subject ID')
    parser.add_argument('--seed', type=int, choices=(666, 667, 668), required=True)
    parser.add_argument('--shots', type=int, choices=(5, 10, 20), required=True)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--teacher-epochs', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if args.epochs < 1 or (args.teacher_epochs is not None and args.teacher_epochs < 1):
        parser.error('epoch counts must be positive')
    if not args.run_id or Path(args.run_id).name != args.run_id:
        parser.error('run-id must be a single path component')
    run(args)


if __name__ == '__main__':
    main()
