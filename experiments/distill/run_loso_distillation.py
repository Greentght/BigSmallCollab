"""Frozen-teacher KD, native feature alignment, and CE-only warmup under LOSO.

Teacher targets are source-training trials only. The completed scratch LOSO
baseline supplies the student config, never the student initial weights. Every
cell saves optimizer, scheduler and RNG state so interrupted training resumes.
"""
import argparse
import csv
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

from models import get_adapter
from experiments.finetune import run_loso_five_datasets as protocol
from experiments.distill import loso_teacher_cache as teachers

PLAN_PATH = protocol.ROOT / 'configs/reproductions/loso_distillation_v1.yaml'
STAGES = ('logits_kd', 'kd_feature', 'warmup10_kd', 'warmup10_kd_feature')
STUDENTS = ('ifnet', 'eegnet', 'adfcnn')
TEACHERS = ('mirepnet', 'cbramod')
FEATURE_ATTR = {'ifnet': 'feat_dim', 'eegnet': 'flatten_size', 'adfcnn': 'feat_size'}
METRICS = ('accuracy', 'balanced_accuracy', 'kappa', 'macro_f1', 'auroc')


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=PLAN_PATH)
    p.add_argument('--teacher', choices=TEACHERS)
    p.add_argument('--stage', choices=STAGES)
    p.add_argument('--datasets', nargs='+', choices=protocol.DATASET_NAMES)
    p.add_argument('--students', nargs='+', choices=STUDENTS)
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--folds', nargs='+', type=int, help='zero-based held-out subject IDs')
    p.add_argument('--gpu', type=int)
    p.add_argument('--smoke-only', action='store_true')
    p.add_argument('--status', action='store_true')
    p.add_argument('--summarize', action='store_true')
    p.add_argument('--assert-stage-complete', choices=STAGES)
    return p.parse_args()


def _plan(path):
    plan = yaml.safe_load(path.read_text())
    if list(plan['stages']) != list(STAGES):
        raise ValueError(f'Expected stage order {STAGES}')
    if plan['loss']['feature_transform'] != 'native_penultimate':
        raise ValueError('This runner aligns the native teacher feature only')
    if plan['loss']['teacher_correct_mask'] or float(plan['loss']['ce_weight']) != 1.0:
        raise ValueError('This protocol uses unmasked KD and unit-weight CE')
    if float(plan['loss']['temperature']) <= 0:
        raise ValueError('temperature must be positive')
    for name, stage in plan['stages'].items():
        if min(float(stage['lam_kd']), float(stage['lam_feat'])) < 0:
            raise ValueError(f'negative loss weight in {name}')
        if not 0 <= int(stage['distill_warmup_epochs']) < 100:
            raise ValueError(f'invalid warmup in {name}')
    plan['_file_sha256'] = protocol._sha256(path)
    return plan


def _root(plan):
    return (protocol.ROOT / plan['output_root']).resolve()


def _json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=True) + '\n')
    os.replace(tmp, path)


def _torch_write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def _state_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _student_code(student):
    paths = [protocol.ROOT / 'models/base.py',
             *sorted((protocol.ROOT / 'models' / student).glob('*.py'))]
    return {str(p.relative_to(protocol.ROOT)): protocol._sha256(p) for p in paths}


def _rng_state(device, generator):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': (torch.cuda.get_rng_state(device)
                           if device.type == 'cuda' else None),
            'loader_generator': generator.get_state()}


def _restore_rng(state, device, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['torch_cuda'], device)
    generator.set_state(state['loader_generator'])


def _projector(student, model, target_dim, seed, device, enabled):
    if not enabled:
        return None
    # Do not run an extra student forward: IFNet eval forward can renorm its
    # classifier, and a train-mode probe can alter BN and dropout RNG.
    dim = int(getattr(model, FEATURE_ATTR[student]))
    devices = [device.index] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(int(seed) + 7919)
        projection = nn.Linear(dim, int(target_dim)).to(device)
    return projection


def _optimizer(model, projection, cfg):
    params = list(model.parameters())
    if projection is not None:
        params.extend(projection.parameters())
    name = str(cfg.get('optimizer', cfg.get('optimizer_type', 'adamw'))).lower()
    kwargs = {'lr': float(cfg['lr']), 'weight_decay': float(cfg['weight_decay'])}
    if name == 'adamw':
        return torch.optim.AdamW(params, **kwargs)
    if name == 'adam':
        return torch.optim.Adam(params, **kwargs)
    if name == 'sgd':
        return torch.optim.SGD(params, momentum=float(cfg.get('momentum', .9)), **kwargs)
    raise ValueError(f'Unsupported optimizer {name}')


def _losses(logits, feat, labels, teacher_logits, teacher_feat, projection,
            stage, temperature, epoch):
    ce = F.cross_entropy(logits, labels)
    kd, feature = ce.new_zeros(()), ce.new_zeros(())
    active = epoch >= int(stage['distill_warmup_epochs'])
    if active and float(stage['lam_kd']) > 0:
        kd = F.kl_div(F.log_softmax(logits / temperature, dim=1),
                      F.softmax(teacher_logits.detach() / temperature, dim=1),
                      reduction='batchmean') * temperature ** 2
    if active and float(stage['lam_feat']) > 0:
        feature = (1 - F.cosine_similarity(
            projection(feat), teacher_feat.detach(), dim=1, eps=1e-8)).mean()
    total = ce + float(stage['lam_kd']) * kd + float(stage['lam_feat']) * feature
    return total, ce, kd, feature, active


def _baseline(dataset, student, fold, seed, uid_train, uid_test, y_test, classes):
    cell = (protocol.RESULTS_ROOT / dataset / student / 'scratch_supervised_loso_v1'
            / f'subject_{fold + 1:02d}' / f'seed_{seed}')
    names = ('model.pt', 'manifest.json', 'result.npz', 'train_history.csv')
    if not all((cell / name).is_file() for name in names):
        raise FileNotFoundError(f'Complete scratch baseline required: {cell}')
    manifest = json.loads((cell / 'manifest.json').read_text())
    if manifest['test_subject'] != fold + 1 or manifest['seed'] != seed:
        raise RuntimeError(f'Baseline fold/seed mismatch: {cell}')
    if manifest.get('label_values') != classes:
        raise RuntimeError(f'Baseline class mapping mismatch: {cell}')
    if (not np.array_equal(manifest['train_sample_uids'], uid_train)
            or not np.array_equal(manifest['test_sample_uids'], uid_test)):
        raise RuntimeError(f'Baseline/source UID mismatch: {cell}')
    cfg = manifest['model_config']
    if int(cfg['epochs']) != 100:
        raise RuntimeError('This fixed-budget experiment requires the 100-epoch baseline')
    with (cell / 'train_history.csv').open() as f:
        history = list(csv.DictReader(f))
    if len(history) != 100 or int(history[-1]['epoch']) != 100:
        raise RuntimeError(f'Baseline history incomplete: {cell}')
    with np.load(cell / 'result.npz', allow_pickle=False) as saved:
        if (not np.array_equal(saved['sample_uid'], uid_test)
                or not np.array_equal(saved['y'], y_test)):
            raise RuntimeError(f'Baseline test artifact mismatch: {cell}')
        metrics = json.loads(str(saved['metrics_json'].item()))
    return cfg, {'path': str(cell),
                 'source_files_sha256': protocol._json_hash(manifest['source_files']),
                 'manifest_sha256': protocol._sha256(cell / 'manifest.json'),
                 'result_sha256': protocol._sha256(cell / 'result.npz')}, metrics


def _completed(cell, fingerprint, uid_test, labels, feature_enabled):
    paths = [cell / name for name in ('result.npz', 'model.pt', 'manifest.json',
                                      'train_history.csv')]
    if feature_enabled:
        paths.append(cell / 'projector.pt')
    if not all(p.is_file() for p in paths):
        return False
    manifest = json.loads((cell / 'manifest.json').read_text())
    if manifest.get('run_fingerprint') != fingerprint:
        raise RuntimeError(f'Completed cell belongs to another configuration: {cell}')
    with (cell / 'train_history.csv').open() as f:
        history = list(csv.DictReader(f))
    if len(history) != 100 or int(history[-1]['epoch']) != 100:
        raise RuntimeError(f'Completed cell has invalid epoch history: {cell}')
    with np.load(cell / 'result.npz', allow_pickle=False) as saved:
        if (not np.array_equal(saved['sample_uid'], uid_test)
                or not np.array_equal(saved['y'], labels)):
            raise RuntimeError(f'Completed cell has different test trials: {cell}')
        if not np.isfinite(saved['logits']).all():
            raise RuntimeError(f'Non-finite saved logits: {cell}')
    return True


def _fit(adapter, model, projection, x_tensor, labels, targets, cfg, stage,
         temperature, seed, cell, fingerprint, initial_hash, checkpoint_every):
    device = adapter.device
    generator = torch.Generator().manual_seed(int(seed))
    tl = torch.from_numpy(targets['logits'])
    # logits-only cells do not carry the large native feature matrix in batches.
    tf = (torch.from_numpy(targets['feats']) if projection is not None
          else torch.zeros((len(labels), 1), dtype=torch.float32))
    loader = DataLoader(TensorDataset(x_tensor, torch.as_tensor(labels, dtype=torch.long),
                                    tl, tf, torch.arange(len(labels))), batch_size=int(cfg['batch_size']),
                        shuffle=True, generator=generator, num_workers=0)
    opt = _optimizer(model, projection, cfg)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg['epochs']))
    history, epoch_start, elapsed_before = [], 0, 0.0
    resume_path = cell / 'resume.pt'
    if resume_path.is_file():
        saved = torch.load(resume_path, map_location='cpu')
        if (saved['run_fingerprint'] != fingerprint
                or saved['initial_state_sha256'] != initial_hash):
            raise RuntimeError(f'Resume state/configuration mismatch: {cell}')
        model.load_state_dict(saved['model'], strict=True)
        if projection is not None:
            projection.load_state_dict(saved['projection'], strict=True)
        opt.load_state_dict(saved['optimizer'])
        schedule.load_state_dict(saved['scheduler'])
        history = saved['history']
        epoch_start = int(saved['completed_epoch'])
        elapsed_before = float(saved['elapsed_sec'])
        _restore_rng(saved['rng'], device, generator)
        if len(history) != epoch_start or not 0 <= epoch_start <= int(cfg['epochs']):
            raise RuntimeError(f'Invalid resume epoch: {cell}')
        print(f'[resume] {cell} completed_epoch={epoch_start}', flush=True)
        del saved
    started = time.monotonic()
    for epoch in range(epoch_start, int(cfg['epochs'])):
        model.train()
        if projection is not None:
            projection.train()
        sums = torch.zeros(4, device=device, dtype=torch.float64)
        seen = 0
        batch_order = hashlib.sha256()
        lr = float(opt.param_groups[0]['lr'])
        epoch_start_time = time.monotonic()
        for xb, yb, teacher_lg, teacher_f, batch_ids in loader:
            batch_order.update(batch_ids.numpy().tobytes())
            xb, yb = xb.to(device), yb.to(device)
            teacher_lg = teacher_lg.to(device)
            if projection is not None:
                teacher_f = teacher_f.to(device)
            opt.zero_grad(set_to_none=True)
            feat, logits = adapter.forward(model, xb)
            total, ce, kd, feature, active = _losses(
                logits, feat, yb, teacher_lg, teacher_f, projection,
                stage, temperature, epoch)
            total.backward()
            opt.step()
            sums += torch.stack([total.detach(), ce.detach(), kd.detach(),
                                 feature.detach()]).to(torch.float64) * len(yb)
            seen += len(yb)
        schedule.step()
        means = (sums / max(seen, 1)).cpu().tolist()
        if not np.isfinite(means).all():
            raise RuntimeError(f'Non-finite training loss at epoch {epoch+1}: {cell}')
        entry = {'epoch': epoch + 1, 'train_loss': means[0], 'ce_loss': means[1],
                 'kd_loss': means[2], 'feature_loss': means[3],
                 'distillation_active': active,
                 'effective_lam_kd': float(stage['lam_kd']) if active else 0.0,
                 'effective_lam_feat': float(stage['lam_feat']) if active else 0.0,
                 'n_train': seen, 'lr_used': lr,
                 'lr_next': float(opt.param_groups[0]['lr']),
                 'batch_index_order_sha256': batch_order.hexdigest(),
                 'student_state_sha256': (_state_hash(model) if epoch + 1 in (1, 10, 100) else ''),
                 'epoch_elapsed_sec': round(time.monotonic()-epoch_start_time, 3)}
        history.append(entry)
        elapsed = elapsed_before + time.monotonic() - started
        _json_write(cell / 'progress.json', {
            'status': 'training', 'completed_epoch': epoch + 1,
            'total_epochs': int(cfg['epochs']), 'elapsed_sec': round(elapsed, 2),
            'run_fingerprint': fingerprint, 'updated_unix': time.time(), **entry})
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == int(cfg['epochs']):
            print(f'  epoch={epoch+1:03d}/{cfg["epochs"]} loss={means[0]:.5f} '
                  f'ce={means[1]:.5f} kd={means[2]:.5f} feat={means[3]:.5f} '
                  f'active={active} sec={entry["epoch_elapsed_sec"]:.2f}', flush=True)
        if (epoch == 0 or (epoch + 1) % checkpoint_every == 0
                or epoch + 1 in (10, int(cfg['epochs']))):
            saved = {'run_fingerprint': fingerprint,
                     'initial_state_sha256': initial_hash,
                     'completed_epoch': epoch + 1, 'history': history,
                     'elapsed_sec': elapsed, 'model': model.state_dict(),
                     'projection': projection.state_dict() if projection is not None else None,
                     'optimizer': opt.state_dict(), 'scheduler': schedule.state_dict(),
                     'rng': _rng_state(device, generator)}
            _torch_write(resume_path, saved)
            protocol._write_history(cell / 'train_history.csv', history)
            del saved
    return history, elapsed_before + time.monotonic() - started


def _run(args, plan, spec, snapshot, device):
    stage = plan['stages'][args.stage]
    datasets = args.datasets or plan['datasets']
    students = args.students or plan['students']
    seeds = args.seeds or plan['seeds']
    if len(set(seeds)) != len(seeds):
        raise ValueError('duplicate seeds')
    for dataset in datasets:
        ds = spec['datasets'][dataset]
        folds = list(range(int(ds['subjects']))) if args.folds is None else args.folds
        if any(f < 0 or f >= int(ds['subjects']) for f in folds):
            raise ValueError(f'Invalid held-out subject for {dataset}')
        protocol._verify_source_files(dataset, spec, snapshot)
        x, y, subject_ids, uids, local_uids, *_ = protocol._load_trials(dataset, spec)
        for student in students:
            out = _root(plan) / args.stage / dataset / f'{args.teacher}__{student}'
            for fold in folds:
                tr, te = subject_ids != fold, subject_ids == fold
                x_tensor, shared_cfg = None, None
                for seed in seeds:
                    cfg, baseline_record, baseline_metrics = _baseline(
                        dataset, student, fold, seed, uids[tr], uids[te], y[te], ds['classes'])
                    targets, target_meta = teachers.load_cache(
                        args.teacher, dataset, fold, seed, uids[tr], y[tr])
                    if baseline_record['source_files_sha256'] != target_meta['source_files_sha256']:
                        raise RuntimeError('Teacher and student baseline have different source fingerprints')
                    if (targets['logits'].shape != (int(tr.sum()), len(ds['classes']))
                            or np.any(targets['sample_uid'][:, 0] == fold)):
                        raise RuntimeError('Teacher cache is not this source-training fold')
                    teacher_source = teachers.teacher_cell(args.teacher, dataset, fold, seed, spec)
                    expected_recipe = plan['teachers'][args.teacher].get('recipe_overrides', {}).get(
                        dataset, plan['teachers'][args.teacher]['default_recipe'])
                    if teacher_source.parent.parent.name != expected_recipe:
                        raise RuntimeError('Teacher cache recipe does not match the plan')
                    run_config = {
                        'protocol': plan['protocol_id'], 'source_protocol': protocol.PROTOCOL,
                        'plan_sha256': protocol._sha256(args.config),
                        'runner_sha256': protocol._sha256(Path(__file__)),
                        'student_code_files': _student_code(student),
                        'source_spec_sha256': snapshot['spec_sha256'],
                        'trial_manifest_sha256': protocol._sha256(protocol.TRIALS_PATH),
                        'teacher': args.teacher, 'student': student, 'dataset': dataset,
                        'stage': args.stage, 'stage_parameters': stage,
                        'loss_parameters': plan['loss'], 'seed': seed,
                        'test_subject': fold+1, 'model_config': cfg,
                        'teacher_cache_metadata': target_meta,
                        'baseline': baseline_record,
                        'initialization': 'fresh_random_from_baseline_seed; not_baseline_final_weights',
                        'selection_policy': 'fixed_final_epoch_no_validation',
                    }
                    fingerprint = protocol._json_hash(run_config)
                    cell, rp, cp, mp, hp = protocol._cell_paths(out, fold, seed)
                    cell.mkdir(parents=True, exist_ok=True)
                    with (cell / '.lock').open('a') as lock:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                        if _completed(cell, fingerprint, uids[te], y[te], stage['lam_feat'] > 0):
                            print(f'[skip] {args.stage} {dataset} {args.teacher}->{student} '
                                  f'S{fold+1} seed={seed}', flush=True)
                            del targets
                            continue
                        # Teacher construction/export and reads happen before resetting
                        # the student RNG; its init follows the existing CE runner.
                        protocol._set_seed(seed)
                        adapter = get_adapter(student, device=device, **cfg)
                        model = adapter.build(len(ds['classes']))
                        initial_hash = _state_hash(model)
                        projection = _projector(student, model, targets['feats'].shape[1],
                                                seed, adapter.device, stage['lam_feat'] > 0)
                        if shared_cfg is not None and cfg != shared_cfg:
                            raise RuntimeError('Different student input configs within a fold')
                        if x_tensor is None:
                            x_tensor = adapter.preprocess(x[tr])
                            shared_cfg = cfg
                        print(f'[start] {args.stage} {dataset} {args.teacher}->{student} '
                              f'S{fold+1} seed={seed} train={int(tr.sum())} '
                              f'feat_dim={targets["feats"].shape[1]} device={device}', flush=True)
                        if adapter.device.type == 'cuda':
                            torch.cuda.reset_peak_memory_stats(adapter.device)
                        history, elapsed = _fit(
                            adapter, model, projection, x_tensor, y[tr], targets, cfg,
                            stage, float(plan['loss']['temperature']), seed, cell,
                            fingerprint, initial_hash,
                            int(plan['execution']['checkpoint_every_epochs']))
                        feat, logits = adapter.infer(model, x[te])
                        metrics, probs, pred, cm = protocol._metrics(y[te], logits)
                        metrics.update(dataset=dataset, model=student, teacher=args.teacher,
                                       student=student, recipe=args.stage,
                                       test_subject=fold+1, seed=seed, n_train=int(tr.sum()),
                                       n_test=int(te.sum()), elapsed_sec=round(elapsed, 2),
                                       artifact_origin='distilled_frozen_source_teacher')
                        for metric in METRICS:
                            metrics[f'baseline_{metric}'] = baseline_metrics[metric]
                            metrics[f'delta_{metric}'] = metrics[metric] - baseline_metrics[metric]
                        result = {'y': y[te], 'pred': pred, 'probs': probs, 'logits': logits,
                                  'feats': feat, 'sample_uid': uids[te],
                                  'local_sample_uid': local_uids[te], 'confusion_matrix': cm,
                                  'metrics_json': np.asarray(json.dumps(metrics, sort_keys=True))}
                        manifest = {**run_config, 'run_fingerprint': fingerprint,
                                    'initial_state_sha256': initial_hash,
                                    'environment_snapshot': protocol._environment_snapshot(device),
                                    'teacher_feature_dim': int(targets['feats'].shape[1]),
                                    'student_feature_dim': int(getattr(model, FEATURE_ATTR[student])),
                                    'student_parameters': sum(p.numel() for p in model.parameters()),
                                    'peak_gpu_allocated_bytes': (torch.cuda.max_memory_allocated(adapter.device)
                                        if adapter.device.type == 'cuda' else 0),
                                    'peak_gpu_reserved_bytes': (torch.cuda.max_memory_reserved(adapter.device)
                                        if adapter.device.type == 'cuda' else 0),
                                    'projector_parameters': (sum(p.numel() for p in projection.parameters())
                                                             if projection is not None else 0),
                                    'train_sample_uids': uids[tr].tolist(),
                                    'test_sample_uids': uids[te].tolist(),
                                    'label_values': ds['classes'], 'window_label': ds.get(
                                        'window_label', '4s_from_acquired_signal'),
                                    'artifact_origin': metrics['artifact_origin'],
                                    'test_metrics': metrics}
                        if projection is not None:
                            _torch_write(cell / 'projector.pt', projection.state_dict())
                        protocol._save_cell(rp, cp, mp, hp, model, result, manifest, history)
                        _json_write(cell / 'progress.json', {
                            'status': 'complete', 'completed_epoch': 100,
                            'run_fingerprint': fingerprint, 'updated_unix': time.time(),
                            'test_metrics': metrics})
                        (cell / 'resume.pt').unlink(missing_ok=True)
                        print(f'[done] {args.stage} {dataset} {args.teacher}->{student} '
                              f'S{fold+1} seed={seed} acc={metrics["accuracy"]:.4f} '
                              f'delta={metrics["delta_accuracy"]:+.4f} sec={elapsed:.1f}', flush=True)
                        del model, projection, adapter, targets, feat, logits, result
                        gc.collect()
                        if device.startswith('cuda'):
                            torch.cuda.empty_cache()
                del x_tensor
            _summarize(plan, spec)
        del x, y, subject_ids, uids, local_uids
        gc.collect()


def _scan(plan, spec):
    counts, rows, latest = {}, [], None
    current_runner = protocol._sha256(Path(__file__))
    current_spec = protocol._sha256(protocol.SPEC_PATH)
    current_trials = protocol._sha256(protocol.TRIALS_PATH)
    student_codes = {name: _student_code(name) for name in plan['students']}
    for stage in STAGES:
        n = 0
        for ds in plan['datasets']:
            for teacher in plan['teachers']:
                for student in plan['students']:
                    for fold in range(spec['datasets'][ds]['subjects']):
                        for seed in plan['seeds']:
                            cell, rp, cp, mp, hp = protocol._cell_paths(
                                _root(plan) / stage / ds / f'{teacher}__{student}', fold, seed)
                            progress = cell / 'progress.json'
                            if progress.is_file():
                                ts = progress.stat().st_mtime
                                if latest is None or ts > latest['mtime']:
                                    latest = {'path': str(progress), 'mtime': ts,
                                              'details': json.loads(progress.read_text())}
                            needed = [rp, cp, mp, hp]
                            if plan['stages'][stage]['lam_feat'] > 0:
                                needed.append(cell / 'projector.pt')
                            if not all(p.is_file() for p in needed):
                                continue
                            manifest = json.loads(mp.read_text())
                            if (manifest.get('protocol') != plan['protocol_id']
                                    or manifest.get('stage') != stage
                                    or manifest.get('dataset') != ds
                                    or manifest.get('teacher') != teacher
                                    or manifest.get('student') != student
                                    or manifest.get('plan_sha256') != plan['_file_sha256']
                                    or manifest.get('runner_sha256') != current_runner
                                    or manifest.get('source_spec_sha256') != current_spec
                                    or manifest.get('trial_manifest_sha256') != current_trials
                                    or manifest.get('student_code_files') != student_codes[student]
                                    or manifest.get('stage_parameters') != plan['stages'][stage]
                                    or manifest.get('loss_parameters') != plan['loss']
                                    or manifest.get('test_subject') != fold+1
                                    or manifest.get('seed') != seed):
                                raise RuntimeError(f'Inconsistent result identity: {mp}')
                            with hp.open() as stream:
                                hist = list(csv.DictReader(stream))
                            if len(hist) != 100 or int(hist[-1]['epoch']) != 100:
                                raise RuntimeError(f'Incomplete finalized cell history: {hp}')
                            metrics = dict(manifest['test_metrics'])
                            metrics['stage'] = stage
                            rows.append(metrics)
                            n += 1
        expected = (sum(spec['datasets'][d]['subjects'] for d in plan['datasets'])
                    * len(plan['seeds']) * len(plan['teachers']) * len(plan['students']))
        counts[stage] = {'completed': n, 'expected': expected, 'complete': n == expected}
    return counts, rows, latest


def _summarize(plan, spec):
    counts, rows, latest = _scan(plan, spec)
    root = _root(plan)
    root.mkdir(parents=True, exist_ok=True)
    columns = ['stage', 'dataset', 'teacher', 'student', 'test_subject', 'seed',
               'n_train', 'n_test', *METRICS,
               *(f'baseline_{m}' for m in METRICS),
               *(f'delta_{m}' for m in METRICS), 'elapsed_sec']
    path = root / 'all_results.csv'
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in columns})
    os.replace(tmp, path)
    groups = []
    for stage in STAGES:
        for ds in plan['datasets']:
            for teacher in plan['teachers']:
                for student in plan['students']:
                    selected = [r for r in rows if r['stage'] == stage and r['dataset'] == ds
                                and r['teacher'] == teacher and r['student'] == student]
                    n_subjects = spec['datasets'][ds]['subjects']
                    expected = n_subjects * len(plan['seeds'])
                    if not selected:
                        continue
                    group = {'stage': stage, 'dataset': ds, 'teacher': teacher,
                             'student': student, 'completed': len(selected),
                             'expected': expected, 'complete': len(selected) == expected}
                    if group['complete']:
                        for metric in (*METRICS, *(f'delta_{m}' for m in METRICS)):
                            means = [float(np.nanmean([r[metric] for r in selected if r['seed'] == seed]))
                                     for seed in plan['seeds']]
                            group[f'{metric}_mean'] = float(np.mean(means))
                            group[f'{metric}_std_seeds'] = float(np.std(means, ddof=1))
                            group[f'{metric}_per_seed_subject_mean'] = dict(zip(map(str, plan['seeds']), means))
                    groups.append(group)
    _json_write(root / 'summary.json', {'protocol': plan['protocol_id'],
                'stage_counts': counts, 'groups': groups, 'latest_progress': latest})
    return counts, rows, latest


def _smoke(args, plan, spec, device):
    for ds_name in args.datasets or plan['datasets']:
        ds = spec['datasets'][ds_name]
        x, y, subs, uid, *_ = protocol._load_trials(ds_name, spec)
        fold = args.folds[0] if args.folds else 0
        seed = (args.seeds or plan['seeds'])[0]
        mask = subs != fold
        targets, _ = teachers.load_cache(args.teacher, ds_name, fold, seed, uid[mask], y[mask])
        for student in args.students or plan['students']:
            cfg = json.loads((protocol.RESULTS_ROOT / ds_name / student
                             / 'scratch_supervised_loso_v1' / f'subject_{fold+1:02d}'
                             / f'seed_{seed}' / 'manifest.json').read_text())['model_config']
            for stage_name, stage in plan['stages'].items():
                protocol._set_seed(seed)
                adapter = get_adapter(student, device=device, **cfg)
                model = adapter.build(len(ds['classes']))
                projection = _projector(student, model, targets['feats'].shape[1],
                                        seed, adapter.device, stage['lam_feat'] > 0)
                xb = adapter.preprocess(x[mask][:4]).to(device)
                feat, logits = adapter.forward(model, xb)
                teacher_lg = torch.tensor(targets['logits'][:4], device=device, requires_grad=True)
                teacher_f = torch.tensor(targets['feats'][:4], device=device, requires_grad=True)
                labels = torch.tensor(y[mask][:4], device=device)
                total, ce, kd, fl, active = _losses(logits, feat, labels, teacher_lg,
                    teacher_f, projection, stage, float(plan['loss']['temperature']),
                    int(stage['distill_warmup_epochs']))
                if not torch.isfinite(total) or logits.shape != (4, len(ds['classes'])):
                    raise RuntimeError('Invalid distillation loss or class head')
                total.backward()
                if teacher_lg.grad is not None or teacher_f.grad is not None:
                    raise RuntimeError('Teacher target unexpectedly received gradients')
                if not any(p.grad is not None and torch.isfinite(p.grad).all()
                           and p.grad.abs().sum() > 0 for p in model.parameters()):
                    raise RuntimeError('Student received no finite gradient')
                if projection is not None and not any(p.grad is not None and p.grad.abs().sum() > 0
                                                       for p in projection.parameters()):
                    raise RuntimeError('Feature projector received no gradient')
                equal_kd = F.kl_div(F.log_softmax(logits.detach()/2, dim=1),
                    F.softmax(logits.detach()/2, dim=1), reduction='batchmean') * 4
                if abs(float(equal_kd)) > 1e-5:
                    raise RuntimeError('KD must be zero for identical logits')
                if stage['distill_warmup_epochs']:
                    for warmup_epoch in (0, 9):
                        warm, ce_w, kd_w, feat_w, enabled = _losses(
                            logits, feat, labels, teacher_lg, teacher_f, projection,
                            stage, float(plan['loss']['temperature']), warmup_epoch)
                        if enabled or float(kd_w) != 0 or float(feat_w) != 0 or not torch.equal(warm, ce_w):
                            raise RuntimeError('CE-only warmup is not exact')
                optimizer = _optimizer(model, projection, cfg)
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise RuntimeError('Student optimizer update produced non-finite parameters')
                if projection is not None and not all(torch.isfinite(p).all() for p in projection.parameters()):
                    raise RuntimeError('Projection optimizer update produced non-finite parameters')
                print(f'[smoke-ok] {ds_name} {args.teacher}->{student} {stage_name} '
                      f'student_dim={feat.shape[1]} teacher_dim={teacher_f.shape[1]}', flush=True)
                del adapter, model, projection, optimizer, xb, feat, logits, total, ce, kd, fl
                gc.collect()
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
        del x, y, subs, uid, targets


def main():
    args = _args()
    plan = _plan(args.config)
    spec, snapshot = protocol._load_spec()
    if args.status or args.summarize or args.assert_stage_complete:
        counts, _, latest = (_summarize(plan, spec) if args.summarize
                              else _scan(plan, spec))
        print(json.dumps({'stage_counts': counts, 'latest_progress': latest}, indent=2), flush=True)
        if args.assert_stage_complete and not counts[args.assert_stage_complete]['complete']:
            raise RuntimeError(f'Stage barrier not satisfied: {args.assert_stage_complete}')
        return
    if args.teacher is None or (args.stage is None and not args.smoke_only):
        raise ValueError('--teacher and --stage are required for training')
    if args.gpu is not None and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU requested but CUDA is unavailable')
    device = f'cuda:{args.gpu}' if args.gpu is not None else 'cpu'
    torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '4')))
    if args.smoke_only:
        _smoke(args, plan, spec, device)
    else:
        _run(args, plan, spec, snapshot, device)


if __name__ == '__main__':
    main()
