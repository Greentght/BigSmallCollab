"""Paired HFD on each subject's 30/70 calibration/test split."""

from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['BIGSMALLCOLLAB_CPU_THREADS'] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import fcntl
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

from collab.mutual import _kd
from data.loso_source import load_dataset, sha256_file
from experiments.bidir.run_hfd_loso import (atomic_json, atomic_save, build, make_adapters,
                                             predict, state_copy, state_digest,
                                             implementation_hashes as hfd_implementation_hashes)
from experiments.distill.mirepnet_t2s_data import ensure_uid_alignment
from experiments.distill.run_fewshot_structure_kd import _split_subject
from experiments.storage import require_external_output

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/experiments/fewshot_hfd_004.yaml'


def implementation_hashes(pair):
    hashes = hfd_implementation_hashes(pair)
    for path in (Path(__file__), ROOT / 'data/split.py',
                 ROOT / 'experiments/distill/run_fewshot_structure_kd.py'):
        hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    return hashes


def uid_hash(values):
    return hashlib.sha256(json.dumps(list(values), ensure_ascii=False).encode()).hexdigest()


def load_views(cfg, pair, subject, seed, smoke):
    dataset = cfg['dataset']
    big_name, small_name = [cfg['pairs'][pair][role]['model'] for role in ('big', 'small')]
    source = load_dataset(dataset)
    big, small = [load_dataset(dataset, name) for name in (big_name, small_name)]
    if big.input_manifest_path is None or small.input_manifest_path is None:
        raise ValueError('Both model views require verified model-input caches')
    big, small = ensure_uid_alignment(big, small)
    source, small = ensure_uid_alignment(source, small)
    if (source.class_mapping != {'left_hand': 0, 'right_hand': 1} or
            set(np.unique(source.subjects)) != set(cfg['subjects'])):
        raise ValueError('Unexpected subjects or class mapping')
    if any(view.input_manifest.get('selected_session') != cfg['selected_session']
           for view in (big, small)):
        raise ValueError('Model-input cache session mismatch')
    split = _split_subject(source, dataset, subject, seed,
                           cfg['calibration_fraction'], cfg['test_fraction'])
    calibration_uid = split['trial_uid_cal'].tolist()
    test_uid = split['trial_uid_test'].tolist()
    if (len(set(calibration_uid)) != len(calibration_uid) or
            len(set(test_uid)) != len(test_uid) or set(calibration_uid) & set(test_uid)):
        raise ValueError('Calibration and test trials must be unique and disjoint')
    lookup = {uid: i for i, uid in enumerate(np.asarray(small.trial_uid, dtype=str))}
    cal_idx = np.asarray([lookup[uid] for uid in calibration_uid], dtype=np.int64)
    test_idx = np.asarray([lookup[uid] for uid in test_uid], dtype=np.int64)
    for indices, labels in ((cal_idx, split['y_cal']), (test_idx, split['y_test'])):
        if (not np.all(np.asarray(small.subjects)[indices] == subject) or
                not np.array_equal(np.asarray(small.y)[indices], labels)):
            raise ValueError('Subject or label mismatch after UID alignment')
    eval_idx = cal_idx if smoke else test_idx
    adapters = make_adapters(cfg, pair, 'cpu')
    train = [adapter.preprocess(np.array(view.x[cal_idx], copy=True))
             for adapter, view in zip(adapters, (big, small))]
    evaluation = [adapter.preprocess(np.array(view.x[eval_idx], copy=True))
                  for adapter, view in zip(adapters, (big, small))]
    shapes = {'mirepnet_ifnet': ((45, 1000), (6, 1000)),
              'cbramod_eegnet': ((3, 4, 200), (3, 1000))}
    if any(tuple(x.shape[1:]) != shape for x, shape in zip(train, shapes[pair])):
        raise ValueError(f'Unexpected HFD input sizes: {[tuple(x.shape) for x in train]}')
    return {
        'train': train, 'evaluation': evaluation,
        'y_train': torch.as_tensor(split['y_cal'], dtype=torch.long),
        'y_evaluation': np.asarray(split['y_cal'] if smoke else split['y_test'], dtype=np.int64),
        'evaluation_uid': np.asarray(calibration_uid if smoke else test_uid, dtype=str),
        'source_manifest_sha256': source.source_manifest_sha256,
        'input_manifest_sha256': [big.input_manifest_sha256, small.input_manifest_sha256],
        'input_preprocessing': [big.input_manifest['preprocessing'], small.input_manifest['preprocessing']],
        'calibration_uid_sha256': uid_hash(calibration_uid),
        'test_uid_sha256': uid_hash(test_uid),
        'calibration_count': len(calibration_uid), 'test_count': len(test_uid),
        'pretrained_sha256': sha256_file(cfg['pairs'][pair]['big']['pretrained']),
        'mirepnet_ea_policy': ('full_subject_unlabeled_transductive'
                               if big_name == 'mirepnet' else None),
    }


def run(cfg, args, out):
    data = load_views(cfg, args.pair, args.subject, args.seed, args.smoke)
    identity = {key: data[key] for key in (
        'source_manifest_sha256', 'input_manifest_sha256', 'input_preprocessing',
        'calibration_uid_sha256', 'test_uid_sha256', 'calibration_count', 'test_count',
        'pretrained_sha256', 'mirepnet_ea_policy')}
    identity.update(experiment_id=cfg['experiment_id'], dataset=cfg['dataset'],
                    pair=args.pair, condition=args.condition, subject=args.subject,
                    seed=args.seed, protocol='fewshot_stratified_random_30_70',
                    checkpoint_selection='final_epoch', smoke=args.smoke,
                    config_sha256=sha256_file(args.config),
                    implementation_sha256=implementation_hashes(args.pair))
    adapters, models, hfd, opts, schedulers = build(
        cfg, args.pair, args.seed, args.device, args.condition.endswith('hfd'))
    identity['initial_state_sha256'] = [state_digest(state_copy(model)) for model in models]
    state_path, result_path = out / 'training_state.pt', out / 'evaluation.json'
    if result_path.exists():
        record = json.loads(result_path.read_text())
        prediction, training = out / 'predictions.npz', out / 'training.json'
        if (record['identity'] != identity or not prediction.is_file() or not training.is_file() or
                record['predictions_sha256'] != sha256_file(prediction) or
                record.get('evaluation_split') != ('calibration' if args.smoke else 'test') or
                json.loads(training.read_text()).get('identity') != identity):
            raise RuntimeError('Completed few-shot result identity mismatch')
        state_path.unlink(missing_ok=True)
        if hfd:
            hfd.close()
        return
    generator = torch.Generator().manual_seed(args.seed + 100003)
    loader = DataLoader(TensorDataset(*data['train'], data['y_train']),
                        batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=0, generator=generator)
    epochs = 2 if args.smoke else int(cfg['epochs'])
    history, start = [], 0
    if state_path.exists():
        saved = torch.load(state_path, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise RuntimeError('Few-shot resume identity mismatch')
        for model, state in zip(models, saved['models']):
            model.load_state_dict(state)
        if hfd:
            hfd.load_state_dict(saved['hfd'])
        for opt, state in zip(opts, saved['optimizers']):
            opt.load_state_dict(state)
        for scheduler, state in zip(schedulers, saved['schedulers']):
            scheduler.load_state_dict(state)
        generator.set_state(saved['loader_rng'])
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if args.device.type == 'cuda':
            torch.cuda.set_rng_state(saved['cuda_rng'], args.device)
        start, history = saved['epoch'], saved['history']
    try:
        for epoch in range(start, epochs):
            for model in models:
                model.train()
            ramp = 1.0 if args.smoke else min(
                1.0, max(0.0, (epoch + 1 - cfg['warmup_epochs']) / cfg['ramp_epochs']))
            totals, batches = np.zeros(4, dtype=np.float64), 0
            for x_big, x_small, target in loader:
                for opt in opts:
                    opt.zero_grad(set_to_none=True)
                if hfd:
                    hfd.clear()
                z_big = adapters[0].forward(models[0], x_big.to(args.device))[1]
                z_small = adapters[1].forward(models[1], x_small.to(args.device))[1]
                target = target.to(args.device)
                pair_cfg = cfg['pairs'][args.pair]
                ce_big = F.cross_entropy(z_big, target, label_smoothing=float(pair_cfg['big']['label_smoothing']))
                ce_small = F.cross_entropy(z_small, target, label_smoothing=float(pair_cfg['small']['label_smoothing']))
                loss = ce_big + ce_small
                dml_value = torch.zeros((), device=args.device)
                feature_value = torch.zeros((), device=args.device)
                if 'dml_kl' in args.condition and ramp:
                    temperature = float(cfg['temperature'])
                    dml_value = temperature**2 * (
                        _kd(z_big, z_small, temperature).mean() +
                        _kd(z_small, z_big, temperature).mean())
                    loss = loss + ramp * float(cfg['dml_weight']) * dml_value
                if hfd and ramp:
                    losses = hfd.losses()
                    if args.smoke and epoch == 0 and batches == 0:
                        captured = (hfd.features['big'], hfd.features['small'])
                        for i, projector in enumerate((hfd.big_to_small, hfd.small_to_big)):
                            grads = torch.autograd.grad(
                                losses[i], (captured[i], captured[1 - i], projector.weight),
                                allow_unused=True, retain_graph=True)
                            if (grads[0] is None or grads[1] is not None or grads[2] is None or
                                    not torch.isfinite(grads[0]).all() or not torch.isfinite(grads[2]).all() or
                                    not grads[0].abs().sum() > 0 or not grads[2].abs().sum() > 0):
                                raise RuntimeError('Few-shot HFD directional gradient isolation failed')
                        print('Few-shot HFD directional gradient isolation passed', flush=True)
                    feature_value = sum(losses)
                    loss = loss + ramp * float(cfg['hfd_weight']) * feature_value
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite few-shot HFD loss')
                loss.backward()
                for opt in opts:
                    opt.step()
                totals += [ce_big.item(), ce_small.item(), dml_value.item(), feature_value.item()]
                batches += 1
                if hfd:
                    hfd.clear()
                if args.smoke and batches >= 2:
                    break
            for scheduler in schedulers:
                scheduler.step()
            history.append({'epoch': epoch + 1, 'mean_losses': (totals / batches).tolist(), 'ramp': ramp})
            stop_requested = bool(args.stop_file and args.stop_file.exists())
            stop_after = bool(args.stop_after and epoch + 1 >= args.stop_after)
            if (stop_requested or stop_after or epoch + 1 == epochs or
                    (epoch + 1) % int(cfg['checkpoint_interval']) == 0):
                saved = {'identity': identity, 'epoch': epoch + 1,
                         'models': [state_copy(model) for model in models],
                         'hfd': hfd.state_dict() if hfd else None,
                         'optimizers': [opt.state_dict() for opt in opts],
                         'schedulers': [scheduler.state_dict() for scheduler in schedulers],
                         'history': history, 'loader_rng': generator.get_state(),
                         'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                         'torch_rng': torch.get_rng_state(),
                         'cuda_rng': torch.cuda.get_rng_state(args.device) if args.device.type == 'cuda' else None}
                atomic_save(state_path, saved)
            print(f'{args.pair} {args.condition} subject={args.subject} seed={args.seed} '
                  f'epoch={epoch + 1}/{epochs} ce={totals[0]/batches:.4f},{totals[1]/batches:.4f} '
                  f'hfd={totals[3]/batches:.4f}', flush=True)
            if args.progress:
                atomic_json(args.progress, {'status': 'running', 'epoch': epoch + 1,
                                             'epochs': epochs, 'pair': args.pair,
                                             'condition': args.condition,
                                             'subject': args.subject, 'seed': args.seed})
            if stop_requested:
                if args.progress:
                    atomic_json(args.progress, {'status': 'paused', 'epoch': epoch + 1})
                raise SystemExit(75)
            if stop_after:
                return
        if hfd:
            hfd.close()
        final_states = [state_copy(model) for model in models]
        results, logits = {}, []
        eval_indices = np.arange(len(data['evaluation_uid']))
        truth = data['y_evaluation']
        for role, model, adapter, x, saved in zip(
                ('big', 'small'), models, adapters, data['evaluation'], final_states):
            output = predict(adapter, model, x, eval_indices, cfg['batch_size'], args.device)
            predicted = output.argmax(axis=1)
            results[role] = {'accuracy': float(np.mean(predicted == truth) * 100),
                             'balanced_accuracy': float(balanced_accuracy_score(truth, predicted) * 100),
                             'kappa': float(cohen_kappa_score(truth, predicted)),
                             'selected_epoch': epochs, 'state_sha256': state_digest(saved)}
            logits.append(output)
        temporary = out / f'.predictions.{os.getpid()}.tmp.npz'
        try:
            np.savez_compressed(temporary, uid=data['evaluation_uid'], y=truth,
                                big_logits=logits[0], small_logits=logits[1])
            os.replace(temporary, out / 'predictions.npz')
        finally:
            temporary.unlink(missing_ok=True)
        atomic_json(out / 'training.json', {'identity': identity, 'history': history})
        results.update(identity=identity, evaluation_split='calibration' if args.smoke else 'test',
                       predictions_sha256=sha256_file(out / 'predictions.npz'))
        atomic_json(result_path, results)
        state_path.unlink(missing_ok=True)
    finally:
        if hfd:
            hfd.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--pair', required=True)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--subject', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-id', default='default')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--expected-config-sha256')
    parser.add_argument('--expected-implementation-sha256')
    parser.add_argument('--progress', type=Path)
    parser.add_argument('--stop-file', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.config = args.config.resolve()
    cfg = yaml.safe_load(args.config.read_text())
    if args.expected_config_sha256 and sha256_file(args.config) != args.expected_config_sha256:
        raise ValueError('Queued few-shot configuration changed')
    if (args.pair not in cfg['pairs'] or args.condition not in cfg['conditions'] or
            args.subject not in cfg['subjects'] or args.seed not in cfg['seeds'] or
            cfg['checkpoint_selection'] != 'final_epoch' or int(cfg['checkpoint_interval']) < 1):
        raise ValueError('Pair, condition, subject, seed or checkpoint policy is outside the experiment')
    digest = hashlib.sha256(json.dumps(implementation_hashes(args.pair), sort_keys=True).encode()).hexdigest()
    if args.expected_implementation_sha256 and digest != args.expected_implementation_sha256:
        raise ValueError('Queued few-shot implementation changed')
    if (args.stop_after or args.cpu) and not args.smoke:
        parser.error('--stop-after and --cpu are available only for smoke')
    if Path(args.smoke_id).name != args.smoke_id:
        parser.error('invalid smoke id')
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required outside CPU smoke checks')
    args.device = torch.device('cpu' if args.cpu else f'cuda:{args.gpu}')
    if args.device.type == 'cpu':
        torch.backends.mkldnn.enabled = False
    else:
        torch.cuda.set_device(args.device)
    prefix = Path(cfg['output_root']) / ('smoke' if args.smoke else 'runs')
    if args.smoke:
        prefix /= args.smoke_id
    out = require_external_output(prefix / args.pair / args.condition /
                                  f'subject_{args.subject}' / f'seed_{args.seed}')
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(cfg, args, out)
    if args.progress and not args.stop_after:
        atomic_json(args.progress, {'status': 'completed', 'pair': args.pair,
                                    'condition': args.condition,
                                    'subject': args.subject, 'seed': args.seed})


if __name__ == '__main__':
    main()
