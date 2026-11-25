"""Paired, full-trial shallow HFD LOSO experiment on BNCI2014004."""

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
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
import yaml

from collab.hfd import ShallowHFD
from collab.mutual import _kd
from collab.seed import set_seed
from data.loso_source import load_dataset, sha256_file
from experiments.distill.mirepnet_t2s_data import ensure_uid_alignment
from experiments.distill.sample_utility_splits import fold_masks
from experiments.storage import require_external_output
from models import get_adapter

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/experiments/loso/hfd_loso_004.yaml'


def atomic_json(path, data):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save(path, data):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        torch.save(data, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def state_copy(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def implementation_hashes(pair):
    big, small = pair.split('_')
    small_file = 'ifnet/ifnet.py' if small == 'ifnet' else 'eegnet/residual_eegnet.py'
    paths = [
        Path(__file__), ROOT / 'collab/hfd.py', ROOT / 'collab/mutual.py',
        ROOT / 'collab/seed.py', ROOT / 'data/loso_source.py',
        ROOT / 'data/preproc.py', ROOT / 'data/channels.py', ROOT / 'config.py',
        ROOT / 'models/base.py', ROOT / 'models/__init__.py',
        ROOT / 'experiments/distill/mirepnet_t2s_data.py',
        ROOT / 'experiments/distill/sample_utility_splits.py',
        ROOT / 'configs/datasets/BNCI2014004.yaml',
        ROOT / f'models/{big}/adapter.py', ROOT / f'models/{small}/adapter.py',
        ROOT / f'models/{small_file}',
        ROOT / ('models/mirepnet/mlm.py' if big == 'mirepnet'
                else 'models/cbramod/cbramod.py'),
    ]
    if big == 'cbramod':
        paths.append(ROOT / 'models/cbramod/criss_cross_transformer.py')
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def make_adapters(cfg, pair_name, device):
    adapters = []
    pair = cfg['pairs'][pair_name]
    for role in ('big', 'small'):
        settings = dict(pair[role])
        name = settings.pop('model')
        if role == 'big':
            settings['pretrain'] = settings.pop('pretrained')
        settings.update(epochs=int(cfg['epochs']), batch_size=int(cfg['batch_size']))
        adapters.append(get_adapter(name, device=device, dataset_name=cfg['dataset'], **settings))
    return adapters


def load_views(cfg, pair_name):
    names = [cfg['pairs'][pair_name][role]['model'] for role in ('big', 'small')]
    big, small = [load_dataset(cfg['dataset'], name) for name in names]
    if big.input_manifest_path is None or small.input_manifest_path is None:
        raise ValueError('Both views must use verified canonical model-input caches')
    big, small = ensure_uid_alignment(big, small)
    mapping = {'left_hand': 0, 'right_hand': 1}
    if big.class_mapping != mapping or set(np.unique(small.subjects)) != set(cfg['folds']):
        raise ValueError('Unexpected classes or incomplete LOSO subjects')
    for view in (big, small):
        if view.input_manifest.get('selected_session') != cfg['selected_session']:
            raise ValueError('Incorrect cache session')
    # Materialize CPU tensors once. The IFNet adapter applies its filter bank once.
    adapters = make_adapters(cfg, pair_name, 'cpu')
    arrays = [adapter.preprocess(np.array(view.x, copy=True))
              for adapter, view in zip(adapters, (big, small))]
    expected = {'mirepnet_ifnet': ((1400, 45, 1000), (1400, 6, 1000)),
                'cbramod_eegnet': ((1400, 3, 4, 200), (1400, 3, 1000))}
    if tuple(tuple(x.shape) for x in arrays) != expected[pair_name]:
        raise ValueError(f'Unexpected paired input sizes: {[x.shape for x in arrays]}')
    source = {
        'source_manifest_sha256': big.source_manifest_sha256,
        'input_manifest_sha256': [big.input_manifest_sha256, small.input_manifest_sha256],
        'input_preprocessing': [big.input_manifest['preprocessing'], small.input_manifest['preprocessing']],
        'pretrained_sha256': sha256_file(cfg['pairs'][pair_name]['big']['pretrained']),
    }
    return arrays, torch.as_tensor(small.y, dtype=torch.long), np.asarray(small.subjects), np.asarray(small.trial_uid, dtype=str), source


def build(cfg, pair_name, seed, device, with_hfd):
    set_seed(seed)
    adapters = make_adapters(cfg, pair_name, device)
    models = [adapter.build(2) for adapter in adapters]
    hfd = ShallowHFD(pair_name, models[0], models[1], seed).to(device) if with_hfd else None
    optimizers, schedulers = [], []
    for i, role in enumerate(('big', 'small')):
        settings = cfg['pairs'][pair_name][role]
        params = list(models[i].parameters())
        if hfd:
            projector = hfd.big_to_small if i == 0 else hfd.small_to_big
            params.extend(projector.parameters())
        cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[settings['optimizer']]
        opt = cls(params, lr=float(settings['lr']), weight_decay=float(settings['weight_decay']))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=int(cfg['epochs']), eta_min=float(settings['min_lr']))
        optimizers.append(opt)
        schedulers.append(scheduler)
    return adapters, models, hfd, optimizers, schedulers


@torch.no_grad()
def predict(adapter, model, data, indices, batch_size, device):
    # IFNet's max-norm layer mutates its weights even in eval mode.
    saved = state_copy(model)
    model.eval()
    outputs = []
    try:
        for start in range(0, len(indices), batch_size):
            batch = data[indices[start:start + batch_size]].to(device)
            outputs.append(adapter.forward(model, batch)[1].detach().cpu())
    finally:
        model.load_state_dict(saved)
    return torch.cat(outputs).numpy()


def evaluate(cfg, args, out, data, models, adapters, best_states, best_epochs, identity):
    arrays, labels, subjects, uids, _ = data
    _, masks = fold_masks(subjects, args.fold)
    # The preflight checks the whole training/evaluation path without using
    # the final held-out subject for experimental feedback.
    split = 'feedback' if args.smoke else 'test'
    test_idx = np.flatnonzero(masks[split])
    predictions = []
    results = {}
    for role, model, adapter, x, saved, epoch in zip(
            ('big', 'small'), models, adapters, arrays, best_states, best_epochs):
        model.load_state_dict(saved)
        logits = predict(adapter, model, x, test_idx, cfg['batch_size'], args.device)
        predicted = logits.argmax(axis=1)
        truth = labels[test_idx].numpy()
        results[role] = {
            'accuracy': float(np.mean(predicted == truth) * 100),
            'balanced_accuracy': float(balanced_accuracy_score(truth, predicted) * 100),
            'kappa': float(cohen_kappa_score(truth, predicted)),
            'selected_epoch': epoch, 'state_sha256': state_digest(saved),
        }
        predictions.append(logits)
    temporary = out / f'.predictions.{os.getpid()}.tmp.npz'
    try:
        np.savez_compressed(temporary, uid=uids[test_idx], y=labels[test_idx].numpy(),
                            big_logits=predictions[0], small_logits=predictions[1])
        os.replace(temporary, out / 'predictions.npz')
    finally:
        temporary.unlink(missing_ok=True)
    results.update(identity=identity, evaluation_split=split,
                   predictions_sha256=sha256_file(out / 'predictions.npz'))
    atomic_json(out / 'evaluation.json', results)


def run(cfg, args, out):
    data = load_views(cfg, args.pair)
    arrays, labels, subjects, uids, source = data
    val_subject, masks = fold_masks(subjects, args.fold)
    train_idx = np.flatnonzero(masks['train'])
    val_idx = np.flatnonzero(masks['feedback'])
    test_idx = np.flatnonzero(masks['test'])
    if len(set(uids)) != len(uids) or not all(len(indices) for indices in (train_idx, val_idx, test_idx)):
        raise ValueError('Missing subjects or duplicated trial IDs')
    identity = {
        'experiment_id': cfg['experiment_id'], 'dataset': cfg['dataset'],
        'pair': args.pair, 'condition': args.condition, 'fold': args.fold,
        'validation_subject': int(val_subject), 'seed': args.seed,
        'smoke': args.smoke, 'config_sha256': sha256_file(args.config),
        'implementation_sha256': implementation_hashes(args.pair), **source,
    }
    adapters, models, hfd, opts, schedulers = build(
        cfg, args.pair, args.seed, args.device, args.condition.endswith('hfd'))
    identity['initial_state_sha256'] = [state_digest(state_copy(model)) for model in models]
    state_path = out / 'training_state.pt'
    result_path = out / 'evaluation.json'
    if result_path.exists():
        record = json.loads(result_path.read_text())
        prediction = out / 'predictions.npz'
        training = out / 'training.json'
        if (record['identity'] != identity or not prediction.is_file() or
                record['predictions_sha256'] != sha256_file(prediction) or
                record.get('evaluation_split') != ('feedback' if args.smoke else 'test') or
                not training.is_file() or json.loads(training.read_text()).get('identity') != identity):
            raise RuntimeError('Completed result or prediction identity mismatch')
        state_path.unlink(missing_ok=True)
        if hfd:
            hfd.close()
        return
    generator = torch.Generator().manual_seed(args.seed + 100003)
    dataset = TensorDataset(*(x[train_idx] for x in arrays), labels[train_idx])
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=0, generator=generator)
    epochs = 2 if args.smoke else int(cfg['epochs'])
    best_scores = [-1.0, -1.0]
    best_epochs = [0, 0]
    best_states = [None, None]
    history = []
    start = 0
    if state_path.exists():
        saved = torch.load(state_path, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise RuntimeError('Resume identity mismatch')
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
        start, best_scores, best_epochs, best_states, history = (
            saved[k] for k in ('epoch', 'best_scores', 'best_epochs', 'best_states', 'history'))
    pair_cfg = cfg['pairs'][args.pair]
    try:
        for epoch in range(start, epochs):
            for model in models:
                model.train()
            ramp = 1.0 if args.smoke else min(1.0, max(0.0, (epoch + 1 - cfg['warmup_epochs']) / cfg['ramp_epochs']))
            totals = np.zeros(4, dtype=np.float64)
            batches = 0
            for x_big, x_small, target in loader:
                for opt in opts:
                    opt.zero_grad(set_to_none=True)
                if hfd:
                    hfd.clear()
                z_big = adapters[0].forward(models[0], x_big.to(args.device))[1]
                z_small = adapters[1].forward(models[1], x_small.to(args.device))[1]
                target = target.to(args.device)
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
                    feature_losses = hfd.losses()
                    if args.smoke and epoch == 0 and batches == 0:
                        captured = (hfd.features['big'], hfd.features['small'])
                        for i, projector in enumerate((hfd.big_to_small, hfd.small_to_big)):
                            grads = torch.autograd.grad(
                                feature_losses[i], (captured[i], captured[1-i], projector.weight),
                                allow_unused=True, retain_graph=True)
                            if (grads[0] is None or grads[1] is not None or grads[2] is None or
                                    not torch.isfinite(grads[0]).all() or not torch.isfinite(grads[2]).all() or
                                    not grads[0].abs().sum() > 0 or not grads[2].abs().sum() > 0):
                                raise RuntimeError('HFD directional gradient isolation failed')
                        print('HFD directional gradient isolation passed', flush=True)
                    feature_value = sum(feature_losses)
                    loss = loss + ramp * float(cfg['hfd_weight']) * feature_value
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite HFD training loss')
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
            for i, (adapter, model, x) in enumerate(zip(adapters, models, arrays)):
                logits = predict(adapter, model, x, val_idx, cfg['batch_size'], args.device)
                score = float(np.mean(logits.argmax(axis=1) == labels[val_idx].numpy()) * 100)
                if score > best_scores[i]:
                    best_scores[i] = score
                    best_epochs[i] = epoch + 1
                    best_states[i] = state_copy(model)
            if hfd:
                hfd.clear()
            history.append({'epoch': epoch + 1, 'mean_losses': (totals / batches).tolist(),
                            'best_validation_accuracy': list(best_scores), 'ramp': ramp})
            saved = {
                'identity': identity, 'epoch': epoch + 1,
                'models': [state_copy(model) for model in models],
                'hfd': hfd.state_dict() if hfd else None,
                'optimizers': [opt.state_dict() for opt in opts],
                'schedulers': [scheduler.state_dict() for scheduler in schedulers],
                'best_scores': best_scores, 'best_epochs': best_epochs, 'best_states': best_states,
                'history': history, 'loader_rng': generator.get_state(),
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state(args.device) if args.device.type == 'cuda' else None,
            }
            atomic_save(state_path, saved)
            print(f'{args.pair} {args.condition} fold={args.fold} seed={args.seed} epoch={epoch+1}/{epochs} '
                  f'ce={totals[0]/batches:.4f},{totals[1]/batches:.4f} hfd={totals[3]/batches:.4f}', flush=True)
            if args.progress:
                atomic_json(args.progress, {'status': 'running', 'epoch': epoch + 1,
                                             'epochs': epochs, 'pair': args.pair,
                                             'condition': args.condition, 'fold': args.fold, 'seed': args.seed})
            if args.stop_file and args.stop_file.exists():
                if args.progress:
                    atomic_json(args.progress, {'status': 'paused', 'epoch': epoch + 1})
                raise SystemExit(75)
            if args.stop_after and epoch + 1 >= args.stop_after:
                return
        if hfd:
            hfd.close()
        # Publish evaluation last: it is the completed-run marker.
        atomic_json(out / 'training.json', {'identity': identity, 'history': history})
        evaluate(cfg, args, out, data, models, adapters, best_states, best_epochs, identity)
        state_path.unlink(missing_ok=True)
    finally:
        if hfd:
            hfd.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--pair', required=True)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--fold', required=True, type=int)
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
        raise ValueError('Queued configuration changed')
    if (args.pair not in cfg['pairs'] or args.condition not in cfg['conditions'] or
            args.fold not in cfg['folds'] or args.seed not in cfg['seeds']):
        raise ValueError('Pair, condition, fold or seed is outside the experiment')
    implementation = implementation_hashes(args.pair)
    digest = hashlib.sha256(json.dumps(implementation, sort_keys=True).encode()).hexdigest()
    if args.expected_implementation_sha256 and digest != args.expected_implementation_sha256:
        raise ValueError('Queued implementation changed')
    if args.stop_after and not args.smoke:
        parser.error('--stop-after is available only for smoke')
    if args.cpu and not args.smoke:
        parser.error('--cpu is available only for smoke')
    if Path(args.smoke_id).name != args.smoke_id:
        parser.error('invalid smoke id')
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required outside CPU smoke checks')
    args.device = torch.device('cpu' if args.cpu else f'cuda:{args.gpu}')
    if args.device.type == 'cpu':
        # oneDNN backward was nondeterministic for this CBraMod model on this host.
        torch.backends.mkldnn.enabled = False
    if args.device.type == 'cuda':
        torch.cuda.set_device(args.device)
    prefix = Path(cfg['output_root']) / ('smoke' if args.smoke else 'runs')
    if args.smoke:
        prefix = prefix / args.smoke_id
    out = require_external_output(prefix / args.pair / args.condition / f'fold_{args.fold}' / f'seed_{args.seed}')
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(cfg, args, out)
    if args.progress and not args.stop_after:
        atomic_json(args.progress, {'status': 'completed', 'pair': args.pair, 'condition': args.condition,
                                    'fold': args.fold, 'seed': args.seed})


if __name__ == '__main__':
    main()
