"""One fixed-GPU, native-logit DML-DS + terminal cross-operator HFD unit."""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['BIGSMALLCOLLAB_CPU_THREADS'] = '1'

import argparse
import fcntl
import hashlib
import json
import random
from pathlib import Path
import time

import numpy as np
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score

import config
from collab.hfd_terminal import BIG_LAYERS, SMALL_LAYERS, TerminalHFD
from collab.dcm import mutual_losses
from data.loso_source import sha256_file
from experiments.bidir.run_dml_ds_cross_hfd_all import (
    ROOT, implementation_hashes, make_models, source_and_split, validate_baseline,
)
from experiments.bidir.run_dml_ds_cross_hfd_fewshot import atomic_json, atomic_save, state_digest, uid_hash
from experiments.storage import require_external_output

CONFIG = ROOT / 'configs/experiments/fewshot_dml_ds_terminal_hfd_v1.yaml'


def fingerprint():
    hashes = implementation_hashes()
    for path in ('collab/hfd_terminal.py', 'collab/hfd_position.py',
                 'experiments/bidir/run_terminal_hfd_fewshot.py'):
        hashes[path] = sha256_file(ROOT / path)
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def run_folder(cfg, dataset, method, subject, seed, smoke_id=None):
    root = Path(cfg['output_root']) / ('smoke' if smoke_id else 'runs')
    if smoke_id:
        root /= smoke_id
    return require_external_output(root / dataset / cfg['pair'] / method /
                                   f'subject_{subject}' / f'seed_{seed}')


def json_values(values):
    return {key: value.detach().cpu().tolist() for key, value in values.items()}


def check_gradients(directions, features, losses):
    for role in range(2):
        params = list(directions[role].parameters())
        tensors = features[role][:1] + features[1-role] + params
        grads = torch.autograd.grad(losses[role], tensors, allow_unused=True, retain_graph=True)
        own, peer, projection = grads[:1], grads[1:3], grads[3:]
        if (any(g is None or not torch.isfinite(g).all() or g.abs().sum() == 0 for g in own)
                or any(g is not None for g in peer)
                or any(g is None or not torch.isfinite(g).all() for g in projection)
                or not any(g.abs().sum() > 0 for g in projection)):
            raise RuntimeError(f'Terminal HFD gradient direction {role} failed')


def native_terminal_check(adapters, models, heads, feature, inputs, device):
    from collab.hfd_cross_all import frozen_call
    for model in models: model.eval()
    rng = torch.cuda.get_rng_state(device).clone()
    with torch.no_grad():
        logits = [a.forward(m, x[:2].to(device))[1] for a, m, x in zip(adapters, models, inputs)]
        maps = feature.take()
        for role, direction in enumerate(feature.directions):
            peer = maps[1-role]
            reconstructed = frozen_call(direction._operator, peer[0])
            if not torch.allclose(reconstructed, peer[1], rtol=1e-6, atol=1e-6):
                raise RuntimeError('Terminal operator does not reproduce its native output')
        reproduced = [models[0].classifier(maps[0][1].flatten(1)),
                      models[1].fc1(maps[1][1].flatten(1))]
        if any(not torch.equal(a, b) for a, b in zip(logits, reproduced)):
            raise RuntimeError('HFD target is not the final native classification feature')
    if not torch.equal(rng, torch.cuda.get_rng_state(device)):
        raise RuntimeError('Native terminal verification consumed CUDA RNG')
    feature.clear()
    for head in heads: head.features.clear()
    print('Terminal operators and original standalone classifiers reproduced exactly', flush=True)


def train(cfg, base, args, out):
    source, split = source_and_split(args.dataset, args.subject, args.seed, base)
    baseline = validate_baseline(args.dataset, cfg['pair'], args.subject, args.seed, source, split, base)
    channels, classes = split['x_cal_raw'].shape[1], config.load_dataset_config(args.dataset)['num_classes']
    bt, st = base['teacher_training']['cbramod'], base['student_training']['eegnet']
    adapters, models, heads, previous, optimizers = make_models(
        args.dataset, cfg['pair'], channels, classes, base, args.seed, args.device)
    previous.close()
    for opt in optimizers:
        opt.param_groups.pop()
    del previous
    feature = TerminalHFD(*models, args.method, bt['batch_size'], args.seed).to(args.device)
    for opt, direction in zip(optimizers, feature.directions):
        params = list(direction.parameters())
        if params:
            opt.add_param_group({'params': params})
    epochs = 2 if args.smoke else bt['epochs']
    weight = float(cfg['methods'][args.method]['weight'])
    train_uid, test_uid = split['trial_uid_cal'].tolist(), split['trial_uid_test'].tolist()
    identity = {
        'experiment_id': cfg['experiment_id'], 'dataset': args.dataset, 'pair': cfg['pair'],
        'method': args.method, 'subject': args.subject, 'seed': args.seed, 'smoke': args.smoke,
        'protocol': 'fewshot_stratified_random_30_70', 'epochs': epochs, 'batch_size': bt['batch_size'],
        'big_training': bt, 'small_training': st, 'feature_weight': weight,
        'warmup_epochs': cfg['warmup_epochs'], 'ramp_epochs': cfg['ramp_epochs'],
        'temperature': base['temperature'], 'alpha': base['alpha'], 'lambda': base['lambda'],
        'big_layers': list(BIG_LAYERS), 'small_layers': list(SMALL_LAYERS),
        'channels': channels, 'num_classes': classes, 'full_trial_seconds': 4,
        'source_manifest_sha256': source.source_manifest_sha256,
        'train_uid_sha256': uid_hash(train_uid), 'test_uid_sha256': uid_hash(test_uid),
        'train_count': len(train_uid), 'test_count': len(test_uid),
        'pretrained_sha256': sha256_file(config.weight_path('cbramod')),
        'config_sha256': sha256_file(args.config), 'implementation_sha256': fingerprint(),
        'baseline': baseline, 'initial_model_sha256': [state_digest(x) for x in models],
        'initial_head_sha256': [state_digest(x) for x in heads],
        'initial_feature_sha256': state_digest(feature),
        'adaptation': 'last_native_block_cross_operator_cosine; learned_32_knot_EEG_adapter; native_classifier_KL',
        'auxiliary_rng_isolated': True,
        'selection': 'fixed final epoch; no test selection', 'cpu_threads': 1, 'num_workers': 0,
    }
    state_path = out / 'training_state.pt'
    try:
        if (out / 'evaluation.json').is_file():
            result = json.loads((out / 'evaluation.json').read_text())
            if (result['identity'] != identity or
                    result['predictions_sha256'] != sha256_file(out / 'predictions.npz')):
                raise ValueError('Existing completion identity differs')
            return
        inputs = [adapter.preprocess(split['x_cal_raw']) for adapter in adapters]
        labels = torch.as_tensor(split['y_cal'], dtype=torch.long)
        if args.smoke: native_terminal_check(adapters, models, heads, feature, inputs, args.device)
        generator = torch.Generator().manual_seed(args.seed + 100003)
        history, start = [], 0
        if state_path.is_file():
            saved = torch.load(state_path, map_location='cpu', weights_only=False)
            if saved['identity'] != identity:
                raise ValueError('Resume identity differs')
            for module, state in zip(models, saved['models']): module.load_state_dict(state)
            for module, state in zip(heads, saved['heads']): module.load_state_dict(state)
            feature.load_state_dict(saved['feature'])
            feature.set_rng_states(saved['feature_rng'])
            for opt, state in zip(optimizers, saved['optimizers']): opt.load_state_dict(state)
            generator.set_state(saved['generator'])
            random.setstate(saved['python_rng'])
            np.random.set_state(saved['numpy_rng'])
            torch.set_rng_state(saved['torch_rng'])
            torch.cuda.set_rng_state(saved['cuda_rng'], args.device)
            start, history = saved['epoch'], saved['history']
        for epoch in range(start, epochs):
            started = time.monotonic()
            for module in models + heads + [feature]: module.train()
            order = torch.randperm(len(labels), generator=generator)
            ramp = 1.0 if args.smoke else min(1.0, max(
                0.0, (epoch + 1 - cfg['warmup_epochs']) / cfg['ramp_epochs']))
            loss_sum, seen, batch_count = np.zeros(6), 0, 0
            diagnostics, shapes, gradient = [{}, {}], None, None
            for pos in range(0, len(order), bt['batch_size']):
                rows = order[pos:pos + bt['batch_size']]
                target = labels[rows].to(args.device)
                for opt in optimizers: opt.zero_grad(set_to_none=True)
                feature.clear()
                logits = [adapter.forward(model, x[rows].to(args.device))[1]
                          for adapter, model, x in zip(adapters, models, inputs)]
                outputs = [[z] + heads[j](len(target)) for j, z in enumerate(logits)]
                ds_losses = mutual_losses(outputs, target, 'dml_ds', base['temperature'],
                                          base['alpha'], base['lambda'])
                maps = feature.take()
                aux = [torch.zeros((), device=args.device)] * 2
                details = [{}, {}]
                checking = args.smoke and epoch == 0 and batch_count == 0
                before_rng = torch.cuda.get_rng_state(args.device).clone() if checking else None
                before_buffers = [b.clone() for m in models for b in m.buffers()] if checking else None
                if ramp:
                    for role, direction in enumerate(feature.directions):
                        aux[role], details[role] = direction(maps[role], maps[1-role])
                    if checking:
                        if not torch.equal(before_rng, torch.cuda.get_rng_state(args.device)):
                            raise RuntimeError('Auxiliary HFD consumed native CUDA RNG')
                        after_buffers = [b for m in models for b in m.buffers()]
                        if any(not torch.equal(a, b) for a, b in zip(before_buffers, after_buffers)):
                            raise RuntimeError('Frozen peer HFD changed native BatchNorm buffers')
                        print('Native CUDA RNG and peer buffers preserved', flush=True)
                        check_gradients(feature.directions, maps, aux)
                        print('Both feature directions: own gradients finite, peer gradients absent', flush=True)
                    if batch_count == 0:
                        gradient = []
                        anchors = (models[0].backbone.patch_embedding.proj_in[0].weight, models[1].conv1.weight)
                        for role, anchor in enumerate(anchors):
                            a = torch.autograd.grad(ds_losses[role], anchor, retain_graph=True)[0].detach().flatten()
                            b = torch.autograd.grad(ramp * weight * aux[role], anchor, retain_graph=True)[0].detach().flatten()
                            gradient.append({'base_norm': a.norm().item(), 'weighted_aux_norm': b.norm().item(),
                                             'cosine': torch.nn.functional.cosine_similarity(a, b, dim=0).item()})
                feature.clear()
                for head in heads: head.features.clear()
                loss = sum(ds_losses) + ramp * weight * sum(aux)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite feature training loss')
                loss.backward()
                for opt in optimizers:
                    torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 1.0)
                    opt.step()
                n = len(rows)
                loss_sum += np.asarray([x.item() for x in ds_losses + aux] +
                                       [loss.item(), ramp * weight * sum(aux).item()]) * n
                for role in range(2):
                    for key, value in details[role].items():
                        diagnostics[role][key] = diagnostics[role].get(key, 0) + value * n
                shapes = [[list(x.shape[1:]) for x in side] for side in maps]
                seen += n
                batch_count += 1
                del maps, outputs, logits, aux, details, loss
                if args.smoke and batch_count >= 2: break
            factor = 0.5 * (1 + np.cos(np.pi * (epoch + 1) / epochs))
            for opt, training in zip(optimizers, (bt, st)):
                for group in opt.param_groups: group['lr'] = float(training['lr']) * factor
            entry = {'epoch': epoch + 1, 'ramp': ramp, 'examples': seen, 'batches': batch_count,
                     'mean_losses': (loss_sum / seen).tolist(), 'feature_shapes': shapes,
                     'diagnostics': [json_values({k: v / seen for k, v in d.items()}) for d in diagnostics],
                     'first_batch_gradient': gradient, 'seconds': time.monotonic() - started}
            history.append(entry)
            stop = args.stop_file and args.stop_file.exists()
            checkpoint = stop or args.stop_after or epoch + 1 == epochs or (epoch + 1) % cfg['checkpoint_interval'] == 0
            if checkpoint:
                atomic_save(state_path, {'identity': identity, 'epoch': epoch + 1,
                    'models': [x.state_dict() for x in models], 'heads': [x.state_dict() for x in heads],
                    'feature': feature.state_dict(), 'feature_rng': feature.rng_states(), 'optimizers': [x.state_dict() for x in optimizers],
                    'history': history, 'generator': generator.get_state(), 'python_rng': random.getstate(),
                    'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state(args.device)})
            print(f'{args.dataset} {args.method} subject={args.subject} seed={args.seed} '
                  f'epoch={epoch+1}/{epochs} base={loss_sum[:2]/seen} '
                  f'feature={loss_sum[2:4]/seen} ramp={ramp:.2f}', flush=True)
            if args.progress:
                atomic_json(args.progress, {'status': 'running', 'epoch': epoch + 1, 'epochs': epochs,
                                           'dataset': args.dataset, 'method': args.method,
                                           'subject': args.subject, 'seed': args.seed, 'gpu': args.gpu})
            if stop: raise SystemExit(75)
            if args.stop_after and epoch + 1 >= args.stop_after: return
        hashes = {'model_sha256': [state_digest(x) for x in models],
                  'head_sha256': [state_digest(x) for x in heads], 'feature_sha256': state_digest(feature),
                  'feature_rng_sha256': [hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in feature.rng_states()]}
        for head in heads: head.close()
        feature.close()
        # Smoke evaluation uses calibration only; formal evaluation is once at
        # the precommitted final epoch, through the original standalone models.
        raw = split['x_cal_raw'] if args.smoke else split['x_test_raw']
        truth = np.asarray(split['y_cal'] if args.smoke else split['y_test'], dtype=np.int64)
        uids = np.asarray(train_uid if args.smoke else test_uid, dtype=str)
        predictions = []
        for adapter, model in zip(adapters, models):
            model.eval()
            x = adapter.preprocess(raw)
            with torch.no_grad():
                pred = [adapter.forward(model, x[pos:pos + bt['batch_size']].to(args.device))[1].argmax(1).cpu()
                        for pos in range(0, len(x), bt['batch_size'])]
            predictions.append(torch.cat(pred).numpy())
        temporary = out / f'.predictions.{os.getpid()}.npz'
        np.savez_compressed(temporary, uids=uids, y=truth, big=predictions[0], small=predictions[1])
        os.replace(temporary, out / 'predictions.npz')
        atomic_json(out / 'training.json', {'identity': identity, 'history': history})
        result = {'identity': identity, 'evaluation_split': 'calibration' if args.smoke else 'test',
                  'predictions_sha256': sha256_file(out / 'predictions.npz'), **hashes}
        for role, pred in zip(('big', 'small'), predictions):
            result[role] = {'accuracy': float(np.mean(pred == truth) * 100),
                            'balanced_accuracy': float(balanced_accuracy_score(truth, pred) * 100),
                            'selected_epoch': epochs}
        atomic_json(out / 'evaluation.json', result)
        # Only this run's temporary resume state is retired after all retained
        # predictions, metrics, diagnostics and identity records are complete.
        state_path.unlink(missing_ok=True)
    finally:
        for head in heads: head.close()
        feature.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--subject', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-id', default='default')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--progress', type=Path)
    parser.add_argument('--expected-fingerprint')
    parser.add_argument('--expected-config-sha256')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    base = yaml.safe_load((ROOT / cfg['baseline_config']).read_text())
    if (cfg['pair'] != 'cbramod_eegnet' or args.dataset not in cfg['datasets'] or
            args.method not in cfg['methods'] or args.seed not in cfg['seeds'] or
            not 0 <= args.subject < cfg['datasets'][args.dataset] or
            base['calibration_fraction'] != .3 or base['test_fraction'] != .7 or
            cfg['checkpoint_interval'] < 1 or cfg['ramp_epochs'] < 1):
        parser.error('Invalid or unmatched experiment setting')
    if (args.stop_after and not args.smoke) or not args.smoke_id.isalnum():
        parser.error('Stop-after is smoke only; smoke id must be alphanumeric')
    if args.expected_fingerprint and fingerprint() != args.expected_fingerprint:
        raise ValueError('Frozen implementation changed')
    if args.expected_config_sha256 and sha256_file(args.config) != args.expected_config_sha256:
        raise ValueError('Frozen config changed')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        raise RuntimeError('Expected one PyTorch inter-op thread')
    torch.cuda.set_device(args.gpu)
    args.device = torch.device(f'cuda:{args.gpu}')
    out = run_folder(cfg, args.dataset, args.method, args.subject, args.seed,
                     args.smoke_id if args.smoke else None)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(cfg, base, args, out)
    if args.progress and not args.stop_after:
        atomic_json(args.progress, {'status': 'completed', 'dataset': args.dataset,
                                  'method': args.method, 'subject': args.subject, 'seed': args.seed,
                                  'gpu': args.gpu})


if __name__ == '__main__':
    main()
