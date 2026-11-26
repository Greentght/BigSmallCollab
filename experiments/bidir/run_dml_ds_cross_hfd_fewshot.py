"""Add cross-operator HFD to the existing CBraMod–EEGNet DML-DS run."""

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

import numpy as np
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score

import config
from collab.dcm import mutual_losses
from collab.dml_ds_heads import build_heads
from collab.hfd_cross_operator import CrossOperatorHFD
from data.loso_source import load_dataset, sha256_file
from experiments.distill.run_fewshot_structure_kd import _split_subject
from experiments.storage import require_external_output
from models import get_adapter

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/experiments/fewshot_dml_ds_cross_hfd_004.yaml'


def atomic_json(path, value):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save(path, value):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def uid_hash(values):
    return hashlib.sha256(json.dumps(list(values), ensure_ascii=False).encode()).hexdigest()


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def implementation_hashes():
    paths = [Path(__file__), ROOT / 'collab/hfd_cross_operator.py', ROOT / 'collab/dcm.py',
             ROOT / 'collab/dml_ds_heads.py', ROOT / 'config.py', ROOT / 'data/split.py',
             ROOT / 'data/loso_source.py', ROOT / 'data/preproc.py', ROOT / 'data/channels.py',
             ROOT / 'models/base.py', ROOT / 'models/__init__.py',
             ROOT / 'models/cbramod/adapter.py', ROOT / 'models/cbramod/cbramod.py',
             ROOT / 'models/cbramod/criss_cross_transformer.py',
             ROOT / 'models/eegnet/adapter.py', ROOT / 'models/eegnet/residual_eegnet.py',
             ROOT / 'configs/datasets/BNCI2014004.yaml', ROOT / 'configs/models/cbramod.yaml',
             ROOT / 'configs/models/eegnet.yaml',
             ROOT / 'experiments/distill/run_fewshot_structure_kd.py',
             ROOT / 'experiments/bidir/run_dml_ds_fewshot_pairs.py']
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def baseline_for(cfg, base, subject, seed, source, split):
    folder = Path(cfg['baseline_root']) / f'subject_{subject}' / f'seed_{seed}'
    result_path, prediction_path = folder / 'evaluation.json', folder / 'predictions.npz'
    if not result_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(f'Matching DML-DS baseline is not complete: {folder}')
    result = json.loads(result_path.read_text())
    i = result['identity']
    expected = (cfg['dataset'], cfg['pair'], 'dml_ds', subject, seed,
                'fewshot_stratified_random_30_70', False,
                base['teacher_training']['cbramod']['epochs'],
                base['teacher_training']['cbramod']['batch_size'])
    actual = tuple(i[k] for k in ('dataset', 'pair', 'condition', 'subject', 'seed',
                                  'protocol', 'smoke', 'epochs', 'batch_size'))
    if (actual != expected or i['big_training'] != base['teacher_training']['cbramod'] or
            i['small_training'] != base['student_training']['eegnet'] or
            i['source_manifest_sha256'] != source.source_manifest_sha256 or
            i['pretrained_sha256'] != sha256_file(config.weight_path('cbramod')) or
            i['train_uid_sha256'] != uid_hash(split['trial_uid_cal'].tolist()) or
            i['test_uid_sha256'] != uid_hash(split['trial_uid_test'].tolist())):
        raise ValueError(f'DML-DS baseline identity differs: {folder}')
    for name, expected_hash in i['code_sha256'].items():
        path = ROOT / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f'DML-DS baseline code/config changed: {name}')
    with np.load(prediction_path, allow_pickle=False) as pred:
        uids, truth = pred['uids'].tolist(), pred['y']
        if (len(uids) != result['test_count'] or len(set(uids)) != len(uids) or
                uid_hash(uids) != i['test_uid_sha256'] or
                not np.array_equal(truth, split['y_test'])):
            raise ValueError(f'DML-DS baseline test identity differs: {folder}')
        for role in ('big', 'small'):
            accuracy = float(np.mean(pred[role] == truth) * 100)
            if not np.isclose(accuracy, result[f'{role}_accuracy'], atol=1e-9, rtol=0):
                raise ValueError(f'DML-DS baseline prediction mismatch: {folder}')
    return {'result_path': str(result_path), 'result_sha256': sha256_file(result_path),
            'predictions_sha256': sha256_file(prediction_path),
            'code_sha256': i['code_sha256']}


def make_models(base, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    big_cfg = config.load_model_config('cbramod', 'BNCI2014004', 'fewshot')
    big_cfg.update(in_channels=3, samples=1000, skip_preprocess=False)
    small_cfg = config.load_model_config('eegnet', 'BNCI2014004', 'fewshot')
    small_cfg.update(in_channels=3, samples=1000)
    adapters = [get_adapter(name, device=device, dataset_name='BNCI2014004', **settings)
                for name, settings in (('cbramod', big_cfg), ('eegnet', small_cfg))]
    models = [adapter.build(2) for adapter in adapters]
    heads = [build_heads(model, name, 2).to(device)
             for model, name in zip(models, ('cbramod', 'eegnet'))]
    # Construct HFD only after the baseline models and auxiliary heads. Its
    # local CPU generator does not alter either model's dropout RNG stream.
    hfd = CrossOperatorHFD(models[0], models[1], seed).to(device)
    optimizers = []
    for model, head, projector, training in zip(
            models, heads, (hfd.big_to_small, hfd.small_to_big),
            (base['teacher_training']['cbramod'], base['student_training']['eegnet'])):
        model_params = list(model.parameters()) + list(head.parameters())
        # Keep model/head clipping identical to the baseline; the projector is
        # its own optimizer group and does not alter that clipping norm.
        groups = [{'params': model_params}, {'params': list(projector.parameters())}]
        cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[training['optimizer']]
        optimizers.append(cls(groups, lr=float(training['lr']), weight_decay=float(training['weight_decay'])))
    return adapters, models, heads, hfd, optimizers


def run(cfg, base, args, out):
    source = load_dataset(cfg['dataset'])
    split = _split_subject(source, cfg['dataset'], args.subject, args.seed,
                           base['calibration_fraction'], base['test_fraction'])
    cal_uid, test_uid = split['trial_uid_cal'].tolist(), split['trial_uid_test'].tolist()
    if (len(set(cal_uid)) != len(cal_uid) or len(set(test_uid)) != len(test_uid) or
            set(cal_uid) & set(test_uid)):
        raise ValueError('Calibration/test trial UIDs are not unique and disjoint')
    baseline = baseline_for(cfg, base, args.subject, args.seed, source, split)
    adapters, models, heads, hfd, optimizers = make_models(base, args.seed, args.device)
    train = [adapter.preprocess(split['x_cal_raw']) for adapter in adapters]
    labels = torch.as_tensor(split['y_cal'], dtype=torch.long)
    if (tuple(train[0].shape[1:]) != (3, 4, 200) or
            tuple(train[1].shape[1:]) != (3, 1000)):
        raise ValueError(f'Unexpected few-shot HFD input shapes: {[x.shape for x in train]}')
    epochs = 2 if args.smoke else int(base['teacher_training']['cbramod']['epochs'])
    batch_size = int(base['teacher_training']['cbramod']['batch_size'])
    identity = {'experiment_id': cfg['experiment_id'], 'dataset': cfg['dataset'], 'pair': cfg['pair'],
                'condition': cfg['condition'], 'subject': args.subject, 'seed': args.seed,
                'protocol': 'fewshot_stratified_random_30_70', 'smoke': args.smoke,
                'hfd_method': 'paper_eq2_native_peer_operators_eeg_adaptation',
                'epochs': epochs, 'batch_size': batch_size, 'final_epoch_only': True,
                'source_manifest_sha256': source.source_manifest_sha256,
                'train_uid_sha256': uid_hash(cal_uid), 'test_uid_sha256': uid_hash(test_uid),
                'train_count': len(cal_uid), 'test_count': len(test_uid),
                'pretrained_sha256': sha256_file(config.weight_path('cbramod')),
                'big_training': base['teacher_training']['cbramod'],
                'small_training': base['student_training']['eegnet'],
                'baseline_config_sha256': sha256_file(ROOT / cfg['baseline_config']),
                'baseline_result_sha256': baseline['result_sha256'],
                'baseline_predictions_sha256': baseline['predictions_sha256'],
                'implementation_sha256': implementation_hashes(),
                'config_sha256': sha256_file(args.config),
                'initial_model_sha256': [state_digest(model) for model in models],
                'initial_head_sha256': [state_digest(head) for head in heads]}
    state_path, result_path = out / 'training_state.pt', out / 'evaluation.json'
    try:
        if result_path.exists():
            result = json.loads(result_path.read_text())
            prediction, training = out / 'predictions.npz', out / 'training.json'
            if (result['identity'] != identity or not prediction.is_file() or not training.is_file() or
                    result['predictions_sha256'] != sha256_file(prediction) or
                    result['evaluation_split'] != ('calibration' if args.smoke else 'test') or
                    json.loads(training.read_text()).get('identity') != identity):
                raise RuntimeError('Completed DML-DS+HFD identity mismatch')
            state_path.unlink(missing_ok=True)
            return
        generator = torch.Generator().manual_seed(args.seed + 100003)
        history, start = [], 0
        if state_path.exists():
            saved = torch.load(state_path, map_location='cpu', weights_only=False)
            if saved['identity'] != identity:
                raise RuntimeError('DML-DS+HFD resume identity mismatch')
            for model, value in zip(models, saved['models']): model.load_state_dict(value)
            for head, value in zip(heads, saved['heads']): head.load_state_dict(value)
            hfd.load_state_dict(saved['hfd'])
            for opt, value in zip(optimizers, saved['optimizers']): opt.load_state_dict(value)
            generator.set_state(saved['generator'])
            random.setstate(saved['python_rng'])
            np.random.set_state(saved['numpy_rng'])
            torch.set_rng_state(saved['torch_rng'])
            if args.device.type == 'cuda': torch.cuda.set_rng_state(saved['cuda_rng'], args.device)
            start, history = saved['epoch'], saved['history']
        for epoch in range(start, epochs):
            for model in models: model.train()
            order = torch.randperm(len(labels), generator=generator)
            ramp = 1.0 if args.smoke else min(1.0, max(
                0.0, (epoch + 1 - cfg['hfd_warmup_epochs']) / cfg['hfd_ramp_epochs']))
            sums, batches = np.zeros(3, dtype=np.float64), 0
            gradient_probe = None
            for pos in range(0, len(order), batch_size):
                rows = order[pos:pos + batch_size]
                inputs = [x[rows].to(args.device) for x in train]
                target = labels[rows].to(args.device)
                for opt in optimizers: opt.zero_grad(set_to_none=True)
                hfd.clear()
                logits = [adapter.forward(model, x)[1]
                          for adapter, model, x in zip(adapters, models, inputs)]
                outputs = [[z] + heads[j](len(target)) for j, z in enumerate(logits)]
                ds_loss = sum(mutual_losses(outputs, target, 'dml_ds', base['temperature'],
                                            base['alpha'], base['lambda']))
                hfd_loss = torch.zeros((), device=args.device)
                if ramp:
                    captured = (hfd.features['big_first'], hfd.features['small_first'])
                    losses = hfd.losses()
                    if args.smoke and epoch == 0 and batches == 0:
                        for j, projector in enumerate((hfd.big_to_small, hfd.small_to_big)):
                            grads = torch.autograd.grad(losses[j],
                                (captured[j], captured[1-j], projector.weight),
                                allow_unused=True, retain_graph=True)
                            if (grads[0] is None or grads[1] is not None or grads[2] is None or
                                    not torch.isfinite(grads[0]).all() or not torch.isfinite(grads[2]).all() or
                                    not grads[0].abs().sum() > 0 or not grads[2].abs().sum() > 0):
                                raise RuntimeError('DML-DS+HFD directional gradient isolation failed')
                        print('DML-DS+HFD directional gradient isolation passed', flush=True)
                    hfd_loss = sum(losses)
                    if batches == 0:
                        # One calibration minibatch per epoch; gradients are
                        # measured before backward without changing optimizer state.
                        parameters = (models[0].backbone.patch_embedding.proj_in[0].weight,
                                      models[1].conv1.weight)
                        base_grads = torch.autograd.grad(ds_loss, parameters,
                                                         retain_graph=True, allow_unused=True)
                        gradient_probe = []
                        for j, parameter in enumerate(parameters):
                            grad = torch.autograd.grad(ramp * float(cfg['hfd_weight']) * losses[j],
                                                       parameter, retain_graph=True, allow_unused=True)[0]
                            ref = base_grads[j]
                            if grad is None or ref is None:
                                raise RuntimeError('Missing shallow-parameter gradient in HFD probe')
                            ref_norm, hfd_norm = ref.norm(), grad.norm()
                            denom = ref_norm * hfd_norm
                            cosine = float((ref * grad).sum().div(denom).item()) if denom > 0 else None
                            gradient_probe.append({'baseline_norm': float(ref_norm.item()),
                                                   'weighted_hfd_norm': float(hfd_norm.item()),
                                                   'cosine': cosine})
                loss = ds_loss + ramp * float(cfg['hfd_weight']) * hfd_loss
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite DML-DS+HFD loss')
                loss.backward()
                for opt in optimizers:
                    torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 1.0)
                    opt.step()
                sums += [ds_loss.item(), hfd_loss.item(), loss.item()]
                batches += 1
                hfd.clear()
                if args.smoke and batches >= 2: break
            factor = 0.5 * (1 + np.cos(np.pi * (epoch + 1) / epochs))
            for opt, training in zip(optimizers, (base['teacher_training']['cbramod'],
                                                  base['student_training']['eegnet'])):
                for group in opt.param_groups: group['lr'] = float(training['lr']) * factor
            history.append({'epoch': epoch + 1, 'ramp': ramp, 'mean_losses': (sums / batches).tolist(),
                            'first_calibration_batch_gradient_probe': gradient_probe})
            stop = bool(args.stop_file and args.stop_file.exists())
            stop_after = bool(args.stop_after and epoch + 1 >= args.stop_after)
            if stop or stop_after or epoch + 1 == epochs or (epoch + 1) % cfg['checkpoint_interval'] == 0:
                atomic_save(state_path, {'identity': identity, 'epoch': epoch + 1,
                    'models': [model.state_dict() for model in models],
                    'heads': [head.state_dict() for head in heads], 'hfd': hfd.state_dict(),
                    'optimizers': [opt.state_dict() for opt in optimizers],
                    'history': history, 'generator': generator.get_state(),
                    'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                    'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state(args.device) if args.device.type == 'cuda' else None})
            print(f'DML-DS+HFD subject={args.subject} seed={args.seed} epoch={epoch+1}/{epochs} '
                  f'ds={sums[0]/batches:.5f} hfd={sums[1]/batches:.5f} ramp={ramp:.2f}', flush=True)
            if args.progress:
                atomic_json(args.progress, {'status': 'running', 'epoch': epoch+1, 'epochs': epochs,
                                             'subject': args.subject, 'seed': args.seed})
            if stop:
                if args.progress: atomic_json(args.progress, {'status': 'paused', 'epoch': epoch+1})
                raise SystemExit(75)
            if stop_after: return
        head_hashes = [state_digest(head) for head in heads]
        hfd_hash = state_digest(hfd)
        for head in heads: head.close()
        hfd.close()
        raw = split['x_cal_raw'] if args.smoke else split['x_test_raw']
        truth = np.asarray(split['y_cal'] if args.smoke else split['y_test'], dtype=np.int64)
        uids = np.asarray(cal_uid if args.smoke else test_uid, dtype=str)
        predictions = []
        for adapter, model in zip(adapters, models):
            model.eval()
            data = adapter.preprocess(raw)
            parts = []
            with torch.no_grad():
                for pos in range(0, len(data), batch_size):
                    parts.append(adapter.forward(model, data[pos:pos+batch_size].to(args.device))[1]
                                 .argmax(1).cpu())
            predictions.append(torch.cat(parts).numpy())
        temporary = out / f'.predictions.{os.getpid()}.tmp.npz'
        try:
            np.savez_compressed(temporary, uids=uids, y=truth, big=predictions[0], small=predictions[1])
            os.replace(temporary, out / 'predictions.npz')
        finally:
            temporary.unlink(missing_ok=True)
        atomic_json(out / 'training.json', {'identity': identity, 'history': history})
        result = {'identity': identity, 'evaluation_split': 'calibration' if args.smoke else 'test',
                  'predictions_sha256': sha256_file(out / 'predictions.npz'),
                  'model_sha256': [state_digest(model) for model in models],
                  'head_sha256': head_hashes, 'hfd_sha256': hfd_hash}
        for role, predicted in zip(('big', 'small'), predictions):
            result[role] = {'accuracy': float(np.mean(predicted == truth) * 100),
                            'balanced_accuracy': float(balanced_accuracy_score(truth, predicted) * 100),
                            'selected_epoch': epochs}
        atomic_json(result_path, result)
        state_path.unlink(missing_ok=True)
    finally:
        for head in heads: head.close()
        hfd.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG)
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
    base = yaml.safe_load((ROOT / cfg['baseline_config']).read_text())
    if args.expected_config_sha256 and sha256_file(args.config) != args.expected_config_sha256:
        raise ValueError('Queued DML-DS+HFD configuration changed')
    digest = hashlib.sha256(json.dumps(implementation_hashes(), sort_keys=True).encode()).hexdigest()
    if args.expected_implementation_sha256 and digest != args.expected_implementation_sha256:
        raise ValueError('Queued DML-DS+HFD implementation changed')
    if (cfg['dataset'] != 'BNCI2014004' or cfg['pair'] != 'cbramod_eegnet' or
            cfg['condition'] != 'dml_ds_cross_hfd' or args.subject not in cfg['subjects'] or
            args.seed not in cfg['seeds'] or cfg['checkpoint_interval'] < 1 or
            base['calibration_fraction'] != 0.3 or base['test_fraction'] != 0.7):
        raise ValueError('Invalid DML-DS+HFD experiment identity or protocol')
    if (args.cpu or args.stop_after) and not args.smoke:
        parser.error('--cpu and --stop-after are available only for smoke')
    if Path(args.smoke_id).name != args.smoke_id: parser.error('invalid smoke id')
    if not args.cpu and not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    args.device = torch.device('cpu' if args.cpu else f'cuda:{args.gpu}')
    if args.device.type == 'cpu': torch.backends.mkldnn.enabled = False
    else: torch.cuda.set_device(args.device)
    prefix = Path(cfg['output_root']) / ('smoke' if args.smoke else 'runs')
    if args.smoke: prefix /= args.smoke_id
    out = require_external_output(prefix / f'subject_{args.subject}' / f'seed_{args.seed}')
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(cfg, base, args, out)
    if args.progress and not args.stop_after:
        atomic_json(args.progress, {'status': 'completed', 'subject': args.subject, 'seed': args.seed})


if __name__ == '__main__':
    main()
