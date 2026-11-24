"""Fixed-endpoint DCM training for the two BNCI2014001-4 model pairs."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

import argparse
import fcntl
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

import config
from collab.dcm import DCMHeads, mutual_losses
from data.loso_source import load_dataset, sha256_file
from experiments.distill.mirepnet_t2s_data import ensure_uid_alignment
from experiments.finetune.run_neurogpt_loso import _adapter_for, _load_spec, DEFAULT_CONFIG, _BatchSchedule
from experiments.storage import require_external_output
from models import get_adapter

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/experiments/loso/dcm.yaml'


def atomic_save(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair', required=True, choices=('cbramod_ifnet', 'neurogpt_eegnet'))
    parser.add_argument('--condition', required=True, choices=('none', 'dml_kl', 'dml_ds', 'dcm'))
    parser.add_argument('--fold', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if args.stop_after and not args.smoke:
        parser.error('--stop-after is only supported for smoke runs')
    cfg = yaml.safe_load(CONFIG.read_text())
    if args.fold not in cfg['folds'] or args.seed not in cfg['seeds']:
        raise ValueError('Fold or seed outside configured experiment')
    out = require_external_output(Path(cfg['output_root']) / ('smoke' if args.smoke else 'runs') / args.pair / args.condition /
                                  f'fold_{args.fold}' / f'seed_{args.seed}')
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, cfg, out)


def run(args, cfg, out):
    dataset = cfg['dataset']
    pair = cfg['pairs'][args.pair]
    big_name, small_name = args.pair.split('_')
    big = load_dataset(dataset, 'cbramod' if big_name == 'cbramod' else None)
    small = load_dataset(dataset, small_name)
    big, small = ensure_uid_alignment(big, small)
    subjects = np.asarray(small.subjects)
    train_idx = np.flatnonzero(subjects != args.fold)
    test_idx = np.flatnonzero(subjects == args.fold)
    if set(np.unique(subjects)) != set(cfg['folds']) or not len(test_idx):
        raise ValueError('Incomplete LOSO subject set')
    if sorted(np.unique(small.y).tolist()) != [0, 1, 2, 3]:
        raise ValueError('Expected four contiguous classes')
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if big_name == 'neurogpt':
        exp, spec, _ = _load_spec(DEFAULT_CONFIG, dataset)
        big_adapter, _ = _adapter_for(dataset, 22, device, exp, spec)
    else:
        model_cfg = config.load_model_config(big_name, dataset, 'loso')
        model_cfg.update(in_channels=22, skip_preprocess=big.input_manifest_path is not None)
        big_adapter = get_adapter(big_name, device=device, dataset_name=dataset, **model_cfg)
    small_cfg = config.load_model_config(small_name, dataset, 'loso')
    small_cfg.update(in_channels=22, samples=1000)
    small_adapter = get_adapter(small_name, device=device, dataset_name=dataset, **small_cfg)
    adapters = [big_adapter, small_adapter]
    models = [adapter.build(4) for adapter in adapters]
    with_heads = args.condition in ('dml_ds', 'dcm')
    heads = ([DCMHeads(model, name, 4).to(device)
              for model, name in zip(models, (big_name, small_name))] if with_heads else [])
    optimizers = []
    for index, role in enumerate(('big', 'small')):
        params = list(models[index].parameters())
        if with_heads:
            params += list(heads[index].parameters())
        optimizers.append(torch.optim.AdamW(params, lr=float(pair[role + '_lr']),
                                             weight_decay=float(pair[role + '_weight_decay'])))
    batch_size = int(pair['batch_size'])
    total_steps = int(pair.get('steps', pair.get('epochs', 0) * int(np.ceil(len(train_idx) / batch_size))))
    if args.smoke:
        total_steps = 4
    schedule = _BatchSchedule(len(train_idx), batch_size, args.seed)
    source_identity = {
        'experiment': cfg['experiment_id'], 'pair': args.pair, 'condition': args.condition,
        'fold': args.fold, 'seed': args.seed, 'config_sha256': sha256_file(CONFIG),
        'implementation_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in (
            Path(__file__), ROOT / 'collab/dcm.py', ROOT / f'models/{big_name}/adapter.py',
            ROOT / f'models/{big_name}/{big_name}.py', ROOT / f'models/{small_name}/adapter.py',
            ROOT / ('models/ifnet/ifnet.py' if small_name == 'ifnet' else 'models/eegnet/residual_eegnet.py'),
            ROOT / 'models/cbramod/criss_cross_transformer.py', ROOT / 'models/base.py',
            ROOT / 'experiments/finetune/run_neurogpt_loso.py',
            ROOT / 'experiments/distill/mirepnet_t2s_data.py', ROOT / 'config.py',
            ROOT / 'data/preproc.py', ROOT / 'data/loso_source.py', ROOT / f'configs/models/{big_name}.yaml',
            ROOT / f'configs/models/{small_name}.yaml', ROOT / 'configs/datasets/BNCI2014001-4.yaml')},
        'source_manifest_sha256': small.source_manifest_sha256,
        'input_manifest_sha256': [big.input_manifest_sha256, small.input_manifest_sha256],
        'pretrained_sha256': sha256_file(config.weight_path(big_name)),
        'channel_matrix_sha256': sha256_file('/data1/llx/pre_weight/neurogpt/tMatrix_value.npy') if big_name == 'neurogpt' else None,
        'total_steps': total_steps, 'smoke': args.smoke,
    }
    result_path, state_path = out / 'evaluation.json', out / 'training_state.pt'
    if result_path.exists():
        recorded = json.loads(result_path.read_text())
        if recorded['identity'] != source_identity or not (out / 'predictions.npz').is_file():
            raise RuntimeError('Completed result identity or predictions mismatch')
        state_path.unlink(missing_ok=True)
        return
    start = 0
    if state_path.exists():
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        if state['identity'] != source_identity:
            raise RuntimeError('Resume identity mismatch')
        for model, saved in zip(models, state['models']): model.load_state_dict(saved)
        for head, saved in zip(heads, state['heads']): head.load_state_dict(saved)
        for optimizer, saved in zip(optimizers, state['optimizers']): optimizer.load_state_dict(saved)
        schedule.load_state_dict(state['schedule'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'], device)
        start = state['step']
    train_inputs = [adapter.preprocess(np.asarray(view.x[train_idx])) for adapter, view in zip(adapters, (big, small))]
    labels = torch.as_tensor(small.y[train_idx], dtype=torch.long)
    for model in models: model.train()
    for step in range(start, total_steps):
        rows = schedule.next()
        batch = [x[rows].to(device) for x in train_inputs]
        target = labels[rows].to(device)
        for optimizer in optimizers: optimizer.zero_grad(set_to_none=True)
        logits = [adapter.forward(model, x)[1] for adapter, model, x in zip(adapters, models, batch)]
        outputs = [[z] + (heads[i](len(target)) if with_heads else []) for i, z in enumerate(logits)]
        losses = mutual_losses(outputs, target, args.condition, cfg['temperature'], cfg['alpha'], cfg['lambda'])
        loss = sum(losses)
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite DCM loss')
        loss.backward()
        for index, optimizer in enumerate(optimizers):
            if args.pair == 'neurogpt_eegnet':
                warmup = int(pair['warmup_steps'])
                factor = ((step + 1) / warmup if step < warmup else (total_steps - step) / (total_steps - warmup))
            else:
                factor = 0.5 * (1.0 + np.cos(np.pi * step / total_steps))
            role = ('big', 'small')[index]
            for group in optimizer.param_groups: group['lr'] = float(pair[role + '_lr']) * factor
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 1.0)
            optimizer.step()
        if args.smoke or (step + 1) % int(cfg['checkpoint_interval']) == 0 or step + 1 == total_steps:
            state = {'identity': source_identity, 'step': step + 1,
                     'models': [m.state_dict() for m in models], 'heads': [h.state_dict() for h in heads],
                     'optimizers': [o.state_dict() for o in optimizers], 'schedule': schedule.state_dict(),
                     'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                     'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(device)}
            atomic_save(state_path, state)
            print(f'{args.pair} {args.condition} fold={args.fold} seed={args.seed} step={step+1}/{total_steps} loss={loss.item():.5f}', flush=True)
        if args.stop_after and step + 1 >= args.stop_after:
            return
    for head in heads: head.close()
    predictions = []
    for adapter, model, view in zip(adapters, models, (big, small)):
        model.eval()
        x = adapter.preprocess(np.asarray(view.x[test_idx]))
        parts = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                parts.append(adapter.forward(model, x[start:start + batch_size].to(device))[1].argmax(1).cpu())
        predictions.append(torch.cat(parts).numpy())
    tmp = out / 'predictions.tmp.npz'
    np.savez_compressed(tmp, uids=np.asarray(small.trial_uid[test_idx], dtype=str),
                        y=small.y[test_idx], big=predictions[0], small=predictions[1])
    os.replace(tmp, out / 'predictions.npz')
    result = {'identity': source_identity, 'big_accuracy': float(np.mean(predictions[0] == small.y[test_idx]) * 100),
              'small_accuracy': float(np.mean(predictions[1] == small.y[test_idx]) * 100)}
    tmp = out / 'evaluation.tmp.json'
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, result_path)
    state_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
