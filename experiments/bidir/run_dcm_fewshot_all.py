"""Multi-dataset subject-wise 30/70 broadband DCM experiment."""
import os
os.environ['BIGSMALLCOLLAB_CPU_THREADS'] = '1'
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

import argparse
import fcntl
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

import config
from collab.dcm import DCMHeads, mutual_losses
from data.loso_source import load_dataset_subjects, sha256_file
from experiments.distill.run_fewshot_structure_kd import _split_subject
from experiments.finetune.labram_alexmi_broadband import load_alexmi_broadband
from experiments.finetune.run_neurogpt_loso import _adapter_for, _load_spec, DEFAULT_CONFIG
from experiments.storage import require_external_output
from models import get_adapter

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/experiments/fewshot_dcm_all.yaml'


def atomic_save(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def digest_uids(uids):
    return hashlib.sha256(json.dumps(list(uids), ensure_ascii=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=('BNCI2014004', 'BNCI2014001', 'BNCI2015001', 'AlexMI'))
    parser.add_argument('--pair', required=True, choices=('cbramod_ifnet', 'neurogpt_eegnet'))
    parser.add_argument('--condition', required=True, choices=('none', 'dml_kl', 'dml_ds', 'dcm'))
    parser.add_argument('--subject', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if args.stop_after and not args.smoke:
        parser.error('--stop-after requires --smoke')
    cfg = yaml.safe_load(CONFIG.read_text())
    if not 0 <= args.subject < cfg['datasets'][args.dataset] or args.seed not in cfg['seeds']:
        raise ValueError('Subject or seed outside experiment')
    root = require_external_output(Path(cfg['output_root']) / ('smoke' if args.smoke else 'runs') / args.dataset /
                                   args.pair / args.condition / f'subject_{args.subject}' / f'seed_{args.seed}')
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, cfg, root)


def run(args, cfg, root):
    dataset = args.dataset
    pair = cfg['pairs'][args.pair]
    big_name, small_name = args.pair.split('_')
    if dataset == 'AlexMI':
        from mne.filter import resample
        source = load_alexmi_broadband()
        # Reproduce EEGDataset's existing AlexMI 512 -> 250 Hz conversion
        # and concatenate the first second to its three-second signal.
        x = resample(source.x, up=125, down=256, axis=2)[..., :750].astype(np.float32)
        source.x = np.concatenate([x, x[..., :250]], axis=2)
    else:
        source = load_dataset_subjects(dataset, [args.subject])
    split = _split_subject(source, dataset, args.subject, args.seed,
                           cfg['calibration_fraction'], cfg['test_fraction'])
    train_uid = split['trial_uid_cal'].tolist()
    test_uid = split['trial_uid_test'].tolist()
    if len(set(train_uid)) != len(train_uid) or len(set(test_uid)) != len(test_uid) or set(train_uid) & set(test_uid):
        raise ValueError('Calibration and test UIDs must be unique and disjoint')
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    channels = int(split['x_cal_raw'].shape[1])
    num_classes = int(config.load_dataset_config(dataset)['num_classes'])
    if big_name == 'neurogpt':
        exp, spec, _ = _load_spec(DEFAULT_CONFIG, dataset)
        big_adapter, _ = _adapter_for(dataset, channels, device, exp, spec)
    else:
        big_cfg = config.load_model_config(big_name, dataset, 'fewshot')
        big_cfg.update(in_channels=channels, samples=1000)
        big_adapter = get_adapter(big_name, device=device, dataset_name=dataset, **big_cfg)
    small_cfg = config.load_model_config(small_name, dataset, 'fewshot')
    small_cfg.update(in_channels=channels, samples=1000)
    small_adapter = get_adapter(small_name, device=device, dataset_name=dataset, **small_cfg)
    adapters = [big_adapter, small_adapter]
    models = [adapter.build(num_classes) for adapter in adapters]
    with_heads = args.condition in ('dml_ds', 'dcm')
    heads = ([DCMHeads(model, name, num_classes).to(device)
              for model, name in zip(models, (big_name, small_name))] if with_heads else [])
    optimizers = []
    for index, role in enumerate(('big', 'small')):
        params = list(models[index].parameters()) + (list(heads[index].parameters()) if with_heads else [])
        optimizers.append(torch.optim.AdamW(params, lr=float(pair[role + '_lr']),
                                             weight_decay=float(pair[role + '_weight_decay'])))
    epochs = 2 if args.smoke else int(pair['epochs'])
    batch_size = int(pair['batch_size'])
    code_paths = [Path(__file__), CONFIG, ROOT / 'collab/dcm.py', ROOT / 'config.py',
                  ROOT / 'data/split.py', ROOT / 'data/loso_source.py', ROOT / 'data/preproc.py',
                  ROOT / 'models/base.py', ROOT / 'models/cbramod/criss_cross_transformer.py',
                  ROOT / 'experiments/distill/run_fewshot_structure_kd.py',
                  ROOT / 'experiments/finetune/run_neurogpt_loso.py',
                  ROOT / 'experiments/finetune/labram_alexmi_broadband.py',
                  ROOT / f'configs/datasets/{dataset}.yaml',
                  ROOT / f'configs/models/{big_name}.yaml', ROOT / f'configs/models/{small_name}.yaml',
                  ROOT / f'models/{big_name}/adapter.py', ROOT / f'models/{big_name}/{big_name}.py',
                  ROOT / f'models/{small_name}/adapter.py',
                  ROOT / ('models/ifnet/ifnet.py' if small_name == 'ifnet' else 'models/eegnet/residual_eegnet.py')]
    identity = {'experiment': cfg['experiment_id'], 'dataset': dataset, 'pair': args.pair,
                'condition': args.condition, 'subject': args.subject, 'seed': args.seed,
                'protocol': 'fewshot_stratified_random_30_70', 'epochs': epochs,
                'batch_size': batch_size, 'smoke': args.smoke, 'source_manifest_sha256': source.source_manifest_sha256,
                'channels': channels, 'num_classes': num_classes,
                'input_preprocessing': ('AlexMI broadband 512Hz to 250Hz, first 750 samples then repeat first 250 samples' if dataset == 'AlexMI' else 'verified broadband canonical 4s at 250Hz'),
                'train_uid_sha256': digest_uids(train_uid), 'test_uid_sha256': digest_uids(test_uid),
                'pretrained_sha256': sha256_file(config.weight_path(big_name)),
                'channel_matrix_sha256': sha256_file('/data1/llx/pre_weight/neurogpt/tMatrix_value.npy') if big_name == 'neurogpt' and channels == 22 else None,
                'code_sha256': {str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths}}
    result_path, state_path = root / 'evaluation.json', root / 'training_state.pt'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['identity'] != identity or not (root / 'predictions.npz').is_file():
            raise RuntimeError('Completed result identity or predictions mismatch')
        state_path.unlink(missing_ok=True)
        return
    x_train = [adapter.preprocess(split['x_cal_raw']) for adapter in adapters]
    labels = torch.as_tensor(split['y_cal'], dtype=torch.long)
    generator = torch.Generator().manual_seed(args.seed + 100003)
    start = 0
    if state_path.exists():
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        if state['identity'] != identity: raise RuntimeError('Resume identity mismatch')
        for model, value in zip(models, state['models']): model.load_state_dict(value)
        for head, value in zip(heads, state['heads']): head.load_state_dict(value)
        for optimizer, value in zip(optimizers, state['optimizers']): optimizer.load_state_dict(value)
        generator.set_state(state['generator'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'], device)
        start = state['epoch']
    for epoch in range(start, epochs):
        for model in models: model.train()
        order = torch.randperm(len(labels), generator=generator)
        for start_batch in range(0, len(order), batch_size):
            rows = order[start_batch:start_batch + batch_size]
            inputs = [x[rows].to(device) for x in x_train]
            y = labels[rows].to(device)
            for optimizer in optimizers: optimizer.zero_grad(set_to_none=True)
            logits = [adapter.forward(model, x)[1] for adapter, model, x in zip(adapters, models, inputs)]
            outputs = [[z] + (heads[i](len(y)) if with_heads else []) for i, z in enumerate(logits)]
            loss = sum(mutual_losses(outputs, y, args.condition, cfg['temperature'], cfg['alpha'], cfg['lambda']))
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
            loss.backward()
            for optimizer in optimizers:
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 1.0)
                optimizer.step()
        for optimizer, role in zip(optimizers, ('big', 'small')):
            factor = 0.5 * (1 + np.cos(np.pi * (epoch + 1) / epochs))
            for group in optimizer.param_groups: group['lr'] = float(pair[role + '_lr']) * factor
        state = {'identity': identity, 'epoch': epoch + 1, 'models': [m.state_dict() for m in models],
                 'heads': [h.state_dict() for h in heads], 'optimizers': [o.state_dict() for o in optimizers],
                 'generator': generator.get_state(), 'python_rng': random.getstate(),
                 'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state(device)}
        atomic_save(state_path, state)
        print(f'{args.pair} {args.condition} subject={args.subject} seed={args.seed} epoch={epoch+1}/{epochs} loss={loss.item():.5f}', flush=True)
        if args.stop_after and epoch + 1 >= args.stop_after: return
    for head in heads: head.close()
    predictions = []
    for adapter, model in zip(adapters, models):
        model.eval()
        test = adapter.preprocess(split['x_test_raw'])
        parts = []
        with torch.no_grad():
            for pos in range(0, len(test), batch_size):
                parts.append(adapter.forward(model, test[pos:pos + batch_size].to(device))[1].argmax(1).cpu())
        predictions.append(torch.cat(parts).numpy())
    temp = root / 'predictions.tmp.npz'
    np.savez_compressed(temp, uids=np.asarray(test_uid, dtype=str), y=split['y_test'],
                        big=predictions[0], small=predictions[1])
    os.replace(temp, root / 'predictions.npz')
    result = {'identity': identity, 'train_count': len(train_uid), 'test_count': len(test_uid),
              'big_accuracy': float(np.mean(predictions[0] == split['y_test']) * 100),
              'small_accuracy': float(np.mean(predictions[1] == split['y_test']) * 100)}
    temp = root / 'evaluation.tmp.json'
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    os.replace(temp, result_path)
    state_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
