"""Train one matched few-shot DML-DS + paper-path HFD unit."""
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
from collab.hfd_cross_all import CrossOperatorHFDAll
from data.loso_source import load_dataset_subjects, sha256_file
from experiments.bidir.run_dml_ds_cross_hfd_fewshot import atomic_json, atomic_save, state_digest, uid_hash
from experiments.distill.run_fewshot_structure_kd import _prepare_mirepnet_input, _split_subject
from experiments.finetune.labram_alexmi_broadband import load_alexmi_broadband
from experiments.finetune.run_neurogpt_loso import _adapter_for, _load_spec, DEFAULT_CONFIG as NEURO_CONFIG
from experiments.storage import require_external_output
from models import get_adapter

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/experiments/fewshot_dml_ds_cross_hfd_all.yaml'


def baseline_dir(dataset, pair, subject, seed, base):
    if pair in base['reuse_pairs']:
        root = Path('/data1/llx/BigSmallcollab/results/bidir')
        root = (root / 'dcm_fewshot_0014_v1/runs' if dataset == 'BNCI2014001-4'
                else root / 'dcm_fewshot_all_v1/runs' / dataset)
    else:
        root = Path(base['output_root']) / 'runs' / dataset
    return root / pair / 'dml_ds' / f'subject_{subject}' / f'seed_{seed}'


def implementation_hashes():
    names = [
        'collab/hfd_cross_all.py', 'collab/hfd_cross_operator.py', 'collab/dcm.py',
        'collab/dml_ds_heads.py', 'config.py', 'data/split.py', 'data/loso_source.py',
        'data/preproc.py', 'data/channels.py', 'models/base.py', 'models/__init__.py',
        'experiments/bidir/run_dml_ds_cross_hfd_all.py',
        'experiments/bidir/run_dml_ds_cross_hfd_fewshot.py',
        'experiments/bidir/run_dml_ds_fewshot_pairs.py',
        'experiments/bidir/run_dcm_fewshot.py', 'experiments/bidir/run_dcm_fewshot_all.py',
        'experiments/distill/run_fewshot_structure_kd.py',
        'experiments/finetune/labram_alexmi_broadband.py', 'experiments/finetune/run_neurogpt_loso.py',
        'configs/experiments/fewshot_dml_ds_pairs.yaml', 'configs/experiments/fewshot_dcm.yaml',
        'configs/experiments/fewshot_dcm_all.yaml',
        'models/cbramod/criss_cross_transformer.py',
    ]
    for dataset in ('BNCI2014004', 'BNCI2014001', 'BNCI2014001-4', 'BNCI2015001', 'AlexMI'):
        names.append(f'configs/datasets/{dataset}.yaml')
    for model in ('mirepnet', 'cbramod', 'neurogpt', 'ifnet', 'eegnet', 'adfcnn'):
        module = 'mlm.py' if model == 'mirepnet' else ('residual_eegnet.py' if model == 'eegnet' else f'{model}.py')
        names += [f'configs/models/{model}.yaml', f'models/{model}/adapter.py', f'models/{model}/{module}']
    return {name: sha256_file(ROOT / name) for name in names}


def source_and_split(dataset, subject, seed, base):
    if dataset == 'AlexMI':
        from mne.filter import resample
        source = load_alexmi_broadband()
        x = resample(source.x, up=125, down=256, axis=2)[..., :750].astype(np.float32)
        source.x = np.concatenate([x, x[..., :250]], axis=2)
    else:
        source = load_dataset_subjects(dataset, [subject])
    split = _split_subject(source, dataset, subject, seed,
                           base['calibration_fraction'], base['test_fraction'])
    train, test = split['trial_uid_cal'].tolist(), split['trial_uid_test'].tolist()
    if len(set(train)) != len(train) or len(set(test)) != len(test) or set(train) & set(test):
        raise ValueError('Calibration and test trial UIDs are not unique and disjoint')
    return source, split


def check_reused_training_config(dataset, pair, base):
    path = ROOT / ('configs/experiments/fewshot_dcm.yaml' if dataset == 'BNCI2014001-4'
                   else 'configs/experiments/fewshot_dcm_all.yaml')
    old = yaml.safe_load(path.read_text())
    for key in ('calibration_fraction', 'test_fraction', 'temperature', 'alpha', 'lambda'):
        if old[key] != base[key]:
            raise ValueError(f'Reused DML-DS baseline differs in {key}: {path}')
    big_name, small_name = pair.split('_')
    b, s, actual = base['teacher_training'][big_name], base['student_training'][small_name], old['pairs'][pair]
    expected = {'epochs': b['epochs'], 'batch_size': b['batch_size'], 'big_lr': b['lr'],
                'big_weight_decay': b['weight_decay'], 'small_lr': s['lr'], 'small_weight_decay': s['weight_decay']}
    if actual != expected or b['optimizer'] != 'adamw' or s['optimizer'] != 'adamw':
        raise ValueError(f'Reused DML-DS optimizer differs: {path} {pair}')


def validate_baseline(dataset, pair, subject, seed, source, split, base, ea_sha=None):
    folder = baseline_dir(dataset, pair, subject, seed, base)
    result_path, pred_path = folder / 'evaluation.json', folder / 'predictions.npz'
    if not result_path.is_file() or not pred_path.is_file():
        raise FileNotFoundError(f'Matching DML-DS baseline not complete: {folder}')
    result = json.loads(result_path.read_text())
    ident = result['identity']
    big_name, small_name = pair.split('_')
    bt, st = base['teacher_training'][big_name], base['student_training'][small_name]
    expected = (dataset, pair, 'dml_ds', subject, seed, 'fewshot_stratified_random_30_70', False,
                bt['epochs'], bt['batch_size'])
    actual = tuple(ident[k] for k in ('dataset', 'pair', 'condition', 'subject', 'seed',
                                       'protocol', 'smoke', 'epochs', 'batch_size'))
    if (actual != expected or ident['source_manifest_sha256'] != source.source_manifest_sha256 or
            ident['pretrained_sha256'] != sha256_file(config.weight_path(big_name)) or
            ident['train_uid_sha256'] != uid_hash(split['trial_uid_cal'].tolist()) or
            ident['test_uid_sha256'] != uid_hash(split['trial_uid_test'].tolist()) or
            ident.get('channels', split['x_cal_raw'].shape[1]) != split['x_cal_raw'].shape[1] or
            ident.get('num_classes', config.load_dataset_config(dataset)['num_classes']) !=
            config.load_dataset_config(dataset)['num_classes']):
        raise ValueError(f'DML-DS baseline identity or split differs: {folder}')
    if pair in base['reuse_pairs']:
        check_reused_training_config(dataset, pair, base)
    elif ident.get('big_training') != bt or ident.get('small_training') != st:
        raise ValueError(f'DML-DS optimizer or schedule differs: {folder}')
    if big_name == 'mirepnet' and ident.get('mirepnet_ea_sha256') != ea_sha:
        raise ValueError(f'MIRepNet alignment differs from baseline: {folder}')
    if big_name == 'neurogpt' and int(split['x_cal_raw'].shape[1]) == 22:
        if ident.get('channel_matrix_sha256') != sha256_file('/data1/llx/pre_weight/neurogpt/tMatrix_value.npy'):
            raise ValueError(f'NeuroGPT channel matrix differs from baseline: {folder}')
    for name, digest in ident['code_sha256'].items():
        path = ROOT / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f'DML-DS baseline code differs: {name}')
    with np.load(pred_path, allow_pickle=False) as pred:
        uids, truth = pred['uids'].tolist(), pred['y']
        if (len(uids) != result['test_count'] or len(set(uids)) != len(uids) or
                uid_hash(uids) != ident['test_uid_sha256'] or not np.array_equal(truth, split['y_test'])):
            raise ValueError(f'DML-DS baseline predictions differ: {folder}')
        for role in ('big', 'small'):
            if not np.isclose(np.mean(pred[role] == truth) * 100, result[role + '_accuracy'], atol=1e-9, rtol=0):
                raise ValueError(f'DML-DS baseline accuracy differs from predictions: {folder}')
    return {'result_sha256': sha256_file(result_path), 'predictions_sha256': sha256_file(pred_path),
            'result_path': str(result_path)}


def make_models(dataset, pair, channels, classes, base, seed, device):
    big_name, small_name = pair.split('_')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if big_name == 'neurogpt':
        exp, spec, _ = _load_spec(NEURO_CONFIG, dataset)
        big_adapter, _ = _adapter_for(dataset, channels, device, exp, spec)
    else:
        big_cfg = config.load_model_config(big_name, dataset, 'fewshot')
        big_cfg.update(in_channels=45 if big_name == 'mirepnet' else channels, samples=1000,
                       skip_preprocess=big_name == 'mirepnet')
        big_adapter = get_adapter(big_name, device=device, dataset_name=dataset, **big_cfg)
    small_cfg = config.load_model_config(small_name, dataset, 'fewshot')
    small_cfg.update(in_channels=channels, samples=1000)
    small_adapter = get_adapter(small_name, device=device, dataset_name=dataset, **small_cfg)
    adapters = [big_adapter, small_adapter]
    models = [adapter.build(classes) for adapter in adapters]
    heads = [build_heads(model, name, classes).to(device)
             for model, name in zip(models, (big_name, small_name))]
    hfd = CrossOperatorHFDAll(big_name, small_name, models[0], models[1], channels, seed).to(device)
    optimizers = []
    for model, head, projector, training in zip(
            models, heads, (hfd.big_to_small, hfd.small_to_big),
            (base['teacher_training'][big_name], base['student_training'][small_name])):
        groups = [{'params': list(model.parameters()) + list(head.parameters())},
                  {'params': list(projector.parameters())}]
        cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[training['optimizer']]
        optimizers.append(cls(groups, lr=float(training['lr']), weight_decay=float(training['weight_decay'])))
    return adapters, models, heads, hfd, optimizers


def run(cfg, base, args, out):
    dataset, pair, subject, seed = args.dataset, args.pair, args.subject, args.seed
    big_name, small_name = pair.split('_')
    source, split = source_and_split(dataset, subject, seed, base)
    channels = int(split['x_cal_raw'].shape[1])
    classes = int(config.load_dataset_config(dataset)['num_classes'])
    adapters, models, heads, hfd, optimizers = make_models(
        dataset, pair, channels, classes, base, seed, args.device)
    try:
        if big_name == 'mirepnet':
            big_cal, big_test, ea_sha = _prepare_mirepnet_input(
                dataset, split['x_cal_raw'], split['x_test_raw'], adapters[0])
        else:
            big_cal, big_test, ea_sha = split['x_cal_raw'], split['x_test_raw'], None
        baseline = validate_baseline(dataset, pair, subject, seed, source, split, base, ea_sha)
        inputs = [adapters[0].preprocess(big_cal), adapters[1].preprocess(split['x_cal_raw'])]
        labels = torch.as_tensor(split['y_cal'], dtype=torch.long)
        train_uid, test_uid = split['trial_uid_cal'].tolist(), split['trial_uid_test'].tolist()
        bt, st = base['teacher_training'][big_name], base['student_training'][small_name]
        epochs = 2 if args.smoke else int(bt['epochs'])
        identity = {'experiment_id': cfg['experiment_id'], 'dataset': dataset, 'pair': pair,
                    'condition': cfg['condition'], 'subject': subject, 'seed': seed,
                    'protocol': 'fewshot_stratified_random_30_70', 'smoke': args.smoke,
                    'hfd_method': 'paper_eq2_native_peer_operators_eeg_adaptation',
                    'epochs': epochs, 'batch_size': int(bt['batch_size']), 'final_epoch_only': True,
                    'channels': channels, 'num_classes': classes,
                    'source_manifest_sha256': source.source_manifest_sha256,
                    'train_uid_sha256': uid_hash(train_uid), 'test_uid_sha256': uid_hash(test_uid),
                    'train_count': len(train_uid), 'test_count': len(test_uid),
                    'pretrained_sha256': sha256_file(config.weight_path(big_name)),
                    'mirepnet_ea_sha256': ea_sha, 'big_training': bt, 'small_training': st,
                    'baseline_result_sha256': baseline['result_sha256'],
                    'baseline_predictions_sha256': baseline['predictions_sha256'],
                    'baseline_result_path': baseline['result_path'],
                    'config_sha256': sha256_file(args.config), 'implementation_sha256': implementation_hashes(),
                    'initial_model_sha256': [state_digest(model) for model in models],
                    'initial_head_sha256': [state_digest(head) for head in heads],
                    'input_preprocessing': ('AlexMI broadband 512Hz to 250Hz, first 750 samples then repeat first 250 samples'
                                            if dataset == 'AlexMI' else 'verified broadband canonical 4s at 250Hz')}
        state_path, result_path = out / 'training_state.pt', out / 'evaluation.json'
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            pred, history_path = out / 'predictions.npz', out / 'training.json'
            if (result['identity'] != identity or not pred.is_file() or not history_path.is_file() or
                    result['predictions_sha256'] != sha256_file(pred) or
                    json.loads(history_path.read_text()).get('identity') != identity):
                raise ValueError(f'Completed experiment identity differs: {out}')
            state_path.unlink(missing_ok=True)
            return
        generator = torch.Generator().manual_seed(seed + 100003)
        history, start = [], 0
        if state_path.is_file():
            saved = torch.load(state_path, map_location='cpu', weights_only=False)
            if saved['identity'] != identity:
                raise ValueError(f'Cannot resume with different experiment identity: {out}')
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
            sums, batches = np.zeros(4, dtype=np.float64), 0
            for pos in range(0, len(order), bt['batch_size']):
                rows = order[pos:pos + bt['batch_size']]
                batch_inputs = [x[rows].to(args.device) for x in inputs]
                target = labels[rows].to(args.device)
                for opt in optimizers: opt.zero_grad(set_to_none=True)
                hfd.clear()
                logits = [adapter.forward(model, x)[1]
                          for adapter, model, x in zip(adapters, models, batch_inputs)]
                outputs = [[z] + heads[j](len(target)) for j, z in enumerate(logits)]
                ds_loss = sum(mutual_losses(outputs, target, 'dml_ds', base['temperature'],
                                            base['alpha'], base['lambda']))
                hfd_losses = (torch.zeros((), device=args.device), torch.zeros((), device=args.device))
                if ramp:
                    native = dict(hfd.features)
                    hfd_losses = hfd.losses()
                    if args.smoke and epoch == 0 and batches == 0:
                        own = ([native['big_first']], [native[f'small_first_{i}'] for i in range(len(hfd.small_dims))])
                        for j, projector in enumerate((hfd.big_to_small, hfd.small_to_big)):
                            p = list(projector.parameters())
                            grads = torch.autograd.grad(hfd_losses[j], own[j] + own[1-j] + p,
                                                        allow_unused=True, retain_graph=True)
                            own_grads = grads[:len(own[j])]
                            peer_grads = grads[len(own[j]):len(own[j]) + len(own[1-j])]
                            proj_grads = grads[len(own[j]) + len(own[1-j]):]
                            if (any(g is None or not torch.isfinite(g).all() or not g.abs().sum() > 0 for g in own_grads) or
                                    any(g is not None for g in peer_grads) or
                                    any(g is None or not torch.isfinite(g).all() or not g.abs().sum() > 0 for g in proj_grads)):
                                raise RuntimeError('HFD directional gradient isolation failed')
                        print('HFD directional gradient isolation passed', flush=True)
                loss = ds_loss + ramp * float(cfg['hfd_weight']) * sum(hfd_losses)
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
                loss.backward()
                for opt in optimizers:
                    torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 1.0)
                    opt.step()
                sums += [ds_loss.item(), hfd_losses[0].item(), hfd_losses[1].item(), loss.item()]
                batches += 1
                hfd.clear()
                if args.smoke and batches >= 2: break
            factor = 0.5 * (1 + np.cos(np.pi * (epoch + 1) / epochs))
            for opt, training in zip(optimizers, (bt, st)):
                for group in opt.param_groups: group['lr'] = float(training['lr']) * factor
            history.append({'epoch': epoch + 1, 'ramp': ramp, 'mean_losses': (sums / batches).tolist()})
            stop = bool(args.stop_file and args.stop_file.exists())
            stop_after = bool(args.stop_after and epoch + 1 >= args.stop_after)
            if stop or stop_after or epoch + 1 == epochs or (epoch + 1) % cfg['checkpoint_interval'] == 0:
                atomic_save(state_path, {'identity': identity, 'epoch': epoch + 1,
                    'models': [model.state_dict() for model in models], 'heads': [head.state_dict() for head in heads],
                    'hfd': hfd.state_dict(), 'optimizers': [opt.state_dict() for opt in optimizers],
                    'history': history, 'generator': generator.get_state(), 'python_rng': random.getstate(),
                    'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state(args.device) if args.device.type == 'cuda' else None})
            print(f'{dataset} {pair} subject={subject} seed={seed} epoch={epoch+1}/{epochs} '
                  f'ds={sums[0]/batches:.5f} hfd_big={sums[1]/batches:.5f} '
                  f'hfd_small={sums[2]/batches:.5f} ramp={ramp:.2f}', flush=True)
            if args.progress:
                atomic_json(args.progress, {'status': 'running', 'epoch': epoch + 1, 'epochs': epochs,
                                             'dataset': dataset, 'pair': pair, 'subject': subject, 'seed': seed})
            if stop: raise SystemExit(75)
            if stop_after: return
        model_hashes, head_hashes, hfd_hash = ([state_digest(model) for model in models],
                                                [state_digest(head) for head in heads], state_digest(hfd))
        for head in heads: head.close()
        hfd.close()
        raw = (big_cal, split['x_cal_raw']) if args.smoke else (big_test, split['x_test_raw'])
        truth = np.asarray(split['y_cal'] if args.smoke else split['y_test'], dtype=np.int64)
        uids = np.asarray(train_uid if args.smoke else test_uid, dtype=str)
        predictions = []
        for adapter, model, x in zip(adapters, models, raw):
            model.eval()
            test = adapter.preprocess(x)
            parts = []
            with torch.no_grad():
                for pos in range(0, len(test), bt['batch_size']):
                    parts.append(adapter.forward(model, test[pos:pos + bt['batch_size']].to(args.device))[1].argmax(1).cpu())
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
                  'model_sha256': model_hashes, 'head_sha256': head_hashes, 'hfd_sha256': hfd_hash}
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
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--pair', required=True)
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
    # run_neurogpt_loso sets inter-op threads during import; setting them a
    # second time aborts some PyTorch builds even when the value is unchanged.
    if torch.get_num_interop_threads() != 1:
        raise RuntimeError('Training requires one PyTorch inter-op thread')
    args.config = args.config.resolve()
    cfg = yaml.safe_load(args.config.read_text())
    base = yaml.safe_load((ROOT / cfg['baseline_config']).read_text())
    if args.expected_config_sha256 and sha256_file(args.config) != args.expected_config_sha256:
        raise ValueError('Experiment configuration changed')
    digest = hashlib.sha256(json.dumps(implementation_hashes(), sort_keys=True).encode()).hexdigest()
    if args.expected_implementation_sha256 and digest != args.expected_implementation_sha256:
        raise ValueError('Experiment implementation changed')
    if (args.dataset not in cfg['datasets'] or args.seed not in cfg['seeds'] or
            not 0 <= args.subject < cfg['datasets'][args.dataset] or
            args.pair not in {f'{a}_{b}' for a in cfg['teachers'] for b in cfg['students']} or
            cfg['condition'] != 'dml_ds_cross_hfd' or base['calibration_fraction'] != 0.3 or
            base['test_fraction'] != 0.7 or cfg['checkpoint_interval'] < 1):
        raise ValueError('Invalid experiment identity or protocol')
    if (args.cpu or args.stop_after) and not args.smoke:
        parser.error('--cpu and --stop-after require --smoke')
    if Path(args.smoke_id).name != args.smoke_id:
        parser.error('invalid smoke id')
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    args.device = torch.device('cpu' if args.cpu else f'cuda:{args.gpu}')
    if args.device.type == 'cpu': torch.backends.mkldnn.enabled = False
    else: torch.cuda.set_device(args.device)
    root = Path(cfg['output_root']) / ('smoke' if args.smoke else 'runs')
    if args.smoke: root /= args.smoke_id
    out = require_external_output(root / args.dataset / args.pair / f'subject_{args.subject}' / f'seed_{args.seed}')
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(cfg, base, args, out)
    if args.progress and not args.stop_after:
        atomic_json(args.progress, {'status': 'completed', 'dataset': args.dataset, 'pair': args.pair,
                                    'subject': args.subject, 'seed': args.seed})


if __name__ == '__main__':
    main()
