"""Independent CE baselines and DML-DS with MIRepNet's 8–30 Hz input."""
import os
os.environ['BIGSMALLCOLLAB_CPU_THREADS'] = '1'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import config
from collab.dcm import mutual_losses
from collab.dml_ds_heads import build_heads
from data.loso_source import load_dataset_subjects, sha256_file
from data.preproc import bandpass
from experiments.distill.run_fewshot_structure_kd import _split_subject, _prepare_mirepnet_input
from experiments.finetune.labram_alexmi_broadband import load_alexmi_broadband
from experiments.storage import require_external_output
from models import get_adapter

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/experiments/fewshot_dml_ds_mirepnet_8_30.yaml'
STUDENT_OFFSET = {'ifnet': 1009, 'eegnet': 2017, 'adfcnn': 3019}


def uid_hash(uids):
    return hashlib.sha256(json.dumps(list(uids), ensure_ascii=False).encode()).hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        data = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(data.dtype).encode())
        digest.update(str(tuple(data.shape)).encode()); digest.update(data.numpy().tobytes())
    return digest.hexdigest()


class ModelRNG:
    """Independent and resumable dropout streams for each backbone and head."""
    def __init__(self, seed, device):
        self.device = device
        self.cpu = torch.Generator(device='cpu').manual_seed(seed).get_state()
        self.cuda = (torch.Generator(device=device).manual_seed(seed).get_state()
                     if device.type == 'cuda' else None)

    @contextmanager
    def use(self):
        devices = [self.device.index] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self.cpu)
            if self.cuda is not None: torch.cuda.set_rng_state(self.cuda, self.device)
            try:
                yield
            finally:
                self.cpu = torch.get_rng_state()
                if self.cuda is not None: self.cuda = torch.cuda.get_rng_state(self.device)

    def save(self): return {'cpu': self.cpu, 'cuda': self.cuda}
    def load(self, state): self.cpu, self.cuda = state['cpu'], state['cuda']


def model_paths(name):
    module = {'mirepnet': 'mlm.py', 'eegnet': 'residual_eegnet.py'}.get(name, f'{name}.py')
    return [ROOT / f'configs/models/{name}.yaml', ROOT / f'models/{name}/adapter.py',
            ROOT / f'models/{name}/{module}']


def code_hashes(dataset, names):
    paths = [Path(__file__), CONFIG, ROOT / 'collab/dcm.py', ROOT / 'collab/dml_ds_heads.py',
             ROOT / 'config.py', ROOT / 'data/split.py', ROOT / 'data/loso_source.py',
             ROOT / 'data/preproc.py', ROOT / 'data/channels.py', ROOT / 'models/base.py',
             ROOT / 'experiments/distill/run_fewshot_structure_kd.py',
             ROOT / 'experiments/finetune/labram_alexmi_broadband.py',
             ROOT / f'configs/datasets/{dataset}.yaml']
    for name in names: paths.extend(model_paths(name))
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def data_for(dataset, subject, seed, cfg):
    if dataset == 'AlexMI':
        from mne.filter import resample
        source = load_alexmi_broadband()
        x = resample(source.x, up=125, down=256, axis=2)[..., :750].astype(np.float32)
        source.x = np.concatenate([x, x[..., :250]], axis=2)
    else:
        source = load_dataset_subjects(dataset, [subject])
    split = _split_subject(source, dataset, subject, seed,
                           cfg['calibration_fraction'], cfg['test_fraction'])
    train, test = split['trial_uid_cal'].tolist(), split['trial_uid_test'].tolist()
    if len(set(train)) != len(train) or len(set(test)) != len(test) or set(train) & set(test):
        raise ValueError('Duplicate or overlapping calibration/test trials')
    return source, split, train, test


def prepare_teacher(dataset, split, adapter):
    cal = bandpass(np.asarray(split['x_cal_raw'], dtype=np.float64), 250, 8, 30).astype(np.float32)
    test = bandpass(np.asarray(split['x_test_raw'], dtype=np.float64), 250, 8, 30).astype(np.float32)
    return _prepare_mirepnet_input(dataset, cal, test, adapter)


def atomic_torch(path, value):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        torch.save(value, tmp); os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def run(args, cfg, folder):
    dataset, seed = args.dataset, args.seed
    device = torch.device('cpu' if args.cpu else f'cuda:{args.gpu}')
    if device.type == 'cuda': torch.cuda.set_device(device)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    random.seed(seed); np.random.seed(seed)
    source, split, train_uid, test_uid = data_for(dataset, args.subject, seed, cfg)
    channels = int(split['x_cal_raw'].shape[1])
    classes = int(config.load_dataset_config(dataset)['num_classes'])
    names = ['mirepnet'] if args.kind == 'teacher' else [args.student] if args.kind == 'student' else ['mirepnet', args.student]
    model_seeds = [seed if name == 'mirepnet' else seed + STUDENT_OFFSET[name] for name in names]
    adapters, models, model_rng = [], [], []
    for name, model_seed in zip(names, model_seeds):
        model_cfg = config.load_model_config(name, dataset, 'fewshot')
        model_cfg.update(in_channels=45 if name == 'mirepnet' else channels,
                         samples=1000, skip_preprocess=name == 'mirepnet')
        adapter = get_adapter(name, device=device, dataset_name=dataset, **model_cfg)
        torch.manual_seed(model_seed); torch.cuda.manual_seed_all(model_seed)
        model = adapter.build(classes)
        adapters.append(adapter); models.append(model); model_rng.append(ModelRNG(model_seed + 40001, device))
    initial_hashes = {name: state_hash(model) for name, model in zip(names, models)}
    heads, head_rng = [], []
    if args.kind == 'dml_ds':
        for name, model, model_seed in zip(names, models, model_seeds):
            torch.manual_seed(model_seed + 50021); torch.cuda.manual_seed_all(model_seed + 50021)
            heads.append(build_heads(model, name, classes).to(device))
            head_rng.append(ModelRNG(model_seed + 60013, device))
    opts = []
    for i, (name, model) in enumerate(zip(names, models)):
        training = cfg['teacher_training'] if name == 'mirepnet' else cfg['student_training'][name]
        params = list(model.parameters()) + (list(heads[i].parameters()) if heads else [])
        opt = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[training['optimizer']]
        opts.append(opt(params, lr=float(training['lr']), weight_decay=float(training['weight_decay'])))
    inputs, tests = [], []
    ea_sha = None
    for name, adapter in zip(names, adapters):
        if name == 'mirepnet':
            cal, test, ea_sha = prepare_teacher(dataset, split, adapter)
        else: cal, test = split['x_cal_raw'], split['x_test_raw']
        inputs.append(adapter.preprocess(cal)); tests.append(adapter.preprocess(test))
    epochs = 2 if args.smoke else int(cfg['epochs'])
    identity = {'experiment': cfg['experiment_id'], 'dataset': dataset, 'kind': args.kind,
        'student': args.student if args.kind != 'teacher' else None, 'subject': args.subject,
        'seed': seed, 'smoke': args.smoke, 'protocol': 'fewshot_stratified_random_30_70',
        'epochs': epochs, 'batch_size': cfg['batch_size'], 'initial_hashes': initial_hashes,
        'source_manifest_sha256': source.source_manifest_sha256, 'train_uid_sha256': uid_hash(train_uid),
        'test_uid_sha256': uid_hash(test_uid), 'mirepnet_ea_sha256': ea_sha,
        'mirepnet_preprocessing': 'broadband_250Hz_8_30Hz_order4_filtfilt_calibration_EA_pad45' if ea_sha else None,
        'pretrained_sha256': sha256_file(config.weight_path('mirepnet')) if ea_sha else None,
        'code_sha256': code_hashes(dataset, names)}
    result_path, state_path = folder / 'evaluation.json', folder / 'training_state.pt'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['identity'] != identity or not (folder / 'predictions.npz').is_file():
            raise RuntimeError('Completed result identity mismatch')
        state_path.unlink(missing_ok=True); return
    labels = torch.as_tensor(split['y_cal'], dtype=torch.long)
    generator = torch.Generator().manual_seed(seed + 100003)
    start = 0
    if state_path.exists():
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        if state['identity'] != identity: raise RuntimeError('Resume identity mismatch')
        for obj, saved in zip(models, state['models']): obj.load_state_dict(saved)
        for obj, saved in zip(heads, state['heads']): obj.load_state_dict(saved)
        for obj, saved in zip(opts, state['optimizers']): obj.load_state_dict(saved)
        for obj, saved in zip(model_rng, state['model_rng']): obj.load(saved)
        for obj, saved in zip(head_rng, state['head_rng']): obj.load(saved)
        generator.set_state(state['generator']); start = state['epoch']
    for epoch in range(start, epochs):
        for model in models: model.train()
        order = torch.randperm(len(labels), generator=generator)
        for pos in range(0, len(order), cfg['batch_size']):
            rows = order[pos:pos + cfg['batch_size']]; y = labels[rows].to(device)
            for opt in opts: opt.zero_grad(set_to_none=True)
            outputs = []
            for i, (adapter, model, x) in enumerate(zip(adapters, models, inputs)):
                with model_rng[i].use(): _, logits = adapter.forward(model, x[rows].to(device))
                if logits.shape != (len(y), classes): raise ValueError('Main logits shape mismatch')
                output = [logits]
                if heads:
                    with head_rng[i].use(): output.extend(heads[i](len(y)))
                    if len(output) != 3 or any(z.shape != logits.shape for z in output):
                        raise ValueError('Auxiliary logits shape mismatch')
                outputs.append(output)
            loss = (sum(mutual_losses(outputs, y, 'dml_ds', cfg['temperature'], cfg['alpha'], cfg['lambda']))
                    if heads else F.cross_entropy(outputs[0][0], y))
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            loss.backward()
            if epoch == 0 and pos == 0:
                for name, model in zip(names, models):
                    if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                               for p in model.parameters()):
                        raise RuntimeError(f'No finite nonzero backbone gradient for {name}')
                for head in heads:
                    if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                               for p in head.parameters()): raise RuntimeError('No auxiliary-head gradient')
            for opt in opts:
                torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 1.0); opt.step()
        for opt, name in zip(opts, names):
            training = cfg['teacher_training'] if name == 'mirepnet' else cfg['student_training'][name]
            for group in opt.param_groups: group['lr'] = float(training['lr']) * 0.5 * (1 + np.cos(np.pi * (epoch + 1) / epochs))
        atomic_torch(state_path, {'identity': identity, 'epoch': epoch+1, 'models': [m.state_dict() for m in models],
            'heads': [h.state_dict() for h in heads], 'optimizers': [o.state_dict() for o in opts],
            'model_rng': [r.save() for r in model_rng], 'head_rng': [r.save() for r in head_rng],
            'generator': generator.get_state()})
        print(f'{dataset} {args.kind} {args.student} subject={args.subject} seed={seed} epoch={epoch+1}/{epochs} loss={loss.item():.5f}', flush=True)
        if args.stop_after and epoch + 1 >= args.stop_after: return
    for head in heads: head.close()
    predictions = {}
    for name, adapter, model, x in zip(names, adapters, models, tests):
        model.eval(); parts = []
        with torch.no_grad():
            for pos in range(0, len(x), cfg['batch_size']):
                parts.append(adapter.forward(model, x[pos:pos+cfg['batch_size']].to(device))[1].argmax(1).cpu())
        predictions[name] = torch.cat(parts).numpy()
    tmp = folder / f'predictions.{os.getpid()}.tmp.npz'
    np.savez_compressed(tmp, uids=np.asarray(test_uid, dtype=str), y=split['y_test'], **predictions)
    os.replace(tmp, folder / 'predictions.npz')
    result = {'identity': identity, 'train_count': len(train_uid), 'test_count': len(test_uid),
              'accuracy_percent': {name: float(np.mean(p == split['y_test']) * 100) for name,p in predictions.items()}}
    tmp = folder / f'evaluation.{os.getpid()}.tmp.json'
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n'); os.replace(tmp, result_path)
    state_path.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True); p.add_argument('--kind', choices=('teacher','student','dml_ds'), required=True)
    p.add_argument('--student', choices=('ifnet','eegnet','adfcnn'))
    p.add_argument('--subject', type=int, required=True); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--gpu', type=int, required=True); p.add_argument('--cpu', action='store_true')
    p.add_argument('--smoke', action='store_true'); p.add_argument('--smoke-tag', default='default')
    p.add_argument('--stop-after', type=int)
    args = p.parse_args(); cfg = yaml.safe_load(CONFIG.read_text())
    if args.dataset not in cfg['datasets'] or args.seed not in cfg['seeds'] or not 0 <= args.subject < cfg['datasets'][args.dataset]: p.error('Outside experiment scope')
    if (args.kind == 'teacher') != (args.student is None): p.error('Teacher takes no student; other kinds require one')
    if (args.cpu or args.stop_after) and not args.smoke: p.error('CPU and stop-after are for smoke only')
    if not args.smoke_tag.isalnum(): p.error('Invalid smoke tag')
    folder = require_external_output(Path(cfg['output_root']) / ('smoke' if args.smoke else 'runs') /
              (args.smoke_tag if args.smoke else '') / args.dataset / args.kind / (args.student or 'mirepnet') /
              f'subject_{args.subject}' / f'seed_{args.seed}')
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); run(args, cfg, folder)


if __name__ == '__main__': main()
