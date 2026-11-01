"""Resumable stages for fixed teacher-to-student feature distillation.

Inputs are already preprocessed model tensors in CPU arrays. Each caller owns
the source/split/UID audit and puts its identity into ``fingerprint``. No stage
selects a checkpoint from held-out scores. Resume checkpoints live under the
external weights tree and are removed after their terminal state is saved.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from collab.mcc_diffkd import EpsilonPredictor, ddpm_schedule, ddim_denoise, minimum_class_confusion
from experiments.storage import require_external_output, resolve_local_file


def state_hash(value) -> str:
    state = value.state_dict() if isinstance(value, torch.nn.Module) else value
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(str(key).encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _file_hash(path) -> str:
    digest = hashlib.sha256()
    with Path(resolve_local_file(path)).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def seed_all(seed: int, device) -> None:
    device = torch.device(device)
    random.seed(int(seed))
    np.random.seed(int(seed) % 2**32)
    torch.manual_seed(int(seed))
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(int(seed))
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng(device) -> dict:
    device = torch.device(device)
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def restore_rng(state: dict, device) -> None:
    device = torch.device(device)
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if device.type == 'cuda' and state.get('torch_cuda') is not None:
        torch.cuda.set_rng_state(state['torch_cuda'].cpu(), device)


def atomic_json(path, value) -> Path:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)
    return path


def atomic_torch(path, value) -> Path:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(_cpu(value), temporary)
    os.replace(temporary, path)
    return path


def _csv(path, rows):
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_completion(results_dir, fingerprint: str) -> dict | None:
    """Read a completed stage without requiring weights or starting training."""
    path = require_external_output(Path(results_dir) / 'complete.json')
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if value.get('status') != 'complete' or value.get('fingerprint') != fingerprint:
        raise RuntimeError(f'completed stage identity differs: {path}')
    return value


def _paths(results_dir, weights_dir):
    results = require_external_output(results_dir)
    weights = require_external_output(weights_dir)
    results.mkdir(parents=True, exist_ok=True)
    weights.mkdir(parents=True, exist_ok=True)
    return results, weights / 'final.pt', weights / 'resume.pt'


def _forward(adapter, model, xb, forward_fn):
    native, logits = (forward_fn(adapter, model, xb) if forward_fn else adapter.forward(model, xb))
    if native.ndim != 2 or logits.ndim != 2 or native.shape[0] != logits.shape[0]:
        raise RuntimeError('native classifier feature and logits must both be batch vectors')
    return native, logits


def _batch(x, ids, device):
    if torch.is_tensor(x):
        return x[torch.as_tensor(ids, dtype=torch.long)].to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(x[ids]), dtype=torch.float32, device=device)


def _finite(value, context):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f'non-finite {context}')


def _load_final(model, final_path, manifest):
    if not final_path.is_file():
        raise FileNotFoundError(f'completed stage weights were pruned; refusing retraining: {final_path}')
    if manifest['checkpoint_sha256'] != _file_hash(final_path):
        raise RuntimeError(f'completed checkpoint hash differs: {final_path}')
    payload = torch.load(resolve_local_file(final_path), map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model'], strict=True)
    if state_hash(model) != manifest['model_state_sha256']:
        raise RuntimeError(f'completed tensor state hash differs: {final_path}')
    return model


def _finish(model, final_path, resume_path, results, fingerprint, history, elapsed, **metadata):
    model_state = state_hash(model)
    atomic_torch(final_path, {'fingerprint': fingerprint, 'model': model.state_dict(),
                              'model_state_sha256': model_state})
    manifest = {'status': 'complete', 'fingerprint': fingerprint,
        'checkpoint': str(final_path), 'checkpoint_sha256': _file_hash(final_path),
        'model_state_sha256': model_state, 'elapsed_seconds': float(elapsed),
        'history': str(results / 'history.csv'), **metadata}
    _csv(results / 'history.csv', history)
    atomic_json(results / 'complete.json', manifest)
    resume_path.unlink(missing_ok=True)
    return manifest


def extract(adapter, model, x, *, batch_size=64, device=None, forward_fn=None,
            cache_path=None, fingerprint=None, uids=None, labels=None) -> dict:
    """Extract native features/logits, optionally reusing a hash-checked NPZ cache.

    The model and RNG are restored after inference, including classifiers that
    renormalize their weights during eval. ``x`` must be preprocessed once.
    """
    device = torch.device(device or adapter.device)
    cache = require_external_output(cache_path) if cache_path is not None else None
    meta_path = cache.with_suffix('.json') if cache is not None else None
    if cache is not None and (cache.exists() or meta_path.exists()):
        if not (cache.is_file() and meta_path.is_file()):
            raise RuntimeError(f'incomplete extraction cache: {cache}')
        meta = json.loads(meta_path.read_text())
        if meta.get('fingerprint') != fingerprint or meta.get('file_sha256') != _file_hash(cache):
            raise RuntimeError(f'extraction cache identity/hash differs: {cache}')
        with np.load(resolve_local_file(cache), allow_pickle=False) as saved:
            arrays = {name: np.asarray(saved[name]) for name in saved.files}
        if uids is not None and not np.array_equal(arrays.get('trial_uid'), np.asarray(uids)):
            raise RuntimeError('extraction cache trial UID order differs')
        if labels is not None and not np.array_equal(arrays.get('y'), np.asarray(labels)):
            raise RuntimeError('extraction cache labels differ')
        return arrays
    original, rng, training = _cpu(model.state_dict()), capture_rng(device), model.training
    parts, logits = [], []
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(x), int(batch_size)):
                ids = np.arange(start, min(start + int(batch_size), len(x)))
                feat, out = _forward(adapter, model, _batch(x, ids, device), forward_fn)
                _finite(feat, 'extracted features'); _finite(out, 'extracted logits')
                parts.append(feat.float().cpu().numpy()); logits.append(out.float().cpu().numpy())
    finally:
        model.load_state_dict(original, strict=True)
        model.train(training)
        restore_rng(rng, device)
    arrays = {'features': np.concatenate(parts), 'logits': np.concatenate(logits)}
    if uids is not None:
        arrays['trial_uid'] = np.asarray(uids)
    if labels is not None:
        arrays['y'] = np.asarray(labels)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(cache.name + '.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, cache)
        atomic_json(meta_path, {'fingerprint': fingerprint, 'file_sha256': _file_hash(cache),
            'model_state_sha256': state_hash(original), 'n_trials': len(x),
            'feature_dimension': int(arrays['features'].shape[1])})
    return arrays


def adapt_teacher(adapter, model, x, y, cfg, *, results_dir, weights_dir,
                  fingerprint, seed, initial_state, mcc_weight=0.0, epochs=None,
                  schedule_seed=None, max_batches=None, forward_fn=None, stop_file=None):
    """Return ``(model,manifest)`` for full-backbone CE or CE+MCC adaptation."""
    results, final_path, resume_path = _paths(results_dir, weights_dir)
    done = read_completion(results, fingerprint)
    if done:
        _load_final(model, final_path, done)
        model.eval()
        return model, done
    device = torch.device(adapter.device)
    seed_all(seed, device)
    model.load_state_dict(initial_state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(True); parameter.grad = None
    model.eval()
    initial_hash = state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get('lr', 1e-4)),
        betas=tuple(cfg.get('betas', (.9, .999))), eps=float(cfg.get('eps', 1e-8)),
        weight_decay=float(cfg.get('weight_decay', 1e-6)))
    total_epochs = int(epochs if epochs is not None else cfg.get('epochs', 10))
    order_rng = np.random.default_rng(int(schedule_seed if schedule_seed is not None else seed))
    history, start_epoch, elapsed_before = [], 0, 0.0
    if resume_path.is_file():
        saved = torch.load(resolve_local_file(resume_path), map_location='cpu', weights_only=False)
        if saved['fingerprint'] != fingerprint or saved['initial_state_sha256'] != initial_hash:
            raise RuntimeError('teacher resume identity differs')
        model.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'], device); order_rng.bit_generator.state = saved['order_rng']
        history, start_epoch, elapsed_before = saved['history'], int(saved['epoch']), saved['elapsed_seconds']
        if len(history) != start_epoch or not 0 <= start_epoch <= total_epochs:
            raise RuntimeError('teacher resume epoch/history cursor differs')
    started = time.monotonic()
    for epoch in range(start_epoch, total_epochs):
        model.eval()
        order, totals, seen = order_rng.permutation(len(y)), np.zeros(3), 0
        for batch_index, start in enumerate(range(0, len(order), int(cfg.get('batch_size', 8)))):
            if max_batches is not None and batch_index >= int(max_batches): break
            ids = order[start:start + int(cfg.get('batch_size', 8))]
            labels = torch.as_tensor(np.asarray(y)[ids], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            _, logits = _forward(adapter, model, _batch(x, ids, device), forward_fn)
            ce = F.cross_entropy(logits, labels)
            mcc = logits.new_zeros(())
            if float(mcc_weight) != 0.0:
                mcc_cfg = cfg.get('mcc', {})
                mcc, _ = minimum_class_confusion(logits,
                    temperature=float(mcc_cfg.get('temperature', 2.0)),
                    entropy_epsilon=float(mcc_cfg.get('entropy_epsilon', 1e-5)),
                    row_epsilon=float(mcc_cfg.get('row_denominator_epsilon', 1e-8)))
            loss = ce + float(mcc_weight) * mcc
            _finite(loss, 'teacher loss'); loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get('gradient_clip_norm', 1.0)))
            _finite(grad, 'teacher gradient'); optimizer.step()
            totals += np.array([ce.item(), mcc.item(), loss.item()]) * len(ids); seen += len(ids)
        history.append({'epoch': epoch + 1, 'n_train': seen,
            'ce': float(totals[0] / seen), 'mcc': float(totals[1] / seen),
            'loss': float(totals[2] / seen), 'mcc_weight': float(mcc_weight)})
        elapsed = elapsed_before + time.monotonic() - started
        _csv(results / 'history.csv', history)
        atomic_torch(resume_path, {'fingerprint': fingerprint, 'epoch': epoch + 1,
            'initial_state_sha256': initial_hash, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'rng': capture_rng(device),
            'order_rng': order_rng.bit_generator.state, 'history': history, 'elapsed_seconds': elapsed})
        if stop_file is not None and Path(stop_file).exists():
            return None, {'status': 'paused', 'epoch': epoch + 1}
    elapsed = elapsed_before + time.monotonic() - started
    manifest = _finish(model, final_path, resume_path, results, fingerprint, history, elapsed,
        final_epoch=total_epochs, initial_state_sha256=initial_hash, mcc_weight=float(mcc_weight),
        seed=int(seed), schedule_seed=int(schedule_seed if schedule_seed is not None else seed),
        config=cfg, forward_mode='eval_with_autograd')
    return model, manifest


def fit_prior(features, cfg, *, results_dir, weights_dir, fingerprint, seed,
              device, initial_state=None, feedback_features=None, updates=None, stop_file=None):
    """Fit an unconditional vector prior; return ``(frozen_prior,manifest)``."""
    results, final_path, resume_path = _paths(results_dir, weights_dir)
    device = torch.device(device)
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('prior features must be finite [N,D] vectors')
    dimension = values.shape[1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        prior = EpsilonPredictor(dimension, int(cfg.get('hidden_dimensions', 128)),
                                 int(cfg.get('time_embedding_dimensions', 32)))
    if initial_state is not None: prior.load_state_dict(initial_state, strict=True)
    prior = prior.to(device)
    done = read_completion(results, fingerprint)
    if done:
        _load_final(prior, final_path, done)
        prior.eval()
        for parameter in prior.parameters(): parameter.requires_grad_(False); parameter.grad = None
        return prior, done
    initial_hash = state_hash(prior)
    optimizer = torch.optim.AdamW(prior.parameters(), lr=float(cfg.get('lr', 3e-4)),
        betas=tuple(cfg.get('betas', (.9, .999))), eps=float(cfg.get('eps', 1e-8)),
        weight_decay=float(cfg.get('weight_decay', 1e-4)))
    _, alpha, sigma = ddpm_schedule(int(cfg.get('steps', 1000)),
        float(cfg.get('beta_start', 1e-4)), float(cfg.get('beta_end', .02)), device)
    generator = torch.Generator(device=device).manual_seed(int(seed) + 1)
    train = torch.as_tensor(values, dtype=torch.float32, device=device)
    batch_size = min(int(cfg.get('batch_size_cap', 128)), len(values))
    total_updates = int(updates if updates is not None else cfg.get('training_updates', 5000))
    history, start_update, elapsed_before = [], 0, 0.0
    fixed = {}
    for part, data in (('train', values), ('feedback', feedback_features)):
        if data is None: continue
        data = np.asarray(data, dtype=np.float32)
        rng = np.random.default_rng(int(seed) + 9100 + (part == 'feedback'))
        fixed[part] = []
        for low, high in ((1,250),(251,500),(501,750),(751,1000)):
            count = min(64, len(data)); ids = rng.choice(len(data), count, replace=False)
            fixed[part].append((low, high, data[ids], rng.integers(low, high+1, count),
                                rng.standard_normal((count, dimension)).astype(np.float32)))
    if resume_path.is_file():
        saved = torch.load(resolve_local_file(resume_path), map_location='cpu', weights_only=False)
        if saved['fingerprint'] != fingerprint or saved['initial_state_sha256'] != initial_hash:
            raise RuntimeError('prior resume identity differs')
        prior.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator_state'].cpu()); restore_rng(saved['rng'], device)
        history, start_update, elapsed_before = saved['history'], int(saved['update']), saved['elapsed_seconds']
        if (not 0 <= start_update <= total_updates
                or history and int(history[-1]['update']) != start_update):
            raise RuntimeError('prior resume update/history cursor differs')
    started = time.monotonic()
    for update in range(start_update, total_updates + 1):
        if update > start_update:
            ids = torch.randint(len(train), (batch_size,), device=device, generator=generator)
            times = torch.randint(1, len(alpha), (batch_size,), device=device, generator=generator)
            eps = torch.randn((batch_size, dimension), device=device, generator=generator)
            prediction = prior(alpha[times,None]*train[ids] + sigma[times,None]*eps, times)
            loss = F.mse_loss(prediction, eps); _finite(loss, 'prior loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(prior.parameters(), float(cfg.get('gradient_clip_norm',1.0)))
            _finite(grad, 'prior gradient'); optimizer.step()
        if (update == 0 or update == total_updates or update % int(cfg.get('diagnostic_interval',500)) == 0) and (not history or history[-1]['update'] != update):
            row = {'update': update}
            with torch.no_grad():
                for part, bins in fixed.items():
                    for low, high, clean_np, times_np, eps_np in bins:
                        clean = torch.as_tensor(clean_np, device=device)
                        times = torch.as_tensor(times_np, dtype=torch.long, device=device)
                        eps = torch.as_tensor(eps_np, device=device)
                        pred = prior(alpha[times,None]*clean + sigma[times,None]*eps, times)
                        row[f'{part}_{low}_{high}_noise_mse'] = float(F.mse_loss(pred,eps).cpu())
                        row[f'{part}_{low}_{high}_zero_mse'] = float(eps.square().mean().cpu())
            history.append(row); _csv(results / 'history.csv',history)
            elapsed = elapsed_before + time.monotonic() - started
            atomic_torch(resume_path, {'fingerprint': fingerprint, 'update': update,
                'initial_state_sha256': initial_hash, 'model': prior.state_dict(),
                'optimizer': optimizer.state_dict(), 'rng': capture_rng(device),
                'generator_state': generator.get_state(), 'history': history,'elapsed_seconds':elapsed})
            if stop_file is not None and Path(stop_file).exists(): return None, {'status':'paused','update':update}
    elapsed = elapsed_before + time.monotonic() - started
    manifest = _finish(prior,final_path,resume_path,results,fingerprint,history,elapsed,
        final_update=total_updates,feature_dimension=int(dimension),seed=int(seed),
        initial_state_sha256=initial_hash,config=cfg)
    prior.eval()
    for parameter in prior.parameters(): parameter.requires_grad_(False); parameter.grad = None
    return prior,manifest


def train_student(adapter, model, x, y, cfg, *, results_dir, weights_dir, fingerprint,
                  seed, initial_state, teacher_targets=None, prior=None, feature_weight=.5,
                  epochs=None, schedule_seed=None, noise_seed=None, max_batches=None,
                  forward_fn=None, diffusion_cfg=None, stop_file=None):
    """Train CE, paired FEATURE, or differentiable frozen-prior DIFFKD.

    All five conditions must receive the same initial_state, seed, schedule_seed
    and epoch budget. The separate noise generator leaves Dropout RNG paired.
    A completed manifest is returned even when final student weights are pruned.
    """
    results, final_path, resume_path = _paths(results_dir, weights_dir)
    done = read_completion(results,fingerprint)
    if done:
        if final_path.is_file(): _load_final(model,final_path,done)
        return done
    device = torch.device(adapter.device)
    seed_all(seed,device); model.load_state_dict(initial_state,strict=True)
    initial_hash = state_hash(model)
    for parameter in model.parameters(): parameter.requires_grad_(True); parameter.grad = None
    optimizer_name = str(cfg.get('optimizer','adamw')).lower()
    kwargs = dict(lr=float(cfg['lr']),weight_decay=float(cfg.get('weight_decay',.01)),
                  betas=tuple(cfg.get('betas',(.9,.999))),eps=float(cfg.get('eps',1e-8)),
                  foreach=False)
    if optimizer_name == 'adamw': optimizer=torch.optim.AdamW(model.parameters(),**kwargs)
    elif optimizer_name == 'adam': optimizer=torch.optim.Adam(model.parameters(),**kwargs)
    else: raise ValueError(f'unsupported student optimizer {optimizer_name}')
    total_epochs = int(epochs if epochs is not None else cfg.get('epochs',100))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=int(cfg.get('epochs',100)),eta_min=0)
    order_rng = np.random.default_rng(int(schedule_seed if schedule_seed is not None else seed))
    active = float(feature_weight)!=0.0
    use_diffusion = active and prior is not None
    noise_generator = None
    prior_hash = None
    diffusion_cfg = diffusion_cfg or {}
    if active:
        if teacher_targets is None or len(teacher_targets)!=len(y):
            raise ValueError('teacher targets must align with every student training trial')
        targets = torch.as_tensor(np.asarray(teacher_targets),dtype=torch.float32,device=device)
        if targets.ndim!=2: raise ValueError('teacher targets must be [N,D]')
        _finite(targets,'teacher targets')
    if use_diffusion:
        prior.eval()
        for parameter in prior.parameters(): parameter.requires_grad_(False); parameter.grad=None
        prior_hash=state_hash(prior)
        noise_generator=torch.Generator(device=device).manual_seed(int(noise_seed if noise_seed is not None else seed+100003))
        _,alpha,sigma=ddpm_schedule(int(diffusion_cfg.get('steps',1000)),
            float(diffusion_cfg.get('beta_start',1e-4)),float(diffusion_cfg.get('beta_end',.02)),device)
    history,start_epoch,elapsed_before=[],0,0.0
    if resume_path.is_file():
        saved=torch.load(resolve_local_file(resume_path),map_location='cpu',weights_only=False)
        if saved['fingerprint']!=fingerprint or saved['initial_state_sha256']!=initial_hash:
            raise RuntimeError('student resume identity differs')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);restore_rng(saved['rng'],device)
        order_rng.bit_generator.state=saved['order_rng']
        if use_diffusion:noise_generator.set_state(saved['noise_generator_state'].cpu())
        history,start_epoch,elapsed_before=saved['history'],int(saved['epoch']),saved['elapsed_seconds']
        if len(history)!=start_epoch or not 0<=start_epoch<=total_epochs:
            raise RuntimeError('student resume epoch/history cursor differs')
    started=time.monotonic()
    for epoch in range(start_epoch,total_epochs):
        model.train();order=order_rng.permutation(len(y));totals=np.zeros(3);seen=0
        gradients={}
        for batch_index,start in enumerate(range(0,len(order),int(cfg['batch_size']))):
            if max_batches is not None and batch_index>=int(max_batches):break
            ids=order[start:start+int(cfg['batch_size'])]
            labels=torch.as_tensor(np.asarray(y)[ids],dtype=torch.long,device=device)
            optimizer.zero_grad(set_to_none=True)
            native,logits=_forward(adapter,model,_batch(x,ids,device),forward_fn)
            ce=F.cross_entropy(logits,labels);feature=ce.new_zeros(())
            if active:
                if native.shape[1]!=targets.shape[1]:raise RuntimeError('projected teacher and native student dimensions differ')
                auxiliary=F.layer_norm(native,(native.shape[1],),weight=None,bias=None,eps=1e-5)
                matched=auxiliary
                if use_diffusion:
                    noise=torch.randn(auxiliary.shape,dtype=auxiliary.dtype,device=device,generator=noise_generator)
                    matched=ddim_denoise(auxiliary,noise,prior,alpha,sigma,tuple(diffusion_cfg.get('ddim_timesteps',(250,200,150,100,50,0))))
                feature=F.mse_loss(matched,targets[ids].detach())
                if epoch==0 and batch_index==0:
                    native_gradient=torch.autograd.grad(feature,native,retain_graph=True)[0]
                    parameter_gradients=torch.autograd.grad(feature,[p for p in model.parameters() if p.requires_grad],retain_graph=True,allow_unused=True)
                    norm=sum(float(g.detach().square().sum().cpu()) for g in parameter_gradients if g is not None)**.5
                    _finite(native_gradient,'feature gradient')
                    if native_gradient.norm().item()<=0 or norm<=0:raise RuntimeError('feature-only loss did not reach student backbone')
                    gradients={'feature_native_gradient_norm':float(native_gradient.norm().detach().cpu()),'feature_backbone_gradient_norm':float(norm)}
            loss=ce+float(feature_weight)*feature;_finite(loss,'student loss');loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:_finite(parameter.grad,'student gradient')
            optimizer.step();totals+=np.array([ce.item(),feature.item(),loss.item()])*len(ids);seen+=len(ids)
        lr=float(optimizer.param_groups[0]['lr']);scheduler.step()
        history.append({'epoch':epoch+1,'n_train':seen,'ce':float(totals[0]/seen),
            'feature_mse':float(totals[1]/seen),'loss':float(totals[2]/seen),'lr_used':lr,
            'lr_next':float(optimizer.param_groups[0]['lr']),**gradients})
        elapsed=elapsed_before+time.monotonic()-started;_csv(results/'history.csv',history)
        atomic_torch(resume_path,{'fingerprint':fingerprint,'epoch':epoch+1,
            'initial_state_sha256':initial_hash,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'rng':capture_rng(device),'order_rng':order_rng.bit_generator.state,
            'noise_generator_state':noise_generator.get_state() if use_diffusion else None,
            'history':history,'elapsed_seconds':elapsed})
        if stop_file is not None and Path(stop_file).exists():return {'status':'paused','epoch':epoch+1}
    if use_diffusion:
        if state_hash(prior)!=prior_hash or any(p.grad is not None for p in prior.parameters()):
            raise RuntimeError('frozen prior changed during student training')
    elapsed=elapsed_before+time.monotonic()-started
    return _finish(model,final_path,resume_path,results,fingerprint,history,elapsed,
        final_epoch=total_epochs,initial_state_sha256=initial_hash,feature_weight=float(feature_weight),
        uses_diffusion=use_diffusion,seed=int(seed),schedule_seed=int(schedule_seed if schedule_seed is not None else seed),
        noise_seed=int(noise_seed if noise_seed is not None else seed+100003) if use_diffusion else None,config=cfg)
