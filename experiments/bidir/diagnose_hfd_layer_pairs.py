"""Cross-fitted trial-level layer similarity on 004 calibration data only."""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['BIGSMALLCOLLAB_CPU_THREADS'] = '1'

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold

from collab.dcm import mutual_losses
from data.loso_source import sha256_file
from experiments.bidir.run_dml_ds_cross_hfd_all import CONFIG, ROOT, make_models, source_and_split
from experiments.bidir.run_dml_ds_cross_hfd_fewshot import atomic_json, uid_hash
from experiments.bidir.summarize_dml_ds_cross_hfd_all import write_csv
from experiments.storage import REPORTS_ROOT, require_report_output


def u_centered_gram(x):
    # Inner product of two U-centered Gram matrices is proportional to
    # unbiased HSIC. Removing the diagonal matters with few trials and many features.
    x = x.detach().cpu().reshape(x.shape[0], -1).double()
    x = x - x.mean(dim=0)
    n = x.shape[0]
    if n < 4:
        raise ValueError('At least four paired trials are required for U-centering')
    gram = (x @ x.T).numpy()
    np.fill_diagonal(gram, 0.0)
    row = gram.sum(axis=1)
    centered = gram - (row[:, None] + row[None, :]) / (n - 2) + row.sum() / ((n - 1) * (n - 2))
    np.fill_diagonal(centered, 0.0)
    norm = np.linalg.norm(centered)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('Degenerate layer representation')
    return centered / norm


def measure_fold(base, seed, epochs, x, y, fit, valid):
    big, small = 'cbramod', 'eegnet'
    adapters, models, heads, hfd, optimizers = make_models(
        'BNCI2014004', f'{big}_{small}', x.shape[1], 2, base, seed, torch.device('cpu'))
    hfd.close()  # No HFD operator is active during the protocol's five warm-up epochs.
    hooks = []
    try:
        inputs = [adapter.preprocess(x[fit]) for adapter in adapters]
        yfit = torch.as_tensor(y[fit], dtype=torch.long)
        generator = torch.Generator().manual_seed(seed + 100003)
        big_cfg, small_cfg = base['teacher_training'][big], base['student_training'][small]
        for epoch in range(epochs):
            for model in models:
                model.train()
            for rows in torch.randperm(len(fit), generator=generator).split(big_cfg['batch_size']):
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                logits = [a.forward(m, v[rows])[1] for a, m, v in zip(adapters, models, inputs)]
                outputs = [[v] + head(len(rows)) for head, v in zip(heads, logits)]
                loss = sum(mutual_losses(outputs, yfit[rows], 'dml_ds', base['temperature'],
                                         base['alpha'], base['lambda']))
                loss.backward()
                for opt in optimizers:
                    torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 1.0)
                    opt.step()
            factor = 0.5 * (1 + np.cos(np.pi * (epoch + 1) / big_cfg['epochs']))
            for opt, cfg in zip(optimizers, (big_cfg, small_cfg)):
                for group in opt.param_groups:
                    group['lr'] = cfg['lr'] * factor
        for head in heads:
            head.close()
        modules = {'patch_embedding': models[0].backbone.patch_embedding,
                   'block4': models[0].backbone.encoder.layers[3],
                   'batchnorm1': models[1].batchnorm1, 'pooling2': models[1].pooling2}
        grams = {}
        def capture(layer):
            def hook(_module, _inputs, output):
                grams[layer] = u_centered_gram(output)
            return hook
        hooks = [module.register_forward_hook(capture(name)) for name, module in modules.items()]
        with torch.no_grad():
            for adapter, model in zip(adapters, models):
                model.eval()
                adapter.forward(model, adapter.preprocess(x[valid]))
        return grams
    finally:
        for hook in hooks:
            hook.remove()
        for head in heads:
            head.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', type=int, default=0, choices=range(9))
    parser.add_argument('--seed', type=int, default=666, choices=(666, 667, 668))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    cfg = yaml.safe_load(CONFIG.read_text())
    base = yaml.safe_load((ROOT / cfg['baseline_config']).read_text())
    source, split = source_and_split('BNCI2014004', args.subject, args.seed, base)
    x, y, uids = split['x_cal_raw'], split['y_cal'], split['trial_uid_cal']
    root = REPORTS_ROOT / 'dml_ds_hfd_layer_pilot_004_v1' / f'subject_{args.subject}_seed_{args.seed}'
    output = require_report_output(root / 'pair_means.csv')
    root.mkdir(parents=True, exist_ok=True)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    rows, provenance = [], []
    pairs = list(product(('patch_embedding', 'block4'), ('batchnorm1', 'pooling2')))
    draws = 199
    for fold, (fit, valid) in enumerate(splitter.split(x, y)):
        grams = measure_fold(base, args.seed, cfg['hfd_warmup_epochs'], x, y, fit, valid)
        rng = np.random.default_rng(args.seed * 100 + fold)
        perm = np.tile(np.arange(len(valid)), (draws, 1))
        for label in np.unique(y[valid]):
            idx = np.flatnonzero(y[valid] == label)
            for order in perm:
                order[idx] = rng.permutation(idx)
        for big_layer, small_layer in pairs:
            a, b = grams[big_layer], grams[small_layer]
            score = float(np.sum(a * b))
            null = np.sum(a * b[perm[:, :, None], perm[:, None, :]], axis=(1, 2))
            rows.append(dict(fold=fold + 1, big_layer=big_layer, small_layer=small_layer,
                             n_fit=len(fit), n_valid=len(valid), cka=score,
                             within_class_null_mean=float(null.mean()),
                             paired_minus_null=score - float(null.mean())))
        provenance.append(dict(fold=fold + 1, fit_uid_sha256=uid_hash(uids[fit].tolist()),
                               valid_uid_sha256=uid_hash(uids[valid].tolist())))
    means = []
    for big_layer, small_layer in pairs:
        subset = [r for r in rows if (r['big_layer'], r['small_layer']) == (big_layer, small_layer)]
        means.append(dict(big_layer=big_layer, small_layer=small_layer,
                          **{k: float(np.mean([r[k] for r in subset]))
                             for k in ('cka', 'within_class_null_mean', 'paired_minus_null')}))
    write_csv(root / 'by_fold.csv', rows, list(rows[0]))
    write_csv(output, means, list(means[0]))
    atomic_json(require_report_output(root / 'provenance.json'), {
        'dataset': 'BNCI2014004', 'pair': 'cbramod_eegnet', 'subject_zero_based': args.subject,
        'seed': args.seed, 'calibration_n': len(y), 'calibration_uid_sha256': uid_hash(uids.tolist()),
        'source_manifest_sha256': source.source_manifest_sha256, 'folds': provenance,
        'diagnostic_code_sha256': sha256_file(Path(__file__)),
        'preprocessing_and_warmup_config': str(CONFIG.relative_to(ROOT)),
        'warmup_epochs': cfg['hfd_warmup_epochs'], 'device': 'cpu', 'training_threads': 1,
        'measurement': 'flatten native activations per trial in eval mode, linear CKA with U-centered Gram matrices',
        'null': 'permute trial pairing inside each validation class', 'null_draws_per_fold': draws,
        'outer_test_used_for_training_or_measurement': False, 'trained_layer_selection_variants': 0})
    print(f'{len(y)} calibration trials, 3 folds, {output}', flush=True)
    for row in means:
        print(row)


if __name__ == '__main__':
    main()
