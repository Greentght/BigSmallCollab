"""A — the end-to-end fine-tune baseline (the critical control for F+T).

Our balance-gated result compares fusion to *linear heads on frozen features*
(head_big/head_small). A reviewer's real baseline is: take the LOSO-pretrained
model and actually FINE-TUNE it end-to-end on the same K labeled test-subject
trials. If full fine-tuning closes the gap, the frozen-feature story weakens; if
not, our cheap frozen fusion holds up. This script measures it.

For each (cell, test subject, seed): retrain the LOSO base (all-but-subject) once
— saving the model — then for each K and draw, clone it, fine-tune end-to-end on
the K support trials, and evaluate on the same held-out rows the frozen artifacts
use (identical StratifiedShuffleSplit(seed)). Reports, per subject:
  ft_big  : full end-to-end fine-tune of the big model on K
  ft_small: full end-to-end fine-tune of the small model on K
  head_big/head_small/fusion : the frozen-feature heads (recomputed here to be on
            the exact same split) — so ft-vs-frozen is apples-to-apples.

Focused by default on a couple of cells to bound GPU cost; widen with --cells.

    conda run -n mirepnet python experiments/fusion/run_finetune_baseline.py \
        --cells BNCI2014001-4:mirepnet:ifnet --Ks 20 30 --gpu 2 --ft_epochs 30
"""
import argparse
import copy
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import config
import data
from collab import fusion as fz
from collab.router import _softmax_np as _sm
from models import get_adapter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS = os.path.join(ROOT, 'results', 'metrics')


_SPLIT_CACHE = {}


def cached_loso_split(dataset, fold):
    """loso_split is seed-independent (the fold is fixed by the held-out subject),
    so cache it — otherwise the mne/EEGDataset full-dataset load repeats per seed
    and per model (6x redundant I/O, the dominant cost)."""
    key = (dataset, fold)
    if key not in _SPLIT_CACHE:
        _SPLIT_CACHE[key] = data.loso_split(dataset, fold)
    return _SPLIT_CACHE[key]


def build_loso_base(model_name, dataset, fold, seed, nc, mcfg, device, base_epochs=None):
    """Train the LOSO base (all-but-`fold`) and return (adapter, model, X_te, y_te)."""
    X_tr, y_tr, _subj, X_te, y_te = cached_loso_split(dataset, fold)
    cfg = dict(mcfg); cfg.update(in_channels=X_tr.shape[1], samples=X_tr.shape[2],
                                 dataset_name=dataset)
    if base_epochs is not None:
        cfg['epochs'] = base_epochs
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    ad = get_adapter(model_name, device=device, **cfg)
    model = ad.build(nc)
    model = ad.finetune(model, X_tr, y_tr, nc)
    return ad, model, X_te, y_te


def finetune_on_support(ad, base_model, X_sup, y_sup, nc, ft_epochs, ft_lr, seed):
    """Clone the LOSO base and continue-train end-to-end on the K support trials."""
    import torch
    torch.manual_seed(seed)
    model = copy.deepcopy(base_model)
    saved = ad.cfg.get('epochs'), ad.cfg.get('lr')
    ad.cfg['epochs'], ad.cfg['lr'] = ft_epochs, ft_lr
    model = ad.finetune(model, X_sup, y_sup, nc)
    ad.cfg['epochs'], ad.cfg['lr'] = saved
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='BNCI2014001-4:mirepnet:ifnet,BNCI2014004:cbramod_native:ifnet',
                    help='comma of dataset:big:small')
    ap.add_argument('--Ks', type=int, nargs='+', default=[20, 30])
    ap.add_argument('--seeds', type=int, nargs='+', default=[666, 667, 668])
    ap.add_argument('--draws', type=int, default=3)
    ap.add_argument('--subjects', type=int, nargs='+', default=None,
                    help='restrict to these held-out subjects (default all)')
    ap.add_argument('--base_epochs', type=int, default=None,
                    help='override LOSO base training epochs (speed control)')
    ap.add_argument('--ft_epochs', type=int, default=30)
    ap.add_argument('--ft_lr', type=float, default=5e-4)
    ap.add_argument('--gpu', type=int, default=None)
    a = ap.parse_args()
    import torch
    device = (f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available() else 'cpu')
    cells = [tuple(c.split(':')) for c in a.cells.split(',')]
    os.makedirs(METRICS, exist_ok=True)
    rows = []
    for ds, big, small in cells:
        dcfg = config.load_dataset_config(ds)
        nc, n_sub = dcfg['num_classes'], dcfg['num_subjects']
        bcfg, scfg = config.load_model_config(big), config.load_model_config(small)
        subjects = a.subjects if a.subjects is not None else list(range(n_sub))
        for seed in a.seeds:
            for t in subjects:
                adb, mb, Xb_te, yb = build_loso_base(big, ds, t, seed, nc, bcfg, device, a.base_epochs)
                ads, ms, Xs_te, ys = build_loso_base(small, ds, t, seed, nc, scfg, device, a.base_epochs)
                assert np.array_equal(yb, ys), 'big/small held-out labels misaligned'
                y = yb
                if np.bincount(y, minlength=nc).min() < 2:
                    continue
                # frozen features for this held-out subject (same rows)
                bf = adb.infer(mb, Xb_te)[0]; sf = ads.infer(ms, Xs_te)[0]
                for K in a.Ks:
                    if len(y) <= K + nc:
                        continue
                    for di, (tr, te) in enumerate(StratifiedShuffleSplit(
                            n_splits=a.draws, train_size=K, random_state=seed).split(bf, y)):
                        yt = y[te]
                        # end-to-end fine-tune BOTH models on the K support
                        mbf = finetune_on_support(adb, mb, Xb_te[tr], y[tr], nc, a.ft_epochs, a.ft_lr, seed)
                        msf = finetune_on_support(ads, ms, Xs_te[tr], y[tr], nc, a.ft_epochs, a.ft_lr, seed)
                        # eval logits/feats of the FINE-TUNED (subject-adapted) models
                        fbf_te, lb_te = adb.infer(mbf, Xb_te[te])
                        fsf_te, ls_te = ads.infer(msf, Xs_te[te])
                        pbf = _sm(lb_te); psf = _sm(ls_te)
                        ftb = (pbf.argmax(1) == yt).mean() * 100
                        fts = (psf.argmax(1) == yt).mean() * 100
                        # COLLABORATION of the fine-tuned models (the fair scheme):
                        ft_ens = ((pbf + psf).argmax(1) == yt).mean() * 100      # softmax average
                        ft_conf = (np.where(pbf.max(1) >= psf.max(1), pbf.argmax(1),
                                            psf.argmax(1)) == yt).mean() * 100     # confidence route
                        ft_orac = ((pbf.argmax(1) == yt) | (psf.argmax(1) == yt)).mean() * 100
                        # feature fusion of the fine-tuned features (support is in-sample -> caveat)
                        fbf_tr = adb.infer(mbf, Xb_te[tr])[0]; fsf_tr = ads.infer(msf, Xs_te[tr])[0]
                        ft_fuse = (fz.fusion_concat(fbf_tr, fsf_tr, y[tr], fbf_te, fsf_te, nc, hidden=0) == yt).mean() * 100
                        # frozen-feature heads on the exact same split (the old scheme)
                        hb = (fz.head_single(bf[tr], y[tr], bf[te], nc) == yt).mean() * 100
                        hs = (fz.head_single(sf[tr], y[tr], sf[te], nc) == yt).mean() * 100
                        fu = (fz.fusion_concat(bf[tr], sf[tr], y[tr], bf[te], sf[te], nc, hidden=0) == yt).mean() * 100
                        rows.append(dict(dataset=ds, big=big, small=small, seed=seed,
                                         subject=t, K=K, draw=di, ft_big=round(ftb, 2),
                                         ft_small=round(fts, 2), ft_ens=round(ft_ens, 2),
                                         ft_conf=round(ft_conf, 2), ft_fuse=round(ft_fuse, 2),
                                         ft_oracle=round(ft_orac, 2), head_big=round(hb, 2),
                                         head_small=round(hs, 2), fusion=round(fu, 2)))
                    print(f'[{ds} {big}x{small}] seed{seed} S{t} K{K} done', flush=True)
                del mb, ms
                if device != 'cpu':
                    torch.cuda.empty_cache()

    import pandas as pd
    df = pd.DataFrame(rows)
    out = os.path.join(METRICS, 'finetune_baseline.csv')
    df.to_csv(out, index=False)
    import numpy as _np
    from scipy.stats import wilcoxon as _wil
    print('\n=== collaboration OF FINE-TUNED models vs best single fine-tuned model ===')
    for (ds, big, small, K), g in df.groupby(['dataset', 'big', 'small', 'K']):
        m = g[['ft_big', 'ft_small', 'ft_ens', 'ft_conf', 'ft_fuse', 'ft_oracle',
               'fusion', 'head_big']].mean()
        # subject-level: best single fine-tuned vs the collaboration schemes
        gs = g.groupby('subject')[['ft_big', 'ft_small', 'ft_ens', 'ft_fuse']].mean()
        best_single_ft = _np.maximum(gs.ft_big, gs.ft_small)
        def _p(col):
            try:
                return _wil(gs[col], best_single_ft)[1]
            except ValueError:
                return 1.0
        print(f'  {ds} {big}x{small} K{K}:')
        print(f'     ft_big={m.ft_big:.2f} ft_small={m.ft_small:.2f} '
              f'best_single_ft={best_single_ft.mean():.2f} | ft_ens={m.ft_ens:.2f} '
              f'ft_conf={m.ft_conf:.2f} ft_fuse={m.ft_fuse:.2f} | ft_oracle={m.ft_oracle:.2f}')
        print(f'     ft_ens−best_single_ft={m.ft_ens-best_single_ft.mean():+.2f} '
              f'(p={_p("ft_ens"):.3f})   ft_fuse−best={m.ft_fuse-best_single_ft.mean():+.2f} '
              f'(p={_p("ft_fuse"):.3f})   [frozen fusion was {m.fusion:.2f}]')
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
