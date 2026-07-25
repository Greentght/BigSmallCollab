"""Offline KD + feature-align: distill a cached (frozen) teacher into a student.

Run in the STUDENT's conda env (small models -> mirepnet). The teacher's
train-split artifact (feats + logits) must already exist — exported earlier via
finetune_export.py in the teacher's own env. The big teacher is never loaded here.

    conda run -n mirepnet python scripts/run_distill.py \
        --dataset BNCI2014004 --teacher cbramod --student ifnet \
        --lam_kd 0.5 --lam_feat 0.5

Trains two conditions per (subject, seed): the plain student (lam=0) baseline and
the distilled student, and writes acc/kappa to
results/metrics/<dataset>_distill_<teacher>_to_<student>.csv.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from collab.distill import distill_student
from core import artifacts, config, data, metrics
from core.registry import get_adapter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='BNCI2014004')
    p.add_argument('--teacher', required=True)
    p.add_argument('--student', required=True)
    p.add_argument('--lam_kd', type=float, default=0.5)
    p.add_argument('--lam_feat', type=float, default=0.5)
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--teacher_correct_only', dest='teacher_correct_only',
                   action='store_true', default=True,
                   help='mask KD/feat align to teacher-correct samples (default)')
    p.add_argument('--no_teacher_correct_only', dest='teacher_correct_only',
                   action='store_false',
                   help='align on all samples (legacy behaviour)')
    p.add_argument('--mask_ablation', action='store_true',
                   help='run base + {KD,Combo} x {all,masked} in one pass')
    p.add_argument('--adaptive', action='store_true',
                   help='run base + {KD,Combo} x {all,masked,mc} in one pass; '
                        'mc = MC-dropout entropy weight from <subj>_<seed>_train_mc.npz')
    p.add_argument('--dkd_ablation', action='store_true',
                   help='decoupled-KD (logits only): base / KD_all / KD_masked / '
                        'DKD_all / DKD_tmask (teacher-wrong -> drop TCKD, keep NCKD)')
    p.add_argument('--dkd_alpha', type=float, default=1.0, help='TCKD weight')
    p.add_argument('--dkd_beta', type=float, default=1.0, help='NCKD weight')
    p.add_argument('--proto_ablation', action='store_true',
                   help='class-prototype alignment: base / KD / GlobalFeat / '
                        'Proto / Proto_w (class-reliability + margin weighted)')
    p.add_argument('--lam_proto', type=float, default=0.5)
    p.add_argument('--rel_delta', type=float, default=0.0, help='class-reliability margin delta')
    p.add_argument('--rel_tau', type=float, default=0.1, help='class-reliability temperature')
    p.add_argument('--margin_gamma', type=float, default=0.0, help='sample prototype-margin threshold')
    p.add_argument('--margin_tau', type=float, default=0.1, help='sample prototype-margin temperature')
    p.add_argument('--relational_ablation', action='store_true',
                   help='relational KD: base / SampleCos / SimFull / IntraInter / ProtoSim '
                        '(batch B x B similarity, class-balanced sampling)')
    p.add_argument('--lam_sim', type=float, default=0.5)
    p.add_argument('--lam_intra', type=float, default=0.5)
    p.add_argument('--lam_inter', type=float, default=0.5)
    p.add_argument('--fewshot_pearson', action='store_true',
                   help='few-shot big->small KD with the Pearson logit-distance '
                        'regularizer (L_inter): base / KD / Pearson / KD+Pearson, '
                        'subsampling --shots per class from the teacher train pool')
    p.add_argument('--shots', type=int, nargs='+', default=[5, 10, 20],
                   help='few-shot: number of labeled trials PER CLASS')
    p.add_argument('--lam_pearson', type=float, default=0.5,
                   help='weight of the Pearson logit-distance term L_inter')
    p.add_argument('--rel_conds', default=None,
                   help='comma list to restrict relational conditions, e.g. '
                        'base,IntraOnly,InterOnly,IntraInter')
    p.add_argument('--subjects', type=int, nargs='+', default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--gpu', type=int, default=None)
    p.add_argument('--out_csv', default=None)
    return p.parse_args()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _prototypes(feats, y, nc):
    """teacher class means (nc, D)."""
    M = np.zeros((nc, feats.shape[1]), np.float32)
    for c in range(nc):
        m = y == c
        if m.any():
            M[c] = feats[m].mean(0)
    return M


def _proto_margins(feats, M, y):
    """per-sample teacher prototype margin: cos(f_i,M_{y_i}) - max_{k!=y} cos."""
    fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    cs = fn @ Mn.T                                    # (N, nc)
    own = cs[np.arange(len(y)), y]
    other = cs.copy(); other[np.arange(len(y)), y] = -np.inf
    return (own - other.max(1)).astype(np.float32)


_per_class = metrics.per_class   # per-class breakdown (shared, core.metrics)


def run_proto_ablation(a, scfg, subjects, seeds, val_split, nc, device, out_csv):
    """base / KD / GlobalFeat / Proto / Proto_w. Records overall + per-class."""
    def make_student():
        acfg = dict(scfg)
        return acfg
    rows = []
    for seed in seeds:
        for subj in subjects:
            try:
                tch = artifacts.load(a.dataset, a.teacher, subj, seed, 'train')
            except FileNotFoundError as e:
                print(f'[miss teacher] S{subj} seed{seed}: {e}'); continue
            X_tr, y_tr, X_te, y_te = data.subject_split(
                a.dataset, subj, val_split=val_split, seed=seed)
            assert np.array_equal(tch['y'], y_tr), 'teacher rows misaligned'

            # teacher-side, cached & static: prototypes, per-class reliability, margins
            M = _prototypes(tch['feats'], y_tr, nc)
            tp = tch['logits'].argmax(1)
            Rt = np.array([(tp[y_tr == c] == c).mean() if (y_tr == c).any() else 0.0
                           for c in range(nc)], np.float32)
            margins = _proto_margins(tch['feats'], M, y_tr)

            common = dict(epochs=scfg.get('epochs', 50), lr=scfg.get('lr', 1e-3),
                          weight_decay=scfg.get('weight_decay', 0.01),
                          batch_size=scfg.get('batch_size', 16),
                          temperature=a.temperature)

            def adapter():
                acfg = dict(scfg); acfg.update(
                    in_channels=X_tr.shape[1], samples=X_tr.shape[2],
                    dataset_name=a.dataset)
                return get_adapter(a.student, device=device, **acfg)

            # base (2-pass): capture train preds -> student per-class reliability R_c^S
            te_pred, tr_pred = distill_student(
                adapter(), nc, X_tr, y_tr, tch['feats'], tch['logits'], X_te,
                lam_kd=0.0, lam_feat=0.0, teacher_correct_only=False,
                return_train_preds=True, **common)
            Rs = np.array([(tr_pred[y_tr == c] == c).mean() if (y_tr == c).any() else 0.0
                           for c in range(nc)], np.float32)
            wc = _sigmoid((Rt - Rs - a.rel_delta) / a.rel_tau)          # (nc,)
            w_sample = (wc[y_tr] * _sigmoid((margins - a.margin_gamma) / a.margin_tau)
                        ).astype(np.float32)

            lp = a.lam_proto
            conds = {
                'base':       dict(lam_kd=0.0, lam_feat=0.0, teacher_correct_only=False),
                'KD':         dict(lam_kd=a.lam_kd, lam_feat=0.0, teacher_correct_only=False),
                'GlobalFeat': dict(lam_kd=0.0, lam_feat=lp, teacher_correct_only=False),
                'Proto':      dict(lam_kd=0.0, lam_feat=lp, feat_proto=True,
                                   teacher_correct_only=False),
                'Proto_w':    dict(lam_kd=0.0, lam_feat=lp, feat_proto=True,
                                   sample_weight=w_sample),
            }
            for cond, kw in conds.items():
                preds = (te_pred if cond == 'base' else distill_student(
                    adapter(), nc, X_tr, y_tr, tch['feats'], tch['logits'], X_te,
                    **kw, **common))
                m = metrics.evaluate(y_te, preds)
                row = dict(dataset=a.dataset, subject=subj, seed=seed,
                           condition=f'{a.student}_{cond}',
                           acc=m['acc'], kappa=m['kappa'])
                row.update(_per_class(y_te, preds, nc))
                rows.append(row)
                print(f"S{subj} seed{seed} {a.student}_{cond} | acc={m['acc']} "
                      f"kappa={m['kappa']} f1={row['macro_f1']}", flush=True)

    if not rows:
        print('No teacher artifacts found.'); return
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f'\nWrote {out_csv}')
    print(df.groupby('condition')[['acc', 'kappa', 'macro_f1']].mean().round(3))


def run_relational_ablation(a, scfg, subjects, seeds, val_split, nc, device, out_csv):
    """base / SampleCos / SimFull / IntraInter / ProtoSim. Class-balanced
    sampling for ALL conditions so only the loss differs. Records per-class."""
    lp = a.lam_proto
    conds = {
        'base':       dict(lam_kd=0.0, lam_feat=0.0),
        'VanillaKD':  dict(lam_kd=a.lam_kd, lam_feat=0.0),
        'ProtoOnly':  dict(lam_kd=0.0, lam_feat=lp, feat_proto=True),
        'KDProto':    dict(lam_kd=a.lam_kd, lam_feat=lp, feat_proto=True),
        'SampleCos':  dict(lam_kd=0.0, lam_feat=lp),
        'SimFull':    dict(lam_kd=0.0, lam_feat=0.0,
                           relational=dict(mode='sim', lam_sim=a.lam_sim)),
        'IntraInter': dict(lam_kd=0.0, lam_feat=0.0,
                           relational=dict(mode='intra_inter',
                                           lam_intra=a.lam_intra, lam_inter=a.lam_inter)),
        'IntraOnly':  dict(lam_kd=0.0, lam_feat=0.0,
                           relational=dict(mode='intra_inter',
                                           lam_intra=a.lam_intra, lam_inter=0.0)),
        'InterOnly':  dict(lam_kd=0.0, lam_feat=0.0,
                           relational=dict(mode='intra_inter',
                                           lam_intra=0.0, lam_inter=a.lam_inter)),
        'ProtoSim':   dict(lam_kd=0.0, lam_feat=lp, feat_proto=True,
                           relational=dict(mode='sim', lam_sim=a.lam_sim)),
        # EA-KD (真 EA-KD: w=1/2 H_T(1+H_S/logC), 温度 T'=ea_temp, 高教师熵=高价值)
        'EA_KD':         dict(lam_kd=a.lam_kd, lam_feat=0.0, ea_kd=True),
        'CorrectMaskKD': dict(lam_kd=a.lam_kd, lam_feat=0.0, weight_src='mask'),
        'CorrectMaskEA': dict(lam_kd=a.lam_kd, lam_feat=0.0, ea_kd=True, weight_src='mask'),
        # Combo = KD + 逐样本 feat-align; EA_Combo = EA 权重同时作用于 KD 与 feat
        'Combo':         dict(lam_kd=a.lam_kd, lam_feat=lp),
        'EA_Combo':      dict(lam_kd=a.lam_kd, lam_feat=lp, ea_kd=True),
    }
    if a.rel_conds:
        keep = set(a.rel_conds.split(','))
        conds = {k: v for k, v in conds.items() if k in keep}
    rows = []
    for seed in seeds:
        for subj in subjects:
            try:
                tch = artifacts.load(a.dataset, a.teacher, subj, seed, 'train')
            except FileNotFoundError as e:
                print(f'[miss teacher] S{subj} seed{seed}: {e}'); continue
            X_tr, y_tr, X_te, y_te = data.subject_split(
                a.dataset, subj, val_split=val_split, seed=seed)
            assert np.array_equal(tch['y'], y_tr), 'teacher rows misaligned'
            common = dict(epochs=scfg.get('epochs', 50), lr=scfg.get('lr', 1e-3),
                          weight_decay=scfg.get('weight_decay', 0.01),
                          batch_size=scfg.get('batch_size', 16),
                          temperature=a.temperature, teacher_correct_only=False,
                          balanced_batch=True, seed=seed)
            t_mask = (tch['logits'].argmax(1) == y_tr).astype(np.float32)
            for cond, kw0 in conds.items():
                kw = dict(kw0)
                if kw.pop('weight_src', None) == 'mask':
                    kw['sample_weight'] = t_mask
                acfg = dict(scfg); acfg.update(
                    in_channels=X_tr.shape[1], samples=X_tr.shape[2],
                    dataset_name=a.dataset)
                student = get_adapter(a.student, device=device, **acfg)
                preds = distill_student(
                    student, nc, X_tr, y_tr, tch['feats'], tch['logits'], X_te,
                    **kw, **common)
                m = metrics.evaluate(y_te, preds)
                row = dict(dataset=a.dataset, subject=subj, seed=seed,
                           condition=f'{a.student}_{cond}',
                           acc=m['acc'], kappa=m['kappa'])
                row.update(_per_class(y_te, preds, nc))
                rows.append(row)
                print(f"S{subj} seed{seed} {a.student}_{cond} | acc={m['acc']} "
                      f"kappa={m['kappa']} f1={row['macro_f1']}", flush=True)
    if not rows:
        print('No teacher artifacts found.'); return
    df = pd.DataFrame(rows); df.to_csv(out_csv, index=False)
    print(f'\nWrote {out_csv}')
    print(df.groupby('condition')[['acc', 'kappa', 'macro_f1']].mean().round(3))


def _fewshot_idx(y_tr, n_shot, nc, seed):
    """Pick n_shot indices per class from the train pool (seeded). If a class has
    fewer than n_shot samples, take all of them."""
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(nc):
        pool = np.where(y_tr == c)[0]
        k = min(n_shot, len(pool))
        idx += list(rng.choice(pool, k, replace=False))
    rng.shuffle(idx)
    return np.array(sorted(idx))


def run_fewshot_pearson(a, scfg, subjects, seeds, val_split, nc, device, out_csv):
    """Few-shot big->small distillation with the Pearson logit-distance term.

    The teacher artifact is cached at the standard 70%-train split; we subsample
    ``--shots`` labelled trials per class as the student's few-shot calibration
    set (teacher logits/feats stay row-aligned), and evaluate on the full 30%
    test split. Conditions per (shots, subject, seed):
      base       - student only (no teacher)
      KD         - vanilla logit KD (KL @ T)
      Pearson    - only L_inter = mean_i (1 - corr(s_logits_i, t_logits_i))
      KD+Pearson - KD + L_inter (regularizer on top of KD)
    """
    conds = {
        'base':       dict(lam_kd=0.0, lam_feat=0.0),
        'KD':         dict(lam_kd=a.lam_kd, lam_feat=0.0),
        'Pearson':    dict(lam_kd=0.0, lam_feat=0.0,
                           pearson=dict(lam=a.lam_pearson)),
        'KD+Pearson': dict(lam_kd=a.lam_kd, lam_feat=0.0,
                           pearson=dict(lam=a.lam_pearson)),
    }
    if nc == 2:
        print('[warn] C=2: Pearson over 2 logits is degenerate (d_p in {0,2}); '
              'results only sanity-check the plumbing.', flush=True)
    rows = []
    for n_shot in a.shots:
        for seed in seeds:
            for subj in subjects:
                try:
                    tch = artifacts.load(a.dataset, a.teacher, subj, seed, 'train')
                except FileNotFoundError as e:
                    print(f'[miss teacher] S{subj} seed{seed}: {e}'); continue
                X_tr, y_tr, X_te, y_te = data.subject_split(
                    a.dataset, subj, val_split=val_split, seed=seed)
                assert np.array_equal(tch['y'], y_tr), 'teacher rows misaligned'

                sel = _fewshot_idx(y_tr, n_shot, nc, seed)
                Xs, ys = X_tr[sel], y_tr[sel]
                fts, lts = tch['feats'][sel], tch['logits'][sel]
                common = dict(epochs=scfg.get('epochs', 50), lr=scfg.get('lr', 1e-3),
                              weight_decay=scfg.get('weight_decay', 0.01),
                              batch_size=scfg.get('batch_size', 16),
                              temperature=a.temperature, teacher_correct_only=False,
                              balanced_batch=True, seed=seed)
                for cond, kw in conds.items():
                    acfg = dict(scfg); acfg.update(
                        in_channels=X_tr.shape[1], samples=X_tr.shape[2],
                        dataset_name=a.dataset)
                    student = get_adapter(a.student, device=device, **acfg)
                    preds = distill_student(
                        student, nc, Xs, ys, fts, lts, X_te, **kw, **common)
                    m = metrics.evaluate(y_te, preds)
                    row = dict(dataset=a.dataset, subject=subj, seed=seed,
                               shots=n_shot, n_train=len(sel),
                               condition=f'{a.student}_{cond}',
                               acc=m['acc'], kappa=m['kappa'])
                    row.update(_per_class(y_te, preds, nc))
                    rows.append(row)
                    print(f"shots{n_shot} S{subj} seed{seed} {a.student}_{cond} | "
                          f"acc={m['acc']} kappa={m['kappa']}", flush=True)
    if not rows:
        print('No teacher artifacts found.'); return
    df = pd.DataFrame(rows); df.to_csv(out_csv, index=False)
    print(f'\nWrote {out_csv}')
    print(df.groupby(['shots', 'condition'])[['acc', 'kappa']].mean().round(3))


def main():
    a = parse_args()
    dcfg = config.load_dataset_config(a.dataset)
    scfg = config.load_model_config(a.student)
    subjects = a.subjects or list(range(dcfg['num_subjects']))
    seeds = a.seeds or dcfg['seeds']
    val_split = dcfg['val_split']
    nc = dcfg['num_classes']
    device = (f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available()
              else 'cpu')

    out_csv = a.out_csv or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'metrics',
        f'{a.dataset}_distill_{a.teacher}_to_{a.student}.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    if a.proto_ablation:
        return run_proto_ablation(a, scfg, subjects, seeds, val_split, nc,
                                  device, out_csv)
    if a.relational_ablation:
        return run_relational_ablation(a, scfg, subjects, seeds, val_split, nc,
                                       device, out_csv)
    if a.fewshot_pearson:
        return run_fewshot_pearson(a, scfg, subjects, seeds, val_split, nc,
                                   device, out_csv)

    # each condition = (lam_kd, lam_feat, weight_mode). weight_mode selects the
    # per-sample alignment weight: 'all'=1 / 'masked'=1[teacher_correct] /
    # 'mc'=1-H_mc/logC (MC-dropout entropy). Irrelevant when both lams are 0.
    s = a.student
    if a.adaptive:
        conditions = {
            f'{s}_base':         (0.0, 0.0, 'all'),
            f'{s}_KD_all':       (a.lam_kd, 0.0, 'all'),
            f'{s}_KD_masked':    (a.lam_kd, 0.0, 'masked'),
            f'{s}_KD_mc':        (a.lam_kd, 0.0, 'mc'),
            f'{s}_Combo_all':    (a.lam_kd, a.lam_feat, 'all'),
            f'{s}_Combo_masked': (a.lam_kd, a.lam_feat, 'masked'),
            f'{s}_Combo_mc':     (a.lam_kd, a.lam_feat, 'mc'),
        }
    elif a.dkd_ablation:
        conditions = {
            f'{s}_base':        (0.0, 0.0, 'all'),
            f'{s}_KD_all':      (a.lam_kd, 0.0, 'all'),
            f'{s}_KD_masked':   (a.lam_kd, 0.0, 'masked'),
            f'{s}_DKD_all':     (a.lam_kd, 0.0, 'dkd_all'),
            f'{s}_DKD_tmask':   (a.lam_kd, 0.0, 'dkd_tmask'),
        }
    elif a.mask_ablation:
        conditions = {
            f'{s}_base':         (0.0, 0.0, 'all'),
            f'{s}_KD_all':       (a.lam_kd, 0.0, 'all'),
            f'{s}_KD_masked':    (a.lam_kd, 0.0, 'masked'),
            f'{s}_Combo_all':    (a.lam_kd, a.lam_feat, 'all'),
            f'{s}_Combo_masked': (a.lam_kd, a.lam_feat, 'masked'),
        }
    else:
        mode = 'masked' if a.teacher_correct_only else 'all'
        conditions = {
            f'{s}_base': (0.0, 0.0, mode),
            f'{s}_KD<-{a.teacher}': (a.lam_kd, a.lam_feat, mode),
        }

    rows = []
    for seed in seeds:
        for subj in subjects:
            try:
                tch = artifacts.load(a.dataset, a.teacher, subj, seed, 'train')
            except FileNotFoundError as e:
                print(f'[miss teacher] S{subj} seed{seed}: {e}'); continue

            X_tr, y_tr, X_te, y_te = data.subject_split(
                a.dataset, subj, val_split=val_split, seed=seed)
            assert np.array_equal(tch['y'], y_tr), (
                f'teacher train rows misaligned for S{subj} seed{seed}')

            # precompute per-sample alignment weights shared by all conditions
            w_all = np.ones(len(y_tr), dtype=np.float32)
            w_masked = (tch['logits'].argmax(1) == y_tr).astype(np.float32)
            w_mc = None
            if a.adaptive:
                mcp = os.path.join(os.path.dirname(artifacts.artifact_path(
                    a.dataset, a.teacher, subj, seed, 'train')),
                    f'{subj}_{seed}_train_mc.npz')
                d = np.load(mcp)
                assert np.array_equal(d['y'], y_tr), (
                    f'MC sidecar rows misaligned for S{subj} seed{seed}')
                logC = np.log(nc)
                w_mc = np.clip(1.0 - d['pred_entropy'] / logC, 0.0, 1.0).astype(np.float32)
            weights = {'all': w_all, 'masked': w_masked, 'mc': w_mc}

            correct = (tch['logits'].argmax(1) == y_tr).astype(np.float32)

            for cond, (lk, lf, mode) in conditions.items():
                acfg = dict(scfg)
                acfg.update(in_channels=X_tr.shape[1], samples=X_tr.shape[2],
                            dataset_name=a.dataset)
                student = get_adapter(a.student, device=device, **acfg)
                kw = dict(lam_kd=lk, lam_feat=lf, temperature=a.temperature,
                          epochs=scfg.get('epochs', 50), lr=scfg.get('lr', 1e-3),
                          weight_decay=scfg.get('weight_decay', 0.01),
                          batch_size=scfg.get('batch_size', 16))
                if mode.startswith('dkd'):
                    # DKD: NCKD always on; TCKD gated by teacher correctness.
                    wt = correct if mode == 'dkd_tmask' else np.ones_like(correct)
                    kw.update(dkd=True, w_target=wt,
                              w_nontarget=np.ones_like(correct),
                              dkd_alpha=a.dkd_alpha, dkd_beta=a.dkd_beta)
                else:
                    kw['sample_weight'] = weights[mode]
                preds = distill_student(
                    student, nc, X_tr, y_tr, tch['feats'], tch['logits'], X_te, **kw)
                m = metrics.evaluate(y_te, preds)
                rows.append(dict(dataset=a.dataset, subject=subj, seed=seed,
                                 condition=cond, acc=m['acc'], kappa=m['kappa'],
                                 lam_kd=lk, lam_feat=lf, weight_mode=mode))
                print(f"S{subj} seed{seed} {cond} | acc={m['acc']} "
                      f"kappa={m['kappa']}", flush=True)

    if not rows:
        print('No teacher artifacts found.'); return
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f'\nWrote {out_csv}')
    print(df.groupby('condition')[['acc', 'kappa']].mean().round(3))


if __name__ == '__main__':
    main()
