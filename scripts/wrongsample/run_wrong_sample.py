"""E0-E6: Wrong-sample utilisation ablation (fixed teacher, student-only update).

All conditions share: teacher frozen (cached logits), same student init seed,
same CE on ALL samples. Only the handling of teacher-WRONG samples differs.

  E0_CE              : CE only, no KD anywhere
  E1_KD_all          : KD on all samples (vanilla KD)
  E2_KD_correct      : KD only on teacher-correct samples      [anchor]
  E3_WrongCE         : E2 + upweighted CE on wrong
  E4_FlipKD          : E2 + flip(y_true<->y_pred) on wrong     [raw flip baseline]
  E5_CalibRevision   : E2 + minimal alpha correction on wrong
  ---- diagnostic variants (added for information-content test) ----
  E4_ShuffledFlip    : E4 but wrong-sample flip targets shuffled among wrong samples
                       (breaks sample<->confidence correspondence)
  E4_ConstantFlip    : E4 but all wrong samples share the mean flip target
                       (tests whether any soft label = plain label smoothing)
  E4_GatedFlip_q25   : E4 only on lowest-25% margin wrong samples (barely wrong)
  E4_GatedFlip_q50   : E4 only on lowest-50% margin wrong samples
  E6_PartialKD       : wrong samples: zero out top-1 wrong class, renorm remaining
                       (preserves C-1 class relations; only meaningful for C≥3)

Diagnostic logic:
  Original > Shuffled ≈ Constant → sample-level confidence carries information
  All three ≈ each other         → E4 gain is just label-smoothing noise
  All three < E2                 → binary wrong logits have no extra info → close line
  Gated > E4_full                → low-margin errors have useful boundary info
  E6_PartialKD > E2              → class-relation structure in wrong logits is useful (4-cls)

Wrong-sample KD weight: lam_err in {0.1, 0.25, 0.5}
Margin (GatedFlip): m_i = p_pred - p_y; τ = q-th percentile over wrong samples (no val leakage)

    conda run -n mirepnet python -u scripts/wrongsample/run_wrong_sample.py \\
        --dataset BNCI2015001 --teacher mirepnet --student ifnet --gpu 2 \\
        --methods E2_KD_correct,E4_FlipKD,E4_ShuffledFlip,E4_ConstantFlip,\\
E4_GatedFlip_q25,E4_GatedFlip_q50 --tag v2_diag
    conda run -n mirepnet python -u scripts/wrongsample/run_wrong_sample.py \\
        --dataset BNCI2014001-4 --teacher mirepnet --student ifnet --gpu 2 \\
        --methods E2_KD_correct,E4_FlipKD,E4_ShuffledFlip,E4_ConstantFlip,\\
E4_GatedFlip_q25,E4_GatedFlip_q50,E6_PartialKD --tag v2_diag
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import config
import data
from collab import artifacts
from collab.distill import _set_seed
from eval import metrics
from models import get_adapter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS = os.path.join(ROOT, 'results', 'metrics')


# ---------------------------------------------------------------------------
# Temperature calibration (training-side val only, never touches test set)
# ---------------------------------------------------------------------------

def _calibrate_temperature(logits_np, y_np, device, val_frac=0.2, seed=0):
    """Fit scalar temperature T on a held-out val_frac of training data.

    Minimises NLL of softmax(logits / T) on the held-out set via grid search
    over T in [0.2, 5.0]. Returns T as a float."""
    n = len(y_np)
    idx = np.arange(n)
    _, val_idx = train_test_split(idx, test_size=val_frac, random_state=seed,
                                  stratify=y_np)
    lv = torch.as_tensor(logits_np[val_idx], dtype=torch.float32).to(device)
    yv = torch.as_tensor(y_np[val_idx],    dtype=torch.long).to(device)
    best_T, best_nll = 1.0, float('inf')
    for t in np.linspace(0.2, 5.0, 49):  # 49 points in [0.2, 5.0]
        t_f = float(t)
        nll = F.cross_entropy(lv / t_f, yv).item()
        if nll < best_nll:
            best_nll = nll; best_T = t_f
    return best_T


# ---------------------------------------------------------------------------
# Per-condition soft-target and per-sample weight computation
# ---------------------------------------------------------------------------

def _prepare_targets(lt_np, y_np, T, method, lam_err, device, cal_T=None, rng_seed=0):
    """Return (kd_target, w_kd, ce_weight) tensors for the full training set.

    kd_target : (N, C) probabilities used as the KD soft label
    w_kd      : (N,)   per-sample KD weight (0 = skip, 1 = full, lam_err = down-weighted)
    ce_weight : (N,)   per-sample CE weight (1 normally, 1+lam_err for E3 wrong)
    rng_seed  : seed for stochastic diagnostics (ShuffledFlip)
    """
    lt = torch.as_tensor(lt_np, dtype=torch.float32).to(device)
    yt = torch.as_tensor(y_np,  dtype=torch.long).to(device)
    N, C = lt.shape

    tc = (lt.argmax(1) == yt)  # teacher-correct (N,)
    tw = ~tc                    # teacher-wrong

    # ---- E0: CE only ---------------------------------------------------------
    if method == 'E0_CE':
        return (torch.zeros(N, C, device=device),
                torch.zeros(N, device=device),
                torch.ones(N, device=device))

    # ---- base soft labels (standard T scaling) --------------------------------
    kd_base = F.softmax(lt / T, dim=1)   # (N, C), teacher at temperature T

    if method == 'E1_KD_all':
        return kd_base, torch.ones(N, device=device), torch.ones(N, device=device)

    if method == 'E2_KD_correct':
        return kd_base, tc.float(), torch.ones(N, device=device)

    if method == 'E3_WrongCE':
        ce_w = torch.ones(N, device=device) + lam_err * tw.float()
        return kd_base, tc.float(), ce_w

    # ---- E4: Raw Flip-KD (shared helper) --------------------------------------
    def _flip_logits(lt_in):
        """Return lt_flip where wrong samples have y_true <-> y_pred swapped."""
        lt_f = lt_in.clone()
        wrong_idx = tw.nonzero(as_tuple=True)[0]
        if len(wrong_idx) > 0:
            pred_cls = lt_in.argmax(1)[wrong_idx]
            true_cls = yt[wrong_idx]
            tmp = lt_f[wrong_idx, true_cls].clone()
            lt_f[wrong_idx, true_cls] = lt_f[wrong_idx, pred_cls]
            lt_f[wrong_idx, pred_cls] = tmp
        return lt_f, tw.nonzero(as_tuple=True)[0]

    if method == 'E4_FlipKD':
        lt_flip, _ = _flip_logits(lt)
        kd_flip = F.softmax(lt_flip / T, dim=1)
        w_kd = tc.float() + lam_err * tw.float()
        return kd_flip, w_kd, torch.ones(N, device=device)

    # ---- E4_ShuffledFlip: flip targets shuffled among wrong samples -----------
    # Tests: does sample<->confidence correspondence matter?
    # If Original ≈ Shuffled, the specific confidence value is not informative.
    if method == 'E4_ShuffledFlip':
        lt_flip, wrong_idx = _flip_logits(lt)
        kd_flip = F.softmax(lt_flip / T, dim=1)
        if len(wrong_idx) > 1:
            rng = np.random.RandomState(rng_seed)
            perm = rng.permutation(len(wrong_idx))
            kd_flip[wrong_idx] = kd_flip[wrong_idx[torch.from_numpy(perm).to(device)]]
        w_kd = tc.float() + lam_err * tw.float()
        return kd_flip, w_kd, torch.ones(N, device=device)

    # ---- E4_ConstantFlip: all wrong samples share mean flip target ------------
    # Tests: is E4 just equivalent to plain label smoothing?
    # If Original ≈ Constant, the information is only "wrong samples get a soft
    # label" (label-smoothing effect), not the specific confidence structure.
    if method == 'E4_ConstantFlip':
        lt_flip, wrong_idx = _flip_logits(lt)
        kd_flip = F.softmax(lt_flip / T, dim=1)
        if len(wrong_idx) > 0:
            mean_target = kd_flip[wrong_idx].mean(0, keepdim=True)
            kd_flip[wrong_idx] = mean_target.expand(len(wrong_idx), -1)
        w_kd = tc.float() + lam_err * tw.float()
        return kd_flip, w_kd, torch.ones(N, device=device)

    # ---- E4_GatedFlip_q25 / _q50: only flip lowest-margin wrong samples ------
    # margin m_i = p_pred - p_y (teacher confidence of being wrong).
    # Low margin → barely wrong, may encode decision-boundary info.
    # τ = q-th percentile of margins on training wrong samples (no val leakage).
    if method in ('E4_GatedFlip_q25', 'E4_GatedFlip_q50'):
        q = 0.25 if method.endswith('q25') else 0.50
        pt_raw = F.softmax(lt / T, dim=1)
        lt_flip, wrong_idx = _flip_logits(lt)
        kd_gated = kd_base.clone()         # start with standard targets
        gate_wrong = torch.zeros(N, dtype=torch.bool, device=device)
        if len(wrong_idx) > 0:
            pred_cls_w = lt.argmax(1)[wrong_idx]
            true_cls_w = yt[wrong_idx]
            margin = pt_raw[wrong_idx, pred_cls_w] - pt_raw[wrong_idx, true_cls_w]
            tau = torch.quantile(margin.float(), q)
            sel_mask = margin <= tau
            selected = wrong_idx[sel_mask]
            kd_gated[selected] = F.softmax(lt_flip[selected] / T, dim=1)
            gate_wrong[selected] = True
        w_kd = tc.float() + lam_err * gate_wrong.float()
        return kd_gated, w_kd, torch.ones(N, device=device)

    # ---- E5: Calibrated Revision (minimal alpha correction) -------------------
    if method == 'E5_CalibRevision':
        T_use = cal_T if cal_T is not None else T
        pt = F.softmax(lt / T_use, dim=1)

        a = pt[torch.arange(N), yt]
        pt_no_y = pt.clone()
        pt_no_y[torch.arange(N), yt] = -1e9
        b = pt_no_y.max(1).values

        eps = 1e-6
        alpha = torch.clamp((b - a + eps) / (1 - a + b + eps), 0.0, 1.0)
        alpha[tc] = 0.0

        one_hot = torch.zeros_like(pt)
        one_hot[torch.arange(N), yt] = 1.0
        q_target = (1.0 - alpha.unsqueeze(1)) * pt + alpha.unsqueeze(1) * one_hot

        w_kd = tc.float() + lam_err * tw.float()
        return q_target, w_kd, torch.ones(N, device=device)

    # ---- E6_PartialKD: zero out top-1 wrong class, renorm remaining ----------
    # Meaningful only for C≥3 (4-class MI). For wrong samples, the teacher
    # class prediction is likely wrong but its beliefs about the other C-1
    # classes may still preserve useful ordinal relationships.
    if method == 'E6_PartialKD':
        kd_partial = kd_base.clone()
        wrong_idx = tw.nonzero(as_tuple=True)[0]
        if len(wrong_idx) > 0:
            pred_cls_w = lt.argmax(1)[wrong_idx]
            kd_partial[wrong_idx, pred_cls_w] = 0.0
            row_sums = kd_partial[wrong_idx].sum(1, keepdim=True).clamp(min=1e-8)
            kd_partial[wrong_idx] = kd_partial[wrong_idx] / row_sums
        w_kd = tc.float() + lam_err * tw.float()
        return kd_partial, w_kd, torch.ones(N, device=device)

    raise ValueError(f'Unknown method: {method}')


# ---------------------------------------------------------------------------
# Student training loop
# ---------------------------------------------------------------------------

def _train_and_eval(ad, nc, Xp_tr, ytr_t, kd_target, w_kd, ce_weight,
                    Xp_te, y_te, lam_kd, T, epochs, lr, wd, bs, seed, device):
    """One student training run. Returns test acc (%)."""
    _set_seed(seed)
    model = ad.build(nc)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    dataset = TensorDataset(Xp_tr.cpu(), ytr_t.cpu(),
                            kd_target.cpu(), w_kd.cpu(), ce_weight.cpu())
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb, kd_b, wkd_b, cew_b in loader:
            xb, yb, kd_b = xb.to(device), yb.to(device), kd_b.to(device)
            wkd_b, cew_b = wkd_b.to(device), cew_b.to(device)

            _, logits = ad.forward(model, xb)
            # CE (per-sample weighted)
            loss = (cew_b * F.cross_entropy(logits, yb, reduction='none')).mean()
            # KD (weighted sum normalised by sum of weights)
            wsum = wkd_b.sum()
            if lam_kd > 0 and wsum > 0:
                kd_loss = F.kl_div(F.log_softmax(logits / T, dim=1),
                                   kd_b,  # already probabilities
                                   reduction='none').sum(1)
                loss = loss + lam_kd * (T * T) * (wkd_b * kd_loss).sum() / wsum

            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    # eval
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(Xp_te), 64):
            _, lg = ad.forward(model, Xp_te[i:i+64].to(device))
            preds.append(lg.argmax(1).cpu().numpy())
    return (np.concatenate(preds) == y_te).mean() * 100


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METHODS = ['E0_CE', 'E1_KD_all', 'E2_KD_correct',
           'E3_WrongCE', 'E4_FlipKD', 'E5_CalibRevision',
           'E4_ShuffledFlip', 'E4_ConstantFlip',
           'E4_GatedFlip_q25', 'E4_GatedFlip_q50',
           'E6_PartialKD']

# methods that do not sweep lam_err (single run at lam_err=0)
_NO_LAM_METHODS = {'E0_CE', 'E1_KD_all', 'E2_KD_correct'}
# methods that sweep lam_err (E3+ including diagnostic variants)
_LAM_METHODS = set(METHODS) - _NO_LAM_METHODS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset',  default='BNCI2015001')
    ap.add_argument('--teacher',  default='cbramod_native')
    ap.add_argument('--student',  default='ifnet')
    ap.add_argument('--methods',  default=None,
                    help='comma subset of METHODS (default: all)')
    ap.add_argument('--lam_kd',   type=float, default=0.5)
    ap.add_argument('--lam_errs', type=float, nargs='+', default=[0.1, 0.25, 0.5],
                    help='wrong-sample weight grid for E3/E4/E5')
    ap.add_argument('--temperature', type=float, default=2.0)
    ap.add_argument('--seeds',    type=int, nargs='+', default=None)
    ap.add_argument('--subjects', type=int, nargs='+', default=None)
    ap.add_argument('--gpu',      type=int, default=None)
    ap.add_argument('--tag',      default='v1')
    a = ap.parse_args()

    device = (f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available()
              else 'cpu')
    dcfg = config.load_dataset_config(a.dataset)
    scfg = config.load_model_config(a.student)
    nc = dcfg['num_classes']
    val_split = dcfg.get('val_split', 0.3)
    seeds = a.seeds or dcfg['seeds']
    n_sub = dcfg['num_subjects']
    subjects = a.subjects if a.subjects is not None else list(range(n_sub))
    methods = a.methods.split(',') if a.methods else METHODS

    # E0-E2 have no lam_err; E3/E4/E5 sweep lam_errs
    os.makedirs(METRICS, exist_ok=True)
    out_csv = os.path.join(METRICS,
                           f'wrong_sample_{a.dataset}_{a.teacher}_{a.student}_{a.tag}.csv')

    import pandas as pd
    rows = []

    for seed in seeds:
        for subj in subjects:
            # load teacher artifacts (cached, frozen)
            try:
                tch = artifacts.load(a.dataset, a.teacher, subj, seed, 'train')
            except FileNotFoundError as e:
                print(f'[miss artifact] S{subj} seed{seed}: {e}'); continue

            X_tr, y_tr, X_te, y_te = data.subject_split(
                a.dataset, subj, val_split=val_split, seed=seed)
            assert np.array_equal(tch['y'], y_tr), \
                f'S{subj} seed{seed}: artifact/split label mismatch'

            # student adapter
            scfg2 = dict(scfg); scfg2.update(in_channels=X_tr.shape[1],
                                               samples=X_tr.shape[2],
                                               dataset_name=a.dataset)
            ad = get_adapter(a.student, device=device, **scfg2)
            Xp_tr = ad.preprocess(X_tr).to(device)
            Xp_te = ad.preprocess(X_te).to(device)
            ytr_t = torch.as_tensor(y_tr, dtype=torch.long).to(device)

            epochs = scfg.get('epochs', 50)
            lr     = scfg.get('lr', 1e-3)
            wd     = scfg.get('weight_decay', 0.01)
            bs     = scfg.get('batch_size', 16)
            T      = a.temperature

            # calibrate temperature for E5 (training-side val, not test)
            cal_T = None
            if 'E5_CalibRevision' in methods:
                cal_T = _calibrate_temperature(tch['logits'], y_tr, device, seed=seed)

            # teacher correctness stats
            tc_mask = (tch['logits'].argmax(1) == y_tr)
            n_correct = tc_mask.sum()
            n_wrong   = (~tc_mask).sum()
            t_acc     = n_correct / len(y_tr) * 100

            def _run(method, lam_err=0.0):
                kd_t, w_kd, ce_w = _prepare_targets(
                    tch['logits'], y_tr, T, method, lam_err, device,
                    cal_T=(cal_T if method == 'E5_CalibRevision' else None),
                    rng_seed=seed)
                acc = _train_and_eval(
                    ad, nc, Xp_tr, ytr_t, kd_t, w_kd, ce_w,
                    Xp_te, y_te, a.lam_kd, T, epochs, lr, wd, bs, seed, device)
                return acc

            # no-lam methods (E0/E1/E2)
            for m in [m for m in methods if m in _NO_LAM_METHODS]:
                acc = _run(m)
                rows.append(dict(dataset=a.dataset, teacher=a.teacher, student=a.student,
                                 seed=seed, subject=subj, method=m, lam_err=0.0,
                                 acc=round(acc, 2), t_acc=round(t_acc, 2),
                                 n_correct=int(n_correct), n_wrong=int(n_wrong),
                                 cal_T=round(cal_T, 3) if cal_T else 1.0))
                print(f'S{subj} seed{seed} {m} | acc={acc:.2f}  t_acc={t_acc:.1f}', flush=True)

            # lam-sweep methods (E3/E4/E5 + diagnostic variants)
            for m in [m for m in methods if m in _LAM_METHODS]:
                for lam_err in a.lam_errs:
                    acc = _run(m, lam_err)
                    rows.append(dict(dataset=a.dataset, teacher=a.teacher, student=a.student,
                                     seed=seed, subject=subj, method=m, lam_err=lam_err,
                                     acc=round(acc, 2), t_acc=round(t_acc, 2),
                                     n_correct=int(n_correct), n_wrong=int(n_wrong),
                                     cal_T=round(cal_T, 3) if cal_T else 1.0))
                    print(f'S{subj} seed{seed} {m} λ={lam_err} | acc={acc:.2f}', flush=True)

            pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f'\nWrote {out_csv}')

    from scipy.stats import wilcoxon
    print('\n=== all methods vs E2 (subject-level Wilcoxon, best lam_err per method) ===')
    for (ds, tch_n, stu), g in df.groupby(['dataset', 'teacher', 'student']):
        gm = g.groupby(['method', 'lam_err', 'subject'])['acc'].mean().reset_index()
        e2 = gm[gm.method == 'E2_KD_correct'].groupby('subject')['acc'].mean()
        if len(e2) == 0:
            continue
        print(f'\n  {ds} {tch_n}->{stu}  (teacher acc={g.t_acc.mean():.1f}%  '
              f'n_wrong={g.n_wrong.mean():.1f}  cal_T={g.cal_T.mean():.2f})')
        for m in sorted(gm.method.unique()):
            sub = gm[gm.method == m]
            if len(sub) == 0: continue
            if m in _NO_LAM_METHODS:
                gx = sub.groupby('subject')['acc'].mean()
            else:
                best_lam = sub.groupby('lam_err')['acc'].mean().idxmax()
                gx = sub[sub.lam_err == best_lam].groupby('subject')['acc'].mean()
                m = f'{m}(λ={best_lam})'
            if len(gx) == 0: continue
            d = gx - e2
            wins = int((d > 0).sum())
            try: _, p = wilcoxon(gx, e2)
            except: p = 1.0
            star = ' **' if p < 0.05 else (' .' if p < 0.10 else '')
            print(f'  {m:<35} Δ={d.mean():+.2f}  wins={wins}/{len(gx)}  p={p:.3f}{star}')


if __name__ == '__main__':
    main()
