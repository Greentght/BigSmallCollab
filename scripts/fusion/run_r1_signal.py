"""R1 signal hunt — leakage-free meta-LOSO learned router on cached artifacts.

Flagship cell (default): MIRepNet x IFNet / BNCI2014001-4 / LOSO (biggest D0 oracle
headroom, most decoupled errors). No base model is retrained — the gate is trained
on cached out-of-sample test-split logits via nested subject-LOSO (see collab.router).

Reports, per base seed and averaged, the router (soft/hard) vs the four references
(best-single, avg-ensemble, conf-route, oracle-union) plus the conditional-compute
wake-rate/accuracy Pareto, and a subject-level paired Wilcoxon of router vs the best
achievable baseline. Appends rows to results/leaderboard.csv.

    conda run -n mirepnet python scripts/fusion/run_r1_signal.py --big mirepnet --small ifnet \
        --dataset BNCI2014001-4 --seeds 666 667 668
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score, f1_score

import config
from collab import artifacts
from collab.router import (apply_variants, gate_alpha, gate_features, train_gate,
                           _softmax_np)

LB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                  'results', 'leaderboard.csv')


def load_cell(dataset, big, small, seed, n_sub):
    """Per-subject out-of-sample test artifacts: dict subj -> (big_logits, small_logits, y)."""
    bdir, sdir = f'{big}_loso', f'{small}_loso'
    out = {}
    for t in range(n_sub):
        try:
            per, y = artifacts.load_aligned(dataset, [bdir, sdir], t, seed, 'test')
        except (FileNotFoundError, ValueError):
            continue
        out[t] = (per[bdir]['logits'], per[sdir]['logits'], y)
    return out


def metrics(pred, y, nc):
    return (float((pred == y).mean() * 100),
            float(f1_score(y, pred, average='macro', labels=list(range(nc)),
                           zero_division=0) * 100),
            float(cohen_kappa_score(y, pred, labels=list(range(nc)))))


def run_seed(cell, nc, seed, device='cpu'):
    """Nested subject-LOSO. Returns per-subject dict of accuracies for each method
    and pooled arrays for the conditional-compute Pareto."""
    subs = sorted(cell)
    per_sub = {}                       # subj -> {method: acc}
    pool_alpha, pool_pb, pool_ps, pool_y = [], [], [], []
    for t in subs:
        bl_te, sl_te, y_te = cell[t]
        pb_te, ps_te = _softmax_np(bl_te), _softmax_np(sl_te)
        # train pool = all other subjects (each out-of-sample for the base models)
        bl_tr = np.concatenate([cell[s][0] for s in subs if s != t])
        sl_tr = np.concatenate([cell[s][1] for s in subs if s != t])
        y_tr = np.concatenate([cell[s][2] for s in subs if s != t])
        gate, mu, sd = train_gate(
            gate_features(bl_tr, sl_tr), _softmax_np(bl_tr), _softmax_np(sl_tr),
            y_tr, seed=seed, device=device)
        alpha = gate_alpha(gate, gate_features(bl_te, sl_te), mu, sd, device)
        v = apply_variants(alpha, pb_te, ps_te, y_te)
        accs = {}
        accs['router_soft'] = (v['soft'] == y_te).mean() * 100
        accs['router_hard'] = (v['hard'] == y_te).mean() * 100
        accs['big'] = (pb_te.argmax(1) == y_te).mean() * 100
        accs['small'] = (ps_te.argmax(1) == y_te).mean() * 100
        accs['avg_ens'] = ((pb_te + ps_te).argmax(1) == y_te).mean() * 100
        pick = pb_te.max(1) >= ps_te.max(1)
        accs['conf_route'] = (np.where(pick, pb_te.argmax(1), ps_te.argmax(1)) == y_te).mean() * 100
        accs['oracle'] = ((pb_te.argmax(1) == y_te) | (ps_te.argmax(1) == y_te)).mean() * 100
        per_sub[t] = accs
        pool_alpha.append(alpha); pool_pb.append(pb_te); pool_ps.append(ps_te); pool_y.append(y_te)
    pareto = cond_compute_pareto(np.concatenate(pool_alpha), np.concatenate(pool_pb),
                                 np.concatenate(pool_ps), np.concatenate(pool_y))
    return per_sub, pareto


def cond_compute_pareto(alpha, pb, ps, y, budgets=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)):
    """Small-default; wake the big model on the top-`w` fraction of samples by alpha
    (gate's preference for big). Returns list of (wake_rate, acc)."""
    order = np.argsort(-alpha)                       # most big-favouring first
    N = len(y); out = []
    small_pred, big_pred = ps.argmax(1), pb.argmax(1)
    for w in budgets:
        k = int(round(w * N))
        wake = np.zeros(N, bool); wake[order[:k]] = True
        pred = np.where(wake, big_pred, small_pred)
        out.append((round(wake.mean(), 3), round(float((pred == y).mean() * 100), 2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--big', default='mirepnet')
    ap.add_argument('--small', default='ifnet')
    ap.add_argument('--dataset', default='BNCI2014001-4')
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--gpu', type=int, default=None)
    a = ap.parse_args()
    import torch
    device = (f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available() else 'cpu')
    dcfg = config.load_dataset_config(a.dataset)
    seeds = a.seeds or dcfg['seeds']
    nc, n_sub = dcfg['num_classes'], dcfg['num_subjects']

    methods = ['router_soft', 'router_hard', 'avg_ens', 'conf_route', 'big', 'small', 'oracle']
    seed_means = {m: [] for m in methods}
    per_sub_over_seeds = {m: {} for m in methods}   # subj -> list of accs (avg later)
    paretos = []
    for seed in seeds:
        cell = load_cell(a.dataset, a.big, a.small, seed, n_sub)
        if not cell:
            print(f'[skip] no artifacts for seed {seed}'); continue
        per_sub, pareto = run_seed(cell, nc, seed, device)
        paretos.append(pareto)
        for m in methods:
            vals = [per_sub[t][m] for t in per_sub]
            seed_means[m].append(np.mean(vals))
            for t in per_sub:
                per_sub_over_seeds[m].setdefault(t, []).append(per_sub[t][m])
        print(f"[seed {seed}] " + "  ".join(
            f"{m}={np.mean([per_sub[t][m] for t in per_sub]):.2f}" for m in methods))

    if not any(seed_means[m] for m in methods):
        print('No data.'); return

    print(f"\n=== R1 signal: {a.big} x {a.small} / {a.dataset} / LOSO "
          f"(mean over {len(seeds)} seeds, {n_sub} subjects) ===")
    for m in methods:
        print(f"  {m:12s} acc={np.mean(seed_means[m]):6.2f}  (± {np.std(seed_means[m]):.2f})")

    # subject-level paired Wilcoxon: router_soft vs best achievable baseline (per subject)
    subj_router = np.array([np.mean(per_sub_over_seeds['router_soft'][t]) for t in sorted(cell)])
    base_best = np.maximum.reduce([
        np.array([np.mean(per_sub_over_seeds[m][t]) for t in sorted(cell)])
        for m in ['big', 'small', 'avg_ens', 'conf_route']])
    diff = subj_router - base_best
    try:
        stat, p = wilcoxon(subj_router, base_best)
    except ValueError:
        stat, p = float('nan'), float('nan')
    print(f"\n  router_soft − best_baseline (per subject): mean Δ={diff.mean():+.2f} "
          f"acc, wins {int((diff > 0).sum())}/{len(diff)}, Wilcoxon p={p:.4f}")
    print(f"  vs oracle ceiling: {np.mean(seed_means['oracle']):.2f}  "
          f"(router captures {100*diff.mean()/max(np.mean(seed_means['oracle'])-base_best.mean(),1e-9):.1f}% of the gap)")
    print("\n  conditional-compute Pareto (wake_rate, acc) averaged over seeds:")
    par = np.array(paretos).mean(0)
    for wr, ac in par:
        print(f"    wake={wr:.2f} -> acc={ac:.2f}")

    # append to leaderboard
    sig = 'yes' if (p == p and p < 0.05 and diff.mean() > 0) else 'no'
    os.makedirs(os.path.dirname(LB), exist_ok=True)
    write_header = not os.path.exists(LB) or os.path.getsize(LB) == 0
    with open(LB, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['dataset', 'protocol', 'method', 'variant', 'big', 'small',
                        'seed', 'acc', 'f1', 'kappa', 'wake_rate', 'params',
                        'vs_best_baseline_p', 'significant'])
        for m in ['router_soft', 'router_hard', 'avg_ens', 'conf_route', 'oracle']:
            w.writerow([a.dataset, 'loso', 'R1', m, a.big, a.small, 'mean3',
                        round(np.mean(seed_means[m]), 2), '', '', '', '',
                        round(p, 4) if m == 'router_soft' else '',
                        sig if m == 'router_soft' else ''])
    print(f"\nAppended R1 rows to {LB}")


if __name__ == '__main__':
    main()
