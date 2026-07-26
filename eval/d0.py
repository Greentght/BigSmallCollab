"""Phase-D0 diagnostics — the "where is the synergy?" headroom map.

Consumes the standardized per-sample artifacts (``results/artifacts/<dataset>/
<model>[_loso]/<key>_<seed>_<split>.npz``; see ``collab.artifacts``) for a big /
small model pair and, WITHOUT training anything, quantifies how much collaboration
headroom exists and which mechanism family it points to. For every
``(dataset, big, small, protocol)`` cell it computes, on the held-out *test* rows
pooled over folds/subjects × seeds:

  ① error decoupling  — disagreement rate, Yule's Q + phi on the (big✓,small✓)
     2×2 table, and the four-cell fractions a/b/c/d.
  ② oracle upper bounds — single-model accs, oracle-union (≥1 correct = routing/
     fusion ceiling), realistic max-confidence routing + softmax-avg ensemble
     baselines, teacher-win-rate, and the win-rate sliced by small-model
     confidence tercile (does the big model win exactly where the small one is
     unsure? → conditional-computation headroom).
  ③ calibration — ECE (15-bin) for big and small + reliability-curve points.
  ④ representation — linear & RBF CKA between big and small penultimate features.

It then applies the user's five routing rules to tag each cell with a recommended
mechanism family (R+F / F,B,R / projector / K / drop-negative) and writes:
  - results/headroom_map.csv     one row per cell (the leaderboard of headroom)
  - results/d0/<cell>_reliability.png / _confbucket.png   per-cell visuals
  - results/d0/decision_report.md  human-readable routing decision

Run (any env with numpy/pandas; matplotlib optional for the PNGs):
    python -m eval.d0 --protocols within loso
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

from collab import artifacts
import config

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, 'results', 'd0')

# artifact model-dir names per protocol (big models were cached under these)
BIG = {'mirepnet': 'mirepnet', 'cbramod_native': 'cbramod_native'}
SMALL = ('ifnet', 'eegnet', 'adfcnn')


def _dir(model, protocol):
    return model if protocol == 'within' else f'{model}_loso'


# ---------- core numeric helpers ------------------------------------------
def softmax(logits):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(probs, y, n_bins=15):
    """Expected Calibration Error (max-confidence binning) + reliability points."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e, pts = 0.0, []
    N = len(y)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        acc_b, conf_b, w = correct[m].mean(), conf[m].mean(), m.sum() / N
        e += w * abs(acc_b - conf_b)
        pts.append((conf_b, acc_b, int(m.sum())))
    return float(e), pts


def yule_q(a, b, c, d):
    """Yule's Q on the (big✓/✗ × small✓/✗) 2×2 counts. +1 concordant errors,
    0 independent, -1 fully decoupled. Low/negative ⇒ diverse errors ⇒ fusion room."""
    num, den = a * d - b * c, a * d + b * c
    return float(num / den) if den != 0 else 0.0


def phi_coef(a, b, c, d):
    n1, n0 = a + b, c + d
    m1, m0 = a + c, b + d
    den = np.sqrt(n1 * n0 * m1 * m0)
    return float((a * d - b * c) / den) if den != 0 else 0.0


def linear_cka(X, Y):
    """Linear CKA between two feature sets (rows aligned; cols may differ)."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xy = np.linalg.norm(Y.T @ X, 'fro') ** 2
    xx = np.linalg.norm(X.T @ X, 'fro')
    yy = np.linalg.norm(Y.T @ Y, 'fro')
    return float(xy / (xx * yy)) if xx > 0 and yy > 0 else 0.0


def _rbf_gram(X):
    sq = (X ** 2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    d2 = np.maximum(d2, 0)
    med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
    return np.exp(-d2 / (med + 1e-12))


def rbf_cka(X, Y, max_n=800):
    """RBF (median-heuristic) CKA via centered HSIC. Subsampled for O(n^2) cost."""
    n = len(X)
    if n > max_n:  # deterministic stride subsample (rows already aligned)
        idx = np.linspace(0, n - 1, max_n).astype(int)
        X, Y = X[idx], Y[idx]
        n = len(X)
    H = np.eye(n) - np.ones((n, n)) / n
    K, L = _rbf_gram(X), _rbf_gram(Y)
    Kc, Lc = H @ K @ H, H @ L @ H
    hsic = (Kc * Lc).sum()
    nx = np.sqrt((Kc * Kc).sum())
    ny = np.sqrt((Lc * Lc).sum())
    return float(hsic / (nx * ny)) if nx > 0 and ny > 0 else 0.0


# ---------- per-cell computation ------------------------------------------
def collect_cell(dataset, big, small, protocol, seeds, n_keys):
    """Pool per-sample predictions/feats for one cell over folds × seeds.

    Returns None if no aligned artifacts are found (cell not yet exported)."""
    bl, sl, bf, sf, ys, keyid = [], [], [], [], [], []
    bdir, sdir = _dir(big, protocol), _dir(small, protocol)
    found = 0
    for seed in seeds:
        for k in range(n_keys):
            try:
                per, y = artifacts.load_aligned(
                    dataset, [bdir, sdir], k, seed, 'test')
            except (FileNotFoundError, ValueError):
                continue
            bl.append(per[bdir]['logits']); sl.append(per[sdir]['logits'])
            bf.append(per[bdir]['feats']);  sf.append(per[sdir]['feats'])
            ys.append(y); keyid.append(np.full(len(y), k))
            found += 1
    if found == 0:
        return None
    return dict(
        big_logits=np.concatenate(bl), small_logits=np.concatenate(sl),
        big_feats=bf, small_feats=sf,  # keep per-fold for CKA (dims align within fold)
        y=np.concatenate(ys), key=np.concatenate(keyid), n_folds=found)


def analyze_cell(dataset, big, small, protocol, c, expected_folds=None):
    bp = softmax(c['big_logits']); sp = softmax(c['small_logits'])
    y = c['y']
    bpred, spred = bp.argmax(1), sp.argmax(1)
    bc, sc = (bpred == y), (spred == y)

    a = int((bc & sc).sum()); b = int((bc & ~sc).sum())
    cc = int((~bc & sc).sum()); d = int((~bc & ~sc).sum())
    N = a + b + cc + d

    acc_big, acc_small = bc.mean() * 100, sc.mean() * 100
    best_single = max(acc_big, acc_small)
    oracle_union = (bc | sc).mean() * 100          # ≥1 correct — routing/fusion ceiling
    # realistic max-confidence routing (no labels): take more confident model's pred
    pick_big = bp.max(1) >= sp.max(1)
    conf_route_pred = np.where(pick_big, bpred, spred)
    conf_route = (conf_route_pred == y).mean() * 100
    avg_ens = ((bp + sp).argmax(1) == y).mean() * 100    # softmax-avg ensemble

    headroom = oracle_union - best_single           # oracle upper bound (needs labels)
    ens_gain = avg_ens - best_single                 # achievable: plain softmax-avg
    route_gain = conf_route - best_single            # achievable: max-confidence routing
    teacher_win = b / N * 100                        # big uniquely correct
    student_win = cc / N * 100                        # small uniquely correct
    # win-rate sliced by small-model confidence tercile
    sconf = sp.max(1)
    qs = np.quantile(sconf, [1 / 3, 2 / 3])
    buckets = np.digitize(sconf, qs)                 # 0 low,1 mid,2 high
    bucket_stats = {}
    for bk in (0, 1, 2):
        m = buckets == bk
        if m.sum() == 0:
            continue
        bucket_stats[bk] = dict(
            n=int(m.sum()),
            twin=float((bc[m] & ~sc[m]).mean() * 100),
            swin=float((~bc[m] & sc[m]).mean() * 100),
            small_acc=float(sc[m].mean() * 100))

    ece_big, rel_big = ece(bp, y)
    ece_small, rel_small = ece(sp, y)

    # CKA per fold (feature dims align within a fold), averaged
    cka_lin = np.mean([linear_cka(bf, sf)
                       for bf, sf in zip(c['big_feats'], c['small_feats'])])
    cka_rbf = np.mean([rbf_cka(bf, sf)
                       for bf, sf in zip(c['big_feats'], c['small_feats'])])

    return dict(
        dataset=dataset, big=big, small=small, protocol=protocol,
        n_folds=c['n_folds'], expected_folds=expected_folds or c['n_folds'],
        complete=bool(expected_folds is None or c['n_folds'] >= expected_folds), N=N,
        acc_big=round(acc_big, 2), acc_small=round(acc_small, 2),
        best_single=round(best_single, 2),
        oracle_union=round(oracle_union, 2), headroom=round(headroom, 2),
        conf_route=round(conf_route, 2), avg_ensemble=round(avg_ens, 2),
        ens_gain=round(ens_gain, 2), route_gain=round(route_gain, 2),
        disagree=round((bpred != spred).mean() * 100, 2),
        yule_q=round(yule_q(a, b, cc, d), 4), phi=round(phi_coef(a, b, cc, d), 4),
        cell_a_bothok=a, cell_b_bigonly=b, cell_c_smallonly=cc, cell_d_bothwrong=d,
        teacher_win=round(teacher_win, 2), student_win=round(student_win, 2),
        ece_big=round(ece_big, 4), ece_small=round(ece_small, 4),
        cka_linear=round(float(cka_lin), 4), cka_rbf=round(float(cka_rbf), 4),
        big_minus_small=round(acc_big - acc_small, 2),
        _rel_big=rel_big, _rel_small=rel_small, _bucket=bucket_stats)


# ---------- routing decision ----------------------------------------------
def decide(r, headroom_thresh=3.0, cka_low=0.3, gap_big=5.0, ece_bad=0.15):
    tags = []
    if not r.get('complete', True):
        tags.append('PARTIAL(incomplete folds — provisional)')
    best_achievable = max(r['ens_gain'], r['route_gain'])
    if r['headroom'] >= headroom_thresh and r['yule_q'] <= 0.5:
        tags.append('R+F(route/fuse: high oracle headroom + decoupled errors)')
    # the crux: big oracle gap that naive label-free fusion FAILS to capture
    if r['headroom'] >= headroom_thresh and best_achievable < 1.0:
        tags.append('LEARNED-R/F(oracle headroom uncaptured by avg/conf-route '
                    f'[best achievable {best_achievable:+.1f}] → needs learned gate/fusion)')
    if r['headroom'] < 1.0:
        tags.append('DROP(no synergy: oracle≈best single → negative result)')
    if r['teacher_win'] < r['student_win'] or r['ece_big'] > ece_bad:
        tags.append('avoid-softKD(teacher not uniquely-right / poorly calibrated → F,B,R)')
    if r['cka_linear'] < cka_low:
        tags.append('projector(low CKA → featureKD needs projector; prefer logit/relational/route)')
    if r['big_minus_small'] >= gap_big:
        tags.append('K(big≫small here → distillation big→small has room)')
    if not tags:
        tags.append('weak/marginal(monitor)')
    return ' | '.join(tags)


# ---------- plots ----------------------------------------------------------
def plot_cell(r):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return
    tag = f"{r['dataset']}_{r['big']}_{r['small']}_{r['protocol']}"
    # reliability
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    for pts, lab in [(r['_rel_big'], f"big {r['big']} ECE{r['ece_big']}"),
                     (r['_rel_small'], f"small {r['small']} ECE{r['ece_small']}")]:
        if pts:
            xs, ys, _ = zip(*pts)
            ax.plot(xs, ys, 'o-', ms=3, label=lab)
    ax.set_xlabel('confidence'); ax.set_ylabel('accuracy')
    ax.set_title(tag, fontsize=7); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f'{tag}_reliability.png'), dpi=90)
    plt.close(fig)
    # confidence-bucket win rates
    if r['_bucket']:
        fig, ax = plt.subplots(figsize=(4, 3))
        ks = sorted(r['_bucket'])
        ax.plot(ks, [r['_bucket'][k]['twin'] for k in ks], 'o-', label='big-only-correct %')
        ax.plot(ks, [r['_bucket'][k]['swin'] for k in ks], 's-', label='small-only-correct %')
        ax.set_xticks(ks); ax.set_xticklabels(['low', 'mid', 'high'])
        ax.set_xlabel('small-model confidence tercile'); ax.set_ylabel('%')
        ax.set_title(f'{tag} win-by-conf', fontsize=7); ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f'{tag}_confbucket.png'), dpi=90)
        plt.close(fig)


# ---------- driver ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['BNCI2014001-4', 'BNCI2014004'])
    ap.add_argument('--bigs', nargs='+', default=list(BIG))
    ap.add_argument('--smalls', nargs='+', default=list(SMALL))
    ap.add_argument('--protocols', nargs='+', default=['within', 'loso'])
    ap.add_argument('--no_plots', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    for dataset in a.datasets:
        dcfg = config.load_dataset_config(dataset)
        seeds, n_keys = dcfg['seeds'], dcfg['num_subjects']
        for big, small, proto in itertools.product(a.bigs, a.smalls, a.protocols):
            c = collect_cell(dataset, big, small, proto, seeds, n_keys)
            if c is None:
                print(f'[skip-missing] {dataset} {big} {small} {proto}', flush=True)
                continue
            r = analyze_cell(dataset, big, small, proto, c,
                             expected_folds=len(seeds) * n_keys)
            r['decision'] = decide(r)
            if not a.no_plots:
                plot_cell(r)
            rows.append(r)
            print(f"[ok] {dataset} {big}->{small} {proto} "
                  f"accB={r['acc_big']} accS={r['acc_small']} "
                  f"oracle={r['oracle_union']} headroom={r['headroom']} "
                  f"Q={r['yule_q']} CKA={r['cka_linear']} :: {r['decision']}",
                  flush=True)

    if not rows:
        print('No cells with artifacts found.'); return
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')}
                       for r in rows])
    df = df.sort_values(['protocol', 'dataset', 'headroom'],
                        ascending=[True, True, False])
    out_csv = os.path.join(_ROOT, 'results', 'headroom_map.csv')
    df.to_csv(out_csv, index=False)
    print(f'\nWrote {out_csv} ({len(df)} cells)')

    # decision report
    cols = ['dataset', 'big', 'small', 'protocol', 'acc_big', 'acc_small',
            'oracle_union', 'headroom', 'yule_q', 'cka_linear',
            'ece_big', 'teacher_win', 'student_win', 'decision']
    try:
        tbl = df[cols].to_markdown(index=False)
    except ImportError:  # tabulate missing in this env
        tbl = df[cols].to_string(index=False)
    lines = ['# Phase-D0 Headroom Map & Routing Decision\n',
             f'Cells: {len(df)}  |  protocols: {sorted(df.protocol.unique())}  |  '
             f'datasets: {sorted(df.dataset.unique())}\n',
             '## Per-cell headroom (sorted)\n', tbl]
    with open(os.path.join(OUT_DIR, 'decision_report.md'), 'w') as f:
        f.write('\n'.join(lines))
    print(f'Wrote {os.path.join(OUT_DIR, "decision_report.md")}')


if __name__ == '__main__':
    main()
