"""cbramod 超参网格扫描：固定 seed + 配对检验 + bootstrap CI。

设计要点：
  - 数据切分只加载一次（切分不依赖模型超参），每个 combo 在同一批
    (subject, seed) 切分上训练，因此结果是配对的。
  - 每个 variant 与 baseline 做配对比较：均值 Δ、配对 bootstrap 的 95% CI、
    配对 Wilcoxon p 值，并对多个 variant 做 Holm 校正。
  - acc 是 0.5pp 粒度的小整数，配对差含大量 tie，Wilcoxon p 会偏保守；
    因此以 bootstrap CI 为主判据（CI 不含 0 才算可靠）。

用法：
  conda run -n cbramod python experiments/finetune/sweep_cbramod.py \
      --dataset BNCI2014001-4 \
      --variant 'lr=0.0003' --variant 'lr=0.0001' \
      --variant 'l_freq=8,h_freq=32' --variant 'wd=0.3' \
      --gpu 7 --out_csv results/cbramod_sweep.csv
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from scipy import stats

import config
import data
from models import get_adapter


def parse_overrides(s):
    """'k1=v1,k2=v2' -> {k1: v1, k2: v2}，数值自动转 float，否则保留字符串。"""
    d = {}
    if not s:
        return d
    for kv in s.split(','):
        k, v = kv.split('=')
        k, v = k.strip(), v.strip()
        try:
            v = float(v)
        except ValueError:
            pass
        d[k] = v
    return d


def train_eval(model, dataset, protocol, key, seed, overrides, device, num_classes, raw):
    X_tr, y_tr, X_te, y_te = raw[(key, seed)]
    mcfg = config.load_model_config(model, dataset, protocol)
    mcfg.update(overrides)  # 只覆盖要扫的键，其余沿用 yaml
    torch.manual_seed(seed)
    np.random.seed(seed)
    ad = get_adapter(model, device=device, **mcfg,
                     in_channels=X_tr.shape[1], samples=X_tr.shape[2],
                     dataset_name=dataset)
    m = ad.build(num_classes)
    m = ad.finetune(m, X_tr, y_tr, num_classes)
    _, logits = ad.infer(m, X_te)
    acc = float((logits.argmax(1) == y_te).mean() * 100)
    del m
    if device != 'cpu':
        torch.cuda.empty_cache()
    return acc


def paired_bootstrap_ci(diff, n_boot=10000, ci=95, seed=0):
    """配对 bootstrap：按 unit（subject,seed）成对重采样，求均值 Δ 的百分位 CI。"""
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(n_boot)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)  # 成对重采样，保留配对结构
        means[_] = diff[idx].mean()
    lo, hi = np.percentile(means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='cbramod')
    ap.add_argument('--dataset', default='BNCI2014001-4')
    ap.add_argument('--protocol', default='fewshot')
    ap.add_argument('--baseline', default='', help='k=v 逗号分隔；空 = yaml 原样')
    ap.add_argument('--variant', action='append', default=[], help='k=v，可重复')
    ap.add_argument('--seeds', type=int, nargs='+', default=[666, 667, 668])
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out_csv', default='results/cbramod_sweep.csv')
    ap.add_argument('--summary_csv', default=None, help='可选：汇总表 CSV')
    ap.add_argument('--n_boot', type=int, default=10000)
    a = ap.parse_args()

    # 共享 128 核机器，压一下线程
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')
    torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '4')))

    dcfg = config.load_dataset_config(a.dataset)
    num_classes = dcfg['num_classes']
    n_sub = dcfg['num_subjects']
    train_pct = 1.0 - float(dcfg['val_split'])
    device = f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu'

    # 1) 原始切分只加载一次（不依赖模型超参）
    raw = {(k, s): data.subject_split_ordered_fewshot(
                a.dataset, k, train_percentage=train_pct)
           for k in range(n_sub) for s in a.seeds}

    combos = [('baseline', parse_overrides(a.baseline))] + \
             [(v, parse_overrides(v)) for v in a.variant]

    rows, per = [], {name: [] for name, _ in combos}
    print(f'[sweep] {a.model} {a.dataset} {a.protocol} device={device} '
          f'seeds={a.seeds} n_sub={n_sub} train_pct={train_pct:.2f}', flush=True)
    for name, ov in combos:
        for k in range(n_sub):
            for s in a.seeds:
                acc = train_eval(a.model, a.dataset, a.protocol,
                                 k, s, ov, device, num_classes, raw)
                rows.append({'combo': name, 'overrides': str(ov),
                             'subject': k + 1, 'seed': s, 'acc': round(acc, 2)})
                per[name].append(acc)
        print(f'  {name:24s} {np.mean(per[name]):6.2f} ± '
              f'{np.std(per[name], ddof=1):5.2f}', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)
    with open(a.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['combo', 'overrides', 'subject', 'seed', 'acc'])
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {a.out_csv} ({len(rows)} rows)')

    # 2) 配对比较 vs baseline
    base = np.array(per['baseline'])
    variants = [(name, np.array(per[name])) for name, _ in combos if name != 'baseline']
    print(f'\n--- 配对 vs baseline (n={len(base)} units = {n_sub} subj x {len(a.seeds)} seed) ---')
    print(f'{"variant":24s} {"mean±std":14s} {"Δmean":>8s} {"95% CI":>18s} {"Wilcoxon p":>11s}')
    summary = []
    pvals = []
    for name, acc in variants:
        d = acc - base
        lo, hi = paired_bootstrap_ci(d, n_boot=a.n_boot, seed=0)
        _, p = stats.wilcoxon(acc, base)
        pvals.append((name, p))
        summary.append({'variant': name, 'mean': round(acc.mean(), 2),
                        'std': round(acc.std(ddof=1), 2),
                        'delta': round(d.mean(), 2),
                        'ci_lo': round(lo, 2), 'ci_hi': round(hi, 2),
                        'wilcoxon_p': round(p, 4), 'holm_adj': None})
        print(f'{name:24s} {acc.mean():6.2f}±{acc.std(ddof=1):5.2f} '
              f'{d.mean():+8.2f} [{lo:8.2f},{hi:8.2f}] {p:11.4f}')

    # Holm 校正
    pvals.sort(key=lambda x: x[1])
    m = len(pvals)
    holm = {name: min(1.0, p * (m - r)) for r, (name, p) in enumerate(pvals)}
    print('\nHolm 校正后 p：')
    for name, p in pvals:
        print(f'  {name:24s} raw p={p:.4f} -> adj={holm[name]:.4f}')
    for s in summary:
        s['holm_adj'] = round(holm[s['variant']], 4)

    if a.summary_csv:
        with open(a.summary_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['variant', 'mean', 'std', 'delta',
                                              'ci_lo', 'ci_hi', 'wilcoxon_p', 'holm_adj'])
            w.writeheader()
            w.writerows(summary)
        print(f'wrote {a.summary_csv}')

    print('\n判读：95% CI 不含 0 才算相对 baseline 有可靠变化（Wilcoxon p 因 tie 偏保守，仅参考）。')


if __name__ == '__main__':
    main()
