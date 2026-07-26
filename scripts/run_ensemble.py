"""Test-time ensemble over cached artifacts (any env — pure arrays).

    python scripts/run_ensemble.py --dataset BNCI2014004 \
        --models mirepnet cbramod labram ifnet adfcnn eegnet \
        --big mirepnet cbramod labram

Loads each model's split='test' artifact per (subject, seed), asserts label
alignment, then evaluates the confidence-gated rule (big side gates the small
side) plus conf-weighted and voting baselines. Writes per-(subject,seed) acc +
kappa to results/metrics/<dataset>_ensemble.csv, and prints per-model solo acc.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from collab import ensemble
import config
from collab import artifacts
from eval import metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='BNCI2014004')
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--big', nargs='+', default=None,
                   help='big side for the gate; default = models with size:big')
    p.add_argument('--subjects', type=int, nargs='+', default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--out_csv', default=None)
    return p.parse_args()


def main():
    a = parse_args()
    dcfg = config.load_dataset_config(a.dataset)
    subjects = a.subjects or list(range(dcfg['num_subjects']))
    seeds = a.seeds or dcfg['seeds']

    big = a.big or [m for m in a.models
                    if config.load_model_config(m).get('size') == 'big']
    smalls = [m for m in a.models if m not in big]
    if not big or not smalls:
        print(f'[warn] big={big} smalls={smalls}: gate needs both sides; '
              f'gate will fall back to whatever is provided.')

    out_csv = a.out_csv or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'metrics', f'{a.dataset}_ensemble.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    rows = []
    for seed in seeds:
        for subj in subjects:
            try:
                per_model, y = artifacts.load_aligned(
                    a.dataset, a.models, subj, seed, 'test')
            except FileNotFoundError as e:
                print(f'[miss] S{subj} seed{seed}: {e}')
                continue

            logits = {m: per_model[m]['logits'] for m in a.models}
            row = dict(dataset=a.dataset, subject=subj, seed=seed)
            for m in a.models:
                row[f'{m}_acc'] = metrics.evaluate(
                    y, metrics.preds_from_logits(logits[m]))['acc']

            big_logits = [logits[m] for m in big] if big else list(logits.values())
            small_logits = [logits[m] for m in smalls] if smalls else big_logits
            p_gate = ensemble.gate(big_logits, small_logits)
            row.update(metrics.evaluate(y, p_gate.argmax(1)))   # acc, kappa = gate
            row['gate_acc'] = row.pop('acc'); row['gate_kappa'] = row.pop('kappa')

            p_cw = ensemble.conf_weighted(list(logits.values()))
            row['confw_acc'] = metrics.evaluate(y, p_cw.argmax(1))['acc']
            pred_vote = ensemble.voting(list(logits.values()))
            row['vote_acc'] = metrics.evaluate(y, pred_vote)['acc']
            rows.append(row)
            print(f"S{subj} seed{seed} | gate acc={row['gate_acc']} "
                  f"kappa={row['gate_kappa']}", flush=True)

    if not rows:
        print('No artifacts found.'); return
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f'\nWrote {out_csv}')
    print('Mean over subjects/seeds:')
    print(df.drop(columns=['subject', 'seed']).select_dtypes('number').mean().round(2))


if __name__ == '__main__':
    main()
