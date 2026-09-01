"""Config-driven experiment runner.

    python -m experiments.run configs/exp/<name>.yaml [--gpu 0] [--report]

Walks (dataset x unit x seed x condition), consumes cached teacher artifacts,
trains the student via ``collab.distill.distill_student``, and writes a long-form
metrics CSV (``results/metrics/<name>.csv``) that the ``eval`` layer consumes.
With ``--report`` (or a ``report:`` block) it prints the paired-stats report.

Must run in the STUDENT's conda env; the teacher is never built here — its
train-split artifact (feats+logits) must already exist (exported once in the
teacher's env via ``scripts/export/finetune_export.py``). Conditions with no teacher
signal (all methods reduce to lam=0) run even without an artifact.

YAML schema (see configs/exp/*.yaml):
    name:      str                    # -> results/metrics/<name>.csv
    protocol:  fewshot | loso             # within accepted as a legacy alias
    datasets:  [str, ...]
    teacher:   str | null             # cached-artifact model name (null = none)
    student:   str
    seeds:     [int,...] | null        # null -> dataset config default
    units:     [int,...] | null        # subjects (fewshot) / folds (loso); null -> all
    train_percentage: float | null     # fewshot TRAIN fraction; null -> dataset val_split
    distill:   {temperature, epochs, lr, ...}   # shared distill_student kwargs
    conditions: {name: {method, masked, lam_kd, ...}, ...}
    report:    {baseline: str, metrics: [acc, kappa]}   # optional auto-eval
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import yaml

from collab.distill import distill_student
import config
import data
from collab import artifacts
from eval import metrics
from models import get_adapter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


METHOD_REGISTRY = {
    'baseline': dict(lam_kd=0.0, lam_feat=0.0, teacher_correct_only=False),
    'kd': dict(lam_kd=0.5, lam_feat=0.0, teacher_correct_only=False),
    'feat': dict(lam_kd=0.0, lam_feat=0.5, teacher_correct_only=False),
    'combo': dict(lam_kd=0.5, lam_feat=0.5, teacher_correct_only=False),
    'proto': dict(lam_kd=0.0, lam_feat=0.5, feat_proto=True,
                  teacher_correct_only=False),
    'dkd': dict(lam_kd=0.5, lam_feat=0.0, dkd=True,
                teacher_correct_only=False),
}
_CONDITION_SUGAR = ('method', 'masked')


def resolve_condition(cond, defaults):
    """Resolve one YAML condition into ``distill_student`` kwargs.

    Conditions may either use a ``method`` shorthand from ``METHOD_REGISTRY`` or
    spell out the final kwargs directly. The runner-level ``masked`` flag is
    returned separately because it depends on per-cell teacher logits.
    """
    cond = dict(cond or {})
    method = cond.get('method')
    kwargs = dict(defaults)
    if method is not None:
        if method not in METHOD_REGISTRY:
            raise KeyError(f'unknown method {method!r}; known: {list(METHOD_REGISTRY)}')
        kwargs.update(METHOD_REGISTRY[method])
    kwargs.update({k: v for k, v in cond.items() if k not in _CONDITION_SUGAR})
    return kwargs, bool(cond.get('masked', False))


def _teacher_signal(teacher, dataset, unit, seed, y_tr):
    """Load the cached teacher train artifact and assert row alignment. Returns
    ``(feats, logits, correct_mask)`` or ``(None, None, None)`` if no teacher."""
    if not teacher:
        n, = y_tr.shape
        return None, None, None
    tch = artifacts.load(dataset, teacher, unit, seed, 'train')
    if not np.array_equal(tch['y'], y_tr):
        raise ValueError(f'teacher rows misaligned for {dataset} u{unit} seed{seed}')
    mask = (tch['logits'].argmax(1) == y_tr).astype(np.float32)
    return tch['feats'], tch['logits'], mask


def run_dataset(cfg, dataset, device):
    dcfg = config.load_dataset_config(dataset)
    scfg = config.load_model_config(cfg['student'])
    nc = dcfg['num_classes']
    seeds = cfg.get('seeds') or dcfg['seeds']
    units = cfg.get('units') or list(range(dcfg['num_subjects']))
    defaults = dict(cfg.get('distill', {}))
    teacher = cfg.get('teacher')
    student = cfg['student']

    protocol = data.canonical_protocol(cfg['protocol'])
    if 'val_split' in cfg and cfg.get('train_percentage') is not None:
        raise ValueError('pass only one of val_split or train_percentage')
    val_split = cfg.get('val_split', dcfg['val_split'])
    cells = data.iter_cells(protocol, dataset, units, seeds, val_split,
                            dcfg['num_subjects'], cfg.get('train_percentage'))
    rows = []
    for cell in cells:
        # dummy zeros stand in for teacher feats/logits when there's no teacher;
        # every condition then reduces to lam=0 (distill_student ignores them).
        try:
            feat_t, log_t, mask = _teacher_signal(
                teacher, dataset, cell.unit, cell.seed, cell.y_tr)
        except FileNotFoundError as e:
            print(f'[miss teacher] u{cell.unit} seed{cell.seed}: {e}', flush=True)
            continue
        if feat_t is None:
            feat_t = np.zeros((len(cell.y_tr), 1), np.float32)
            log_t = np.zeros((len(cell.y_tr), nc), np.float32)

        for name, cond in cfg['conditions'].items():
            kwargs, masked = resolve_condition(cond, defaults)
            if masked:
                if mask is None:
                    raise ValueError(f'condition {name!r} is masked but no teacher')
                kwargs['sample_weight'] = mask
            acfg = dict(scfg)
            acfg.update(in_channels=cell.X_tr.shape[1], samples=cell.X_tr.shape[2],
                        dataset_name=dataset)
            adapter = get_adapter(student, device=device, **acfg)
            preds = distill_student(
                adapter, nc, cell.X_tr, cell.y_tr, feat_t, log_t, cell.X_te,
                subject_ids=cell.subj_ids, **kwargs)
            m = metrics.evaluate(cell.y_te, preds)
            rows.append(dict(dataset=dataset, subject=cell.unit, seed=cell.seed,
                             condition=f'{student}_{name}',
                             acc=m['acc'], kappa=m['kappa']))
            print(f"[{dataset}] u{cell.unit} seed{cell.seed} {student}_{name} | "
                  f"acc={m['acc']} kappa={m['kappa']}", flush=True)
    return rows


def main():
    # thread cap: shared box discipline (default 4, setdefault keeps pre-set env)
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')
    torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '4')))

    ap = argparse.ArgumentParser(prog='python -m experiments.run')
    ap.add_argument('config', help='path to experiment YAML')
    ap.add_argument('--gpu', type=int, default=None)
    ap.add_argument('--seed', type=int, default=None,
                    help='process-level seed set once before the run loop '
                         '(default: leave global RNG untouched — historical behavior)')
    ap.add_argument('--report', action='store_true',
                    help='print the eval paired-stats report after running')
    a = ap.parse_args()

    if a.seed is not None:
        from collab.seed import set_seed
        set_seed(a.seed)

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    device = (f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available()
              else 'cpu')

    rows = []
    for ds in cfg['datasets']:
        rows += run_dataset(cfg, ds, device)
    if not rows:
        print('No rows produced (missing teacher artifacts?).'); return

    out_csv = os.path.join(os.environ.get('REPRO_OUT',
                                          os.path.join(_ROOT, 'results', 'metrics')),
                           f"{cfg['name']}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f'\nWrote {out_csv}')
    print(df.groupby('condition')[['acc', 'kappa']].mean().round(3))

    rep = cfg.get('report')
    if a.report or rep:
        from eval import report_contrasts
        rep = rep or {}
        student = cfg['student']
        base = rep.get('baseline', f'{student}_base')
        methods_list = [c for c in df.condition.unique() if c != base]
        for ds in cfg['datasets']:
            print(f'\n########## report: {ds} ##########')
            report_contrasts(df[df.dataset == ds], baseline=base,
                             methods=methods_list,
                             metrics=tuple(rep.get('metrics', ['acc', 'kappa'])))


if __name__ == '__main__':
    main()
