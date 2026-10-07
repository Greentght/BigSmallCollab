"""Validate and summarize the three-seed warm-up + KD + MI combination run."""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from experiments.storage import external_path, require_external_output, resolve_local_file

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (666, 667, 668)
TASKS = ('BNCI2014001', 'BNCI2014001-4', 'BNCI2014004', 'BNCI2015001', 'AlexMI')
SUBJECTS = {'BNCI2014001': 9, 'BNCI2014001-4': 9, 'BNCI2014004': 9,
            'BNCI2015001': 12, 'AlexMI': 8}
ARMS = ('BASE_CE', 'KD_ALL', 'WARMUP10', 'KD_MI_ALL', 'WARMUP10_KD_MI')
ALL_PERFORMANCE_ARMS = ARMS + ('MIRepNet_teacher',)
CONTRASTS = {
    'combo - IFNet CE': ('WARMUP10_KD_MI', 'BASE_CE'),
    'combo - logits KD': ('WARMUP10_KD_MI', 'KD_ALL'),
    'combo - warm-up only': ('WARMUP10_KD_MI', 'WARMUP10'),
    'combo - KD+MI only': ('WARMUP10_KD_MI', 'KD_MI_ALL'),
    'combo - MIRepNet standalone': ('WARMUP10_KD_MI', 'MIRepNet_teacher'),
}
SOURCES = {
    'BNCI2014001 pair': ('BNCI2014001', 'BNCI2014001-4'),
    'BNCI2014004': ('BNCI2014004',),
    'BNCI2015001': ('BNCI2015001',),
    'AlexMI': ('AlexMI',),
}
OUT = Path('/data1/llx/BigSmallcollab/results/distill/warmup_mi_combo_three_seed')
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def seed_root(seed):
    if seed == 666:
        return Path('/data1/llx/BigSmallcollab/results/distill/seed666_three_ablation')
    return ROOT / f'/data1/llx/BigSmallcollab/results/distill/three_seed_ablation_seed{seed}'


def combo_root(seed):
    return ROOT / f'/data1/llx/BigSmallcollab/results/distill/warmup_mi_combo_seed{seed}'


# Validate each run and index baseline/combination results by the matched unit.
lookup = {}
audit = []
for seed in SEEDS:
    combo_rows = read_csv(combo_root(seed) / 'results_per_run.csv')
    assert len(combo_rows) == 47, (seed, 'combo rows', len(combo_rows))
    assert all(r['condition'] == 'WARMUP10_KD_MI' and r['status'] == 'complete'
               and int(r['epochs_completed']) == 100 for r in combo_rows)
    combo_by = {(r['dataset'], int(r['subject_index'])): r for r in combo_rows}
    assert len(combo_by) == 47
    base_rows = read_csv(seed_root(seed) / 'results_per_run.csv')
    assert len(base_rows) == 235, (seed, 'baseline rows', len(base_rows))
    assert all(r['status'] == 'complete' and int(r['epochs_completed']) == 100
               for r in base_rows)
    base_by = {(r['dataset'], int(r['subject_index']), r['condition']): r for r in base_rows}
    assert len(base_by) == 235
    prov = json.loads((resolve_local_file(combo_root(seed) / 'execution_provenance.json')).read_text())
    assert prov['status'] == 'complete' and prov['new_runs'] == 47, (seed, prov)
    for ds in TASKS:
        for subject in range(SUBJECTS[ds]):
            c = combo_by[(ds, subject)]
            for arm in ARMS[:-1]:
                b = base_by[(ds, subject, arm)]
                fields = ('train_uid_hash', 'test_uid_hash', 'split_uid_hash',
                          'initial_state_hash', 'preprocessing_hash',
                          'batch_order_hash', 'batch_order_hashes')
                bad = [field for field in fields if c.get(field) != b.get(field)]
                audit.append({'seed': seed, 'dataset': ds, 'subject_index': subject,
                              'baseline': arm, 'hashes_equal': not bad,
                              'mismatched_fields': ';'.join(bad)})
                if bad:
                    raise AssertionError(f'{seed} {ds} S{subject} vs {arm}: hash mismatch {bad}')
                lookup[(seed, ds, subject, 'WARMUP10_KD_MI')] = c
                lookup[(seed, ds, subject, arm)] = b
    teacher_rows = read_csv(combo_root(seed) / 'teacher_baselines_per_subject.csv')
    assert len(teacher_rows) == 47 and all(r['uid_alignment'] == 'pass' for r in teacher_rows)
    teacher_by = {(r['dataset'], int(r['subject_index'])): r for r in teacher_rows}
    assert len(teacher_by) == 47
    for ds in TASKS:
        for subject in range(SUBJECTS[ds]):
            teacher = teacher_by[(ds, subject)]
            combo = combo_by[(ds, subject)]
            assert teacher['test_uid_hash'] == combo['test_uid_hash'], (seed, ds, subject, 'teacher test UIDs')
            lookup[(seed, ds, subject, 'MIRepNet_teacher')] = teacher
write_csv(OUT / 'paired_hash_audit.csv', audit)

# Keep subject-level paired outcomes, using balanced accuracy as the primary metric.
paired = []
delta = {}
for seed in SEEDS:
    for ds in TASKS:
        for subject in range(SUBJECTS[ds]):
            combo = lookup[(seed, ds, subject, 'WARMUP10_KD_MI')]
            for label, (method, base) in CONTRASTS.items():
                baseline = lookup[(seed, ds, subject, base)]
                dba = float(combo['test_balanced_accuracy']) - float(baseline['test_balanced_accuracy'])
                dacc = float(combo['test_accuracy']) - float(baseline['test_accuracy'])
                dkappa = float(combo['test_kappa']) - float(baseline['test_kappa'])
                delta[(seed, label, ds, subject)] = dba
                paired.append({'seed': seed, 'dataset': ds, 'subject_index': subject,
                               'comparison': label, 'combo_condition': method,
                               'baseline_condition': base,
                               'combo_balanced_accuracy': float(combo['test_balanced_accuracy']),
                               'baseline_balanced_accuracy': float(baseline['test_balanced_accuracy']),
                               'delta_balanced_accuracy': dba,
                               'combo_accuracy': float(combo['test_accuracy']),
                               'baseline_accuracy': float(baseline['test_accuracy']),
                               'delta_accuracy': dacc,
                               'combo_kappa': float(combo['test_kappa']),
                               'baseline_kappa': float(baseline['test_kappa']),
                               'delta_kappa': dkappa})
write_csv(OUT / 'paired_subject_comparisons.csv', paired)

# Equal-weight sources; the two BNCI2014001 views share subjects and count as one source.
source_effects = {}
macro_rows = []
task_rows = []
for label in CONTRASTS:
    source_effects[label] = {}
    for seed in SEEDS:
        per_ds = {}
        for ds in TASKS:
            per_ds[ds] = {s: delta[(seed, label, ds, s)] for s in range(SUBJECTS[ds])}
        pair_ids = set(per_ds['BNCI2014001']) & set(per_ds['BNCI2014001-4'])
        values = {
            'BNCI2014001 pair': {s: np.mean([per_ds['BNCI2014001'][s],
                                            per_ds['BNCI2014001-4'][s]]) for s in pair_ids},
            'BNCI2014004': per_ds['BNCI2014004'],
            'BNCI2015001': per_ds['BNCI2015001'],
            'AlexMI': per_ds['AlexMI'],
        }
        source_effects[label][seed] = values
        means = {source: float(np.mean(list(subject_deltas.values())))
                 for source, subject_deltas in values.items()}
        macro_rows.append({'comparison': label, 'seed': seed,
                           'four_source_macro_delta_ba': float(np.mean(list(means.values()))),
                           'positive_sources': sum(v > 1e-12 for v in means.values()),
                           'negative_sources': sum(v < -1e-12 for v in means.values()),
                           'tied_sources': sum(abs(v) <= 1e-12 for v in means.values()),
                           'source_deltas': json.dumps(means, sort_keys=True)})
        for ds in TASKS:
            vals = np.asarray(list(per_ds[ds].values()), dtype=float)
            task_rows.append({'comparison': label, 'seed': seed, 'dataset': ds,
                              'mean_delta_ba': float(vals.mean()),
                              'positive_subjects': int((vals > 1e-12).sum()),
                              'negative_subjects': int((vals < -1e-12).sum()),
                              'tied_subjects': int((np.abs(vals) <= 1e-12).sum()),
                              'subject_count': len(vals)})

# Bootstrap seeds first, then people within each fixed source.
rng = np.random.RandomState(20260930)
B = 20000
summary = []
for label in CONTRASTS:
    seed_macros = [next(r['four_source_macro_delta_ba'] for r in macro_rows
                        if r['comparison'] == label and r['seed'] == seed)
                   for seed in SEEDS]
    draws = np.empty(B, dtype=float)
    for i in range(B):
        sampled_seeds = rng.choice(SEEDS, size=len(SEEDS), replace=True)
        sampled_macros = []
        for seed in sampled_seeds:
            source_means = []
            for source in SOURCES:
                vals = np.asarray(list(source_effects[label][seed][source].values()), dtype=float)
                source_means.append(float(vals[rng.randint(0, len(vals), size=len(vals))].mean()))
            sampled_macros.append(float(np.mean(source_means)))
        draws[i] = float(np.mean(sampled_macros))
    source_mean = {}
    for source in SOURCES:
        seed_values = [float(np.mean(list(source_effects[label][seed][source].values())))
                       for seed in SEEDS]
        source_mean[source] = float(np.mean(seed_values))
    summary.append({'comparison': label,
                    'four_source_macro_delta_ba_mean': float(np.mean(seed_macros)),
                    'four_source_macro_delta_ba_seed_sd': float(np.std(seed_macros, ddof=1)),
                    'positive_seed_count': int(sum(v > 1e-12 for v in seed_macros)),
                    'seed666_delta': seed_macros[0], 'seed667_delta': seed_macros[1],
                    'seed668_delta': seed_macros[2],
                    'bootstrap_ci95_low': float(np.percentile(draws, 2.5)),
                    'bootstrap_ci95_high': float(np.percentile(draws, 97.5)),
                    'source_mean_deltas': json.dumps(source_mean, sort_keys=True),
                    'bootstrap_seed': 20260930, 'bootstrap_replicates': B})
write_csv(OUT / 'four_source_macro_by_seed.csv', macro_rows)
write_csv(OUT / 'task_deltas_by_seed.csv', task_rows)
write_csv(OUT / 'three_seed_comparison_summary.csv', summary)

# Report absolute test performance by task, with means and SD over seed-level task means.
performance = []
for ds in TASKS:
    for arm in ALL_PERFORMANCE_ARMS:
        seed_means = []
        for seed in SEEDS:
            rows = [lookup[(seed, ds, s, arm)] for s in range(SUBJECTS[ds])]
            seed_means.append({metric: float(np.mean([float(r[col]) for r in rows]))
                               for metric, col in (('balanced_accuracy', 'test_balanced_accuracy'),
                                                   ('accuracy', 'test_accuracy'),
                                                   ('kappa', 'test_kappa'))})
        row = {'dataset': ds, 'condition': arm}
        for metric in ('balanced_accuracy', 'accuracy', 'kappa'):
            vals = [x[metric] for x in seed_means]
            row[f'{metric}_mean_across_seeds'] = float(np.mean(vals))
            row[f'{metric}_sd_across_seeds'] = float(np.std(vals, ddof=1))
            for seed, value in zip(SEEDS, vals):
                row[f'seed{seed}_{metric}'] = value
        performance.append(row)
write_csv(OUT / 'performance_by_task.csv', performance)

lines = [
    '# 10-epoch warm-up + logits KD + MI: three-seed results', '',
    '- MIRepNet → IFNet; seeds 666, 667, 668; five tasks and 47 task-subject units per seed.',
    '- Every combination run completed 100 epochs. Paired initialization, split, preprocessing, and batch-order hashes matched all four corresponding arms for every subject.',
    '- Combination: epochs 1–10 CE only; epochs 11–100 CE + all-sample logits KD (T=2, weight 0.5) + class-probability MI (T=1, weight 0.1).',
    '- Primary metric: paired test balanced-accuracy change in percentage points. Four sources are equally weighted; the two BNCI2014001 views are averaged by shared subject and treated as one source.', '',
    '## Main paired comparisons', '',
    '| comparison | macro Δ BA (pp) | seed SD | positive seeds | 95% hierarchical bootstrap interval |',
    '|---|---:|---:|---:|---:|']
for r in summary:
    lines.append(f"| {r['comparison']} | {r['four_source_macro_delta_ba_mean']:+.3f} | {r['four_source_macro_delta_ba_seed_sd']:.3f} | {r['positive_seed_count']}/3 | [{r['bootstrap_ci95_low']:+.3f}, {r['bootstrap_ci95_high']:+.3f}] |")
lines += ['', '## Absolute balanced accuracy by task', '',
          '| task | IFNet CE | logits KD | warm-up only | KD+MI only | warm-up + KD+MI | MIRepNet standalone |',
          '|---|---:|---:|---:|---:|---:|---:|']
perf = {(r['dataset'], r['condition']): r for r in performance}
for ds in TASKS:
    cells = []
    for arm in ALL_PERFORMANCE_ARMS:
        r = perf[(ds, arm)]
        cells.append(f"{r['balanced_accuracy_mean_across_seeds']:.3f} ± {r['balanced_accuracy_sd_across_seeds']:.3f}")
    lines.append(f'| {ds} | ' + ' | '.join(cells) + ' |')
lines += ['', '## Interpretation', '',
          'The two primary rows compare the combined schedule with IFNet trained by CE and standard all-sample logits KD. The two secondary rows show whether the combination exceeds either component alone. A positive mean with a confidence interval crossing zero is suggestive, not conclusive. Three seeds quantify some training variation but do not establish broad generalization.', '',
          'BNCI2014001 and BNCI2014001-4 share subjects. The bootstrap resamples the three seeds and then subjects within each of four fixed sources; it does not sample new datasets.', '',
          'Files: `three_seed_comparison_summary.csv`, `four_source_macro_by_seed.csv`, `task_deltas_by_seed.csv`, `paired_subject_comparisons.csv`, `performance_by_task.csv`, and `paired_hash_audit.csv`.']
(OUT / 'report.md').write_text('\n'.join(lines) + '\n')
print(f'[complete] 3 seeds × 47 combination runs; hash checks={len(audit)} all matched; report={OUT / "report.md"}')
