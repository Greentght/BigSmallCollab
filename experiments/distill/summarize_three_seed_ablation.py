"""Aggregate matched seed-666/667/668 warm-up, MI, and prototype ablations."""
from __future__ import annotations
import csv, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from experiments.storage import external_path, require_external_output, resolve_local_file

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (666, 667, 668)
TASKS = ('BNCI2014001', 'BNCI2014001-4', 'BNCI2014004', 'BNCI2015001', 'AlexMI')
CONTRASTS = {
    '10-epoch warm-up': ('WARMUP10', 'KD_ALL'),
    'KD + MI': ('KD_MI_ALL', 'KD_ALL'),
    'prototype gate': ('KD_PROTO', 'KD_ALL'),
}
SOURCES = {
    'BNCI2014001 pair': ('BNCI2014001', 'BNCI2014001-4'),
    'BNCI2014004': ('BNCI2014004',),
    'BNCI2015001': ('BNCI2015001',),
    'AlexMI': ('AlexMI',),
}
OUT = Path('/data1/llx/BigSmallcollab/results/distill/three_seed_ablation')
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    rows = list(rows)
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with Path(path).open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def seed_root(seed):
    if seed == 666: return Path('/data1/llx/BigSmallcollab/results/distill/seed666_three_ablation')
    return ROOT / f'/data1/llx/BigSmallcollab/results/distill/three_seed_ablation_seed{seed}'


all_rows=[]; teacher_rows=[]; perf_by_seed={}; delta_by_seed={}; source_arrays={}
for seed in SEEDS:
    folder=seed_root(seed)
    rows=read_csv(folder/'results_per_run.csv')
    assert len(rows)==235, (seed,len(rows))
    assert all(r['status']=='complete' and int(r['epochs_completed'])==100 for r in rows)
    counts=Counter(r['condition'] for r in rows)
    assert counts==Counter({'BASE_CE':47,'KD_ALL':47,'WARMUP10':47,'KD_MI_ALL':47,'KD_PROTO':47}), (seed,counts)
    units=defaultdict(list)
    for r in rows: units[(r['dataset'],int(r['subject_index']))].append(r)
    assert len(units)==47 and all(len(v)==5 for v in units.values())
    for key,group in units.items():
        for field in ('train_uid_hash','test_uid_hash','initial_state_hash','batch_order_hash','batch_order_hashes'):
            assert len({r[field] for r in group})==1, (seed,key,field)
    prov=json.loads((resolve_local_file(folder/'execution_provenance.json')).read_text())
    assert prov['status']=='complete' and prov['new_runs']==235
    assert all(x['hashes_equal'] for x in prov['new_triplet_hashes'])
    for r in rows:
        rr=dict(r); rr['seed']=seed; all_rows.append(rr)
    teachers=read_csv(folder/'teacher_baselines_per_subject.csv')
    assert len(teachers)==47 and all(r['uid_alignment']=='pass' for r in teachers)
    teacher_rows.extend([{**r,'seed':seed} for r in teachers])
    perf=defaultdict(list)
    for r in rows:
        perf[(r['dataset'],r['condition'])].append(r)
    for (ds,cond),group in perf.items():
        perf_by_seed[(seed,ds,cond)]={m:float(np.mean([float(r[col]) for r in group])) for m,col in (
            ('balanced_accuracy','test_balanced_accuracy'),('accuracy','test_accuracy'),('kappa','test_kappa'))}
    for ds in TASKS:
        for label,(method,base) in CONTRASTS.items():
            for subject in range({'BNCI2014001':9,'BNCI2014001-4':9,'BNCI2014004':9,'BNCI2015001':12,'AlexMI':8}[ds]):
                a=next(r for r in rows if r['dataset']==ds and int(r['subject_index'])==subject and r['condition']==method)
                b=next(r for r in rows if r['dataset']==ds and int(r['subject_index'])==subject and r['condition']==base)
                delta_by_seed[(seed,label,ds,subject)]=float(a['test_balanced_accuracy'])-float(b['test_balanced_accuracy'])
    source_arrays[seed]={}
    for label in CONTRASTS:
        source_arrays[seed][label]={}
        for source,datasets in SOURCES.items():
            if len(datasets)==1:
                ds=datasets[0]
                source_arrays[seed][label][source]={s:delta_by_seed[(seed,label,ds,s)] for s in range({'BNCI2014004':9,'BNCI2015001':12,'AlexMI':8}[ds])}
            else:
                common=set(range(9))
                source_arrays[seed][label][source]={s:np.mean([delta_by_seed[(seed,label,ds,s)] for ds in datasets]) for s in common}

# Validate that the subject-derived seed macro equals each per-seed recorded macro.
for seed in SEEDS:
    recorded={r['comparison']:float(r['equal_weight_source_macro_delta_ba']) for r in read_csv(seed_root(seed)/'source_macro_comparisons.csv')}
    for label,(method,base) in CONTRASTS.items():
        comparison=f'{method}-{base}'
        derived=np.mean([np.mean(list(source_arrays[seed][label][source].values())) for source in SOURCES])
        assert abs(derived-recorded[comparison]) < 1e-9, (seed,label,derived,recorded[comparison])

write_csv(OUT/'all_seeds_results_per_run.csv',all_rows)
write_csv(OUT/'all_seeds_teacher_baselines_per_subject.csv',teacher_rows)

macro_rows=[]; source_summary=[]; task_summary=[]
rng=np.random.RandomState(20260930); B=20000
for label in CONTRASTS:
    per_seed=[]
    for seed in SEEDS:
        vals={source:float(np.mean(list(source_arrays[seed][label][source].values()))) for source in SOURCES}
        macro=float(np.mean(list(vals.values())))
        per_seed.append(macro)
        macro_rows.append({'contrast':label,'seed':seed,'four_source_macro_delta_ba':macro,
                           'positive_sources':sum(v>1e-12 for v in vals.values()),
                           'negative_sources':sum(v< -1e-12 for v in vals.values()),
                           'tied_sources':sum(abs(v)<=1e-12 for v in vals.values()),
                           'source_deltas':json.dumps(vals,sort_keys=True)})
    boot=np.empty(B,dtype=float)
    for i in range(B):
        sampled_seeds=rng.choice(SEEDS,size=len(SEEDS),replace=True)
        seed_macros=[]
        for seed in sampled_seeds:
            source_means=[]
            for source in SOURCES:
                values=np.asarray(list(source_arrays[seed][label][source].values()),dtype=float)
                sampled=values[rng.randint(0,len(values),size=len(values))]
                source_means.append(float(sampled.mean()))
            seed_macros.append(float(np.mean(source_means)))
        boot[i]=float(np.mean(seed_macros))
    mean=float(np.mean(per_seed)); sd=float(np.std(per_seed,ddof=1))
    macro_rows.append({'contrast':label,'seed':'MEAN','four_source_macro_delta_ba':mean,
                       'seed_sd':sd,'positive_seed_count':sum(v>1e-12 for v in per_seed),
                       'positive_sources':'','negative_sources':'','tied_sources':'',
                       'hierarchical_bootstrap_ci95_low':float(np.percentile(boot,2.5)),
                       'hierarchical_bootstrap_ci95_high':float(np.percentile(boot,97.5)),
                       'source_deltas':''})
    for source in SOURCES:
        vals=[float(np.mean(list(source_arrays[seed][label][source].values()))) for seed in SEEDS]
        source_summary.append({'contrast':label,'source':source,
                               **{f'seed{seed}_delta':v for seed,v in zip(SEEDS,vals)},
                               'mean_across_seeds':float(np.mean(vals)),
                               'sd_across_seeds':float(np.std(vals,ddof=1)),
                               'positive_seed_count':sum(v>1e-12 for v in vals)})
    for ds in TASKS:
        vals=[]
        for seed in SEEDS:
            n_subject={'BNCI2014001':9,'BNCI2014001-4':9,'BNCI2014004':9,'BNCI2015001':12,'AlexMI':8}[ds]
            vals.append(float(np.mean([delta_by_seed[(seed,label,ds,s)] for s in range(n_subject)])))
        task_summary.append({'contrast':label,'task':ds,
                             **{f'seed{seed}_delta':v for seed,v in zip(SEEDS,vals)},
                             'mean_across_seeds':float(np.mean(vals)),
                             'sd_across_seeds':float(np.std(vals,ddof=1)),
                             'positive_seed_count':sum(v>1e-12 for v in vals)})

write_csv(OUT/'four_source_macro_by_seed.csv',macro_rows)
write_csv(OUT/'source_effects_across_seeds.csv',source_summary)
write_csv(OUT/'task_effects_across_seeds.csv',task_summary)

performance=[]
conditions=('BASE_CE','KD_ALL','WARMUP10','KD_MI_ALL','KD_PROTO')
for ds in TASKS:
    for cond in conditions:
        vals=[perf_by_seed[(seed,ds,cond)] for seed in SEEDS]
        performance.append({'dataset':ds,'condition':cond,
            'balanced_accuracy_mean_across_seeds':float(np.mean([x['balanced_accuracy'] for x in vals])),
            'balanced_accuracy_sd_across_seeds':float(np.std([x['balanced_accuracy'] for x in vals],ddof=1)),
            'accuracy_mean_across_seeds':float(np.mean([x['accuracy'] for x in vals])),
            'accuracy_sd_across_seeds':float(np.std([x['accuracy'] for x in vals],ddof=1)),
            'kappa_mean_across_seeds':float(np.mean([x['kappa'] for x in vals])),
            'kappa_sd_across_seeds':float(np.std([x['kappa'] for x in vals],ddof=1)),
            **{f'seed{seed}_balanced_accuracy':perf_by_seed[(seed,ds,cond)]['balanced_accuracy'] for seed in SEEDS}})
for ds in TASKS:
    vals=[]
    for seed in SEEDS:
        rr=[r for r in teacher_rows if int(r['seed'])==seed and r['dataset']==ds]
        vals.append(float(np.mean([float(r['test_balanced_accuracy']) for r in rr])))
    performance.append({'dataset':ds,'condition':'MIRepNet_teacher',
        'balanced_accuracy_mean_across_seeds':float(np.mean(vals)),
        'balanced_accuracy_sd_across_seeds':float(np.std(vals,ddof=1)),
        **{f'seed{seed}_balanced_accuracy':v for seed,v in zip(SEEDS,vals)}})
write_csv(OUT/'performance_by_task_across_seeds.csv',performance)

lines=['# MIRepNet → IFNet three-seed ablation summary','',
       '- Seeds: 666, 667, 668; five tasks; each seed has 47 task-subject cells and five student conditions (BASE_CE, KD_ALL, WARMUP10, KD_MI_ALL, KD_PROTO).',
       '- Total: 705 complete student runs, each 100 epochs. Seed 666 is the completed prior matched run; seeds 667 and 668 were newly run (470 runs).',
       '- All runs were checked within each seed for equal train/test UIDs, IFNet initial-state hash, and minibatch-order hash across the five conditions. Teacher test predictions were UID/label aligned and used only as a reference.',
       '- Deltas are within-seed paired differences in final test balanced accuracy. The four-source macro averages the two BNCI2014001 tasks within the same subject first, then gives that combined source and the other three sources equal weight.',
       '- The hierarchical 95% interval resamples seeds, then subjects within each fixed source; it is descriptive with only three seeds.','',
       '## Primary effects: four-source macro ΔBA (percentage points)','',
       '| contrast | seed 666 | seed 667 | seed 668 | mean ± SD across seeds | positive seeds | hierarchical 95% interval |',
       '|---|---:|---:|---:|---:|---:|---:|']
for label in CONTRASTS:
    rr=[r for r in macro_rows if r['contrast']==label]
    byseed={int(r['seed']):float(r['four_source_macro_delta_ba']) for r in rr if r['seed']!='MEAN'}
    m=next(r for r in rr if r['seed']=='MEAN')
    lines.append(f"| {label} | {byseed[666]:+.3f} | {byseed[667]:+.3f} | {byseed[668]:+.3f} | {float(m['four_source_macro_delta_ba']):+.3f} ± {float(m['seed_sd']):.3f} | {m['positive_seed_count']}/3 | [{float(m['hierarchical_bootstrap_ci95_low']):+.3f}, {float(m['hierarchical_bootstrap_ci95_high']):+.3f}] |")
lines += ['', '## Per-task mean paired ΔBA across seeds','',
          '| contrast | BNCI2014001 | BNCI2014001-4 | BNCI2014004 | BNCI2015001 | AlexMI |',
          '|---|---:|---:|---:|---:|---:|']
for label in CONTRASTS:
    d={r['task']:r for r in task_summary if r['contrast']==label}
    lines.append('| '+label+' | '+' | '.join(f"{float(d[t]['mean_across_seeds']):+.3f} ({d[t]['positive_seed_count']}/3)" for t in TASKS)+' |')
lines += ['', '## Interpretation','','Positive seeds indicate how many of the three observed seed-level macro deltas exceed zero; this is not a significance test. A direction recurring in 3/3 seeds is a repeatability signal, while the interval and effect size show its uncertainty and magnitude. The warm-up comparison is still WARMUP10 (90 KD epochs) versus KD_ALL (100 KD epochs), so it includes a difference in KD exposure. Prototype gate effectiveness is assessed as currently implemented; no feature-alignment loss or warm-up interaction was included.','',
          '## Files','','- `all_seeds_results_per_run.csv`: 705 subject-condition results.','- `all_seeds_teacher_baselines_per_subject.csv`: teacher reference scores with aligned held-out UIDs.','- `four_source_macro_by_seed.csv`: per-seed macro effects and hierarchical summary.','- `source_effects_across_seeds.csv` and `task_effects_across_seeds.csv`: source and task level detail.','- `performance_by_task_across_seeds.csv`: CE/KD/ablation/teacher performance means and seed-level variation.','']
(OUT/'report.md').write_text('\n'.join(lines))
print('validated 705 runs across three seeds; wrote',OUT)
for label in CONTRASTS:
    rr=[r for r in macro_rows if r['contrast']==label and r['seed']!='MEAN']
    print(label,[round(float(r['four_source_macro_delta_ba']),3) for r in rr],
          'mean',round(float(next(r for r in macro_rows if r['contrast']==label and r['seed']=='MEAN')['four_source_macro_delta_ba']),3))
