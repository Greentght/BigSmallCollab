"""Inventory every reported 20-epoch run and create separate 100-epoch configs."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path

import yaml

from experiments.bidir.budget100 import (
    ARTIFACT, CONFIGS, FAMILIES, OLD, REPORT, ROOT, digest, epoch_count, sha, write_json,
)


def configure():
    CONFIGS.mkdir(parents=True, exist_ok=True)
    filenames = {v[1] for v in FAMILIES.values() if v[1]}
    for filename in sorted(filenames):
        original = ROOT / 'configs/experiments' / filename
        cfg = yaml.safe_load(original.read_text())
        cfg['source_experiment_id'] = cfg['experiment_id']
        cfg['experiment_id'] += '_100epoch'
        cfg['joint_epochs'] = 100
        if 'epochs' in cfg: cfg['epochs'] = 100
        if 'teacher_training' in cfg:
            for value in cfg['teacher_training'].values():
                if isinstance(value, dict) and 'epochs' in value: value['epochs'] = 100
        for value in cfg.get('pairs', {}).values():
            if 'epochs' in value: value['epochs'] = 100
        family = cfg['source_experiment_id']
        cfg['output_root'] = str(ARTIFACT / family)
        if 'report_root' in cfg: cfg['report_root'] = str(REPORT / family)
        if 'report_name' in cfg: cfg['report_name'] = str(REPORT.relative_to(ROOT / 'results') / family)
        if 'baseline_config' in cfg:
            cfg['baseline_config'] = str((CONFIGS / Path(cfg['baseline_config']).name).relative_to(ROOT))
        if 'baseline_root' in cfg:
            cfg['baseline_root'] = str(ARTIFACT / 'dml_ds_fewshot_pairs_v1/runs/BNCI2014004/cbramod_eegnet/dml_ds')
        if 'existing_004_cbramod_eegnet' in cfg:
            cfg['existing_004_cbramod_eegnet'] = str(ARTIFACT / 'dml_ds_cross_hfd_fewshot_004_v1')
        if filename == 'fewshot_dml_ds_pairs.yaml':
            # Supply native matched 100-epoch baselines also for formerly reused pairs.
            cfg['reuse_pairs'] = []
        target = CONFIGS / filename
        text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
        if target.exists() and target.read_text() != text:
            raise ValueError(f'Existing 100-epoch config differs: {target}')
        target.write_text(text)
    # The validation-only tuning search gets a fresh 100-epoch configuration.
    p = ROOT / 'configs/experiments/fewshot_dml_ds_tuning_v1.yaml'
    cfg = yaml.safe_load(p.read_text())
    cfg['experiment_id'] += '_100epoch'
    cfg['teachers'] = ['mirepnet', 'cbramod']
    cfg['joint_epochs'] = 100
    for value in cfg['teacher_training'].values(): value['epochs'] = 100
    cfg['output_root'] = str(ARTIFACT / 'fewshot_dml_ds_tuning_v1')
    cfg['report_subdirectory'] = str(REPORT.relative_to(ROOT / 'results') / 'fewshot_dml_ds_tuning_v1')
    target = CONFIGS / p.name
    text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
    if target.exists() and target.read_text() != text: raise ValueError('Tuning config changed')
    target.write_text(text)


def baseline_output(dataset, pair, subject, seed):
    return ARTIFACT / 'dml_ds_fewshot_pairs_v1/runs' / dataset / pair / 'dml_ds' / f'subject_{subject}' / f'seed_{seed}'


def make_job(family, path, doc, supplement=False):
    module, config_name, mode = FAMILIES[family]
    i = doc['identity']
    dataset, subject, seed = i['dataset'], int(i['subject']), int(i['seed'])
    condition = i.get('condition', i.get('method', i.get('kind')))
    pair = i.get('pair') or ('mirepnet_' + i['student'] if i.get('student') else 'mirepnet')
    rel = path.relative_to(OLD / family)
    out = ARTIFACT / family / rel.parent
    args = ['--dataset', dataset, '--subject', str(subject), '--seed', str(seed)]
    if mode in ('dcm_one', 'pilot'):
        args = ['--subject', str(subject), '--seed', str(seed)]
    dependency = None
    if mode in ('pair', 'dcm_one', 'cross'):
        args += ['--pair', pair, '--condition', condition] if mode != 'cross' else ['--pair', pair]
    elif mode == 'pilot':
        pass
    elif mode == 'features':
        args += ['--method', condition]
    elif mode == 'mirep':
        args += ['--kind', i['kind']]
        if i.get('student'): args += ['--student', i['student']]
    elif mode == 'position':
        args += ['--pair', pair, '--condition', condition]
        if condition in ('ce', 'ds'):
            module = 'run_hfd_position_baselines'
            args += ['--epochs', '100']
            out = ARTIFACT / family / 'runs' / dataset / pair / condition / '100' / f'subject_{subject}' / f'seed_{seed}'
        config_name = 'fewshot_dml_ds_pairs.yaml'
    if mode in ('pilot', 'features', 'cross', 'position'):
        dependency = str(baseline_output(dataset, pair, subject, seed))
    if mode == 'position' and pair.startswith('neurogpt_') and condition == 'ce':
        # The same study already ran the exact 100-epoch control.
        old100 = OLD / family / 'runs' / dataset / pair / 'ce/100' / f'subject_{subject}' / f'seed_{seed}'
        if not (old100 / 'evaluation.json').is_file(): raise ValueError('Missing existing 100-epoch NeuroGPT CE control')
        old_doc = json.loads((old100 / 'evaluation.json').read_text())
        if epoch_count(old_doc) != 100: raise ValueError('Wrong reused epoch count')
        for key in ('source_manifest_sha256', 'train_uid_sha256', 'test_uid_sha256',
                    'initial_model_sha256', 'initial_head_sha256', 'normalization'):
            if old_doc['identity'].get(key) != i.get(key): raise ValueError('Existing 100-epoch control does not match')
        return None, {'source20': str(path), 'reuse100': str(old100 / 'evaluation.json'), 'family': family}
    profile = ('mirepnet_8_30_calibration_EA' if mode == 'mirep' else
               'legacy_mirepnet_broadband_teacher_input_control' if pair.startswith('mirepnet_') else
               i.get('normalization', 'native_fixed_preprocessing'))
    job = {'family': family, 'module': module, 'mode': mode, 'config': str(CONFIGS / config_name),
           'dataset': dataset, 'pair': pair, 'condition': condition, 'subject': subject, 'seed': seed,
           'profile': profile, 'epochs': 100, 'source_epochs': 20,
           'source_evaluation': str(path), 'source_sha256': sha(path), 'output': str(out),
           'argv': args, 'dependency': dependency, 'matching_baseline_supplement': supplement}
    if mode in ('pilot', 'features', 'cross'):
        job['argv'] += ['--config', str(CONFIGS / config_name)]
    job['id'] = digest(job)
    return job, None


def main():
    configure()
    jobs, reused = [], []
    for family in FAMILIES:
        for path in sorted((OLD / family / 'runs').rglob('evaluation.json')):
            doc = json.loads(path.read_text())
            if doc.get('identity', {}).get('smoke') or 'smoke' in path.parts: continue
            if epoch_count(doc) == 100:
                reused.append({'family': family, 'reuse100': str(path), 'source_sha256': sha(path)})
                continue
            if epoch_count(doc) != 20: raise ValueError(f'Unclassified epoch count: {path}')
            job, reuse = make_job(family, path, doc)
            if job: jobs.append(job)
            if reuse: reused.append(reuse)
    # HFD's generic baseline reader requires the formerly reused CBraMod–IFNet
    # baseline in the new main-baseline tree. Its original DCM records are also replayed.
    for family in ('dcm_fewshot_0014_v1', 'dcm_fewshot_all_v1'):
        for path in sorted((OLD / family / 'runs').rglob('evaluation.json')):
            doc = json.loads(path.read_text()); i = doc['identity']
            if i.get('smoke') or i.get('pair') != 'cbramod_ifnet' or i.get('condition') not in ('none', 'dml_ds'): continue
            shadow = copy.deepcopy(doc)
            source = OLD / 'dml_ds_fewshot_pairs_v1/runs' / i['dataset'] / i['pair'] / i['condition'] / f"subject_{i['subject']}" / f"seed_{i['seed']}" / 'evaluation.json'
            # Use a virtual main-tree location while retaining the real source record.
            job, _ = make_job('dml_ds_fewshot_pairs_v1', source, shadow, supplement=True) if source.exists() else (None, None)
            if job is None:
                job = {'family': 'dml_ds_fewshot_pairs_v1', 'module': 'run_dml_ds_fewshot_pairs', 'mode': 'pair',
                       'config': str(CONFIGS / 'fewshot_dml_ds_pairs.yaml'), 'dataset': i['dataset'],
                       'pair': i['pair'], 'condition': i['condition'], 'subject': i['subject'], 'seed': i['seed'],
                       'profile': 'native_fixed_preprocessing', 'epochs': 100, 'source_epochs': 20,
                       'source_evaluation': str(path), 'source_sha256': sha(path),
                       'output': str(ARTIFACT / 'dml_ds_fewshot_pairs_v1/runs' / i['dataset'] / i['pair'] / i['condition'] / f"subject_{i['subject']}" / f"seed_{i['seed']}"),
                       'argv': ['--dataset', i['dataset'], '--pair', i['pair'], '--condition', i['condition'],
                                '--subject', str(i['subject']), '--seed', str(i['seed'])],
                       'dependency': None, 'matching_baseline_supplement': True}
                job['id'] = digest(job)
            jobs.append(job)
    by_output = {}
    for job in jobs:
        if job['output'] in by_output: raise ValueError('Duplicate output identity')
        by_output[job['output']] = job
    for job in jobs:
        if job['dependency'] and job['dependency'] not in by_output:
            raise ValueError(f"Missing new 100-epoch dependency: {job['dependency']}")
    # Cover the currently discussed pair first, then all models and datasets.
    def priority(j):
        current = j['pair'] == 'cbramod_eegnet' and j['dataset'] in ('BNCI2014004', 'BNCI2014001')
        base = j['family'] == 'dml_ds_fewshot_pairs_v1'
        return (not current, j['subject'], j['seed'], not base, j['family'], j['dataset'], j['pair'], j['condition'])
    jobs.sort(key=priority)
    summary = dict(Counter(j['family'] for j in jobs))
    manifest = {'experiment_id': 'joint_fewshot_100epoch_v1', 'epochs': 100, 'jobs': jobs,
                'reused_100epoch_records': reused, 'family_counts': summary,
                'tuning_config': str(CONFIGS / 'fewshot_dml_ds_tuning_v1.yaml'),
                'policy': 'fresh initialization and 100-epoch cosine; historical inputs remain separate profiles'}
    target = REPORT / 'manifest.json'
    if target.exists() and json.loads(target.read_text()) != manifest and (REPORT / 'controller.json').exists():
        raise ValueError('An existing replay manifest differs; audit before changing it')
    write_json(target, manifest)
    write_json(REPORT / 'inventory.json', {'new_training_units': len(jobs), 'reused_100epoch_records': len(reused),
                                         'families': summary, 'all_joint_epochs': 100})
    print(json.dumps({'new_training_units': len(jobs), 'reused100': len(reused), 'families': summary}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
