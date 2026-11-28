"""Paths and immutable protocol for replaying reported few-shot runs at 100 epochs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.storage import PROJECT_ROOT as ROOT, REPORTS_ROOT, require_external_output, require_report_output

NAME = 'joint_fewshot_100epoch_v1'
ARTIFACT = require_external_output('/data1/llx/BigSmallcollab/results/bidir/' + NAME)
REPORT = require_report_output(REPORTS_ROOT / NAME / 'manifest.json').parent
CONFIGS = ROOT / 'configs/experiments/budget100'
OLD = Path('/data1/llx/BigSmallcollab/results/bidir')

# Every reported 20-epoch family is included. Position studies have no own YAML.
FAMILIES = {
    'dml_ds_fewshot_pairs_v1': ('run_dml_ds_fewshot_pairs', 'fewshot_dml_ds_pairs.yaml', 'pair'),
    'dcm_fewshot_0014_v1': ('run_dcm_fewshot', 'fewshot_dcm.yaml', 'dcm_one'),
    'dcm_fewshot_all_v1': ('run_dcm_fewshot_all', 'fewshot_dcm_all.yaml', 'pair'),
    'dml_ds_hfd_fewshot_004_v1': ('run_dml_ds_hfd_fewshot', 'fewshot_dml_ds_hfd_004.yaml', 'pilot'),
    'dml_ds_cross_hfd_fewshot_004_v1': ('run_dml_ds_cross_hfd_fewshot', 'fewshot_dml_ds_cross_hfd_004.yaml', 'pilot'),
    'dml_ds_cross_hfd_fewshot_all_v1': ('run_dml_ds_cross_hfd_all', 'fewshot_dml_ds_cross_hfd_all.yaml', 'cross'),
    'hfd_position_fewshot_v1': ('run_hfd_position_fewshot', None, 'position'),
    'dml_ds_mirepnet_fewshot_8_30_v1': ('run_mirepnet_fewshot_8_30', 'fewshot_dml_ds_mirepnet_8_30.yaml', 'mirep'),
    'ds_only_mirepnet_fewshot_8_30_v1': ('run_mirepnet_ds_only_fewshot', 'fewshot_ds_only_mirepnet_8_30.yaml', 'mirep'),
    'dml_ds_cross_domain_features_fewshot_v1': ('run_cross_domain_features_fewshot', 'fewshot_cross_domain_features_v1.yaml', 'features'),
    'dml_ds_semckd_weight50_fewshot_v1': ('run_cross_domain_features_fewshot', 'fewshot_semckd_weight50_v1.yaml', 'features'),
    'dml_ds_semckd_weight400_fewshot_v1': ('run_cross_domain_features_fewshot', 'fewshot_semckd_weight400_v1.yaml', 'features'),
    'dml_ds_terminal_hfd_fewshot_v1': ('run_terminal_hfd_fewshot', 'fewshot_dml_ds_terminal_hfd_v1.yaml', 'features'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    import os
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
    tmp.replace(path)


def runtime_fingerprint():
    paths = [ROOT / 'experiments/bidir/budget100.py', ROOT / 'experiments/bidir/run_budget100_job.py']
    # Freeze the reused numerical implementations, not mutable result exporters.
    for module, _, _ in FAMILIES.values(): paths.append(ROOT / 'experiments/bidir' / (module + '.py'))
    paths += [ROOT / 'experiments/bidir/run_hfd_position_baselines.py', ROOT / 'config.py',
              ROOT / 'experiments/distill/run_fewshot_structure_kd.py',
              ROOT / 'experiments/finetune/run_neurogpt_loso.py',
              ROOT / 'experiments/finetune/labram_alexmi_broadband.py']
    paths += [ROOT / 'collab' / name for name in ('dcm.py', 'dml_ds_heads.py', 'hfd.py',
              'hfd_cross_operator.py', 'hfd_cross_all.py', 'hfd_position.py',
              'cross_domain_features.py', 'hfd_terminal.py')]
    for pattern in ('data/*.py', 'configs/models/*.yaml', 'configs/datasets/*.yaml',
                    'configs/experiments/budget100/*.yaml'):
        paths += list(ROOT.glob(pattern))
    for name in ('mirepnet', 'cbramod', 'neurogpt', 'ifnet', 'eegnet', 'adfcnn'):
        paths += list((ROOT / 'models' / name).rglob('*.py'))
    paths += [ROOT / 'models/base.py', ROOT / 'models/__init__.py']
    values = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths)) if p.is_file()}
    return digest(values), values


def epoch_count(doc):
    i = doc.get('identity', {})
    if 'epochs' in i: return int(i['epochs'])
    selected = [doc.get(role, {}).get('selected_epoch') for role in ('big', 'small')]
    if selected[0] is not None and selected[0] == selected[1]: return int(selected[0])
    return None


def verify_completion(job):
    out = Path(job['output'])
    complete = out / 'budget100_completion.json'
    if not complete.is_file(): return False
    d = json.loads(complete.read_text())
    if (d['job_id'] != job['id'] or d['epochs'] != 100 or
            d['evaluation_sha256'] != sha(out / 'evaluation.json') or
            d['predictions_sha256'] != sha(out / 'predictions.npz')):
        raise ValueError(f'Invalid 100-epoch completion: {out}')
    if epoch_count(json.loads((out / 'evaluation.json').read_text())) != 100:
        raise ValueError('A non-100-epoch result entered the replay cohort')
    return True
