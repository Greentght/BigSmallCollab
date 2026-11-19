"""Validate and summarize complete BNCI2014001 AFID few-shot results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from experiments.storage import REPORTS_ROOT, RESULTS_ROOT, require_report_output

TEACHERS = ('mirepnet', 'cbramod')
STUDENTS = ('ifnet', 'eegnet', 'adfcnn')
SHOTS = (10, 5, 20)
SEEDS = (666, 667, 668)
SUBJECTS = tuple(range(9))
METHODS = ('none', 'kd', 'dml', 'afid_self_gate', 'afid')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_record(path, report_path, expected):
    with path.open(encoding='utf-8') as stream:
        record = json.load(stream)
    identity = record['identity']
    if any(identity.get(key) != value for key, value in expected.items() if key != 'method'):
        raise ValueError(f'{path}: result identity does not match its location')
    if record.get('method') != expected['method'] or record.get('identity_sha256') != digest(identity):
        raise ValueError(f'{path}: invalid method or identity hash')
    train, test = record['support_trial_uids'], record['test_trial_uids']
    expected_train = 43 if expected.get('protocol') == 'calibration_30pct' else 2 * expected['shot']
    if (len(train) != expected_train or len(set(train)) != len(train)
            or len(test) != 101 or len(set(test)) != len(test) or set(train) & set(test)):
        raise ValueError(f'{path}: invalid support or test UIDs')
    if digest(train) != identity['support_uid_sha256'] or digest(test) != identity['test_uid_sha256']:
        raise ValueError(f'{path}: trial UID hash mismatch')
    if not identity.get('source_manifest_sha256'):
        raise ValueError(f'{path}: source identity is missing')
    if Path(record['predictions_path']).resolve() != report_path.resolve():
        raise ValueError(f'{path}: predictions path differs from expected report')
    with report_path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(test) or [row['trial_uid'] for row in rows] != test:
        raise ValueError(f'{report_path}: prediction UIDs do not match completion record')
    labels = [int(row['y_true']) for row in rows]
    student = [int(row['y_pred']) for row in rows]
    teacher = [int(row['teacher_y_pred']) for row in rows]
    if any(value not in (0, 1) for value in labels + student + teacher):
        raise ValueError(f'{report_path}: invalid binary class')
    for name, predictions in (('accuracy_pct', student), ('teacher_accuracy_pct', teacher)):
        actual = 100.0 * np.mean(np.equal(labels, predictions))
        if not math.isclose(float(record[name]), actual, rel_tol=0, abs_tol=1e-8):
            raise ValueError(f'{path}: {name} does not match predictions')
    if record.get('n_test') != len(test) or record.get('student_inference') != 'standalone_original_classifier':
        raise ValueError(f'{path}: invalid evaluation protocol')
    return record, tuple(labels)


def atomic_csv(path, fields, rows):
    path = require_report_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def collect(run_id, protocol='kshot'):
    root = RESULTS_ROOT / 'fewshot' / 'afid' / run_id
    reports = REPORTS_ROOT / 'fewshot' / 'afid' / run_id
    cells, splits, identities, protocols = {}, {}, {}, {}
    shots = (None,) if protocol == 'calibration_30pct' else SHOTS
    source_identity = None
    for teacher in TEACHERS:
        for student in STUDENTS:
            for shot in shots:
                for subject in SUBJECTS:
                    for seed in SEEDS:
                        setting = 'calibration30' if protocol == 'calibration_30pct' else f'shot{shot}'
                        rel = Path(teacher) / student / setting / f'subject{subject + 1:02d}' / f'seed{seed}'
                        for method in METHODS:
                            path = root / rel / method / 'completed.json'
                            if not path.is_file():
                                continue
                            key = (teacher, student, shot, subject, seed, method)
                            if key in cells:
                                raise ValueError(f'duplicate result {key}')
                            expected = dict(dataset='BNCI2014001', teacher=teacher,
                                            student=student, shot=shot, subject=subject,
                                            seed=seed, method=method)
                            if protocol == 'calibration_30pct':
                                expected['protocol'] = protocol
                            record, labels = read_record(path, reports / rel / f'{method}_predictions.csv', expected)
                            identity = record['identity']
                            if source_identity is None:
                                source_identity = identity['source_manifest_sha256']
                            elif source_identity != identity['source_manifest_sha256']:
                                raise ValueError(f'{path}: source manifest differs within this run')
                            group_identity = {key: value for key, value in identity.items()
                                        if key not in ('subject', 'seed', 'support_uid_sha256',
                                                       'test_uid_sha256', 'ea_sha256')}
                            protocol_key = (teacher, student, shot)
                            if protocol_key in protocols and protocols[protocol_key] != group_identity:
                                raise ValueError(f'{path}: configuration or code differs within this group')
                            protocols[protocol_key] = group_identity
                            cell = key[:-1]
                            if cell in identities and identities[cell] != identity:
                                raise ValueError(f'{path}: methods do not share the same identity')
                            identities[cell] = identity
                            split_key = (shot, subject, seed)
                            split = (identity['source_manifest_sha256'], tuple(record['support_trial_uids']),
                                     tuple(record['test_trial_uids']), labels)
                            if split_key in splits and splits[split_key] != split:
                                raise ValueError(f'{path}: models do not share the same trial split')
                            splits[split_key] = split
                            cells[key] = record
    if protocol == 'kshot':
        for subject in SUBJECTS:
            for seed in SEEDS:
                available = [(shot, splits[(shot, subject, seed)]) for shot in sorted(SHOTS)
                             if (shot, subject, seed) in splits]
                for (_, smaller), (_, larger) in zip(available, available[1:]):
                    if (smaller[0] != larger[0] or not set(smaller[1]).issubset(larger[1])
                            or smaller[2:] != larger[2:]):
                        raise ValueError(f'subject {subject + 1}, seed {seed}: K-shot splits are not nested')
    return cells


def summarize(cells, bootstrap_seed=2026, bootstrap_samples=10000, protocol='kshot'):
    groups, comparisons = [], []
    shots = (None,) if protocol == 'calibration_30pct' else SHOTS
    for teacher in TEACHERS:
        for student in STUDENTS:
            for shot in shots:
                values = {}
                cell_counts = {}
                for method in METHODS:
                    table = {(subject, seed): cells[(teacher, student, shot, subject, seed, method)]
                             for subject in SUBJECTS for seed in SEEDS
                             if (teacher, student, shot, subject, seed, method) in cells}
                    complete = len(table) == len(SUBJECTS) * len(SEEDS)
                    cell_counts[method] = len(table)
                    values[method] = table if complete else None
                    averages = [np.mean([table[(subject, seed)]['accuracy_pct'] for subject in SUBJECTS])
                                for seed in SEEDS] if complete else []
                    groups.append(dict(dataset='BNCI2014001', teacher=teacher, student=student,
                                       protocol=protocol, shot=shot, n_train=43 if shot is None else 2 * shot,
                                       method=method, completed_cells=len(table),
                                       expected_cells=27, complete=complete,
                                       seed_666=averages[0] if complete else '',
                                       seed_667=averages[1] if complete else '',
                                       seed_668=averages[2] if complete else '',
                                       accuracy_pct=float(np.mean(averages)) if complete else ''))
                teacher_table = values['afid']
                teacher_averages = ([np.mean([teacher_table[(subject, seed)]['teacher_accuracy_pct']
                                              for subject in SUBJECTS]) for seed in SEEDS]
                                    if teacher_table is not None else [])
                groups.append(dict(dataset='BNCI2014001', teacher=teacher, student=student,
                                   protocol=protocol, shot=shot, n_train=43 if shot is None else 2 * shot,
                                   method='teacher',
                                   completed_cells=cell_counts['afid'],
                                   expected_cells=27, complete=bool(teacher_averages),
                                   seed_666=teacher_averages[0] if teacher_averages else '',
                                   seed_667=teacher_averages[1] if teacher_averages else '',
                                   seed_668=teacher_averages[2] if teacher_averages else '',
                                   accuracy_pct=float(np.mean(teacher_averages)) if teacher_averages else ''))
                for reference in ('afid_self_gate', 'none', 'kd', 'teacher'):
                    afid = values['afid']
                    baseline = afid if reference == 'teacher' else values[reference]
                    if afid is None or baseline is None:
                        continue
                    reference_key = 'teacher_accuracy_pct' if reference == 'teacher' else 'accuracy_pct'
                    diffs = np.asarray([np.mean([afid[(subject, seed)]['accuracy_pct'] -
                                                 baseline[(subject, seed)][reference_key]
                                                 for seed in SEEDS]) for subject in SUBJECTS])
                    seed_material = f'{bootstrap_seed}:{teacher}:{student}:{shot}:{reference}'.encode()
                    comparison_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], 'big')
                    rng = np.random.default_rng(comparison_seed)
                    samples = rng.choice(diffs, size=(bootstrap_samples, len(SUBJECTS)), replace=True).mean(axis=1)
                    lower, upper = np.percentile(samples, [2.5, 97.5])
                    comparisons.append(dict(dataset='BNCI2014001', teacher=teacher, student=student,
                                            protocol=protocol, shot=shot, n_train=43 if shot is None else 2 * shot,
                                            method='afid', reference=reference,
                                            mean_delta_pp=float(diffs.mean()), ci95_lower_pp=float(lower),
                                            ci95_upper_pp=float(upper), n_subjects=len(SUBJECTS),
                                            bootstrap_samples=bootstrap_samples, bootstrap_seed=comparison_seed))
    return groups, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--protocol', choices=('kshot', 'calibration_30pct'), default='kshot')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in ('', '.', '..'):
        raise ValueError('run ID must be a single directory name')
    groups, comparisons = summarize(collect(args.run_id, protocol=args.protocol), protocol=args.protocol)
    output = REPORTS_ROOT / 'fewshot' / 'afid' / args.run_id
    atomic_csv(output / 'group_summary.csv', list(groups[0]), groups)
    fields = ['dataset', 'teacher', 'student', 'protocol', 'shot', 'n_train', 'method', 'reference', 'mean_delta_pp',
              'ci95_lower_pp', 'ci95_upper_pp', 'n_subjects', 'bootstrap_samples', 'bootstrap_seed']
    atomic_csv(output / 'paired_comparisons.csv', fields, comparisons)
    print(f'Complete groups: {sum(row["complete"] for row in groups)}/{len(groups)}')
    print(f'Reports: {output}')


if __name__ == '__main__':
    main()
