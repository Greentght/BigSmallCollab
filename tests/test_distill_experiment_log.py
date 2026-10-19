"""CPU-only reporting tests using synthetic completed folds and temporary CSVs."""
from copy import deepcopy
import csv
import json
from multiprocessing import Process
from pathlib import Path

import pytest

from experiments.distill import experiment_log as log


HEADER = ['model', 'method', 'seed', 'Epoch', 'lr', 'Batch', 'T', 'λ',
          '004', '001', '001-4', '15001', 'AlexMI']


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def make_log(path, extra=(), templates=True):
    names = HEADER[:8] + list(extra) + HEADER[8:]
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\r\n')
        writer.writerow(names)
        for teacher in ('MIRepNet', 'CbraMod'):
            for model in (teacher, 'IFnet', 'EEGNet', 'ADFCNN'):
                row = dict.fromkeys(names, '')
                row.update(model=model, method='none', seed='666,667,668', **{'004': '70.00'})
                writer.writerow([row[name] for name in names])
            if templates:
                for model in ('IFnet', 'EEGNet', 'ADFCNN'):
                    writer.writerow([model, 'kd'])  # Intentionally short.
        writer.writerow(['', '', ''])
    return names


def make_job(root, name='trial', dataset='BNCI2014004', teacher='cbramod', student='ifnet',
             methods=('KD_all',), epochs=50, lr=.001, weight_decay=.01, batch_size=16,
             temperature=1., lam_kd=.25, **cfg_updates):
    folder = root / 'jobs' / name
    count = log.DATASETS[dataset][1]
    scfg = dict(epochs=epochs, lr=lr, weight_decay=weight_decay, batch_size=batch_size,
                optimizer='adamw', use_filter_bank=True, **cfg_updates)
    args = dict(dataset=dataset, datasets=[dataset], teacher=teacher, teachers=[teacher],
                student=student, students=[student], protocol='loso', seeds=[666, 667, 668],
                keys=list(range(count)), methods=list(methods), epochs=epochs, lr=lr,
                weight_decay=weight_decay, batch_size=batch_size, temperature=temperature,
                lam_kd=lam_kd, lam_mmd=.5, lam_mi=.1, teacher_seed=None,
                mmd_sigmas=[1., 2.], mmd_normalize=True, mmd_class_conditional=False)
    snapshot = dict(schema_version=1, args=args,
                    dataset_configs={dataset: {'num_subjects': count,
                                     'loso': {} if dataset == 'AlexMI' else
                                     {'source_directory': f'/data1/llx/{dataset}/wide_fre_band'}}},
                    student_configs={f'{dataset}/{student}': scfg},
                    teacher_configs={f'{dataset}/{teacher}': {'epochs': 20, 'lr': .0001}},
                    code_sha256={'runner': 'same-code'})
    spec = dict(snapshot=snapshot, snapshot_sha256=log._hash(snapshot))
    progress = dict(schema_version=1, run_spec=snapshot,
                    run_spec_hash=log._hash(snapshot), status='completed',
                    total_cells=count * 3 * len(methods), records={})
    for method in methods:
        parameters = log._expected_parameters(args, scfg, method)
        for seed in args['seeds']:
            for key in args['keys']:
                cell = [dataset, teacher, student, seed, key, method]
                identity = dict(dataset=dataset, teacher=teacher, student=student,
                                seed=seed, key=key, method=method,
                                teacher_seed=seed if method != 'Base' else None,
                                student_config=dict(scfg))
                acc = 50. + 2 * (seed - 666) + key
                row = dict(dataset=dataset, teacher=teacher, student=student,
                           seed=seed, key=key, method=method, subject=key + 1,
                           protocol='loso', failure_status='complete', acc=acc,
                           test_accuracy=acc, test_balanced_accuracy=acc + 9,
                           n_test=500 if key == 0 else 10,
                           teacher_seed=seed if method != 'Base' else '',
                           student_epochs=epochs, student_lr=lr,
                           student_weight_decay=weight_decay, student_batch_size=batch_size,
                           temperature=parameters['temperature'],
                           lam_kd=parameters['lam_kd'], lam_mmd=parameters['lam_mmd'],
                           lam_mi=parameters['lam_mi'], optimizer='adamw',
                           scheduler='CosineAnnealingLR', weight_mode=parameters['weight_mode'],
                           mmd_sigmas='1.0 2.0', mmd_normalize=True,
                           mmd_class_conditional=False)
                progress['records'][log._canonical(cell)] = dict(cell=cell, identity=identity,
                    identity_hash=log._hash(identity), row=row)
    dump(folder / 'spec.json', spec)
    dump(folder / 'progress.json', progress)
    export_results(folder, progress)
    return folder


def export_results(folder, progress):
    rows = [record['row'] for record in progress['records'].values()]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (folder / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def register(job, path, registry=None):
    return log.register_completed_job(job, path, registry_path=registry)


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / 'exp_log.csv'
    make_log(path)
    return tmp_path, path


def test_accuracy_equal_subject_then_seed_mean_and_actual_student_parameters(setup):
    root, path = setup
    before = path.read_bytes()
    report = register(make_job(root), path)
    assert report['status'] == 'written', report
    row = next(row for row in read_rows(path) if row['model'] == 'IFnet' and row['Epoch'] == '50')
    assert row['004'] == '56.00'  # Not BA=65 or n_test-weighted=52.65.
    assert row['seed'] == '666,667,668'
    assert (row['Epoch'], row['lr'], row['Batch'], row['T'], row['λ']) == ('50', '0.001', '16', '1.0', '0.25')
    assert report['entries'][0]['fold_count'] == 27
    assert report['entries'][0]['seed_means'] == {'666': 54., '667': 56., '668': 58.}
    assert Path(report['backup']).read_bytes() == before
    assert (root / 'jobs/trial/experiment_log_registration.json').exists()


def test_baselines_and_other_raw_records_preserved_byte_for_byte(setup):
    root, path = setup
    before = path.read_bytes().splitlines(keepends=True)
    assert register(make_job(root), path)['status'] == 'written'
    after = path.read_bytes().splitlines(keepends=True)
    for line in before:
        if b',none,' in line or line.startswith((b'EEGNet,kd', b'ADFCNN,kd')):
            assert after.count(line) == before.count(line)
    assert path.read_bytes().endswith(b',,\r\n')
    assert path.read_bytes().startswith(before[0])


def test_idempotent_repeated_source_has_no_duplicate_and_no_csv_write(setup):
    root, path = setup
    job = make_job(root)
    assert register(job, path)['status'] == 'written'
    data, stat = path.read_bytes(), path.stat()
    repeated = register(job, path)
    assert repeated['status'] == 'unchanged', repeated
    assert path.read_bytes() == data
    assert path.stat().st_mtime_ns == stat.st_mtime_ns
    registry = log._read_json(root / 'experiment_log_registry.json')
    assert len(registry['entries']) == 1
    assert len(registry['entries'][0]['datasets']['BNCI2014004']['sources']) == 1


def test_matching_profile_other_dataset_completes_owned_row(setup):
    root, path = setup
    assert register(make_job(root, '004'), path)['status'] == 'written'
    assert register(make_job(root, '001', dataset='BNCI2014001'), path)['status'] == 'written'
    rows = [row for row in read_rows(path) if row['method'] == 'kd' and row['Epoch'] == '50']
    assert len(rows) == 1
    assert rows[0]['004'] == rows[0]['001'] == '56.00'


def test_same_settings_across_alexmi_and_four_class_task_share_row(setup):
    root, path = setup
    for name, dataset in (('004', 'BNCI2014004'), ('alex', 'AlexMI'), ('0014', 'BNCI2014001-4')):
        report = register(make_job(root, name, dataset=dataset), path)
        assert report['status'] == 'written', report
    rows = [row for row in read_rows(path) if row['method'] == 'kd' and row['Epoch'] == '50']
    assert len(rows) == 1
    assert (rows[0]['004'], rows[0]['AlexMI'], rows[0]['001-4']) == ('56.00', '55.50', '56.00')


def test_runtime_shape_overrides_are_verified_against_actual_preprocessing(setup):
    root, path = setup
    job = make_job(root, samples=2500, in_channels=3)
    progress = log._read_json(job / 'progress.json')
    for record in progress['records'].values():
        record['identity']['student_config'].update(samples=1000, in_channels=6,
                                                    dataset_name='BNCI2014004')
        record['identity_hash'] = log._hash(record['identity'])
        record['row']['preprocessing'] = json.dumps({'samples': 1000, 'in_channels': 6})
    dump(job / 'progress.json', progress)
    export_results(job, progress)
    report = register(job, path)
    assert report['status'] == 'written', report


def test_changed_data_source_same_dataset_does_not_reuse_old_row(setup):
    root, path = setup
    register(make_job(root, 'first'), path)
    job = make_job(root, 'new-source')
    progress = log._read_json(job / 'progress.json')
    for record in progress['records'].values():
        record['row']['source_manifest_sha256'] = 'new-source-content'
        record['identity']['source_manifest_sha256'] = 'new-source-content'
        record['identity_hash'] = log._hash(record['identity'])
    dump(job / 'progress.json', progress)
    export_results(job, progress)
    report = register(job, path)
    assert report['status'] == 'written', report
    assert len([row for row in read_rows(path) if row['Epoch'] == '50']) == 2


def test_same_source_manually_edited_row_is_not_duplicated(setup):
    root, path = setup
    job = make_job(root)
    register(job, path)
    original = path.read_bytes()
    manual = original.replace(b'56.00', b'55.55')
    path.write_bytes(manual)
    report = register(job, path)
    assert report['status'] == 'skipped', report
    assert 'manually changed' in report['reason']
    assert path.read_bytes() == manual
    assert len([row for row in read_rows(path) if row['Epoch'] == '50']) == 1


def test_unshown_weight_decay_does_not_mix_rows(setup):
    root, path = setup
    assert register(make_job(root, 'a'), path)['status'] == 'written'
    assert register(make_job(root, 'b', dataset='BNCI2014001', weight_decay=.0001), path)['status'] == 'written'
    rows = [row for row in read_rows(path) if row['method'] == 'kd' and row['Epoch'] == '50']
    assert len(rows) == 2
    assert [row['004'] for row in rows] == ['56.00', '']
    assert [row['001'] for row in rows] == ['', '56.00']


def test_new_parameter_same_method_inserted_after_same_student_method(setup):
    root, path = setup
    register(make_job(root, 'a'), path)
    register(make_job(root, 'b', epochs=100), path)
    rows = read_rows(path)
    ix = next(i for i, row in enumerate(rows) if row['model'] == 'CbraMod')
    assert [row['model'] for row in rows[ix:ix + 4]] == ['CbraMod', 'IFnet', 'EEGNet', 'ADFCNN']
    assert [(row['model'], row['method'], row['Epoch']) for row in rows[ix + 4:ix + 8]] == [
        ('IFnet', 'kd', '50'), ('IFnet', 'kd', '100'), ('EEGNet', 'kd', None), ('ADFCNN', 'kd', None)]


def test_new_method_after_student_last_experiment(setup):
    root, path = setup
    register(make_job(root, 'a'), path)
    report = register(make_job(root, 'b', methods=('KD_MMD',)), path)
    assert report['status'] == 'written', report
    rows = read_rows(path)
    matching = [row for row in rows if row['model'] == 'IFnet' and row['Epoch'] == '50']
    assert [row['method'] for row in matching] == ['kd', 'kd+mmd']


def test_teacher_sections_separate_and_only_requested_student(setup):
    root, path = setup
    report = register(make_job(root, student='adfcnn', teacher='mirepnet', batch_size=32), path)
    assert report['status'] == 'written'
    rows = read_rows(path)
    boundary = next(i for i, row in enumerate(rows) if row['model'] == 'CbraMod')
    assert any(row['model'] == 'ADFCNN' and row['004'] == '56.00' for row in rows[:boundary])
    assert not any(row['Epoch'] == '50' for row in rows[boundary:])
    assert sum(row['Epoch'] == '50' for row in rows) == 1


def test_unknown_header_and_exact_verified_model_key_preserved(setup):
    root, path = setup
    make_log(path, extra=('weight_decay', 'dropout', 'user_new_parameter'))
    report = register(make_job(root, dropout=.25), path)
    assert report['status'] == 'written', report
    row = next(row for row in read_rows(path) if row['Epoch'] == '50')
    assert row['weight_decay'] == '0.01'
    assert row['dropout'] == '0.25'
    assert row['user_new_parameter'] == ''
    assert any('user_new_parameter' in note for note in report['notes'])
    assert path.read_text().splitlines()[0].split(',') == HEADER[:8] + ['weight_decay', 'dropout', 'user_new_parameter'] + HEADER[8:]


def test_untracked_populated_history_is_not_reinterpreted(setup):
    root, path = setup
    text = path.read_bytes().replace(b'IFnet,kd\r\n', b'IFnet,kd,"666,667,668",50,0.001,16,,,70.00\r\n')
    path.write_bytes(text)
    report = register(make_job(root), path)
    assert report['status'] == 'written', report
    assert b'IFnet,kd,"666,667,668",50,0.001,16,,,70.00\r\n' in path.read_bytes()
    assert len([row for row in read_rows(path) if row['model'] == 'IFnet' and row['Epoch'] == '50']) == 3


def test_base_only_never_changes_none_comparison_block(setup):
    root, path = setup
    original = path.read_bytes()
    report = register(make_job(root, methods=('Base',)), path)
    assert report['status'] == 'skipped'
    assert 'none baseline' in report['reason']
    assert path.read_bytes() == original


@pytest.mark.parametrize('mutation,reason', [
    ('missing_seed', 'seeds'), ('partial_keys', 'all LOSO'),
    ('missing_fold', 'incomplete'), ('wrong_status', 'not completed'),
    ('duplicate', 'identity'), ('wrong_metric', 'metric'),
    ('no_acc', 'Accuracy'), ('nan', 'nonfinite'), ('range', '0–100'),
    ('wrong_protocol', 'LOSO'), ('wrong_params', 'parameter mismatch'),
    ('wrong_teacher_seed', 'teacher seed'), ('wrong_snapshot', 'snapshot integrity'),
    ('csv_duplicate', 'duplicate'), ('csv_accuracy_changed', 'CSV/progress')])
def test_reject_incomplete_or_inconsistent_results_without_changing_log(setup, mutation, reason):
    root, path = setup
    original = path.read_bytes()
    job = make_job(root)
    spec = log._read_json(job / 'spec.json')
    progress = log._read_json(job / 'progress.json')
    first_key = next(iter(progress['records']))
    record = progress['records'][first_key]
    if mutation == 'missing_seed':
        spec['snapshot']['args']['seeds'] = [666, 667]
    elif mutation == 'partial_keys':
        spec['snapshot']['args']['keys'] = [0]
    elif mutation == 'missing_fold':
        del progress['records'][first_key]
    elif mutation == 'wrong_status':
        progress['status'] = 'running'
    elif mutation == 'duplicate':
        progress['records']['duplicate-key'] = record
        del progress['records'][list(progress['records'])[1]]
    elif mutation == 'wrong_metric':
        record['row']['metric'] = 'Balanced Accuracy'
    elif mutation == 'no_acc':
        record['row'].pop('acc')
    elif mutation == 'nan':
        record['row']['acc'] = 'nan'
    elif mutation == 'range':
        record['row']['acc'] = 101.
    elif mutation == 'wrong_protocol':
        spec['snapshot']['args']['protocol'] = 'fewshot'
    elif mutation == 'wrong_params':
        record['row']['student_lr'] = .03
    elif mutation == 'wrong_teacher_seed':
        record['row']['teacher_seed'] = 0
    elif mutation == 'wrong_snapshot':
        spec['snapshot_sha256'] = 'wrong'
    if mutation != 'wrong_snapshot':
        spec['snapshot_sha256'] = log._hash(spec['snapshot'])
    progress['run_spec'] = deepcopy(spec['snapshot'])
    progress['run_spec_hash'] = log._hash(spec['snapshot'])
    dump(job / 'spec.json', spec)
    dump(job / 'progress.json', progress)
    if mutation == 'csv_duplicate':
        rows = (job / 'results.csv').read_text().splitlines()
        rows[-1] = rows[1]
        (job / 'results.csv').write_text('\n'.join(rows) + '\n')
    elif mutation == 'csv_accuracy_changed':
        text = (job / 'results.csv').read_text()
        (job / 'results.csv').write_text(text.replace('50.0,50.0', '49.0,49.0', 1))
    else:
        export_results(job, progress)
    report = register(job, path)
    assert report['status'] == 'skipped', report
    assert reason in report['reason'], report
    assert path.read_bytes() == original


@pytest.mark.parametrize('method,expected', [('KD_all', 'kd'), ('KD_masked', 'kd_masked'),
                                            ('KD_MMD', 'kd+mmd'), ('MMD', 'mmd')])
def test_method_mapping_and_unused_loss_defaults(method, expected, setup):
    root, path = setup
    report = register(make_job(root, methods=(method,)), path)
    assert report['status'] == 'written', report
    assert report['entries'][0]['method'] == expected
    row = next(row for row in read_rows(path) if row['Epoch'] == '50')
    assert row['method'] == expected
    assert row['λ'] == ('' if method == 'MMD' else '0.25')


def _register_child(job, path):
    result = register(Path(job), Path(path))
    if result['status'] not in ('written', 'unchanged'):
        raise AssertionError(result)


def test_concurrent_completions_preserve_both_and_shared_registry(setup):
    root, path = setup
    jobs = [make_job(root, 'a'), make_job(root, 'b', dataset='BNCI2014001')]
    processes = [Process(target=_register_child, args=(str(job), str(path))) for job in jobs]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    rows = [row for row in read_rows(path) if row['Epoch'] == '50']
    assert len(rows) == 1
    assert rows[0]['004'] == rows[0]['001'] == '56.00'


def test_manual_csv_edit_during_registration_is_preserved(setup, monkeypatch):
    root, path = setup
    job = make_job(root)
    original = path.read_bytes()
    actual = log._apply_outcome
    manual = original + b'user edit,,\r\n'
    def edit(*args):
        result = actual(*args)
        path.write_bytes(manual)
        return result
    monkeypatch.setattr(log, '_apply_outcome', edit)
    report = register(job, path)
    assert report['status'] == 'skipped', report
    assert 'manual edit' in report['reason']
    assert path.read_bytes() == manual


def test_csv_written_registry_failure_recovered_without_duplicate(setup, monkeypatch):
    root, path = setup
    job = make_job(root)
    actual = log._atomic_json
    def crash_registry(destination, value):
        if Path(destination).name == 'experiment_log_registry.json':
            raise OSError('simulated crash after CSV replacement')
        return actual(destination, value)
    with monkeypatch.context() as context:
        context.setattr(log, '_atomic_json', crash_registry)
        report = register(job, path)
    assert report['status'] == 'error'
    already_written = path.read_bytes()
    assert (root / 'experiment_log_registry.json.pending.json').exists()
    report = register(job, path)
    assert report['status'] == 'unchanged', report
    assert path.read_bytes() == already_written
    assert not (root / 'experiment_log_registry.json.pending.json').exists()


def test_log_io_error_never_raises_into_training(setup):
    root, path = setup
    report = register(make_job(root), path.with_name('does-not-exist.csv'))
    assert report['status'] == 'error'
    assert 'FileNotFoundError' in report['reason']


def test_malformed_baseline_block_refuses_registration(setup):
    root, path = setup
    original = path.read_bytes().replace(b'CbraMod,none', b'CbraMod,other')
    path.write_bytes(original)
    report = register(make_job(root), path)
    assert report['status'] == 'skipped'
    assert 'baseline' in report['reason'] or 'none block' in report['reason']
    assert path.read_bytes() == original
