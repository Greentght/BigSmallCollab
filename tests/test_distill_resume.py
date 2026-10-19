"""Direct-run fold recovery and frozen YAMLs, using no real model training."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import config
from experiments.distill import progress, run_distill


@pytest.fixture
def fake_run(tmp_path, monkeypatch):
    config_dir = tmp_path / 'configs'
    (config_dir / 'datasets').mkdir(parents=True)
    (config_dir / 'models').mkdir()
    dataset = {'num_subjects': 2, 'num_classes': 2, 'seeds': [666],
               'val_split': 0.2, 'loso': {'seeds': [666]}}
    (config_dir / 'datasets' / 'BNCI2014004.yaml').write_text(yaml.safe_dump(dataset))
    for model in ('ifnet', 'cbramod'):
        value = {'finetune': {'defaults': {'epochs': 2, 'lr': 0.001,
                                         'batch_size': 2, 'weight_decay': 0.01},
                              'BNCI2014004': {'loso': {}}}}
        (config_dir / 'models' / f'{model}.yaml').write_text(yaml.safe_dump(value))
    monkeypatch.setattr(config, 'CONFIG_DIR', str(config_dir))
    monkeypatch.setenv('BIGSMALL_RUN_SNAPSHOT', '')
    config._queued_yaml_snapshot.cache_clear()
    monkeypatch.setattr(run_distill, 'require_report_output', lambda path: Path(path))
    monkeypatch.setattr(progress, 'require_report_output', lambda path: Path(path))
    monkeypatch.setattr(run_distill, '_set_thread_defaults', lambda: None)
    monkeypatch.setattr(run_distill, '_device', lambda _: 'cpu')
    student_data = SimpleNamespace(
        x=np.arange(12, dtype=np.float32).reshape(4, 1, 3),
        y=np.asarray([0, 1, 0, 1], dtype=np.int64),
        subjects=np.asarray([0, 0, 1, 1], dtype=np.int64),
        sample_uid=np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64),
        source_manifest_sha256='source-hash',
        input_manifest_sha256='student-input-hash',
        identity={'input_policy_sha256': 'student-policy-hash'},
    )
    state = {'training_calls': [], 'teacher_calls': [], 'fail_fold': None,
             'stop_after': None, 'stop_file': tmp_path / 'stop', 'row_extra': {}}
    monkeypatch.setattr(run_distill, '_load_or_prepare_loso_dataset',
                        lambda *_, **__: student_data)

    def load_teacher(dataset, teacher, seed, fold, **kwargs):
        state['teacher_calls'].append(fold)
        return {'_artifact_path': '/unused/fake-teacher-target.npz'}

    monkeypatch.setattr(run_distill, '_load_or_ensure_loso_teacher_targets', load_teacher)
    monkeypatch.setattr(run_distill, '_check_teacher', lambda *_, **__: {'status': 'pass'})
    monkeypatch.setattr(run_distill, '_artifact_info',
                        lambda path: {'path': str(path), 'sha256': 'fake-teacher-hash'})

    def train(args, dataset, protocol, teacher, student, artifact, fold, seed,
              method, X_tr, y_tr, X_te, y_te, subj_tr, targets, nc, device, **kwargs):
        state['training_calls'].append(fold)
        if fold == state['fail_fold']:
            raise RuntimeError('simulated interrupted fold')
        if fold == state['stop_after']:
            state['stop_file'].write_text('pause at next fold boundary')
        return {'dataset': dataset, 'teacher': teacher, 'student': student,
                'subject': fold + 1, 'seed': seed, 'method': method,
                'acc': 75.0 + fold, 'kappa': 0.5, 'n_test': len(y_te),
                'failure_status': 'complete', **state['row_extra']}

    monkeypatch.setattr(run_distill, '_run_method', train)
    argv = ['--dataset', 'BNCI2014004', '--teacher', 'cbramod',
            '--student', 'ifnet', '--protocol', 'loso', '--seeds', '666',
            '--methods', 'kd', '--resume', '--out_csv', str(tmp_path / 'scores.csv'),
            '--progress', str(tmp_path / 'scores.progress.json'), '--fail_fast']
    yield SimpleNamespace(state=state, data=student_data, argv=argv,
                          output=tmp_path / 'scores.csv',
                          progress=tmp_path / 'scores.progress.json',
                          config_dir=config_dir, root=tmp_path)
    config._queued_yaml_snapshot.cache_clear()


def _execute(fake_run, *extra):
    return run_distill._run(run_distill.parse_args([*fake_run.argv, *extra]))


def _csv_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def test_resume_preserves_first_fold_and_does_not_load_its_teacher_again(fake_run):
    fake_run.state['fail_fold'] = 1
    with pytest.raises(RuntimeError, match='interrupted fold'):
        _execute(fake_run)
    saved = json.loads(fake_run.progress.read_text())
    assert saved['status'] == 'failed'
    assert len(saved['records']) == 1
    assert [int(row['subject']) for row in _csv_rows(fake_run.output)] == [1]
    assert fake_run.state['training_calls'] == [0, 1]

    fake_run.state['fail_fold'] = None
    fake_run.state['training_calls'].clear()
    fake_run.state['teacher_calls'].clear()
    assert _execute(fake_run) == 0
    assert fake_run.state['training_calls'] == [1]
    assert fake_run.state['teacher_calls'] == [1]
    completed = json.loads(fake_run.progress.read_text())
    assert completed['status'] == 'completed'
    assert len(completed['records']) == completed['total_cells'] == 2
    assert [int(row['subject']) for row in _csv_rows(fake_run.output)] == [1, 2]

    fake_run.state['training_calls'].clear()
    fake_run.state['teacher_calls'].clear()
    assert _execute(fake_run) == 0
    assert not fake_run.state['training_calls']
    assert not fake_run.state['teacher_calls']
    assert len(_csv_rows(fake_run.output)) == 2


@pytest.mark.parametrize('changed', ['student_config', 'source', 'uid'])
def test_changed_identity_refuses_reuse_without_overwriting_results(fake_run, changed):
    assert _execute(fake_run) == 0
    previous_records = json.loads(fake_run.progress.read_text())['records']
    previous_csv = fake_run.output.read_bytes()
    if changed == 'student_config':
        path = fake_run.config_dir / 'models' / 'ifnet.yaml'
        value = yaml.safe_load(path.read_text())
        value['finetune']['defaults']['lr'] = 0.002
        path.write_text(yaml.safe_dump(value))
    elif changed == 'source':
        fake_run.data.source_manifest_sha256 = 'different-source-hash'
    else:
        fake_run.data.sample_uid[0, 1] = 999
    fake_run.state['training_calls'].clear()
    fake_run.state['teacher_calls'].clear()

    with pytest.raises(ValueError, match='identity mismatch'):
        _execute(fake_run)

    assert fake_run.output.read_bytes() == previous_csv
    assert json.loads(fake_run.progress.read_text())['records'] == previous_records
    assert not fake_run.state['training_calls']
    assert not fake_run.state['teacher_calls']


def test_preflight_checks_later_completed_fold_before_training_missing_early_fold(fake_run):
    # Without fail_fast, fold 0 fails and the later fold still completes. The
    # resume preflight must inspect that later retained row before retrying 0.
    fake_run.argv.remove('--fail_fast')
    fake_run.state['fail_fold'] = 0
    assert _execute(fake_run) == 1
    previous_records = json.loads(fake_run.progress.read_text())['records']
    previous_csv = fake_run.output.read_bytes()
    assert [int(row['subject']) for row in _csv_rows(fake_run.output)] == [2]
    assert len(previous_records) == 1
    fake_run.data.source_manifest_sha256 = 'changed-after-later-fold-completed'
    fake_run.state['fail_fold'] = None
    fake_run.state['training_calls'].clear()
    fake_run.state['teacher_calls'].clear()

    with pytest.raises(run_distill.ResumeIdentityError, match='identity mismatch'):
        _execute(fake_run)

    assert not fake_run.state['training_calls']
    assert not fake_run.state['teacher_calls']
    assert fake_run.output.read_bytes() == previous_csv
    assert json.loads(fake_run.progress.read_text())['records'] == previous_records


@pytest.mark.parametrize('changed', ['teacher_target', 'teacher_checkpoint'])
def test_completed_teacher_metadata_drift_stops_resume_before_any_training(
        fake_run, monkeypatch, changed):
    # Both source files are small metadata JSONs. No teacher weights are opened.
    manifest = fake_run.root / 'teacher-target-manifest.json'
    manifest.write_text(json.dumps({'checkpoint': 'original-checkpoint'}))
    baseline = fake_run.root / 'teacher-baseline-manifest.json'
    baseline.write_text(json.dumps({'final_model_sha256': 'original-checkpoint'}))
    monkeypatch.setattr(run_distill.loso_source, '_baseline_teacher_manifest',
                        lambda *_args, **_kwargs: baseline)
    fake_run.state['row_extra'] = {
        'teacher_target_manifest_path': str(manifest),
        'teacher_target_manifest_sha256': run_distill._sha256_file(manifest),
        'teacher_checkpoint_sha256': 'original-checkpoint',
    }
    fake_run.argv.remove('--fail_fast')
    assert _execute(fake_run) == 0
    previous_records = json.loads(fake_run.progress.read_text())['records']
    previous_csv = fake_run.output.read_bytes()
    if changed == 'teacher_target':
        manifest.write_text(json.dumps({'checkpoint': 'changed-checkpoint'}))
        expected = 'teacher target identity mismatch'
    else:
        baseline.write_text(json.dumps({'final_model_sha256': 'changed-checkpoint'}))
        expected = 'teacher checkpoint identity mismatch'
    fake_run.state['training_calls'].clear()
    fake_run.state['teacher_calls'].clear()

    with pytest.raises(run_distill.ResumeIdentityError, match=expected):
        _execute(fake_run)

    assert not fake_run.state['training_calls']
    assert not fake_run.state['teacher_calls']
    assert fake_run.output.read_bytes() == previous_csv
    assert json.loads(fake_run.progress.read_text())['records'] == previous_records


def test_stop_before_run_pauses_without_training(fake_run):
    fake_run.state['stop_file'].write_text('pause')
    assert _execute(fake_run, '--stop-file', str(fake_run.state['stop_file'])) == 75
    saved = json.loads(fake_run.progress.read_text())
    assert saved['status'] == 'paused'
    assert not saved['records']
    assert not fake_run.state['training_calls']
    assert not fake_run.state['teacher_calls']


def test_stop_during_fold_saves_current_fold_and_resumes_at_next(fake_run):
    fake_run.state['stop_after'] = 0
    stop_path = fake_run.state['stop_file']
    assert _execute(fake_run, '--stop-file', str(stop_path)) == 75
    saved = json.loads(fake_run.progress.read_text())
    assert saved['status'] == 'paused'
    assert len(saved['records']) == 1
    assert [int(row['subject']) for row in _csv_rows(fake_run.output)] == [1]
    assert fake_run.state['training_calls'] == [0]
    stop_path.unlink()
    fake_run.state['stop_after'] = None
    fake_run.state['training_calls'].clear()
    fake_run.state['teacher_calls'].clear()
    assert _execute(fake_run, '--stop-file', str(stop_path)) == 0
    assert fake_run.state['training_calls'] == [1]
    assert fake_run.state['teacher_calls'] == [1]


def test_snapshot_preserves_resolved_yaml_and_reaches_teacher_subprocess(fake_run):
    args = run_distill.parse_args(fake_run.argv)
    snapshot = run_distill.freeze_run(args)
    snapshot_path = fake_run.root / 'snapshot.json'
    snapshot_path.write_text(json.dumps({'snapshot': snapshot}))
    model_path = fake_run.config_dir / 'models' / 'ifnet.yaml'
    current = yaml.safe_load(model_path.read_text())
    current['finetune']['defaults']['lr'] = 99.0
    model_path.write_text(yaml.safe_dump(current))
    args.snapshot = str(snapshot_path)

    activated = run_distill._activate_snapshot(args)
    assert activated._frozen['student_configs']['BNCI2014004/ifnet']['lr'] == 0.001
    assert config.load_model_config('ifnet', 'BNCI2014004', 'loso')['lr'] == 0.001
    # Read only configuration in a fresh interpreter, as a cache-export teacher
    # subprocess does. No torch models or datasets are instantiated.
    program = ('import json,sys,config; config.CONFIG_DIR=sys.argv[1]; '
               'print(json.dumps(config.load_model_config("ifnet", '
               '"BNCI2014004", "loso")))')
    child = subprocess.run([sys.executable, '-B', '-c', program, str(fake_run.config_dir)],
                           check=True, text=True, capture_output=True, env=os.environ.copy())
    assert json.loads(child.stdout)['lr'] == 0.001
    assert run_distill._run(activated) == 0
    assert len(fake_run.state['training_calls']) == 2
