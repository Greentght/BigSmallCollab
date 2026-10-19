"""Default direct-LOSO log paths and exception capture without training."""

import pytest
from pathlib import Path

from experiments.distill import run_distill


def _fake_run(calls):
    def run(args):
        calls.append(args)
        print('[mock-run] no model training')
        return 0
    return run


def test_loso_default_log_sits_next_to_explicit_csv_and_never_overwrites(
        monkeypatch, tmp_path):
    reports = tmp_path / 'reports'
    out_csv = tmp_path / 'custom' / 'scores.csv'
    monkeypatch.setattr(run_distill, 'require_report_output', Path)
    monkeypatch.setattr(run_distill, 'REPORTS_ROOT', reports)
    monkeypatch.setattr(run_distill, '_log_timestamp', lambda: '20300102T030405.000006Z')
    monkeypatch.setattr(run_distill.os, 'getpid', lambda: 12345)
    calls = []
    monkeypatch.setattr(run_distill, '_run', _fake_run(calls))
    argv = ['--protocol', 'loso', '--out_csv', str(out_csv), '--seeds', '666', '667']

    assert run_distill.main(argv) == 0
    first = out_csv.parent / 'scores_20300102T030405.000006Z_12345.log'
    assert first.is_file()
    first.write_text('preserve existing log')

    assert run_distill.main(argv) == 0
    second = out_csv.parent / 'scores_20300102T030405.000006Z_12345_1.log'

    assert first.read_text() == 'preserve existing log'
    assert second.is_file()
    assert len(calls) == 2
    content = second.read_text()
    assert '[mock-run] no model training' in content
    assert '"--seeds", "666", "667"' in content
    assert 'student=explicit --seeds [666, 667]' in content
    assert 'teacher=same-as-student-seed' in content


def test_loso_default_log_uses_checkout_reports_root_without_out_csv(
        monkeypatch, tmp_path):
    reports = tmp_path / 'checkout-results'
    monkeypatch.setattr(run_distill, 'REPORTS_ROOT', reports)
    monkeypatch.setattr(run_distill, '_log_timestamp', lambda: '20300102T030405.000006Z')
    monkeypatch.setattr(run_distill.os, 'getpid', lambda: 777)
    monkeypatch.setattr(run_distill, '_run', _fake_run([]))

    assert run_distill.main(['--protocol', 'loso', '--seeds', '668']) == 0

    path = reports / 'loso' / 'distill' / 'logs' / '20300102T030405.000006Z_777.log'
    assert path.is_file()
    content = path.read_text()
    assert 'student=explicit --seeds [668]' in content


def test_explicit_log_file_path_remains_in_use(monkeypatch, tmp_path):
    monkeypatch.setattr(run_distill, 'REPORTS_ROOT', tmp_path / 'reports')
    monkeypatch.setattr(run_distill, '_run', _fake_run([]))
    path = tmp_path / 'explicit' / 'run.log'
    path.parent.mkdir()
    path.write_text('old content')

    assert run_distill.main([
        '--protocol', 'loso', '--log_file', str(path), '--seeds', '666',
    ]) == 0

    content = path.read_text()
    assert 'old content' not in content
    assert f'[log] writing stdout/stderr to {path}' in content
    assert 'student=explicit --seeds [666]' in content
    assert not (tmp_path / 'reports' / 'loso' / 'distill' / 'logs').exists()


def test_uncaught_run_exception_traceback_is_written_to_log_then_raised(
        monkeypatch, tmp_path, capsys):
    reports = tmp_path / 'checkout-results'
    monkeypatch.setattr(run_distill, 'REPORTS_ROOT', reports)
    monkeypatch.setattr(run_distill, '_log_timestamp', lambda: '20300102T030405.000006Z')
    monkeypatch.setattr(run_distill.os, 'getpid', lambda: 888)

    def fail(_args):
        print('[mock-run] about to fail')
        raise RuntimeError('mock distillation failure')

    monkeypatch.setattr(run_distill, '_run', fail)
    with pytest.raises(RuntimeError, match='mock distillation failure'):
        run_distill.main(['--protocol', 'loso', '--seeds', '666'])

    path = reports / 'loso' / 'distill' / 'logs' / '20300102T030405.000006Z_888.log'
    content = path.read_text()
    assert '[mock-run] about to fail' in content
    assert '[exception] direct distillation run failed' in content
    assert 'Traceback (most recent call last):' in content
    assert 'RuntimeError: mock distillation failure' in content
    # The explicit traceback is persisted once to the log. Python's normal
    # propagation remains the only path that could print it to the console.
    assert 'Traceback (most recent call last):' not in capsys.readouterr().err
