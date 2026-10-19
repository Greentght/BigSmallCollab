"""Register verified completed queue trials in the user-maintained tuning CSV.

Only JSON/CSV reports are read. This module never loads EEG, model files or a
training framework. Registration failures are returned to the caller so a log
permission problem cannot turn a successful training run into a failed run.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

from experiments.storage import PROJECT_ROOT


DEFAULT_LOG = PROJECT_ROOT / 'docs/exp_log.csv'
DATASETS = {'BNCI2014004': ('004', 9), 'BNCI2014001': ('001', 9),
            'BNCI2014001-4': ('001-4', 9), 'BNCI2015001': ('15001', 12),
            'AlexMI': ('AlexMI', 8)}
TEACHERS = {'mirepnet': 'MIRepNet', 'cbramod': 'CbraMod'}
STUDENTS = {'ifnet': 'IFnet', 'eegnet': 'EEGNet', 'adfcnn': 'ADFCNN'}
METHODS = {'KD_all': 'kd', 'KD_masked': 'kd_masked', 'KD_MMD': 'kd+mmd',
           'MMD': 'mmd', 'Base': 'none'}
SEEDS = (666, 667, 668)
_DATA_COLUMNS = tuple(value[0] for value in DATASETS.values())


class RegistrationSkipped(ValueError):
    """A completed report does not meet the register's fixed inclusion rules."""


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _bytes_hash(value):
    return hashlib.sha256(value).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def _atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name,
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
        temporary = None
        fd = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_json(path, value):
    _atomic_bytes(path, (json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   indent=2, allow_nan=False) + '\n').encode())


@contextmanager
def _lock(path):
    # Never lock the CSV inode itself: atomic replacement changes that inode.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _finite(value, name):
    try:
        if isinstance(value, bool):
            raise ValueError
        result = float(value)
        if not math.isfinite(result):
            raise ValueError
        return result
    except (ValueError, TypeError, OverflowError) as exc:
        raise RegistrationSkipped(f'invalid/nonfinite {name}: {value!r}') from exc


def _same(left, right):
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return str(left).lower() == str(right).lower()
    try:
        return float(left) == float(right)
    except (ValueError, TypeError, OverflowError):
        return left == right or str(left) == str(right)


def _one(args, singular, plural):
    values = args.get(plural, [args.get(singular)])
    if not isinstance(values, list) or len(values) != 1:
        raise RegistrationSkipped('automatic registration requires one dataset/teacher/student')
    value = values[0]
    if args.get(singular, value) != value:
        raise RegistrationSkipped(f'inconsistent {singular} identity')
    return value


def _expected_parameters(args, student_config, method):
    values = {}
    for key, default in (('epochs', 50), ('lr', .001), ('weight_decay', .01),
                         ('batch_size', 16)):
        values[key] = student_config.get(key, default)
        if key in args and not _same(args[key], values[key]):
            raise RegistrationSkipped(f'args/student configuration disagree on {key}')
    for key in ('epochs', 'batch_size'):
        numeric = _finite(values[key], key)
        if numeric <= 0 or numeric != int(numeric):
            raise RegistrationSkipped(f'invalid integer {key}')
        values[key] = int(numeric)
    values['lr'] = _finite(values['lr'], 'lr')
    values['weight_decay'] = _finite(values['weight_decay'], 'weight_decay')
    if values['lr'] <= 0 or values['weight_decay'] < 0:
        raise RegistrationSkipped('invalid learning rate or weight decay')
    values.update(temperature=args.get('temperature'),
                  lam_kd=args.get('lam_kd') if method in {'KD_all', 'KD_masked', 'KD_MMD'} else 0.,
                  lam_mmd=args.get('lam_mmd') if method in {'MMD', 'KD_MMD'} else 0.,
                  lam_mi=0.)
    for key in ('temperature', 'lam_kd', 'lam_mmd', 'lam_mi'):
        values[key] = _finite(values[key], key)
        if values[key] < 0 or key == 'temperature' and values[key] <= 0:
            raise RegistrationSkipped(f'invalid {key}')
    values.update(optimizer=str(student_config.get('optimizer', 'adamw')).lower(),
                  scheduler='CosineAnnealingLR',
                  weight_mode='teacher_correct' if method == 'KD_masked' else
                              'none' if method == 'Base' else 'all')
    if method in {'MMD', 'KD_MMD'}:
        sigmas = args.get('mmd_sigmas')
        if not sigmas or any(_finite(value, 'mmd_sigmas') <= 0 for value in sigmas):
            raise RegistrationSkipped('missing/invalid MMD bandwidths')
        values.update(mmd_sigmas=list(sigmas),
                      mmd_normalize=args.get('mmd_normalize', True),
                      mmd_class_conditional=args.get('mmd_class_conditional', False))
    return values


def _validate_job(folder):
    """Validate provenance, complete folds, Accuracy and actual effective settings."""
    spec = _read_json(folder / 'spec.json')
    snapshot = spec['snapshot']
    if spec.get('snapshot_sha256') != _hash(snapshot):
        raise RegistrationSkipped('queued configuration snapshot integrity mismatch')
    progress = _read_json(folder / 'progress.json')
    if progress.get('status') != 'completed':
        raise RegistrationSkipped('training progress is not completed')
    if progress.get('run_spec') != snapshot or progress.get('run_spec_hash') != _hash(snapshot):
        raise RegistrationSkipped('progress/snapshot identity mismatch')
    args = snapshot['args']
    if args.get('protocol') != 'loso':
        raise RegistrationSkipped('only LOSO Accuracy can be registered')
    seeds = args.get('seeds')
    if not isinstance(seeds, list) or any(type(s) is not int for s in seeds) \
            or len(seeds) != 3 or set(seeds) != set(SEEDS):
        raise RegistrationSkipped('registration requires exact seeds 666,667,668')
    dataset = _one(args, 'dataset', 'datasets')
    teacher = _one(args, 'teacher', 'teachers')
    student = _one(args, 'student', 'students')
    if dataset not in DATASETS or teacher not in TEACHERS or student not in STUDENTS:
        raise RegistrationSkipped('unsupported dataset, teacher or student')
    dcfg = snapshot['dataset_configs'][dataset]
    expected_count = DATASETS[dataset][1]
    if dcfg.get('num_subjects') != expected_count:
        raise RegistrationSkipped('dataset subject count differs from the supported LOSO task')
    keys = args.get('keys')
    if not isinstance(keys, list) or any(type(k) is not int for k in keys) \
            or len(keys) != expected_count or set(keys) != set(range(expected_count)):
        raise RegistrationSkipped('all LOSO subject folds are required before registration')
    methods = args.get('methods')
    if not isinstance(methods, list) or not methods or len(set(methods)) != len(methods) \
            or any(method not in METHODS for method in methods):
        raise RegistrationSkipped('unsupported/duplicate method selection')
    scfg = snapshot['student_configs'][f'{dataset}/{student}']
    tcfg = snapshot.get('teacher_configs', {}).get(f'{dataset}/{teacher}')
    if not isinstance(scfg, dict) or not isinstance(tcfg, dict):
        raise RegistrationSkipped('missing resolved student or teacher configuration')
    expected_total = expected_count * 3 * len(methods)
    records = progress.get('records')
    if not isinstance(records, dict) or len(records) != expected_total \
            or progress.get('total_cells') != expected_total:
        raise RegistrationSkipped('incomplete progress fold records')
    parameters = {m: _expected_parameters(args, scfg, m) for m in methods}
    by_method = {m: {} for m in methods}
    provenance = {m: {} for m in methods}
    rows_by_cell = {}
    for record_key, record in records.items():
        row, identity, cell = record['row'], record['identity'], record['cell']
        if record_key != _canonical(cell) or record.get('identity_hash') != _hash(identity):
            raise RegistrationSkipped('invalid completed fold identity hash')
        if not isinstance(cell, list) or len(cell) != 6:
            raise RegistrationSkipped('invalid completed fold cell')
        d, t, s, seed, key, method = cell
        if (d, t, s) != (dataset, teacher, student) or seed not in SEEDS \
                or type(seed) is not int or type(key) is not int \
                or key not in keys or method not in methods:
            raise RegistrationSkipped('unexpected completed fold identity')
        for name, expected in (('dataset', d), ('teacher', t), ('student', s),
                               ('seed', seed), ('key', key), ('method', method)):
            if row.get(name) != expected or identity.get(name) != expected:
                raise RegistrationSkipped(f'completed fold disagrees on {name}')
        if row.get('protocol') != 'loso' or row.get('subject') != key + 1 \
                or row.get('failure_status') != 'complete':
            raise RegistrationSkipped('fold is not a successful LOSO subject evaluation')
        if row.get('metric') not in (None, 'Accuracy', 'accuracy', 'acc', 'Accuracy (%)'):
            raise RegistrationSkipped('fold metric is not Accuracy')
        acc = _finite(row.get('acc'), 'Accuracy acc')
        if not 0 <= acc <= 100:
            raise RegistrationSkipped('Accuracy must be on a 0–100 scale')
        if 'test_accuracy' in row and not _same(row['test_accuracy'], acc):
            raise RegistrationSkipped('acc/test_accuracy disagree')
        expected_teacher_seed = None if method == 'Base' else args.get('teacher_seed')
        if expected_teacher_seed is None and method != 'Base':
            expected_teacher_seed = seed
        if identity.get('teacher_seed') != expected_teacher_seed \
                or method != 'Base' and row.get('teacher_seed') != expected_teacher_seed:
            raise RegistrationSkipped('teacher seed provenance mismatch')
        runtime_config = identity.get('student_config')
        if not isinstance(runtime_config, dict):
            raise RegistrationSkipped('missing actual student configuration')
        runtime_keys = {'in_channels', 'samples', 'dataset_name'}
        for name, value in scfg.items():
            if name in runtime_keys:
                continue  # _student_runtime_cfg explicitly replaces these.
            if name not in runtime_config or not _same(runtime_config[name], value):
                raise RegistrationSkipped(f'actual/snapshot student configuration differs: {name}')
        if runtime_config.get('dataset_name', dataset) != dataset:
            raise RegistrationSkipped('actual student dataset_name mismatch')
        if row.get('preprocessing'):
            preprocessing = row['preprocessing']
            if isinstance(preprocessing, str):
                preprocessing = json.loads(preprocessing)
            for name in ('in_channels', 'samples'):
                if runtime_config.get(name) != preprocessing.get(name):
                    raise RegistrationSkipped(f'actual student/preprocessing shape mismatch: {name}')
        p = parameters[method]
        for name, row_name in (('epochs', 'student_epochs'), ('lr', 'student_lr'),
                               ('weight_decay', 'student_weight_decay'),
                               ('batch_size', 'student_batch_size'), ('temperature', 'temperature'),
                               ('lam_kd', 'lam_kd'), ('lam_mmd', 'lam_mmd'), ('lam_mi', 'lam_mi'),
                               ('optimizer', 'optimizer'), ('scheduler', 'scheduler'),
                               ('weight_mode', 'weight_mode')):
            if not _same(row.get(row_name), p[name]):
                raise RegistrationSkipped(f'actual fold/snapshot parameter mismatch: {row_name}')
        if method in {'MMD', 'KD_MMD'}:
            sigmas = row.get('mmd_sigmas', '')
            if isinstance(sigmas, str):
                sigmas = [_finite(v, 'mmd_sigmas') for v in sigmas.split()]
            if sigmas != p['mmd_sigmas']:
                raise RegistrationSkipped('actual MMD bandwidth mismatch')
            for name in ('mmd_normalize', 'mmd_class_conditional'):
                if not _same(row.get(name), p[name]):
                    raise RegistrationSkipped(f'actual {name} mismatch')
        if (seed, key) in by_method[method]:
            raise RegistrationSkipped('duplicate seed/subject/method result')
        by_method[method][seed, key] = acc
        rows_by_cell[tuple(cell)] = row
        source_fields = ('source_manifest_sha256', 'student_input_manifest_sha256',
                         'student_input_policy_sha256', 'teacher_input_policy_sha256')
        provenance[method][f'{seed}/{key}'] = {
            name: row.get(name, '') for name in (*source_fields, 'teacher_checkpoint_sha256')}
        for name in source_fields[:3]:
            if name in identity and identity[name] != row.get(name):
                raise RegistrationSkipped(f'actual fold/source identity mismatch: {name}')
    # CSV is an independent final export; require it to contain exactly the
    # same records as durable progress, not just the same rounded aggregate.
    csv_path = folder / 'results.csv'
    with csv_path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        exported = list(reader)
    if len(exported) != expected_total:
        raise RegistrationSkipped('final result CSV has missing/extra folds')
    seen = set()
    for row in exported:
        cell = (row.get('dataset'), row.get('teacher'), row.get('student'),
                int(row['seed']), int(row['key']), row.get('method'))
        if cell in seen or cell not in rows_by_cell:
            raise RegistrationSkipped('final CSV has duplicate/unexpected folds')
        seen.add(cell)
        original = rows_by_cell[cell]
        for name, value in original.items():
            if row.get(name) != ('' if value is None else str(value)):
                raise RegistrationSkipped(f'final CSV/progress disagree on {name}')
    outcomes = []
    for method in methods:
        if method == 'Base':
            continue  # The user's fixed none block is never auto-written.
        seed_means = {str(seed): sum(by_method[method][seed, k] for k in keys) /
                      len(keys) for seed in SEEDS}
        outcomes.append({'dataset': dataset, 'dataset_column': DATASETS[dataset][0],
                         'teacher': TEACHERS[teacher], 'student': STUDENTS[student],
                         'method': METHODS[method], 'internal_method': method,
                         'parameters': parameters[method], 'student_config': scfg,
                         'teacher_config': tcfg, 'dataset_config': dcfg,
                         'data_provenance': provenance[method],
                         'seed_means': seed_means,
                         'accuracy': sum(seed_means.values()) / len(SEEDS),
                         'fold_count': expected_count * len(SEEDS)})
    source = {'job_dir': str(folder.resolve()), 'spec_sha256': _bytes_hash((folder / 'spec.json').read_bytes()),
              'snapshot_sha256': spec['snapshot_sha256'],
              'progress_sha256': _bytes_hash((folder / 'progress.json').read_bytes()),
              'results_sha256': _bytes_hash(csv_path.read_bytes()),
              'seeds': list(SEEDS), 'protocol': 'loso', 'metric': 'Accuracy (%)'}
    return outcomes, source, snapshot


class _CsvDocument:
    """Keep every untouched CSV record byte-for-byte, including short rows."""
    def __init__(self, original):
        self.original = original
        text = original.decode('utf-8')
        self.bom = text.startswith('\ufeff')
        if self.bom:
            text = text[1:]
        lines = text.splitlines(keepends=True)
        reader = csv.reader(io.StringIO(text, newline=''))
        self.rows = []
        previous = 0
        for values in reader:
            raw = ''.join(lines[previous:reader.line_num])
            previous = reader.line_num
            self.rows.append({'values': values, 'raw': raw})
        if not self.rows:
            raise RegistrationSkipped('experiment log is empty')
        self.header = self.rows.pop(0)
        self.names = self.header['values']
        if self.names[:2] != ['model', 'method'] or len(set(self.names)) != len(self.names):
            raise RegistrationSkipped('experiment log must start with unique model,method columns')
        if any(name not in self.names for name in ('seed', *_DATA_COLUMNS)):
            raise RegistrationSkipped('experiment log is missing seed or dataset columns')
        self.newline = '\r\n' if self.header['raw'].endswith('\r\n') else '\n'
        for row in self.rows:
            if len(row['values']) > len(self.names):
                raise RegistrationSkipped('experiment log has a row wider than its header')

    def mapping(self, row):
        return dict(zip(self.names, row['values'] + [''] * (len(self.names) - len(row['values']))))

    def encode_row(self, values):
        stream = io.StringIO(newline='')
        csv.writer(stream, lineterminator=self.newline).writerow(values)
        return stream.getvalue()

    def replace(self, index, mapping):
        values = [mapping.get(name, '') for name in self.names]
        self.rows[index] = {'values': values, 'raw': self.encode_row(values)}

    def insert(self, index, mapping):
        values = [mapping.get(name, '') for name in self.names]
        self.rows.insert(index, {'values': values, 'raw': self.encode_row(values)})

    def bytes(self):
        records = [self.header, *self.rows]
        pieces = []
        for index, row in enumerate(records):
            raw = row['raw']
            if index < len(records) - 1 and raw and not raw.endswith(('\n', '\r')):
                raw += self.newline
            pieces.append(raw)
        return (('\ufeff' if self.bom else '') + ''.join(pieces)).encode('utf-8')


_ALIASES = {'epoch': 'epochs', 'epochs': 'epochs', 'lr': 'lr',
            'learning_rate': 'lr', 'batch': 'batch_size', 'batch_size': 'batch_size',
            't': 'temperature', 'temperature': 'temperature', 'λ': 'lam_kd',
            'lambda': 'lam_kd', 'lam_kd': 'lam_kd', 'lambda_kd': 'lam_kd',
            'weight_decay': 'weight_decay', 'wd': 'weight_decay',
            'lam_mmd': 'lam_mmd', 'lambda_mmd': 'lam_mmd', 'λ_mmd': 'lam_mmd',
            'lam_mi': 'lam_mi', 'teacher_seed': 'teacher_seed'}


def _format(value):
    if value is None:
        return ''
    if isinstance(value, (dict, list, tuple)):
        return _canonical(value)
    return str(value)


def _visible_parameters(names, outcome, snapshot):
    values, notes = {}, []
    actual = outcome['parameters']
    for name in names:
        if name in {'model', 'method', *_DATA_COLUMNS}:
            continue
        if name == 'seed':
            values[name] = '666,667,668'
            continue
        key = _ALIASES.get(name.strip().lower(), name)
        # KD temperature/lambda describe an active KD term, not parser defaults.
        if key in {'temperature', 'lam_kd'} and outcome['internal_method'] == 'MMD':
            values[name] = ''
        elif key == 'teacher_seed':
            values[name] = _format(snapshot['args'].get('teacher_seed')) or 'same-as-student'
        elif key in actual:
            values[name] = _format(actual[key])
        elif key in outcome['student_config']:
            values[name] = _format(outcome['student_config'][key])
        else:
            values[name] = ''
            notes.append(f'column {name!r} left blank: no verified student parameter mapping')
    return values, notes


def _profile(outcome, snapshot):
    # Shapes/source paths vary naturally across tasks; retain them per-dataset
    # below. All remaining model/training options stay in the merge identity.
    task_keys = {'dataset', 'dataset_name', 'num_classes', 'n_classes', 'channels',
                 'in_channels', 'n_channels', 'samples', 'n_times', 'sample_rate',
                 'sampling_rate', 'duration_seconds', 'env', 'size',
                 'input_root', 'source_directory', 'result_root', 'teacher_target_root'}
    def normalized(config):
        return {key: value for key, value in config.items() if key not in task_keys}
    dataset_source = outcome['dataset_config'].get('loso', {}).get('source_directory', '')
    source_kind = Path(dataset_source).name if dataset_source else 'canonical-dataset-default'
    if source_kind == 'wide_fre_band' or (outcome['dataset'] == 'AlexMI' and not dataset_source):
        source_kind = 'active-loso-sources'
    return {'teacher': outcome['teacher'], 'student': outcome['student'],
            'method': outcome['method'], 'parameters': outcome['parameters'],
            'student_config': normalized(outcome['student_config']),
            'teacher_config': normalized(outcome['teacher_config']),
            'teacher_seed': snapshot['args'].get('teacher_seed'),
            'source_kind': source_kind, 'protocol': 'loso', 'seeds': list(SEEDS),
            'code_sha256': snapshot.get('code_sha256', {}),
            'environment': snapshot.get('environment', {})}


def _section(document, teacher):
    starts = [(index, document.mapping(row)['model']) for index, row in enumerate(document.rows)
              if document.mapping(row)['model'] in TEACHERS.values()]
    matches = [index for index, model in starts if model == teacher]
    if len(matches) != 1:
        raise RegistrationSkipped(f'expected exactly one {teacher} section')
    start = matches[0]
    end = next((index for index, _ in starts if index > start), len(document.rows))
    initial = [document.mapping(row) for row in document.rows[start:start + 4]]
    if len(initial) != 4 or [row['model'] for row in initial] != [teacher, *STUDENTS.values()] \
            or any(row['method'] != 'none' for row in initial):
        raise RegistrationSkipped(f'{teacher} fixed teacher/student none block is malformed')
    return start, start + 4, end


def _locate_entry(document, entry, baseline_end, end, visible):
    remembered = entry['row_values']
    matches = []
    for index in range(baseline_end, end):
        current = document.mapping(document.rows[index])
        if all(name in current and current[name] == value for name, value in remembered.items()) \
                and all(not current[name] or
                        name in visible and _same(current[name], visible[name])
                        for name in current if name not in remembered):
            matches.append(index)
    return matches[0] if len(matches) == 1 else None


def _apply_outcome(document, registry, outcome, snapshot, source):
    _, baseline_end, end = _section(document, outcome['teacher'])
    visible, notes = _visible_parameters(document.names, outcome, snapshot)
    profile = _profile(outcome, snapshot)
    fingerprint = _hash(profile)
    column = outcome['dataset_column']
    score = f"{outcome['accuracy']:.2f}"
    index, owned = None, None
    for entry in registry['entries']:
        same_source = any(
            receipt['job_dir'] == source['job_dir'] and
            receipt['results_sha256'] == source['results_sha256']
            for registered in entry.get('datasets', {}).values()
            for receipt in registered.get('sources', []))
        if same_source and entry['profile']['method'] == outcome['method'] \
                and (entry['fingerprint'] != fingerprint or
                     _locate_entry(document, entry, baseline_end, end, visible) is None):
            raise RegistrationSkipped('this source was already registered but its CSV row was manually changed; review it before registering again')
        if entry['fingerprint'] != fingerprint:
            continue
        candidate = _locate_entry(document, entry, baseline_end, end, visible)
        if candidate is None:
            continue
        previous = entry.get('datasets', {}).get(outcome['dataset'])
        if previous is not None and previous['dataset_config'] != outcome['dataset_config']:
            continue
        if previous is not None and previous['student_config'] != outcome['student_config']:
            continue
        if previous is not None and previous['teacher_config'] != outcome['teacher_config']:
            continue
        if previous is not None and previous.get('data_provenance') != outcome['data_provenance']:
            continue
        current = document.mapping(document.rows[candidate])
        # A newly introduced parameter column cannot be filled by guessing
        # historical values. Existing ledger metadata does verify these values.
        if any(current[name] and not _same(current[name], value)
               for name, value in visible.items()):
            continue
        if current[column] and current[column] != score:
            raise RegistrationSkipped('refusing to overwrite a different recorded Accuracy')
        index, owned = candidate, entry
        break
    if index is None:
        # Only completely untouched template rows may be adopted without a
        # provenance ledger. Populated manual history remains untouched.
        for candidate in range(baseline_end, end):
            current = document.mapping(document.rows[candidate])
            if current['model'] == outcome['student'] and current['method'] == outcome['method'] \
                    and all(not value for key, value in current.items()
                            if key not in {'model', 'method'}):
                index = candidate
                break
    inserted = index is None
    if inserted:
        matching_method = [i for i in range(baseline_end, end)
                           if document.mapping(document.rows[i])['model'] == outcome['student']
                           and document.mapping(document.rows[i])['method'] == outcome['method']]
        matching_student = [i for i in range(baseline_end, end)
                            if document.mapping(document.rows[i])['model'] == outcome['student']]
        index = (matching_method[-1] + 1 if matching_method else
                 matching_student[-1] + 1 if matching_student else baseline_end)
        current = dict.fromkeys(document.names, '')
    else:
        current = document.mapping(document.rows[index])
    before = dict(current)
    current.update(model=outcome['student'], method=outcome['method'], **visible)
    current[column] = score
    if inserted:
        document.insert(index, current)
    elif current != before:
        document.replace(index, current)
    if owned is None:
        owned = {'fingerprint': fingerprint, 'profile': profile, 'datasets': {}}
        registry['entries'].append(owned)
    owned['row_values'] = current
    dataset_entry = owned['datasets'].setdefault(outcome['dataset'], {
        'dataset_config': outcome['dataset_config'], 'student_config': outcome['student_config'],
        'teacher_config': outcome['teacher_config'], 'data_provenance': outcome['data_provenance'],
        'sources': []})
    receipt = {**source, 'accuracy': outcome['accuracy'], 'seed_means': outcome['seed_means'],
               'fold_count': outcome['fold_count']}
    if not any(item['job_dir'] == source['job_dir'] and
               item['results_sha256'] == source['results_sha256'] for item in dataset_entry['sources']):
        dataset_entry['sources'].append(receipt)
    return {'teacher': outcome['teacher'], 'student': outcome['student'],
            'method': outcome['method'], 'dataset': outcome['dataset'],
            'accuracy': score, 'parameters': outcome['parameters'],
            'row': index + 2, 'fingerprint': fingerprint,
            'seed_means': outcome['seed_means'], 'fold_count': outcome['fold_count'],
            'changed': inserted or current != before}, notes


def _recover_transaction(transaction_path, registry_path, log_path):
    if not transaction_path.exists():
        return
    pending = _read_json(transaction_path)
    current = _bytes_hash(log_path.read_bytes())
    if current == pending['after_sha256']:
        _atomic_json(registry_path, pending['registry'])
    elif current != pending['before_sha256']:
        raise RegistrationSkipped('unfinished registration conflicts with a manual CSV edit')
    transaction_path.unlink()


def register_completed_job(job_dir, log_path=DEFAULT_LOG, *, registry_path=None):
    """Safely register a completed trial; return written/unchanged/skipped/error.

    ``registry_path`` may be supplied for isolated CPU tests or another queue.
    Every writer of the same CSV uses its stable ``.lock`` file. JSON receipts,
    content-addressed backups and a recoverable transaction stay in the report
    tree, not beside the user-maintained CSV.
    """
    folder, log_path = Path(job_dir).resolve(), Path(log_path).resolve()
    registry_path = (Path(registry_path).resolve() if registry_path is not None else
                     folder.parent.parent / 'experiment_log_registry.json')
    report = {'status': 'error', 'job_id': folder.name, 'log_path': str(log_path),
              'checked_at': _now(), 'entries': [], 'notes': []}
    try:
        outcomes, source, snapshot = _validate_job(folder)
        report['source'] = source
        if not outcomes:
            raise RegistrationSkipped('fixed none baselines are not automatically changed or duplicated')
        with _lock(log_path.with_name(log_path.name + '.lock')):
            transaction = registry_path.with_name(registry_path.name + '.pending.json')
            _recover_transaction(transaction, registry_path, log_path)
            original = log_path.read_bytes()
            document = _CsvDocument(original)
            registry = (_read_json(registry_path) if registry_path.exists() else
                        {'schema_version': 1, 'log_path': str(log_path), 'entries': []})
            if registry.get('schema_version') != 1 or registry.get('log_path') != str(log_path):
                raise RegistrationSkipped('experiment-log registry belongs to a different log/schema')
            for outcome in outcomes:
                entry, notes = _apply_outcome(document, registry, outcome, snapshot, source)
                report['entries'].append(entry)
                report['notes'].extend(notes)
            updated = document.bytes()
            before_hash, after_hash = _bytes_hash(original), _bytes_hash(updated)
            report.update(before_sha256=before_hash, after_sha256=after_hash,
                          registry_path=str(registry_path))
            registry.update(csv_sha256=after_hash, updated_at=_now())
            if original != updated:
                backup = registry_path.parent / 'experiment_log_backups' / f'{log_path.name}.{before_hash}.csv'
                if not backup.exists():
                    _atomic_bytes(backup, original)
                report['backup'] = str(backup)
                # Preserve a human edit that ignored our advisory lock.
                if log_path.read_bytes() != original:
                    raise RegistrationSkipped('CSV changed during registration; preserving the manual edit')
                _atomic_json(transaction, {'before_sha256': before_hash, 'after_sha256': after_hash,
                                           'registry': registry})
                if log_path.read_bytes() != original:
                    transaction.unlink()
                    raise RegistrationSkipped('CSV changed before replacement; preserving the manual edit')
                _atomic_bytes(log_path, updated)
            _atomic_json(registry_path, registry)
            transaction.unlink(missing_ok=True)
            report['status'] = 'written' if original != updated else 'unchanged'
    except RegistrationSkipped as exc:
        report.update(status='skipped', reason=str(exc))
    except Exception as exc:  # A reporting failure must never fail training.
        report.update(status='error', reason=f'{type(exc).__name__}: {exc}')
    try:
        _atomic_json(folder / 'experiment_log_registration.json', report)
    except Exception as exc:
        report['notes'].append(f'could not save registration receipt: {type(exc).__name__}: {exc}')
    return report
