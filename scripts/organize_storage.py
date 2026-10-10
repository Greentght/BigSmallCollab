#!/usr/bin/env python
"""Apply the audited, user-authorized October 2026 storage cleanup.

Default is read-only. An explicit audit manifest, its SHA256, and --apply are
required to delete completed resume states and CodeBrain artifacts. Current
LOSO models, projectors, teacher targets, shared datasets and Git history are
outside the deletion scope. This is a maintenance operation, not a trainer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.storage import (DATA_CACHE_ROOT, LEGACY_LOGS_ROOT,
                                 PRETRAINED_WEIGHTS_ROOT, PROJECT_DATA_ROOT,
                                 REPORTS_ROOT, RESULTS_ROOT, WEIGHTS_ROOT,
                                 external_path, require_report_output)

TEXT_SUFFIXES = {'.json', '.jsonl', '.csv', '.md', '.yaml', '.yml', '.sha256', '.txt', '.log'}
DUPLICATE_METADATA = {
    'manifest.json', 'cache_manifest.json', 'input_manifest.json',
    'split_manifest.json', 'config_resolved.json', 'config_resolved_all.json',
    'status.json', 'progress.json', 'metadata.json', 'dispatch_status.json',
    'train_history.csv', 'history.csv', 'history.jsonl', 'epoch_metrics.csv',
    'epoch_mask_metrics.csv', 'queue_events.jsonl',
    'result.json', 'execution_provenance.json', 'config_resolved.yaml',
    'config_resolved.yml', 'run_manifest.csv', 'git_status_before.txt',
}
FEWSHOT_TARGET_DIRECTORIES = (
    'bnci2014001_4_supplement/stage5_cbramod200/teacher_artifacts',
    'task_feature_logit_kd_six_pairs_seed666_cbramod200/teacher_artifacts',
    'task_feature_logit_kd_six_pairs_seed666_cbramod200_originalhead/teacher_artifacts',
)
PILOT_PROTOCOL = 'sample_utility_adaptive_rl_loso_pilot_v1'


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def inventory(path: Path) -> dict:
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                inode=stat.st_ino, device=stat.st_dev)


def check_record(record: dict) -> Path:
    path = Path(record['path'])
    if inventory(path) != {key: record[key] for key in
                           ('path', 'size', 'mtime_ns', 'inode', 'device')}:
        raise RuntimeError(f'Audited file changed: {path}')
    return path


def within(path: Path, root: Path) -> bool:
    # Check the entry's lexical location; never follow a link when deleting it.
    return path.is_absolute() and path != root and root in path.parents


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def disk_snapshot() -> dict:
    paths = (PROJECT_DATA_ROOT, ROOT, REPORTS_ROOT, ROOT / '.git',
             DATA_CACHE_ROOT, RESULTS_ROOT, WEIGHTS_ROOT)
    occupied = {}
    for path in paths:
        if path.exists():
            output = subprocess.check_output(['du', '-s', '-B1', str(path)], text=True)
            occupied[str(path)] = int(output.split()[0])
    stat = os.statvfs(PROJECT_DATA_ROOT)
    return dict(allocated_bytes=occupied,
                filesystem_available_bytes=stat.f_bavail * stat.f_frsize)


def assert_idle() -> None:
    rows = subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True).splitlines()
    active = []
    for row in rows:
        parts = row.strip().split(None, 1)
        if len(parts) != 2 or int(parts[0]) == os.getpid():
            continue
        command = parts[1]
        executable = command.split()[0]
        if ('python' in Path(executable).name
                and any(token in command for token in
                        ('experiments/finetune/', 'experiments/distill/',
                         'dispatch_loso', 'run_loso_distillation', 'train_codebrain'))):
            active.append(row.strip())
    if active:
        raise RuntimeError('Training/dispatcher is active; stop cleanup:\n' + '\n'.join(active))


def preflight(data: dict) -> None:
    assert_idle()
    if Path(data['root']) != PROJECT_DATA_ROOT:
        raise ValueError('Audit does not describe the canonical project data root')
    if data['training_state_blocked'] or data['codebrain_model_blocked']:
        raise ValueError('Audit contains incomplete or blocked candidates')
    allowed_codebrain = [Path(row['path']) for row in data['codebrain_related_directories']]
    for root in allowed_codebrain:
        if not within(root, RESULTS_ROOT) or 'codebrain' not in root.name:
            raise ValueError(f'Unexpected CodeBrain root: {root}')
    for item in data['training_state_candidates']:
        path = check_record(item)
        if not within(path, RESULTS_ROOT) or path.name != 'training_state.pt':
            raise ValueError(f'Unexpected resume-state deletion: {path}')
        if (path.parent / 'completed_state.pt').exists():
            raise RuntimeError(f'Completion sidecar already exists: {path.parent}')
        evidence = item['completion_evidence']
        result = read_json(Path(evidence['result_path']))
        status = read_json(Path(evidence['status_path']))
        with Path(evidence['history_path']).open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        epochs = int(result['resolved_training_config']['epochs'])
        if (result.get('status') != 'complete' or status.get('status') != 'complete'
                or epochs != evidence['configured_epochs'] or len(rows) != epochs
                or int(rows[-1]['epoch']) != epochs):
            raise RuntimeError(f'Completion proof changed: {path}')
        for artifact in evidence['final_artifacts']:
            check_record(artifact)
    for item in data['codebrain_all_files']:
        path = check_record(item)
        if not any(within(path, root) for root in allowed_codebrain):
            raise ValueError(f'Unexpected CodeBrain deletion: {path}')
    for item in data['codebrain_model_candidates']:
        check_record(item)
        proof = item['completion_evidence']
        metrics = read_json(Path(proof['metrics_path']))
        if (metrics.get('status') != 'complete'
                or metrics.get('epochs_completed') != proof['configured_epochs']):
            raise RuntimeError(f'CodeBrain completion proof changed: {item["path"]}')


def preflight_moves(links: list[dict]) -> None:
    old_artifacts = RESULTS_ROOT / 'artifacts'
    new_artifacts = DATA_CACHE_ROOT / 'artifacts'
    moves = (
        (old_artifacts, new_artifacts),
        (RESULTS_ROOT / 'distill/loso_five_settings_kd_feature_warmup10_v1/teacher_cache',
         DATA_CACHE_ROOT / 'teacher_targets/loso_five_settings_kd_feature_warmup10_v1'),
        *((RESULTS_ROOT / 'distill' / relative,
           DATA_CACHE_ROOT / 'teacher_targets/fewshot' / relative)
          for relative in FEWSHOT_TARGET_DIRECTORIES),
    )
    for source, destination in moves:
        if (not source.is_dir() or source.is_symlink() or destination.exists()
                or destination.is_symlink()):
            raise RuntimeError(f'Unexpected migration paths: {source} -> {destination}')
    for entry in links:
        path = Path(entry['link'])
        old, new = Path(entry['old_target']), Path(entry['new_target'])
        if (not within(path, RESULTS_ROOT / 'distill') or not within(old, old_artifacts)
                or new != new_artifacts / old.relative_to(old_artifacts)
                or '..' in path.parts or '..' in old.parts or '..' in new.parts
                or not path.is_symlink() or path.resolve() != old or not old.is_file()):
            raise RuntimeError(f'Unreviewed migration symlink: {entry}')
    for name in ('mirepnet.pth', 'cbramod.pth', 'labram-base.pth'):
        path, target = WEIGHTS_ROOT / name, PRETRAINED_WEIGHTS_ROOT / name
        if not target.is_file():
            raise RuntimeError(f'Shared pretrained weight missing: {target}')
        if path.is_symlink() and path.resolve() != target.resolve():
            raise RuntimeError(f'Unexpected official weight alias: {path}')
    codebrain = WEIGHTS_ROOT / 'codebrain.pth'
    if codebrain.is_symlink():
        raise RuntimeError('CodeBrain weight is a link to an unreviewed target')
    if (REPORTS_ROOT / 'archive/codebrain/codebrain_records.zip').exists():
        raise RuntimeError('Existing CodeBrain report archive; inspect before continuing')
    pilot = WEIGHTS_ROOT / PILOT_PROTOCOL
    for path in pilot.rglob('teacher_train.npz'):
        target = DATA_CACHE_ROOT / 'teacher_targets' / PILOT_PROTOCOL / path.relative_to(pilot)
        if path.is_symlink() or target.exists() or target.is_symlink():
            raise RuntimeError(f'Unexpected pilot target migration: {path} -> {target}')


class Organizer:
    def __init__(self, data: dict, manifest: Path, link_manifest: Path, report: Path):
        self.data, self.report = data, report
        self.value = dict(status='in_progress', started_unix=time.time(),
                          audit_manifest_sha256=digest(manifest), actions=[],
                          symlink_inventory_sha256=digest(link_manifest),
                          deleted_bytes=0, retained_completion_bytes=0,
                          archived_report_bytes=0, before_disk=disk_snapshot())
        self.save()

    def save(self) -> None:
        atomic_json(self.report, self.value)

    def event(self, action: str, **fields) -> None:
        self.value['actions'].append(dict(action=action, **fields))

    def compact_states(self) -> None:
        import torch
        from experiments.finetune.run_loso_config_alignment import compact_completed_state
        total = len(self.data['training_state_candidates'])
        for index, record in enumerate(self.data['training_state_candidates'], 1):
            path = check_record(record)
            state = torch.load(path, map_location='cpu', weights_only=False)
            compact = compact_completed_state(state)
            if int(compact['next_epoch']) != record['completion_evidence']['configured_epochs']:
                raise RuntimeError(f'Wrong terminal epoch: {path}')
            if not all(key in compact['rng_state'] for key in ('python', 'numpy', 'torch_cpu')):
                raise RuntimeError(f'Incomplete terminal RNG: {path}')
            for artifact in record['completion_evidence']['final_artifacts']:
                check_record(artifact)
            destination = path.with_name('completed_state.pt')
            temporary = destination.with_suffix('.pt.partial')
            torch.save(compact, temporary)
            restored = torch.load(temporary, map_location='cpu', weights_only=False)
            if set(restored) != set(compact) or restored.get('completed') is not True:
                raise RuntimeError(f'Invalid compact completion state: {temporary}')
            os.replace(temporary, destination)
            check_record(record).unlink()
            self.value['deleted_bytes'] += record['size']
            self.value['retained_completion_bytes'] += destination.stat().st_size
            self.event('compact_completed_resume', source=str(path),
                       destination=str(destination), removed_bytes=record['size'],
                       retained_bytes=destination.stat().st_size,
                       retained_sha256=digest(destination))
            del state, compact, restored
            if index % 25 == 0 or index == total:
                self.save()
                print(f'Completed resume states: {index}/{total}', flush=True)

    def archive_codebrain(self) -> None:
        folder = REPORTS_ROOT / 'archive/codebrain'
        archive = require_report_output(folder / 'codebrain_records.zip')
        if archive.exists():
            raise RuntimeError(f'Refusing to overwrite an existing archive: {archive}')
        folder.mkdir(parents=True, exist_ok=True)
        entries = []
        for record in self.data['codebrain_all_files']:
            path = check_record(record)
            if path.suffix.lower() in TEXT_SUFFIXES:
                entries.append((path, 'external/' + str(path.relative_to(PROJECT_DATA_ROOT))))
        local = REPORTS_ROOT / 'codebrain'
        tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT,
                                              text=True).split('\0'))
        for path in sorted(local.rglob('*')):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                if str(path.relative_to(ROOT)) in tracked or path.is_symlink():
                    raise RuntimeError(f'Not an untracked real CodeBrain report: {path}')
                entries.append((path, 'checkout/' + str(path.relative_to(ROOT))))
        contents = []
        temporary = archive.with_suffix('.zip.partial')
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
            for path, member in entries:
                raw = path.read_bytes()
                contents.append(dict(source=str(path), member=member, size=len(raw),
                                     sha256=hashlib.sha256(raw).hexdigest()))
                handle.writestr(member, raw)
            handle.writestr('archive_manifest.json', json.dumps(dict(
                audit_manifest_sha256=self.value['audit_manifest_sha256'],
                scope='CodeBrain text reports, configurations and histories; no EEG or weights',
                files=contents), indent=2, sort_keys=True) + '\n')
        with zipfile.ZipFile(temporary) as handle:
            for entry in contents:
                if hashlib.sha256(handle.read(entry['member'])).hexdigest() != entry['sha256']:
                    raise RuntimeError(f'Archive integrity failed: {entry["member"]}')
        os.replace(temporary, archive)
        score_csv = require_report_output(folder / 'codebrain_scores.csv')
        columns = ('run_id', 'dataset', 'seed', 'subject', 'initialization', 'accuracy',
                   'balanced_accuracy', 'kappa', 'epochs_completed', 'elapsed_seconds',
                   'protocol', 'formal_result', 'metrics_path')
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=columns)
        writer.writeheader()
        for entry in self.data['codebrain_score_records']:
            metrics = read_json(Path(entry['path']))
            row = {key: metrics.get(key) for key in columns}
            row['metrics_path'] = entry['path']
            writer.writerow(row)
        score_csv.write_text(buffer.getvalue())
        for path, member in entries:
            if member.startswith('checkout/'):
                self.value['deleted_bytes'] += path.stat().st_size
                path.unlink()
        self.value['archived_report_bytes'] += archive.stat().st_size + score_csv.stat().st_size
        self.event('archive_codebrain_text', archive=str(archive), sha256=digest(archive),
                   text_files=len(contents), score_rows=len(self.data['codebrain_score_records']),
                   score_csv=str(score_csv), bytes=archive.stat().st_size)
        self.save()
        print(f'CodeBrain records archived: {len(contents)} text files', flush=True)

    def remove_codebrain(self) -> None:
        for record in self.data['codebrain_all_files']:
            path = check_record(record)
            path.unlink()
            self.value['deleted_bytes'] += record['size']
            self.event('delete_codebrain_artifact', path=str(path), bytes=record['size'])
        weight = WEIGHTS_ROOT / 'codebrain.pth'
        if weight.exists():
            if weight.is_symlink():
                raise RuntimeError('CodeBrain weight is a link; refusing to delete an unreviewed target')
            size = weight.stat().st_size
            weight.unlink()
            self.value['deleted_bytes'] += size
            self.event('delete_codebrain_pretrained', path=str(weight), bytes=size)
        self.save()
        print('CodeBrain model and run artifacts removed', flush=True)

    def relocate_tree(self, source: Path, destination: Path, legacy_alias=False) -> None:
        if not source.exists() or source.is_symlink() or destination.exists():
            raise RuntimeError(f'Unexpected migration paths: {source} -> {destination}')
        count, size = 0, 0
        for path in source.rglob('*'):
            if path.is_file() and not path.is_symlink():
                count += 1
                size += path.stat().st_size
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Both locations are on the same filesystem: rename keeps bytes/inodes.
        source.rename(destination)
        if legacy_alias:
            source.symlink_to(destination, target_is_directory=True)
        self.event('relocate_cache', source=str(source), destination=str(destination),
                   files=count, bytes=size, legacy_external_alias=legacy_alias)
        self.save()

    def relocate_caches(self, links: list[dict]) -> None:
        for entry in links:
            path = Path(entry['link'])
            if not path.is_symlink() or str(path.resolve()) != entry['old_target']:
                raise RuntimeError(f'Changed legacy target symlink: {path}')
        self.relocate_tree(RESULTS_ROOT / 'artifacts', DATA_CACHE_ROOT / 'artifacts',
                           legacy_alias=True)
        self.relocate_tree(
            RESULTS_ROOT / 'distill/loso_five_settings_kd_feature_warmup10_v1/teacher_cache',
            DATA_CACHE_ROOT / 'teacher_targets/loso_five_settings_kd_feature_warmup10_v1',
            legacy_alias=True)
        for entry in links:
            path, target = Path(entry['link']), Path(entry['new_target'])
            if not target.is_file():
                raise RuntimeError(f'Migrated symlink target missing: {target}')
            temporary = path.with_name(path.name + '.relocated')
            temporary.symlink_to(target)
            os.replace(temporary, path)
            self.event('retarget_external_link', path=str(path), target=str(target))
        for name in ('mirepnet.pth', 'cbramod.pth', 'labram-base.pth'):
            path = WEIGHTS_ROOT / name
            if path.is_symlink():
                target = PRETRAINED_WEIGHTS_ROOT / name
                if path.resolve() != target.resolve() or not target.is_file():
                    raise RuntimeError(f'Unexpected official weight alias: {path}')
                path.unlink()
                self.event('remove_pretrained_alias', path=str(path), retained=str(target))
        self.save()
        print('Artifact caches and teacher targets consolidated', flush=True)

    def move_local_records(self, source: Path, destination: Path) -> None:
        moved = identical = bytes_removed = 0
        for path in sorted(source.rglob('*')):
            if not path.is_file() or path.is_symlink():
                continue
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if path.stat().st_size == target.stat().st_size and digest(path) == digest(target):
                    bytes_removed += path.stat().st_size
                    path.unlink()
                    identical += 1
                    continue
                target = target.with_name(target.name + '.checkout-' + digest(path)[:12])
                if target.exists():
                    raise RuntimeError(f'Conflicting local record: {path}')
            shutil.copy2(path, target)
            if digest(path) != digest(target):
                raise RuntimeError(f'Copied log/report failed integrity: {path}')
            path.unlink()
            moved += 1
        self.value['deleted_bytes'] += bytes_removed
        self.event('move_local_operational_records', source=str(source), destination=str(destination),
                   moved=moved, identical_duplicates_removed=identical,
                   duplicate_bytes_removed=bytes_removed)
        self.save()

    def relocate_auxiliary_teacher_caches(self) -> None:
        for relative in FEWSHOT_TARGET_DIRECTORIES:
            source = RESULTS_ROOT / 'distill' / relative
            destination = DATA_CACHE_ROOT / 'teacher_targets/fewshot' / relative
            # Keep the raw historical path for manifests and readers that do
            # not call the resolver; only the physical directory changes.
            self.relocate_tree(source, destination, legacy_alias=True)
        pilot = WEIGHTS_ROOT / PILOT_PROTOCOL
        paths = sorted(pilot.rglob('teacher_train.npz'))
        size = 0
        for path in paths:
            if path.is_symlink():
                raise RuntimeError(f'Pilot target was already moved: {path}')
            target = (DATA_CACHE_ROOT / 'teacher_targets' / PILOT_PROTOCOL
                      / path.relative_to(pilot))
            if target.exists() or target.is_symlink():
                raise RuntimeError(f'Pilot target destination exists: {target}')
            before = inventory(path)
            sha = digest(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)
            path.symlink_to(target)
            if digest(target) != sha or target.stat().st_size != before['size']:
                raise RuntimeError(f'Moved pilot target changed: {target}')
            size += before['size']
            self.event('relocate_pilot_teacher_target', source=str(path),
                       destination=str(target), bytes=before['size'], sha256=sha,
                       legacy_external_alias=True)
        self.save()
        print(f'Auxiliary teacher targets consolidated; pilot files: {len(paths)} ({size} bytes)',
              flush=True)

    def clean_local_duplicates(self) -> None:
        count, size, skipped = 0, 0, []
        for path in sorted(REPORTS_ROOT.rglob('*')):
            if (not path.is_file() or path.is_symlink()
                    or (path.name not in DUPLICATE_METADATA and not path.name.endswith('.lock'))):
                continue
            external = external_path(path)
            if (external == path or not external.is_file()
                    or path.stat().st_size != external.stat().st_size
                    or digest(path) != digest(external)):
                skipped.append(str(path))
                continue
            file_size = path.stat().st_size
            path.unlink()
            count += 1
            size += file_size
        self.value['deleted_bytes'] += size
        self.event('remove_identical_checkout_training_metadata', files=count, bytes=size,
                   retained_external=True, changed_or_unique_files_retained=skipped)
        self.save()
        print(f'Identical checkout training records removed: {count}', flush=True)

    def remove_runtime_debris(self) -> None:
        tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT,
                                              text=True).split('\0'))
        count, size = 0, 0
        protected = {'.git', '.agents', '.codex', '.aws'}
        for base, directories, names in os.walk(ROOT):
            directories[:] = [name for name in directories if name not in protected]
            for name in names:
                path = Path(base) / name
                relative = path.relative_to(ROOT)
                remove = (name.endswith('.pyc') or '__pycache__' in relative.parts
                          or '.pytest_cache' in relative.parts
                          or relative.parts[:2] == ('.deps', 'codebrain'))
                if remove and str(relative) not in tracked and not path.is_symlink():
                    size += path.stat().st_size
                    path.unlink()
                    count += 1
        for base, directories, _ in os.walk(ROOT, topdown=False):
            current = Path(base)
            relative = current.relative_to(ROOT)
            if protected.intersection(relative.parts):
                continue
            if (current.name == '__pycache__' and not current.is_symlink()
                    and not any(current.iterdir())):
                current.rmdir()
        self.value['deleted_bytes'] += size
        self.event('remove_untracked_runtime_debris', files=count, bytes=size)
        # Remove empty ignored directories, never .git or the source packages.
        for top in (REPORTS_ROOT, ROOT / 'logs', ROOT / 'test/qc/artifacts',
                    ROOT / '.deps/codebrain', ROOT / '.pytest_cache',
                    RESULTS_ROOT / 'codebrain', DATA_CACHE_ROOT / 'artifacts'):
            if not top.exists() or top.is_symlink():
                continue
            for base, _, _ in os.walk(top, topdown=False, followlinks=False):
                path = Path(base)
                if not path.is_symlink() and not any(path.iterdir()):
                    path.rmdir()
        self.save()

    def apply(self, links: list[dict]) -> None:
        try:
            assert_idle()
            self.archive_codebrain()
            self.compact_states()
            self.remove_codebrain()
            self.relocate_caches(links)
            self.relocate_auxiliary_teacher_caches()
            self.move_local_records(ROOT / 'logs', LEGACY_LOGS_ROOT)
            self.move_local_records(ROOT / 'test/qc/artifacts', RESULTS_ROOT / 'qc_artifacts')
            self.clean_local_duplicates()
            self.remove_runtime_debris()
            self.value.update(status='complete', finished_unix=time.time(),
                              after_disk=disk_snapshot())
        except Exception as error:
            self.value.update(status='failed', failure=repr(error), finished_unix=time.time())
            raise
        finally:
            self.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-manifest', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--symlink-inventory', type=Path, required=True)
    parser.add_argument('--report', type=Path, default=REPORTS_ROOT / 'storage_cleanup/cleanup_report.json')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if digest(args.audit_manifest) != args.expected_sha256:
        raise ValueError('Audit manifest SHA256 does not match the reviewed manifest')
    data = read_json(args.audit_manifest)
    preflight(data)
    links = json.loads(args.symlink_inventory.read_text())
    preflight_moves(links)
    print(json.dumps(dict(audited=data['counts'], apply=args.apply,
                          external_links_to_retarget=len(links)), indent=2), flush=True)
    if args.apply:
        report = require_report_output(args.report)
        if report.exists():
            raise RuntimeError(f'Existing cleanup report; inspect before continuing: {report}')
        migration = PROJECT_DATA_ROOT / 'migrations/storage_cleanup/20261010'
        migration.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.audit_manifest, migration / args.audit_manifest.name)
        shutil.copy2(args.symlink_inventory, migration / args.symlink_inventory.name)
        Organizer(data, args.audit_manifest, args.symlink_inventory, report).apply(links)


if __name__ == '__main__':
    main()
