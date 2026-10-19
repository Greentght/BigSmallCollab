"""Durable completed-fold metrics for a single distillation invocation.

This ledger contains JSON configuration and metrics only. The caller supplies
the source/configuration identity for each cell; model artifacts stay in the
external artifact tree. A separate lock inode survives atomic JSON replacement.
"""
from __future__ import annotations

from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from experiments.storage import require_report_output


SCHEMA_VERSION = 1
_STATUSES = {'running', 'paused', 'completed', 'failed'}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode('utf-8')).hexdigest()


def _validate_row(row: dict[str, Any]) -> None:
    if not isinstance(row, dict):
        raise ValueError('a completed progress row must be a JSON object')
    if row.get('failure_status', 'complete') != 'complete':
        raise ValueError('progress records must have failure_status=complete')
    for name in ('acc', 'kappa'):
        if name in row:
            try:
                valid = not isinstance(row[name], bool) and math.isfinite(float(row[name]))
            except (TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                raise ValueError(f'completed progress row has nonfinite/invalid {name}')
    # Catch NaN or unsupported Python objects in every remaining field too.
    _canonical(row)


class ProgressBusyError(RuntimeError):
    """Another process already holds this invocation's progress lock."""


class ProgressStore:
    """Keep a nonblocking run lock and atomically persist each completed fold.

    ``cell`` may be a string or a JSON list. ``identity`` and ``run_spec`` are
    dictionaries of resolved settings and provenance. Changing either identity
    is an error, rather than an instruction to replace earlier results.
    """

    def __init__(self, path: str | os.PathLike[str], run_spec: dict[str, Any]):
        self.path = require_report_output(path)
        if not isinstance(run_spec, dict):
            raise ValueError('run_spec must be a JSON object')
        self.run_spec = json.loads(_canonical(run_spec))
        self.run_spec_hash = _hash(self.run_spec)
        self._lock_fd: int | None = None
        self._data: dict[str, Any] | None = None

    def __enter__(self) -> 'ProgressStore':
        if self._lock_fd is not None:
            raise RuntimeError('ProgressStore is already open')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + '.lock')
        flags = os.O_CREAT | os.O_RDWR | getattr(os, 'O_CLOEXEC', 0)
        flags |= getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(lock_path, flags, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ProgressBusyError(
                    f'another run holds the progress lock: {self.path}') from exc
            self._lock_fd = fd
            if self.path.exists():
                with self.path.open(encoding='utf-8') as handle:
                    data = json.load(handle)
                self._validate_document(data)
                self._data = data
            else:
                self._save({
                    'schema_version': SCHEMA_VERSION,
                    'run_spec': self.run_spec,
                    'run_spec_hash': self.run_spec_hash,
                    'status': 'running',
                    'total_cells': None,
                    'records': {},
                })
            return self
        except BaseException:
            self._data = None
            self._lock_fd = None
            os.close(fd)
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._lock_fd is not None:
            fd, self._lock_fd = self._lock_fd, None
            self._data = None
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    @staticmethod
    def _cell_key(cell: str | list[Any]) -> str:
        if not isinstance(cell, (str, list)):
            raise ValueError('cell must be a string or JSON list')
        return _canonical(cell)

    def _open_data(self) -> dict[str, Any]:
        if self._lock_fd is None or self._data is None:
            raise RuntimeError('ProgressStore must be used inside a with block')
        return self._data

    def _validate_document(self, data: Any) -> None:
        if not isinstance(data, dict) or type(data.get('schema_version')) is not int \
                or data['schema_version'] != SCHEMA_VERSION:
            raise ValueError(f'unsupported progress schema_version: {self.path}')
        if (data.get('run_spec_hash') != self.run_spec_hash
                or _canonical(data.get('run_spec')) != _canonical(self.run_spec)):
            raise ValueError(f'progress run_spec identity mismatch: {self.path}')
        if data.get('status') not in _STATUSES:
            raise ValueError('invalid progress status')
        records = data.get('records')
        if not isinstance(records, dict):
            raise ValueError('progress records must be a JSON object')
        total = data.get('total_cells')
        if total is not None and (type(total) is not int or total < len(records)):
            raise ValueError('invalid progress total_cells')
        for key, record in records.items():
            if not isinstance(record, dict) or key != self._cell_key(record.get('cell')):
                raise ValueError('invalid or duplicate progress cell')
            identity = record.get('identity')
            if not isinstance(identity, dict) or record.get('identity_hash') != _hash(identity):
                raise ValueError('invalid progress cell identity')
            _validate_row(record.get('row'))

    def _save(self, data: dict[str, Any]) -> None:
        # Encode before touching disk, and replace only after the complete file
        # is durable. If writing or replacing fails, the earlier JSON is intact.
        payload = json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False,
                             allow_nan=False) + '\n'
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode='w', encoding='utf-8', dir=self.path.parent,
                    prefix='.' + self.path.name + '.', suffix='.tmp',
                    delete=False) as handle:
                temp_path = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
            temp_path = None
            # Persist the rename as well as its contents on POSIX filesystems.
            directory_fd = os.open(self.path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self._data = data
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def get(self, cell: str | list[Any], identity: dict[str, Any]) -> dict[str, Any] | None:
        data = self._open_data()
        if not isinstance(identity, dict):
            raise ValueError('cell identity must be a JSON object')
        record = data['records'].get(self._cell_key(cell))
        if record is None:
            return None
        if record['identity_hash'] != _hash(identity):
            raise ValueError(f'progress cell identity mismatch: {cell!r}')
        return deepcopy(record['row'])

    def put(self, cell: str | list[Any], identity: dict[str, Any], row: dict[str, Any]) -> None:
        data = self._open_data()
        _validate_row(row)
        existing = self.get(cell, identity)
        if existing is not None:
            if _canonical(existing) != _canonical(row):
                raise ValueError(f'refusing to replace a completed progress row: {cell!r}')
            return
        updated = deepcopy(data)
        total = updated['total_cells']
        if total is not None and len(updated['records']) >= total:
            raise ValueError('completed progress cells would exceed total_cells')
        updated['records'][self._cell_key(cell)] = {
            'cell': json.loads(_canonical(cell)),
            'identity': json.loads(_canonical(identity)),
            'identity_hash': _hash(identity),
            'row': json.loads(_canonical(row)),
        }
        self._save(updated)

    def all_rows(self) -> list[dict[str, Any]]:
        return [deepcopy(record['row']) for record in self._open_data()['records'].values()]

    def all_records(self) -> list[dict[str, Any]]:
        """Return copied cells, identities and rows for resume preflight."""
        return [deepcopy(record) for record in self._open_data()['records'].values()]

    def set_status(self, status: str, total_cells: int | None = None) -> None:
        data = self._open_data()
        if status not in _STATUSES:
            raise ValueError(f'invalid progress status: {status!r}')
        updated = deepcopy(data)
        if total_cells is not None:
            if type(total_cells) is not int or total_cells < len(data['records']):
                raise ValueError('total_cells must be an integer >= completed record count')
            if data['total_cells'] is not None and total_cells != data['total_cells']:
                raise ValueError('refusing to change an established total_cells')
            updated['total_cells'] = total_cells
        if status == 'completed' and updated['total_cells'] is not None \
                and len(updated['records']) != updated['total_cells']:
            raise ValueError('cannot mark incomplete progress completed')
        updated['status'] = status
        self._save(updated)
