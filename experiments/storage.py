"""Shared paths for experiment data and artifacts stored outside the checkout.

Historical paths in manifests may still name ``data_cache``, ``results`` or
``weights`` inside the project. Resolve them here before reading or writing.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CACHE_ROOT = Path('/data1/llx/data_cache')
RESULTS_ROOT = Path('/data1/llx/BigSmallCollab_results')
WEIGHTS_ROOT = Path('/data1/llx/BigSmallCollab_weights')
LFS_ROOT = Path('/data1/llx/BigSmallCollab_git_lfs')

_ROOTS = {
    'data_cache': DATA_CACHE_ROOT,
    'results': RESULTS_ROOT,
    'weights': WEIGHTS_ROOT,
}
_LFS_VERSION = b'version https://git-lfs.github.com/spec/v1'
_VERIFIED_BLOBS: set[tuple[str, int, int, int, str]] = set()


def external_path(path: str | os.PathLike[str]) -> Path:
    """Map historical project artifact paths to their external locations.

    Absolute paths outside this checkout, including existing ``/data1/llx``
    datasets and pretrained weights, retain their original location.
    """
    value = Path(path).expanduser()
    if value.is_absolute():
        try:
            relative = value.relative_to(PROJECT_ROOT)
        except ValueError:
            return value
    else:
        relative = value
    if relative.parts[:3] == ('test', 'qc', 'artifacts'):
        return (RESULTS_ROOT / 'qc_artifacts').joinpath(*relative.parts[3:])
    if relative.parts and relative.parts[0] in _ROOTS:
        return _ROOTS[relative.parts[0]].joinpath(*relative.parts[1:])
    return value


def _verified_blob(path: Path, oid: str, size: int) -> bool:
    if not path.is_file():
        return False
    stat = path.stat()
    if stat.st_size != size:
        return False
    identity = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ino, oid)
    if identity in _VERIFIED_BLOBS:
        return True
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != oid:
        return False
    _VERIFIED_BLOBS.add(identity)
    return True


def resolve_local_file(path: str | os.PathLike[str]) -> Path:
    """Return a real local file, resolving verified Git LFS pointers.

    Prefer external storage. The old file and local LFS object locations are
    accepted as a transition fallback while existing caches are being moved.
    """
    original = Path(path).expanduser()
    value = external_path(original)
    if not value.is_file() and value != original and original.is_file():
        value = original
    with value.open('rb') as handle:
        header = handle.read(4096)
    if not header.startswith(_LFS_VERSION):
        return value

    try:
        pointer = header.decode('ascii')
    except UnicodeDecodeError as error:
        raise RuntimeError(f'Invalid Git LFS pointer: {value}') from error
    lines = pointer.splitlines()
    if not lines or lines[0] != _LFS_VERSION.decode('ascii'):
        raise RuntimeError(f'Invalid Git LFS pointer version: {value}')
    oid_lines = [line for line in lines if line.startswith('oid ')]
    size_lines = [line for line in lines if line.startswith('size ')]
    if len(oid_lines) != 1 or len(size_lines) != 1:
        raise RuntimeError(f'Git LFS pointer must have one oid and size: {value}')
    oid_match = re.fullmatch(r'oid sha256:([0-9a-f]{64})', oid_lines[0])
    size_match = re.fullmatch(r'size (0|[1-9][0-9]*)', size_lines[0])
    if oid_match is None or size_match is None:
        raise RuntimeError(f'Invalid Git LFS oid or size: {value}')
    oid = oid_match.group(1)
    size = int(size_match.group(1))
    relative = Path('objects') / oid[:2] / oid[2:4] / oid
    candidates = (LFS_ROOT / relative, PROJECT_ROOT / '.git/lfs' / relative)
    for candidate in candidates:
        if _verified_blob(candidate, oid, size):
            return candidate
    locations = ', '.join(str(candidate) for candidate in candidates)
    raise RuntimeError(
        f'Local Git LFS object missing or invalid for {value}: '
        f'sha256={oid}, size={size}; searched {locations}'
    )


def require_external_output(path: str | os.PathLike[str]) -> Path:
    """Map a legacy output path and reject any destination inside the repo.

    Resolving symlinks also prevents an external-looking path from writing
    back into the checkout. This function does not create directories.
    """
    destination = external_path(path).resolve(strict=False)
    project = PROJECT_ROOT.resolve(strict=False)
    if destination == project or project in destination.parents:
        raise ValueError(
            f'禁止把数据、模型或实验产物保存在项目目录内：{destination}。'
            '请使用 /data1/llx 下的路径。'
        )
    return destination
