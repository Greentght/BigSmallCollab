"""External training artifacts and checkout-local result reports.

Historical paths in manifests may still name ``data_cache``, ``results`` or
``weights`` inside the project. Resolve them here before reading or writing.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_DATA_ROOT = Path('/data1/llx')
PROJECT_DATA_ROOT = SHARED_DATA_ROOT / 'BigSmallcollab'
DATA_CACHE_ROOT = PROJECT_DATA_ROOT / 'cache'
RESULTS_ROOT = PROJECT_DATA_ROOT / 'results'
REPORTS_ROOT = PROJECT_ROOT / 'results'
WEIGHTS_ROOT = PROJECT_DATA_ROOT / 'weights'
LFS_ROOT = PROJECT_DATA_ROOT / 'git_lfs'
BNCI14001_SOURCE_ROOT = SHARED_DATA_ROOT / 'BNCI2014001/broadband_0p1_75hz'

# Versioned execution specifications have been removed from the active config
# tree. Preserve read compatibility for in-flight jobs and historical manifests;
# the current KD specification is byte-identical at its canonical location.
_CANONICAL_CONFIGS = {
    'loso_source_refresh_distillation_004_5001_v1.yaml':
        PROJECT_ROOT / 'configs/experiments/loso_distillation.yaml',
    'loso_source_refresh_004_5001_v1.yaml':
        PROJECT_ROOT / 'configs/protocols/loso.yaml',
    'bnci14001_wideband_loso_v3.yaml':
        PROJECT_ROOT / 'configs/protocols/loso_001.yaml',
}
_RETIRED_CONFIGS = {
    name: PROJECT_DATA_ROOT / 'migrations/retired_loso_configs' / name
    for name in (
        'loso_five_datasets_v1.yaml', 'loso_distillation_v1.yaml',
        'loso_config_alignment_v2.yaml',
        'loso_cbramod_0014_npy_source_control_20261006.yaml',
        'loso_001_all_models_wideband_v1.yaml',
    )
}

_SOURCE_RELATIVE = Path('loso_source_v3/BNCI2014001')
_LEGACY_ABSOLUTE_ROOTS = (
    (SHARED_DATA_ROOT / 'data_cache' / _SOURCE_RELATIVE, BNCI14001_SOURCE_ROOT),
    (DATA_CACHE_ROOT / _SOURCE_RELATIVE, BNCI14001_SOURCE_ROOT),
    (SHARED_DATA_ROOT / 'data_cache', DATA_CACHE_ROOT),
    (SHARED_DATA_ROOT / 'BigSmallCollab_results', RESULTS_ROOT),
    (SHARED_DATA_ROOT / 'BigSmallCollab_weights', WEIGHTS_ROOT),
    (SHARED_DATA_ROOT / 'BigSmallCollab_git_lfs', LFS_ROOT),
    (PROJECT_ROOT / '.git/lfs', LFS_ROOT),
)

_ROOTS = {
    'data_cache': DATA_CACHE_ROOT,
    'results': RESULTS_ROOT,
    'weights': WEIGHTS_ROOT,
}
_LFS_VERSION = b'version https://git-lfs.github.com/spec/v1'
_VERIFIED_BLOBS: set[tuple[str, int, int, int, str]] = set()


def external_path(path: str | os.PathLike[str]) -> Path:
    """Map historical project artifact paths to their external locations.

    Existing shared datasets and pretrained weights retain their locations.
    The earlier external flat directories also map to the new layout, so
    historical manifest bytes and training input hashes remain unchanged.
    """
    value = Path(path).expanduser()
    if value.is_absolute():
        for old_root, new_root in _LEGACY_ABSOLUTE_ROOTS:
            try:
                return new_root / value.relative_to(old_root)
            except ValueError:
                pass
        try:
            relative = value.relative_to(PROJECT_ROOT)
        except ValueError:
            return value
    else:
        relative = value
    if relative.parts[:2] == ('configs', 'reproductions'):
        replacement = (_CANONICAL_CONFIGS | _RETIRED_CONFIGS).get(relative.name)
        if replacement is not None:
            return replacement
    if relative.parts[:3] == ('data_cache', 'loso_source_v3', 'BNCI2014001'):
        return BNCI14001_SOURCE_ROOT.joinpath(*relative.parts[3:])
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


def require_report_output(path: str | os.PathLike[str]) -> Path:
    """Resolve a user-facing report inside the checkout's real results folder.

    Report exporters use this explicitly instead of the historical external
    artifact mapper. No directories are created by this function.
    """
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = (PROJECT_ROOT / value if value.parts and value.parts[0] == 'results'
                 else REPORTS_ROOT / value)
    project = PROJECT_ROOT.resolve(strict=False)
    reports = REPORTS_ROOT.resolve(strict=False)
    destination = value.resolve(strict=False)
    if REPORTS_ROOT.is_symlink() or reports != project / 'results':
        raise ValueError('报告目录必须是项目内真实的 results/ 目录，不能链接到外部存储。')
    if destination == reports or reports not in destination.parents:
        raise ValueError(f'结果报告必须保存在 {reports} 下：{destination}')
    if destination.suffix.lower() not in {'.xlsx', '.csv', '.json', '.md', '.html', '.pdf', '.png', '.svg'}:
        raise ValueError(f'该文件不是汇总报告；数据、缓存和模型请使用外部存储：{destination}')
    return destination
