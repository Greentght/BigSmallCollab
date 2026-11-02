#!/usr/bin/env python3
"""Recover and aggregate completed CBraMod→IFNet T2S fold evaluations.

The existing evaluator wrote target-projection diagnostic CSV bytes under the
fixed-projector NPZ filename. This entry point restores those deterministic
projectors from their recorded seeds and places each CSV at its intended path,
then aggregates saved fold predictions. It does not train or prune weights.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.distill import run_cbramod_t2s_full_feature_loso as runner
from experiments.storage import REPORTS_ROOT, require_report_output


EXPECTED_SEEDS = (666, 667, 668)
EXPECTED_FOLDS = 9


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight(config: dict) -> tuple[list[Path], list[str]]:
    checkpoints: list[Path] = []
    fold_ids: list[str] = []
    for seed in EXPECTED_SEEDS:
        run_id = f"{config['storage']['external_run_id_prefix']}{seed}"
        for fold in range(EXPECTED_FOLDS):
            roots = runner._roots(run_id, seed, fold)
            complete_path = roots["run_result"] / "fold_complete.json"
            if not complete_path.is_file():
                raise FileNotFoundError(f"missing completed training fold: {complete_path}")
            complete = runner._read_json(complete_path)
            if (complete.get("status") != "trained"
                    or int(complete.get("seed", -1)) != seed
                    or int(complete.get("fold_zero_based", -1)) != fold):
                raise RuntimeError(f"invalid training completion record: {complete_path}")
            for condition in runner.CONDITIONS:
                manifest_path = roots["run_result"] / f"student_{condition}.json"
                if not manifest_path.is_file():
                    raise FileNotFoundError(f"missing student manifest: {manifest_path}")
                manifest = runner._read_json(manifest_path)
                if (manifest.get("status") != "complete"
                        or manifest.get("run_fingerprint") != complete.get("run_fingerprint")
                        or int(manifest.get("final_epoch", -1)) != 100):
                    raise RuntimeError(f"invalid student manifest: {manifest_path}")
                checkpoint = Path(manifest["checkpoint"])
                if (not checkpoint.is_file()
                        or file_sha256(checkpoint) != manifest.get("checkpoint_sha256")):
                    raise RuntimeError(f"missing/changed student checkpoint: {checkpoint}")
                checkpoints.append(checkpoint)
            fold_ids.append(f"seed_{seed}/subject_{fold + 1:02d}")
    if len(fold_ids) != len(EXPECTED_SEEDS) * EXPECTED_FOLDS:
        raise RuntimeError(f"expected 27 folds, found {len(fold_ids)}")
    return checkpoints, fold_ids


def restore_misplaced_projection_diagnostics(config: dict) -> int:
    """Recover CSV diagnostics and deterministic projectors from completed folds.

    The existing evaluator shadows its result CSV path with the fixed-projector
    NPZ path before writing diagnostics. In that case the NPZ pathname contains
    the diagnostic CSV bytes, while fold_evaluation.json records their hash.
    Preserve those bytes as the correctly named diagnostic file, then recreate
    the fixed projection from its recorded deterministic seed.
    """
    plans = []
    for seed in EXPECTED_SEEDS:
        run_id = f"{config['storage']['external_run_id_prefix']}{seed}"
        for fold in range(EXPECTED_FOLDS):
            roots = runner._roots(run_id, seed, fold)
            result_root, cache_root = roots["run_result"], roots["run_cache"]
            eval_path = result_root / "fold_evaluation.json"
            metrics_path = result_root / "fold_evaluation_metrics.csv"
            predictions_path = result_root / "fold_predictions.csv"
            diagnostics_path = result_root / "test_projection_diagnostics.csv"
            projector_path = cache_root / "fixed_teacher_to_IFNet_P.npz"
            projector_meta_path = cache_root / "fixed_teacher_to_IFNet_P.json"
            required = (eval_path, metrics_path, predictions_path,
                        projector_path, projector_meta_path)
            if not all(path.is_file() for path in required):
                raise FileNotFoundError(
                    f"cannot recover completed target evaluation for seed={seed}, fold={fold + 1}")
            evaluation = runner._read_json(eval_path)
            complete = runner._read_json(result_root / "fold_complete.json")
            if (evaluation.get("status") != "complete"
                    or evaluation.get("run_fingerprint") != complete.get("run_fingerprint")
                    or evaluation.get("metrics_sha256") != file_sha256(metrics_path)
                    or evaluation.get("predictions_sha256") != file_sha256(predictions_path)):
                raise RuntimeError(f"saved fold evaluation identity/hash mismatch: {eval_path}")
            metadata = runner._read_json(projector_meta_path)
            if metadata.get("fingerprint") != complete.get("run_fingerprint"):
                raise RuntimeError(f"projector fingerprint mismatch: {projector_meta_path}")

            if diagnostics_path.is_file():
                if evaluation.get("projection_diagnostics_sha256") != file_sha256(diagnostics_path):
                    raise RuntimeError(f"saved projection diagnostics hash mismatch: {diagnostics_path}")
                continue

            current_hash = file_sha256(projector_path)
            recorded_diagnostic_hash = evaluation.get("projection_diagnostics_sha256")
            if (not recorded_diagnostic_hash or current_hash != recorded_diagnostic_hash
                    or current_hash == metadata.get("sha256")):
                raise RuntimeError(
                    f"cannot prove projector path contains the saved diagnostic CSV: {projector_path}")
            plans.append({
                "seed": seed, "fold": fold, "evaluation": evaluation,
                "metadata": metadata, "diagnostics_path": diagnostics_path,
                "projector_path": projector_path,
                "projector_meta_path": projector_meta_path,
                "diagnostic_bytes": projector_path.read_bytes(),
            })

    # Do not partially modify external artifacts unless all affected folds passed
    # the identity and hash checks above.
    for item in plans:
        diag_path = item["diagnostics_path"]
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        temp_diag = diag_path.with_suffix(diag_path.suffix + ".tmp")
        temp_diag.write_bytes(item["diagnostic_bytes"])
        os.replace(temp_diag, diag_path)
        if file_sha256(diag_path) != item["evaluation"]["projection_diagnostics_sha256"]:
            raise RuntimeError(f"recovered diagnostic CSV hash mismatch: {diag_path}")

        metadata = item["metadata"]
        matrix64 = runner.projection_core.fixed_teacher_projection(int(metadata["seed"]))
        matrix32 = matrix64.astype("float32")
        runner._atomic_npz(item["projector_path"],
                           P_float64=matrix64, P_float32=matrix32)
        with np.load(item["projector_path"], allow_pickle=False) as saved:
            if (not np.array_equal(saved["P_float64"], matrix64)
                    or not np.array_equal(saved["P_float32"], matrix32)):
                raise RuntimeError(f"reconstructed projector values differ: {item['projector_path']}")
        metadata["restored_original_sha256"] = metadata.get("sha256")
        metadata["sha256"] = file_sha256(item["projector_path"])
        metadata["restored_from_deterministic_seed"] = True
        metadata["restored_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        runner._atomic_json(item["projector_meta_path"], metadata)
    return len(plans)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=runner.DEFAULT_CONFIG)
    args = parser.parse_args()
    config = runner._read_config(args.config.resolve())
    report_dir = REPORTS_ROOT / config["storage"]["report_directory"]
    final_files = ("metric_summary.csv", "paired_differences.csv",
                   "projection_diagnostics.csv", "teacher_rebuild_identity.csv",
                   "trial_predictions.csv", "aggregate.json", "summary.xlsx")
    existing = [report_dir / name for name in final_files if (report_dir / name).exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing CBraMod target-evaluation reports: "
            + ", ".join(map(str, existing)))

    checkpoints_before, fold_ids = preflight(config)
    restored_projectors = restore_misplaced_projection_diagnostics(config)
    # This runner passes its report directory through a helper that accepts
    # report *files* only. Permit that one directory after checking it remains
    # inside the real checkout results tree; individual files still use the
    # normal suffix and location validation.
    original_report_guard = runner.require_report_output
    report_root = REPORTS_ROOT.resolve(strict=False)

    def report_guard_with_directory(path):
        candidate = Path(path).expanduser()
        resolved = candidate.resolve(strict=False)
        if (candidate.suffix == "" and resolved != report_root
                and report_root in resolved.parents
                and not REPORTS_ROOT.is_symlink()):
            return candidate
        return original_report_guard(candidate)

    original_evaluate_fold = runner._evaluate_fold

    def load_fold_with_progress(config_value, seed, run_id, fold, device):
        print(f"[load-saved-evaluation] seed={seed} target={fold + 1}/{EXPECTED_FOLDS} start",
              flush=True)
        value = original_evaluate_fold(config_value, seed, run_id, fold, device)
        print(f"[load-saved-evaluation] seed={seed} target={fold + 1}/{EXPECTED_FOLDS} complete",
              flush=True)
        return value

    runner.require_report_output = report_guard_with_directory
    runner._evaluate_fold = load_fold_with_progress
    try:
        # All 27 fold-level prediction files were verified above; this path
        # loads them and writes summaries without loading or pruning weights.
        aggregate = runner._evaluate_all(config, torch.device("cpu"))
    finally:
        runner._evaluate_fold = original_evaluate_fold
        runner.require_report_output = original_report_guard

    # The runner's atomic JSON helper routes through the historical external
    # artifact mapper. Keep this user-facing aggregate in the checkout reports
    # tree, alongside its CSV/XLSX siblings.
    aggregate_target = require_report_output(report_dir / "aggregate.json")
    aggregate_target.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")

    checkpoints_after, _ = preflight(config)
    if checkpoints_before != checkpoints_after:
        raise RuntimeError("student checkpoint list changed during read-only evaluation")
    report = {
        "status": aggregate.get("status"),
        "experiment_id": config["experiment_id"],
        "seeds": list(EXPECTED_SEEDS),
        "completed_folds": fold_ids,
        "training_started": False,
        "new_inference_started": False,
        "saved_fold_evaluations_aggregated": len(fold_ids),
        "checkpoint_cleanup_started": False,
        "student_manifest_name": "student_<condition>.json",
        "aggregate_read_manifest_fallbacks": 0,
        "restored_projector_count": restored_projectors,
        "checkpoint_count_preserved": len(checkpoints_after),
        "aggregate_report": str(report_dir / "aggregate.json"),
    }
    target = require_report_output(report_dir / "evaluation_recovery.json")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
