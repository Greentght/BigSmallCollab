"""Config-driven fusion backend used by ``run_fusion.py``."""

from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stderr, redirect_stdout
import sys
from types import SimpleNamespace

from experiments import config_loader
from experiments.fusion import run_fusion as legacy


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python experiments/fusion/run_fusion.py",
        description="Run a fusion experiment from configs/experiments/*.yaml")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, default=None)
    write_mode = parser.add_mutually_exclusive_group()
    write_mode.add_argument("--resume", action="store_true")
    write_mode.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def _device(gpu):
    import torch
    if gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        return f"cuda:{gpu}"
    return "cpu"


def _namespace(spec, artifact_root):
    params = dict(spec["params"])
    training = dict(spec["training"])
    return SimpleNamespace(
        dataset=spec["dataset"], protocol=spec["protocol"],
        big=spec["big_model"], small=spec["small_model"],
        keys=spec.get("subjects"), seeds=spec.get("seeds"),
        methods=spec["methods"], gpu=None,
        epochs=int(training["epochs"]), lr=float(training["lr"]),
        weight_decay=float(training["weight_decay"]),
        batch_size=int(training["batch_size"]),
        hidden=int(params.get("hidden", 128)),
        dropout=float(params.get("dropout", 0.2)),
        alpha=float(params.get("alpha", 1.0)),
        beta=float(params.get("beta", 1.0)),
        big_temperature=float(params.get("big_temperature", 1.0)),
        small_temperature=float(params.get("small_temperature", 1.0)),
        big_weight=float(params.get("big_weight", 0.5)),
        artifact_root=str(artifact_root),
    )


def _row_key(row):
    values = (
        row.get("run_id", ""), row.get("dataset", ""),
        row.get("big", ""), row.get("small", ""),
        row.get("key", row.get("subject", "")), row.get("seed", ""),
        row.get("method", ""),
    )
    return tuple(str(value) for value in values)


def _read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    priority = [
        "config_name", "run_id", "grid_index", "dataset", "protocol", "big",
        "small", "big_artifact", "small_artifact", "subject", "key", "seed",
        "method", "acc", "kappa", "n_train", "n_test", "num_classes",
        "train_acc_big_pct", "train_acc_small_pct", "epochs", "lr",
        "weight_decay", "batch_size", "hidden", "dropout", "alpha", "beta",
        "big_temperature", "small_temperature", "big_weight",
    ]
    fields = list(priority)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_spec(spec, device, existing_keys, fail_fast):
    dcfg = legacy.config.load_dataset_config(spec["dataset"])
    keys = spec.get("subjects")
    if keys is None:
        keys = list(range(dcfg["num_subjects"]))
    seeds = spec.get("seeds")
    if seeds is None:
        seeds = list(dcfg["seeds"])
    args = _namespace(spec, config_loader.resolve_path(spec["artifact_root"]))
    protocol = spec["protocol"]
    methods = spec["methods"]
    expected_policy = legacy._expected_split_policy(protocol)
    big_artifact = legacy._artifact_model(spec["big_model"], protocol)
    small_artifact = legacy._artifact_model(spec["small_model"], protocol)
    rows, errors = [], []
    for seed in seeds:
        for key in keys:
            prefix = (spec["run_id"], spec["dataset"], spec["big_model"],
                      spec["small_model"], str(key), str(seed))
            pending_methods = [
                method for method in methods
                if prefix + (method,) not in existing_keys
            ]
            if not pending_methods:
                print(f"[resume] skip dataset={spec['dataset']} key={key} seed={seed} run_id={spec['run_id']}", flush=True)
                continue
            try:
                cell_rows = legacy._run_cell(
                    args, protocol, pending_methods, device,
                    big_artifact, small_artifact,
                    expected_policy, key, seed)
                for row in cell_rows:
                    row.update({
                        "config_name": spec["config_name"],
                        "run_id": spec["run_id"],
                        "grid_index": spec["grid_index"],
                    })
                    if row["method"] == spec["runtime_method"]:
                        row.update(spec["params"])
                        if spec["runtime_method"] == "concat_mlp":
                            row.update(spec["training"])
                    row_key = _row_key(row)
                    if row_key not in existing_keys:
                        rows.append(row)
                        existing_keys.add(row_key)
                print(
                    f"[ok] dataset={spec['dataset']} key={key} seed={seed} "
                    f"rows={len(cell_rows)} run_id={spec['run_id']}", flush=True)
            except Exception as exc:  # noqa: BLE001
                message = (f"{spec['dataset']} {spec['big_model']}->{spec['small_model']} "
                           f"key={key} seed={seed} run_id={spec['run_id']}: {exc}")
                errors.append(message)
                print(f"[ERR] {message}", flush=True)
                if fail_fast:
                    raise
    return rows, errors


def run(args, loaded):
    config_loader.require_experiment_type(loaded, "fusion")
    specs = loaded["runs"]
    output = config_loader.resolve_path(specs[0]["output"])
    resolved = config_loader.resolved_path(output, specs[0]["config_name"])
    if output.exists() and not (args.force or args.resume):
        raise FileExistsError(f"refusing to overwrite {output}; use --resume or --force")
    if resolved.exists() and not (args.force or args.resume):
        raise FileExistsError(f"refusing to overwrite {resolved}; use --force")
    config_loader.save_resolved(loaded, output)
    device = _device(args.gpu)
    prior = _read_rows(output) if args.resume else []
    existing_keys = {_row_key(row) for row in prior}
    rows, errors = list(prior), []
    print(f"[config] {loaded['source_path']} type=fusion specs={len(specs)} device={device}", flush=True)
    print(f"[resolved] {resolved}", flush=True)
    for spec in specs:
        new_rows, spec_errors = _run_spec(
            spec, device, existing_keys,
            bool(args.fail_fast or spec.get("fail_fast", False)))
        rows.extend(new_rows)
        errors.extend(spec_errors)
        _write_rows(output, rows)
    _write_rows(output, rows)
    print(f"[write] {output} rows={len(rows)}", flush=True)
    if errors:
        print(f"[errors] {len(errors)} cells failed", flush=True)
        for error in errors:
            print(f"  {error}", flush=True)
    return 1 if errors else 0


def main(argv=None):
    args = parse_args(argv)
    loaded = config_loader.load_experiment(args.config)
    config_loader.require_experiment_type(loaded, "fusion")
    log_path = config_loader.resolve_path(loaded["runs"][0]["log_file"])
    if log_path.exists() and not (args.force or args.resume):
        raise FileExistsError(f"refusing to overwrite {log_path}; use --resume or --force")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with log_path.open(mode, buffering=1) as log_handle:
        tee_out = _Tee(sys.stdout, log_handle)
        tee_err = _Tee(sys.stderr, log_handle)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"[log] writing stdout/stderr to {log_path}", flush=True)
            return run(args, loaded)
