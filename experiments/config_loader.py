"""Shared loader for method-centric experiment YAML files.

One YAML describes one collaboration method. Each big/small pair owns its
``params`` and ``grid`` blocks; training settings use the same blocks and are
separated only in the resolved runtime spec. The loader expands
dataset x protocol x pair x grid into runner specifications.
"""

from __future__ import annotations

from itertools import product
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

import config as project_config
import data
from models import BIG_MODELS, SMALL_MODELS
from experiments.storage import external_path, require_external_output


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_KEYS = ("epochs", "lr", "weight_decay", "batch_size")

COMMON_FIELDS = {
    "name", "type", "method", "datasets", "protocols", "seeds", "subjects",
    "artifact_root", "output_dir", "fail_fast", "pairs",
}
DISTILL_FIELDS = COMMON_FIELDS | {"include_baseline"}
FUSION_FIELDS = COMMON_FIELDS | {"include_controls"}

DISTILL_RUNTIME_METHODS = {
    "kd": "KD_all",
    "kd_masked": "KD_masked",
    "mmd": "MMD",
    "kd_mmd": "KD_MMD",
    "mi": "CE_MI",
}
DISTILL_METHOD_ALIASES = {
    "kd": "kd",
    "kd_all": "kd",
    "kd_masked": "kd_masked",
    "mmd": "mmd",
    "kd_mmd": "kd_mmd",
    "mi": "mi",
}
FUSION_RUNTIME_METHODS = {
    "avg_prob": "avg_prob",
    "concat_mlp": "concat_mlp",
    "gate_conf_acc": "gate_conf_acc",
}
FUSION_METHOD_ALIASES = {
    "avg": "avg_prob",
    "avg_prob": "avg_prob",
    "concat": "concat_mlp",
    "concat_mlp": "concat_mlp",
    "gate": "gate_conf_acc",
    "gate_conf_acc": "gate_conf_acc",
}

MMD_FIELDS = {
    "lam_mmd", "mmd_sigmas", "mmd_normalize", "mmd_class_conditional",
}
PARAM_FIELDS = {
    "distill": {
        "kd": {"temperature", "lam_kd"},
        "kd_masked": {"temperature", "lam_kd"},
        "mmd": set(MMD_FIELDS),
        "kd_mmd": {"temperature", "lam_kd"} | MMD_FIELDS,
        "mi": {"lam_mi"},
    },
    "fusion": {
        "avg_prob": {"big_temperature", "small_temperature", "big_weight"},
        "concat_mlp": {"hidden", "dropout"},
        "gate_conf_acc": {
            "alpha", "beta", "big_temperature", "small_temperature",
        },
    },
}

METHOD_DEFAULT_PARAMS = {
    "distill": {
        "kd": {"temperature": 2.0, "lam_kd": 0.5},
        "kd_masked": {"temperature": 2.0, "lam_kd": 0.5},
        "mmd": {
            "lam_mmd": 0.5,
            "mmd_sigmas": [0.5, 1.0, 2.0, 4.0],
            "mmd_normalize": True,
            "mmd_class_conditional": False,
        },
        "kd_mmd": {
            "temperature": 2.0,
            "lam_kd": 0.5,
            "lam_mmd": 0.5,
            "mmd_sigmas": [0.5, 1.0, 2.0, 4.0],
            "mmd_normalize": True,
            "mmd_class_conditional": False,
        },
        "mi": {"lam_mi": 0.1},
    },
    "fusion": {
        "avg_prob": {
            "big_temperature": 1.0,
            "small_temperature": 1.0,
            "big_weight": 0.5,
        },
        "concat_mlp": {"hidden": 128, "dropout": 0.2},
        "gate_conf_acc": {
            "alpha": 1.0,
            "beta": 1.0,
            "big_temperature": 1.0,
            "small_temperature": 1.0,
        },
    },
}


class ExperimentConfigError(ValueError):
    """Raised when an experiment YAML is ambiguous or invalid."""


def _fail(context: str, message: str) -> None:
    raise ExperimentConfigError(f"{context}: {message}")


def _require_mapping(value: Any, context: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        _fail(context, "must be a mapping")
    return dict(value)


def _check_unknown(mapping: Mapping, allowed: set[str], context: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        _fail(context, f"unknown field(s): {unknown}")


def _string_list(value: Any, field: str, allow_none: bool = False) -> list[str] | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, list) or not value or not all(isinstance(x, str) for x in value):
        _fail(field, "must be a non-empty list of strings or null")
    return list(value)


def _int_list(value: Any, field: str, allow_none: bool = False) -> list[int] | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, list) or not value or not all(
            isinstance(x, int) and not isinstance(x, bool) for x in value):
        _fail(field, "must be a non-empty list of integers or null")
    return [int(x) for x in value]


def _number(value: Any, field: str, *, minimum: float | None = None,
            maximum: float | None = None, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(field, "must be numeric")
    if integer and (not isinstance(value, int) or isinstance(value, bool)):
        _fail(field, "must be an integer")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        _fail(field, f"must be >= {minimum}")
    if maximum is not None and numeric > maximum:
        _fail(field, f"must be <= {maximum}")


def _validate_param_value(name: str, value: Any, context: str) -> None:
    if value is None:
        return
    if name in {"temperature", "big_temperature", "small_temperature", "lr"}:
        _number(value, f"{context}.{name}", minimum=0.0)
        if float(value) <= 0:
            _fail(f"{context}.{name}", "must be > 0")
    elif name in {"lam_kd", "lam_mi", "lam_mmd", "weight_decay", "alpha", "beta"}:
        _number(value, f"{context}.{name}", minimum=0.0)
    elif name == "big_weight":
        _number(value, f"{context}.{name}", minimum=0.0, maximum=1.0)
    elif name in {"epochs", "batch_size", "hidden"}:
        _number(value, f"{context}.{name}",
                minimum=1 if name != "hidden" else 0, integer=True)
    elif name == "dropout":
        _number(value, f"{context}.{name}", minimum=0.0, maximum=1.0)
    elif name == "mmd_sigmas":
        if not isinstance(value, list) or not value:
            _fail(f"{context}.{name}", "must be a non-empty list")
        for index, sigma in enumerate(value):
            _number(sigma, f"{context}.{name}[{index}]", minimum=0.0)
            if float(sigma) <= 0:
                _fail(f"{context}.{name}[{index}]", "must be > 0")
    elif name in {"mmd_normalize", "mmd_class_conditional"}:
        if not isinstance(value, bool):
            _fail(f"{context}.{name}", "must be boolean")
    else:
        _fail(context, f"unknown parameter {name!r}")


def _validate_block(value: Any, allowed: set[str], context: str) -> dict:
    block = _require_mapping(value, context)
    _check_unknown(block, allowed, context)
    for key, item in block.items():
        _validate_param_value(key, item, context)
    return block


def _canonical_protocols(value: Any) -> list[str]:
    raw = _string_list(value, "protocols")
    protocols = []
    for item in raw:
        protocol = data.canonical_protocol(item)
        if protocol not in {"fewshot", "loso"}:
            _fail("protocols", f"unsupported protocol {item!r}")
        if protocol not in protocols:
            protocols.append(protocol)
    return protocols


def _canonical_method(value: Any, kind: str) -> str:
    if not isinstance(value, str):
        _fail("method", "must be a string")
    key = value.strip().lower()
    aliases = (DISTILL_METHOD_ALIASES if kind == "distill"
               else FUSION_METHOD_ALIASES)
    if key not in aliases:
        valid = sorted(DISTILL_RUNTIME_METHODS if kind == "distill"
                       else FUSION_RUNTIME_METHODS)
        _fail("method", f"unknown method {value!r}; expected one of {valid}")
    return aliases[key]


def _validate_common(raw: Mapping, kind: str) -> dict:
    _check_unknown(raw, DISTILL_FIELDS if kind == "distill" else FUSION_FIELDS,
                   "experiment")
    name = raw.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        _fail("name", "must contain only letters, digits, '.', '_' or '-'")
    if raw.get("type") != kind:
        _fail("type", f"must be {kind!r}")
    datasets = _string_list(raw.get("datasets"), "datasets")
    if len(set(datasets)) != len(datasets):
        _fail("datasets", "must not contain duplicates")
    protocols = _canonical_protocols(raw.get("protocols"))
    subjects = _int_list(raw.get("subjects"), "subjects", allow_none=True)
    seeds = _int_list(raw.get("seeds"), "seeds", allow_none=True)
    if subjects is not None and any(subject < 0 for subject in subjects):
        _fail("subjects", "must contain only non-negative indices")
    artifact_root = raw.get("artifact_root", "/data1/llx/BigSmallCollab_results/artifacts")
    output_dir = raw.get("output_dir", f"/data1/llx/BigSmallCollab_results/{kind}")
    for field, value in (("artifact_root", artifact_root), ("output_dir", output_dir)):
        if not isinstance(value, str) or not value:
            _fail(field, "must be a non-empty string")
    artifact_root = str(external_path(artifact_root))
    output_dir = str(require_external_output(output_dir))
    fail_fast = raw.get("fail_fast", False)
    if not isinstance(fail_fast, bool):
        _fail("fail_fast", "must be boolean")
    method = _canonical_method(raw.get("method"), kind)
    if kind == "distill" and method == "mi":
        if protocols != ["fewshot"]:
            _fail("protocols", "method 'mi' is subject-wise fewshot only")
        if subjects is not None:
            _fail("subjects", "method 'mi' requires subjects: null (all subjects)")
        if raw.get("include_baseline", True) is not True:
            _fail("include_baseline", "method 'mi' always expands Base, KD_all and CE_MI")
    common = {
        "name": name,
        "type": kind,
        "method": method,
        "datasets": datasets,
        "protocols": protocols,
        "subjects": subjects,
        "seeds": seeds,
        "artifact_root": artifact_root,
        "output_dir": output_dir,
        "fail_fast": fail_fast,
    }
    control_field = "include_baseline" if kind == "distill" else "include_controls"
    include_control = raw.get(control_field, True)
    if not isinstance(include_control, bool):
        _fail(control_field, "must be boolean")
    common[control_field] = include_control
    return common


def _validate_grid(value: Any, allowed: set[str], context: str) -> dict:
    grid = _require_mapping(value, context)
    _check_unknown(grid, allowed, context)
    result = {}
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            _fail(f"{context}.{key}", "grid values must be a non-empty list")
        for item in values:
            _validate_param_value(key, item, context)
        result[key] = list(values)
    return result


def _validate_pair(raw: Any, kind: str, method: str, index: int) -> dict:
    if not isinstance(raw, Mapping):
        _fail(f"pairs[{index}]", "must be a mapping")
    model_fields = ({"teacher", "student"} if kind == "distill"
                    else {"big", "small"})
    allowed = model_fields | {"params", "training", "grid", "datasets"}
    _check_unknown(raw, allowed, f"pairs[{index}]")
    for required in (*model_fields, "params", "grid"):
        if required not in raw:
            _fail(f"pairs[{index}]", f"missing required field {required!r}")
    if kind == "distill":
        teacher = raw.get("teacher")
        student = raw.get("student")
        if not isinstance(teacher, str) or teacher not in BIG_MODELS:
            _fail(f"pairs[{index}].teacher", f"unknown model {teacher!r}")
        if not isinstance(student, str) or student not in SMALL_MODELS:
            _fail(f"pairs[{index}].student", f"unknown model {student!r}")
    else:
        teacher = raw.get("big")
        student = raw.get("small")
        if not isinstance(teacher, str) or teacher not in BIG_MODELS:
            _fail(f"pairs[{index}].big", f"unknown model {teacher!r}")
        if not isinstance(student, str) or student not in SMALL_MODELS:
            _fail(f"pairs[{index}].small", f"unknown model {student!r}")
    allowed_params = PARAM_FIELDS[kind][method] | set(TRAINING_KEYS)
    params = _validate_block(raw.get("params"), allowed_params, f"pairs[{index}].params")
    training = _validate_block(raw.get("training"), set(TRAINING_KEYS),
                               f"pairs[{index}].training")
    duplicate_training = sorted(set(params) & set(training))
    if duplicate_training:
        _fail(
            f"pairs[{index}]",
            f"training value(s) specified twice: {duplicate_training}",
        )
    grid = _validate_grid(raw.get("grid"), allowed_params, f"pairs[{index}].grid")
    overrides = raw.get("datasets")
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        _fail(f"pairs[{index}].datasets", "must be a mapping")
    return {
        "index": index,
        "teacher": teacher,
        "student": student,
        "params": params,
        "training": training,
        "grid": grid,
        "datasets": dict(overrides),
    }


def _validate_dataset_overrides(pair: Mapping, datasets: list[str], kind: str,
                                method: str) -> None:
    allowed = {"params", "training", "grid"}
    all_params = PARAM_FIELDS[kind][method] | set(TRAINING_KEYS)
    for dataset, raw in pair["datasets"].items():
        if dataset not in datasets:
            _fail(f"pairs[{pair['index']}].datasets", f"unknown dataset {dataset!r}")
        if not isinstance(raw, Mapping):
            _fail(f"pairs[{pair['index']}].datasets.{dataset}", "must be a mapping")
        _check_unknown(raw, allowed, f"pairs[{pair['index']}].datasets.{dataset}")
        params = _validate_block(
            raw.get("params"), all_params,
            f"pairs[{pair['index']}].datasets.{dataset}.params")
        training = _validate_block(
            raw.get("training"), set(TRAINING_KEYS),
            f"pairs[{pair['index']}].datasets.{dataset}.training")
        duplicate_training = sorted(set(params) & set(training))
        if duplicate_training:
            _fail(
                f"pairs[{pair['index']}].datasets.{dataset}",
                f"training value(s) specified twice: {duplicate_training}",
            )
        _validate_grid(raw.get("grid"), all_params,
                       f"pairs[{pair['index']}].datasets.{dataset}.grid")


def _split_values(params: Mapping) -> tuple[dict, dict]:
    """Split a pair's flat params into method and runtime-training values."""
    out_params = dict(params)
    out_training = {}
    for key in TRAINING_KEYS:
        if key in out_params:
            out_training[key] = out_params.pop(key)
    return out_params, out_training


def _merge_pair_blocks(pair: Mapping, dataset: str, kind: str,
                       method: str) -> tuple[dict, dict, dict]:
    params, training = _split_values(pair["params"])
    training.update(pair["training"])
    grid = dict(pair["grid"])
    override = pair["datasets"].get(dataset, {})
    over_params, over_training = _split_values(override.get("params", {}))
    params.update(over_params)
    training.update(over_training)
    training.update(override.get("training", {}))
    grid.update(override.get("grid", {}))
    defaults = METHOD_DEFAULT_PARAMS[kind][method]
    effective_params = dict(defaults)
    effective_params.update(params)
    return effective_params, training, grid


def _resolve_training(model: str, dataset: str, protocol: str,
                      configured: Mapping) -> dict:
    try:
        defaults = project_config.load_model_config(model, dataset, protocol)
    except Exception as exc:
        _fail(f"{model}/{dataset}/{protocol}", str(exc))
    resolved = {key: defaults.get(key) for key in TRAINING_KEYS}
    for key, value in configured.items():
        if key in TRAINING_KEYS and value is not None:
            resolved[key] = value
    for key, value in resolved.items():
        if value is None:
            _fail("training", f"could not resolve {key!r} for {model}/{dataset}/{protocol}")
        _validate_param_value(key, value, "resolved training")
    return resolved


def _grid_combinations(params: Mapping, training: Mapping, grid: Mapping,
                       kind: str, method: str):
    keys = list(grid)
    choices = [grid[key] for key in keys]
    if not keys:
        yield 0, dict(params), dict(training)
        return
    method_fields = PARAM_FIELDS[kind][method]
    for index, values in enumerate(product(*choices)):
        p = dict(params)
        t = dict(training)
        for key, value in zip(keys, values):
            if key in TRAINING_KEYS:
                t[key] = value
            elif key in method_fields:
                p[key] = value
            else:
                _fail("grid", f"unknown parameter {key!r}")
        for key, value in p.items():
            _validate_param_value(key, value, "expanded params")
        for key, value in t.items():
            _validate_param_value(key, value, "expanded training")
        yield index, p, t


def _stable_run_id(name: str, pair_index: int, dataset: str, protocol: str,
                   teacher: str, student: str, grid_index: int,
                   params: Mapping, training: Mapping) -> str:
    payload = json.dumps({
        "pair_index": pair_index, "dataset": dataset, "protocol": protocol,
        "teacher": teacher, "student": student, "grid_index": grid_index,
        "params": params, "training": training,
    }, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return (f"{name}__p{pair_index:02d}__{dataset}__{protocol}__"
            f"g{grid_index:03d}__{digest}")


def _derived_paths(name: str, output_dir: str) -> tuple[str, str]:
    out_dir = Path(output_dir)
    output = out_dir / f"{name}.csv"
    log_file = out_dir.parent / "logs" / f"{name}.log"
    return str(output), str(log_file)


def load_experiment(path: str | Path) -> dict:
    """Load, validate, merge defaults, and expand one method YAML."""
    source = Path(path)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    if not source.exists():
        raise FileNotFoundError(source)
    with source.open() as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        _fail("experiment", "top-level YAML value must be a mapping")
    kind = raw.get("type")
    if kind not in {"distill", "fusion"}:
        _fail("type", "must be 'distill' or 'fusion'")
    common = _validate_common(raw, kind)
    pairs_raw = raw.get("pairs")
    if not isinstance(pairs_raw, list) or not pairs_raw:
        _fail("pairs", "must be a non-empty list")
    pairs = [_validate_pair(item, kind, common["method"], index)
             for index, item in enumerate(pairs_raw)]
    seen_pairs = {}
    for pair in pairs:
        pair_key = (pair["teacher"], pair["student"])
        if pair_key in seen_pairs:
            _fail(
                f"pairs[{pair['index']}]",
                f"duplicates pairs[{seen_pairs[pair_key]}] for "
                f"{pair['teacher']} x {pair['student']}",
            )
        seen_pairs[pair_key] = pair["index"]
        _validate_dataset_overrides(
            pair, common["datasets"], kind, common["method"])

    output, log_file = _derived_paths(common["name"], common["output_dir"])
    config = dict(common)
    config.update({"pairs": pairs, "output": output, "log_file": log_file})
    runs = []
    seen_control_keys = set()
    runtime_method = (DISTILL_RUNTIME_METHODS[common["method"]]
                      if kind == "distill" else
                      FUSION_RUNTIME_METHODS[common["method"]])
    for dataset in common["datasets"]:
        try:
            dataset_config = project_config.load_dataset_config(dataset)
        except Exception as exc:
            _fail(f"datasets.{dataset}", str(exc))
        if common["subjects"] is not None:
            num_subjects = int(dataset_config["num_subjects"])
            invalid = [subject for subject in common["subjects"]
                       if subject >= num_subjects]
            if invalid:
                _fail(
                    "subjects",
                    f"indices {invalid} are out of range for {dataset} "
                    f"(num_subjects={num_subjects})",
                )
        for protocol in common["protocols"]:
            for pair in pairs:
                params, explicit_training, grid = _merge_pair_blocks(
                    pair, dataset, kind, common["method"])
                training = _resolve_training(pair["student"], dataset, protocol,
                                             explicit_training)
                for grid_index, final_params, final_training in _grid_combinations(
                        params, training, grid, kind, common["method"]):
                    control_identity = {
                        "dataset": dataset,
                        "protocol": protocol,
                        "pair_index": pair["index"],
                        "teacher": pair["teacher"],
                        "student": pair["student"],
                    }
                    if kind == "distill":
                        control_identity["training"] = final_training
                    control_key = json.dumps(
                        control_identity, sort_keys=True, separators=(",", ":"))
                    first_control = control_key not in seen_control_keys
                    seen_control_keys.add(control_key)
                    if kind == "distill":
                        if common["method"] == "mi":
                            # The MI pilot is deliberately a fixed three-way
                            # comparison; do not let control de-duplication or
                            # include_baseline silently remove a condition.
                            methods = ["Base", "KD_all", "CE_MI"]
                        else:
                            methods = [runtime_method]
                            if common["include_baseline"] and first_control:
                                methods.insert(0, "Base")
                    else:
                        methods = [runtime_method]
                        if common["include_controls"] and first_control:
                            methods = ["big_only", "small_only"] + methods
                    run_id = _stable_run_id(
                        common["name"], pair["index"], dataset, protocol,
                        pair["teacher"], pair["student"], grid_index,
                        final_params, final_training)
                    spec = {
                        "type": kind,
                        "method": common["method"],
                        "config_name": common["name"],
                        "run_id": run_id,
                        "grid_index": grid_index,
                        "pair_index": pair["index"],
                        "dataset": dataset,
                        "protocol": protocol,
                        "teacher": pair["teacher"],
                        "student": pair["student"],
                        "big_model": pair["teacher"] if kind == "fusion" else None,
                        "small_model": pair["student"] if kind == "fusion" else None,
                        "runtime_method": runtime_method,
                        "methods": methods,
                        "params": final_params,
                        "training": final_training,
                        "subjects": common["subjects"],
                        "seeds": common["seeds"],
                        "artifact_root": common["artifact_root"],
                        "fail_fast": common["fail_fast"],
                        "output": output,
                        "log_file": log_file,
                    }
                    runs.append(spec)
    if not runs:
        _fail("experiment", "expansion produced no runs")
    return {"source_path": str(source), "config": config, "runs": runs}


def require_experiment_type(loaded: Mapping, expected: str) -> None:
    """Reject dispatching a valid config to the wrong experiment runner."""
    if expected not in {"distill", "fusion"}:
        raise ValueError(f"unknown experiment type {expected!r}")
    actual = loaded.get("config", {}).get("type")
    if actual != expected:
        raise ExperimentConfigError(
            f"type: config is {actual!r}, but the {expected} runner was selected")


def resolve_path(value: str | Path) -> Path:
    """Resolve code paths locally and historical artifact paths externally."""
    path = external_path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def jsonable(value: Any):
    """Convert resolved config objects to YAML/JSON-safe Python values."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def resolved_payload(loaded: Mapping) -> dict:
    return {
        "source_path": loaded["source_path"],
        "config": jsonable(loaded["config"]),
        "runs": jsonable(loaded["runs"]),
    }


def resolved_path(output: str | Path, name: str | None = None) -> Path:
    """Map foo.csv to the requested sibling foo.resolved.yaml."""
    path = resolve_path(output)
    stem = name or path.stem
    return path.with_name(f"{stem}.resolved.yaml")


def save_resolved(loaded: Mapping, output: str | Path) -> Path:
    path = require_external_output(resolved_path(output, loaded["config"].get("name")))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(resolved_payload(loaded), handle,
                       sort_keys=False, allow_unicode=True)
    return path
