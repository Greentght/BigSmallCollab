"""Focused LOSO hyperparameter tuning for MIRepNet.

Runs full leave-one-subject-out folds for each candidate config on one search
seed, chooses the best mean balanced accuracy per dataset, then confirms the
chosen config with three seeds. Designed for long detached launches on a shared
machine: subprocesses cap BLAS/PyTorch CPU threads by default.
"""
import argparse
import itertools
import json
import os
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PYBIN = os.environ.get("MIREPNET_PYBIN", "/home/lixinli/anaconda3/envs/mirepnet/bin/python")
RUNNER = os.path.join(ROOT, "scripts", "bigmodel", "mirepnet_loso_adapt.py")
TUNE_DIR = os.path.join(ROOT, "results", "mirepnet_loso", "tune")
TUNED_DIR = os.path.join(ROOT, "results", "mirepnet_loso", "tuned")
LOG_DIR = os.path.join(ROOT, "logs", "tune_mirepnet_loso")

N_SUBJECTS = {
    "BNCI2014004": 9,
    "BNCI2014001-4": 9,
    "BNCI2014001": 9,
    "AlexMI": 8,
    "BNCI2015001": 12,
}
ALIASES = {
    "BNCI2014001_4c": "BNCI2014001-4",
    "14001-4": "BNCI2014001-4",
    "BNCI2014001_2c": "BNCI2014001",
    "14001-2": "BNCI2014001",
    "14004": "BNCI2014004",
    "AlexMI_2c": "AlexMI",
    "15001": "BNCI2015001",
}

SEARCH_SEED = 666
CONFIRM_SEEDS = [666, 667, 668]
ALL_DATASETS = list(N_SUBJECTS)

# 3 x 2 x 2 x 2 = 24 configs, matching the focused-grid decision.
EPOCHS = [10, 30, 50]
LRS = [5e-4, 1e-3]
WEIGHT_DECAYS = [1e-6, 1e-4]
BATCH_SIZES = [8, 16]


def canonical_dataset_name(name):
    return ALIASES.get(name, name)


def cfg_grid():
    for epochs, lr, wd, bs in itertools.product(EPOCHS, LRS, WEIGHT_DECAYS, BATCH_SIZES):
        yield {"epochs": epochs, "lr": lr, "weight_decay": wd, "batch_size": bs}


def cfg_tag(c):
    return f"lr{c['lr']:g}_ep{c['epochs']}_wd{c['weight_decay']:g}_bs{c['batch_size']}"


def build_cmd(dataset, c, seeds, gpu, out, threads):
    return [
        PYBIN, RUNNER,
        "--dataset", dataset,
        "--gpu", str(gpu),
        "--seeds", *[str(s) for s in seeds],
        "--epochs", str(c["epochs"]),
        "--lr", str(c["lr"]),
        "--weight_decay", str(c["weight_decay"]),
        "--batch_size", str(c["batch_size"]),
        "--torch_threads", str(threads),
        "--out", out,
    ]


def run_env(threads):
    env = os.environ.copy()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = str(threads)
    env["TORCH_THREADS"] = str(threads)
    return env


def is_complete(out, dataset, n_seeds):
    if not os.path.exists(out):
        return False
    try:
        df = pd.read_csv(out)
    except Exception:
        return False
    return len(df) >= N_SUBJECTS[dataset] * n_seeds


def run_job(job, gpu_q, log_prefix, threads):
    dataset, c, seeds, out, log = job
    gpu = gpu_q.get()
    try:
        if is_complete(out, dataset, len(seeds)):
            print(f"[skip] {log_prefix} {dataset} {cfg_tag(c)} (done)", flush=True)
            return dataset, c, out, "skip"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        os.makedirs(os.path.dirname(log), exist_ok=True)
        cmd = build_cmd(dataset, c, seeds, gpu, out, threads)
        print(f"[run ] {log_prefix} {dataset} {cfg_tag(c)} gpu={gpu}", flush=True)
        with open(log, "w") as fh:
            rc = subprocess.run(
                cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=run_env(threads)
            ).returncode
        status = "ok" if rc == 0 else f"FAIL(rc={rc})"
        print(f"[done] {log_prefix} {dataset} {cfg_tag(c)} gpu={gpu} -> {status}", flush=True)
        return dataset, c, out, status
    finally:
        gpu_q.put(gpu)


def mean_bac(out):
    return float(pd.read_csv(out)["bac"].mean())


def phase_search(datasets, gpus, threads):
    gpu_q = queue.Queue()
    for g in gpus:
        gpu_q.put(g)
    jobs = []
    for ds in datasets:
        for c in cfg_grid():
            out = os.path.join(TUNE_DIR, ds, f"{cfg_tag(c)}.csv")
            log = os.path.join(LOG_DIR, f"search_{ds}_{cfg_tag(c)}.log")
            jobs.append((ds, c, [SEARCH_SEED], out, log))
    print(f"[search] {len(jobs)} jobs across gpus {gpus}, threads/job={threads}", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        list(ex.map(lambda j: run_job(j, gpu_q, "search", threads), jobs))

    chosen = {}
    rows = []
    for ds in datasets:
        best = None
        for c in cfg_grid():
            out = os.path.join(TUNE_DIR, ds, f"{cfg_tag(c)}.csv")
            if not is_complete(out, ds, 1):
                print(f"[warn] incomplete: {out}", flush=True)
                continue
            bac = mean_bac(out)
            rows.append({"dataset": ds, **c, "search_bac": round(bac, 6)})
            if best is None or bac > best[0]:
                best = (bac, c)
        if best is not None:
            chosen[ds] = {**best[1], "search_bac": round(best[0], 6)}
            print(f"[best ] {ds}: {cfg_tag(best[1])} bac={best[0]:.4f}", flush=True)
    os.makedirs(TUNED_DIR, exist_ok=True)
    if rows:
        pd.DataFrame(rows).sort_values(
            ["dataset", "search_bac"], ascending=[True, False]
        ).to_csv(os.path.join(TUNE_DIR, "search_summary.csv"), index=False)
    with open(os.path.join(TUNED_DIR, "chosen_configs.json"), "w") as fh:
        json.dump(chosen, fh, indent=2)
    return chosen


def load_chosen():
    with open(os.path.join(TUNED_DIR, "chosen_configs.json")) as fh:
        return json.load(fh)


def phase_confirm(datasets, gpus, chosen, threads):
    gpu_q = queue.Queue()
    for g in gpus:
        gpu_q.put(g)
    jobs = []
    for ds in datasets:
        if ds not in chosen:
            print(f"[warn] no chosen config for {ds}", flush=True)
            continue
        c = {k: chosen[ds][k] for k in ("epochs", "lr", "weight_decay", "batch_size")}
        out = os.path.join(TUNED_DIR, f"{ds}.csv")
        log = os.path.join(LOG_DIR, f"confirm_{ds}.log")
        jobs.append((ds, c, CONFIRM_SEEDS, out, log))
    print(f"[confirm] {len(jobs)} jobs across gpus {gpus}, threads/job={threads}", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        list(ex.map(lambda j: run_job(j, gpu_q, "confirm", threads), jobs))

    rows = []
    for ds in datasets:
        out = os.path.join(TUNED_DIR, f"{ds}.csv")
        if not os.path.exists(out):
            continue
        df = pd.read_csv(out)
        rows.append({
            "dataset": ds,
            "acc": round(df.acc.mean(), 2),
            "bac": round(df.bac.mean(), 4),
            "kappa": round(df.kappa.mean(), 4),
            "n": len(df),
        })
    if rows:
        summ = pd.DataFrame(rows)
        summ.to_csv(os.path.join(TUNED_DIR, "summary_tuned.csv"), index=False)
        print(summ.to_string(index=False), flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["search", "confirm", "all"], default="all")
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 2, 3])
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    datasets = [canonical_dataset_name(d) for d in args.datasets]
    unknown = [d for d in datasets if d not in N_SUBJECTS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}; known={list(N_SUBJECTS)}")
    if args.phase in ("search", "all"):
        chosen = phase_search(datasets, args.gpus, args.threads)
    else:
        chosen = load_chosen()
    if args.phase in ("confirm", "all"):
        phase_confirm(datasets, args.gpus, chosen, args.threads)


if __name__ == "__main__":
    main()
