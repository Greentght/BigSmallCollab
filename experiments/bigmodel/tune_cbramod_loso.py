"""Focused LOSO hyperparameter tuning for native CBraMod.

Uses ``cbramod_native_adapt.py --protocol loso``. The focused grid follows the
handoff decision: keep preprocessing fixed per dataset and sweep optimizer/head
training levers only (lr x epochs x weight_decay x dropout = 24 configs).

Preprocessing defaults encode the current project findings:
  * BNCI2014004 (3 channels): norm=none, scale=1, b75n60; CAR is harmful there.
  * Other datasets: CAR-only high-score path, scale=1; b75n60 except AlexMI=b50.
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
PYBIN = os.environ.get("CBRAMOD_PYBIN", "/home/lixinli/anaconda3/envs/cbramod/bin/python")
RUNNER = os.path.join(ROOT, "experiments", "bigmodel", "cbramod_native_adapt.py")
TUNE_DIR = os.path.join(ROOT, "results", "cbramod_native", "loso_tune")
TUNED_DIR = os.path.join(ROOT, "results", "cbramod_native", "loso_tuned")
LOG_DIR = os.path.join(ROOT, "logs", "tune_cbramod_loso")

N_SUBJECTS = {
    "BNCI2014001_4c": 9,
    "BNCI2014001_2c": 9,
    "BNCI2014004": 9,
    "AlexMI_2c": 8,
    "BNCI2015001": 12,
}
ALIASES = {
    "BNCI2014001": "BNCI2014001_4c",
    "BNCI2014001-4": "BNCI2014001_4c",
    "14001-4": "BNCI2014001_4c",
    "14001-2": "BNCI2014001_2c",
    "AlexMI": "AlexMI_2c",
    "14004": "BNCI2014004",
    "15001": "BNCI2015001",
}

BANDS = {
    "b50": (0.3, 50.0, None),
    "b75n60": (0.3, 75.0, 60.0),
}
PREPROC = {
    "BNCI2014001_4c": {"norm": "car", "scale": 1.0, "band": "b75n60"},
    "BNCI2014001_2c": {"norm": "car", "scale": 1.0, "band": "b75n60"},
    "BNCI2014004": {"norm": "none", "scale": 1.0, "band": "b75n60"},
    "AlexMI_2c": {"norm": "car", "scale": 1.0, "band": "b50"},
    "BNCI2015001": {"norm": "car", "scale": 1.0, "band": "b75n60"},
}

SEARCH_SEED = 666
CONFIRM_SEEDS = [666, 667, 668]
ALL_DATASETS = list(N_SUBJECTS)

FIXED = dict(
    batch_size=16,
    warmup_epochs=5,
    min_lr=1e-6,
    clip_grad_norm=1.0,
    label_smoothing=0.0,
)

LRS = [5e-4, 1e-3]
EPOCHS = [20, 50]
WEIGHT_DECAYS = [0.01, 0.05, 0.1]
DROPOUTS = [0.1, 0.5]


def canonical_dataset_name(name):
    return ALIASES.get(name, name)


def cfg_grid():
    for lr, epochs, wd, dropout in itertools.product(LRS, EPOCHS, WEIGHT_DECAYS, DROPOUTS):
        yield {"lr": lr, "epochs": epochs, "weight_decay": wd, "dropout": dropout}


def cfg_tag(c):
    return f"lr{c['lr']:g}_ep{c['epochs']}_do{c['dropout']:g}_wd{c['weight_decay']:g}"


def build_cmd(dataset, c, seeds, gpu, out):
    prep = PREPROC[dataset]
    l_freq, h_freq, notch = BANDS[prep["band"]]
    cmd = [
        PYBIN, RUNNER,
        "--dataset", dataset,
        "--protocol", "loso",
        "--preset", "native70",
        "--gpu", str(gpu),
        "--seeds", *[str(s) for s in seeds],
        "--epochs", str(c["epochs"]),
        "--batch_size", str(FIXED["batch_size"]),
        "--dropout", str(c["dropout"]),
        "--label_smoothing", str(FIXED["label_smoothing"]),
        "--lr", str(c["lr"]),
        "--weight_decay", str(c["weight_decay"]),
        "--warmup_epochs", str(FIXED["warmup_epochs"]),
        "--min_lr", str(FIXED["min_lr"]),
        "--clip_grad_norm", str(FIXED["clip_grad_norm"]),
        "--norm_method", prep["norm"],
        "--scale_divisor", str(prep["scale"]),
        "--l_freq", str(l_freq),
        "--h_freq", str(h_freq),
        "--dataloader_workers", "0",
        "--out", out,
    ]
    if notch is not None:
        cmd += ["--notch_freq", str(notch)]
    return cmd


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
        cmd = build_cmd(dataset, c, seeds, gpu, out)
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


def with_preproc(dataset, c):
    prep = PREPROC[dataset]
    l_freq, h_freq, notch = BANDS[prep["band"]]
    return {
        **c,
        "batch_size": FIXED["batch_size"],
        "norm_method": prep["norm"],
        "scale_divisor": prep["scale"],
        "band": prep["band"],
        "l_freq": l_freq,
        "h_freq": h_freq,
        "notch_freq": notch,
    }


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
            rows.append({"dataset": ds, **with_preproc(ds, c), "search_bac": round(bac, 6)})
            if best is None or bac > best[0]:
                best = (bac, c)
        if best is not None:
            chosen[ds] = {**with_preproc(ds, best[1]), "search_bac": round(best[0], 6)}
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
        c = {k: chosen[ds][k] for k in ("lr", "epochs", "weight_decay", "dropout")}
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
