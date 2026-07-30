"""Per-(dataset, split) hyperparameter tuning of the *native* CBraMod adapter.

Drives ``cbramod_native_adapt.py`` (the native ``all_patch_reps`` model + faithful
native preprocessing recorded in PROGRESS.md 2026-07-09). It does NOT re-implement
training; it only sweeps a grid and selects the best config per (dataset, split)
by mean balanced accuracy on one search seed, then confirms with 3 seeds.

Protocol (user decisions 2026-07-12/13):
  * The most-native CBraMod is the target model (official all_patch_reps head +
    12-layer backbone + pretrained weights); nothing about the model is changed.
  * Splits = within-subject stratified train_percentage in {0.7, 0.3}. Each
    (dataset, split) gets its own separately-tuned optimal config.
  * Swept levers (all treated as archives): scale_divisor, dropout, weight_decay,
    band (l/h/notch), lr, epochs. Everything else fixed (see FIXED).

Selection is config-level (matches how the reference tuning record was produced);
each run reports last-epoch metrics (no best-epoch leakage). Resumable via
per-config CSVs.
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
PYBIN = os.environ.get(
    "CBRAMOD_PYBIN", "/home/lixinli/anaconda3/envs/cbramod/bin/python"
)
RUNNER = os.path.join(ROOT, "scripts", "bigmodel", "cbramod_native_adapt.py")
TUNE_DIR = os.path.join(ROOT, "results", "cbramod_native", "tune")
TUNED_DIR = os.path.join(ROOT, "results", "cbramod_native", "tuned")
LOG_DIR = os.path.join(ROOT, "logs", "tune_cbramod")

# subjects per source (for resume/completeness checks)
N_SUBJECTS = {
    "BNCI2014001_4c": 9,
    "BNCI2014001_2c": 9,
    "BNCI2014004": 9,
    "AlexMI_2c": 8,
    "BNCI2015001": 12,
}

# Fixed hyperparameters (harness defaults / constant across the reference CBraMod MI runs)
FIXED = dict(
    batch_size=16,
    warmup_epochs=5,
    min_lr=1e-6,
    clip_grad_norm=1.0,
    label_smoothing=0.0,
)

# ---- swept levers (the "档位") -----------------------------------------------
SCALES = [10.0, 100.0]           # native /100 vs the 07-09 runner's /10
DROPOUTS = [0.1, 0.5]            # CBraMod baseline 0.1 vs 07-09 runner's 0.5
WEIGHT_DECAYS = [0.01, 0.05, 0.1]
LRS = [5e-4, 1e-3]
EPOCHS = [20, 50]
BANDS = {
    # name: (l_freq, h_freq, notch_freq or None)
    "b50": (0.3, 50.0, None),      # native default (07-09 baseline)
    "b75n60": (0.3, 75.0, 60.0),   # reference record's CBraMod MI band
}

# splits: within-subject stratified train fraction
TRAIN_PCTS = [0.7, 0.3]

SEARCH_SEED = 666
CONFIRM_SEEDS = [666, 667, 668]
ALL_DATASETS = list(N_SUBJECTS)


def cfg_grid():
    for scale, dropout, wd, lr, epochs, band in itertools.product(
        SCALES, DROPOUTS, WEIGHT_DECAYS, LRS, EPOCHS, BANDS
    ):
        yield {
            "scale": scale, "dropout": dropout, "weight_decay": wd,
            "lr": lr, "epochs": epochs, "band": band,
        }


def cfg_tag(c):
    return (f"lr{c['lr']:g}_ep{c['epochs']}_do{c['dropout']:g}"
            f"_wd{c['weight_decay']:g}_sc{c['scale']:g}_{c['band']}")


def tp_tag(tp):
    return f"tp{tp:g}"


def build_cmd(dataset, c, tp, seeds, gpu, out):
    l, h, notch = BANDS[c["band"]]
    cmd = [
        PYBIN, RUNNER,
        "--dataset", dataset,
        "--preset", "native70",
        "--train_percentage", str(tp),
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
        "--scale_divisor", str(c["scale"]),
        "--l_freq", str(l),
        "--h_freq", str(h),
        "--dataloader_workers", "0",
        "--out", out,
    ]
    if notch is not None:
        cmd += ["--notch_freq", str(notch)]
    return cmd


def is_complete(out, dataset, n_seeds):
    if not os.path.exists(out):
        return False
    try:
        df = pd.read_csv(out)
    except Exception:
        return False
    return len(df) >= N_SUBJECTS[dataset] * n_seeds


def run_job(job, gpu_q, log_prefix):
    dataset, c, tp, seeds, out, log = job
    gpu = gpu_q.get()
    try:
        if is_complete(out, dataset, len(seeds)):
            print(f"[skip] {log_prefix} {dataset} {tp_tag(tp)} {cfg_tag(c)} (done)", flush=True)
            return dataset, c, tp, out, "skip"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        os.makedirs(os.path.dirname(log), exist_ok=True)
        cmd = build_cmd(dataset, c, tp, seeds, gpu, out)
        print(f"[run ] {log_prefix} {dataset} {tp_tag(tp)} {cfg_tag(c)} gpu={gpu}", flush=True)
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        status = "ok" if rc == 0 else f"FAIL(rc={rc})"
        print(f"[done] {log_prefix} {dataset} {tp_tag(tp)} {cfg_tag(c)} gpu={gpu} -> {status}", flush=True)
        return dataset, c, tp, out, status
    finally:
        gpu_q.put(gpu)


def mean_bac(out):
    return float(pd.read_csv(out)["bac"].mean())


def phase_search(datasets, gpus):
    gpu_q = queue.Queue()
    for g in gpus:
        gpu_q.put(g)
    jobs = []
    for ds in datasets:
        for tp in TRAIN_PCTS:
            for c in cfg_grid():
                out = os.path.join(TUNE_DIR, ds, tp_tag(tp), f"{cfg_tag(c)}.csv")
                log = os.path.join(LOG_DIR, f"search_{ds}_{tp_tag(tp)}_{cfg_tag(c)}.log")
                jobs.append((ds, c, tp, [SEARCH_SEED], out, log))
    print(f"[search] {len(jobs)} jobs across gpus {gpus}", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        list(ex.map(lambda j: run_job(j, gpu_q, "search"), jobs))

    # pick best config per (dataset, split) by mean BAC
    chosen = {}
    rows = []
    for ds in datasets:
        for tp in TRAIN_PCTS:
            best = None
            for c in cfg_grid():
                out = os.path.join(TUNE_DIR, ds, tp_tag(tp), f"{cfg_tag(c)}.csv")
                if not is_complete(out, ds, 1):
                    print(f"[warn] incomplete: {out}", flush=True)
                    continue
                bac = mean_bac(out)
                rows.append({"dataset": ds, "train_pct": tp, **c, "search_bac": round(bac, 6)})
                if best is None or bac > best[0]:
                    best = (bac, c)
            if best is not None:
                chosen[f"{ds}|{tp}"] = {**best[1], "search_bac": round(best[0], 6)}
                print(f"[best ] {ds} {tp_tag(tp)}: {cfg_tag(best[1])} bac={best[0]:.4f}", flush=True)
    os.makedirs(TUNED_DIR, exist_ok=True)
    if rows:
        pd.DataFrame(rows).sort_values(
            ["dataset", "train_pct", "search_bac"], ascending=[True, True, False]
        ).to_csv(os.path.join(TUNE_DIR, "search_summary.csv"), index=False)
    with open(os.path.join(TUNED_DIR, "chosen_configs.json"), "w") as fh:
        json.dump(chosen, fh, indent=2)
    return chosen


def load_chosen():
    with open(os.path.join(TUNED_DIR, "chosen_configs.json")) as fh:
        return json.load(fh)


def phase_confirm(datasets, gpus, chosen):
    gpu_q = queue.Queue()
    for g in gpus:
        gpu_q.put(g)
    jobs = []
    for ds in datasets:
        for tp in TRAIN_PCTS:
            key = f"{ds}|{tp}"
            if key not in chosen:
                print(f"[warn] no chosen config for {key}", flush=True)
                continue
            c = {k: chosen[key][k] for k in ("scale", "dropout", "weight_decay", "lr", "epochs", "band")}
            out = os.path.join(TUNED_DIR, f"{ds}_{tp_tag(tp)}.csv")
            log = os.path.join(LOG_DIR, f"confirm_{ds}_{tp_tag(tp)}.log")
            jobs.append((ds, c, tp, CONFIRM_SEEDS, out, log))
    print(f"[confirm] {len(jobs)} jobs across gpus {gpus}", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        list(ex.map(lambda j: run_job(j, gpu_q, "confirm"), jobs))
    # final summary
    rows = []
    for ds in datasets:
        for tp in TRAIN_PCTS:
            out = os.path.join(TUNED_DIR, f"{ds}_{tp_tag(tp)}.csv")
            if not os.path.exists(out):
                continue
            df = pd.read_csv(out)
            rows.append({
                "dataset": ds, "train_pct": tp,
                "acc": round(df.acc.mean(), 2), "bac": round(df.bac.mean(), 4),
                "kappa": round(df.kappa.mean(), 4), "n": len(df),
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
    return p.parse_args()


def main():
    args = parse_args()
    datasets = [d for d in args.datasets if d in N_SUBJECTS]
    if args.phase in ("search", "all"):
        chosen = phase_search(datasets, args.gpus)
    else:
        chosen = load_chosen()
    if args.phase in ("confirm", "all"):
        phase_confirm(datasets, args.gpus, chosen)
    print("[tune] all phases done", flush=True)


if __name__ == "__main__":
    main()
