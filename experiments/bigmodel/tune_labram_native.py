"""Per-dataset hyperparameter tuning of the *native* LaBraM adapter.

Parallel to ``tune_cbramod_native.py``: drives ``labram_native_adapt.py`` (the
faithful native preprocessing + pretrained labram_base backbone + native
mean-pool/Linear head). The **model structure is never changed** — only optimizer
/ regularization / preprocessing-band levers are swept, mirroring the CBraMod
tuning record's spirit (dropout->drop_path; +layer_decay, LaBraM's key finetune
lever).

Two phases:
  * search  — every config on one seed (666), all subjects; pick best per dataset
    by mean **validation** balanced-acc (``val_bac``). Selecting on val, not test,
    avoids the test leakage the CBraMod tuner had (it selected on test ``bac``).
  * confirm — rerun the chosen config with 3 seeds; report last (best-val) test.

Split = within-subject native70 (tp0.7) only, matching the user's LaBraM run.
Resumable via per-config CSVs.
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
PYBIN = os.environ.get("LABRAM_PYBIN", "/home/lixinli/anaconda3/envs/labram/bin/python")
RUNNER = os.path.join(ROOT, "experiments", "bigmodel", "labram_native_adapt.py")
TUNE_DIR = os.path.join(ROOT, "results", "labram_native", "tune2")
TUNED_DIR = os.path.join(ROOT, "results", "labram_native", "tuned2")
LOG_DIR = os.path.join(ROOT, "logs", "tune_labram2")

N_SUBJECTS = {
    "BNCI2014001_4c": 9,
    "BNCI2014001_2c": 9,
    "BNCI2014004": 9,
    "AlexMI_2c": 8,
    "BNCI2015001": 12,
}

# Fixed (structure + constants; not swept). Aligned to the benchmark's LaBraM
# reference: label_smoothing 0, min_lr 1e-5, no grad clipping, native band.
FIXED = dict(
    scale_divisor=100.0,   # shown ~irrelevant (LayerNorm after patch_embed)
    epochs=50,             # budget; val-select picks the epoch, so no epoch sweep
    drop_path=0.1,         # rely on weight_decay (not drop_path) for regularization
    band="b75n50",         # LaBraM native band (reference preprocessing)
    warmup_epochs=5,
    min_lr=1e-5,
    clip_grad_norm=0.0,    # reference: use_grad_clipping=False
    label_smoothing=0.0,
    val_split=0.2,
)

# ---- swept levers (refined grid, centered on the benchmark's LaBraM config:
#      low lr 1e-4 + high wd 0.5 + small batch 8 + layer_decay 1.0) -----------
LRS = [1e-4, 3e-4, 5e-4]
WEIGHT_DECAYS = [0.1, 0.5]
LAYER_DECAYS = [1.0, 0.65]       # 1.0 = uniform lr (reference); 0.65 = strong decay
BATCH_SIZES = [8, 16]
BANDS = {
    "b75n50": (0.1, 75.0, 50.0),  # LaBraM native band
    "b50": (0.3, 50.0, None),     # CBraMod-style band (kept for build_cmd lookup)
}

TRAIN_PCTS = [0.7]
SEARCH_SEED = 666
CONFIRM_SEEDS = [666, 667, 668]
ALL_DATASETS = list(N_SUBJECTS)


def cfg_grid():
    for lr, wd, ld, bs in itertools.product(
        LRS, WEIGHT_DECAYS, LAYER_DECAYS, BATCH_SIZES
    ):
        yield {"lr": lr, "weight_decay": wd, "layer_decay": ld,
               "batch_size": bs, "drop_path": FIXED["drop_path"],
               "band": FIXED["band"]}


def cfg_tag(c):
    return (f"lr{c['lr']:g}_wd{c['weight_decay']:g}_ld{c['layer_decay']:g}"
            f"_bs{c['batch_size']}_{c['band']}")


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
        "--epochs", str(FIXED["epochs"]),
        "--batch_size", str(c["batch_size"]),
        "--val_split", str(FIXED["val_split"]),
        "--drop_path", str(c["drop_path"]),
        "--label_smoothing", str(FIXED["label_smoothing"]),
        "--lr", str(c["lr"]),
        "--layer_decay", str(c["layer_decay"]),
        "--weight_decay", str(c["weight_decay"]),
        "--warmup_epochs", str(FIXED["warmup_epochs"]),
        "--min_lr", str(FIXED["min_lr"]),
        "--clip_grad_norm", str(FIXED["clip_grad_norm"]),
        "--scale_divisor", str(FIXED["scale_divisor"]),
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


def mean_val_bac(out):
    df = pd.read_csv(out)
    col = "val_bac" if "val_bac" in df.columns else "bac"
    return float(df[col].mean())


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
                vbac = mean_val_bac(out)
                tbac = float(pd.read_csv(out)["bac"].mean())
                rows.append({"dataset": ds, "train_pct": tp, **c,
                             "search_val_bac": round(vbac, 6), "search_test_bac": round(tbac, 6)})
                if best is None or vbac > best[0]:
                    best = (vbac, c)
            if best is not None:
                chosen[f"{ds}|{tp}"] = {**best[1], "search_val_bac": round(best[0], 6)}
                print(f"[best ] {ds} {tp_tag(tp)}: {cfg_tag(best[1])} val_bac={best[0]:.4f}", flush=True)
    os.makedirs(TUNED_DIR, exist_ok=True)
    if rows:
        pd.DataFrame(rows).sort_values(
            ["dataset", "train_pct", "search_val_bac"], ascending=[True, True, False]
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
            c = {k: chosen[key][k] for k in ("lr", "weight_decay", "layer_decay", "batch_size", "drop_path", "band")}
            out = os.path.join(TUNED_DIR, f"{ds}_{tp_tag(tp)}.csv")
            log = os.path.join(LOG_DIR, f"confirm_{ds}_{tp_tag(tp)}.log")
            jobs.append((ds, c, tp, CONFIRM_SEEDS, out, log))
    print(f"[confirm] {len(jobs)} jobs across gpus {gpus}", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        list(ex.map(lambda j: run_job(j, gpu_q, "confirm"), jobs))
    rows = []
    for ds in datasets:
        for tp in TRAIN_PCTS:
            out = os.path.join(TUNED_DIR, f"{ds}_{tp_tag(tp)}.csv")
            if not os.path.exists(out):
                continue
            df = pd.read_csv(out)
            key = f"{ds}|{tp}"
            rows.append({
                "dataset": ds, "train_pct": tp,
                "acc": round(df.acc.mean(), 2), "acc_std": round(df.acc.std(), 2),
                "bac": round(df.bac.mean(), 4), "kappa": round(df.kappa.mean(), 4),
                "n": len(df), "cfg": cfg_tag(chosen[key]) if key in chosen else "",
            })
    if rows:
        summ = pd.DataFrame(rows)
        summ.to_csv(os.path.join(TUNED_DIR, "summary_tuned.csv"), index=False)
        print(summ.to_string(index=False), flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["search", "confirm", "all"], default="all")
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    p.add_argument("--gpus", type=int, nargs="+", default=[2, 3, 4, 5, 7])
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
