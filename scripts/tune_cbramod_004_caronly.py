"""Focused CAR-only re-tune of native CBraMod on BNCI2014004 (the one dataset still
lagging the EEGFMBench record after the CAR-only fix). Fixes norm=car + scale=1;
sweeps lr/epochs/dropout/weight_decay/band per split {0.7,0.3}; picks best BAC on
1 seed then confirms with 3 seeds. Resumable per-config CSVs. GPUs default 1,7,8,9.
"""
import itertools, json, os, queue, subprocess
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.environ.get("CBRAMOD_PYBIN", "/home/lixinli/anaconda3/envs/cbramod/bin/python")
RUNNER = os.path.join(ROOT, "scripts", "cbramod_native_adapt.py")
TUNE = os.path.join(ROOT, "results", "cbramod_native", "tune004")
TUNED = os.path.join(ROOT, "results", "cbramod_native", "tuned004")
LOG = os.path.join(ROOT, "logs", "tune_cbramod")
DS, NSUBJ = "BNCI2014004", 9
GPUS = [1, 7, 8, 9]

LRS, EPOCHS, DROPOUTS, WDS = [5e-4, 1e-3], [20, 50], [0.1, 0.5], [0.01, 0.05, 0.1]
BANDS = {"b50": (0.3, 50.0, None), "b75n60": (0.3, 75.0, 60.0)}
TPS = [0.7, 0.3]
SEARCH_SEED, CONFIRM = 666, [666, 667, 668]


def grid():
    for lr, ep, do, wd, band in itertools.product(LRS, EPOCHS, DROPOUTS, WDS, BANDS):
        yield dict(lr=lr, epochs=ep, dropout=do, weight_decay=wd, band=band)


def tag(c):
    return f"lr{c['lr']:g}_ep{c['epochs']}_do{c['dropout']:g}_wd{c['weight_decay']:g}_{c['band']}"


def cmd(c, tp, seeds, gpu, out):
    l, h, notch = BANDS[c["band"]]
    x = [PY, RUNNER, "--dataset", DS, "--preset", "native70", "--train_percentage", str(tp),
         "--gpu", str(gpu), "--seeds", *map(str, seeds), "--epochs", str(c["epochs"]),
         "--batch_size", "16", "--dropout", str(c["dropout"]), "--label_smoothing", "0.0",
         "--lr", str(c["lr"]), "--weight_decay", str(c["weight_decay"]), "--warmup_epochs", "5",
         "--min_lr", "1e-6", "--clip_grad_norm", "1.0", "--norm_method", "car", "--scale_divisor", "1",
         "--l_freq", str(l), "--h_freq", str(h), "--dataloader_workers", "0", "--overwrite", "--out", out]
    if notch is not None:
        x += ["--notch_freq", str(notch)]
    return x


def done(out, nseeds):
    if not os.path.exists(out):
        return False
    try:
        return len(pd.read_csv(out)) >= NSUBJ * nseeds
    except Exception:
        return False


def run(job, gq, pref):
    c, tp, seeds, out, log = job
    g = gq.get()
    try:
        if done(out, len(seeds)):
            print(f"[skip] {pref} tp{tp} {tag(c)}", flush=True); return
        os.makedirs(os.path.dirname(out), exist_ok=True); os.makedirs(os.path.dirname(log), exist_ok=True)
        print(f"[run ] {pref} tp{tp} {tag(c)} gpu={g}", flush=True)
        with open(log, "w") as fh:
            subprocess.run(cmd(c, tp, seeds, g, out), stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    finally:
        gq.put(g)


def main():
    gq = queue.Queue()
    [gq.put(g) for g in GPUS]
    jobs = [(c, tp, [SEARCH_SEED], os.path.join(TUNE, f"tp{tp}", f"{tag(c)}.csv"),
             os.path.join(LOG, f"s004_tp{tp}_{tag(c)}.log")) for tp in TPS for c in grid()]
    print(f"[search] {len(jobs)} jobs", flush=True)
    with ThreadPoolExecutor(max_workers=len(GPUS)) as ex:
        list(ex.map(lambda j: run(j, gq, "search"), jobs))
    chosen = {}
    for tp in TPS:
        best = None
        for c in grid():
            out = os.path.join(TUNE, f"tp{tp}", f"{tag(c)}.csv")
            if not done(out, 1):
                continue
            b = float(pd.read_csv(out)["bac"].mean())
            if best is None or b > best[0]:
                best = (b, c)
        if best:
            chosen[str(tp)] = {**best[1], "search_bac": round(best[0], 6)}
            print(f"[best ] tp{tp}: {tag(best[1])} bac={best[0]:.4f}", flush=True)
    os.makedirs(TUNED, exist_ok=True)
    json.dump(chosen, open(os.path.join(TUNED, "chosen.json"), "w"), indent=2)
    # confirm
    cjobs = [({k: chosen[str(tp)][k] for k in ("lr", "epochs", "dropout", "weight_decay", "band")},
              tp, CONFIRM, os.path.join(TUNED, f"tp{tp}.csv"),
              os.path.join(LOG, f"c004_tp{tp}.log")) for tp in TPS if str(tp) in chosen]
    print(f"[confirm] {len(cjobs)} jobs", flush=True)
    with ThreadPoolExecutor(max_workers=len(GPUS)) as ex:
        list(ex.map(lambda j: run(j, gq, "confirm"), cjobs))
    for tp in TPS:
        out = os.path.join(TUNED, f"tp{tp}.csv")
        if os.path.exists(out):
            d = pd.read_csv(out)
            print(f"[final] tp{tp}: acc={d.acc.mean():.2f} bac={d.bac.mean():.4f} kappa={d.kappa.mean():.4f}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
