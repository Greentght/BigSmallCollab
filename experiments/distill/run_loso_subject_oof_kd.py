"""LOSO KD with subject cross-fitted teacher supervision.

Compares exactly three student conditions for each outer LOSO fold:

  CE             : no teacher supervision
  InSampleKD     : teacher logits from the ordinary LOSO teacher trained on all
                   eight source subjects, including the sample's subject
  SubjectOOFKD   : teacher logits from source-subject OOF inner teachers; each
                   source subject is predicted by a teacher that did not see it

No confidence gating, correct-only masking, flipped targets, feature loss, or new
KD objective is applied. The script also writes teacher diagnostics and in-sample
vs OOF logit-difference summaries aligned to the student train rows.

Prerequisite:
    conda run -n mirepnet python scripts/export/export_teacher_loso_subjoof.py \
        --dataset BNCI2014001-4 --seeds 666 --gpu 0

Run:
    conda run -n mirepnet python experiments/distill/run_loso_subject_oof_kd.py \
        --dataset BNCI2014001-4 --seeds 666 --gpu 0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch

from collab.distill import distill_student
import config
import data
from collab import artifacts
from eval import metrics
from models import get_adapter

INSAMPLE_NAME = "mirepnet_loso"
OOF_NAME = "mirepnet_loso_subjoof"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="BNCI2014001-4")
    p.add_argument("--student", default="ifnet")
    p.add_argument("--folds", type=int, nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--lam_kd", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--insample_teacher", default=INSAMPLE_NAME)
    p.add_argument("--oof_teacher", default=OOF_NAME)
    p.add_argument("--out_csv", default=None)
    p.add_argument("--teacher_diag_csv", default=None)
    p.add_argument("--logit_diff_csv", default=None)
    p.add_argument("--sample_diff_csv", default=None)
    p.add_argument("--summary_csv", default=None)
    p.add_argument("--no_sample_diffs", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--torch_threads", type=int, default=int(os.environ.get("TORCH_THREADS", "4")))
    return p.parse_args()


def _repo_path(*parts):
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        *parts,
    )


def _softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _ece(probs, y, n_bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = pred == y
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    n = len(y)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        out += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def _teacher_metrics(logits, y):
    probs = _softmax(logits)
    idx = np.arange(len(y))
    true_prob = probs[idx, y]
    ent = -(probs * np.log(probs + 1e-12)).sum(1)
    pred = probs.argmax(1)
    return {
        "acc": round(float((pred == y).mean() * 100.0), 4),
        "entropy": round(float(ent.mean()), 6),
        "nll": round(float((-np.log(true_prob + 1e-12)).mean()), 6),
        "ece": round(_ece(probs, y), 6),
        "conf": round(float(probs.max(1).mean()), 6),
        "true_prob": round(float(true_prob.mean()), 6),
    }


def _logit_diff(in_logits, oof_logits, y):
    pin = _softmax(in_logits)
    poof = _softmax(oof_logits)
    in_pred = pin.argmax(1)
    oof_pred = poof.argmax(1)
    idx = np.arange(len(y))
    in_ent = -(pin * np.log(pin + 1e-12)).sum(1)
    oof_ent = -(poof * np.log(poof + 1e-12)).sum(1)
    in_centered = in_logits - in_logits.mean(1, keepdims=True)
    oof_centered = oof_logits - oof_logits.mean(1, keepdims=True)
    kl_in_oof = (pin * (np.log(pin + 1e-12) - np.log(poof + 1e-12))).sum(1)
    kl_oof_in = (poof * (np.log(poof + 1e-12) - np.log(pin + 1e-12))).sum(1)
    in_correct = in_pred == y
    oof_correct = oof_pred == y
    return {
        "top1_agree": round(float((in_pred == oof_pred).mean() * 100.0), 4),
        "prob_l1": round(float(np.abs(pin - poof).sum(1).mean()), 6),
        "centered_logit_l2": round(float(np.linalg.norm(in_centered - oof_centered, axis=1).mean()), 6),
        "kl_in_to_oof": round(float(kl_in_oof.mean()), 6),
        "kl_oof_to_in": round(float(kl_oof_in.mean()), 6),
        "entropy_delta_oof_minus_in": round(float((oof_ent - in_ent).mean()), 6),
        "true_prob_delta_oof_minus_in": round(float((poof[idx, y] - pin[idx, y]).mean()), 6),
        "both_correct": round(float((in_correct & oof_correct).mean() * 100.0), 4),
        "in_only_correct": round(float((in_correct & ~oof_correct).mean() * 100.0), 4),
        "oof_only_correct": round(float((~in_correct & oof_correct).mean() * 100.0), 4),
        "both_wrong": round(float((~in_correct & ~oof_correct).mean() * 100.0), 4),
    }


def _sample_diff_rows(dataset, fold, seed, subj, in_logits, oof_logits, y):
    pin = _softmax(in_logits)
    poof = _softmax(oof_logits)
    idx = np.arange(len(y))
    in_pred = pin.argmax(1)
    oof_pred = poof.argmax(1)
    in_ent = -(pin * np.log(pin + 1e-12)).sum(1)
    oof_ent = -(poof * np.log(poof + 1e-12)).sum(1)
    in_centered = in_logits - in_logits.mean(1, keepdims=True)
    oof_centered = oof_logits - oof_logits.mean(1, keepdims=True)
    kl_in_oof = (pin * (np.log(pin + 1e-12) - np.log(poof + 1e-12))).sum(1)
    kl_oof_in = (poof * (np.log(poof + 1e-12) - np.log(pin + 1e-12))).sum(1)
    rows = []
    for i in range(len(y)):
        rows.append({
            "dataset": dataset,
            "fold": fold,
            "seed": seed,
            "train_row": int(i),
            "source_subject": int(subj[i]),
            "y": int(y[i]),
            "insample_pred": int(in_pred[i]),
            "oof_pred": int(oof_pred[i]),
            "pred_agree": bool(in_pred[i] == oof_pred[i]),
            "insample_correct": bool(in_pred[i] == y[i]),
            "oof_correct": bool(oof_pred[i] == y[i]),
            "insample_conf": float(pin[i].max()),
            "oof_conf": float(poof[i].max()),
            "insample_true_prob": float(pin[i, y[i]]),
            "oof_true_prob": float(poof[i, y[i]]),
            "insample_entropy": float(in_ent[i]),
            "oof_entropy": float(oof_ent[i]),
            "kl_in_to_oof": float(kl_in_oof[i]),
            "kl_oof_to_in": float(kl_oof_in[i]),
            "centered_logit_l2": float(np.linalg.norm(in_centered[i] - oof_centered[i])),
        })
    return rows


def _apply_student_overrides(cfg, args):
    cfg = dict(cfg)
    for key in ("epochs", "lr", "weight_decay", "batch_size"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    return cfg


def _device(gpu):
    if gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        return f"cuda:{gpu}"
    return "cpu"


def _append_teacher_diag(rows, dataset, fold, seed, subj, y, tag, logits):
    scopes = [("all", -1, np.ones(len(y), dtype=bool))]
    scopes += [(f"subject_{int(s)}", int(s), subj == s) for s in sorted(np.unique(subj))]
    for scope, source_subject, mask in scopes:
        m = _teacher_metrics(logits[mask], y[mask])
        rows.append({
            "dataset": dataset,
            "fold": fold,
            "seed": seed,
            "teacher_type": tag,
            "scope": scope,
            "source_subject": source_subject,
            "n": int(mask.sum()),
            **m,
        })


def _append_logit_diffs(rows, dataset, fold, seed, subj, y, in_logits, oof_logits):
    scopes = [("all", -1, np.ones(len(y), dtype=bool))]
    scopes += [(f"subject_{int(s)}", int(s), subj == s) for s in sorted(np.unique(subj))]
    for scope, source_subject, mask in scopes:
        rows.append({
            "dataset": dataset,
            "fold": fold,
            "seed": seed,
            "scope": scope,
            "source_subject": source_subject,
            "n": int(mask.sum()),
            **_logit_diff(in_logits[mask], oof_logits[mask], y[mask]),
        })


def _summarize(student_rows):
    df = pd.DataFrame(student_rows)
    if df.empty:
        return pd.DataFrame()

    summaries = []
    for cond, g in df.groupby("condition"):
        summaries.append({
            "comparison": f"{cond}_absolute",
            "condition": cond,
            "n": len(g),
            "mean_acc": round(float(g.acc.mean()), 4),
            "std_acc": round(float(g.acc.std(ddof=1)), 4) if len(g) > 1 else 0.0,
            "mean_kappa": round(float(g.kappa.mean()), 6),
            "std_kappa": round(float(g.kappa.std(ddof=1)), 6) if len(g) > 1 else 0.0,
        })

    piv_acc = df.pivot_table(index=["dataset", "fold", "seed"], columns="condition", values="acc")
    piv_kappa = df.pivot_table(index=["dataset", "fold", "seed"], columns="condition", values="kappa")
    if "CE" in piv_acc:
        for cond in ("InSampleKD", "SubjectOOFKD"):
            if cond not in piv_acc:
                continue
            gain_acc = (piv_acc[cond] - piv_acc["CE"]).dropna()
            gain_kappa = (piv_kappa[cond] - piv_kappa["CE"]).dropna()
            summaries.append({
                "comparison": f"{cond}_minus_CE",
                "condition": cond,
                "n": len(gain_acc),
                "mean_acc": round(float(gain_acc.mean()), 4),
                "std_acc": round(float(gain_acc.std(ddof=1)), 4) if len(gain_acc) > 1 else 0.0,
                "median_acc": round(float(gain_acc.median()), 4),
                "min_acc": round(float(gain_acc.min()), 4),
                "max_acc": round(float(gain_acc.max()), 4),
                "win_rate_acc": round(float((gain_acc > 0).mean()), 6),
                "nonnegative_rate_acc": round(float((gain_acc >= 0).mean()), 6),
                "mean_kappa": round(float(gain_kappa.mean()), 6),
                "std_kappa": round(float(gain_kappa.std(ddof=1)), 6) if len(gain_kappa) > 1 else 0.0,
                "median_kappa": round(float(gain_kappa.median()), 6),
                "min_kappa": round(float(gain_kappa.min()), 6),
                "max_kappa": round(float(gain_kappa.max()), 6),
                "win_rate_kappa": round(float((gain_kappa > 0).mean()), 6),
                "nonnegative_rate_kappa": round(float((gain_kappa >= 0).mean()), 6),
            })

    if {"InSampleKD", "SubjectOOFKD"}.issubset(set(piv_acc.columns)):
        delta_acc = (piv_acc["SubjectOOFKD"] - piv_acc["InSampleKD"]).dropna()
        delta_kappa = (piv_kappa["SubjectOOFKD"] - piv_kappa["InSampleKD"]).dropna()
        summaries.append({
            "comparison": "SubjectOOFKD_minus_InSampleKD",
            "condition": "SubjectOOFKD",
            "n": len(delta_acc),
            "mean_acc": round(float(delta_acc.mean()), 4),
            "std_acc": round(float(delta_acc.std(ddof=1)), 4) if len(delta_acc) > 1 else 0.0,
            "median_acc": round(float(delta_acc.median()), 4),
            "min_acc": round(float(delta_acc.min()), 4),
            "max_acc": round(float(delta_acc.max()), 4),
            "win_rate_acc": round(float((delta_acc > 0).mean()), 6),
            "nonnegative_rate_acc": round(float((delta_acc >= 0).mean()), 6),
            "mean_kappa": round(float(delta_kappa.mean()), 6),
            "std_kappa": round(float(delta_kappa.std(ddof=1)), 6) if len(delta_kappa) > 1 else 0.0,
            "median_kappa": round(float(delta_kappa.median()), 6),
            "min_kappa": round(float(delta_kappa.min()), 6),
            "max_kappa": round(float(delta_kappa.max()), 6),
            "win_rate_kappa": round(float((delta_kappa > 0).mean()), 6),
            "nonnegative_rate_kappa": round(float((delta_kappa >= 0).mean()), 6),
        })
    return pd.DataFrame(summaries)


def main():
    a = parse_args()
    if a.torch_threads > 0:
        torch.set_num_threads(a.torch_threads)

    dcfg = config.load_dataset_config(a.dataset)
    scfg = _apply_student_overrides(config.load_model_config(a.student), a)
    n_sub = dcfg["num_subjects"]
    nc = dcfg["num_classes"]
    folds = a.folds if a.folds is not None else list(range(n_sub))
    seeds = a.seeds if a.seeds is not None else [dcfg["seeds"][0]]
    device = _device(a.gpu)

    out_csv = a.out_csv or _repo_path(
        "results", "metrics",
        f"{a.dataset}_subject_oof_kd_{a.insample_teacher}_to_{a.student}.csv",
    )
    teacher_diag_csv = a.teacher_diag_csv or _repo_path(
        "results", "metrics",
        f"{a.dataset}_subject_oof_teacher_diag_{a.insample_teacher}.csv",
    )
    logit_diff_csv = a.logit_diff_csv or _repo_path(
        "results", "metrics",
        f"{a.dataset}_subject_oof_logit_diff_{a.insample_teacher}.csv",
    )
    sample_diff_csv = a.sample_diff_csv or _repo_path(
        "results", "metrics",
        f"{a.dataset}_subject_oof_sample_diffs_{a.insample_teacher}.csv",
    )
    summary_csv = a.summary_csv or _repo_path(
        "results", "metrics",
        f"{a.dataset}_subject_oof_kd_summary_{a.insample_teacher}_to_{a.student}.csv",
    )
    for path in (out_csv, teacher_diag_csv, logit_diff_csv, summary_csv):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    if not a.no_sample_diffs:
        os.makedirs(os.path.dirname(sample_diff_csv), exist_ok=True)

    common = dict(
        temperature=a.temperature,
        teacher_correct_only=False,
        balanced_batch=True,
        epochs=scfg.get("epochs", 100),
        lr=scfg.get("lr", 1e-3),
        weight_decay=scfg.get("weight_decay", 0.01),
        batch_size=scfg.get("batch_size", 16),
    )
    print(
        f"[cfg] dataset={a.dataset} student={a.student} seeds={seeds} "
        f"lam_kd={a.lam_kd} T={a.temperature} epochs={common['epochs']} "
        f"device={device}",
        flush=True,
    )

    student_rows = []
    teacher_rows = []
    logit_diff_rows = []
    sample_rows = []
    conditions = (
        ("CE", "insample", 0.0),
        ("InSampleKD", "insample", a.lam_kd),
        ("SubjectOOFKD", "oof", a.lam_kd),
    )

    for seed in seeds:
        for fold in folds:
            fold = int(fold)
            try:
                in_t = artifacts.load(a.dataset, a.insample_teacher, fold, seed, "train")
                oof_t = artifacts.load(a.dataset, a.oof_teacher, fold, seed, "train")
            except FileNotFoundError as e:
                print(f"[miss teacher] fold={fold} seed={seed}: {e}", flush=True)
                continue

            X_tr, y_tr, subj_tr, X_te, y_te = data.loso_split(a.dataset, fold)
            if not np.array_equal(in_t["y"], y_tr):
                raise ValueError(f"in-sample teacher rows misaligned for fold={fold} seed={seed}")
            if not np.array_equal(oof_t["y"], y_tr):
                raise ValueError(f"OOF teacher rows misaligned for fold={fold} seed={seed}")

            _append_teacher_diag(
                teacher_rows, a.dataset, fold, seed, subj_tr, y_tr,
                "InSample", in_t["logits"],
            )
            _append_teacher_diag(
                teacher_rows, a.dataset, fold, seed, subj_tr, y_tr,
                "SubjectOOF", oof_t["logits"],
            )
            _append_logit_diffs(
                logit_diff_rows, a.dataset, fold, seed, subj_tr, y_tr,
                in_t["logits"], oof_t["logits"],
            )
            if not a.no_sample_diffs:
                sample_rows.extend(_sample_diff_rows(
                    a.dataset, fold, seed, subj_tr,
                    in_t["logits"], oof_t["logits"], y_tr,
                ))

            acfg = dict(scfg)
            acfg.update(
                in_channels=X_tr.shape[1],
                samples=X_tr.shape[2],
                dataset_name=a.dataset,
            )
            teacher_by_key = {"insample": in_t, "oof": oof_t}
            for condition, teacher_key, lam_kd in conditions:
                tch = teacher_by_key[teacher_key]
                student = get_adapter(a.student, device=device, **acfg)
                preds = distill_student(
                    student, nc, X_tr, y_tr, tch["feats"], tch["logits"], X_te,
                    lam_kd=lam_kd, lam_feat=0.0, seed=seed, subject_ids=subj_tr,
                    **common,
                )
                m = metrics.evaluate(y_te, preds)
                row = {
                    "dataset": a.dataset,
                    "fold": fold,
                    "seed": seed,
                    "student": a.student,
                    "condition": condition,
                    "teacher_artifact": "" if condition == "CE" else (
                        a.insample_teacher if teacher_key == "insample" else a.oof_teacher
                    ),
                    "acc": m["acc"],
                    "kappa": m["kappa"],
                }
                row.update(metrics.per_class(y_te, preds, nc))
                student_rows.append(row)
                print(
                    f"fold={fold} seed={seed} {condition} | "
                    f"acc={m['acc']} kappa={m['kappa']}",
                    flush=True,
                )
                if device != "cpu":
                    torch.cuda.empty_cache()

    if not student_rows:
        print("No complete in-sample + Subject-OOF teacher artifacts found.")
        return

    pd.DataFrame(student_rows).to_csv(out_csv, index=False)
    pd.DataFrame(teacher_rows).to_csv(teacher_diag_csv, index=False)
    pd.DataFrame(logit_diff_rows).to_csv(logit_diff_csv, index=False)
    if not a.no_sample_diffs:
        pd.DataFrame(sample_rows).to_csv(sample_diff_csv, index=False)

    summary = _summarize(student_rows)
    summary.to_csv(summary_csv, index=False)
    print(f"\nWrote {out_csv}")
    print(f"Wrote {teacher_diag_csv}")
    print(f"Wrote {logit_diff_csv}")
    if not a.no_sample_diffs:
        print(f"Wrote {sample_diff_csv}")
    print(f"Wrote {summary_csv}")
    print("\nStudent means:")
    print(pd.DataFrame(student_rows).groupby("condition")[["acc", "kappa"]].mean().round(4))
    print("\nStability summary:")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
