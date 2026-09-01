"""Subject-OOF LOSO teacher finetune.

For an outer LOSO fold f, the student trains on all subjects except f. This
script creates teacher targets for those source-subject samples by cross-fitting
inside the outer training set:

    q_i = T_{outer=f, heldout=subj(i)}(x_i)

The inner teacher for source subject s is trained on the other seven source
subjects only. It never sees s, and it never uses the outer test subject f. The
saved artifact is row-aligned with data.loso_split(dataset, f)'s train split and
is keyed as:

    results/artifacts/<dataset>/mirepnet_loso_subjoof/<f>_<seed>_train.npz

Example:
    conda run -n mirepnet python experiments/finetune/finetune_teacher_loso_subjoof.py \
        --dataset BNCI2014001-4 --seeds 666 --gpu 0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from collab.distill import _set_seed
import config
import data
from collab import artifacts
from models import get_adapter

TEACHER = "mirepnet"
OOF_NAME = "mirepnet_loso_subjoof"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="BNCI2014001-4")
    p.add_argument("--folds", type=int, nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--torch_threads", type=int, default=int(os.environ.get("TORCH_THREADS", "4")))
    return p.parse_args()


def _apply_overrides(cfg, args):
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


def main():
    a = parse_args()
    if a.torch_threads > 0:
        torch.set_num_threads(a.torch_threads)

    dcfg = config.load_dataset_config(a.dataset)
    mcfg = _apply_overrides(config.load_model_config(TEACHER), a)
    n_sub = dcfg["num_subjects"]
    nc = dcfg["num_classes"]
    folds = a.folds if a.folds is not None else list(range(n_sub))
    seeds = a.seeds if a.seeds is not None else [dcfg["seeds"][0]]
    device = _device(a.gpu)

    print(
        f"[cfg] dataset={a.dataset} artifact={OOF_NAME} epochs={mcfg.get('epochs')} "
        f"bs={mcfg.get('batch_size')} lr={mcfg.get('lr'):g} "
        f"wd={mcfg.get('weight_decay'):g} device={device}",
        flush=True,
    )

    for seed in seeds:
        for outer_fold in folds:
            outer_fold = int(outer_fold)
            if (artifacts.exists(a.dataset, OOF_NAME, outer_fold, seed, "train")
                    and not a.overwrite):
                print(f"[skip] outer_fold={outer_fold} seed={seed}", flush=True)
                continue

            X_tr, y_tr, subj_tr, _X_te, _y_te = data.loso_split(a.dataset, outer_fold)
            logits_oof = None
            feats_oof = None
            source_subjects = sorted(int(s) for s in np.unique(subj_tr))

            for source_subject in source_subjects:
                src_mask = subj_tr == source_subject
                fit_mask = subj_tr != source_subject
                X_fit, y_fit, subj_fit = X_tr[fit_mask], y_tr[fit_mask], subj_tr[fit_mask]
                X_src, y_src = X_tr[src_mask], y_tr[src_mask]

                acfg = dict(mcfg)
                acfg.update(
                    in_channels=X_fit.shape[1],
                    samples=X_fit.shape[2],
                    dataset_name=a.dataset,
                    skip_preprocess=True,
                )
                ad = get_adapter(TEACHER, device=device, **acfg)

                Xp_fit = ad.ea_pad_per_subject(X_fit, subj_fit)
                Xp_src = ad.ea_pad_per_subject(
                    X_src, np.full(len(y_src), source_subject, dtype=np.int64)
                )

                _set_seed(seed)
                model = ad.build(nc)
                model = ad.finetune(model, Xp_fit, y_fit, nc)
                feats, logits = ad.infer(model, Xp_src)

                if logits_oof is None:
                    logits_oof = np.empty((len(y_tr), logits.shape[1]), dtype=np.float32)
                    feats_oof = np.empty((len(y_tr), feats.shape[1]), dtype=np.float32)
                logits_oof[src_mask] = logits.astype(np.float32)
                feats_oof[src_mask] = feats.astype(np.float32)

                acc = (logits.argmax(1) == y_src).mean() * 100.0
                print(
                    f"[inner] outer_fold={outer_fold} source_subject={source_subject} "
                    f"seed={seed} n_fit={len(y_fit)} n_oof={len(y_src)} acc={acc:.2f}",
                    flush=True,
                )
                del model
                if device != "cpu":
                    torch.cuda.empty_cache()

            path = artifacts.save(
                a.dataset, OOF_NAME, outer_fold, seed, "train",
                logits=logits_oof, feats=feats_oof, y=y_tr,
            )
            print(
                f"[ok] outer_fold={outer_fold} seed={seed} "
                f"n_train={len(y_tr)} wrote={path}",
                flush=True,
            )

    print("Done.")


if __name__ == "__main__":
    main()
