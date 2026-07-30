# Big/Small Collaboration Experiment Context

Last checked: 2026-07-28, from the local repo and `/data1/llx/*`.

This note separates three things that were easy to conflate:

1. What result was obtained.
2. Under which training/fine-tuning scenario it was obtained.
3. Which raw dataset rows enter each subject/session pool.

## Bottom Line

The current honest conclusion is:

- The original target, "end-to-end collaborative adaptation beats the better of end-to-end fine-tuned big/small single models", has not been achieved.
- The robust positive result is narrower: in a frozen-feature / cached-artifact setting, adding a feature-fusion head as a candidate in a support-CV selector (`cv_sel3`) improves over selecting only the frozen big/small heads (`cv_sel2`) by about +0.5 to +0.6 accuracy points across 4 datasets and 24 cells.
- The fine-tune baseline narrows the scope: once a single model is allowed to be end-to-end fine-tuned on the K support trials, the best single fine-tuned model is stronger than frozen fusion in the recorded flagship run.

## Experiment Families

| Family | Scenario | Datasets | Models | Split/adaptation | Result status | Main files |
|---|---|---|---|---|---|---|
| D0 headroom diagnosis | Cached predictions/features, within and LOSO | BNCI2014001-4, BNCI2014004 | big: MIRepNet, CBraMod native; small: IFNet, EEGNet, ADFCNN | within = per-subject 70/30 on downstream session; LOSO = train all other subjects, test held-out subject | Oracle union headroom is real, about +9 to +17, but static cross-subject route/fuse does not capture it | `results/d0/decision_report.md`, `results/headroom_map.csv` |
| R1 logit router | Cached LOSO logits, learned gate | flagship: BNCI2014001-4, MIRepNet x IFNet | MIRepNet, IFNet | nested subject-LOSO gate training over cached OOS logits | Negative: static gate worse than baseline; same-subject calibration only +0.4 ns | `collab/router.py`, `scripts/fusion/run_r1_signal.py`, `PROGRESS.md` |
| KD / offline distillation | Student trained from raw train split against frozen teacher artifacts | mainly BNCI2014001-4, BNCI2014004; some BNCI2015001 | teacher: MIRepNet/CBraMod; student: IFNet/EEGNet/ADFCNN | within 70/30 or LOSO, depending script | Mechanistic negative / no robust main positive | `scripts/distill/*`, `results/metrics/*distill*` |
| Bidirectional / mutual | Both models trained together in same env | mainly MIRepNet x IFNet | MIRepNet, IFNet | LOSO or few-shot variants | Negative / unstable; feature-level mutual did not beat concat fusion | `scripts/bidir/*`, `results/metrics/*bidir*`, `*cramd*`, `*bdeeg*` |
| Wrong-sample utilization | Student uses teacher-correct / teacher-wrong subsets | BNCI2014001-4, BNCI2015001 | teacher: MIRepNet/CBraMod; student: IFNet | mostly within split | Closed as null except ordinary correct-only KD-like signal | `scripts/wrongsample/run_wrong_sample.py`, `results/metrics/wrong_sample_*` |
| F+T initial feature fusion | Frozen LOSO features, K-shot support on test subject | BNCI2014001-4, BNCI2014004 | big: MIRepNet/CBraMod; small: IFNet/EEGNet/ADFCNN | train linear fusion head on K support trials; evaluate on rest | Initial "12/12 win" was a baseline artifact because it compared to `head_big` only | `results/metrics/ft_generality.csv` |
| Corrected balance-gated selection | Frozen LOSO features, K-shot support on test subject | BNCI2014001-4, BNCI2014004, BNCI2015001, AlexMI | big: MIRepNet/CBraMod; small: IFNet/EEGNet/ADFCNN | `cv_sel2` selects frozen big/small head by support CV; `cv_sel3` selects frozen big/small/fusion by support CV | Positive but scoped: +0.51 K20, +0.58 K30 over 24 cells | `scripts/fusion/run_balance_gate.py`, `results/metrics/balance_gate.csv` |
| A fine-tune baseline | True end-to-end fine-tune of single models on K support | recorded flagship: BNCI2014001-4, MIRepNet x IFNet | MIRepNet, IFNet | retrain LOSO base, deepcopy, then fine-tune full model on same K support | Recorded summary: best single FT beats frozen fusion by about +3.9 K20 / +4.8 K30 | `scripts/fusion/run_finetune_baseline.py`, `PROGRESS.md`; current CSV is incomplete |

## Protocol Definitions

### Within-subject 70/30

Implemented by `data.subject_split(dataset, subject, val_split=0.3, seed=...)`.

- First load the downstream subject pool via `data_mode='session3'`.
- Then do stratified `train_test_split(..., test_size=0.3, random_state=seed)`.
- Train/calibration = 70%, test = 30%.
- This is used by standard per-subject finetune/export and many early KD/ensemble diagnostics.

### LOSO

Implemented by `data.loso_split(dataset, test_subject)`.

- For fold `t`, test pool = downstream pool of subject `t`.
- Training pool = downstream pools of all other subjects.
- No split randomness at the fold level.
- Cached LOSO artifacts are written as `results/artifacts/<dataset>/<model>_loso/<subject>_<seed>_{train,test}.npz`.

The artifact files contain only:

- `logits`
- `feats`
- `y`

They do not store raw row IDs. Raw row IDs are reconstructed from the loader rules below.

### Few-shot on the LOSO test subject

Used by `run_ft_fusion.py`, `run_balance_gate.py`, and `run_finetune_baseline.py`.

- Start from the held-out subject's LOSO test pool.
- Draw K labeled support trials using `StratifiedShuffleSplit(train_size=K, random_state=seed)`.
- Evaluate on the remaining held-out subject trials.
- `run_balance_gate.py`: K = 20/30, seeds = 666/667/668, draws = 5.
- `run_ft_fusion.py`: K = 10/20/30, seeds = 666/667/668, draws = 5.
- `run_finetune_baseline.py`: defaults to K = 20/30, seeds = 666/667/668, draws = 3, but the recorded A run used a GPU-limited flagship setup.

Important: support/eval trial IDs are not contiguous ranges. They are random stratified indices inside the subject's test pool.

## Main Numeric Results

### D0 Diagnosis

From `results/d0/decision_report.md`:

- BNCI2014001-4 LOSO, MIRepNet x IFNet: big 48.37, small 41.31, oracle union 64.72, headroom +16.36.
- BNCI2014004 LOSO, MIRepNet x IFNet: big 76.77, small 74.35, oracle union 85.87, headroom +9.10.
- Several CBraMod x small LOSO cells also have large oracle headroom, but CBraMod can be weaker than the small model.

Interpretation: the two models make different mistakes, so oracle complementarity exists; the hard part is identifying, without oracle labels, which model is right on each sample.

### R1 Logit Router

Recorded in `PROGRESS.md`:

- Flagship: BNCI2014001-4, MIRepNet x IFNet, LOSO.
- big / small / avg ensemble = 48.37 / 41.31 / 48.59.
- Static cross-subject soft gate = 46.71, significantly worse than baseline.
- Same-subject 2-fold calibration = 50.10, only +0.40 over max(big, avg ensemble), non-significant.
- Oracle union = 64.72.

Interpretation: logits contain weak same-subject information, but not a stable cross-subject routing signal.

### Corrected Frozen Feature Selection

From `results/metrics/balance_gate.csv`, 4 datasets x 2 big models x 3 small models = 24 cells, K = 20/30:

| K | `cv_sel3 - cv_sel2` | Positive cells | `bal_gate - cv_sel2` | Positive cells |
|---|---:|---:|---:|---:|
| 20 | +0.51 | 20/24 | +0.59 | 19/24 |
| 30 | +0.58 | 21/24 | +0.71 | 19/24 |

Per-dataset mean `cv_sel3 - cv_sel2`:

| Dataset | Mean delta |
|---|---:|
| AlexMI | +0.60 |
| BNCI2014001-4 | +0.93 |
| BNCI2014004 | +0.31 |
| BNCI2015001 | +0.34 |

Meaning:

- `head_big`: linear head on frozen big-model features using K support labels.
- `head_small`: linear head on frozen small-model features using K support labels.
- `fusion`: concat linear head on frozen `[big_feat, small_feat]` using K support labels.
- `cv_sel2`: support-CV selects `head_big` or `head_small`.
- `cv_sel3`: support-CV selects `head_big`, `head_small`, or `fusion`.

This is the stable positive result, but only under frozen/cached-feature constraints.

### A End-to-end Fine-tune Baseline

From `PROGRESS.md` and memory:

- Flagship: BNCI2014001-4, MIRepNet x IFNet, LOSO.
- Procedure: train LOSO base on all non-held-out subjects, deepcopy base, then end-to-end fine-tune the whole single model on the same K support trials.
- Recorded run: 9 subjects x 2 seeds, `base_epochs=40`, `ft_epochs=30`.

| K | `ft_big` | `ft_small` | frozen `fusion` | frozen `head_big` | frozen `head_small` |
|---|---:|---:|---:|---:|---:|
| 20 | 51.4 | 52.9 | 50.0 | 43.3 | 45.7 |
| 30 | 53.7 | 54.2 | 51.9 | 44.7 | 47.7 |

Recorded interpretation:

- `best_ft = max(ft_big, ft_small)` beats frozen fusion by about +3.9 at K20 and +4.8 at K30.
- K30: 8/9 subjects, p = 0.008.
- `ft_big > frozen head_big`: 9/9 subjects, p = 0.0039.

Reproducibility caveat:

- Current `results/metrics/finetune_baseline.csv` contains only one row: BNCI2014001-4, MIRepNet x IFNet, subject 0, seed 666, K30, draw0.
- The full A summary is preserved in `PROGRESS.md` and `/home/lixinli/.claude/projects/-home-lixinli-BigSmallCollab/memory/ft-fusion-positive.md`, not in the current CSV.

## Dataset Pools Used by the Main LOSO/Few-shot Experiments

All row indices below are zero-based raw rows in `/data1/llx/<dataset>/X.npy` before loader filtering, unless noted otherwise. The artifact row order follows the selected pool order after filtering.

### BNCI2014001-4

Source: `/data1/llx/BNCI2014001/X.npy`, shape `(5184, 22, 1001)`.

- Task: 4-class MI: `feet`, `left_hand`, `right_hand`, `tongue`.
- Loader dataset name: `BNCI2014001-4`.
- Uses `session_E` only for downstream pool (`data_mode='session3'`).
- Truncates 1001 samples to 1000.
- 9 subjects, 288 selected trials per subject, 72 trials per class.
- Each selected session has 6 runs, 48 trials each, 12 per class.

For zero-index subject `s`:

- raw subject block = `[576*s, 576*s+575]`
- selected downstream/test pool = `[576*s+288, 576*s+575]`
- run `r` inside selected pool = `[576*s+288+48*r, 576*s+335+48*r]`

| 0-index subject | raw subject id | selected session | selected rows | n |
|---:|---:|---|---|---:|
| 0 | 1 | session_E | 288-575 | 288 |
| 1 | 2 | session_E | 864-1151 | 288 |
| 2 | 3 | session_E | 1440-1727 | 288 |
| 3 | 4 | session_E | 2016-2303 | 288 |
| 4 | 5 | session_E | 2592-2879 | 288 |
| 5 | 6 | session_E | 3168-3455 | 288 |
| 6 | 7 | session_E | 3744-4031 | 288 |
| 7 | 8 | session_E | 4320-4607 | 288 |
| 8 | 9 | session_E | 4896-5183 | 288 |

LOSO fold `t`: train all rows above except subject `t`; test rows = subject `t` selected rows.

### BNCI2014004

Source: `/data1/llx/BNCI2014004/X.npy`, shape `(6520, 3, 1126)`.

- Task: 2-class MI: `left_hand`, `right_hand`.
- Loader dataset name: `BNCI2014004`.
- Uses loader's `session3` range `p1:p2`; rows after `p2` are not used by this loader for downstream experiments.
- Truncates to 1000 samples.
- No local `meta.csv`; ranges below come from `data/eeg_dataset.py`.

| 0-index subject | phase1 rows not used in downstream few-shot | selected `session3` rows | leftover rows not used | selected n | class counts |
|---:|---|---|---|---:|---|
| 0 | 0-399 | 400-559 | 560-719 | 160 | 80/80 |
| 1 | 720-1119 | 1120-1239 | 1240-1399 | 120 | 60/60 |
| 2 | 1400-1799 | 1800-1959 | 1960-2119 | 160 | 80/80 |
| 3 | 2120-2539 | 2540-2699 | 2700-2859 | 160 | 80/80 |
| 4 | 2860-3279 | 3280-3439 | 3440-3599 | 160 | 80/80 |
| 5 | 3600-3999 | 4000-4159 | 4160-4319 | 160 | 80/80 |
| 6 | 4320-4719 | 4720-4879 | 4880-5039 | 160 | 80/80 |
| 7 | 5040-5479 | 5480-5639 | 5640-5799 | 160 | 80/80 |
| 8 | 5800-6199 | 6200-6359 | 6360-6519 | 160 | 80/80 |

LOSO fold `t`: train selected `session3` rows from all other subjects; test selected `session3` rows from subject `t`.

### BNCI2015001

Source: `/data1/llx/BNCI2015001/X.npy`, shape `(5600, 13, 2561)`.

- Task: 2-class MI: `feet`, `right_hand`.
- Loader dataset name: `BNCI2015001`.
- Uses `session_A` only by default (`MI2015001_SESSION` can override, but current config/default is `session_A`).
- Native 512 Hz, resampled to 250 Hz; cropped to a length divisible by 125, capped at 1000.
- 12 subjects, 200 selected trials per subject, 100 per class.

| 0-index subject | raw subject id | selected session | selected rows | n | class counts |
|---:|---:|---|---|---:|---|
| 0 | 1 | session_A | 0-199 | 200 | 100/100 |
| 1 | 2 | session_A | 400-599 | 200 | 100/100 |
| 2 | 3 | session_A | 800-999 | 200 | 100/100 |
| 3 | 4 | session_A | 1200-1399 | 200 | 100/100 |
| 4 | 5 | session_A | 1600-1799 | 200 | 100/100 |
| 5 | 6 | session_A | 2000-2199 | 200 | 100/100 |
| 6 | 7 | session_A | 2400-2599 | 200 | 100/100 |
| 7 | 8 | session_A | 2800-2999 | 200 | 100/100 |
| 8 | 9 | session_A | 3400-3599 | 200 | 100/100 |
| 9 | 10 | session_A | 4000-4199 | 200 | 100/100 |
| 10 | 11 | session_A | 4600-4799 | 200 | 100/100 |
| 11 | 12 | session_A | 5200-5399 | 200 | 100/100 |

Subjects 8-11 in the raw metadata also have `session_C`, but `session_C` is not used by the current loader defaults.

### AlexMI

Source: `/data1/llx/AlexMI/X.npy`, shape `(480, 16, 1537)`.

- Raw labels: `feet`, `right_hand`, `rest`.
- Task used here: 2-class `right_hand` vs `feet`; `rest` is dropped.
- Loader dataset name: `AlexMI`.
- Single metadata session `0`, run `0`.
- Native 512 Hz, resampled to 250 Hz: 1537 -> 750, then first 250 samples are tiled to reach 1000.
- 8 subjects, raw 60 trials each; after dropping `rest`, 40 selected trials per subject, 20 per class.

| 0-index subject | raw subject id | raw block | selected n after dropping rest | class counts |
|---:|---:|---|---:|---|
| 0 | 1 | 0-59 | 40 | 20/20 |
| 1 | 2 | 60-119 | 40 | 20/20 |
| 2 | 3 | 120-179 | 40 | 20/20 |
| 3 | 4 | 180-239 | 40 | 20/20 |
| 4 | 5 | 240-299 | 40 | 20/20 |
| 5 | 6 | 300-359 | 40 | 20/20 |
| 6 | 7 | 360-419 | 40 | 20/20 |
| 7 | 8 | 420-479 | 40 | 20/20 |

The final selected rows are not one contiguous range because `rest` trials are removed. Exact selected raw-index ranges:

| raw subject id | selected non-rest raw ranges |
|---:|---|
| 1 | 0-3, 5-11, 14, 19, 21-25, 28-29, 31-35, 38, 40-41, 43, 45-47, 49, 51-55, 58-59 |
| 2 | 61-63, 66-68, 70-71, 75-78, 80-85, 87-88, 90-91, 93-96, 98, 101-102, 104-105, 107-110, 112-113, 116-117, 119 |
| 3 | 121-123, 126-128, 130-131, 135-138, 140-145, 147-148, 150-151, 153-156, 158, 161-162, 164-165, 167-170, 172-173, 176-177, 179 |
| 4 | 181-183, 186-188, 190-191, 195-198, 200-205, 207-208, 210-211, 213-216, 218, 221-222, 224-225, 227-230, 232-233, 236-237, 239 |
| 5 | 240-245, 248-249, 251-252, 256-257, 261-262, 264-272, 274-275, 277, 279, 282-286, 288, 290-292, 294, 296-298 |
| 6 | 302, 304, 306-308, 310-316, 318-321, 323-324, 327-328, 332-334, 336-339, 341-344, 346-347, 349, 351, 353-355, 357-358 |
| 7 | 360-361, 364-366, 368, 370-373, 375-376, 379-380, 382-385, 387-388, 390-392, 395-403, 405-408, 410, 412, 414, 416 |
| 8 | 420-422, 424-425, 427-429, 431-432, 434, 436-437, 439-442, 446-447, 449, 451-452, 454-458, 460, 463-465, 467, 469, 471-472, 474, 476-479 |

## Reconstructing Exact K-shot Support/Eval Rows

Artifacts do not store support/eval IDs. To reconstruct one exact draw:

```python
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

# y is the artifact y for the held-out subject test split.
# raw_map maps artifact-local row -> raw X.npy row according to the tables above.
sss = StratifiedShuffleSplit(n_splits=5, train_size=K, random_state=seed)
splits = list(sss.split(np.zeros(len(y)), y))
support_local, eval_local = splits[draw]
support_raw = raw_map[support_local]
eval_raw = raw_map[eval_local]
```

For BNCI2014001-4, BNCI2014004, and BNCI2015001, `raw_map` is usually a contiguous selected row range. For AlexMI, `raw_map` is the non-rest selected-row list, not the 60-trial raw block.

## Model Configuration Used by These Experiments

From `configs/models/*.yaml`:

| Model | Role | Env | Default training config |
|---|---|---|---|
| MIRepNet | big/foundation | `mirepnet` | epochs 10, lr 0.001, batch 8, weight_decay 1e-6, emb_size 256, depth 6 |
| CBraMod native | big/foundation | `cbramod` | batch 16; per-dataset lr/epochs/dropout/weight_decay/band filled from adapter table |
| IFNet | small | `mirepnet` | epochs 100, lr 0.001, batch 16, weight_decay 0.01 |
| EEGNet | small | `mirepnet` | epochs 100, lr 0.001, batch 32, weight_decay 1e-4 |
| ADFCNN | small | `mirepnet` | epochs 100, lr 0.001, batch 32, weight_decay 1e-4 |

The A fine-tune baseline intentionally overrode the LOSO base epochs to `base_epochs=40` in the recorded flagship run; this differs from MIRepNet's default 10 epochs and is noted as a caveat in `PROGRESS.md`.
