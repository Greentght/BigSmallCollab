# 当前 004/5001 LOSO 蒸馏

This run repeats four student distillation conditions on the refreshed BNCI
2014004 and BNCI 2015001 source caches. Data and all generated targets,
checkpoints, logs, and results are stored under `/data1/llx`.

## Data and folds

- BNCI 2014004: `session_3`, 9 held-out-subject folds.
- BNCI 2015001: `session_A`, 12 held-out-subject folds.
- Each fold trains on every selected trial from the other subjects and tests
  once on all selected trials from the held-out subject.
- MIRepNet and CBraMod teacher targets use only source-training trials. No
  held-out labels or target-subject inputs are used for distillation.
- MIRepNet inherits the source-refreshed input cache's subject-wise EA. Its
  held-out subject representation therefore uses unlabeled target-subject
  covariance, matching the refreshed baseline's transductive LOSO regime.
- Each model retains its source-refreshed input representation. The shared
  trial UID, label, and subject ordering is checked across the teacher and
  student inputs before training.

## Models and conditions

Teachers are MIRepNet and CBraMod. Students are IFNet, EEGNet, and ADFCNN, for
six directed teacher-to-student pairs. Each fold and seed starts from a fresh
student initialization. Its optimizer, learning rate, batch size, weight
decay, and 100-epoch budget come from the matching refreshed-source student
baseline manifest. The baseline final weights are used only for the paired
metric comparison.

The four conditions run in this order:

| Condition | Loss after warmup | Warmup |
|---|---|---:|
| `logits_kd` | CE + 0.5 × KD | 0 epochs |
| `kd_feature` | CE + 0.5 × KD + 0.5 × feature cosine loss | 0 epochs |
| `warmup10_kd` | CE + 0.5 × KD | 10 CE-only epochs |
| `warmup10_kd_feature` | CE + 0.5 × KD + 0.5 × feature cosine loss | 10 CE-only epochs |

KD uses temperature 2 and `T² × KL(teacher || student)`. Teacher logits and
features are frozen. The feature projection is trained with the student and
is not part of the exported student model. All conditions use 100 epochs and
the same per-epoch cosine learning-rate schedule. Evaluation uses the final
epoch once, without outer-fold validation or checkpoint selection.

Seeds are 666, 667, and 668. Each condition contains 378 fold-seed cells:
21 folds × 6 teacher/student pairs × 3 seeds. The full rerun contains 1,512
training cells and 126 teacher target caches. Reports include Accuracy,
Balanced Accuracy, Kappa, macro F1, AUROC, and paired deltas against the
matching refreshed-source CE student baseline.

## Storage and execution

- Shared inputs: `/data1/llx/BNCI2014004/` and `/data1/llx/BNCI2015001/`.
- Project input caches and teacher targets:
  `/data1/llx/BigSmallcollab/cache/reproductions/`.
- Distillation outputs and dispatcher logs:
  `/data1/llx/BigSmallcollab/results/distill/loso_source_refresh_004_5001_v1/`.
- The dispatcher uses at most two workers and GPUs 1–9; GPU0 is prohibited.
  It validates refreshed inputs and baselines, exports source-only teacher
  targets, smoke-checks each teacher/student/dataset combination, then runs
  the four conditions serially with per-cell resume checkpoints.

The executable configuration is
`configs/experiments/loso_distillation.yaml`.

The teacher/student baseline is the current configuration in `configs/models`
and `configs/datasets`, documented in [loso_baseline.md](loso_baseline.md).
