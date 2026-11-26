# DML-DS + HFD: CBraMod ↔ EEGNet on BNCI2014004

Run only the new DML-DS+HFD condition. Existing DML-DS subject/seed results
are retained and reused; they are not retrained by this experiment. The
comparison covers all nine subjects and seeds 666, 667 and 668. Each subject
uses the project's stratified 30% calibration / 70% test split from the
verified `session_3` broadband source. Both models see the same complete
four-second trial, and final-epoch models are evaluated independently.

The baseline's model configurations, 20 epochs, batch 16, optimizers,
learning rates, weight decay, per-epoch cosine, gradient clipping, auxiliary
head CE, final-logit DML-KL and preprocessing remain unchanged. The extra
term is two stop-gradient cosine distances between CBraMod `proj_in` and
EEGNet `batchnorm1` features, using the same electrode/time alignment and
independent bias-free linear projections as the HFD experiment. HFD weight
is 0.1; epochs 1–5 use only DML-DS, epochs 6–10 ramp HFD linearly to 0.1,
and epochs 11–20 use its full weight. Projections have separate optimizer
parameter groups, leaving baseline model/head gradient clipping unchanged.

Every add-on run requires a matching completed baseline and validates its
source, pretrained weight, split UIDs, training parameters, code hashes and
held-out predictions. No held-out labels are used to choose an epoch or tune
HFD. A GPU preflight checks directional gradients and exact resume on
calibration trials only, before the controller queues 27 add-on runs. Resume
states are stored externally every five epochs and on a requested stop, then
removed after completion.

The companion dispatcher follows the existing shared-GPU rule: fewer than
three compute processes including launch reservations, and at least 2,048 MiB
free. It does not change or stop other jobs. The controller waits for both
baseline and add-on results, validates their paired test UIDs and predictions,
then writes Accuracy, balanced Accuracy and subject-paired gains to the real
checkout `results/dml_ds_hfd_fewshot_004_v1/`. Subject differences first
average the three seeds; 95% bootstrap intervals resample the nine subjects.
All training states and predictions stay under
`/data1/llx/BigSmallcollab/results/bidir/dml_ds_hfd_fewshot_004_v1/`.
