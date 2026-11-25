# Shallow HFD on BNCI2014004

This protocol compares MIRepNet–IFNet and CBraMod–EEGNet on the broadband
`session_3` source. Each example remains one complete four-second trial and
each model receives one classification loss per example. The pair shares
the same trial UID and label; loaders reject incompatible source manifests,
sessions, subjects, labels, or cached inputs. MIRepNet's cached input uses
per-subject unsupervised EA; CBraMod and EEGNet use their non-EA inputs.
Compare methods within each pair rather than interpreting a difference between
the two pairs as an HFD effect.

## Training and feature alignment

For every pair, subject and seed, the four conditions are CE, CE+HFD,
CE+DML-KL, and CE+DML-KL+HFD. Seeds are 666, 667 and 668; the held-out
test subject and the next subject modulo nine as validation leave seven
training subjects. The fixed model settings, loss weights and paths are in
`configs/experiments/loso/hfd_loso_004.yaml`. Each condition trains for 100
epochs, with the first 10 epochs using CE alone and a five-epoch linear ramp
for its enabled auxiliary terms. Each model independently selects its best
validation-accuracy epoch. The formal evaluator runs each selected model
separately on the held-out subject; no projector or partner is used at inference.

HFD captures features in each model's original training forward:

* MIRepNet's `embedding.pool` yields `[B,61,128]`; IFNet's `stem.interFre`
  is average pooled with kernel 99 and stride 15 to `[B,61,64]`. Both
  positions correspond to samples `15*k+49` at 250 Hz.
* CBraMod's `patch_embedding.proj_in` is reshaped to `[B,3,4,8,25]`.
  EEGNet's `batchnorm1` feature map is interpolated at
  `250*(p+q/8)+0.5` to `[B,3,4,8,8]`, preserving electrode identity. The
  four native CBraMod patches are feature coordinates; they are not separate
  classification examples.

Each direction uses its own bias-free linear projector and mean cosine distance
with a detached target. The receiver and its projector receive gradients; the
target does not receive gradients from that direction. The projector initializer
does not advance the model's CPU or CUDA random streams. All workers use one
CPU thread and zero data-loader workers.

## Execution and results

Run `python -B -m experiments.bidir.queue_hfd_loso` under the project's
training Python environment while the shared GPU queue manager is running.
The supervisor submits two GPU preflights, each running two small batches per
epoch for two epochs, with DML and HFD active immediately. It verifies
directional gradients, separate model predictions, and exact agreement between
interrupted/resumed and uninterrupted training on the validation subject.
Only after both preflights pass does it submit the 216 formal paired runs.
The supervisor preserves the existing queue's GPU limits; it does not displace
other jobs. It can be restarted with the same command and frozen configuration.

Queue and supervisor status are in `results/hfd_loso_004_v1/queue_controller.json`
and the shared `results/loso/distill/queue/`. Model training state and internal
predictions are under `/data1/llx/BigSmallcollab/results/bidir/hfd_loso_004_v1/`.
Resume checkpoints are removed after a completed result has been committed;
metrics, predictions, provenance and training histories remain. A failed
preflight blocks formal submission; inspect that job's queue logs before retry.

Once all 216 runs complete, the supervisor validates their identities and
predictions and writes fold results, accuracy and balanced accuracy summaries,
and paired subject differences under the real checkout `results/hfd_loso_004_v1/`.
For each contrast, it first averages each subject's differences over the three
seeds, then bootstraps the nine subjects for a 95% interval. These intervals
describe uncertainty from this small subject sample, not independent trials.
