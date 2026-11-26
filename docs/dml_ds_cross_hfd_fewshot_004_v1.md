# DML-DS + cross-operator HFD on BNCI2014004

This experiment corrects the earlier `dml_ds_hfd_fewshot_004_v1` condition,
which used only pointwise projection and cosine alignment. It keeps that
result intact. The new condition follows both paths in Eq. (2) of Zhu et al.,
*A Good Student is Cooperative and Reliable*, ICCV 2023:

- CNN first-stage feature → dimension/position alignment → the peer ViT's
  next attention operation → cosine distance to that ViT's native attention
  output. This path trains EEGNet and its projection.
- ViT first-stage feature → dimension/position alignment → the peer CNN's
  next layer → cosine distance to that CNN's native next-layer output. This
  path trains CBraMod and its projection.

The native EEG models are not the paper's MobileNetV2 and MiT, so this is an
EEG adaptation of the full two-path HFD equations, not a byte-for-byte reuse
of the paper's segmentation models. The author's project page displayed no
source-code link when checked on 2026-11-26; the implementation is based on
the paper and its supplementary material. The peer transformation parameters
and peer target are detached for each direction, while the peer operation
remains differentiable with respect to its input. Additional BatchNorm
passes use cloned buffers to avoid mutating the peer's running statistics.

CBraMod's `patch_embedding` emits `[B,3,4,200]`; its native layer-0
pre-normalized spatial/temporal attention emits the same shape. EEGNet's
`batchnorm1` emits `[B,8,3,1001]`; its native `conv2`, `batchnorm2`, ELU and
`pooling2` emit `[B,16,1,250]`. EEGNet features are sampled at the 32
CBraMod native temporal coordinates and projected 8→25 before reshaping
into the 200-dimensional attention input. In the other direction a learned
200→64 projection produces eight channels at eight positions per patch;
linear interpolation places those features on EEGNet's temporal feature
axis. These coordinates adapt tensor shapes; each model still receives a
single complete four-second trial and inference uses its original head.

Only the new DML-DS+HFD condition is trained. The existing DML-DS baseline
is checked by source manifest, checkpoint hash, code hashes, subject/seed,
split UIDs, preprocessing, optimizer and predicted held-out trials. The
protocol remains subject-specific stratified 30% calibration / 70% test,
`session_3`, 9 subjects, seeds 666/667/668, 20 epochs and batch 16. HFD has
weight 0.1 per direction, starts after five epochs and ramps to full weight
over the following five epochs, matching the previous DML-DS+HFD comparison.
No test set labels are used for tuning or checkpoint selection.

One training/calibration minibatch per epoch records the norm and cosine of
the baseline and weighted HFD gradients for one shallow parameter in each
model. These are diagnostic minibatch observations, not a guarantee of
improvement or evidence about other layers. Preflight checks native-operator
equivalence, directional gradients and exact interrupt/resume behavior. The
supervisor runs one subject/seed at a time on one fixed GPU, checking fewer
than three visible processes and at least 2048 MiB free before each launch.

Artifacts are stored under
`/data1/llx/BigSmallcollab/results/bidir/dml_ds_cross_hfd_fewshot_004_v1/`;
small result reports belong in
`results/dml_ds_cross_hfd_fewshot_004_v1/`. Completed resume checkpoints are
removed. No results in this document are inferred before the runs complete.
