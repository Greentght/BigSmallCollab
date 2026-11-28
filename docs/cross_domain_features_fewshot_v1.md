# SemCKD and Attention Transfer on the DML-DS few-shot protocol

This pilot tests two cross-domain feature-transfer mechanisms after the fixed
HFD study. It covers CBraMod–EEGNet on BNCI2014004 and BNCI2014001. There are
three new conditions, nine subjects per dataset and seeds 666, 667 and 668:
162 paired-model training units. The 54 matched DML-DS units are reused.

## Scientific comparisons

| Condition | Added objective per receiver | Fixed weight |
| --- | --- | --- |
| DML-DS | Existing main CE + 0.5 mean auxiliary CE + 0.25 T² prediction KL | Reused baseline |
| DML-DS + SemCKD | Learned instance-dependent cross-layer association and projected feature MSE | 0.1 |
| DML-DS + SemCKD uniform | Identical regressors, uniform 1/3 target-layer weights | 0.1 |
| DML-DS + AT | Sum of three normalized activation-energy map MSEs | 1.0 |

The uniform condition isolates the learned-association component from the
benefit of adding multiple feature regression paths. The two SemCKD conditions
have identical initial regressors and model RNG streams. Coefficients are fixed
pilot EEG settings, not a claim to reproduce the papers' CIFAR training recipes.
An unsuccessful pilot at these settings does not establish that a mechanism
cannot work at another regularization strength.

## Paper mechanics and EEG adaptations

SemCKD follows the authors' `models/util.py` and `distiller_zoo/SemCKD.py`:
raw within-batch instance Gram matrices, two-layer ReLU query/key MLPs,
L2-normalized embeddings, softmax over target layers, and a separate nonlinear
1×1/3×3/1×1 Conv–BN–ReLU regressor for every layer pair. Per-example feature
MSEs are weighted and summed, then divided by batch size and receiver layer
count. Query/key output size is batch_size/4, as in the original code.

AT follows the authors' `utils.at` and `utils.at_loss`: squared activation
energy averaged over feature channels, flattened spatial L2 normalization,
squared mean discrepancy, and a sum over the three matched depth levels.
It does not require learnable channel regressors.

Both papers originally use a fixed pretrained teacher. Here both native models
continue their existing DML-DS training, with **two independent directional
feature losses and detached peer targets**. This is an online bidirectional EEG
adaptation, not a reproduction of the original fixed-teacher benchmarks.

Features come from the same single full-trial native forward:

| Model | Features | Map layout |
| --- | --- | --- |
| CBraMod | patch embedding; complete encoder blocks 4 and 8 | B × 200 × electrodes × 4 native patches |
| EEGNet | batchnorm1; pooling2; pooling3 | B × (8/16/16) × (electrodes/1/1) × (1001/250/31) |

Rectangular maps use adaptive average pooling to the smaller length along
each axis. A collapsed electrode axis corresponds to averaging the other
model's electrode responses; it is not treated as a physical single electrode.
SemCKD uses all nine layer pairs. AT uses the three predeclared depth pairs.
No train/test trial is split into separate 1-second classification examples.
The four CBraMod patch positions are native feature coordinates.

The original SemCKD code skips incomplete minibatches. This pilot retains all
CE examples: it zero-pads only the missing Gram **columns** for the fixed-width
MLP, and computes loss only on real examples. Full batches exactly use the
original Gram/MLP computation. This adaptation is important for the small
calibration sets.

## Data, matching and training

Verified broadband sources are the canonical `wide_fre_band` sources. Each
subject uses the existing stratified 30% calibration / 70% test UID split.
The CBraMod adapter's existing fixed preprocessing is retained; EA is disabled.
No test-set fit, test selection, epoch selection or per-pair layer search occurs.
The fixed final epoch is evaluated once with each original standalone model.

Training matches the CBraMod–EEGNet DML-DS baseline: 20 epochs, batch 16,
CBraMod AdamW lr 0.0003 / wd 0.1, EEGNet AdamW lr 0.001 / wd 0.0001,
epoch-wise cosine scheduling, temperature 2. Existing DS/CE definitions and
model-group gradient clipping are preserved. The new feature loss is disabled
for five epochs, ramps linearly over epochs 6–10, then stays at its fixed weight.
There is one CPU thread per process and zero loader workers.

Source manifests, train/test UID digests, pretrained weights, complete baseline
implementation hashes and optimizer settings must match before training.
Historical baseline hashes also cover files irrelevant to this particular pair;
an identity mismatch must be audited, not ignored. The existing
`docs/dml_ds_cross_hfd_fewshot_all_baseline.patch` records the historical runtime
differences required to recreate that validated baseline checkout.

## Execution and reporting

Use the CUDA environment rather than the system CPU-only Python:

```bash
/home/lixinli/anaconda3/envs/mirepnet/bin/python -u -B -m experiments.bidir.queue_cross_domain_features_fewshot --workers 6
```

The persistent controller first performs calibration-only GPU smoke tests,
directional-gradient isolation, and exact continuous-versus-resumed training
checks for each dataset/method. After a group passes, its formal runs start.
Each subject/seed training stays on one GPU throughout all epochs. Admission
requires fewer than three effective processes and at least 4096 MB free. The
controller respects the shared queue's lock and does not modify unrelated jobs.

Training artifacts and temporary resume states live under
`/data1/llx/BigSmallcollab/results/bidir/dml_ds_cross_domain_features_fewshot_v1`.
Every in-progress resume includes models, heads, regressors/query/key modules,
optimizer states, minibatch generator and Python/NumPy/CPU/CUDA RNG states.
Completed results retain predictions, metrics, configuration identity,
training losses, per-epoch layer associations and gradient diagnostics. Their
own temporary resume states are removed only after those records are complete.

Reports live in the real checkout directory
`results/dml_ds_cross_domain_features_fewshot_v1/`. The controller exports them
periodically and upon completion. Manual export:

```bash
/home/lixinli/anaconda3/envs/mirepnet/bin/python -B -m experiments.bidir.summarize_cross_domain_features
```

The workbook separates Accuracy (%) and Balanced Accuracy (%), includes both
models, displays aggregate cells only for complete 27-unit groups, and colors
paired differences green for positive and red for negative. Main comparison
fills follow the repository's teacher/CE reference legend. Each paired
interval uses seed-averaged within-subject differences with a subject-level
two-sided t interval. Subjects are the independent units, not the 27 seed runs.
The final-epoch association table is a calibration-example-weighted training
summary, not causal evidence about a layer's importance.

Sources:

- [Cross-Layer Distillation with Semantic Calibration](https://arxiv.org/abs/2012.03236)
- [Authors' SemCKD implementation](https://github.com/DefangChen/SemCKD)
- [Paying More Attention to Attention](https://arxiv.org/abs/1612.03928)
- [Authors' Attention Transfer implementation](https://github.com/szagoruyko/attention-transfer)

## Supplement: published SemCKD loss strengths

The complete initial 162-unit pilot found identical EEGNet predictions under
SemCKD weight 0.1, its uniform control, and DML-DS. Calibration diagnostics show
that the weighted feature/base gradient-norm ratio on EEGNet's first temporal
convolution has median 0.000857 over the first batches of epochs 10–20.
This does not establish failure of SemCKD at a meaningful strength.

Two additional prespecified strengths, 50 and 400, come from the authors'
ImageNet and CIFAR commands respectively. Both receive the same uniform-control
ablation. Their four conditions add 216 paired units. No other training,
feature, split, ramp, inference or evaluation setting changes. All three
SemCKD strengths are retained and reported; the test set does not select one.
These are published coefficients inside our online EEG adaptation, not claims
to reproduce the complete original fixed-teacher training recipes.

```bash
/home/lixinli/anaconda3/envs/mirepnet/bin/python -u -B -m experiments.bidir.queue_semckd_strengths_fewshot --workers 6
```

The two queues run sequentially, each with six fixed-GPU workers. Both use the
same per-group calibration-only gradient and exact-resume checks. The final
combined report is `results/semckd_strengths_fewshot_v1/SemCKD_AT_全部权重对照.xlsx`.
The combined workbook distinguishes BA and Accuracy and shows a separate
positive/negative colored delta beside each dataset's score. It includes all
378 new paired training units and the matched historical references.
