# Native-logit DML-DS with terminal cross-operator HFD

The user selected: align terminal features, then keep bidirectional KL between
the two models' own original predictions. This experiment keeps the native
classification graph and deep-supervision labels and adds terminal HFD.
Feature alignment and prediction KL are optimized together in one backward
pass; there is no separate alignment-training stage or exchanged classifier.

## Audit of the preceding two adaptations

The executed SemCKD/AT objective for each model was:

`main_CE + 0.5 * mean(two_DS_CE) + 0.25 * T² * native_prediction_KL + ramp * weight * feature_loss`.

Temperature was 2. Both models used their own original classifier outputs for
KL and standalone evaluation. The feature term was additional supervision.

| Detail | SemCKD adaptation | AT adaptation |
| --- | --- | --- |
| CBraMod features | patch embedding; complete blocks 4 and 8 | Same |
| EEGNet features | batchnorm1; pooling2; pooling3 | Same |
| Layer matching | All 3×3 pairs; instance-dependent learned weights | Three fixed depth pairs |
| Signal | Within-batch instance Gram matrices, normalized two-layer query/key MLPs, softmax | Channel-mean squared activation energy, spatial L2 normalization |
| Feature adaptation | Separate 1×1/3×3/1×1 Conv–BN–ReLU regressor per layer pair | No channel regressor |
| Loss | Weighted per-example MSE, averaged over receiving layers and batch | Mean squared normalized-energy difference, summed over three layers |
| Loss weights | 0.1, 50, 400; uniform-association ablation at each strength | 1 |

CBraMod maps were `[B,200,C,4]`; EEGNet maps were
`[B,8,C,1001]`, `[B,16,1,250]`, `[B,16,1,31]`. Both rectangular axes were
average-pooled to their pairwise minimum lengths. Accordingly, temporal maps
were reduced to four bins, and the CBraMod electrode axis was averaged for
comparison to the EEGNet layers that had already mixed electrodes. This is an
explicit EEG adaptation and can discard spatial/temporal detail.

There are two concrete implementation details to scrutinize. With batch 16,
the EEGNet-receiving SemCKD regressors and query/key modules have 13,743,432
trainable parameters. Native EEGNet, excluding its separate DS heads, has
2,146 parameters for three electrodes and 2,450 for 22 electrodes. This is a
large training-only adaptation relative to the receiving model; it does not
by itself establish the cause of the negative result.

AT pools the feature maps **before** squaring them. Thus it uses energy of
temporally averaged activations, not temporally averaged activation energy:
`square(mean(F)) != mean(square(F))`. Signed oscillating EEG activations can
cancel in that pooling step. This is a specific adaptation choice that needs
to be evaluated before making claims about the original AT mechanism.

Other adaptations were online bidirectional detached targets in place of a
fixed teacher, 20-epoch/batch-16 few-shot training, and zero-padding missing
Gram columns for short SemCKD minibatches instead of dropping CE examples.
SemCKD retained its original nonlinear regressors, Gram attention and weighted
MSE; AT retained normalized squared-energy maps. The experiments do not claim
to reproduce the papers' complete fixed-teacher benchmark protocols.

At SemCKD weight 0.1, the first EEGNet convolution's calibration auxiliary/base
gradient-norm ratio was approximately 0.00086 (median over the logged first
batches of epochs 10–20). The 50/400 coefficients were additionally tested and
all settings retained. Negative results apply to these particular online EEG
adaptations and budgets; they do not establish a cause or a general failure of
the original methods.

## Terminal feature graph

| Model | HFD source | Frozen cross operator | Native HFD target |
| --- | --- | --- | --- |
| CBraMod | complete block 11 output `[B,C,4,200]` | complete block 12, including attention, residuals and FFN | complete block 12 output `[B,C,4,200]` |
| EEGNet | pooling2 `[B,16,1,250]` | native dropout → depthwise/pointwise convolutions → BN3 → ELU → pooling3 | pooling3 before the final dropout `[B,16,1,31]` |

For CBraMod learning from EEGNet, its block-11 feature is projected to the CNN
operator's native input, passed through a frozen-parameter copy of that
operator, and compared to EEGNet's detached native pooling3 output. In the
reverse direction EEGNet's pooling2 feature is projected to CBraMod's native
block-12 input, passed through that frozen full block, and compared to the
detached native block-12 output. The discrepancy is the mean `1 - cosine` along
the target feature-channel axis, as in the existing correct HFD implementation.

The paper originally applies cross-operator HFD near the input. This experiment
extends the operator to the complete last feature block so the target is
exactly the feature used by classification. It preserves projection → real
peer operator → native peer target, rather than using only a projected-vector
matching loss. The peer operator is frozen only in the auxiliary functional
call; its native model still trains through its original CE, DS and KL.

### Explicit EEG projection

Eight learned reconstruction slots per native one-second patch form a 32-knot
adapter grid at `(k + 0.5)/8` seconds. These are adapter coordinates, not native
transformer sample points or separately labelled windows.

- CBraMod → CNN input: per patch, linear `C*200 → 8*16`, unfold the eight slots,
  then interpolate to the native pooling2 coordinates `(4*k + 1)/250` seconds.
- CNN → CBraMod input: interpolate pooling2 to the 32 knots, group eight per
  native patch, then linear `8*16 → C*200` and restore `[B,C,4,200]`.

Interpolation uses the existing explicit deterministic dense interpolation
weights and clamps endpoints. For 004 `C=3`; for 001 `C=22`.
Both models still ingest and classify one full four-second trial at a time.

## Loss, conditions and matching

Each model optimizes:

`main_CE + 0.5 * mean(two_DS_CE) + 0.25 * T² * KL(peer_native || own_native) + ramp * w * terminal_HFD`.

`T=2`. The original KL stays active throughout all 20 epochs. Only HFD is zero
for epochs 1–5, ramps over epochs 6–10, and then stays at its fixed strength.
The two conditions are `w=0.1` and `w=1`. Both are reported, without test-based
weight, layer or checkpoint selection.

The scope is CBraMod–EEGNet on 004 and 001, nine subjects, seeds 666–668,
30% calibration / 70% test. This adds 108 paired training units and reuses 54
matched DML-DS results. Initializations, pretrained weights, source manifests,
physical trial UIDs, optimizer groups, 20-epoch/batch-16 budget and cosine
schedule must match the existing baseline. No test-fitted preprocessing occurs.

The exact baseline coefficients are CBraMod AdamW `lr=0.0003, wd=0.1`,
EEGNet AdamW `lr=0.001, wd=0.0001`, one CPU thread and zero loader workers.

## Validation and execution

Before formal work, each dataset/strength undergoes calibration-only GPU checks:

- applying each terminal operator to its own native input reproduces its native
  target; each target reproduces the original classifier output in eval mode;
- HFD backpropagates into the receiving source feature and its projector, with
  no peer-target gradients;
- frozen functional calls do not alter any native BatchNorm buffer;
- extra dropout uses independently seeded CUDA streams and leaves native
  forward RNG unchanged; both auxiliary RNG states are included in resumption;
- continuous and interrupted/resumed runs have exactly matching model, head,
  projector, auxiliary RNG hashes, histories and calibration predictions.

```bash
/home/lixinli/anaconda3/envs/mirepnet/bin/python -u -B -m experiments.bidir.queue_terminal_hfd_fewshot --workers 6
```

Admission requires fewer than three effective GPU processes and at least
4096 MB free. Each subject/seed training stays on its chosen GPU for all epochs.
Unrelated jobs are not modified. Training artifacts and temporary resume files
are external under `/data1/llx/BigSmallcollab/results/bidir/dml_ds_terminal_hfd_fewshot_v1/`.
Completed runs retain small predictions, metrics, training records and identity,
then remove only their own temporary resume checkpoint.

Reports are in the real checkout directory
`results/dml_ds_terminal_hfd_fewshot_v1/`. The workbook is
`DMLDS_末层HFD_详细对照.xlsx`; it separates both models and both metrics, includes
subject-level paired intervals and uses green/red fonts for positive/negative
differences. Incomplete 27-unit groups are left blank.

```bash
/home/lixinli/anaconda3/envs/mirepnet/bin/python -B -m experiments.bidir.summarize_terminal_hfd
```
