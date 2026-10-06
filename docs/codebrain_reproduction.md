# CodeBrain 复现与负结果记录

This document records the reproduction of the authors' public CodeBrain EEGSSM
encoder under the project's subject-wise few-shot protocol. The completed tuned
three-seed run is documented in
[`results/codebrain/codebrain_tuned_d2_cpu_20260929/final_report.md`](../results/codebrain/codebrain_tuned_d2_cpu_20260929/final_report.md).
As of 2026-09-29, CodeBrain is excluded from the active BigSmallCollab
collaboration baseline and teacher sets because its tuned result remained near
chance and below MIRepNet and CBraMod. The implementation and artifacts remain
available as a reproducibility record and negative result. This protocol differs
from the paper's SHU-MI evaluation.

## Fixed protocol

- Datasets: BNCI2014001, BNCI2014004, BNCI2015001, and AlexMI.
- Split: `data.subject_split(..., val_split=0.7, seed=..., return_uid=True)`;
  this means 30% train and 70% test in the project convention.
- Seeds: run all 38 subject cells with seed 666 first, then repeat with 667 and
  668. The seed 666 table is the first-round result; the three-seed table is the
  main summary.
- Initialization: strict load of the author's EEGSSM weights and a new
  three-linear-layer classifier for each dataset's native channel count.
  The paper's SHU-MI head emits one binary logit; this project maps the same
  classifier structure to two logits and uses cross-entropy to match the
  shared BigSmallCollab classifier interface.
- Training: all trainable encoder weights and the new head are fine-tuned with
  AdamW. The official fixed `backbone.patch_embedding.mask_encoding` sentinel
  remains non-trainable by design. Initial learning rate
  `5e-5`, weight decay `5e-3`, dropout `0.3`, batch size `64`, per-batch
  cosine decay to `1e-6`, and gradient norm clipping at `5`. Epoch budgets are
  20/20/20/50 in dataset order above, matching the existing CBraMod few-shot
  budget. The final epoch is fixed in advance. There is no validation set, and
  the test split is evaluated once after the final checkpoint is written.
- No KD, mutual-information loss, or input-mask augmentation is used.

The training recipe uses the CodeBrain downstream optimizer, batch size,
cosine schedule, and clipping rule. Its learning rate, weight decay, dropout,
and project epoch budgets are the pre-fixed baseline values recorded in the
resolved run config; they are not presented as tuned optima for these datasets.

## Input conversion

The project cache provides canonical raw trials with native channels and 1,000
samples at 250 Hz. CodeBrain input preserves the original channel names, count,
and order, applies deterministic polyphase resampling from 250 to 200 Hz,
divides by 100, and reshapes to `[N, C_native, 4, 200]`. It does not filter,
apply EA, pad channels, or fit normalization statistics. The cache unit is
recorded as microvolts in `PROGRESS.md:1135`; each run also records training
split amplitude quantiles and raw train/test array hashes. The `/100` conversion
matches the authors' SHU downstream preprocessing. BNCI2015001 is fixed to
`session_A`. For AlexMI, the existing canonical loader converts the source
3-second 512 Hz epochs to 750 samples at 250 Hz and repeats the first 250
samples to make the project's 1,000-sample trial; CodeBrain then resamples that
canonical trial to 800 samples. UID and window choices remain those of the
project's canonical loader.

## Sources and dependencies

- Code: [official CodeBrain repository](https://github.com/jingyingma01/CodeBrain),
  revision `22d350caf68246d2fda4f630ef837420db3fb130`.
- Weights: [official `CodeBrain.pth`](https://huggingface.co/YjMajy/CodeBrain/resolve/main/CodeBrain.pth),
  revision `bef08d2fdb1759685371cc635aad21ce59163689`, expected SHA256
  `d9714b8732c9883a04d022ee66254cd578ae1fa27f5458e6ab7f1aa96e9a7352`.
- The Apache-2.0 `Models/SSSM.py`, `Models/SGConv.py`, and `LICENSE` are kept
  under `models/codebrain/upstream/CodeBrain-main/`. Their original hashes are
  checked at import:
  - `Models/SSSM.py`: `e2fe5f7364907129507f3a9e946df9f9e44e10c3511efdcf772fab46d6fa278f`
  - `Models/SGConv.py`: `94f12f897ab4e8784a45d64fb65664184c872e6928159a6759fda2920c488be8`
  - `LICENSE`: `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`
  A runtime adapter patch moves the upstream local-attention mask to the
  residual block's device; the mask values and encoder computation remain the
  official implementation.
- The public state dict is loaded with `torch.load(weights_only=True)`,
  strips the official `module.` prefix, and is applied with strict key and shape
  checks. The resolved report records missing/unexpected keys and weight hash.
- The active model environment needs PyTorch, MNE, NumPy, SciPy, scikit-learn,
  einops, and `opt_einsum==3.4.0`. Install the missing package into an isolated
  workspace target rather than changing a shared environment:

  ```bash
  python -m pip install --target .deps/codebrain opt_einsum==3.4.0
  ```

  Then use `PYTHONPATH=.deps/codebrain:$PWD` on the commands below. Set
  `CODEBRAIN_WEIGHT` to the downloaded public checkpoint if it is not at
  `weights/codebrain.pth`; set `DATA_ROOT` if the project data cache is elsewhere.

## Run sequence

The completed tuned result used CPU only. The following commands are for an
intentional archival rerun: they hide CUDA, omit `--gpu`, and use a unique run
ID. Do not treat this as an active collaboration baseline run.

```bash
RUN_ID=codebrain_repro_cpu_$(date +%Y%m%d_%H%M%S)
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH=.deps/codebrain:$PWD
conda run -n cbramod python experiments/finetune/run_codebrain.py --action validate-config --run-id "$RUN_ID"
conda run -n cbramod python experiments/finetune/run_codebrain.py --action preflight --run-id "$RUN_ID"
conda run -n cbramod python experiments/finetune/run_codebrain.py --action smoke --run-id "$RUN_ID"
conda run -n cbramod python experiments/finetune/run_codebrain.py --action check-splits --run-id "$RUN_ID" --seeds 666
conda run -n cbramod python experiments/finetune/run_codebrain.py --action train --run-id "$RUN_ID" --seeds 666
conda run -n cbramod python experiments/finetune/run_codebrain.py --action train --run-id "$RUN_ID" --seeds 667
conda run -n cbramod python experiments/finetune/run_codebrain.py --action train --run-id "$RUN_ID" --seeds 668
```

`preflight` loads one train and one test batch for every dataset and checks
batch sizes one and two. `smoke` performs one short train-only pass per dataset;
its outputs are not formal artifacts. Training aborts if any MIRepNet or CBraMod
train/test artifact lacks matching UID, labels, or `split_policy`.

For the same-architecture random-init control, use `--init random --seeds 666`
after the pretrained runs are stable. It writes under `codebrain_random` and
uses the same head, splits, budget, and optimizer.

## Artifacts

Detailed runs are written below `results/codebrain/train/<init>/<dataset>/S<n>/<seed>/`:
resolved config, split manifest, weight-load report, per-epoch train history,
final checkpoint, train/test predictions, and metrics. Hub-compatible train and
test arrays are written to `results/artifacts/<dataset>/codebrain/` (or
`codebrain_random/`) with logits, features, labels, sample UIDs, and split policy.
The paired summary script consumes these artifacts alongside MIRepNet and
CBraMod. After the desired seeds finish, run
`python eval/codebrain_compare.py --seeds 666 667 668` to write paired summaries
and alignment checks under `results/codebrain/comparison/`; pass `--seeds 666`
for the first-round table. Metrics include accuracy, balanced accuracy, Kappa, per-class
prediction counts, and a single-class prediction-collapse flag.

The disclosed CodeBrain pretraining source is TUEG. The authors do not publish
a per-record training manifest; reports should therefore say “no overlap found
in the disclosed pretraining source,” not claim record-level non-overlap.
