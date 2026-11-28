# Few-shot experiments at a uniform 100-epoch budget

The earlier reports contained both 20-epoch and 100-epoch runs. Those numbers
must not be interpreted as a matched training-budget comparison. At the user's
request, every completed, reported 20-epoch few-shot condition is replayed with
100 epochs, including its supervised and deep-supervision controls.

The initial frozen manifest contains **5,451 new training units**. A unit is
one dataset, model pair or standalone model, condition, subject and seed; a
paired unit trains both models. Existing 100-epoch results are preserved:
there are 2,031 reuse references pointing to **1,977 distinct records**. The
difference comes from position-study aliases to an already completed CE100
control. These are not additional new runs.

| Historical family | New 100-epoch units |
| --- | ---: |
| DML-DS pairs and matched CE | 1,692 |
| DCM/DML/DML-DS/CE, 001 four-class pilot | 108 |
| DCM/DML-DS/CE, remaining datasets | 342 |
| Initial simplified HFD pilot | 27 |
| Cross-operator HFD pilot | 27 |
| Cross-operator HFD, all pairs/datasets | 819 |
| HFD position study and DS-only controls | 540 |
| MIRepNet 8–30 Hz corrected CE and DML-DS | 987 |
| MIRepNet 8–30 Hz corrected DS-only | 423 |
| SemCKD, uniform attention and AT | 162 |
| SemCKD/uniform, weight 50 | 108 |
| SemCKD/uniform, weight 400 | 108 |
| Terminal HFD, weights 0.1 and 1 | 108 |
| **Total** | **5,451** |

The main baseline tree includes 282 matched CBraMod–IFNet CE/DML-DS units
that were previously reused from DCM reports. This gives downstream HFD
readers a native 100-epoch baseline without changing historical reuse paths.
Position C2 continues to reference the ordinary shallow HFD run; it is not
counted again as a separate physical training unit.

## Training and identity

- The evaluation remains within-subject few-shot: 30% calibration, 70% test,
  seeds 666, 667 and 668. This queue does not start LOSO experiments.
- Models restart from their original initialization. No 20-epoch optimizer
  or trained checkpoint is extended. The cosine schedule has a fresh
  100-epoch horizon, and evaluation uses the original final-epoch rule.
- Auxiliary losses, weights, warm-up/ramp settings, preprocessing, trial
  splits and original optimizer recipes are preserved. This isolates the
  training-budget change within each experiment family.
- All BNCI source inputs originate from the verified `wide_fre_band` data.
  MIRepNet's corrected 8–30 Hz model-input branch and its older broadband
  teacher-input control are distinct protocols and are reported separately.
  Historical NeuroGPT normalization variants are also kept separate.
- The worker records source-result SHA256, exact config and numerical-code
  hashes, actual 100-epoch completion, trial UIDs and labels. It compares
  original initialization and preprocessing identities wherever those
  historical fields are present.
- Temporary optimizer/RNG checkpoints are written every five epochs instead
  of every epoch for older entries. This reduces disk I/O without changing
  training steps. Completed results remain readable from metrics and
  predictions without requiring model weights or resume checkpoints.
- Each worker stays on one GPU for the entire training unit. Admission
  requires fewer than three actual/reserved processes and at least 4 GiB
  free memory. The main queue has six workers, one CPU thread per process,
  and zero data-loader workers. Other running jobs are left untouched.

## Additional tuning branch

After the main replay, a separate 100-epoch tuning queue handles the earlier
MIRepNet/CBraMod tuning work. It replays every completed 20-epoch candidate,
reruns the original A/B candidate search at 100 epochs, selects using only
inner calibration validation, and refits the selected settings on the full
calibration split. The number of these tasks depends on validation selections
and is **additional to 5,451**. Existing NeuroGPT 100-epoch tuning is retained.

Old selected settings are also replayed at 100 epochs as fixed candidates;
they are not presented as settings selected by the new 100-epoch validation.
Both task specifications and completed epoch history are checked when reading
the new tuning results. Smoke/resume checks are excluded from reported scores.

## Execution, storage and reporting

```bash
/home/lixinli/anaconda3/envs/mirepnet/bin/python -B -m experiments.bidir.prepare_budget100
/home/lixinli/anaconda3/envs/mirepnet/bin/python -u -B -m experiments.bidir.queue_budget100
/home/lixinli/anaconda3/envs/mirepnet/bin/python -B -m experiments.bidir.summarize_budget100
```

The manifest is immutable once the controller exists. A changed manifest or
numerical implementation causes the controller to drain rather than mix
implementations. Restarting the same queue recovers verified completions and
resumes interrupted 100-epoch jobs from their own temporary states.

- Training artifacts: `/data1/llx/BigSmallcollab/results/bidir/joint_fewshot_100epoch_v1/`.
- Reports: `results/joint_fewshot_100epoch_v1/`, a real checkout directory.
- `manifest.json` lists exact jobs, source records and dependencies;
  `controller.json` records active GPUs, completion counts and errors.
- `100epoch_统一预算重跑进度与结果.xlsx` and matching CSVs contain only verified
  100-epoch results. Different method families, pairs and input profiles have
  separate sheets. Accuracy and balanced accuracy are labelled separately.
- Aggregate cells remain blank until all expected subjects and the exact
  three-seed set are present. Means first average subjects equally within
  each seed, then average the three seed means equally. No partial completion
  is presented as a final result.

The numerical fingerprint also captures two pre-existing working-tree changes
in `data/split.py` and `configs/models/mirepnet.yaml`. Their exact patch is
already preserved in `docs/dml_ds_cross_hfd_fewshot_all_baseline.patch`; apply
that patch when reproducing from a clean checkout. This replay does not alter
those unrelated LOSO settings. All original 20-epoch artifacts are preserved
as historical records and excluded from the new 100-epoch score tables.
