# BNCI2014001 broadband source: complete model LOSO set

The shared source at `/data1/llx/BNCI2014001/broadband_0p1_75hz/` retains both
native sessions. LOSO continues to use only `0train`, the same trials formerly
called `session_T`. The 001 binary and 001-4 four-class tasks reuse the exact
trial UID order and labels already used by the completed wideband MIRepNet and
CBraMod runs.

MIRepNet and CBraMod already have 27 fold/seed results per task (seeds 0, 1, 2).
This supplement trains the remaining small baselines: IFNet, EEGNet and ADFCNN,
for 9 subjects and 3 seeds on both tasks (162 additional fold/seed cells).

| Model | Source processing | Training recipe |
|---|---|---|
| IFNet | Adapter filter bank 4–16 Hz and 16–40 Hz applied directly to the broadband epochs | Random initialization, AdamW, LR 1e-3, batch 16, WD 0.01, 100 epochs |
| EEGNet | Epoch-level order-4 zero-phase Butterworth 8–32 Hz | Random initialization, AdamW, LR 1e-3, batch 32, WD 1e-4, 100 epochs |
| ADFCNN | Epoch-level order-4 zero-phase Butterworth 8–32 Hz | Random initialization, AdamW, LR 1e-3, batch 32, WD 1e-4, 100 epochs |

The explicit 8–32 Hz step for EEGNet and ADFCNN preserves the effective band
assumed by the earlier MOABB-generated NPY. IFNet instead uses its configured
filter bank on the broadband source, so its 4–16 and 16–40 Hz bands are not
truncated by the old 8–32 Hz cache. All models use the same subject folds,
trial UIDs, label mapping, seeds, 4-second window, no validation split, and
final-epoch evaluation. GPU0 is excluded.

Before training, a matching trial (source row 2) was checked between the old
NPY and the new broadband NPY after applying the explicit 8–32 Hz filter. Their
RMS values were 4.3991 and 4.4203 respectively (ratio 1.0048), so the updated
µV data does not introduce an amplitude-unit jump for EEGNet or ADFCNN.

Runner: `experiments/finetune/run_loso_small_wideband_001.py`.
Dispatcher: `experiments/finetune/dispatch_loso_small_wideband_001.py`.
Results and checkpoints are written outside the checkout under
`/data1/llx/BigSmallcollab/results/reproductions/loso_config_alignment_v2/wideband_npy_v3/`.
