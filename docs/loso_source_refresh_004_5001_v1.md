# BNCI2014004 / BNCI2015001 source refresh and LOSO rerun

This run rebuilds reusable, all-session MOABB NPY sources for BNCI2014004 and
BNCI2015001 while preserving the existing flat NPY files. The source variants
are stored with the shared datasets. Model-specific inputs, checkpoints,
predictions, logs, and summaries are stored under
`/data1/llx/BigSmallcollab/`.

The source exporter applies the EEG-FM-Benchmark source frequency ranges:

| Dataset | Source filter | Native rate | All-session trials | LOSO session |
|---|---:|---:|---:|---|
| BNCI2014004 | 0–120 Hz | 250 Hz | 6,520 | `session_3` / MOABB `3test` |
| BNCI2015001 | 0.1–75 Hz | 512 Hz | 5,600 | `session_A` / MOABB `0A` |

Each exported trial is mapped one-to-one to the old NPY row by subject,
session index, run, within-run event ordinal, and class. The mapping and raw
file hashes are retained in each source variant's manifest. The task trial
order follows the old NPY row order so the old and refreshed source results can
be compared subject by subject with the same seeds.

The LOSO run uses the existing project model recipes and seeds 666, 667, 668.
MIRepNet uses 8–30 Hz filtering followed by per-subject EA and 45-channel
interpolation. CBraMod uses the existing full fine-tuning recipe and its
0.3–75 Hz, 60 Hz notch input path. IFNet uses its internal filterbank; EEGNet
and ADFCNN receive 8–32 Hz input. BNCI2015001 is resampled to 250 Hz and cropped
to the first 1,000 samples before model-specific preprocessing.

The run has 315 fold/seed cells across five models and two datasets. GPU 0 is
excluded. The dispatcher waits for an eligible GPU before starting or retrying
workers, and each worker writes a checkpoint after every epoch so it can resume
after interruption.

Execution configuration: `configs/reproductions/loso_source_refresh_004_5001_v1.yaml`.
