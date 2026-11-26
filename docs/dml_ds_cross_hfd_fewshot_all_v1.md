# Few-shot DML-DS with bidirectional HFD

This run tests the fixed shallow HFD design from Zhu et al. on five EEG tasks. The reference condition is the existing DML-DS run for the exact dataset, two architectures, subject, calibration split and seed. Layer selection and frequency perturbation are separate research questions. The full grid gives paired results for one specified layer choice per architecture. [Zhu et al., ICCV 2023, Eq. 2 and supplementary discussion 4.2–4.5](https://arxiv.org/html/2307.12574v1)

| Architecture | HFD input | HFD comparison output after the native operator | DS classifier inputs |
| --- | --- | --- | --- |
| CBraMod | patch embedding | block 1 spatial/temporal attention branch before the residual sum | full blocks 4 and 8 |
| MIRepNet | embedding tokens | block 1 normalized attention branch before the residual sum | full blocks 2 and 4 |
| NeuroGPT | encoder patch tokens | encoder block 1 normalized attention branch | full encoder blocks 2 and 4 |
| EEGNet | batchnorm1 | spatial convolution, batchnorm2, ELU and pooling2 | pooling2 and pooling3 |
| IFNet | stem.sconv | native temporal convolutions and interFre fusion | stem.sconv and stem.interFre |
| ADFCNN | spectral_1 and spectral_2 | spatial_1 and spatial_2, within each parallel branch | spatial_1 and spatial_2 |

The learner projects its input feature, applies the peer's next operator with frozen parameters and compares the result with the peer’s detached native output using a channel-wise cosine distance. Each direction updates the learner's early backbone and its own projector. The time adapters use the models' native time grids. Matching electrodes individually requires an electrode axis on both sides. A learned projection handles models that have mixed electrodes. The two ADFCNN branch losses receive equal weight.

Each model has a main classifier and two auxiliary classifiers. The DML-DS objective uses main CE, 0.5 times the mean of the two auxiliary CEs and 0.25 times the main-head peer KL with temperature 2 and the usual squared temperature factor. Each HFD direction ramps from zero after five warm-up epochs to 0.1 at epoch ten. Both models predict the entire trial independently with their main classifier at the final epoch.

## Data and reporting

The three large models are paired with each of the three small models on BNCI2014004, BNCI2014001, BNCI2014001-4, BNCI2015001 and AlexMI. A stratified within-subject split assigns 30% to calibration and 70% to evaluation for seeds 666, 667 and 668. The four BNCI tasks use the verified broadband source. AlexMI uses the matched baseline's three native seconds, resampled from 512 to 250 Hz, followed by a repeat of the first second. MIRepNet fits EA on calibration trials and uses that matrix on the held-out trials. NeuroGPT’s encoder processes two chunks within each trial.

There are 1,269 paired units. The 27 completed BNCI2014004 CBraMod/EEGNet units use the previously validated implementation. The other 1,242 use the generalized HFD adapters. Preflight compares uninterrupted and resumed training on calibration trials and checks both directions' gradients for each new dataset/pair. A worker runs every epoch of a subject/seed on the same GPU. The dispatcher counts current CUDA processes and launch reservations and requires fewer than three per GPU at dispatch. It uses one CPU thread per worker.

Run the controller from the project root in the training environment:

```sh
python -B -m experiments.bidir.supervise_dml_ds_cross_hfd_all
```

The controller records status under `results/dml_ds_cross_hfd_fewshot_all_v1/controller.json`. The adjacent CSVs and workbook report completed groups. Subject means within each seed and then the three seed means receive equal weight. The confidence intervals resample subjects after averaging each subject’s three paired seed differences. Full training state is temporary under the external experiment result root, which retains each unit's predictions, split identities and model hashes for verification.

The separate calibration diagnostic runs with `python -B -m experiments.bidir.diagnose_hfd_layer_pairs`. For a single subject and seed it fits the five warm-up epochs within each of three inner folds, records four layer pairs’ trial-level linear CKA on the inner validation trials, and permutes the pairings within each class as a reference. Its reports go under `results/dml_ds_hfd_layer_pilot_004_v1/`.

The historical baselines record SHA-256 hashes of whole files, including two files with unrelated LOSO settings. For reproduction from a clean published checkout, apply the accompanying patch in a disposable checkout with `git apply docs/dml_ds_cross_hfd_fewshot_all_baseline.patch` if its two baseline files differ. The active experiment checkout already has the recorded versions.
