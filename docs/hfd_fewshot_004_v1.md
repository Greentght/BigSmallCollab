# HFD subject-wise few-shot on BNCI2014004

This experiment replaces the stopped HFD LOSO run. Each subject is an
independent within-subject experiment: stratify that subject's `session_3`
trials by class using seed 666, 667 or 668; use 30% for labeled calibration
and the remaining 70% only for final test. No other subject's trials enter
training. Subject 2 has 36 calibration and 84 test trials; each other subject
has 48 and 112. The split is the project's existing
`fewshot_stratified_random_30_70` policy, not fixed K-shot sampling.

The original HFD feature coordinates, full four-second inputs, two model pairs,
four conditions, model hyperparameters, 100-epoch budget, loss weights and
10-epoch CE warmup plus five-epoch ramp remain fixed. Conditions are CE,
CE+HFD, DML-KL, and DML-KL+HFD for MIRepNet–IFNet and CBraMod–EEGNet. Each
trial contributes one CE per model; HFD uses aligned shallow features of the
same trial. The pair is trained on one GPU and each model is evaluated alone.

To match the project's 30/70 few-shot protocol, there is no separate validation
split: each model is evaluated at its final epoch. The 70% test labels do not
select epochs or tune weights. The MIRepNet cache retains the previously agreed
per-subject EA fitted using the whole subject's **unlabeled** EEG, including
test signals. Therefore the MIRepNet pair is transductive with respect to its
input preprocessing; it is not a calibration-only EA protocol. CBraMod and
EEGNet use non-EA caches. Comparisons of HFD effects are within each pair and
use identical calibration/test trial UIDs and initial model states.

The two GPU preflights run the full DML+HFD training path for two epochs,
evaluate calibration trials only, and verify gradient isolation and exact
agreement between resumed and uninterrupted runs. After both pass,
`experiments.bidir.queue_hfd_fewshot` submits the 216 subject/seed/condition/
pair runs. `experiments.bidir.dispatch_hfd_shared` uses the existing queue lock
and honors the user's shared-GPU policy: fewer than three GPU compute processes
(including reserved starting jobs) and at least 2,048 MiB free. Other queue
jobs and settings are unchanged.

Model inputs are read from manifest-validated broadband-source caches. Training
states, prediction artifacts and provenance stay under
`/data1/llx/BigSmallcollab/results/bidir/hfd_fewshot_004_v1/`; completed full
resume checkpoints are removed. The controller and reports are in the real
checkout's `results/hfd_fewshot_004_v1/`. Reports include Accuracy, balanced
accuracy and paired HFD gains; gains are averaged across seeds within each
subject before a 95% subject-bootstrap interval is computed.

Resume state is saved every five epochs and on a requested stop, reducing disk
I/O for the short few-shot epochs. An unexpected interruption can replay at
most four epochs from the last saved state.
