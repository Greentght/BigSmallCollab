# Agent Instructions

## Data and model storage

- User requirement: never save datasets, raw EEG downloads, preprocessing caches,
  teacher targets, model weights, or training checkpoints inside this checkout.
- Shared datasets belong in `/data1/llx/<dataset-name>/` and must be reusable
  across projects. Preserve existing versions; put new source variants in
  clearly named subdirectories, such as
  `/data1/llx/BNCI2014001/broadband_0p1_75hz/` for all-session broadband NPYs.
- Project-specific inputs, caches, teacher targets, training artifacts and checkpoints
  belong under `/data1/llx/BigSmallcollab/`, with `cache`, `results`, `weights`,
  `git_lfs` and `migrations` subdirectories. Do not create another top-level
  `data_cache` or scattered `BigSmallCollab_*` directories.
- Existing datasets in `/data1/llx/BNCI*` and pretrained weights in
  `/data1/llx/pre_weight` remain valid inputs.
- Keep one canonical copy of official pretrained weights in
  `/data1/llx/pre_weight/`. Do not duplicate them for each project or run.
- The user plans further distillation. Preserve the current refreshed-source
  001/001-4/004/5001 teacher checkpoints, teacher targets, student weights and
  feature projectors until their reuse or retirement is explicitly decided.
  A projector being unused for classification inference does not authorize
  deleting it when continued training or feature analysis may need it.
- Minimize retained training artifacts: do not keep every completed fold's
  student weights, feature projectors, or optimizer/RNG resume checkpoints
  permanently. Resume checkpoints are temporary during training. Fine-tuned
  teachers are temporary inputs when a planned KD run still needs its targets.
  Keep small metrics, predictions, configuration and provenance for reporting;
  retain model-input and teacher-target caches only for planned reuse.
- Before pruning existing trained-model files, make completed-result readers
  accept the retained metrics and provenance without requiring pruned model
  files; pruning must not silently trigger retraining. Inspect dependencies and
  ongoing processes before deleting existing artifacts.
- Completed 001 LOSO folds retain a small `completed_state.pt` with provenance
  and terminal RNG; their full `training_state.pt` is temporary and is removed
  after completion. Do not remove the small completion record or silently
  retrain a completed fold when a required identity record is missing.
- All physical model-input, Hub and teacher-target caches belong under the
  external `cache/` tree. Historical external path aliases may point there;
  never create checkout-local cache aliases. CodeBrain experiment artifacts
  and its project pretrained weight are retired by user authorization; retain
  its text result archive, not new copies of those models.
- User-facing result reports (Excel, CSV and JSON summaries) belong in the
  checkout's real `results/` directory, not `/data1/llx`. This is the user's
  latest storage requirement. Do not use a symlink to external storage for
  reports. Use `experiments.storage.REPORTS_ROOT` and `require_report_output`
  for report exports; generated reports remain unstaged by default.
- Use `experiments.storage` to resolve historical artifact paths and enforce
  external training output paths. Do not recreate checkout-local data or model
  artifact directories or symlinks to them. The report directory above is the
  explicit exception. Keep source code, configs and documentation in Git.

## Git publishing

- After completing a project change, commit it and push it to `origin/master`.
- Push each completed solution or experiment improvement promptly, rather than leaving it only in the local checkout.
- Before staging, review `git status` and stage only files related to the completed change. Leave generated results, caches, datasets, and unrelated work unstaged unless they are explicitly part of the requested change.
- Verify that the push succeeded. If a push is blocked or rejected, report which commit remains unpushed and why.
