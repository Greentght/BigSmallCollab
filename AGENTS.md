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

## Result-summary Excel format

- Use `docs/loso_distillation_accuracy_3seed_colored.xlsx` as the visual and structural reference when creating comparable result-summary workbooks. Adapt sheet names, model groups, metric columns, and notes to the actual experiment; never copy values or claim an aggregation that the source artifacts do not support.
- For a model-comparison sheet, use a consistent table with `student` and `method` as the first two columns, followed by one column per dataset/task/setting. Put the teacher reference first, then group student models together. Within each student group, list its `none` supervised CE baseline first, followed by the compared methods. Keep model and method names stable across sheets.
- Store percentage metrics on a 0–100 scale and display two decimal places. State the metric explicitly (for example, Accuracy (%) or Balanced Accuracy (%)); do not label one metric as another. Include concise notes for the aggregation, seeds/folds, baseline definitions, and method details. For the referenced three-seed LOSO summary, the stated aggregation is an equal-weight mean over LOSO subjects within each seed, then the mean over seeds 666, 667, and 668. For other experiments, report their actual aggregation and seed/fold set instead of assuming these values.
- Match each method result to the teacher and the `none` baseline of that same student architecture, for the same dataset/task, metric, and evaluation protocol. Color method-result cells using strict `>` comparisons against those two matched reference scores:
  - Red fill `#FFC7CE`: the method score is greater than both the teacher score and the matched small-model `none` score.
  - Green fill `#C6EFCE`: the method score is greater than the teacher score but is not greater than the matched small-model `none` score.
  - Yellow fill `#FFF2CC`: the method score is greater than the matched small-model `none` score but is not greater than the teacher score.
  - No comparison fill: the method score is greater than neither reference, or a valid matched comparison is unavailable. Equality does not count as greater.
- Include a visible legend for the three comparison colors, using the meanings above (in the reference workbook: “大于小模型” = yellow, “大于大模型” = green, “大于二者” = red). Apply comparison fills to method result cells; use a separate, consistent fill to identify the teacher row (reference orange `#FAC090`) and student `none` baseline rows (reference light blue `#93CDDD`).
- Preserve the reference's readable layout: dark navy `#1F4E78` header with bold white text, centered table values, hidden gridlines, a blank spacer before the legend/notes when useful, and left-aligned wrapped footnotes below the table. Adjust merged note ranges and column widths to fit the actual number and length of columns.
- Save user-facing summary workbooks under the checkout's real `results/` directory, following the existing storage policy. Do not stage generated result workbooks unless the user explicitly requests that the workbook itself be versioned.
