# Agent Instructions

## Data and model storage

- User requirement: never save datasets, raw EEG downloads, preprocessing caches,
  teacher targets, model weights, or training checkpoints inside this checkout.
- Shared datasets belong in `/data1/llx/<dataset-name>/` and must be reusable
  across projects. Preserve existing versions; put new source variants in
  clearly named subdirectories, such as
  `/data1/llx/BNCI2014001/broadband_0p1_75hz/` for all-session broadband NPYs.
- Project-specific inputs, caches, teacher targets, results and checkpoints
  belong under `/data1/llx/BigSmallcollab/`, with `cache`, `results`, `weights`,
  `git_lfs` and `migrations` subdirectories. Do not create another top-level
  `data_cache` or scattered `BigSmallCollab_*` directories.
- Existing datasets in `/data1/llx/BNCI*` and pretrained weights in
  `/data1/llx/pre_weight` remain valid inputs.
- Use `experiments.storage` to resolve historical artifact paths and enforce
  external output paths. Do not recreate checkout-local artifact directories
  or symlinks to them. Keep source code, configs and documentation in Git.

## Git publishing

- After completing a project change, commit it and push it to `origin/master`.
- Push each completed solution or experiment improvement promptly, rather than leaving it only in the local checkout.
- Before staging, review `git status` and stage only files related to the completed change. Leave generated results, caches, datasets, and unrelated work unstaged unless they are explicitly part of the requested change.
- Verify that the push succeeded. If a push is blocked or rejected, report which commit remains unpushed and why.
