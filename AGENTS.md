# Agent Instructions

## Data and model storage

- User requirement: never save datasets, raw EEG downloads, preprocessing caches,
  teacher targets, model weights, or training checkpoints inside this checkout.
- Store those files under `/data1/llx`. Defaults are
  `/data1/llx/data_cache`, `/data1/llx/BigSmallCollab_results`,
  `/data1/llx/BigSmallCollab_weights`, and `/data1/llx/BigSmallCollab_git_lfs`.
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
