"""Single-model finetune + artifact export (fewshot / LOSO).

These are the "train ONE model and cache its artifacts" entry points — the shared
prerequisite every collaboration experiment consumes. Each script runs in its
model's own conda env (see configs/models/<model>.yaml `env`) and writes
standardized ``(logits, feats, y)`` artifacts under ``results/artifacts/``.
"""
