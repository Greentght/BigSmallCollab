"""Formal experiment orchestration and active experiment drivers.

Single-model artifact export lives in ``experiments/finetune``. Unified
big-to-small distillation lives in ``experiments/distill/run_distill.py``. Other
line-specific drivers live in their own subdirectories.

Reusable algorithms stay in ``collab/``; experiment code here should orchestrate
protocols, conditions, artifacts, and metrics rather than duplicate core methods.
"""
