"""Formal experiment orchestration and active experiment drivers.

`experiments.run` is the preferred config-driven entry point: one YAML under
`configs/exp/` becomes one metrics CSV and optional eval report. Active line-
specific drivers live here too (`distill/`, `fusion/`, `adapt/`, `bigmodel/`) so
`scripts/` can stay focused on tooling (`check/`, `export/`, `legacy/`).

Reusable algorithms stay in `collab/`; experiment code here should orchestrate
protocols, conditions, artifacts, and metrics rather than duplicate core methods.
"""
