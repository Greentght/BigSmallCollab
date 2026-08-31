"""Formal experiment orchestration and active experiment drivers.

`experiments.run` is the preferred config-driven entry point: one YAML under
`configs/exp/` becomes one metrics CSV and optional eval report. Active line-
specific drivers live here too (`distill/`, `bigmodel/`, `bidir/`, `mask/`) so
`scripts/` can stay focused on tooling (`check/`, `export/`, `legacy/`).
(The D0-onward fusion/adapt line is archived under the `pre-consolidation` tag.)

Reusable algorithms stay in `collab/`; experiment code here should orchestrate
protocols, conditions, artifacts, and metrics rather than duplicate core methods.
"""
