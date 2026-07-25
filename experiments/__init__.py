"""Config-driven experiment orchestration.

One declarative YAML = one experiment, run by ``python -m experiments.run
configs/exp/<name>.yaml`` — replacing the pile of near-duplicate ``scripts/*.sh``
+ ``run_*.py`` drivers. The runner walks (dataset x unit x seed x condition),
consumes cached teacher artifacts, trains the student via ``collab.distill``, and
writes an eval-consumable long-form metrics CSV (then optionally prints the
``eval`` paired-stats report).

Pieces:
  - ``protocols`` — cell generators (within-subject / LOSO) over the canonical data.
  - ``methods``   — condition spec -> ``distill_student`` kwargs (the collab registry).
  - ``run``       — the driver tying config -> cells -> conditions -> CSV -> report.
"""
