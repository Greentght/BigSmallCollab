"""Reusable collaboration algorithms (big<->small distillation / mutual learning).

This package holds the *methods*, not the experiment orchestration. Each module is
self-contained and imports only the cross-env artifact interface (``artifacts``)
and the generic ``seed`` helper — never a concrete model — so a method can run in
whichever conda env owns the student model.

  - ``distill``        : offline KD + penultimate feature align (cached teacher).
  - ``bidirectional``  : sample-routed asymmetric bidirectional distillation.
  - ``mutual``         : CR-AMD — complementarity-routed asymmetric mutual distillation.
  - ``bdeeg``          : BD-EEG — difference-aware bidirectional distillation.
  - ``artifacts``      : standardized ``(logits, feats, y)`` .npz hub interface.
  - ``seed``           : unified full-stack seeding.

Experiment drivers that orchestrate protocols/conditions/artifacts live in
``experiments/`` and must not duplicate the core methods defined here.
"""
