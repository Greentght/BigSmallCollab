"""Collab-method registry: a condition spec -> ``collab.distill.distill_student``
kwargs. This is the pluggable seam that keeps new methods out of a growing
``if/elif`` — a YAML condition names a ``method`` (or gives raw kwargs) and the
runner routes it here.

A condition in YAML is a dict merged over the experiment's ``distill`` defaults,
e.g.::

    conditions:
      base:      {method: baseline}
      KD:        {method: kd}
      KD_masked: {method: kd, masked: true}
      Combo:     {method: combo}
      Proto:     {method: proto}

``method`` picks a base kwarg template below; any other keys (lam_kd, lam_feat,
temperature, masked, ...) override it. The ``masked`` sugar flag is resolved by
the runner into ``sample_weight`` (the teacher-correct mask) since it needs the
per-cell teacher logits.
"""

# method name -> base distill_student kwargs (before per-condition overrides)
REGISTRY = {
    # plain student, no teacher signal (baseline)
    'baseline': dict(lam_kd=0.0, lam_feat=0.0, teacher_correct_only=False),
    # vanilla logit KD
    'kd':       dict(lam_kd=0.5, lam_feat=0.0, teacher_correct_only=False),
    # penultimate feature alignment only (per-sample cosine)
    'feat':     dict(lam_kd=0.0, lam_feat=0.5, teacher_correct_only=False),
    # KD + feature align
    'combo':    dict(lam_kd=0.5, lam_feat=0.5, teacher_correct_only=False),
    # class-prototype feature alignment (align to teacher class means)
    'proto':    dict(lam_kd=0.0, lam_feat=0.5, feat_proto=True,
                     teacher_correct_only=False),
    # decoupled KD (TCKD + NCKD)
    'dkd':      dict(lam_kd=0.5, lam_feat=0.0, dkd=True,
                     teacher_correct_only=False),
}

# keys that are runner-level sugar, not distill_student kwargs
_SUGAR = ('method', 'masked')


def resolve(cond, defaults):
    """Merge into distill_student kwargs with precedence
    ``defaults < method template < condition overrides``.

    ``defaults`` (experiment ``distill:`` block) carries shared hyperparameters
    (temperature, epochs, lr, ...); the named ``method`` template sets the
    method-defining terms (lam_kd/lam_feat/flags) so e.g. ``baseline``'s
    ``lam_kd=0`` is NOT clobbered by a defaults ``lam_kd``; the condition dict has
    the final say. Returns ``(kwargs, masked)`` where ``masked`` is the sugar flag
    the runner turns into a teacher-correct ``sample_weight`` (it needs per-cell
    teacher logits)."""
    method = cond.get('method', 'baseline')
    if method not in REGISTRY:
        raise KeyError(f'unknown method {method!r}; known: {list(REGISTRY)}')
    kwargs = dict(defaults)
    kwargs.update(REGISTRY[method])
    kwargs.update({k: v for k, v in cond.items() if k not in _SUGAR})
    return kwargs, bool(cond.get('masked', False))
