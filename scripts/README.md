# scripts

`scripts/` is the framework self-verification layer — no model training, no experiments.

- `verify_foundation.py` — data layer + small-model vendoring byte-faithful
- `verify_backbones.py --model {mirepnet,cbramod,labram}` — big-model backbone build+forward
- `smoke_test.py` — canonical split + adapter forward contract
