# Phase 2 — development-only simulator (24 Nov 2026)

Run from the RECON root with Python 3.10+:

```bash
python -m pip install numpy
python knight_a/validate_freeze.py
python -m unittest knight_a.test_assessment -v
python -m knight_a.assessment --learners 20 --out knight_a_dev_smoke.json
```

Five unit tests passed in the local validation run: posterior/prior normalization and unary equivalence, equal budget and deterministic replay, development-only seed guard, counterfactual oracle isolation, and a discriminative two-skill toy. The latter checks informative item selection, not a claim of superiority.

Smoke run: 20 simulated learners per regime, one registered DEVELOPMENT world per regime (100, 200, 300), fixed untuned settings. The 12 final Hamming/Brier pairs were:

| Regime | Random | Static pairwise BKT adaptation | Pairwise IG | Hyperedge IG |
|---|---|---|---|---|
| matched | .117/.091 | .125/.112 | .133/.109 | .142/.103 |
| noisy | .267/.180 | .233/.171 | .242/.184 | .242/.178 |
| mismatch | .275/.164 | .267/.186 | .300/.194 | .317/.206 |

This is a functionality check, not scientific evidence, and does not show hyperedge superiority. No held-out worlds accessed. Next: 32-point dev grid per prior family on all registered development worlds, immutable fitted manifest and code hashes, then one independent evaluation with calibration and world-level bootstrap. The static BKT comparator has no learning/forgetting transitions, as prescribed in the frozen assessment-only protocol. Do not alter the question bank or protocol after seeing test results. No professor handoff yet.
