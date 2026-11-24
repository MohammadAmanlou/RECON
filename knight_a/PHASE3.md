# Phase 3 — KNIGHT-A frozen evaluation v1.1 (24 Nov 2026)

## Reproduce from the repository root

Python 3.10+; reference run NumPy 2.3.5. The original protocol is intact. Read PROTOCOL_AMENDMENT.md for two secondary arms declared before held-out evaluation.

```powershell
git pull --ff-only
python -m pip install numpy matplotlib
python knight_a/validate_freeze.py
python -m unittest knight_a.test_assessment -v
python -m knight_a.evaluation tune --out results_knight_a_v1_1
python -m knight_a.evaluation heldout --out results_knight_a_v1_1
python -m knight_a.audit_results --out results_knight_a_v1_1
python -m knight_a.plot_results --out results_knight_a_v1_1
```

Use a fresh output directory. Do not retune after seeing held-out outcomes. The code refuses overwrite of an existing fitted manifest or silent re-execution in an existing test directory. Exact timing and floating point representations may vary by platform. The full reference output (fit manifest, world records, histories, plot and two-page PDF) is in the separately supplied reproducibility ZIP; these large generated artifacts are not committed here.

## Actual run

30 development worlds, 3,000 learners; 60 held-out worlds, 6,000 learners; each of six methods assesses the same underlying learners with an eight-item, no-repeat-family budget. Dev generator equivalence verified for all 30 worlds. Five original tests passed and all six methods passed dev deterministic replay/family checks. Audit passed for 360 world-method rows, 36,000 unique histories, 288,000 item events, frozen hashes, all summary means and paired intervals.

Fitted tuples (r,b,w,slip,guess): additive (0.6,-1.5,2.5,0.12,0.25); conjunctive (0.45,-0.8,1.5,0.22,0.25). Reference manifest SHA-256 875788d2a05eedcc1ad1cc40ba79d8866f5c84609d4bcb7125fff03adf098665.

Hamming error at eight questions:

| Method | Matched | Noisy | Mismatch |
|---|---:|---:|---:|
| Random | .1790 | .2476 | .2337 |
| Static pairwise BKT-style | .1596 | .2173 | .2161 |
| Pairwise IG | .1591 | .2193 | .2225 |
| Hyperedge IG | .1536 | .2200 | .2146 |
| Mixture IG | .1478 | .2188 | .2153 |
| Scalar IRT-CAT proxy | .3011 | .3312 | .3246 |

Primary mismatch paired difference (pairwise minus hyperedge) 0.0079167, percentile world bootstrap 95% interval [0.0015833,0.0138333]. Mixture secondary 0.0071667 [-0.0007500,0.0144167]. Matched hyperedge 0.0055000 [-0.0004167,0.0111667]; noisy -0.0006667 [-0.0054167,0.0045000]. No claim of uniform improvement. Mismatch exact profile recovery at budget eight is .2665/.2645/.2690 and Brier .1642/.1567/.1601 for additive/hyperedge/mixture; mean censored questions to first exact recovery 6.642/6.798/6.889, failures=9. Lower Hamming does not imply faster exact recovery. Parameters are fitted separately, so contrast compares calibrated pipelines, not a fixed-parameter causal effect of conjunction. Model averaging is not structural discovery; question text is genuine but all response data synthetic, manual Q-tags unvalidated. The IRT comparator is a supervised scalar proxy and BKT-style comparator has no temporal learning transitions. Neither model is a validated human assessment. No proof of general novelty or real learner effectiveness.

Do not treat the smoke test as final results or change the bank after seeing these results. Review the two-page plan and prepare a code walkthrough before sending the professor the demo and plan together.
