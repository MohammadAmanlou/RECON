# Executed results — calibrated synthetic RECON-Ω pilot

**Date:** 2026-11-19. **Status:** locally executed. **Evidence type:** planted synthetic worlds; no LLM, no human participants, no pretrained-model unlearning.

The runner generated the JSON/CSV evidence in `results/` directly from the committed code. The same command in CI verifies reproducibility. There is no neural performance claim.

## Controlled comparison

12 independent planted worlds per policy; p=6, max hyperedge order 3, 4 true common edges, 2 alternative-route-only and 2 lesion-impairment-only edges; exactly 48 packets per world, 3 channels, each treatment and matched sham with 1,024 Bernoulli draws. Both policies use **294,912 synthetic observations/world**. Group lasso penalty 0.006 and demonstration threshold 0.03 (not tuned on separate confirmation worlds).

| Method and metric | Uniform random packets | D-optimal packets (nonadaptive) |
|---|---:|---:|
| Three-channel shared-edge F1 | 0.7993 ± 0.2903 | 0.9762 ± 0.0556 |
| Acquisition-only F1 | 0.5383 ± 0.2322 | 0.7478 ± 0.1599 |
| Tri-channel pairwise-only F1 | 0.3190 ± 0.2670 | 0.3012 ± 0.2511 |
| Three-channel precision | 0.8514 ± 0.2937 | 1.0000 ± 0.0000 |
| Three-channel recall | 0.7708 ± 0.3100 | 0.9583 ± 0.0973 |
| Held-out three-channel mean RMSE | 0.0262 ± 0.0160 | 0.0092 ± 0.0028 |
| Held-out fourth-action mean MAE | 0.0237 (corrected; see below) | 0.0081 (corrected; see below) |

*All ± values are standard deviations across planted worlds, **not** confidence intervals. The design is not outcome-adaptive. Results only show comparative behavior of algorithms on this chosen synthetic generator.*

## Algebraic and certificate diagnostics

- Calibrated 3×3 response B: full rank 3; every fixed two-channel menu rank 2 with a null direction changing the shared parameter θ. Synthetic witness θ=0.140, channel means [0.24000000000000002, 0.2, 0.30000000000000004], declared fourth-action mean 0.400. This is an **algebraic construction**, not an experimental LLM finding.
- **Oracle-selected diagnostic** edge `(1, 2, 5)`: θ true=0.0747; independent estimate=0.0656, conservative interval=[0.0286, 0.1025], certified-positive=True; 1,440,000 raw simulated Bernoulli observations. Selection used the planted truth; hence this is **not** a discovery-rate or feature-selection certificate.
- **Oracle-selected diagnostic** edge `(2, 3, 4)`: θ true=0.0777; independent estimate=0.0648, conservative interval=[0.0279, 0.1018], certified-positive=True; 1,440,000 raw simulated Bernoulli observations. Selection used the planted truth; hence this is **not** a discovery-rate or feature-selection certificate.
- **Oracle-selected diagnostic** edge `(1, 5)`: θ true=0.0791; independent estimate=0.0817, conservative interval=[0.0568, 0.1065], certified-positive=True; 720,000 raw simulated Bernoulli observations. Selection used the planted truth; hence this is **not** a discovery-rate or feature-selection certificate.
- Scientific executable tasks: 100 DAG programs independently checked by reverse-mode, forward-mode, and central-difference gradients. Maximum reverse/forward absolute discrepancy 2.220e-16; maximum reverse/finite-difference discrepancy 5.852e-11.

## New adversarial and trainable-learner evidence

The original fourth-action projection used `solve(B, B4)` rather than `solve(B.T, B4)`, and mixed a probability baseline with a contrast. The corrected comparison preserves all three-channel F1 values but **supersedes the old fourth-action MAE**. This was verified by a new population-exact regression test. Full raw evidence and reports:

- [12-world/5-condition stress suite, paired bootstrap, 48 independently confirmed edges including 12 negative controls](ROBUSTNESS.md)
- [8-seed actually trained PyTorch modular-learner experiment showing that reversible lesion does not unlearn parameters](NEURAL_PILOT.md)

## Interpretation and strong caveats

1. The models and loadings are constructed so triple-channel inversion is valid. The advantage is **conditional on those assumptions** and does not establish internal neural mechanisms.
2. D-optimal design is a known method. The observed packet-efficiency gain over uniform is **not a novel algorithmic contribution**, nor a comparison with active factorial or optimal pure exploration.
3. Reported F1 relies on planted hypergraph truth, which has **no analogue known in advance for LLMs**. For LLMs, held-out interventional prediction, transfer and leakage tests matter.
4. The fourth-action positive-control data are generated using the same calibrated response model; a genuine falsification requires independently implemented interventions.
5. Confidence intervals are valid only given independent samples, preselection or data splitting, correctly declared bias bound and the response loading assumption. In the actual neural setting these require new calibration.
6. Results do not prove RECON-Ω is state of the art or novel relative to active factorial design; future experiments must test those claims.

## Reproduce

```bash
python -m pip install -e ".[test,plot]"
pytest -q tests
python -m recon.experiments --out results --seeds 12 --packets 48 --replicates 1024 --confirm-replicates 60000
```