# RECON-Ω: Interventional Competence Hypergraph Discovery

[![Research stage](https://img.shields.io/badge/stage-synthetic--pilot-blue)](#evidence-boundary)

**Research software for** *Recovering Causal Competence Hypergraphs via Controlled Acquisition, Functional Lesions, and Selective Relearning* (Mohammad Amanlou; research proposal for discussion with Prof. Andrey Ustyuzhanin).

**Current release:** a runnable **calibrated synthetic-world pilot**, finite-sample certificate checks, a small executable-science task generator, and honest compute-matched baselines. **No LLM was trained or unlearned in this release.** The simulated latent `theta` is planted ground truth, not evidence of an internal mechanism in a transformer.

## What runs now

| Component | Implementation | Evidence status |
|---|---|---|
| Acquisition / lesion / relearning | Three response channels, independently generated Bernoulli observations with fresh matched controls | Synthetic calibrated simulator |
| Hypergraph mechanism | Sparse order-1/2/3 skill effects, alternative-route and lesion nuisance signals | Planted world |
| Recovery | Multi-channel group-lasso estimator + `theta = m_A + m_L - m_R` | Executed benchmark |
| Experiment policy | Uniform sampling and **nonadaptive** D-optimal packet selection, same action budget | Executed matched-cost comparison |
| Ambiguity | Exact row-space witness; all two-channel menus fail in a constructed alternative | Algebraic unit test |
| Confidence | Independent confirmation on prespecified interactions using weighted Hoeffding + declared bias allowance | Synthetic independent trials; oracle-selected diagnostic, **not** a discovery guarantee |
| Source-to-task compiler | Executable arithmetic DAGs with shared intermediates and reverse/forward/finite-difference gradient verification | 100 generated program tests |
| Neural testbed | Not implemented | **Not evaluated** |

## Quick start

Python 3.10+:

```bash
python -m pip install -e '.[test,plot]'
pytest -q
python -m recon.experiments --out results --seeds 12 --packets 48 --replicates 1024 --confirm-replicates 60000
```

Direct from checkout without installation:

```bash
PYTHONPATH=src pytest -q tests
PYTHONPATH=src python -m recon.experiments --out results
```

Deterministic seeds and protocol parameters are explicitly recorded in `src/recon/experiments.py`. The command writes `results/summary.json`, `results/per_world.csv`, and an optional plot if Matplotlib is available. All Bernoulli draws are simulated and independently seeded.

## Evidence boundary

1. **What the causal claim means here:** under randomized packet interventions and the stipulated cross-channel loading matrix, `theta` is a *functional* multi-channel effect. The three channels do not, without extra validated assumptions, identify neural circuits or erased knowledge.
2. **What the certificate guarantees:** for a fixed, preselected interaction and independently sampled confirmation observations, a union-bound Hoeffding interval simultaneously covers the calibrated functional contrast under the declared maximum systematic bias. It does **not** certify validity of the loading matrix.
3. **What the fourth action does:** it is an overidentifying prediction; in these simulations it is generated using the same stipulated model, so it is a **positive control**, not an independent mechanistic validation. A deliberate violation can be injected.
4. **What comparison is fair:** D-optimal vs uniform has equal numbers of treated, sham, and channel-specific observations. D-optimal is **not** outcome-adaptive, and all observed gains in planted worlds may be model-dependent.
5. **What remains open:** calibrated neural functional lesions, truly novel experimental design guarantees, source-grounded extraction, independent neural confirmation, richer competitor baselines, out-of-distribution generalization, and proof of novelty against active factorial designs.

Read [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) and [`docs/RESULTS.md`](docs/RESULTS.md) before interpreting the experiment tables.

## Repository layout

```
src/recon/core.py             # simulator, design policies, recovery, confidence
src/recon/scientific.py       # executable DAG tasks + independent checkers
src/recon/experiments.py     # reproducible benchmark CLI and results serialization
tests/                       # numerical, scientific, and statistical checks
results/                     # actually executed synthetic-world outcomes
docs/                        # scientific assumptions, results and next steps
.github/workflows/ci.yml     # tests and rerun, with result artifacts
```

## Immediate research next steps

1. Calibrate a fourth protocol on modular learners where intervention targets and leakage are independently verifiable.
2. Compare against channel-aware active factorial and optimized fixed-confidence experimental designs—not only random sampling.
3. Build a trainable, small-transformer task world with model copies, controlled fine-tuning, reversible functional masks, and selective relearning; log independently trained seeds, off-target damage, and prospective transfer.
4. Pre-register a correct-decision-yield vs cost experiment with held-out neural actions; abstain when the calibrated response restrictions fail.

**Reproducibility:** The current claim is that this *code* reproduces a limited synthetic proof-of-concept. It does not establish top-conference novelty, empirical neural effectiveness, supervisor acceptance, or funding.