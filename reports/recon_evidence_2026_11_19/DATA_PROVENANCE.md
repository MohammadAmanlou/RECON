# RECON-Ω evidence data provenance

This report was generated solely from committed RECON numerical outputs. Chart rendering performs no learner training. All quantitative claims belong to a planted synthetic simulator except the separately labelled 8-seed PyTorch modular neural experiment.

| Input | SHA-256 | Bytes |
|---|---|---:|
| `results/per_world.csv` | `b93b5820f23a1f5be340f37bda2c5ed480cdd3f9cfc5f20a6e1ed573cf49d17c` | 2862 |
| `results/summary.json` | `70ba6832bc8a82bd2fea73b8497bc28365328bec000f1be4729d82ab56770e8a` | 4801 |
| `results/robustness/stress_per_world.csv` | `b1c82a77d5f25f508b6e221f9ee081a293446975e9738136616c6d10e1d6b13b` | 23581 |
| `results/robustness/stress_summary.json` | `7bab549d268c35228513b9fff8cca4ffcdca76dd4e213087b132089dee12d5a8` | 27731 |
| `results/neural/neural_module_per_seed.csv` | `7bac841089eb03e2ae8bfb21590f998611b6848b54fc2699be0874573ed2854c` | 1677 |
| `results/neural/neural_module_summary.json` | `50e70513dfc65d6eec0bdcb39b3112d1668e3b9e67630877145f98bcb51231f8` | 2282 |

Generate figures: `python reports/recon_evidence_2026_11_19/make_figures.py --repo-root . --out reports/recon_evidence_2026_11_19/figures`

The initial pilot and five-regime stress suite are distinct runs; do not pool them. 95% paired intervals in the report are reported from `stress_summary.json` and resample planted worlds. The 36/12 confirmations use planted synthetic hidden ground truth and 47,520,000 Bernoulli trials, not 48 human subjects.