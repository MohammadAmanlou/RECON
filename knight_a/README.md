# KNIGHT-A v1.1 — Direction A delivery (27 November 2026)

This release preserves the design freeze and original six-policy evaluation from 24 November. It adds a focused plan, an interactive demo and a compact, separately replayed results snapshot. It does not alter the original question bank, simulator, fitted grid, test seeds, or primary estimand.

- [Focused plan](PLAN.md) (two-page typeset PDF supplied separately with the submission).
- [Interactive demo](demo.html): download the HTML and open it locally; no Python, API, account, network, or GPU needed. Choose a policy and simulated/manual mode, follow eight questions, observe posterior mastery and export a local trace. The demo covers five Bayesian policies; the scalar IRT proxy is included in the Python benchmark only.
- [Single result plot](results_delivery/hamming_curves.svg), [budget curves and final metrics](results_delivery/summary.csv), [paired contrasts](results_delivery/contrasts.json).
- [Frozen design](DESIGN_FREEZE.md), [protocol](data/protocol.json), [amendment](PROTOCOL_AMENDMENT.md), [Phase 3 report](PHASE3.md).
- [Original source-traceable item bank](data/question_bank.json), [candidate graph](data/candidate_graph.json), [semantic evidence](data/kg_evidence.json).

## Reproduce the full frozen benchmark
From RECON root, use a fresh directory:
```bash
git clone --branch knight-a-v1.1-delivery https://github.com/MohammadAmanlou/RECON.git
cd RECON
python -m pip install numpy==2.3.5 matplotlib
python knight_a/validate_freeze.py
python -m unittest knight_a.test_assessment -v
python -m knight_a.evaluation tune --out results_knight_a_v1_1
python -m knight_a.evaluation heldout --out results_knight_a_v1_1
python -m knight_a.audit_results --out results_knight_a_v1_1
python -m knight_a.plot_results --out results_knight_a_v1_1
```
This regenerates full fit, per-world records, histories and the original plot. The reference run audited 360 world-method records, 36,000 histories and 288,000 item events. The 27 Nov numerical replay independently reconstructed the original Q-tag and family arrays and numerical procedures; it matched the reference fitted tuples, 18 endpoint means, and primary CI. Its full output is supplied separately; summary.csv is a rounded snapshot, not the raw reference history. No post-test tuning. The browser demo uses fitted tuples and independent seeded pre-sampled illustrative responses; JS floating-point ties use a 1e-14 tolerance, so it is a teaching interface, not a substitute for the frozen benchmark. The demo engine passed 40 step checks across five policies, normalization, no-repeat families, deterministic replay and equal observed prefixes in a JS runtime. Browser DOM has not been end-to-end tested in this delivery environment.

## Evidence boundary
All responses are synthetic. The original questions are conceptual recognition items, not a validated mastery instrument. Candidate edges and tags are curator hypotheses. Hyperedge IG improves primary Hamming and Brier relative to pairwise IG under mismatch, but does not improve exact recovery or time to first exact recovery and is not superior in noisy worlds. The mixture is exploratory/secondary, not structure discovery. See PLAN.md for the full scope and future validation. Do not interpret as a human study or a general proof of novelty.
