# KNIGHT-A / WARP — exploratory research extension

This folder extends the existing frozen KNIGHT-A v1.1 experiment; it does not replace the eight-question pilot, change its protocol, or claim that WARP was evaluated in it. Both use the same 20 original linear-algebra questions, 11 item families, six skills and curator-authored candidate prerequisites. This extension implements a finite-horizon exact diagnostic planner with action-dependent Walsh moments, a safe branch bound, a finite fixed-model minimax regret solver, and a separate spectral question-bank audit.

## Run (Node.js 18+; no dependencies)
From the RECON repository root:
```sh
node knight_a/warp/test_warp.cjs
node knight_a/warp/experiment.cjs warp_replay.json
node knight_a/warp/witness_audit.cjs
node knight_a/warp/ablation.cjs
```
Open `knight_a/warp/demo.html` locally in any modern browser; the embedded exact policy trees are generated from the same candidate priors and question bank. Choose the policy index or mixture planner, answer two questions and inspect the changing mastery probabilities. The original eight-question browser demo remains `knight_a/demo.html`.

## What was run
- 178 assertions (dense versus spectral, pruning vs unpruned, tree replay, parity marginal and posterior counterexample, finite-game bounds, no hidden-world input to the planner).
- 128 demo engine paths checked (16 policy indexes x two modes x four binary response sequences). Real browser rendering has not been manually tested.
- 36 independent simulated worlds, 64 learners each (2,304 total), seven methods, two distinct question families per learner, common potential responses, 2,000 paired world bootstrap draws per contrast. No tuning was performed: fixed illustrative candidate parameters, new seeds; exploratory rather than confirmatory. The shifted graph is included as a candidate, while response misspecification and world parameters differ. Do not call it zero-shot structure transfer.
- `results.json` contains all per-world metrics and computed intervals, `witness_results.json` all maximizing bundles, `ablation_results.json` dense/pruned/spectral controls. All new execution was in a V8 JavaScript environment using the committed source; local Node/browser replay is recommended before sending.

## Main findings
| Regime | Pairwise IG | Hyperedge IG | Mixture IG | Mixture MER | WARP |
|---|---:|---:|---:|---:|---:|
| Conjunctive | .30686 | .29709 | .30078 | .28038 | .27973 |
| Additive | .31033 | .31554 | .30990 | .30339 | .30621 |
| Shift | .28320 | .31228 | .28060 | .28668 | .28103 |

Final Hamming loss, lower is better. WARP versus mixture MER improvement in shift=.00564, paired 95% percentile interval approximately [0,.01150]; additive=-.00282 [-.00543,-.00065]; conjunctive=.00065 [0,.00195]. WARP does not dominate mixture IG or MER; exact profile accuracy is not universally improved. First-exact metric censors never-exact at 3; it is not time to guaranteed recovery. ECE is a descriptive, world-averaged ten-bin estimate, not coverage. The prior eight-question results include a scalar IRT-CAT proxy and a static BKT-style comparator; new WARP comparisons do not rerun those.

Uniform mixture two-step exact risk .3195464102179067 (spectral) versus .31954641021790664 (dense); 405 evaluated nodes vs 1,473 unpruned dense nodes. The ablation on a conjunctive prior shows 461 nodes for both dense and spectral with the same bound; node savings are due to pruning, not the spectral representation. Unpruned spectral moment entries visited: 11,326 versus 94,272 dense state entries at depth two; counts are representation sizes, not FLOPs or measured runtime. Root uses 56/64 moments and the code still initializes dense priors. No scalability claim.
Finite four-model regret bounds [.00656077,.01219242], gap .00563165 after 16 iterations. These certify optimization of the specified finite game, not validity of student diagnoses under arbitrary shift.
All four candidate interaction masks have zero single-question TV under the parity perturbation audit; positive two-question TV. This is model-specific question-bank observability, not edge validation.

## Before next experiments
Read THEORY.md for assumptions, proofs and related-work threats. Preserve the original bank and existing results. Next: preregister evaluation on genuinely unseen graph structures and separately fitted response models; compare factorized POMDP/value-directed compression, stronger CD-CAT and robust BOED, include actual IRT/BKT where suitable, and assess spectral operator approximation with error bounds before neural training. Real mastery labels and statistical coverage require separate validation. No claims of causal prerequisite validity, human effectiveness, universal superiority, or first-ever novelty.
