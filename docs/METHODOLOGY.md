# Methodology and estimand — RECON-Ω pilot

## Declared scope

This is a finite-domain benchmark with an explicitly planted competence mechanism. The downstream observational data consist of simulated bounded Bernoulli outcomes. The target is the interaction coefficient of a **calibrated functional** response model, **not** a transformer’s internal latent capability.

Let `p=6`, maximal degree `d=3`, and `H_d={S subset[p]: 1<=|S|<=d}`, containing 41 possible interaction terms. The inclusion monomial is `phi_S(T)=1[S subset T]`. For each candidate `S` the planted response vector is `w_S=(theta_S,u_S,v_S)`. Each intervention channel uses the assumed loadings:

```
B = [1 1 0]   # A: theta + alternative route
    [1 0 1]   # L: theta + off-target impairment
    [1 1 1]   # R: theta + alternative + impairment
```

The measured **raw** probability of correct response on fresh tasks is `b + s_j * Σ_{S⊆T}(Bw_S)_j` where `s=(+1,-1,+1)`. The lesion sign is reversed to turn a performance drop into an oriented positive effect. For every packet/channel we sample a separate, matched empty-packet control. `b=0.46` is known to the simulator but is *not* fed to the estimator.

**Mechanistic identifiability requires the loading model.** The algebraic identity `theta=m_A+m_L-m_R` is a restricted model property, not an assumption-free causal inference from three arbitrary editing interventions. A fourth action is declared as `b4=(1,2,1)`; its prediction is `2*m_R-m_L`. Explicitly corrupting that action produces an overidentification failure.

## Synthetic-world design

A fresh planted world selects four nonzero common `theta` interactions (at least two of order three), two alternative-route-only interactions (`u`), and two impairment-only interactions (`v`). Coefficient magnitudes and all data-generator seeds are in `make_world`. Bernoulli response probabilities are checked to remain strictly in `(0,1)` before sampling.

We select 48 unique nonempty skill packets from 63 candidates. Every selected packet receives **3 channels × treated+sham × 1,024 trials**. Both policies have the same total of **294,912 simulated Bernoulli observations per world**, of which 147,456 are treated and 147,456 are matched sham. The policy choices:

- Uniform sample of 48 packets without replacement.
- Classical nonadaptive D-optimal selection: greedily maximize `log det(G + phi phi^T) - log det G`, using a regularizer and no outcome feedback. This is not claimed novel or outcome-adaptive.

The coefficients per channel are estimated using group-sparse multi-task FISTA and converted to the shared component `theta`. We report a **demonstration** support threshold of `theta_hat>0.03`, common to all policies, and comparable acquisition-only and pairwise-only baselines. For prospective accuracy, predict the held-out packet/channel response means on unused packets; compare to the known oracle (possible only in planted worlds). Fourth-action predictions are prospective relative to model fitting, but not independent of the synthetic generating assumption.

## Formal conditions and limitations

For any fully measured packet design `Phi`, the three-channel joint matrix has the Kronecker structure `Phi⊗B`; its rank is `rank(Phi)*rank(B)`. If `Phi` spans all candidate nonempty monomials and `B` has rank 3, all planted response coefficients are identifiable in population. This is classical linear-model identification; our numerical tests check the premises. If only two channels are measured, there are nonzero `h` such that `B_two h=0` but `h_theta != 0`, so there are observationally equivalent mechanisms with different theta. None of these facts alone proves new nonlinear hypergraph recovery theory.

The group lasso is a heuristic estimator in this pilot and has **no implemented exact-support theorem**. Lasso prediction/F1 is empirical and depends on the design matrix, overlap, noise, and tuning. An arbitrary residual lesion leakage invalidates the cross-channel inversion without an independently argued bound.

## Independent confirmation

For each selected edge `S`, `|S|<=3`, an *independent* synthetic confirmatory experiment samples all `T⊆S` for all three channels with `N=60,000` Bernoulli observations **per raw arm**. The estimator is `Omega_S(F_A-F_L-F_R)`; constants cancel for nonempty `S`. Using `m=3·2^|S|` raw arms with weights ±1 and bounded independent trial scores, the weighted Hoeffding interval has simultaneous radius

```
R(S) = sqrt(m * log(2*K/alpha) / (2*N)) + m * rho_raw
```

for `K` preselected edges, confidence `1-alpha`, and declared per-raw-arm systematic mean uncertainty `rho_raw`. The 2026-11-19 pilot sets `alpha=0.05`, `rho_raw=0.00025`, and selects targets from **oracle truth**, which is exclusively a coverage/estimator diagnostic. A valid future selected-discovery certificate requires edge selection from independent exploration data, then fresh confirmation trials, with the same predeclared familywise error accounting. The calibrated loading assumption is not verified by this CI.

## Scientific tasks

The generated DAG domain includes shared intermediate nodes, repeated chain rule and adjoint accumulation. Every generated program is checked with three computations: reverse automatic differentiation, independent forward-mode dual numbers, and finite differences. This proves checker consistency on sampled tasks; it is not a claim that an LLM learned the underlying scientific concepts.

## Future neural protocols (NOT executed)

Matched checkpoints and seeds; randomized LoRA acquisition; independent localization; reversible activation masks and layer-size/norm shams; ability/side-effect probes; selective relearning with ordered packet histories; unexposed controls and independent fourth interventions; OOD exact verifiers; raw blocks = independent training seeds rather than questions. Calibration and invalid interventions must be reported, not silently treated as causal evidence.