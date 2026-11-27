# WARP: technical derivations and claim boundary

## Model
Static binary mastery x in {0,1}^K; a question has known conditional Bernoulli likelihood f(q,y,x), shared by the planning models, conditionally independent responses given x. One item per semantic family. All models specify a prior over x; alternative directed prerequisite CPDs, additive projections and a uniform prior are permitted. This is not causal identification of learning prerequisites. Real assessment may violate all these assumptions.

Let chi_T(x)=(-1)^(sum_{j in T} x_j), and m_T(h)=sum_x p(x) L_h(x) chi_T(x), where L_h is the likelihood of the observed path (the policy's action probabilities can be omitted for a fixed path). With normalized Walsh coefficients fhat(q,y,S), multiplication gives the exact unnormalized transition
m_T(h,q,y)=sum_S fhat(q,y,S) m_(T xor S)(h).
The terminal unnormalized Bayes Hamming risk is r(m)=sum_j (m_empty-|m_{j}|)/(2K). A negative singleton moment implies mastery.

## 1. Action- and horizon-dependent closure
Let D={empty,{1},...,{K}}, A be the remaining items, and A_q remove every item in q's family. Define C(A,0)=D and
C(A,d)=D union union_{q in A} {T xor S : T in C(A_q,d-1), S in support(f(q,0)) union support(f(q,1))}.
Including D at all depths supports pruning and interim reports. By induction on d, every child moment needed at depth d-1 is computable from C(A,d); the leaf loss and probabilities use D alone. Therefore the same value is obtained as full-state finite-horizon Bayes planning, in exact arithmetic. Two priors agreeing on C(A,d) have identical path masses and terminal skill moments for every admissible tree. This is sufficient compression, not a claim of minimality. Dense likelihoods or long horizons can recover the entire 2^K basis. The implementation still starts with full-state priors and a Walsh transform.

## 2. Safe lower bound
Write beta(q)=min_x f(q,0,x)+min_x f(q,1,x)=1-(max_x f(q,1,x)-min_x f(q,1,x)).
For each skill, both nonnegative unnormalized class masses in a response branch are at least min_x f(q,y,x) times their previous values. Taking their minimum, summing over y and averaging skills gives sum_y r(m_y)>=beta(q) r(m). For d remaining questions, multiply the d smallest family-minimum betas among eligible families to obtain a lower bound valid for every adaptive tree: along each branch no family can be repeated. For a fixed next question, multiply the sum of its two child risks by the remaining-family bound. If this exceeds an incumbent feasible risk, safely prune that action (implementation uses floating tolerance). This bound also works in dense coordinates. Our ablation explicitly separates pruning from compression.

## 3. Finite-model minimax regret
For candidate models p_1,...,p_M, nature chooses one model before assessment, not at each step. Let r_m^* be the best achievable terminal Hamming risk knowing m at the same horizon. For tree pi, G(m,pi)=R_m(pi)-r_m^*. A randomized tree is sampled once before observing answers. For any model weights w, a best response is exact Bayes planning with prior sum_m w_m p_m: the oracle offset is independent of pi. We use multiplicative weights on models and this exact best response. At each iteration, the weighted regret is a lower bound on the minimax value; the maximum model regret of the uniform mixture of trees so far is an upper bound. Keep the maximum lower bound and minimum upper bound. Their difference is a computable optimality gap in the finite game, up to numerical tolerance. These are standard game principles, not a new generic minimax theorem. No guarantee extends to models outside the candidate set.

## 4. Spectral question-bank audit
For a diagnostic perturbation p_±(x)=2^-K(1±delta chi_S(x)), any fixed bundle response history h has P_+(h)-P_-(h)=2 delta Lhat_h(S). Thus TV(P_+,P_-)=delta sum_h |Lhat_h(S)|. A single question can be blind to an interaction even if a complementary bundle reveals it. The same pathwise coefficient identity holds for a shared adaptive policy when its action probabilities are included. Our audit enumerates fixed nonadaptive bundles only; it does not estimate maximum adaptive TV or validate causal prerequisites. On KNIGHT, all four candidate interaction masks have zero single-item TV and positive maximum two-item TV under the assumed DINA channel (delta=.4). Results and maximizing bundles are saved.

## 5. Closest related work / what remains new
- Poupart & Boutilier, Value-Directed Compression of POMDPs (NeurIPS 2002): https://papers.neurips.cc/paper_files/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d-Abstract.html . Generic lossless reward/transition-preserving subspaces predate this work. Our specific candidate is a finite-horizon, remaining-family-indexed Walsh closure and its coupling to diagnostic structural observability; prove a genuine complexity advantage over value-directed / factorized methods before claiming novelty.
- Javdani et al., Near Optimal Bayesian Active Learning for Decision Making (AISTATS 2014): https://proceedings.mlr.press/v33/javdani14.html . Decision-region identification and complementary queries already studied.
- Minimum Expected Risk CD-CAT: https://pmc.ncbi.nlm.nih.gov/articles/PMC9073635/ . Risk-directed question selection already exists; compare directly.
- Harikumar et al., Robust Bayesian Decision Making under Adversarial Uncertainty (UAI 2026): https://proceedings.mlr.press/v337/harikumar26a.html . Robust decision-aware sequential design is not itself novel.
- Uncovering Latent Structures (2026): https://pmc.ncbi.nlm.nih.gov/articles/PMC13294626/ . Joint latent hierarchy and Q-matrix learning exists; not claimed here.

Open research targets: (i) witness-directed action ordering and budget-aware rank/minimal closure with a proved computational benefit; (ii) approximate spectral transitions with end-to-end decision error certificates for larger banks; (iii) independently learned and calibrated model uncertainty and human validation. A neural moment predictor or policy distillation is justified only if it beats strong exact/factorized controls at scale. No neural model is trained yet. The current implementation has no statistical mastery coverage guarantee. The witness audit is separate from the planner; a direct witness-conditioned search is future work.
