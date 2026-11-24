"""KNIGHT A: development-only static Bayesian adaptive assessment."""
import argparse
import itertools
import json
from pathlib import Path
from time import perf_counter
import numpy as np

DATA = Path(__file__).resolve().parent / 'data'
BANK = sorted(json.loads((DATA / 'question_bank.json').read_text(encoding='utf-8')), key=lambda q: q['id'])
GRAPH = json.loads((DATA / 'candidate_graph.json').read_text(encoding='utf-8'))
PROTOCOL = json.loads((DATA / 'protocol.json').read_text(encoding='utf-8'))
SKILLS = GRAPH['topological_order']
PARENTS = {SKILLS.index(e['child']): tuple(SKILLS.index(s) for s in e['parents']) for e in GRAPH['candidate_hyperedges']}
STATES = np.array(list(itertools.product((0, 1), repeat=len(SKILLS))), dtype=float)
Q = np.array([[s in item['skills'] for s in SKILLS] for item in BANK], dtype=float)
FAMILIES = np.array([item['item_family'] for item in BANK])
METHODS = ('random', 'pairwise_static_bkt', 'pairwise_ig', 'hyperedge_ig')
DEFAULT = (0.45, -1.5, 2.5, 0.12, 0.25)  # smoke only, not tuned


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def entropy(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -p * np.log(p) - (1-p) * np.log1p(-p)


def rng_for(seed, namespace):
    return np.random.default_rng(np.random.SeedSequence([seed, namespace]))


def prior(family, params, parents=PARENTS):
    r, b, w, _, _ = params
    if family not in ('pairwise', 'hyperedge'):
        raise ValueError(family)
    prob = np.ones(len(STATES))
    for j in range(len(SKILLS)):
        ps = parents.get(j, ())
        if ps:
            x = STATES[:, ps]
            f = x.prod(axis=1) if family == 'hyperedge' else x.mean(axis=1)
            p = sigmoid(b+w*f)
        else:
            p = r
        prob *= np.where(STATES[:, j] == 1, p, 1-p)
    return prob / prob.sum()


def likelihood(params, q=Q, states=STATES):
    _, _, _, slip, guess = params
    mastered = states @ q.T == q.sum(axis=1)
    return np.where(mastered, 1-slip, guess)


def posterior_update(posterior, likelihood_column, correct):
    p = posterior * (likelihood_column if correct else 1-likelihood_column)
    return p / p.sum()


def select(method, posterior, like, eligible, rng, q=Q, states=STATES):
    """Public policy inputs only: posterior, candidate likelihood/tags, eligibility, RNG."""
    candidates = np.flatnonzero(eligible)
    if not len(candidates):
        raise ValueError('No eligible question')
    if method == 'random':
        return int(rng.choice(candidates))
    if method == 'pairwise_static_bkt':
        p = posterior @ states
        scores = q @ (p*(1-p)) / q.sum(axis=1)
    elif method in ('pairwise_ig', 'hyperedge_ig'):
        scores = entropy(posterior @ like) - posterior @ entropy(like)
    else:
        raise ValueError(method)
    return int(candidates[np.argmax(scores[candidates])])


class DevWorld:
    """Oracle environment; only evaluator owns this object. Test seeds refused."""
    def __init__(self, regime, seed, n):
        if seed not in PROTOCOL['worlds']['development'][regime]:
            raise ValueError('Only registered development seeds allowed in phase 2')
        a, b, c, d = (rng_for(seed, ns) for ns in range(1, 5))
        parents = dict(PARENTS)
        if regime == 'mismatch':
            child, removed = ((5, 3), (2, 1))[int(a.integers(2))]
            parents[child] = tuple(s for s in parents[child] if s != removed)
        u = b.normal(0, {'matched': 0., 'noisy': .4, 'mismatch': .6}[regime], n)
        z = np.zeros((n, len(SKILLS)), dtype=int)
        for j in range(len(SKILLS)):
            ps = parents.get(j, ())
            if ps:
                x = z[:, ps]
                f = x.mean(axis=1) if regime == 'mismatch' else x.prod(axis=1)
                p = sigmoid(a.uniform(-1.8, -.6) + a.uniform(1.5, 3.2)*f + u)
            else:
                p = a.uniform(.35, .75)
            z[:, j] = b.random(n) < p
        if regime == 'mismatch':
            p = sigmoid(np.log(.25/.75) + 2.5*(z@Q.T/Q.sum(axis=1)) + .8*u[:, None] - c.uniform(-.6, .6, len(BANK)))
        else:
            sr, gr = ((.06, .18), (.18, .30)) if regime == 'matched' else ((.15, .30), (.20, .35))
            p = np.where(z@Q.T == Q.sum(axis=1), 1-c.uniform(*sr, len(BANK)), c.uniform(*gr, len(BANK)))
        self.mastery = z
        self.answers = d.random(p.shape) < p
        self.true_parents = parents


def run_learner(method, params, respond, rng, budget=8):
    """Assessment controller can request one observed answer; no oracle truth in policy."""
    family = 'hyperedge' if method == 'hyperedge_ig' else 'pairwise'
    p = prior(family, params)
    like = likelihood(params)
    eligible = np.ones(len(BANK), dtype=bool)
    marginals, history = [p @ STATES], []
    for _ in range(budget):
        j = select(method, p, like, eligible, rng)
        predicted = float(p @ like[:, j])
        y = bool(respond(j))
        history.append({'id': BANK[j]['id'], 'family': str(FAMILIES[j]), 'correct': y, 'predicted': predicted})
        p = posterior_update(p, like[:, j], y)
        eligible &= FAMILIES != FAMILIES[j]
        marginals.append(p @ STATES)
    return np.array(marginals), history


def evaluate_dev(regime, seed, n=100):
    world = DevWorld(regime, seed, n)
    output = []
    for method in METHODS:
        start = perf_counter()
        trajectories, histories = [], []
        policy_rng = rng_for(seed, 5)
        for i in range(n):
            p, h = run_learner(method, DEFAULT, lambda j: world.answers[i, j], policy_rng, PROTOCOL['question_budget'])
            trajectories.append(p)
            histories.append(h)
        m = np.array(trajectories)
        error = (m >= .5) != world.mastery[:, None, :]
        output.append(dict(regime=regime, seed=seed, method=method, n=n,
                           hamming=error.mean(axis=(0,2)).tolist(),
                           exact=(~error.any(axis=2)).mean(axis=0).tolist(),
                           final_brier=float(np.mean((m[:, -1]-world.mastery)**2)),
                           seconds=perf_counter()-start, histories=histories))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learners', type=int, default=20)
    parser.add_argument('--out', type=Path, default=Path('knight_a_dev_smoke.json'))
    args = parser.parse_args()
    if not 1 <= args.learners <= PROTOCOL['worlds']['learners_per_world']:
        parser.error('learners must be in 1..100')
    rows = [row for regime, seeds in PROTOCOL['worlds']['development'].items() for row in evaluate_dev(regime, seeds[0], args.learners)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'stage':'development smoke only; fixed untuned params; not scientific evidence', 'rows':rows}, indent=2)+'\n', encoding='utf-8')
    for row in rows:
        print(f"{row['regime']:8} {row['method']:20} Hamming@8={row['hamming'][-1]:.3f} Brier={row['final_brier']:.3f}")
    print('DEV SMOKE ONLY; no held-out worlds accessed. Saved', args.out)


if __name__ == '__main__':
    main()
