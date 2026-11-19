"""RECON-Ω pilot: finite, calibrated interventional competence hypergraphs.

All observations generated here are simulated Bernoulli block scores. An
interventional hyperedge is NOT an internal neural circuit, and the calibrated
cross-channel loading matrix is an explicit (falsifiable) assumption.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

B = np.asarray([[1., 1., 0.], [1., 0., 1.], [1., 1., 1.]])
B4 = np.asarray([1., 2., 1.])  # overidentifying action, specified before testing
CHANNELS = ("acquisition", "lesion", "relearning")
SIGNS = np.asarray([1., -1., 1.])  # lesion decreases raw target accuracy
BASE = 0.46


def hyperedges(p: int, max_order: int) -> tuple[tuple[int, ...], ...]:
    if not 1 <= max_order <= p:
        raise ValueError("invalid p or max_order")
    return tuple(edge for k in range(1, max_order + 1)
                 for edge in combinations(range(p), k))


def feature_matrix(packets: Sequence[int], edges: Sequence[tuple[int, ...]]) -> np.ndarray:
    """Anchored Boolean monomials: phi_S(T)=1[S subset T]."""
    masks = np.asarray(packets, dtype=np.int64).reshape(-1, 1)
    emasks = np.asarray([sum(1 << i for i in edge) for edge in edges],
                        dtype=np.int64).reshape(1, -1)
    return ((masks & emasks) == emasks).astype(float)


@dataclass(frozen=True)
class World:
    p: int
    max_order: int
    edges: tuple[tuple[int, ...], ...]
    w: np.ndarray  # D x 3: common theta, alternative-route u, impairment v
    base: float = BASE

    def __post_init__(self) -> None:
        if self.w.shape != (len(self.edges), 3):
            raise ValueError("coefficient shape mismatch")
        all_masks = np.arange(1 << self.p)
        raw = self.probabilities(all_masks)
        if np.min(raw) < 0.01 or np.max(raw) > 0.99:
            raise ValueError("simulator probabilities out of calibration range")

    def probabilities(self, packets: Sequence[int], fourth: bool = False,
                      violation: float = 0.) -> np.ndarray:
        phi = feature_matrix(packets, self.edges)
        if fourth:
            # controlled, independently falsifiable fourth-action perturbation
            return self.base + phi @ self.w @ B4 + violation * phi[:, -1]
        m = phi @ self.w @ B.T
        return self.base + m * SIGNS.reshape(1, 3)

    def shared_support(self) -> set[tuple[int, ...]]:
        return {edge for edge, row in zip(self.edges, self.w)
                if abs(row[0]) > 1e-10}

    def minimal_packets(self, gain: float) -> set[int]:
        """All inclusion-minimal acquisition packets, no monotonicity assumed."""
        masks = np.arange(1 << self.p)
        values = self.probabilities(masks)[:, 0]
        sufficient = {int(m) for m in masks if values[m] - values[0] >= gain}
        return {m for m in sufficient
                if not any(t != m and (t & m) == t for t in sufficient)}


def make_world(seed: int, p: int = 6, max_order: int = 3,
               shared: int = 4, u_only: int = 2, v_only: int = 2) -> World:
    """A planted sparse world with higher-order edges and confounded channels."""
    rng = np.random.default_rng(seed)
    edges = hyperedges(p, max_order)
    w = np.zeros((len(edges), 3), dtype=float)
    order3 = [i for i, e in enumerate(edges) if len(e) == 3]
    order2 = [i for i, e in enumerate(edges) if len(e) == 2]
    if shared < 2 or shared + u_only + v_only > len(edges):
        raise ValueError("unsupported sparsity configuration")
    # At least two higher-order edges ensure pairwise-only baselines face a real test.
    chosen = list(rng.choice(order3, size=min(2, shared), replace=False))
    available = [i for i in order2 + order3 if i not in chosen]
    chosen += list(rng.choice(available, size=shared - len(chosen), replace=False))
    remaining = [i for i in range(len(edges)) if i not in chosen]
    extras = rng.choice(remaining, size=u_only + v_only, replace=False)
    for i in chosen:
        w[i, 0] = rng.uniform(0.066, 0.082)
    for i in extras[:u_only]:
        w[i, 1] = rng.uniform(0.036, 0.052)
    for i in extras[u_only:]:
        w[i, 2] = rng.uniform(0.026, 0.040)
    return World(p, max_order, edges, w)


def select_packets(p: int, edges: Sequence[tuple[int, ...]], count: int,
                   seed: int, policy: str = "random") -> np.ndarray:
    """Fixed-budget D-optimal design vs uniform random packets.

    The policy is nonadaptive to outcomes and is correctly labeled as such.
    It is a cost-matched experimental-design baseline for later adaptive work.
    """
    rng = np.random.default_rng(seed)
    possibilities = np.arange(1, 1 << p)
    if count > len(possibilities):
        raise ValueError("packet budget exceeds distinct nonempty packets")
    if policy == "random":
        return rng.choice(possibilities, size=count, replace=False)
    if policy != "d_optimal":
        raise ValueError("unknown experimental design")
    phi = feature_matrix(possibilities, edges)
    gram_inv = np.eye(phi.shape[1], dtype=float) / 0.3
    chosen: list[int] = []
    available = np.ones(len(possibilities), dtype=bool)
    for _ in range(count):
        # Matrix determinant lemma: logdet(G+xx')-logdet(G)=log(1+x'G^-1x).
        scores = np.einsum("ij,jk,ik->i", phi, gram_inv, phi)
        scores[~available] = -np.inf
        idx = int(np.argmax(scores + rng.uniform(0, 1e-10, size=len(scores))))
        x = phi[idx]
        gx = gram_inv @ x
        gram_inv -= np.outer(gx, gx) / (1. + x @ gx)
        chosen.append(int(possibilities[idx]))
        available[idx] = False
    return np.asarray(chosen, dtype=int)


def sample_contrasts(world: World, packets: np.ndarray, replicates: int,
                     seed: int) -> np.ndarray:
    """Matched sham controls and treated fresh copies, each charged separately.

    Binomial draws summarize *independent simulated Bernoulli observations*;
    they do not summarize or simulate independently trained neural checkpoints.
    """
    if replicates < 1:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(seed)
    treated = world.probabilities(packets)
    control = np.broadcast_to(world.probabilities([0])[0], treated.shape)
    y_t = rng.binomial(replicates, treated) / replicates
    y_0 = rng.binomial(replicates, control) / replicates
    return SIGNS.reshape(1, 3) * (y_t - y_0)


def group_lasso(phi: np.ndarray, y: np.ndarray, penalty: float,
                iterations: int = 1400) -> np.ndarray:
    """FISTA for multi-task group lasso (one coefficient row per candidate edge)."""
    if y.ndim == 1:
        y = y[:, None]
    n, d = phi.shape
    if n != len(y) or penalty < 0:
        raise ValueError("invalid regression input")
    L = np.linalg.norm(phi, ord=2) ** 2 / n
    z = np.zeros((d, y.shape[1]))
    beta = z.copy()
    momentum = 1.
    xy = phi.T @ y / n
    gram = phi.T @ phi / n
    for _ in range(iterations):
        trial = z - (gram @ z - xy) / L
        length = np.linalg.norm(trial, axis=1)
        shrink = np.maximum(0., 1. - penalty / (L * np.maximum(length, 1e-14)))
        candidate = trial * shrink[:, None]
        m_new = (1. + np.sqrt(1. + 4. * momentum ** 2)) / 2.
        z = candidate + ((momentum - 1.) / m_new) * (candidate - beta)
        beta = candidate
        momentum = m_new
    return beta


def reconstruct_theta(channel_coefs: np.ndarray) -> np.ndarray:
    if channel_coefs.ndim != 2 or channel_coefs.shape[1] != 3:
        raise ValueError("requires all three channels")
    return channel_coefs @ np.array([1., 1., -1.])


def support_f1(truth: set[int], prediction: set[int]) -> dict[str, float]:
    tp = len(truth & prediction)
    prec = tp / len(prediction) if prediction else float(not truth)
    rec = tp / len(truth) if truth else 1.
    return {"precision": prec, "recall": rec,
            "f1": 2. * prec * rec / (prec + rec) if prec + rec else 0.}


def exact_identification_witness() -> dict[str, object]:
    """Exact rank/nullspace witness; theorem assumptions must be explicit."""
    w = np.array([0.14, 0.10, 0.06])
    menus = {"AL": ([0, 1], [1., -1., -1.]),
             "AR": ([0, 2], [1., -1., 0.]),
             "LR": ([1, 2], [1., 0., -1.])}
    assert np.linalg.matrix_rank(B) == 3
    assert np.allclose(np.array([1., 1., -1.]) @ B, [1., 0., 0.])
    for idx, h in menus.values():
        assert np.allclose(B[idx] @ h, 0.) and h[0] != 0
    return {"loading": B.tolist(), "rank_full": int(np.linalg.matrix_rank(B)),
            "rank_two_channel": {name: int(np.linalg.matrix_rank(B[idx]))
                                 for name, (idx, _) in menus.items()},
            "theta": float(w[0]), "means": (B @ w).tolist(),
            "fourth": float(B4 @ w), "two_channel_null_vectors":
            {name: h for name, (_, h) in menus.items()}}


def confirm_interaction(world: World, edge: tuple[int, ...], replicates: int,
                        seed: int, alpha: float = 0.05, n_targets: int = 1,
                        raw_bias_bound: float = 0.) -> dict[str, float | bool]:
    """Independent, finite-sample Hoeffding certificate for a *preselected* edge.

    The edge is selected from exploration data; confirmation trials are fresh.
    This certificate identifies a functional theta_S under calibrated B,
    with a worst-case raw-arm model discrepancy bound supplied by the analyst.
    """
    if replicates < 1 or not (0 < alpha < 1) or n_targets < 1 or raw_bias_bound < 0:
        raise ValueError("invalid confirmation settings")
    rng = np.random.default_rng(seed)
    arms = np.array([sum(1 << i for i in t) for k in range(len(edge) + 1)
                     for t in combinations(edge, k)], dtype=int)
    probabilities = world.probabilities(arms)
    observations = rng.binomial(replicates, probabilities) / replicates
    signs = np.array([(-1) ** (len(edge) - a.bit_count()) for a in arms])
    # θ = Ω(F_A - F_L - F_R), because lesion is oriented as performance *loss*.
    estimate = float(signs @ (observations @ np.array([1., -1., -1.])))
    number_raw_arms = 3 * len(arms)
    # Weighted Hoeffding bound with independent arm-level Bernoulli trials.
    # P(|err| >= t) <= 2exp(-2*n*t^2/sum_raw_weights_squared).
    radius_sampling = float(np.sqrt(number_raw_arms *
                                 np.log(2. * n_targets / alpha) / (2. * replicates)))
    radius_bias = number_raw_arms * raw_bias_bound
    radius = radius_sampling + radius_bias
    return {"estimate": estimate, "radius": radius, "lower": estimate - radius,
            "upper": estimate + radius, "certified_positive": bool(estimate - radius > 0.),
            "replicates_per_raw_arm": replicates,
            "total_simulated_bernoulli_trials": int(number_raw_arms * replicates),
            "true_theta": float(world.w[world.edges.index(edge), 0]),
            "edge": str(edge)}