"""Outcome-adaptive, cost-matched proposal policy for synthetic RECON assays.

This is an *exploratory heuristic*, not a novel optimality guarantee. The policy
observes only previously randomized block contrasts, never the planted graph.
All proposed designs receive the same three-channel treated/sham measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .core import (World, SIGNS, feature_matrix, group_lasso,
                   reconstruct_theta)


@dataclass(frozen=True)
class AdaptiveResult:
    packets: np.ndarray
    contrasts: np.ndarray
    acquisition_trials: int
    policy: str = "adaptive_margin"


def propose_adaptively(world: World, edges: Sequence[tuple[int, ...]],
                       budget: int, replicates: int, seed: int,
                       penalty: float = 0.006,
                       cutoff: float = 0.03,
                       pilot: int = 12,
                       recovery_shift: tuple[tuple[int, ...], float] | None = None,
                       fitter_iterations: int = 300) -> AdaptiveResult:
    """Select packets sequentially using model-dependent uncertainty near edge cutoff.

    The design starts from D-optimal warmup, then uses a ridge surrogate's
    inverse information matrix, with outcome-dependent edge weights estimated
    from multi-task group-lasso. Every sixth post-warmup packet uses plain
    information maximization as a safety/exploration floor.

    ``recovery_shift`` is a deliberately unmodeled off-target response effect,
    not ground-truth causal acquisition, and is disclosed in the stress suite.
    """
    if not (1 <= pilot <= budget < (1 << world.p)) or replicates < 1:
        raise ValueError("invalid experimental budget/pilot/replicates")
    if penalty < 0 or cutoff < 0:
        raise ValueError("invalid penalty/cutoff")
    rng = np.random.default_rng(seed)
    pool = np.arange(1, 1 << world.p, dtype=int)
    pool_phi = feature_matrix(pool, edges)
    d = len(edges)
    gram_inv = np.eye(d) / .3
    available = np.ones(len(pool), dtype=bool)
    packets: list[int] = []
    responses: list[np.ndarray] = []
    off_target_mask = None
    shift = 0.
    if recovery_shift is not None:
        edge, shift = recovery_shift
        off_target_mask = sum(1 << k for k in edge)

    for t in range(budget):
        if t < pilot or (t - pilot) % 6 == 5:
            # D-optimal / exploration step, independent of past outcomes.
            scores = np.einsum('ij,jk,ik->i', pool_phi, gram_inv, pool_phi)
        else:
            x_seen = feature_matrix(packets, edges)
            fitted = group_lasso(x_seen, np.asarray(responses),
                                 penalty=penalty, iterations=fitter_iterations)
            theta = reconstruct_theta(fitted)
            # This scale is a heuristic numerical regularizer, *not* a
            # calibrated Bayesian posterior standard deviation.
            std = np.sqrt(np.maximum(np.diag(gram_inv), 1e-9)) * .025 + .01
            edge_weight = .12 + np.exp(-.5 * ((np.abs(theta) - cutoff) / std) ** 2)
            # Information proxy focused on uncertainty of coefficients near
            # the decision threshold; no claim of optimal information gain.
            effective = (gram_inv * edge_weight[None, :]) @ gram_inv
            scores = np.einsum('ij,jk,ik->i', pool_phi, effective, pool_phi)
            denom = 1. + np.einsum('ij,jk,ik->i', pool_phi, gram_inv, pool_phi)
            scores /= denom
        scores[~available] = -np.inf
        choice = int(np.argmax(scores + rng.uniform(0., 1e-11, len(scores))))
        packet = int(pool[choice])
        trial_probs = world.probabilities([packet])[0].copy()
        if off_target_mask is not None and (packet & off_target_mask) == off_target_mask:
            trial_probs[2] += shift
        if not np.all((trial_probs > 0) & (trial_probs < 1)):
            raise ValueError("off-target response effect made probabilities invalid")
        control_probs = world.probabilities([0])[0]
        treated = rng.binomial(replicates, trial_probs) / replicates
        sham = rng.binomial(replicates, control_probs) / replicates
        responses.append((treated - sham) * SIGNS)
        packets.append(packet)
        x = pool_phi[choice]
        v = gram_inv @ x
        gram_inv -= np.outer(v, v) / (1. + x @ v)
        available[choice] = False
    return AdaptiveResult(np.asarray(packets, dtype=int),
                          np.asarray(responses), 6 * budget * replicates)