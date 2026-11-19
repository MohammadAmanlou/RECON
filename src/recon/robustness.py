"""Predeclared, reproducible stress suite for the RECON-Ω *synthetic* prototype.

The suite intentionally includes calibration violations and misspecified response
surfaces. It is not an evaluation of real pretrained models or human learning.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .adaptive import propose_adaptively
from .core import (B, B4, FOURTH_BASE, SIGNS, World, confirm_interaction,
                   feature_matrix, group_lasso, hyperedges, make_world,
                   reconstruct_theta, sample_contrasts, select_packets,
                   support_f1)

SCENARIOS = ("calibrated", "low_replication", "weak_signal",
             "recovery_interference", "unmodeled_four_way")
POLICIES = ("random", "d_optimal", "adaptive_margin")


def scenario_world(seed: int, scenario: str) -> tuple[World, tuple[tuple[int, ...], float] | None, int]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    if scenario == "unmodeled_four_way":
        base = make_world(seed, max_order=4)
        w = base.w.copy()
        shared_triples = [j for j, e in enumerate(base.edges)
                          if len(e) == 3 and w[j, 0] > 0]
        fourths = [j for j, e in enumerate(base.edges) if len(e) == 4]
        if not shared_triples or not fourths:
            raise AssertionError("cannot place controlled omitted interaction")
        # Move, rather than add, one true effect to a fourth-order edge; this
        # preserves the global effect budget and valid Bernoulli probabilities.
        w[fourths[seed % len(fourths)], 0] = w[shared_triples[0], 0]
        w[shared_triples[0], 0] = 0.
        return World(base.p, base.max_order, base.edges, w), None, 1024
    base = make_world(seed)
    if scenario == "weak_signal":
        w = base.w.copy()
        w[:, 0] *= 0.35
        return World(base.p, base.max_order, base.edges, w), None, 1024
    if scenario == "recovery_interference":
        distractor = next(e for e, row in zip(base.edges, base.w)
                          if row[1] > 0 and row[0] == 0)
        # Negative deviation in selective relearning, counter to the assumed
        # B-loading model: creates a spurious apparent positive theta edge.
        return base, (distractor, -0.06), 1024
    if scenario == "low_replication":
        return base, None, 96
    return base, None, 1024


def _minimal_masks(values: np.ndarray, delta: float = 0.15) -> set[int]:
    sufficient = {i for i in range(len(values))
                  if values[i] - values[0] >= delta}
    return {m for m in sufficient
            if not any(t != m and (t & m) == t for t in sufficient)}


def experiment(seed: int, scenario: str, policy: str,
               packet_budget: int = 48, penalty: float = 0.006,
               support_cutoff: float = 0.03) -> dict:
    world, misspecification, replicates = scenario_world(seed, scenario)
    fit_edges = hyperedges(world.p, 3)
    design_seed = 40000 + 991 * seed + 17 * SCENARIOS.index(scenario)
    observation_seed = 50000 + 991 * seed + 17 * SCENARIOS.index(scenario)
    if policy == "adaptive_margin":
        out = propose_adaptively(world, fit_edges, packet_budget, replicates,
                                 seed=design_seed, penalty=penalty,
                                 cutoff=support_cutoff,
                                 recovery_shift=misspecification)
        packets, obs = out.packets, out.contrasts
    else:
        packets = select_packets(world.p, fit_edges, packet_budget,
                                 seed=design_seed, policy=policy)
        obs = sample_contrasts(world, packets, replicates,
                               seed=observation_seed,
                               recovery_shift=misspecification)
    X = feature_matrix(packets, fit_edges)
    estimates = group_lasso(X, obs, penalty=penalty)
    common = reconstruct_theta(estimates)
    true_support = world.shared_support()
    predicted_support = {e for e, value in zip(fit_edges, common)
                         if value > support_cutoff}
    support_scores = support_f1(true_support, predicted_support)

    train_packets = set(map(int, packets))
    holdout = np.asarray([m for m in range(1, 1 << world.p)
                          if m not in train_packets], dtype=int)
    fit_holdout = feature_matrix(holdout, fit_edges)
    true_holdout = world.probabilities(holdout)
    if misspecification is not None:
        edge, shift = misspecification
        mask = sum(1 << k for k in edge)
        true_holdout[:, 2] += shift * ((holdout & mask) == mask)
    true_oriented = (true_holdout - world.probabilities([0])) * SIGNS
    predicted_oriented = fit_holdout @ estimates
    rmse = float(np.sqrt(np.mean((true_oriented - predicted_oriented) ** 2)))

    # Correct fourth-channel coefficient mapping: B.T lambda = B4; it was
    # previously (incorrectly) solved as B lambda = B4.
    fourth_projection = np.linalg.solve(B.T, B4)
    fourth_pred = fit_holdout @ estimates @ fourth_projection
    fourth_truth = world.probabilities(holdout, fourth=True) - FOURTH_BASE
    fourth_mae = float(np.mean(np.abs(fourth_pred - fourth_truth)))

    all_masks = np.arange(1 << world.p)
    true_acquisition = world.probabilities(all_masks)[:, 0]
    reconstructed = feature_matrix(all_masks, fit_edges) @ estimates[:, 0]
    minimal_true = _minimal_masks(true_acquisition)
    minimal_pred = _minimal_masks(reconstructed)
    packet_scores = support_f1(minimal_true, minimal_pred)
    return {
        "seed": seed, "scenario": scenario, "policy": policy,
        "packets": packet_budget, "replicates_per_raw_arm": replicates,
        "raw_bernoulli_trials": 6 * packet_budget * replicates,
        "planted_shared_edges": len(true_support),
        "predicted_shared_edges": len(predicted_support),
        "true_order4_edges": sum(len(e) == 4 for e in true_support),
        "support_precision": support_scores["precision"],
        "support_recall": support_scores["recall"],
        "support_f1": support_scores["f1"],
        "minimal_packet_f1": packet_scores["f1"],
        "minimal_packet_truth_count": len(minimal_true),
        "minimal_packet_predicted_count": len(minimal_pred),
        "heldout_packet_count": len(holdout),
        "heldout_3channel_rmse": rmse,
        "heldout_4th_action_mae": fourth_mae,
    }


def selected_confirmation(seed: int, budget: int = 48,
                          replicates: int = 1024,
                          confirmation_replicates: int = 60000,
                          max_targets: int = 3) -> dict:
    """Sample-split confirmation of *discovered*, never oracle-chosen edges."""
    world = make_world(seed)
    edges = hyperedges(world.p, 3)
    chosen = select_packets(world.p, edges, budget,
                            seed=75000 + seed, policy="d_optimal")
    observations = sample_contrasts(world, chosen, replicates, seed=76000+seed)
    fit = group_lasso(feature_matrix(chosen, edges), observations, .006)
    theta = reconstruct_theta(fit)
    ranked = sorted((j for j, x in enumerate(theta) if x > .03),
                    key=lambda j: -theta[j])[:max_targets]
    # Audit a likely null edge too, selected solely from small *estimated* effect
    # (not from planted truth). Thus the confirmation records both discoveries
    # and a prospective negative-control decision.
    candidates = [j for j in range(len(edges)) if j not in ranked]
    negative = min(candidates, key=lambda j: abs(theta[j])) if candidates else None
    all_selected = ranked + ([negative] if negative is not None else [])
    records = [confirm_interaction(world, edges[j], confirmation_replicates,
                                   seed=77000 + seed*100+j, alpha=.05,
                                   n_targets=len(all_selected),
                                   raw_bias_bound=.00025) for j in all_selected]
    return {
        "seed": seed,
        "selected": len(ranked),
        "negative_controls": int(negative is not None),
        "negative_controls_certified_positive": int(negative is not None and
                                                       records[-1]["certified_positive"]),
        "negative_controls_true_zero": int(negative is not None and
                                             records[-1]["true_theta"] == 0),
        "total_tested": len(all_selected),
        "certified": sum(bool(r["certified_positive"]) for r in records[:len(ranked)]),
        "false_certifications": sum(bool(r["certified_positive"]) and
                                    r["true_theta"] <= 0 for r in records),
        "true_theta_covered": sum(r["lower"] <= r["true_theta"] <= r["upper"]
                                  for r in records),
        "confirmation_raw_trials": sum(r["total_simulated_bernoulli_trials"]
                                       for r in records),
        "selected_without_oracle": True,
    }


def fourth_action_falsification(seed: int, draws: int = 6000,
                               misspecification: float = 0.,
                               alpha: float = 0.05) -> dict:
    """Independent fixed-action diagnostic with *oracle-calibrated* B/w.

    This isolates the fourth-action test's statistical power, not the error of
    estimating the three-channel model. A real neural model needs an
    independently validated calibration and its uncertainty budget.
    """
    if draws < 1 or not 0 < alpha < 1:
        raise ValueError("invalid falsification experiment")
    world = make_world(seed)
    packet = (1 << world.p) - 1
    predicted = float(world.probabilities([packet], fourth=True)[0])
    truth = float(world.probabilities([packet], fourth=True,
                                       violation=misspecification)[0])
    if not 0 < truth < 1:
        raise ValueError("fourth-action perturbed probability invalid")
    rng = np.random.default_rng(88000+seed)
    observed = float(rng.binomial(draws, truth)/draws)
    radius = float(np.sqrt(np.log(2./alpha)/(2*draws)))
    return {"seed": seed, "violation": misspecification,
            "predicted": predicted, "observed": observed, "actual": truth,
            "radius": radius, "reject": abs(observed-predicted) > radius}


def run(out: Path, seeds: int = 12, packet_budget: int = 48,
        confirmation_replicates: int = 60000,
        fourth_trials: int = 6000) -> dict:
    if seeds < 2 or not 12 <= packet_budget <= 60:
        raise ValueError("invalid configuration")
    out.mkdir(parents=True, exist_ok=True)
    rows = [experiment(seed, scene, policy, packet_budget)
            for scene in SCENARIOS for seed in range(seeds)
            for policy in POLICIES]
    with (out / "stress_per_world.csv").open("w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = ("support_f1", "minimal_packet_f1", "heldout_3channel_rmse",
               "heldout_4th_action_mae", "support_precision", "support_recall")
    aggregate = {}
    for scene in SCENARIOS:
        aggregate[scene] = {}
        for pol in POLICIES:
            xs = [r for r in rows if r["scenario"] == scene and r["policy"] == pol]
            aggregate[scene][pol] = {
                metric: {"mean": float(np.mean([r[metric] for r in xs])),
                         "between_world_std": float(np.std([r[metric] for r in xs], ddof=1))}
                for metric in metrics}
    # Paired bootstrap resamples *worlds*, never millions of Bernoulli probes;
    # uncertainty about the finite suite is between independent planted worlds.
    bootstrap = {}
    bootstrap_rng = np.random.default_rng(20261119)
    for scene in SCENARIOS:
        per_seed = {
            pol: np.asarray([next(r["support_f1"] for r in rows
                                  if r["scenario"] == scene and r["policy"] == pol
                                  and r["seed"] == seed)
                             for seed in range(seeds)])
            for pol in POLICIES
        }
        bootstrap[scene] = {}
        for comparator in ("random", "adaptive_margin"):
            paired = per_seed["d_optimal"] - per_seed[comparator]
            indices = bootstrap_rng.integers(0, seeds, size=(10000, seeds))
            draws = paired[indices].mean(axis=1)
            bootstrap[scene][f"d_optimal_minus_{comparator}"] = {
                "mean_paired_delta_f1": float(paired.mean()),
                "bootstrap_world_ci_95": [float(np.quantile(draws, .025)),
                                          float(np.quantile(draws, .975))],
                "resampling_unit": "independent planted world seed"
            }
    confirm = [selected_confirmation(seed,
                                     budget=packet_budget,
                                     confirmation_replicates=confirmation_replicates)
               for seed in range(seeds)]
    falsification = {
        "calibrated_null": [fourth_action_falsification(seed, fourth_trials, 0.)
                            for seed in range(seeds)],
        "deliberate_violation_plus_0.08": [fourth_action_falsification(seed, fourth_trials, .08)
                                           for seed in range(seeds)]
    }
    summary = {
        "evidence_type": "EXECUTED CONTROLLED SYNTHETIC EXPERIMENTS; NO OPEN-WEIGHT MODELS",
        "seeds": seeds, "packet_budget": packet_budget,
        "scenarios": {"calibrated": "correctly specified planted sparse tri-channel response",
                      "low_replication": "96 draws/raw arm instead of 1024",
                      "weak_signal": "true shared-effect coefficients scaled by 0.35",
                      "recovery_interference": "unmodeled -0.06 recovery mean shift on a nuisance-only skill edge",
                      "unmodeled_four_way": "one real order-4 hyperedge omitted from order-3 fitting dictionary"},
        "policies": {"random": "uniform nonadaptive",
                     "d_optimal": "nonadaptive greedy D-optimal",
                     "adaptive_margin": "outcome-adaptive heuristic, 12 D-optimal pilot packets + margin-targeted design"},
        "aggregate": aggregate,
        "paired_world_bootstrap": bootstrap,
        "discovered_edge_confirmation": {
            "worlds": seeds,
            "selected": sum(r["selected"] for r in confirm),
            "negative_controls": sum(r["negative_controls"] for r in confirm),
            "negative_controls_true_zero": sum(r["negative_controls_true_zero"] for r in confirm),
            "negative_controls_certified_positive": sum(r["negative_controls_certified_positive"] for r in confirm),
            "total_tested": sum(r["total_tested"] for r in confirm),
            "certified_positive": sum(r["certified"] for r in confirm),
            "false_certifications_on_planted_truth": sum(r["false_certifications"] for r in confirm),
            "true_effects_covered": sum(r["true_theta_covered"] for r in confirm),
            "confirmation_raw_trials": sum(r["confirmation_raw_trials"] for r in confirm),
            "selection_uses_ground_truth": False,
            "alpha_familywise_per_world": .05,
            "max_tested_targets_per_world": 4,
            "systematic_raw_arm_bias_bound_assumed": .00025
        },
        "fourth_action_falsification": {
            label: {"worlds": seeds,
                    "rejections": sum(r["reject"] for r in vals),
                    "rejection_rate": float(np.mean([r["reject"] for r in vals])),
                    "draws_per_world": fourth_trials,
                    "note": "oracle-calibrated fourth-action mean, independent Bernoulli test"
                    }
            for label, vals in falsification.items()},
        "falsification_details": falsification,
        "confirmation_details": confirm,
        "warnings": [
            "The outcome-adaptive policy is exploratory, without optimality guarantees.",
            "All numeric results are from planted synthetic worlds. No LLM training, unlearning, or human evaluation was run.",
            "For the fourth-action oracle check, the calibration is known from the simulator; it is not estimated on neural data.",
            "Under intervention model misspecification, classical raw-data intervals may not cover the latent structural coefficient.",
            "The selected-edge confirmation has an independent sample and does not use oracle-selected true support.",
            "F1 and packet recovery here are measured using simulator-provided hidden ground truth."
        ]}
    with (out / "stress_summary.json").open("w", encoding="utf8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/robustness"))
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--packets", type=int, default=48)
    parser.add_argument("--confirmation-replicates", type=int, default=60000)
    parser.add_argument("--fourth-trials", type=int, default=6000)
    args = parser.parse_args()
    summary = run(args.out, seeds=args.seeds, packet_budget=args.packets,
                  confirmation_replicates=args.confirmation_replicates,
                  fourth_trials=args.fourth_trials)
    for scene in SCENARIOS:
        print(scene, {pol: round(summary['aggregate'][scene][pol]['support_f1']['mean'], 3)
                      for pol in POLICIES})
    print("confirmation", summary["discovered_edge_confirmation"])
    print("fourth", summary["fourth_action_falsification"])


if __name__ == "__main__":
    main()