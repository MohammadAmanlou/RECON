"""Reproducible experiment runner. All numeric results are synthetic-world only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .core import (B, B4, confirm_interaction, exact_identification_witness,
                   feature_matrix, group_lasso, make_world, reconstruct_theta,
                   sample_contrasts, select_packets, support_f1)
from .scientific import verify_scientific_tasks


def single_run(seed: int, policy: str, packet_budget: int,
               replicates: int, penalty: float, support_cutoff: float) -> dict:
    world = make_world(seed)
    packets = select_packets(world.p, world.edges, packet_budget,
                             seed=seed + 3000, policy=policy)
    measurements = sample_contrasts(world, packets, replicates,
                                    seed=seed + 9000)
    X = feature_matrix(packets, world.edges)
    full = group_lasso(X, measurements, penalty=penalty)
    theta = reconstruct_theta(full)
    truth = {j for j, w in enumerate(world.w) if w[0] > 1e-12}
    predicted = {j for j, value in enumerate(theta) if value > support_cutoff}
    acq_only = group_lasso(X, measurements[:, :1], penalty=penalty)
    pred_acq = {j for j, value in enumerate(acq_only[:, 0])
                if value > support_cutoff}
    idx_pair = [j for j, edge in enumerate(world.edges) if len(edge) <= 2]
    pair_fit = group_lasso(X[:, idx_pair], measurements, penalty=penalty)
    pair_theta = reconstruct_theta(pair_fit)
    pred_pair = {idx_pair[j] for j, value in enumerate(pair_theta)
                 if value > support_cutoff}
    # Prospective prediction on ALL unseen action packets, not training repeats.
    all_packets = np.arange(1, 1 << world.p)
    heldout = np.asarray([m for m in all_packets if m not in set(packets)])
    Xh = feature_matrix(heldout, world.edges)
    true_means = Xh @ world.w @ B.T
    fitted_means = Xh @ full
    rmse = float(np.sqrt(np.mean((true_means - fitted_means) ** 2)))
    # A fourth action was not used for fitting; calibrated response predicts it.
    fourth_prediction = Xh @ full @ np.linalg.solve(B.T, B4)
    fourth_true = Xh @ world.w @ B4
    fourth_mae = float(np.mean(abs(fourth_prediction - fourth_true)))
    return {"world_seed": seed, "policy": policy, "packet_budget": packet_budget,
            "replicates_per_treated_or_control_arm": replicates,
            "simulated_bernoulli_trials": int(packet_budget * 3 * 2 * replicates),
            "heldout_packets": int(len(heldout)),
            "tri_f1": support_f1(truth, predicted)["f1"],
            "acquisition_only_f1": support_f1(truth, pred_acq)["f1"],
            "pairwise_tri_f1": support_f1(truth, pred_pair)["f1"],
            "tri_precision": support_f1(truth, predicted)["precision"],
            "tri_recall": support_f1(truth, predicted)["recall"],
            "heldout_three_channel_rmse": rmse,
            "heldout_fourth_action_mae": fourth_mae,
            "truth_edges": sorted([str(world.edges[j]) for j in truth]),
            "predicted_edges": sorted([str(world.edges[j]) for j in predicted])}


def run(out: Path, seeds: int = 12, packet_budget: int = 48,
        replicates: int = 1024, penalty: float = 0.006,
        support_cutoff: float = 0.03, confirmations: int = 3,
        confirm_replicates: int = 60000) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rows = [single_run(seed, policy, packet_budget, replicates,
                       penalty, support_cutoff)
            for seed in range(seeds)
            for policy in ("random", "d_optimal")]
    keys = [k for k in rows[0] if k not in ("truth_edges", "predicted_edges")]
    with (out / "per_world.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for item in rows:
            writer.writerow({k: item[k] for k in keys})
    aggregate = {}
    metric_keys = ("tri_f1", "acquisition_only_f1", "pairwise_tri_f1",
                   "tri_precision", "tri_recall", "heldout_three_channel_rmse",
                   "heldout_fourth_action_mae")
    for policy in ("random", "d_optimal"):
        subset = [r for r in rows if r["policy"] == policy]
        aggregate[policy] = {key: {"mean": float(np.mean([r[key] for r in subset])),
                                   "std_between_worlds": float(np.std([r[key] for r in subset], ddof=1))
                                   if len(subset) > 1 else None}
                             for key in metric_keys}
    # Independent confirmation data on preselected true support to audit coverage.
    # This is an *oracle-selection diagnostic*, NOT a discovery/selection success metric.
    world = make_world(0)
    selected = sorted(world.shared_support(), key=lambda edge: (-len(edge), edge))[:confirmations]
    certificates = [confirm_interaction(world, edge, replicates=confirm_replicates,
                                        seed=31000 + j, alpha=.05,
                                        n_targets=len(selected), raw_bias_bound=.00025)
                    for j, edge in enumerate(selected)]
    source_verification = verify_scientific_tasks(100)
    summary = {
        "experiment_type": "CALIBRATED SYNTHETIC WORLD ONLY; NOT LLM RESULTS",
        "seed_world_count": seeds,
        "packet_budget_per_world_policy": packet_budget,
        "replicates_per_treated_or_control_arm": replicates,
        "total_trials_per_policy_world": 6 * packet_budget * replicates,
        "penalty_group_lasso": penalty,
        "edge_support_threshold": support_cutoff,
        "methods": aggregate,
        "identification_witness": exact_identification_witness(),
        "preselected_oracle_confirmation_diagnostics": certificates,
        "scientific_executable_task_checks": source_verification,
        "limitations": [
            "The B loading matrix and the independently replicated Bernoulli block model are constructed, not learned from LLMs.",
            "D-optimal is outcome-nonadaptive and no claim is made of optimal sequential design.",
            "Fourth-action holdout is generated under the same calibrated loading matrix and is only a positive-control sanity check.",
            "Confirmation targets use oracle-selected edges to evaluate the finite-sample certificate; they are not discovery claims.",
            "The support-cutoff and regularization settings are demonstration choices, not preregistered hyperparameters for neural experiments.",
            "There is no open-weight model training, no mechanistic localization, and no human experiment here."
        ]}
    with (out / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        policies = ("random", "d_optimal")
        values = np.asarray([[aggregate[p][k]["mean"] for k in
                             ("acquisition_only_f1", "pairwise_tri_f1", "tri_f1")]
                             for p in policies])
        xs = np.arange(2)
        for i, label in enumerate(("Acquisition only", "Tri-channel pairwise", "Tri-channel higher-order")):
            ax.bar(xs + (i - 1) * .23, values[:, i], width=.21, label=label)
        ax.set_xticks(xs, ("Uniform random", "D-optimal (nonadaptive)"))
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Mean planted θ support F1 (synthetic worlds)")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_title("RECON-Ω pilot — calibrated synthetic-world results")
        fig.tight_layout()
        fig.savefig(out / "synthetic_support_f1.png", dpi=160)
        plt.close(fig)
        summary["plot_generated"] = True
    except ImportError:
        summary["plot_generated"] = False
    with (out / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--packets", type=int, default=48)
    parser.add_argument("--replicates", type=int, default=1024)
    parser.add_argument("--confirm-replicates", type=int, default=60000)
    args = parser.parse_args()
    summary = run(args.out, seeds=args.seeds, packet_budget=args.packets,
                  replicates=args.replicates,
                  confirm_replicates=args.confirm_replicates)
    for method, metrics in summary["methods"].items():
        print(method, {k: round(v["mean"], 4) for k, v in metrics.items()})
    print("Saved:", args.out.resolve())

if __name__ == "__main__":
    main()