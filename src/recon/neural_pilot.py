"""Small *trained PyTorch modular learners* for RECON intervention sanity checks.

This is not a language-model experiment: the chain-rule composition is hardcoded
and the two derivatives have known analytic supervision. The two MLP skill
modules genuinely train; reversible gating does NOT erase model parameters.

As a deliberately falsifiable negative control, restoring an activation mask
recovers the skill without any retraining, showing why lesion != unlearning.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # optional dependency installed only for this pilot
    torch = None
    nn = None


if nn is not None:
    class SkillNetwork(nn.Module):
        def __init__(self, width: int = 32):
            super().__init__()
            self.stack = nn.Sequential(nn.Linear(1, width), nn.Tanh(),
                                       nn.Linear(width, width), nn.Tanh(),
                                       nn.Linear(width, 1))

        def forward(self, x):
            return self.stack(x)


    class ModularDifferentiator(nn.Module):
        """Derivative of sin(x²) assembled from two learned skill modules.

        The composition g_square(x) * g_sine(x²) is a *built-in* prior. This
        bench tests factual acquisition/recovery and lesion confounds, not
        neural discovery of the chain rule or free-form skill prerequisites.
        """
        def __init__(self):
            super().__init__()
            self.square = SkillNetwork()
            self.sine = SkillNetwork()
            self.gate_square = 1.
            self.gate_sine = 1.

        def forward(self, x):
            return (self.gate_square * self.square(x) *
                    self.gate_sine * self.sine(x.square()))


def _train(module, x, target, steps, lr):
    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = (module(x) - target).square().mean()
        loss.backward()
        optimizer.step()


def one_seed(seed: int, steps: int = 225, lr: float = 0.008) -> dict:
    if torch is None:
        raise RuntimeError("install optional torch dependency: pip install -e '.[neural]'")
    if steps <= 0 or lr <= 0:
        raise ValueError("steps and lr must be positive")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 412)
    initial = ModularDifferentiator()
    x_a = torch.tensor(rng.uniform(-1., 1., size=(256, 1)), dtype=torch.float32)
    x_b = torch.tensor(rng.uniform(0., 1.5, size=(256, 1)), dtype=torch.float32)
    t_a = 2. * x_a
    t_b = torch.cos(x_b)
    held = torch.tensor(rng.uniform(-.95, .95, size=(1024, 1)), dtype=torch.float32)
    ood = torch.tensor(rng.uniform(1.05, 1.30, size=(1024, 1)), dtype=torch.float32)

    def score(model, x):
        with torch.no_grad():
            target = 2. * x * torch.cos(x.square())
            mse = (model(x) - target).square().mean().item()
            return 1. / (1. + mse)

    models = {}
    for packet in ("empty", "square", "sine", "both"):
        model = copy.deepcopy(initial)
        if packet in ("square", "both"):
            _train(model.square, x_a, t_a, steps, lr)
        if packet in ("sine", "both"):
            _train(model.sine, x_b, t_b, steps, lr)
        models[packet] = model

    trained = models["both"]
    sham = score(trained, held)
    reversibly_masked = copy.deepcopy(trained)
    reversibly_masked.gate_sine = 0.
    mask_score = score(reversibly_masked, held)
    reversibly_masked.gate_sine = 1.
    unmasked_without_update = score(reversibly_masked, held)

    destructively_perturbed = copy.deepcopy(trained)
    # Structural deletion of ONE learned output projection, not audited
    # knowledge erasure. Compare to sham and no-update restoration.
    with torch.no_grad():
        last = destructively_perturbed.sine.stack[-1]
        last.weight.zero_()
        last.bias.zero_()
    reset_score = score(destructively_perturbed, held)
    _train(destructively_perturbed.sine, x_b, t_b, steps, lr)
    relearn_score = score(destructively_perturbed, held)
    return {
        "seed": seed, "steps_per_exposed_skill": steps,
        "in_domain_empty_score": score(models["empty"], held),
        "in_domain_square_score": score(models["square"], held),
        "in_domain_sine_score": score(models["sine"], held),
        "in_domain_both_score": sham,
        "heldout_ood_both_score": score(trained, ood),
        "reversible_mask_score": mask_score,
        "mask_restored_zero_update_score": unmasked_without_update,
        "destructive_output_projection_reset_score": reset_score,
        "retrained_after_reset_score": relearn_score,
        "modules_trained_per_packet": {"empty": 0, "square": 1,
                                       "sine": 1, "both": 2},
    }


def run(out: Path, seeds: int = 8, steps: int = 225) -> dict:
    if torch is None:
        raise RuntimeError("PyTorch is required for neural pilot")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    rows = [one_seed(seed, steps=steps) for seed in range(seeds)]
    numerical = [k for k in rows[0] if k.endswith("_score")]
    with (out / "neural_module_per_seed.csv").open("w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["seed", "steps_per_exposed_skill"]+numerical)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    summary = {
        "experiment_type": "ACTUAL CPU PYTORCH MODULE TRAINING (NOT AN LLM)",
        "source_program": "analytic derivatives d/dx(sin(x^2)) with fixed chain rule composition",
        "model": "two separate 1-32-32-1 tanh MLP skill modules",
        "seeds": seeds, "training_steps_per_assigned_skill": steps,
        "score": "1/(1+mean squared error) on 1024 heldout x inputs per seed",
        "backend_torch_version": torch.__version__,
        "metrics": {k: {"mean": float(np.mean([r[k] for r in rows])),
                        "between_seed_std": float(np.std([r[k] for r in rows], ddof=1))
                        if seeds > 1 else None} for k in numerical},
        "limitations": [
            "This is a transparent modular neural learner, NOT a transformer, LLM, or demonstration of discovered neural circuits.",
            "Chain-rule composition is hardcoded; the modules only approximate two local derivatives.",
            "Conditions have different training-update budgets (0/1/1/2 modules), so their difference is descriptive, not a compute-matched effect estimate.",
            "Reversible mask is deliberately a zero-parameter-change functional lesion; effortless restoration proves only that the masked weights survived.",
            "Destructive output-projection zeroing is not an established machine-unlearning method.",
            "These observed skill dependencies are architecture/protocol-specific and are not general human prerequisites."
        ]}
    with (out / "neural_module_summary.json").open("w", encoding="utf8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('results/neural'))
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--steps', type=int, default=225)
    args = parser.parse_args()
    summary = run(args.out, seeds=args.seeds, steps=args.steps)
    for k, val in summary['metrics'].items():
        print(k, round(val['mean'], 4))
    print('saved', args.out)


if __name__ == '__main__':
    main()