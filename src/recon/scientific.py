"""Executable scientific micro-domain: reverse-mode differentiation on DAGs.

Ground truth is calculated two independent ways (reverse AD / forward duals)
and stress-tested with central finite differences. This is an exact task-world
compiler, not an LLM or an extraction system for arbitrary textbooks.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Node:
    op: str
    left: int
    right: int


@dataclass(frozen=True)
class Program:
    nodes: tuple[Node, ...]
    inputs: int = 3

    def _forward(self, x: np.ndarray) -> np.ndarray:
        if len(x) != self.inputs:
            raise ValueError("wrong number of inputs")
        values = list(map(float, x))
        for node in self.nodes:
            u, v = values[node.left], values[node.right]
            values.append(u + v if node.op == "add" else u * v)
        return np.asarray(values)

    def evaluate(self, x: np.ndarray) -> float:
        return float(self._forward(np.asarray(x))[-1])

    def reverse_gradient(self, x: np.ndarray) -> np.ndarray:
        """Shared DAG nodes accumulate adjoints from *all* downstream paths."""
        values = self._forward(np.asarray(x))
        grad = np.zeros(len(values)); grad[-1] = 1.
        for index in reversed(range(len(self.nodes))):
            node = self.nodes[index]
            upstream = grad[self.inputs + index]
            if node.op == "add":
                grad[node.left] += upstream
                grad[node.right] += upstream
            elif node.op == "mul":
                grad[node.left] += upstream * values[node.right]
                grad[node.right] += upstream * values[node.left]
            else:
                raise ValueError("unknown op")
        return grad[:self.inputs]

    def forward_gradient(self, x: np.ndarray) -> np.ndarray:
        """Independent forward-mode dual-number oracle."""
        x = np.asarray(x, dtype=float)
        vals = list(x)
        duals = list(np.eye(self.inputs))
        for node in self.nodes:
            a, b = node.left, node.right
            if node.op == "add":
                vals.append(vals[a] + vals[b]); duals.append(duals[a] + duals[b])
            elif node.op == "mul":
                vals.append(vals[a] * vals[b]); duals.append(duals[a] * vals[b] + duals[b] * vals[a])
            else:
                raise ValueError("unknown op")
        return np.asarray(duals[-1])

    def central_difference(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        grad = np.empty(self.inputs)
        for j in range(self.inputs):
            xp = x.copy(); xm = x.copy(); xp[j] += eps; xm[j] -= eps
            grad[j] = (self.evaluate(xp) - self.evaluate(xm)) / (2. * eps)
        return grad


def generate_dag(seed: int, inputs: int = 3, n_ops: int = 6) -> Program:
    """Guarantee a reused intermediate to probe adjoint accumulation."""
    if inputs < 2 or n_ops < 3:
        raise ValueError("requires inputs>=2 and operations>=3")
    rng = np.random.default_rng(seed)
    nodes = [Node("mul", 0, 1), Node("add", inputs, 2 if inputs > 2 else 0),
             Node("mul", inputs, inputs + 1)]
    for i in range(3, n_ops):
        available = inputs + i
        a = int(rng.integers(0, available))
        b = int(rng.integers(0, available))
        nodes.append(Node("mul" if rng.random() < 0.35 else "add", a, b))
    return Program(tuple(nodes), inputs)


def verify_scientific_tasks(seeds: int = 100) -> dict[str, float | int]:
    max_forward = 0.; max_finite = 0.
    for seed in range(seeds):
        p = generate_dag(seed)
        x = np.random.default_rng(1301 + seed).uniform(-0.6, 0.6, p.inputs)
        backward = p.reverse_gradient(x)
        forward = p.forward_gradient(x)
        finite = p.central_difference(x)
        max_forward = max(max_forward, float(np.max(abs(backward - forward))))
        max_finite = max(max_finite, float(np.max(abs(backward - finite))))
        if not np.allclose(backward, forward, atol=1e-9, rtol=1e-9):
            raise AssertionError("reverse/forward verifier disagreement")
        if not np.allclose(backward, finite, atol=1e-4, rtol=1e-4):
            raise AssertionError("reverse/finite-difference disagreement")
    return {"tasks_checked": seeds, "max_reverse_forward_abs_error": max_forward,
            "max_reverse_finite_diff_abs_error": max_finite}