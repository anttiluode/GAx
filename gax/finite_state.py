from __future__ import annotations

import numpy as np


def bitstrings(n_bits: int) -> np.ndarray:
    """Return all {0,1}^n bitstrings in lexicographic integer order."""
    n = 1 << n_bits
    ints = np.arange(n, dtype=np.uint64)[:, None]
    shifts = np.arange(n_bits - 1, -1, -1, dtype=np.uint64)
    return ((ints >> shifts) & 1).astype(float)


def hamming_mutation_matrix(n_bits: int, mu: float) -> np.ndarray:
    """Column-stochastic independent-bit mutation kernel M[y, x]."""
    if not 0.0 <= mu <= 1.0:
        raise ValueError("mu must lie in [0,1]")
    states = bitstrings(n_bits)
    d = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)
    M = (mu ** d) * ((1.0 - mu) ** (n_bits - d))
    M /= M.sum(axis=0, keepdims=True)
    return M


def bimodal_fitness(states: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """Two separated peaks with unequal height, returned as positive weights."""
    n_bits = states.shape[1]
    a = np.zeros(n_bits)
    b = np.ones(n_bits)
    da = np.abs(states - a).sum(axis=1)
    db = np.abs(states - b).sum(axis=1)
    score = np.maximum(1.0 - da / n_bits, 0.93 * (1.0 - db / n_bits))
    return np.exp(beta * score)


def positive_operator(M: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Feynman-Kac / mutation-selection operator L = M diag(weights)."""
    return M @ np.diag(weights)


def projectivize(q: np.ndarray) -> np.ndarray:
    z = float(q.sum())
    if z <= 0:
        raise ValueError("measure has non-positive total mass")
    return q / z


def mean_field_step(p: np.ndarray, L: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalized GA step p -> normalize(L p), plus its normalization constant."""
    q = L @ p
    z = float(q.sum())
    return q / z, z


def unnormalized_trajectory(p0: np.ndarray, L: np.ndarray, generations: int) -> np.ndarray:
    """q_g = L^g p0. Rows are generations."""
    qs = [np.asarray(p0, dtype=float)]
    q = qs[0]
    for _ in range(generations):
        q = L @ q
        qs.append(q)
    return np.stack(qs)


def normalized_trajectory(p0: np.ndarray, L: np.ndarray, generations: int) -> np.ndarray:
    ps = [projectivize(np.asarray(p0, dtype=float))]
    p = ps[0]
    for _ in range(generations):
        p, _ = mean_field_step(p, L)
        ps.append(p)
    return np.stack(ps)


def effective_rank(singular_values: np.ndarray, energy: float = 0.95) -> int:
    e = singular_values**2
    if e.sum() == 0:
        return 0
    return int(np.searchsorted(np.cumsum(e) / e.sum(), energy) + 1)
