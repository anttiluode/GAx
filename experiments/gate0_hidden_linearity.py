from __future__ import annotations

import numpy as np

from gax.finite_state import (
    bimodal_fitness,
    bitstrings,
    effective_rank,
    hamming_mutation_matrix,
    mean_field_step,
    normalized_trajectory,
    positive_operator,
    projectivize,
    unnormalized_trajectory,
)


def main() -> None:
    n_bits = 7
    mu = 0.035
    generations = 30

    states = bitstrings(n_bits)
    M = hamming_mutation_matrix(n_bits, mu)
    w = bimodal_fitness(states, beta=2.4)
    L = positive_operator(M, w)

    rng = np.random.default_rng(7)
    p0 = rng.random(len(states))
    p0 /= p0.sum()

    q_hist = unnormalized_trajectory(p0, L, generations)
    p_hist = normalized_trajectory(p0, L, generations)

    lifted = np.stack([projectivize(q) for q in q_hist])
    projective_error = float(np.max(np.abs(lifted - p_hist)))

    one_step, z = mean_field_step(p_hist[8], L)
    direct_error = float(np.max(np.abs(one_step - p_hist[9])))

    centered = q_hist[1:] / np.linalg.norm(q_hist[1:], axis=1, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    r95 = effective_rank(s, 0.95)

    vals, vecs = np.linalg.eig(L)
    idx = int(np.argmax(np.abs(vals)))
    v = np.real(vecs[:, idx])
    if v.sum() < 0:
        v = -v
    v = np.maximum(v, 0)
    v = projectivize(v)
    final_l1 = float(np.abs(p_hist[-1] - v).sum())

    print("GAx Gate 0 — hidden linearity of selection + mutation")
    print(f"states: {len(states)}  bits: {n_bits}  mu: {mu}")
    print(f"projective lift max error: {projective_error:.3e}")
    print(f"direct one-step max error: {direct_error:.3e}")
    print(f"normalization constant at g=8: {z:.6f}")
    print(f"95% effective rank of normalized Krylov history: {r95}")
    print(f"L1 distance to Perron population after {generations} generations: {final_l1:.6f}")

    assert projective_error < 1e-12
    assert direct_error < 1e-12


if __name__ == "__main__":
    main()
