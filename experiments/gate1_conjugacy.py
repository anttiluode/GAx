from __future__ import annotations

import numpy as np

from gax.finite_state import (
    bimodal_fitness,
    bitstrings,
    hamming_mutation_matrix,
    normalized_trajectory,
    positive_operator,
)


def main() -> None:
    n_bits = 6
    states = bitstrings(n_bits)
    L = positive_operator(
        hamming_mutation_matrix(n_bits, 0.04),
        bimodal_fitness(states, beta=2.0),
    )

    rng = np.random.default_rng(11)
    perm = rng.permutation(len(states))
    P = np.eye(len(states))[perm]
    L2 = P @ L @ P.T

    p0 = rng.random(len(states))
    p0 /= p0.sum()
    p0_2 = P @ p0

    hist1 = normalized_trajectory(p0, L, 18)
    hist2 = normalized_trajectory(p0_2, L2, 18)
    equivariance_error = float(np.max(np.abs(hist2 - hist1 @ P.T)))

    ev1 = np.sort_complex(np.linalg.eigvals(L))
    ev2 = np.sort_complex(np.linalg.eigvals(L2))
    spectrum_error = float(np.max(np.abs(ev1 - ev2)))

    vals, vecs = np.linalg.eig(L)
    i = int(np.argmax(np.abs(vals)))
    v1 = np.real(vecs[:, i])
    vals2, vecs2 = np.linalg.eig(L2)
    j = int(np.argmax(np.abs(vals2)))
    v2 = np.real(vecs2[:, j])
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    aligned = abs(float((P @ v1) @ v2))
    raw = abs(float(v1 @ v2))

    print("GAx Gate 1 — transfer by conjugacy, not raw vectors")
    print(f"trajectory equivariance max error: {equivariance_error:.3e}")
    print(f"spectrum max error: {spectrum_error:.3e}")
    print(f"leading-vector cosine after applying coordinate map: {aligned:.6f}")
    print(f"leading-vector cosine without coordinate map: {raw:.6f}")

    assert equivariance_error < 1e-12
    assert spectrum_error < 1e-10
    assert aligned > 1 - 1e-10


if __name__ == "__main__":
    main()
