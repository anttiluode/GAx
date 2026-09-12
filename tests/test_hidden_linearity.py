import numpy as np

from gax.finite_state import (
    bimodal_fitness,
    bitstrings,
    hamming_mutation_matrix,
    normalized_trajectory,
    positive_operator,
    projectivize,
    unnormalized_trajectory,
)


def test_mutation_columns_sum_to_one():
    M = hamming_mutation_matrix(5, 0.07)
    np.testing.assert_allclose(M.sum(axis=0), 1.0, atol=1e-12)


def test_normalized_ga_is_projectivized_linear_operator():
    states = bitstrings(5)
    M = hamming_mutation_matrix(5, 0.05)
    L = positive_operator(M, bimodal_fitness(states, beta=1.7))
    rng = np.random.default_rng(3)
    p0 = rng.random(len(states))
    p0 /= p0.sum()

    q = unnormalized_trajectory(p0, L, 12)
    p = normalized_trajectory(p0, L, 12)
    p_from_q = np.stack([projectivize(x) for x in q])
    np.testing.assert_allclose(p, p_from_q, atol=1e-12, rtol=1e-12)


def test_conjugacy_preserves_projective_dynamics():
    states = bitstrings(4)
    L = positive_operator(hamming_mutation_matrix(4, 0.06), bimodal_fitness(states, 1.4))
    rng = np.random.default_rng(9)
    perm = rng.permutation(len(states))
    P = np.eye(len(states))[perm]
    L2 = P @ L @ P.T
    p0 = rng.random(len(states))
    p0 /= p0.sum()
    a = normalized_trajectory(p0, L, 7)
    b = normalized_trajectory(P @ p0, L2, 7)
    np.testing.assert_allclose(b, a @ P.T, atol=1e-12, rtol=1e-12)
