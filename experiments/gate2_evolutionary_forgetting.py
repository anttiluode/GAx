from __future__ import annotations

import numpy as np

from gax.finite_state import bitstrings, hamming_mutation_matrix, positive_operator


def two_peak_weights(
    states: np.ndarray,
    *,
    beta: float = 2.0,
    second_height: float = 0.93,
) -> np.ndarray:
    """Positive fitness weights for two distant peaks with tunable near-degeneracy."""
    n_bits = states.shape[1]
    da = np.abs(states).sum(axis=1)
    db = np.abs(states - 1.0).sum(axis=1)
    score = np.maximum(
        1.0 - da / n_bits,
        second_height * (1.0 - db / n_bits),
    )
    return np.exp(beta * score)


def modal_forgetting_receipt(
    *,
    second_height: float,
    n_bits: int = 6,
    mu: float = 0.01,
    beta: float = 2.0,
    generations: int = 60,
    initial_subdominant_ratio: float = 0.25,
) -> dict[str, float | np.ndarray]:
    """Verify Sigh-style modal forgetting in a normalized mutation-selection GA.

    For symmetric Hamming mutation, L = M diag(G) is similar to the symmetric
    operator S = sqrt(G) M sqrt(G).  In the transformed coordinates

        r_g = sqrt(G) q_g,

    the dynamics are exactly r_{g+1} = S r_g.  Population normalization only
    rescales every mode equally, so the ratio of the second mode to the Perron
    mode must decay as (lambda_1 / lambda_0)^g.
    """
    states = bitstrings(n_bits)
    M = hamming_mutation_matrix(n_bits, mu)
    weights = two_peak_weights(states, beta=beta, second_height=second_height)
    L = positive_operator(M, weights)

    sqrt_w = np.sqrt(weights)
    S = sqrt_w[:, None] * M * sqrt_w[None, :]
    evals, evecs = np.linalg.eigh(S)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    # Orient the Perron vector positively.
    if evecs[:, 0].sum() < 0:
        evecs[:, 0] *= -1.0

    u0 = evecs[:, 0]
    u1 = evecs[:, 1]
    r0 = u0 + initial_subdominant_ratio * u1
    if np.min(r0) <= 0:
        raise RuntimeError("constructed positive initial population left the positive cone")

    q0 = r0 / sqrt_w
    p = q0 / q0.sum()

    measured = []
    for _ in range(generations + 1):
        r = sqrt_w * p
        amplitudes = evecs.T @ r
        measured.append(abs(amplitudes[1] / amplitudes[0]))

        q = L @ p
        p = q / q.sum()

    measured = np.asarray(measured)
    ratio = abs(evals[1] / evals[0])
    predicted = measured[0] * ratio ** np.arange(generations + 1)
    half_life = np.log(0.5) / np.log(ratio)

    return {
        "lambda0": float(evals[0]),
        "lambda1": float(evals[1]),
        "spectral_ratio": float(ratio),
        "half_life_generations": float(half_life),
        "initial_ratio": float(measured[0]),
        "max_prediction_error": float(np.max(np.abs(measured - predicted))),
        "measured": measured,
        "predicted": predicted,
    }


def main() -> None:
    ordinary = modal_forgetting_receipt(second_height=0.93)
    near_tie = modal_forgetting_receipt(second_height=0.99)

    print("Gate 2 — evolutionary forgetting times")
    print()
    for name, receipt in (("unequal peaks", ordinary), ("near-tied peaks", near_tie)):
        print(name)
        print(f"  lambda1/lambda0:       {receipt['spectral_ratio']:.12f}")
        print(f"  predicted half-life:   {receipt['half_life_generations']:.6f} generations")
        print(f"  initial modal ratio:   {receipt['initial_ratio']:.6f}")
        print(f"  max curve error:       {receipt['max_prediction_error']:.3e}")
        print()

    print(
        "Interpretation: population normalization turns the GA into a power-method-like "
        "mode purifier. Near-leading modes are not erased immediately; their spectral "
        "ratio assigns them a measurable evolutionary forgetting time."
    )


if __name__ == "__main__":
    main()
