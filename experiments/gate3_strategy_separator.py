from __future__ import annotations

import numpy as np

from gax.finite_state import (
    bitstrings,
    hamming_mutation_matrix,
    mean_field_step,
    positive_operator,
)


def run(p: np.ndarray, L: np.ndarray, generations: int) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    for _ in range(generations):
        p, _ = mean_field_step(p, L)
    return p


def switch_time(
    p: np.ndarray,
    L_target: np.ndarray,
    target_mask: np.ndarray,
    threshold: float = 0.8,
    max_generations: int = 100,
) -> int | None:
    q = np.asarray(p, dtype=float)
    for g in range(max_generations + 1):
        if float(q[target_mask].sum()) >= threshold:
            return g
        q, _ = mean_field_step(q, L_target)
    return None


def build_problem(n_bits: int = 8, mu: float = 0.005, beta: float = 5.0):
    states = bitstrings(n_bits)
    distance_a = states.sum(axis=1)
    distance_b = n_bits - distance_a

    score_a = 1.0 - distance_a / n_bits
    score_b = 1.0 - distance_b / n_bits

    M = hamming_mutation_matrix(n_bits, mu)
    L_a = positive_operator(M, np.exp(beta * score_a))
    L_b = positive_operator(M, np.exp(beta * score_b))

    # A context-free reserve rewards either specialist rather than their average.
    reserve_score = np.maximum(score_a, score_b)
    L_reserve = positive_operator(M, np.exp(beta * reserve_score))

    near_a = distance_a <= 1
    near_b = distance_b <= 1
    return L_a, L_b, L_reserve, near_a, near_b


def main() -> None:
    L_a, L_b, L_reserve, near_a, near_b = build_problem()
    n = L_a.shape[0]
    p0 = np.full(n, 1.0 / n)

    reserve = run(p0, L_reserve, 100)
    reserve_a = float(reserve[near_a].sum())
    reserve_b = float(reserve[near_b].sum())

    asked_a, _ = mean_field_step(reserve, L_a)
    asked_b, _ = mean_field_step(reserve, L_b)

    a_mass_after_a = float(asked_a[near_a].sum())
    b_mass_after_a = float(asked_a[near_b].sum())
    a_mass_after_b = float(asked_b[near_a].sum())
    b_mass_after_b = float(asked_b[near_b].sum())
    query_separation_tv = 0.5 * float(np.abs(asked_a - asked_b).sum())

    print("Gate 3 — strategy separator")
    print(f"reserve specialist mass A: {reserve_a:.6f}")
    print(f"reserve specialist mass B: {reserve_b:.6f}")
    print(f"after asking context A: A={a_mass_after_a:.6f}, B={b_mass_after_a:.6f}")
    print(f"after asking context B: A={a_mass_after_b:.6f}, B={b_mass_after_b:.6f}")
    print(f"A/B query separation (TV): {query_separation_tv:.6f}")
    print()
    print("purify on A -> cost to recover B")

    switch_receipt: dict[int, tuple[float, int | None]] = {}
    for generations_on_a in (0, 2, 5, 10, 20):
        purified = run(reserve, L_a, generations_on_a)
        retained_b = float(purified[near_b].sum())
        recovery = switch_time(purified, L_b, near_b)
        switch_receipt[generations_on_a] = (retained_b, recovery)
        print(
            f"A generations={generations_on_a:2d}: "
            f"retained B={retained_b:.3e}, B recovery={recovery} generations"
        )

    # The reserve contains two separated approaches and a context query amplifies
    # the matching one without averaging the two together.
    assert reserve_a > 0.45 and reserve_b > 0.45
    assert a_mass_after_a > 0.95 and b_mass_after_a < 0.02
    assert b_mass_after_b > 0.95 and a_mass_after_b < 0.02
    assert query_separation_tv > 0.95

    # Repeated use of one approach destroys cheap access to the other.
    assert switch_receipt[0][1] is not None and switch_receipt[0][1] <= 2
    assert switch_receipt[10][0] < 1e-10
    assert switch_receipt[10][1] is not None and switch_receipt[10][1] >= 8


if __name__ == "__main__":
    main()
