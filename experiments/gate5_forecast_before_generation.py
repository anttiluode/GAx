from __future__ import annotations

import numpy as np

from gax.finite_state import (
    bimodal_fitness,
    bitstrings,
    hamming_mutation_matrix,
    mean_field_step,
    normalized_trajectory,
    positive_operator,
    projectivize,
)


def lift_normalized_history(
    p_history: np.ndarray,
    normalizers: np.ndarray,
) -> np.ndarray:
    """Recover q_g from observed p_g and past normalization factors.

    If p_{g+1} = L p_g / z_g and q_0 = p_0, then
    q_g = (prod_{i < g} z_i) p_g.  The caller supplies only factors that
    were already observed before the forecast origin.
    """
    ps = np.asarray(p_history, dtype=float)
    zs = np.asarray(normalizers, dtype=float)
    if ps.ndim != 2:
        raise ValueError("p_history must have shape (generations, states)")
    if zs.shape != (len(ps) - 1,):
        raise ValueError("need exactly one observed normalizer per transition")
    if np.any(zs <= 0.0):
        raise ValueError("normalizers must be positive")

    masses = np.ones(len(ps), dtype=float)
    if len(ps) > 1:
        masses[1:] = np.cumprod(zs)
    return ps * masses[:, None]


def fit_history_operator(q_history: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Minimum-norm linear map supported by the observed Krylov history."""
    qs = np.asarray(q_history, dtype=float)
    if qs.ndim != 2 or len(qs) < 2:
        raise ValueError("q_history must contain at least two generations")
    X = qs[:-1].T
    Y = qs[1:].T
    return Y @ np.linalg.pinv(X, rcond=rcond)


def forecast_projective(
    p_history: np.ndarray,
    normalizers: np.ndarray,
    horizon: int,
    rcond: float = 1e-12,
) -> np.ndarray:
    """Forecast an unseen normalized population using history only."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    qs = lift_normalized_history(p_history, normalizers)
    fitted = fit_history_operator(qs, rcond=rcond)
    q = qs[-1].copy()
    for _ in range(horizon):
        q = fitted @ q
    return projectivize(q)


def persistence_forecast(p_history: np.ndarray) -> np.ndarray:
    return np.asarray(p_history[-1], dtype=float).copy()


def velocity_forecast(p_history: np.ndarray, horizon: int) -> np.ndarray:
    ps = np.asarray(p_history, dtype=float)
    if len(ps) < 2:
        return persistence_forecast(ps)
    q = ps[-1] + horizon * (ps[-1] - ps[-2])
    q = np.clip(q, 0.0, None)
    return projectivize(q)


def oracle_forecast(p: np.ndarray, L: np.ndarray, horizon: int) -> np.ndarray:
    q = np.asarray(p, dtype=float).copy()
    for _ in range(horizon):
        q, _ = mean_field_step(q, L)
    return q


def observed_trajectory_with_normalizers(
    p0: np.ndarray,
    L: np.ndarray,
    generations: int,
) -> tuple[np.ndarray, np.ndarray]:
    ps = [projectivize(np.asarray(p0, dtype=float))]
    zs: list[float] = []
    p = ps[0]
    for _ in range(generations):
        p, z = mean_field_step(p, L)
        ps.append(p)
        zs.append(z)
    return np.stack(ps), np.asarray(zs)


def l1_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).sum())


def main() -> None:
    n_bits = 7
    mu = 0.035
    beta = 2.0
    states = bitstrings(n_bits)
    M = hamming_mutation_matrix(n_bits, mu)
    L = positive_operator(M, bimodal_fitness(states, beta=beta))
    p0 = np.full(len(states), 1.0 / len(states))

    history_lengths = (2, 3, 4, 5, 6, 8)
    horizons = (1, 3, 5)
    max_generation = max(history_lengths) + max(horizons)

    # L is used here only to create the hidden ground-truth process.  The
    # history fit below receives only already-observed p_g and z_g.
    truth, normalizers = observed_trajectory_with_normalizers(
        p0, L, generations=max_generation
    )

    weaker_peak = (n_bits - states.sum(axis=1)) <= 1
    receipts: dict[tuple[int, int], dict[str, float]] = {}

    print("Gate 5 — forecast before generation")
    print("fit uses only past populations + already-observed normalization factors")
    print()
    print("history  rank  horizon    fitted L1    velocity L1  persistence L1")

    for m in history_lengths:
        observed_ps = truth[: m + 1]
        observed_zs = normalizers[:m]
        q_history = lift_normalized_history(observed_ps, observed_zs)
        rank = int(np.linalg.matrix_rank(q_history[:-1].T))

        for h in horizons:
            actual = truth[m + h]
            fitted = forecast_projective(observed_ps, observed_zs, horizon=h)
            velocity = velocity_forecast(observed_ps, horizon=h)
            persistence = persistence_forecast(observed_ps)
            oracle = oracle_forecast(observed_ps[-1], L, horizon=h)

            receipt = {
                "rank": float(rank),
                "fitted": l1_error(fitted, actual),
                "velocity": l1_error(velocity, actual),
                "persistence": l1_error(persistence, actual),
                "oracle": l1_error(oracle, actual),
                "weak_actual": float(actual[weaker_peak].sum()),
                "weak_predicted": float(fitted[weaker_peak].sum()),
            }
            receipts[(m, h)] = receipt
            print(
                f"{m:7d} {rank:5d} {h:8d} "
                f"{receipt['fitted']:12.3e} "
                f"{receipt['velocity']:12.3e} "
                f"{receipt['persistence']:14.3e}"
            )

    early = receipts[(2, 3)]
    middle = receipts[(4, 3)]
    saturated = receipts[(8, 5)]

    print()
    print("weak-branch forecast at history=8, horizon=5")
    print(f"actual mass:    {saturated['weak_actual']:.12f}")
    print(f"predicted mass: {saturated['weak_predicted']:.12f}")
    print(
        "Interpretation: the generational history becomes predictive as its "
        "Krylov span grows; after the accessible rank saturates, an unseen "
        "future generation is forecast almost exactly without consulting L."
    )

    # The history fit should already compete with a velocity extrapolator at
    # short history, improve sharply as new Krylov directions arrive, then be
    # essentially exact once the accessible cyclic subspace has been observed.
    assert early["fitted"] < early["velocity"]
    assert middle["fitted"] < 0.1 * middle["velocity"]
    assert saturated["fitted"] < 1e-8
    assert saturated["fitted"] < 1e-6 * saturated["velocity"]
    assert abs(saturated["weak_predicted"] - saturated["weak_actual"]) < 1e-8
    assert saturated["oracle"] < 1e-12


if __name__ == "__main__":
    main()
