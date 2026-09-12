from __future__ import annotations

import numpy as np

from experiments.gate5_forecast_before_generation import (
    fit_history_operator,
    forecast_projective,
    lift_normalized_history,
)
from gax.finite_state import mean_field_step, normalized_trajectory, unnormalized_trajectory


def test_lift_recovers_hidden_linear_history_from_observed_normalizers() -> None:
    L = np.diag([2.0, 1.2, 0.7])
    p0 = np.array([0.2, 0.3, 0.5])

    ps = [p0 / p0.sum()]
    normalizers: list[float] = []
    p = ps[0]
    for _ in range(5):
        p, z = mean_field_step(p, L)
        ps.append(p)
        normalizers.append(z)

    lifted = lift_normalized_history(np.stack(ps), np.array(normalizers))
    truth = unnormalized_trajectory(p0 / p0.sum(), L, generations=5)

    assert np.max(np.abs(lifted - truth)) < 1e-12


def test_history_fitted_operator_forecasts_unseen_generation() -> None:
    L = np.diag([1.8, 1.25, 0.82])
    p0 = np.array([0.31, 0.27, 0.42])

    observed_generations = 4
    full_generations = observed_generations + 3
    ps = normalized_trajectory(p0, L, generations=full_generations)

    observed_ps = ps[: observed_generations + 1]
    normalizers: list[float] = []
    p = ps[0]
    for _ in range(observed_generations):
        p, z = mean_field_step(p, L)
        normalizers.append(z)

    q_history = lift_normalized_history(observed_ps, np.array(normalizers))
    fitted = fit_history_operator(q_history)
    predicted = forecast_projective(
        observed_ps,
        np.array(normalizers),
        horizon=3,
    )

    assert np.linalg.matrix_rank(q_history[:-1].T) == 3
    assert np.max(np.abs(fitted @ q_history[-2] - q_history[-1])) < 1e-12
    assert np.max(np.abs(predicted - ps[observed_generations + 3])) < 1e-10


def test_forecast_uses_history_not_hidden_oracle_operator() -> None:
    # Two different full operators with identical action on the observed cyclic
    # subspace must produce the same history-fit.  The forecast is allowed to
    # know only the observed generations and normalizers, not either oracle L.
    q_history = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.64, 0.32, 0.0],
        ]
    )
    fitted = fit_history_operator(q_history)

    assert np.allclose(fitted[:, 2], 0.0)
