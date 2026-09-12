import numpy as np

from experiments.gate3_strategy_separator import build_problem, run, switch_time
from gax.finite_state import mean_field_step


def test_context_queries_separate_specialists():
    L_a, L_b, L_reserve, near_a, near_b = build_problem()
    p0 = np.full(L_a.shape[0], 1.0 / L_a.shape[0])
    reserve = run(p0, L_reserve, 100)

    asked_a, _ = mean_field_step(reserve, L_a)
    asked_b, _ = mean_field_step(reserve, L_b)

    assert reserve[near_a].sum() > 0.45
    assert reserve[near_b].sum() > 0.45
    assert asked_a[near_a].sum() > 0.95
    assert asked_b[near_b].sum() > 0.95
    assert 0.5 * np.abs(asked_a - asked_b).sum() > 0.95


def test_purification_makes_context_switch_expensive():
    L_a, L_b, L_reserve, _, near_b = build_problem()
    p0 = np.full(L_a.shape[0], 1.0 / L_a.shape[0])
    reserve = run(p0, L_reserve, 100)

    fresh_switch = switch_time(reserve, L_b, near_b)
    purified = run(reserve, L_a, 10)
    late_switch = switch_time(purified, L_b, near_b)

    assert fresh_switch is not None and fresh_switch <= 2
    assert purified[near_b].sum() < 1e-10
    assert late_switch is not None and late_switch >= 8
    assert late_switch > fresh_switch
