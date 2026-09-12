from __future__ import annotations

import numpy as np

from gax.backprop_fast_forward import (
    CountedObjective,
    FastForwardAttempt,
    ForecastAudit,
    ForecastRecord,
    MLPShape,
    MiniBatchObjective,
    ReducedAffineModel,
    TrainResult,
    attempt_fast_forward,
    audit_frozen_forecast,
    fit_reduced_affine,
    forecast_parameters,
    freeze_forecast,
    gd_step,
    init_params,
    loss_and_grad,
    loss_only,
    make_regression_problem,
    train_minibatch_with_fast_forward,
    train_plain_minibatch_until_target,
    train_plain_until_target,
    train_with_fast_forward,
    train_with_velocity_fast_forward,
)


def run_gate6() -> dict[str, object]:
    x, y = make_regression_problem(n=96)
    shape = MLPShape(1, 12, 1)
    theta0 = init_params(shape, seed=11, scale=0.18)
    lr, target_loss = 1.0, 0.0045
    horizons, history_window = (16, 8, 4, 2), 10

    audit_objective = CountedObjective(shape, x, y)
    theta = theta0.copy()
    history = [theta.copy()]
    for _ in range(20):
        theta, _ = gd_step(theta, audit_objective, lr)
        history.append(theta.copy())
    record = freeze_forecast(np.stack(history[-history_window:]), audit_objective, 16)
    audit = audit_frozen_forecast(record, theta, shape, x, y, lr=lr)
    velocity_theta = theta + 16 * (history[-1] - history[-2])
    velocity_weight_error = float(
        np.linalg.norm(velocity_theta - audit.actual_theta) / max(np.linalg.norm(audit.actual_theta), 1e-15)
    )
    velocity_loss_error = abs(loss_only(velocity_theta, shape, x, y) - audit.actual_loss)

    common = dict(lr=lr, target_loss=target_loss, max_backprops=1000)
    plain = train_plain_until_target(theta0, shape, x, y, **common)
    operator = train_with_fast_forward(theta0, shape, x, y, history_window=history_window,
                                       horizons=horizons, **common)
    velocity = train_with_velocity_fast_forward(theta0, shape, x, y, history_window=history_window,
                                                horizons=horizons, **common)
    noisy_plain = train_plain_minibatch_until_target(theta0, shape, x, y, batch_size=6,
                                                     batch_seed=123, **common)
    noisy_operator = train_minibatch_with_fast_forward(theta0, shape, x, y, batch_size=6,
                                                       batch_seed=123, history_window=history_window,
                                                       horizons=horizons, **common)
    return {
        "shape": shape,
        "learning_rate": lr,
        "target_loss": target_loss,
        "history_window": history_window,
        "horizons": horizons,
        "audit": {
            "record": record,
            "actual": audit,
            "operator_weight_error": audit.weight_relative_error,
            "velocity_weight_error": velocity_weight_error,
            "operator_loss_error": audit.loss_absolute_error,
            "velocity_loss_error": float(velocity_loss_error),
        },
        "primary": {"plain": plain, "operator": operator, "velocity": velocity},
        "stochastic_attack": {"batch_size": 6, "plain": noisy_plain, "operator": noisy_operator},
    }


def _pct_saved(fast: TrainResult, plain: TrainResult) -> float:
    return 100.0 * (1.0 - fast.backward_calls / plain.backward_calls)


def main() -> None:
    report = run_gate6()
    primary, audit, noisy = report["primary"], report["audit"], report["stochastic_attack"]
    plain, operator, velocity = primary["plain"], primary["operator"], primary["velocity"]

    print("Gate 6 — guarded backprop fast-forward")
    print("forecast is frozen before shadow backprops are generated\n")
    print("chronological 16-step forecast audit")
    print(f"  reduced rank:                  {audit['record'].rank}")
    print(f"  operator relative weight err: {audit['operator_weight_error']:.3e}")
    print(f"  velocity relative weight err: {audit['velocity_weight_error']:.3e}")
    print(f"  operator absolute loss err:   {audit['operator_loss_error']:.3e}")
    print(f"  velocity absolute loss err:   {audit['velocity_loss_error']:.3e}\n")
    print("training to matched target loss")
    print("learner       backprops  forward-only  jumps  skipped  final loss")
    for name, result in (("plain", plain), ("operator", operator), ("velocity", velocity)):
        print(f"{name:10s} {result.backward_calls:9d} {result.forward_only_calls:13d} "
              f"{result.accepted_jumps:6d} {result.skipped_steps:8d} {result.final_loss:.6f}")
    print(f"operator backprop reduction: {_pct_saved(operator, plain):.1f}% "
          f"({plain.backward_calls} -> {operator.backward_calls})")
    print(f"velocity backprop reduction: {_pct_saved(velocity, plain):.1f}% "
          f"({plain.backward_calls} -> {velocity.backward_calls})\n")
    print("stochastic minibatch attacker (6 / 96 samples per backward)")
    print(f"plain:    {noisy['plain'].backward_calls} backprops, loss {noisy['plain'].final_loss:.6f}")
    print(f"operator: {noisy['operator'].backward_calls} backprops, loss {noisy['operator'].final_loss:.6f}, "
          f"{noisy['operator'].accepted_jumps} accepted jumps, {noisy['operator'].rejected_jumps} rejected")
    print(f"noisy backprop reduction: {_pct_saved(noisy['operator'], noisy['plain']):.1f}%\n")
    print("Claim boundary: this is a small NumPy MLP/full-batch control. The mini-batch attacker "
          "shows the advantage shrinking sharply; it is not a claim that arbitrary neural training can skip backprop.")

    assert plain.final_loss <= report["target_loss"]
    assert operator.final_loss <= report["target_loss"]
    assert operator.backward_calls < 0.5 * plain.backward_calls
    assert operator.backward_calls < velocity.backward_calls
    assert audit["operator_weight_error"] < 0.01 * audit["velocity_weight_error"]
    assert noisy["operator"].final_loss <= report["target_loss"]
    assert noisy["operator"].backward_calls / noisy["plain"].backward_calls > operator.backward_calls / plain.backward_calls


if __name__ == "__main__":
    main()
