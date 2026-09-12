import numpy as np


def test_manual_backprop_matches_finite_difference():
    try:
        from experiments.gate6_backprop_fast_forward import (
            MLPShape,
            init_params,
            loss_and_grad,
            make_regression_problem,
        )
    except ModuleNotFoundError:
        assert False, "Gate 6 module has not been implemented"

    x, y = make_regression_problem(n=16)
    shape = MLPShape(input_dim=1, hidden_dim=4, output_dim=1)
    theta = init_params(shape, seed=3, scale=0.25)
    loss, grad = loss_and_grad(theta, shape, x, y)

    eps = 1e-6
    probe = np.array([0, len(theta) // 3, len(theta) - 1])
    for idx in probe:
        plus = theta.copy(); plus[idx] += eps
        minus = theta.copy(); minus[idx] -= eps
        lp, _ = loss_and_grad(plus, shape, x, y)
        lm, _ = loss_and_grad(minus, shape, x, y)
        numeric = (lp - lm) / (2 * eps)
        assert np.isclose(grad[idx], numeric, rtol=2e-4, atol=2e-6)
    assert np.isfinite(loss)


def test_reduced_affine_forecaster_extrapolates_unseen_linear_steps():
    from experiments.gate6_backprop_fast_forward import (
        fit_reduced_affine,
        forecast_parameters,
    )

    A = np.array([[0.88, -0.16], [0.12, 0.91]])
    b = np.array([0.015, -0.01])
    xs = [np.array([0.7, -0.35])]
    for _ in range(10):
        xs.append(xs[-1] @ A + b)
    history = np.stack(xs[:7])
    model = fit_reduced_affine(history)
    predicted = forecast_parameters(model, history[-1], horizon=4)
    assert model.rank == 2
    assert np.linalg.norm(predicted - xs[10]) < 1e-8


def test_fast_forward_candidate_uses_forward_check_not_backward_pass():
    from experiments.gate6_backprop_fast_forward import (
        CountedObjective,
        MLPShape,
        attempt_fast_forward,
        gd_step,
        init_params,
        make_regression_problem,
    )

    x, y = make_regression_problem(n=48)
    shape = MLPShape(1, 8, 1)
    objective = CountedObjective(shape, x, y)
    theta = init_params(shape, seed=5, scale=0.2)
    history = [theta.copy()]
    for _ in range(9):
        theta, _ = gd_step(theta, objective, lr=0.08)
        history.append(theta.copy())

    backward_before = objective.backward_calls
    forward_only_before = objective.forward_only_calls
    attempt = attempt_fast_forward(
        np.stack(history),
        objective,
        horizon=2,
        max_fit_ratio=0.25,
        max_step_ratio=6.0,
    )

    assert objective.backward_calls == backward_before
    assert objective.forward_only_calls == forward_only_before + 1
    assert attempt.horizon == 2
    assert np.isfinite(attempt.candidate_loss)


def test_guarded_fast_forward_reaches_same_loss_with_fewer_backprops_on_smooth_full_batch():
    from experiments.gate6_backprop_fast_forward import (
        MLPShape,
        init_params,
        make_regression_problem,
        train_plain_until_target,
        train_with_fast_forward,
    )

    x, y = make_regression_problem(n=96)
    shape = MLPShape(1, 12, 1)
    theta0 = init_params(shape, seed=11, scale=0.18)
    target_loss = 0.0045

    plain = train_plain_until_target(
        theta0, shape, x, y, lr=0.30, target_loss=target_loss, max_backprops=1200
    )
    fast = train_with_fast_forward(
        theta0,
        shape,
        x,
        y,
        lr=0.30,
        target_loss=target_loss,
        max_backprops=1200,
        history_window=10,
        horizons=(8, 4, 2),
    )

    assert plain.final_loss <= target_loss
    assert fast.final_loss <= target_loss
    assert fast.accepted_jumps >= 1
    assert fast.backward_calls < plain.backward_calls


def test_forecast_is_frozen_before_shadow_backprops_are_run():
    from experiments.gate6_backprop_fast_forward import (
        CountedObjective,
        MLPShape,
        audit_frozen_forecast,
        freeze_forecast,
        gd_step,
        init_params,
        make_regression_problem,
    )

    x, y = make_regression_problem(n=64)
    shape = MLPShape(1, 8, 1)
    objective = CountedObjective(shape, x, y)
    theta = init_params(shape, seed=7, scale=0.2)
    history = [theta.copy()]
    for _ in range(10):
        theta, _ = gd_step(theta, objective, lr=0.2)
        history.append(theta.copy())

    before = objective.backward_calls
    record = freeze_forecast(np.stack(history), objective, horizon=4)
    frozen = record.predicted_theta.copy()
    assert objective.backward_calls == before

    audit = audit_frozen_forecast(record, theta, shape, x, y, lr=0.2)
    assert audit.shadow_backward_calls == 4
    assert np.array_equal(record.predicted_theta, frozen)
    assert np.isfinite(audit.actual_loss)
    assert np.isfinite(audit.weight_relative_error)


def test_operator_jump_beats_velocity_jump_on_curved_training_trajectory():
    from experiments.gate6_backprop_fast_forward import (
        MLPShape,
        init_params,
        make_regression_problem,
        train_plain_until_target,
        train_with_fast_forward,
        train_with_velocity_fast_forward,
    )

    x, y = make_regression_problem(n=96)
    shape = MLPShape(1, 12, 1)
    theta0 = init_params(shape, seed=11, scale=0.18)
    kwargs = dict(lr=1.0, target_loss=0.0045, max_backprops=1000)

    plain = train_plain_until_target(theta0, shape, x, y, **kwargs)
    operator = train_with_fast_forward(
        theta0, shape, x, y, history_window=10, horizons=(16, 8, 4, 2), **kwargs
    )
    velocity = train_with_velocity_fast_forward(
        theta0, shape, x, y, history_window=10, horizons=(16, 8, 4, 2), **kwargs
    )

    assert operator.final_loss <= kwargs["target_loss"]
    assert velocity.final_loss <= kwargs["target_loss"]
    assert operator.backward_calls < velocity.backward_calls < plain.backward_calls


def test_minibatch_objective_is_reproducible_and_counts_backprops():
    from experiments.gate6_backprop_fast_forward import (
        MLPShape,
        MiniBatchObjective,
        init_params,
        make_regression_problem,
    )

    x, y = make_regression_problem(n=80)
    shape = MLPShape(1, 6, 1)
    theta = init_params(shape, seed=2, scale=0.2)
    a = MiniBatchObjective(shape, x, y, batch_size=16, seed=123)
    b = MiniBatchObjective(shape, x, y, batch_size=16, seed=123)

    for _ in range(3):
        la, ga = a.loss_grad(theta)
        lb, gb = b.loss_grad(theta)
        assert np.isclose(la, lb)
        assert np.allclose(ga, gb)
    assert a.backward_calls == 3
    assert b.backward_calls == 3


def test_minibatch_fast_forward_run_is_reproducible():
    from experiments.gate6_backprop_fast_forward import (
        MLPShape,
        init_params,
        make_regression_problem,
        train_minibatch_with_fast_forward,
    )

    x, y = make_regression_problem(n=96)
    shape = MLPShape(1, 10, 1)
    theta0 = init_params(shape, seed=4, scale=0.18)
    kwargs = dict(
        lr=0.35,
        target_loss=0.02,
        max_backprops=300,
        batch_size=24,
        batch_seed=99,
        history_window=8,
        horizons=(8, 4, 2),
    )
    a = train_minibatch_with_fast_forward(theta0, shape, x, y, **kwargs)
    b = train_minibatch_with_fast_forward(theta0, shape, x, y, **kwargs)
    assert a.backward_calls == b.backward_calls
    assert a.accepted_jumps == b.accepted_jumps
    assert np.isclose(a.final_loss, b.final_loss)
    assert np.allclose(a.theta, b.theta)


def test_plain_minibatch_run_is_reproducible():
    from experiments.gate6_backprop_fast_forward import (
        MLPShape,
        init_params,
        make_regression_problem,
        train_plain_minibatch_until_target,
    )

    x, y = make_regression_problem(n=96)
    shape = MLPShape(1, 10, 1)
    theta0 = init_params(shape, seed=4, scale=0.18)
    kwargs = dict(
        lr=0.35,
        target_loss=0.02,
        max_backprops=300,
        batch_size=24,
        batch_seed=99,
    )
    a = train_plain_minibatch_until_target(theta0, shape, x, y, **kwargs)
    b = train_plain_minibatch_until_target(theta0, shape, x, y, **kwargs)
    assert a.backward_calls == b.backward_calls
    assert np.isclose(a.final_loss, b.final_loss)
    assert np.allclose(a.theta, b.theta)


def test_gate6_receipt_shows_real_backprop_savings_and_noise_attack():
    from experiments.gate6_backprop_fast_forward import run_gate6

    report = run_gate6()
    primary = report["primary"]
    audit = report["audit"]
    noisy = report["stochastic_attack"]

    assert primary["plain"].final_loss <= report["target_loss"]
    assert primary["operator"].final_loss <= report["target_loss"]
    assert primary["operator"].backward_calls < 0.5 * primary["plain"].backward_calls
    assert primary["operator"].backward_calls < primary["velocity"].backward_calls
    assert audit["operator_weight_error"] < 0.01 * audit["velocity_weight_error"]

    full_ratio = primary["operator"].backward_calls / primary["plain"].backward_calls
    noisy_ratio = noisy["operator"].backward_calls / noisy["plain"].backward_calls
    assert noisy["operator"].final_loss <= report["target_loss"]
    assert noisy_ratio > full_ratio
