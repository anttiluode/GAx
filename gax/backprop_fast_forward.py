from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class MLPShape:
    input_dim: int
    hidden_dim: int
    output_dim: int

    @property
    def n_params(self) -> int:
        return (
            self.input_dim * self.hidden_dim
            + self.hidden_dim
            + self.hidden_dim * self.output_dim
            + self.output_dim
        )


def make_regression_problem(n: int = 128) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, n, dtype=float)[:, None]
    y = 0.65 * np.sin(3.0 * x) + 0.25 * x + 0.08 * np.sin(9.0 * x)
    return x, y


def init_params(shape: MLPShape, seed: int = 0, scale: float = 0.2) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, scale, size=shape.n_params)


def unpack(theta: np.ndarray, shape: MLPShape):
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (shape.n_params,):
        raise ValueError(f"theta must have shape {(shape.n_params,)}")
    i = 0
    n = shape.input_dim * shape.hidden_dim
    W1 = theta[i : i + n].reshape(shape.input_dim, shape.hidden_dim); i += n
    b1 = theta[i : i + shape.hidden_dim]; i += shape.hidden_dim
    n = shape.hidden_dim * shape.output_dim
    W2 = theta[i : i + n].reshape(shape.hidden_dim, shape.output_dim); i += n
    b2 = theta[i : i + shape.output_dim]
    return W1, b1, W2, b2


def loss_only(theta: np.ndarray, shape: MLPShape, x: np.ndarray, y: np.ndarray) -> float:
    W1, b1, W2, b2 = unpack(theta, shape)
    pred = np.tanh(x @ W1 + b1) @ W2 + b2
    err = pred - y
    return 0.5 * float(np.mean(err * err))


def loss_and_grad(theta: np.ndarray, shape: MLPShape, x: np.ndarray, y: np.ndarray):
    W1, b1, W2, b2 = unpack(theta, shape)
    h = np.tanh(x @ W1 + b1)
    pred = h @ W2 + b2
    err = pred - y
    loss = 0.5 * float(np.mean(err * err))
    d_pred = err / err.size
    dW2 = h.T @ d_pred
    db2 = d_pred.sum(axis=0)
    dz1 = (d_pred @ W2.T) * (1.0 - h * h)
    dW1 = x.T @ dz1
    db1 = dz1.sum(axis=0)
    grad = np.concatenate([dW1.ravel(), db1, dW2.ravel(), db2])
    return loss, grad


@dataclass(frozen=True)
class ReducedAffineModel:
    center: np.ndarray
    basis: np.ndarray
    A: np.ndarray
    b: np.ndarray
    rank: int
    fit_rmse: float


def fit_reduced_affine(history: np.ndarray, rcond: float = 1e-10) -> ReducedAffineModel:
    hs = np.asarray(history, dtype=float)
    if hs.ndim != 2 or len(hs) < 3:
        raise ValueError("history must have shape (steps, parameters) with at least 3 steps")
    center = hs.mean(axis=0)
    centered = hs - center
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)
    rank = 1 if len(s) == 0 or s[0] == 0.0 else max(1, int(np.sum(s > rcond * s[0])))
    basis = Vt[:rank].T
    z = centered @ basis
    X = np.column_stack([z[:-1], np.ones(len(z) - 1)])
    coef, *_ = np.linalg.lstsq(X, z[1:], rcond=rcond)
    A, b = coef[:-1], coef[-1]
    fit_rmse = float(np.sqrt(np.mean((z[:-1] @ A + b - z[1:]) ** 2)))
    return ReducedAffineModel(center, basis, A, b, rank, fit_rmse)


def forecast_parameters(model: ReducedAffineModel, theta: np.ndarray, horizon: int) -> np.ndarray:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    z = (np.asarray(theta, dtype=float) - model.center) @ model.basis
    for _ in range(horizon):
        z = z @ model.A + model.b
    return model.center + z @ model.basis.T


class Objective(Protocol):
    backward_calls: int
    forward_only_calls: int
    def loss(self, theta: np.ndarray) -> float: ...
    def loss_grad(self, theta: np.ndarray): ...


@dataclass
class CountedObjective:
    shape: MLPShape
    x: np.ndarray
    y: np.ndarray
    backward_calls: int = 0
    forward_only_calls: int = 0

    def loss(self, theta: np.ndarray) -> float:
        self.forward_only_calls += 1
        return loss_only(theta, self.shape, self.x, self.y)

    def loss_grad(self, theta: np.ndarray):
        self.backward_calls += 1
        return loss_and_grad(theta, self.shape, self.x, self.y)


@dataclass
class MiniBatchObjective(CountedObjective):
    batch_size: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.batch_size > len(self.x):
            raise ValueError("batch_size must be in [1, n_samples]")
        self._rng = np.random.default_rng(self.seed)

    def loss_grad(self, theta: np.ndarray):
        self.backward_calls += 1
        idx = self._rng.choice(len(self.x), size=self.batch_size, replace=False)
        return loss_and_grad(theta, self.shape, self.x[idx], self.y[idx])


def gd_step(theta: np.ndarray, objective: Objective, lr: float):
    loss, grad = objective.loss_grad(theta)
    return np.asarray(theta, dtype=float) - lr * grad, loss


@dataclass(frozen=True)
class FastForwardAttempt:
    candidate_theta: np.ndarray
    candidate_loss: float
    horizon: int
    accepted: bool
    reason: str
    rank: int
    fit_ratio: float
    step_ratio: float


def attempt_fast_forward(
    history: np.ndarray,
    objective: Objective,
    horizon: int,
    *,
    current_loss: float | None = None,
    max_fit_ratio: float = 0.15,
    max_step_ratio: float = 3.0,
    min_relative_improvement: float = 0.0,
) -> FastForwardAttempt:
    hs = np.asarray(history, dtype=float)
    model = fit_reduced_affine(hs)
    candidate = forecast_parameters(model, hs[-1], horizon)
    candidate_loss = objective.loss(candidate)
    diffs = np.diff(hs, axis=0)
    typical_step = float(np.median(np.linalg.norm(diffs, axis=1)))
    fit_scale = float(np.sqrt(np.mean(diffs * diffs)))
    fit_ratio = model.fit_rmse / max(fit_scale, 1e-15)
    step_ratio = float(np.linalg.norm(candidate - hs[-1])) / max(horizon * typical_step, 1e-15)

    accepted, reason = True, "accepted"
    if not np.isfinite(candidate_loss) or not np.all(np.isfinite(candidate)):
        accepted, reason = False, "non-finite forecast"
    elif fit_ratio > max_fit_ratio:
        accepted, reason = False, "trajectory fit too poor"
    elif step_ratio > max_step_ratio:
        accepted, reason = False, "forecast jump too large"
    elif current_loss is not None and candidate_loss > current_loss * (1.0 - min_relative_improvement):
        accepted, reason = False, "forward loss check failed"
    return FastForwardAttempt(candidate, float(candidate_loss), horizon, accepted, reason, model.rank, float(fit_ratio), float(step_ratio))


@dataclass(frozen=True)
class TrainResult:
    theta: np.ndarray
    final_loss: float
    backward_calls: int
    forward_only_calls: int
    accepted_jumps: int = 0
    rejected_jumps: int = 0
    skipped_steps: int = 0
    attempted_jumps: int = 0
    losses: tuple[float, ...] = ()


def _train(
    theta0: np.ndarray,
    objective: Objective,
    *,
    lr: float,
    target_loss: float,
    max_backprops: int,
    jump_mode: str = "none",
    history_window: int = 10,
    horizons: tuple[int, ...] = (8, 4, 2),
    max_fit_ratio: float = 0.15,
    max_step_ratio: float = 3.0,
) -> TrainResult:
    theta = np.asarray(theta0, dtype=float).copy()
    current_loss = objective.loss(theta)
    losses, history = [current_loss], [theta.copy()]
    accepted = rejected = attempted = skipped = 0

    while current_loss > target_loss and objective.backward_calls < max_backprops:
        jumped = False
        if jump_mode != "none" and len(history) >= history_window:
            hs = np.stack(history[-history_window:])
            for horizon in horizons:
                attempted += 1
                if jump_mode == "operator":
                    att = attempt_fast_forward(
                        hs, objective, horizon,
                        current_loss=current_loss,
                        max_fit_ratio=max_fit_ratio,
                        max_step_ratio=max_step_ratio,
                    )
                    candidate, candidate_loss, ok = att.candidate_theta, att.candidate_loss, att.accepted
                elif jump_mode == "velocity":
                    candidate = hs[-1] + horizon * (hs[-1] - hs[-2])
                    candidate_loss = objective.loss(candidate)
                    ok = np.isfinite(candidate_loss) and candidate_loss <= current_loss
                else:
                    raise ValueError(f"unknown jump_mode {jump_mode!r}")
                if ok:
                    theta, current_loss = candidate.copy(), float(candidate_loss)
                    losses.append(current_loss)
                    accepted += 1
                    skipped += horizon
                    history = [theta.copy()]
                    jumped = True
                    break
                rejected += 1
        if jumped:
            continue
        theta, _ = gd_step(theta, objective, lr)
        current_loss = objective.loss(theta)
        losses.append(current_loss)
        history.append(theta.copy())
        if len(history) > history_window:
            history = history[-history_window:]

    return TrainResult(theta, float(current_loss), objective.backward_calls, objective.forward_only_calls,
                       accepted, rejected, skipped, attempted, tuple(float(v) for v in losses))


def train_plain_until_target(theta0, shape, x, y, *, lr, target_loss, max_backprops):
    return _train(theta0, CountedObjective(shape, x, y), lr=lr, target_loss=target_loss, max_backprops=max_backprops)


def train_with_fast_forward(theta0, shape, x, y, *, lr, target_loss, max_backprops,
                            history_window=10, horizons=(8, 4, 2), max_fit_ratio=0.15,
                            max_step_ratio=3.0, min_relative_improvement=0.0):
    del min_relative_improvement
    return _train(theta0, CountedObjective(shape, x, y), lr=lr, target_loss=target_loss,
                  max_backprops=max_backprops, jump_mode="operator", history_window=history_window,
                  horizons=horizons, max_fit_ratio=max_fit_ratio, max_step_ratio=max_step_ratio)


def train_with_velocity_fast_forward(theta0, shape, x, y, *, lr, target_loss, max_backprops,
                                     history_window=10, horizons=(8, 4, 2), min_relative_improvement=0.0):
    del min_relative_improvement
    return _train(theta0, CountedObjective(shape, x, y), lr=lr, target_loss=target_loss,
                  max_backprops=max_backprops, jump_mode="velocity", history_window=history_window,
                  horizons=horizons)


def train_plain_minibatch_until_target(theta0, shape, x, y, *, lr, target_loss, max_backprops,
                                       batch_size, batch_seed):
    obj = MiniBatchObjective(shape, x, y, batch_size=batch_size, seed=batch_seed)
    return _train(theta0, obj, lr=lr, target_loss=target_loss, max_backprops=max_backprops)


def train_minibatch_with_fast_forward(theta0, shape, x, y, *, lr, target_loss, max_backprops,
                                      batch_size, batch_seed, history_window=10, horizons=(8, 4, 2),
                                      max_fit_ratio=0.15, max_step_ratio=3.0,
                                      min_relative_improvement=0.0):
    del min_relative_improvement
    obj = MiniBatchObjective(shape, x, y, batch_size=batch_size, seed=batch_seed)
    return _train(theta0, obj, lr=lr, target_loss=target_loss, max_backprops=max_backprops,
                  jump_mode="operator", history_window=history_window, horizons=horizons,
                  max_fit_ratio=max_fit_ratio, max_step_ratio=max_step_ratio)


@dataclass(frozen=True)
class ForecastRecord:
    predicted_theta: np.ndarray
    predicted_loss: float
    horizon: int
    rank: int
    fit_ratio: float
    step_ratio: float
    history_length: int


@dataclass(frozen=True)
class ForecastAudit:
    actual_theta: np.ndarray
    actual_loss: float
    shadow_backward_calls: int
    weight_relative_error: float
    loss_absolute_error: float


def freeze_forecast(history: np.ndarray, objective: Objective, horizon: int) -> ForecastRecord:
    att = attempt_fast_forward(history, objective, horizon, max_fit_ratio=float("inf"), max_step_ratio=float("inf"))
    predicted = att.candidate_theta.copy(); predicted.setflags(write=False)
    return ForecastRecord(predicted, att.candidate_loss, horizon, att.rank, att.fit_ratio, att.step_ratio, len(history))


def audit_frozen_forecast(record: ForecastRecord, theta_start: np.ndarray, shape: MLPShape,
                          x: np.ndarray, y: np.ndarray, *, lr: float) -> ForecastAudit:
    shadow = CountedObjective(shape, x, y)
    theta = np.asarray(theta_start, dtype=float).copy()
    for _ in range(record.horizon):
        theta, _ = gd_step(theta, shadow, lr)
    actual_loss = loss_only(theta, shape, x, y)
    rel = float(np.linalg.norm(record.predicted_theta - theta) / max(np.linalg.norm(theta), 1e-15))
    return ForecastAudit(theta, float(actual_loss), shadow.backward_calls, rel,
                         abs(float(record.predicted_loss) - float(actual_loss)))
