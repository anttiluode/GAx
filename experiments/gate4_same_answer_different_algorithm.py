from __future__ import annotations

import numpy as np

from gax.finite_state import mean_field_step, positive_operator


def correlated_world() -> tuple[np.ndarray, np.ndarray]:
    """World where the direct cue and relational cue are perfectly correlated."""
    rows = []
    for x1 in (-1.0, 1.0):
        for x2 in (-1.0, 1.0):
            x0 = x1 * x2
            rows.append((x0, x1, x2))
    X = np.asarray(rows)
    return X, X[:, 0]


def intervention_worlds() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Break the correlation; A rewards the direct cue, B the relational cue."""
    rows = []
    for x0 in (-1.0, 1.0):
        for x1 in (-1.0, 1.0):
            for x2 in (-1.0, 1.0):
                rows.append((x0, x1, x2))
    X = np.asarray(rows)
    target_a = X[:, 0]
    target_b = X[:, 1] * X[:, 2]
    return X, target_a, target_b


def predict(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Tiny two-path neural policy: direct path + relational path."""
    w_direct, w_relational = theta
    preactivation = w_direct * X[:, 0] + w_relational * (X[:, 1] * X[:, 2])
    return np.tanh(preactivation)


def mse(theta: np.ndarray, X: np.ndarray, target: np.ndarray) -> float:
    y = predict(theta, X)
    return float(np.mean((y - target) ** 2))


def jacobian_signature(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Average output sensitivity carried by direct vs relational input paths."""
    w_direct, w_relational = theta
    y = predict(theta, X)
    local_gain = 1.0 - y**2

    # dy/dx0 is the direct path. The relational path is represented by
    # sensitivities to x1 and x2, averaged because |x1|=|x2|=1 here.
    direct = float(np.mean(np.abs(local_gain * w_direct)))
    rel_x1 = float(np.mean(np.abs(local_gain * w_relational * X[:, 2])))
    rel_x2 = float(np.mean(np.abs(local_gain * w_relational * X[:, 1])))
    relational = 0.5 * (rel_x1 + rel_x2)
    return np.asarray([direct, relational])


def one_dim_mutation(n: int, mu: float) -> np.ndarray:
    """Nearest-neighbour mutation on one weight coordinate, with reflection."""
    T = np.zeros((n, n))
    for src in range(n):
        T[src, src] += 1.0 - 2.0 * mu
        if src > 0:
            T[src - 1, src] += mu
        else:
            T[src, src] += mu
        if src + 1 < n:
            T[src + 1, src] += mu
        else:
            T[src, src] += mu
    return T


def run(p: np.ndarray, L: np.ndarray, generations: int) -> np.ndarray:
    q = np.asarray(p, dtype=float)
    for _ in range(generations):
        q, _ = mean_field_step(q, L)
    return q


def angle_degrees(signature: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(signature[1], signature[0])))


def main() -> None:
    X_seen, y_seen = correlated_world()
    X, y_a, y_b = intervention_worlds()

    grid = np.linspace(0.0, 4.0, 9)
    params = np.asarray([(wd, wr) for wd in grid for wr in grid])

    errors_a = np.asarray([mse(theta, X, y_a) for theta in params])
    errors_b = np.asarray([mse(theta, X, y_b) for theta in params])

    beta = 8.0
    fit_a = np.exp(-beta * errors_a)
    fit_b = np.exp(-beta * errors_b)

    # Mutation is local in parameter space. The reserve rewards either successful
    # computation instead of forcing their parameter average to become the answer.
    T = one_dim_mutation(len(grid), mu=0.02)
    M = np.kron(T, T)
    L_a = positive_operator(M, fit_a)
    L_b = positive_operator(M, fit_b)
    L_reserve = positive_operator(M, np.maximum(fit_a, fit_b))

    p0 = np.full(len(params), 1.0 / len(params))
    reserve = run(p0, L_reserve, 100)
    asked_a, _ = mean_field_step(reserve, L_a)
    asked_b, _ = mean_field_step(reserve, L_b)

    # A computational approach is defined by which path dominates, not merely by
    # where the weights sit numerically.
    direct_family = params[:, 0] >= params[:, 1] + 1.0
    relational_family = params[:, 1] >= params[:, 0] + 1.0

    reserve_direct = float(reserve[direct_family].sum())
    reserve_relational = float(reserve[relational_family].sum())
    a_direct = float(asked_a[direct_family].sum())
    a_relational = float(asked_a[relational_family].sum())
    b_direct = float(asked_b[direct_family].sum())
    b_relational = float(asked_b[relational_family].sum())
    query_tv = 0.5 * float(np.abs(asked_a - asked_b).sum())

    signatures = np.asarray([jacobian_signature(theta, X) for theta in params])
    reserve_angle = angle_degrees(reserve @ signatures)
    a_angle = angle_degrees(asked_a @ signatures)
    b_angle = angle_degrees(asked_b @ signatures)

    # Two extreme specialists are behaviorally identical in the correlated world,
    # but intervention reveals that they compute the answer differently.
    direct_specialist = np.asarray([4.0, 0.0])
    relational_specialist = np.asarray([0.0, 4.0])
    same_world_difference = float(
        np.max(np.abs(predict(direct_specialist, X_seen) - predict(relational_specialist, X_seen)))
    )

    direct_seen_error = mse(direct_specialist, X_seen, y_seen)
    relational_seen_error = mse(relational_specialist, X_seen, y_seen)
    direct_a_error = mse(direct_specialist, X, y_a)
    direct_b_error = mse(direct_specialist, X, y_b)
    relational_a_error = mse(relational_specialist, X, y_a)
    relational_b_error = mse(relational_specialist, X, y_b)

    # Parameter averaging creates a third computation: it uses both correlated cues.
    # It looks excellent in the seen world, but is mediocre under either intervention.
    centroid = reserve @ params
    centroid_seen_error = mse(centroid, X_seen, y_seen)
    centroid_a_error = mse(centroid, X, y_a)
    centroid_b_error = mse(centroid, X, y_b)

    print("Gate 4 — same answer, different algorithm")
    print(f"seen-world specialist output difference: {same_world_difference:.3e}")
    print(
        "seen-world MSE: "
        f"direct={direct_seen_error:.3e}, relational={relational_seen_error:.3e}, "
        f"reserve-centroid={centroid_seen_error:.3e}"
    )
    print(
        "intervention A MSE: "
        f"direct={direct_a_error:.3e}, relational={relational_a_error:.3e}, "
        f"centroid={centroid_a_error:.3e}"
    )
    print(
        "intervention B MSE: "
        f"direct={direct_b_error:.3e}, relational={relational_b_error:.3e}, "
        f"centroid={centroid_b_error:.3e}"
    )
    print()
    print(f"reserve approach mass: direct={reserve_direct:.6f}, relational={reserve_relational:.6f}")
    print(f"after context A: direct={a_direct:.6f}, relational={a_relational:.6e}")
    print(f"after context B: direct={b_direct:.6e}, relational={b_relational:.6f}")
    print(f"context-separated population TV: {query_tv:.6f}")
    print()
    print(f"reserve centroid weights: direct={centroid[0]:.6f}, relational={centroid[1]:.6f}")
    print(
        "aggregate Jacobian orientation (0 deg=direct, 90 deg=relational): "
        f"reserve={reserve_angle:.3f}, A={a_angle:.3f}, B={b_angle:.3f}"
    )

    # Same observed answers do not identify the computation.
    assert same_world_difference < 1e-12
    assert direct_seen_error < 1e-5 and relational_seen_error < 1e-5

    # Intervention exposes the distinct algorithms.
    assert direct_a_error < 1e-5 and direct_b_error > 1.0
    assert relational_b_error < 1e-5 and relational_a_error > 1.0

    # The reserve keeps both computational families, while context separates them.
    assert reserve_direct > 0.49 and reserve_relational > 0.49
    assert a_direct > 0.999 and a_relational < 1e-5
    assert b_relational > 0.999 and b_direct < 1e-5
    assert query_tv > 0.999

    # Averaging the two approaches is not equivalent to preserving them separately.
    assert centroid_seen_error < 1e-5
    assert centroid_a_error > 0.45 and centroid_b_error > 0.45

    # The internal sensitivity signature rotates with selection even though there is
    # no explicit router network choosing an expert.
    assert 40.0 < reserve_angle < 50.0
    assert a_angle < 25.0
    assert b_angle > 65.0


if __name__ == "__main__":
    main()
