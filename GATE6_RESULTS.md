# Gate 6 — guarded backprop fast-forward

## Question

Can a short history of real neural-network training steps identify enough of the local parameter dynamics to **skip some future backward passes**, while a forward-only loss check prevents obviously bad jumps?

Gate 6 deliberately moves beyond the exact mutation-selection operator of Gate 5. The training dynamics here are nonlinear because the gradient changes with the weights.

The learner observes a short parameter trajectory

\[
\theta_0,\theta_1,\ldots,\theta_m
\]

from ordinary gradient descent. It centers that history, finds its observed SVD basis, fits an affine map in that reduced trajectory subspace,

\[
z_{t+1}\approx z_tA+b,
\]

and forecasts a future parameter state by repeatedly applying that fitted reduced operator.

A proposed jump is **not accepted blindly**. It is checked with one forward-only evaluation of the true loss. No gradient is computed for the skipped steps.

## Frozen setup

```text
model                 1 -> 12 tanh -> 1 MLP
parameters            37
training data          96 deterministic regression samples
optimizer              full-batch gradient descent
learning rate          1.0
target loss            0.0045
history window         10 parameter states
candidate horizons     16, 8, 4, 2
initialization seed    11
```

The synthetic regression target is

\[
y(x)=0.65\sin(3x)+0.25x+0.08\sin(9x),\qquad x\in[-1,1].
\]

The MLP and its backpropagation are implemented directly in NumPy. A finite-difference test checks the analytic gradient.

## Chronological forecast audit

The chronology is explicit.

1. Run 20 real gradient steps.
2. Fit the reduced operator from the last 10 already-observed parameter states.
3. Freeze a 16-step prediction.
4. Only after that prediction exists, run 16 shadow gradient steps to reveal the actual future.

The frozen 16-step forecast gives:

```text
reduced trajectory rank             9
operator relative weight error      5.317e-06
velocity relative weight error      4.431e-03
operator absolute loss error        6.838e-09
velocity absolute loss error        3.519e-05
```

So on this smooth full-batch trajectory, the short history contains much more than the last-step velocity. The reduced operator tracks the curved parameter trajectory over sixteen unseen gradient steps with about three orders of magnitude lower relative weight error than naive velocity extrapolation.

## Training result — actual backward passes counted

All learners start from the same weights and stop after reaching the same frozen loss target.

| learner | backward passes | forward-only checks | accepted jumps | nominal skipped steps | final loss |
|---|---:|---:|---:|---:|---:|
| ordinary GD | 239 | 240 | 0 | 0 | 0.004400 |
| **reduced-operator fast-forward** | **90** | **101** | **10** | **160** | **0.003496** |
| velocity fast-forward | 132 | 255 | 12 | 118 | 0.003559 |

The reduced-operator learner reaches the target with

\[
\boxed{239\to90}
\]

actual backward passes, a **62.3% reduction**.

The velocity baseline also saves work, but requires 132 backward passes and many more forward-only checks because 110 velocity proposals are rejected. In the frozen operator run, all 10 proposed operator jumps pass the guard.

The relevant claim is therefore not merely that the future weights can be fitted retrospectively. The training loop really executes fewer gradient evaluations.

## What a jump costs

A fast-forward proposal performs:

1. SVD / least-squares on the short parameter-history matrix;
2. repeated application of the small reduced affine map;
3. one ordinary **forward-only** loss evaluation.

It does **not** perform the skipped backward passes.

For this tiny NumPy model, wall-clock speed is not a meaningful GPU claim; linear algebra overhead can dominate at this scale. Gate 6 therefore freezes the hardware-independent count that matters first: backward evaluations avoided at matched target loss.

## Stochastic mini-batch attacker

The strongest immediate attacker makes the trajectory less operator-like. Keep everything else fixed, but let each backward pass see only **6 of 96 samples**.

```text
ordinary minibatch GD               228 backward passes   loss 0.004361
operator fast-forward               202 backward passes   loss 0.003872
accepted operator jumps              13
rejected operator proposals         336
```

The backprop reduction falls from **62.3%** to only **11.4%**.

That is important. The forward guard prevents the noisy operator predictor from freely taking bad jumps, but the cost is that most proposed jumps are rejected. The strong full-batch result therefore does not automatically transfer to highly stochastic training.

## Tests

`tests/test_gate6_backprop_fast_forward.py` checks:

- manual MLP backprop against finite differences;
- exact extrapolation of a known affine recurrence by the reduced forecaster;
- a fast-forward proposal performs a forward check without incrementing the backward counter;
- the forecast record is frozen before shadow gradients are generated;
- matched-loss full-batch training uses fewer backward passes;
- the reduced operator beats a last-step velocity extrapolator in the frozen curved-trajectory control;
- mini-batch sampling is reproducible;
- the complete Gate-6 receipt reproduces the backprop saving and the stochastic degradation.

Run:

```bash
python -m experiments.gate6_backprop_fast_forward
pytest -q tests/test_gate6_backprop_fast_forward.py
```

## Claim boundary

Gate 6 does **not** show that arbitrary neural-network training can replace backpropagation with Koopman/DMD steps.

The strong result is currently confined to a small deterministic full-batch MLP whose local weight trajectory is highly forecastable. The mini-batch attacker already shows the limitation clearly: once gradient noise destroys trajectory coherence, most long jumps are rejected and the saving collapses.

The earned statement is:

> **On a smooth deterministic neural-training trajectory, a reduced operator identified only from past weight states can propose parameter fast-forwards that pass a forward-only loss guard and reach the same loss target with substantially fewer actual backward passes. In the frozen control the reduction is 62.3%; with strong mini-batch noise it falls to 11.4%.**

That makes the next question concrete rather than speculative: can the same mechanism survive realistic PyTorch/GPU training strongly enough that backward-pass savings exceed the operator-fit and validation overhead?
