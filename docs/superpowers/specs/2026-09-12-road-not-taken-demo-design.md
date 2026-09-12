# Road Not Taken — GAx Browser Demo Design

Date: 2026-09-12

## Goal

Build a single-page browser demo that makes the core GAx idea legible to a normal person:

> A search process can reveal where it is going before it gets there, and a system can preserve a useful alternative before that alternative is erased.

The demo must be visually attractive enough to feel like a consumer creative tool, while remaining experimentally auditable. It must not replay a prerecorded winner, peek at future generations before forecasting them, or hide prediction failures.

## User story

A person explores visual designs by repeatedly choosing preferred candidates. Under the surface, a finite genetic population evolves. Several genuinely different procedural design families coexist in the population.

After several observed generations, the app forecasts the future family composition from population history only. The forecast is frozen before the future generations are generated. The user can then audit the forecast against the actual future.

The app also identifies a useful family that is approaching extinction and offers two paths:

1. continue ordinary selection;
2. preserve a small reserve of that family.

Later, the objective changes toward the formerly weak family. Both worlds receive the same changed preference. The preserved world should, when the mechanism succeeds, recover the newly useful family faster. Failure cases remain visible.

## Product surface

### Normal mode

The default view should look like a polished creative exploration tool rather than a math dashboard.

The page contains:

- a large central canvas with the current best design;
- a grid of candidate designs from the live population;
- a simple prompt such as `Choose what feels closer`;
- generation count and seed in unobtrusive chrome;
- a compact family trajectory strip;
- a forecast card that appears only after enough history exists;
- an `Audit forecast` action;
- a `One road is disappearing` intervention card when extinction risk becomes high;
- a later `Change your mind` phase that flips the objective toward the endangered family;
- a side-by-side ordinary-vs-preserved race after the flip.

### Machinery mode

A `Show the machinery` toggle reveals:

- the exact random seed;
- population size;
- current family mass;
- observed history length;
- effective numerical rank of the observed history matrix;
- the fitted local predictor;
- frozen forecast values and forecast timestamp/generation;
- actual future values once audited;
- L1 forecast error;
- family half-life estimate when available;
- preservation intervention size;
- a short claim-boundary note stating that this is a finite stochastic browser experiment, not the exact deterministic Gate-5 regime.

## Visual search space

The visual artifact should be an SVG poster/interior-style composition generated from genotype parameters. The rendering must be deterministic from genotype plus seed.

Each genotype contains two layers:

### Procedural family

At least three distinct construction families, for example:

- `geometric` — aligned blocks, straight edges, repeated rectangular motifs;
- `organic` — curved forms, asymmetric flow, rounded masses;
- `experimental` — fragmented layers, offset repetition, unusual negative space.

These families must differ in rendering procedure, not merely in palette values. This preserves the Gate-4 spirit that different modes should correspond to different mechanisms rather than parameter clusters.

### Continuous genes

Within each family, continuous genes control properties such as:

- spacing;
- scale;
- aspect ratio;
- density;
- curvature or corner radius;
- palette hue and warmth;
- luminance contrast;
- accent strength;
- symmetry;
- depth offset;
- repetition count.

Mutation perturbs continuous genes and occasionally changes family according to a low cross-family mutation probability.

## Preference and fitness

The demo supports two preference sources.

### Interactive preference

The user chooses one or more visible candidates. Fitness is based on similarity to the selected candidate(s), with a mild novelty term so the population does not collapse instantly.

### Reproducible showcase preference

A `Run keynote demo` control uses a seeded hidden target preference vector so the full story can be replayed identically without requiring manual clicks. The target is generated from the visible seed at reset; it is not hardcoded to a family name. The selected dominant family therefore depends on the seed.

The showcase mode must visibly state that it is an automated preference stream for reproducibility.

## Evolution engine

Use a finite population, initially targeted at 96 individuals.

Each generation:

1. render/evaluate fitness under the current preference;
2. normalize fitness into selection probabilities;
3. sample parents using the seeded PRNG;
4. copy elites;
5. mutate offspring;
6. optionally apply low-rate family mutation;
7. record the full family histogram and any lower-dimensional summary used by the forecaster.

No future generations are generated speculatively before a forecast is frozen.

## Forecast mechanism

The forecast must be derived only from already observed population history.

For browser robustness, the first implementation predicts the coarse state vector rather than the entire 96-individual population. The state vector should contain at minimum family masses and may include a small set of within-family summary statistics.

Let observed coarse states be

`x_0, x_1, ..., x_m`.

Fit a minimum-norm linear map on the observed span:

`A_hat = Y X^+`

with

`X = [x_0 ... x_{m-1}]`

and

`Y = [x_1 ... x_m]`.

Because the browser cannot assume a numerical linear-algebra dependency, implement a small stable pseudoinverse/SVD or ridge-regularized least-squares routine in plain JavaScript. The machinery panel must report which estimator is used and its regularization/tolerance.

Forecast family composition for horizons 1 through 5. Clamp only tiny numerical negatives and renormalize to a probability simplex; report that this projection occurred.

The main forecast should not be shown until history rank and residual checks pass a minimum confidence criterion.

## Non-cheating audit protocol

This is the core trust requirement.

When the app creates a forecast it must store an immutable forecast record containing:

- seed;
- generation index;
- observed history digest;
- estimator parameters;
- predicted family masses for each horizon;
- current population digest;
- current preference digest.

Only after this record exists may the `Audit forecast` action generate the shadow future.

Audit steps:

1. clone the current population and PRNG state;
2. freeze the current preference;
3. run the requested number of generations in the clone;
4. measure actual family masses;
5. compare them with the frozen prediction;
6. display per-horizon and final-horizon L1 error.

The main live population must remain unchanged by the audit.

The source should contain an explicit comment and assertion guarding against calls that audit before a forecast record is frozen.

## Extinction-risk detector

A family becomes `at risk` when both conditions hold:

- current family mass is above zero but below a configurable reserve threshold; and
- the forecast predicts continued decline or crossing an extinction floor within the forecast horizon.

The UI should say `One road is disappearing`, not claim certainty.

When machinery mode is open, show the exact threshold and predicted mass trajectory.

## Preservation intervention

`Keep this road open` does not manufacture a future design. It changes only the selection policy by enforcing a small reserve for the endangered family.

Implementation options, in order of preference:

1. stratified survivor reservation: reserve a small number of parent slots for the endangered family if it still exists;
2. family-floor reweighting: add the smallest fitness adjustment needed to keep expected selected mass above a floor.

Use the first option because it is easier to explain and audit.

The intervention must be recorded in the timeline. The ordinary comparison arm receives no reserve.

## Counterfactual flip

After the intervention phase, `Change your mind` creates a new preference target favoring the endangered family. In interactive mode, the user can instead choose a candidate from that family.

Fork the current state into two matched worlds:

- `ordinary` — population evolved without preservation;
- `preserved` — population evolved with the reserve intervention.

The two worlds use matched generation counts and deterministic PRNG streams derived from the same seed namespace. They are not required to share identical sampled mutations if population ancestry differs, but the random stream construction must be reproducible and documented.

Show recovery curves for the newly useful family and best-fitness curves. The claim is comparative recovery speed, not guaranteed victory.

## Failure behavior

The app must surface rather than hide these cases:

- forecast confidence too low;
- forecast error large;
- endangered family already extinct before preservation;
- preserved family later turns out not to be useful;
- preservation costs short-term fitness;
- ordinary search rediscovers the family quickly and matches or beats the preserved arm.

A compact message such as `This run did not show an advantage` is preferable to silently reseeding.

The visible seed lets anyone reproduce an unfavorable run.

## Files

Initial implementation should stay deliberately small:

- `docs/index.html` — complete static app shell and markup;
- `docs/app.js` — seeded PRNG, genotype/rendering, GA engine, forecast, audit, preservation, UI state;
- `docs/style.css` — product-like presentation and responsive layout;
- `tests/test_demo_contract.py` — source-level and deterministic contract checks where practical;
- `README.md` — short demo section linking the page and explaining the claim boundary.

No framework, build system, backend, external model, or network dependency is required.

## State model

The app moves through explicit states:

`explore -> forecast_ready -> forecast_frozen -> audited -> intervention_choice -> flip -> comparison -> complete`

Invalid transitions are rejected in code. Reset creates a fresh state from the visible seed.

## Accessibility and device behavior

- works with mouse and keyboard;
- candidate cards have buttons and labels, not click-only divs;
- respects `prefers-reduced-motion`;
- maintains readable contrast;
- responsive down to mobile width, though desktop/tablet is the primary presentation target.

## Testing

### Deterministic engine tests

For a fixed seed:

- reset produces identical population digests;
- one generation produces identical next-state digest;
- rendering parameters are deterministic;
- audit cloning does not mutate live population;
- preservation never resurrects a family with zero members;
- ordinary and preserved forks are reproducible.

### Forecast-contract tests

- forecast uses only history indices `<= currentGeneration`;
- audit is impossible without a frozen forecast;
- forecast record is unchanged after audit;
- actual future values are absent from the forecast record before audit;
- forecast errors are computed from audited future only;
- low-rank/low-confidence histories suppress strong forecast language.

### UI contract tests

At minimum, source-level tests verify the presence of:

- seed control;
- keynote/autoplay mode;
- audit action;
- machinery toggle;
- preservation choice;
- failure language;
- claim-boundary text.

Browser smoke testing should be performed manually or with any available headless browser tooling before merge.

## Success criteria

A successful demo run should make three things visible:

1. **Forecast:** before unseen generations are computed, the app predicts coarse future search composition better than a naive persistence baseline on at least some reproducible seeds.
2. **Audit:** the user can reveal the real future and see honest numerical prediction error.
3. **Preservation:** on at least one documented reproducible seed, an at-risk procedural family is retained and later recovers faster after a matched preference flip than in the ordinary arm.

The repository must also retain documented seeds where the advantage is weak or absent if such cases are encountered during testing.

## Claim boundary

This demo is a finite-population stochastic extension inspired by GAx Gate 5. It does not inherit Gate 5's near-machine-precision forecast claim. Gate 5 used a deterministic fixed infinite-population mutation-selection operator and exact population distributions. This demo deliberately adds finite sampling, nonlinear user fitness, family mutation and an intervention.

The intended earned claim is narrower:

> A short observed history of a finite visual search can sometimes provide useful forward information about which procedural families are growing or disappearing, and that forecast can be used to preserve an alternative for a later objective change.

The demo is valuable only if that statement survives seeded, auditable runs without hiding failures.
