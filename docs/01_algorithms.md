# Algorithms and design rationale

This document describes the optimisation algorithms as they are implemented here. Notation: `x_i` is particle `i`'s position, `v_i` its velocity, `p_i` its personal best, `g` the global best, `L_c` the leader of particle `i`'s group, `⊙` element-wise multiplication.

## 1. Standard PSO (global best)

```
v_i(t+1) = w · v_i(t) + c1 · r1 ⊙ (p_i − x_i) + c2 · r2 ⊙ (g − x_i)
x_i(t+1) = clip(x_i(t) + v_i(t+1), lower, upper)
```

- `r1`, `r2` are independent uniform vectors in `[0, 1]`.
- The Master-file port stores velocity, uses a mirror-boundary velocity flip, `w = 1.0`, `wdamp = 0.99`, `c1 = 1.5`, `c2 = 2.0`, and `Vmax = 0.1 × range`.
- The vectorised benchmark runner uses Latin-hypercube initialisation and calls the objective once per iteration on the whole population.

## 2. PSO-AWDV (adaptive weighted delay velocity)

Adds a delayed-velocity term and adapts both inertia and acceleration coefficients from the swarm's evolutionary state:

```
v_i(t+1) = w(t) · v_i(t) + (1 − w(t)) · v_{i−1}(t)
           + c1(t) · r1 ⊙ (p_i − x_i) + c2(t) · r2 ⊙ (g − x_i)

w(t)   = 1 − a / (1 + exp(b · E(t)))
E(t)   = (f_worst(0) − f_best(t)) / f_worst(0)
c1(t)  = (c1_initial − c1_final) · (T − t)/T + c1_final
c2(t)  = (c2_initial − c2_final) · (T − t)/T + c2_final
```

with `a = 0.9`, `b = 0.5`, `c1: 2.5 → 0.5`, `c2: 0.5 → 2.5`. The delayed term uses the previously updated particle's velocity, exactly as in the MATLAB loop.

## 3. PSOEA (evolutionary PSO initialisation)

1. Generate and evaluate an initial population.
2. Select parents by rank-based roulette (exponential rank weights).
3. Produce `N` children by uniform crossover (per-coordinate parent choice).
4. Run classical PSO from the children, seeded with the best solution found.

## 4. Real-coded GA

- Rank selection with selection pressure `1.0`.
- Arithmetic crossover: `child = α ⊙ parent1 + (1 − α) ⊙ parent2`, `α ~ U(0,1)^d`.
- Gaussian mutation with probability `0.02` per coordinate and a cooling scale `1 − 0.9 · t/T`.
- elitism: `max(2, round(0.05 · N))` individuals.

## 5. PSOMbest (paper algorithm)

Every iteration, k-means clusters the current swarm (warm-started from the previous centroids in the reference port). Each cluster contributes a leader `L_c`, and a third attraction term is added:

```
v_i(t+1) = w · v_i(t)
           + c1 · r1 ⊙ (p_i − x_i)
           + c2 · r2 ⊙ (g − x_i)
           + c3 · r3 ⊙ (L_{c(i)} − x_i)
```

Default reproduction settings: `k = 5`, `w = 1.0`, `wdamp = 0.99`, `c1 = 0.5`, `c2 = 1.5`, `c3 = 1.0`, `Vmax = 1.0 × range`, velocity discarded after each move, LHS initialisation with 5 maximin trials.

**Important implementation detail.** The supplied MATLAB master code builds the leader as a *scalar*: it takes the first coordinate of the first cluster member and copies that number to every coordinate. That is an unusual rule and is the subject of experiments 07 and 08. The "corrected" profile instead stores the full leader vector (the cluster member with the lowest current cost) and keeps velocity memory.

## 6. B1 — frozen grouping

B1 keeps the PSOMbest update but **builds the groups once**, at initialisation, and freezes them. The cluster pull fades linearly:

```
c3(t) = c3_0 · (1 − t / T)
```

This removes the per-iteration k-means cost (the dominant cost in PSOMbest) and prevents the group structure from churning late in the run. Three grouping rules are studied:

### B1-fade (k-means, frozen)

- Run k-means once on the initial swarm; keep the centroids frozen.
- Each iteration, assign particles to the nearest frozen centroid.
- Rebuild the leader table every iteration from the personal bests: for each group, the member with the best personal-best cost.
- Leader is a full vector.

### B1-coord-fade (per-coordinate grouping)

- For every coordinate `j`, cut the initial swarm into `k` equal-count groups using quantiles of `x[:, j]`.
- A particle therefore belongs to a *different* group for each coordinate.
- For each `(group, coordinate)` pair, the leader value is the `j`-th coordinate of the group member with the best personal best.
- The pull is applied coordinate by coordinate.

This is a high-dimensional grouping scheme: it can separate particles that are close in Euclidean space but differ along one axis.

### B1-proj-fade (principal-axis grouping)

- Compute the first principal axis of the initial swarm (first right singular vector of the centred position matrix).
- Project all particles onto that axis and cut the projection into `k` equal-count intervals.
- Freeze the projection and cuts; rebuild the leader table from personal bests each iteration.

This is the cheapest grouping rule (`O(N·d)` projection plus a searchsorted) and is the focus of the ablation experiment.

## 7. B1 static confined clustering (baseline variant)

A one-dimensional version for the PV GMPPT problem:

- The voltage range is split into `k` contiguous regions.
- Particles are assigned to regions and **confined** to their region for the whole run.
- There is no global-best term; the social term is the best personal best *within the region*.
- Regions have a minimum particle count (`N // k` floor check).

This variant is the basis for B2's compute-saving story: because particles are captive, a region that has been fully explored can be frozen without losing coverage elsewhere.

## 8. B2 — coverage-based (tabu-style) region exhaustion

`tabu_coverage.RegionCoverage` discretises each region into bins of width `0.3 V` (three bins per narrowest verified peak width of ~0.96 V). A bin is marked visited the first time any particle sits in it; a region is **exhausted** when every bin has been visited.

The mechanism is deliberately **passive**: it never redirects movement. What each B1 variant does with the "exhausted" signal differs:

### B2-confined (compute saving)

- Freeze all particles in an exhausted region.
- Stop evaluating the objective for frozen particles.
- Only the active (non-frozen) subset is evaluated each iteration.
- Runs seed coverage from the initial scatter, and immediately freezes regions that are already exhausted at iteration 0 (a bug found during verification: without it, those particles would run the full budget with no saving).

### B2-dynamic (accuracy only)

- Particles are never confined and never frozen.
- For a particle currently inside an exhausted region, the cluster-pull term is dropped; it falls back to inertia + personal best.
- Particles keep moving and evaluating, and regain a real pull if they enter an active region.
- Consequently, `n_evals` is identical to plain B1-dynamic by design. Any effect is on accuracy, not compute.

## 9. Scalar-leader experiments

Experiments 07 and 08 isolate the effect of the scalar leader:

- **Leader rules** (experiment 07): `scalar_first` (supplied code), `scalar_mean`, `scalar_best1`, `stored_best` (vector), `pbest_vec` (vector), `none` (no cluster term).
- **Scalar rules applied to B1** (experiment 08): `lead0` (first coordinate of the normal leader), `leadmean` (mean of the normal leader vector), `first` (supplied PSOMbest rule).

The shifted-optimum test (experiment 02) is the control: it moves each benchmark optimum so that its best value differs in every coordinate. A scalar leader pushing all coordinates to one common value cannot be correct there; a vector leader can.

## 10. PV string model

`pv_testbed/pv_model.py` builds a P-V curve for a series string of PV modules:

- Single-diode (Bishop) forward-bias model per module, parameters from Mao et al. (2018).
- Modules that cannot carry the string current are treated as bypass-diode-protected: their terminal voltage is clamped to `−0.6 V` and they stop limiting the string current.
- The curve is obtained by sweeping string current from near zero to `max(Isc)`, summing module voltages, and keeping the positive-voltage branch.
- Ground truth comes from a fine reference sweep and a prominence-based peak detector, not from the number of shading groups.
- `prominence_W` is a tunable design parameter (default `0.02 W`); the adversarial scenario deliberately contains a low-prominence local peak so this sensitivity is visible.

Preset scenarios:

| Scenario | Irradiances (W/m²) | Expected peaks |
|---|---|---|
| `uniform` | 6 × 1000 | 1 |
| `simple_partial` | 3 × 1000, 3 × 500 | 2 |
| `complex_partial` | 1000, 800, 400, 700, 1000, 200, 900, 550 | 3–4, uneven |
| `adversarial_narrow_peak` | 4 × 300, 3 × 600, 1 × 1000 | 2; global is ~5× narrower than local |

`random_scenario(n_modules, n_groups, seed)` supports bulk seeded testing.
