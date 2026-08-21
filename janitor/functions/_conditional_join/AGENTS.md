# Conditional Join Agent Guidance

This file extends the repository-level `AGENTS.md` for work under
`janitor/functions/_conditional_join/`.

## Behavioral Invariants

- Preserve exact output values, row ordering, indexes, column structure,
  dtypes, null semantics, and extension-array behavior.
- Preserve aggregation, `return_matching_indices`, and
  `include_join_positions` behavior when changing index-generation internals.
- Treat `keep="all"` ordering as observable behavior. Changing the anchor or
  sort key can change its row order even when the set of matches is unchanged.
- For `keep="first"` and `keep="last"`, verify that any reordered search path
  still reduces matches using original right-row positions rather than scan
  order.
- Do not assume caller indexes are unique or contiguous unless the public
  dispatch path has normalized them before the relevant operation. Prefer
  position-space internally when positions are the intended representation.
- Preserve stable sorting and the distinctions between `<`, `<=`, `>`, and
  `>=`, especially around duplicates and nulls.

## Architecture Map

- `conditional_join.py` performs public validation, normalizes frames, and
  dispatches to the internal algorithms.
- `_get_indices_non_equi.py` selects the maintained non-equality path.
- `_le_ge_1_or_more.py` handles one or more same-direction inequality
  predicates for the default algorithm.
- `_dual_non_equi.py` implements the two-comparison region-number algorithm.
- `_not_range_join_regions.py` dispatches same-direction predicates for
  `join_algorithm="regions"`.
- `_helpers.py` contains shared filtering and index-materialization logic;
  changes there can affect multiple join modes.

Follow dispatch and result materialization end to end before concluding that
an internal ordering or index representation is unobservable.

## Performance Research

When investigating current or future `conditional_join` performance work,
include the region-number algorithms from Dathan and Trausan-Matu,
*Algorithms for Computing Inequality Joins* (DATA 2018), among the design
options considered: <https://doi.org/10.5220/0006826803570364>.

The paper's two-comparison algorithm underlies the existing `regions` path.
Section 3.2 describes a separate multi-comparison strategy that computes
regions for every predicate and dynamically chooses the driving field.
Consider both the current implementation and the paper's fuller algorithm
when researching:

- predicate or anchor selection;
- ordered field-role assignment in the asymmetric regions algorithm;
- relation orientation when frame sizes differ;
- candidate generation, filtering, and materialization;
- range joins and joins with three or more inequality predicates.

Treat the paper as an option to evaluate, not an automatic implementation
mandate. Account for correlations between predicates: individual selectivity
does not necessarily identify the best predicate pair.

## Benchmark Expectations

Performance work must compare equivalent joins with different predicate
orders and include, where relevant:

- `<`, `<=`, `>`, and `>=`;
- `keep="all"`, `keep="first"`, and `keep="last"`;
- two predicates and three or more predicates;
- selective-first, broad-first, and similarly selective predicates;
- sorted and unsorted inputs;
- dense, sparse, zero-match, and full-match cases;
- narrow and wide frames;
- nullable and extension dtypes;
- small inputs, where estimator overhead dominates;
- large inputs, extending toward 10M-50M rows when memory and runtime permit.

Report absolute timings as well as ratios. Measure already-optimal inputs to
detect regressions introduced by the optimizer itself. For sampled or
estimated selection, include skewed, correlated, periodic, and rare-tail data,
and state the estimator's expected failure modes.

## Verification

Run Python only through `pixi`, as required by the repository-level guidance.
The primary focused suite is:

```bash
pixi run pytest tests/functions/test_conditional_join.py -v
```

During iteration, narrower selections are acceptable, but run the complete
focused file before completion. Add independent-oracle tests where practical;
cross-join-and-filter is suitable for small correctness fixtures.

Tests for an optimizer must assert its decision directly, not only final output
invariance. Final output can remain identical even when the intended anchor,
predicate pair, or algorithmic branch was never selected.

Before treating a pull request as complete, follow the root requirement for a
fresh-context adversarial review, with particular attention to ordering,
nullable data, duplicate values, index representations, and performance on
already-selective inputs.
