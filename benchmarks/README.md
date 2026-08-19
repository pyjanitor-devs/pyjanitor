# Benchmarks

## Conditional join result materialization

`conditional_join_materialization.py` measures either result assembly alone or
the complete public join. It generates deterministic duplicate-key inputs with
NumPy, nullable integer and boolean, categorical, string, datetime, timezone,
and timedelta columns.

### ELI5

A conditional join has two jobs: find matching row numbers, then build a new
table from those numbers. `materialize` times the table-building job;
`end-to-end` times both jobs together.

Run a single case from the repository root:

```shell
pixi run python benchmarks/conditional_join_materialization.py \
  --mode materialize --how outer --density sparse --keep all
```

Options cover all join modes, `zero`, `sparse`, `dense`, and `full` match
densities, and `first`, `last`, and `all` retention. Use `--width` and `--rows`
to exercise narrow or wide results. Each run prints machine-readable JSON with
median and minimum elapsed time plus peak traced allocation.
Timing and memory are measured in separate passes so allocation tracing does
not distort the runtime result.

For a before/after comparison, use the same Python environment, machine,
options, and otherwise idle system on both commits. The minimum width is nine,
which is the size of the mixed-dtype base schema. Compare multiple runs; do
not treat a single small timing difference as meaningful. `tracemalloc` reports
Python and NumPy allocations observed during the operation, not total process
resident memory.
