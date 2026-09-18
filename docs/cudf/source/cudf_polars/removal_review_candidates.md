# cudf-polars local-test removal review queue

This queue is deliberately a review artifact, not an automatic deletion list.
The coverage command initially collects broad upstream compatibility coverage.
It does not claim no-fallback GPU coverage until the strict GPU policy supplies
a corresponding coverage data file. Do not remove a local test based on broad
overlap alone.

Before adding a row, verify that the proposed upstream replacement has a
public-API match and passes through the injected GPU engine with
`raise_on_fail=True` in both in-memory and small-blocksize SPMD modes.

The following superficially similar local tests are intentionally **not**
removal candidates:

- `test_hconcat_strict_different_heights`: its direct upstream lazy counterpart
  fails under small-blocksize SPMD, so it remains a GPU error-path test.
- `test_hconcat_should_broadcast` and `test_empty_init`: these construct
  cudf-polars IR directly and test streaming/internal contracts.
- Eager upstream `pl.concat` and `Series.filter` tests: they do not call
  `LazyFrame.collect` and therefore cannot establish cudf-polars coverage.
