# cudf-polars local-test removal review queue

This queue is deliberately a review artifact, not an automatic deletion list.
Every proposed replacement has a public-API match and an upstream lazy test
that runs through the injected GPU engine with `raise_on_fail=True`.
The initial three rows represent 60 collected local pytest nodes (5, 5, and
50 parameterizations respectively).

| Local test | Upstream strict-GPU replacement | Shared public contract | Proposed action |
| --- | --- | --- | --- |
| `tests/test_hconcat.py::test_hconcat` | `tests/unit/functions/test_concat.py::test_concat_lazyframe_horizontal` | Horizontal concatenation of lazy frames with equal heights | Delete after reviewer confirms the local computed-column variant adds no GPU-specific contract |
| `tests/test_hconcat.py::test_hconcat_different_heights` | `tests/unit/functions/test_concat.py::test_concat_lazyframe_horizontal` | Horizontal concatenation pads the shorter lazy input with nulls | Delete after reviewer confirmation |
| `tests/test_join.py::test_join_where` | `tests/unit/lazyframe/test_lazyframe.py::test_join_where` | Lazy non-equi (`join_where`) execution with multiple predicates and unordered result comparison | Split/reduce the local parameter matrix after reviewer maps each remaining predicate shape |

The following superficially similar local tests are intentionally **not** in
the removal queue:

- `test_hconcat_strict_different_heights`: its direct upstream lazy counterpart
  fails under small-blocksize SPMD, so it remains a GPU error-path test.
- `test_hconcat_should_broadcast` and `test_empty_init`: these construct
  cudf-polars IR directly and test streaming/internal contracts.
- Eager upstream `pl.concat` and `Series.filter` tests: they do not call
  `LazyFrame.collect` and therefore cannot establish cudf-polars coverage.
