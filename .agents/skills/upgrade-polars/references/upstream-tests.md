# Step 5 detail: GPU test selection and triaging upstream failures

## GPU test selection in PR CI

PR CI runs only the upstream tests listed in `ci/cudf_polars/gpu_test_run_list.json` under `<exact polars version>|spmd|default`,
and a selected test that passes must have executed a query on the GPU. The `update-cudf-polars-polars-tests-selection` skill owns
generating and pruning the list. To run the PR configuration locally:

```bash
CUDF_POLARS_GPU_RUN_LIST=ci/cudf_polars/gpu_test_run_list.json POLARS_DIR=<dir>/polars \
    ./ci/run_cudf_polars_polars_tests.sh --engine spmd --inject-gpu-engine-blocksize default
```

What the PR CI errors mean during an upgrade:

- `Cannot select GPU tests for <version>|spmd|default`: the new Polars version (including a patch release) has no entry. Triage the
  failures below first, then generate it with the other skill.
- `Stale GPU run-list nodes`: upstream renamed or removed tests. Update the list in the same PR.
- `Selected GPU test passed without observed GPU execution`: a fix turned GPU execution into a CPU fallback. Remove that node from the list.

Skipped and xfailed items are not checked for GPU execution.

## Triage each failure

1. Real, small bug: fix it in cudf-polars and add a unit test.
2. Unsupported feature: CPU fallback (see the ground rules in `SKILL.md`), and remove the node from the GPU run list if present.
3. Larger fix needed: add the test to `EXPECTED_FAILURES` (or the streaming variants) in
   `python/cudf_polars/cudf_polars/testing/inject_gpu_engine.py` with a reason **and an issue link**.
4. Flaky, non-deterministic, hangs, only checks CPU logs, or would XPASS on one engine and fail on another: `TESTS_TO_SKIP`
   (or the streaming skip dicts).

Conventions for the dicts (`EXPECTED_FAILURES`, `TESTS_TO_SKIP`, `STREAMING_ENGINE_EXPECTED_FAILURES`,
`STREAMING_ENGINE_TESTS_TO_SKIP`, and the `*_SMALL_BLOCKSIZE` variants):
- Key: pytest node id relative to `py-polars` with exact parameter ids, `"tests/unit/io/test_csv.py::test_x[param]": "reason"`.
  Polars often re-parametrizes tests, so stale ids must be updated.
- Reason: a cudf issue URL, a polars issue/PR URL, or the established phrases ("Debug output on stderr doesn't match",
  "Casting that raises not supported on GPU", ...). Streaming engines wrap errors in an `ExceptionGroup`, which has its own phrase.
- Conditional entries use plain Python guards after the dict (not tuples).
- **Audit the existing entries.** xfails are strict: remove entries whose test now passes (XPASS fails the run) or no longer exists.
- Wrong-result failures should become fallbacks, not xfails. In the 2.1 upgrade several "failures" were real wrong results
  (integer-to-`Date` casts relabelled an `INT64` buffer as `INT32`; mixed-scale `Decimal` comparisons wrapped in libcudf; non-strict
  narrowing casts wrapped instead of returning null). Compare the GPU result with the CPU result before assuming a test is flaky.
- When a fix turns a fallback into correct GPU execution (or the reverse), the matching xfail becomes an XPASS: rerun the whole
  upstream suite after merging fixes and delete the stale entries.
- SPMD/streaming failures where the right polars error arrives wrapped in an `ExceptionGroup` go in
  `STREAMING_ENGINE_EXPECTED_FAILURES` with the established phrase ("Correct polars.exceptions.X raised but it's in an ExceptionGroup").

If the new polars adds an optional dependency that its tests need, install it too: read the `[project.optional-dependencies]` of the cloned
`py-polars/pyproject.toml` (for example `polars-iceberg`). Add it to
`ci/test_cudf_polars_polars_tests.sh` unconditionally: CI only runs the upstream tests against the installed Polars version.

If an upstream test requirement breaks (a new dependency release or a warning under `-W error`), patch the requirement with `sed -i`
in `ci/test_cudf_polars_polars_tests.sh` or add a `-W ignore:` filter in `ci/run_cudf_polars_polars_tests.sh`, with a comment that
links the upstream fix so it can be removed later. Tests that cannot run in CI go in `ci/cudf_polars/polars_test_deselections*.txt`.
