# Step 4 detail: oldest supported polars and coverage

## Run the unit tests on the oldest supported polars too

CI tests both ends of the supported range. Overlay the old polars on the same environment so nothing else changes:

```bash
pip install --no-deps --target <tmp>/polars-old "polars==X.Y.Z" "polars-runtime-32==X.Y.Z"
PYTHONPATH=<tmp>/polars-old:$PWD python -m pytest tests --maxfail=300 ...   # from python/cudf_polars
python -c "import sys; sys.path.insert(0, '<tmp>/polars-old'); import polars; print(polars.__version__)"   # confirm
```

What usually breaks, and the fix:

- **A test that asserts new behaviour** (a plan shape, a message, a broadcast rule): `pytest.mark.skipif(POLARS_VERSION_LT_NN, reason=...)`,
  or assert per version when both versions matter (an explain test that expects merged scans on 2.1 and a `UNION` on 2.0).
- **An xfail you deleted because it XPASSes on the new version** still fails on the old one. Keep it, conditional on the old version:
  `pytest.mark.xfail(POLARS_VERSION_LT_NN, reason=...)`. Strict xfail only fires when the condition is true.
- **A new xfail for new behaviour** needs `not POLARS_VERSION_LT_NN` in its condition, or it XPASSes (and fails) on the old version.
- Define the flag next to the others in `cudf_polars/utils/versions.py` and delete it when the lower bound passes it. Write it as
  `POLARS_VERSION_LT_NN = parse("X.Y") > POLARS_VERSION`: the pinned ruff flags `POLARS_VERSION < parse("X.Y")` as a Yoda condition (SIM300),
  and pre-commit.ci fails on it (older flags in this file predate that rule).
- Failures that mention a libcudf feature newer than the nightly in your environment (for example a test from a just-merged C++ change) are
  an environment limit, not a regression. CI builds libcudf from the branch.

## Coverage

CI measures coverage on the newest polars only.

- Code reachable only on an older supported version gets `# pragma: no cover; polars X.Y only` on its `if`/`raise` line, so the whole
  block is excluded. Code that is now live loses its pragma.
- Defensive raises that polars validates first use `# pragma: no cover; polars validates this ahead of time`.
- New behaviour gets a real test. Fallbacks get `assert_ir_translation_raises`; parity gets `assert_gpu_result_equal`;
  error parity gets `assert_collect_raises`.
