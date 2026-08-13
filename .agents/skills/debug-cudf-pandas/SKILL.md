---
name: debug-cudf-pandas
description: Debug and fix pandas test suite failures under the cudf.pandas compatibility layer. Use when given pytest node IDs of failing pandas tests that need to be fixed for cudf.pandas compatibility.
---

# Debug cudf.pandas Compatibility Failures

## Overview

`cudf.pandas` is a zero-code-change GPU accelerator for pandas. It intercepts `import pandas` via a Python `MetaPathFinder` (`python/cudf/cudf/pandas/module_accelerator.py`) and wraps all pandas objects and functions in proxy types that try the cudf (GPU) path first, falling back to pandas (CPU) if cudf raises an exception.

When the pandas test suite is run with `-p cudf.pandas`, test failures indicate one of:
- A **cudf implementation bug** — cudf's DataFrame/Series/Column/Index behavior differs from pandas for an edge case
- A **proxy/dispatch bug** — the wrapping/unwrapping mechanism doesn't correctly handle a type or operation
- A **missing proxy registration** — a pandas type or return value has no registered cudf equivalent
- A **to/from_pandas conversion bug** — data is corrupted or lost when converting between cudf and pandas objects
- A **test setup bug** — the testing scripts or the pandas testing plugin introduce an issue
- A **dependency/environment gap** — the test requires a package (e.g. xlsxwriter) that pandas CI has but our test environment lacks, causing a different code path to execute
- A **pandas bug** — rarely, the expected behavior in the pandas test itself is wrong

Your job is to find the root cause and implement the fix.

## Unacceptable Fix Patterns

The following patterns are prohibited regardless of whether they make a test pass. Stop immediately and ask the user if the only apparent solution falls into one of these categories.

- **Esoteric/test-specific special cases**: Do not implement narrow, brittle logic targeting only the specific failing node IDs (e.g. special-casing a single dtype/shape combination). If a test is failing, find the general API contract mismatch and fix it at that level.
- **Pandas CPU fallback as a fix**: Typically you should not make tests pass by simply raising an exception inside cudf to force CPU fallback. Falling back to pandas/CPU is only an acceptable fix if attempting to fix the bug leads to discovering that large features must be implemented in cudf to support the change (e.g. a completely new class).
- **Private pandas APIs**: Do not import or call any symbol from `pandas.core`, `pandas.compat`, or any underscored pandas module (e.g. `pandas._libs.tslibs.parsing`). These are explicitly unstable per the pandas API policy. Use public pandas APIs or write equivalent local logic instead.
- **PyArrow as a CPU execution backend**: Do not route GPU operations through `pyarrow.compute` on CPU as a substitute for cudf/libcudf semantics. Arrow is an interchange format; it is not an acceptable execution backend for cudf operations.
- **Returning pandas objects from cudf APIs**: cudf public methods (`Series`, `Index`, `DataFrame` operations and accessors) must return cudf-native objects, not `pd.Series`, `pd.Index`, or `pd.DataFrame`. Use `_return_or_inplace` and the existing cudf container reconstruction helpers.
- **Diverging from pandas to pass a test**: The goal is to match pandas behavior exactly. Do not implement proxy overrides that suppress exceptions or alter behavior that vanilla pandas exhibits. If pandas raises an error in a given scenario, cudf.pandas should raise the same error. A fix that makes cudf.pandas behave *differently* from pandas — even if it makes a test pass — is wrong.

---

## Prerequisites

Before starting, verify you are at the repository root. All commands in this skill assume the working directory is the cudf repository root.

**Setting up and refreshing the test checkout.** The test harness lives in `pandas-testing/pandas-tests/`. The runner performs first-time setup (cloning pandas, copying the test tree, rewriting imports) *only* when the relevant directories are missing; once they exist it runs against them as-is and never refreshes them. The xfail/skip markers are applied by a pytest plugin loaded fresh on every run (`-p cudf.pandas.scripts.pandas-testing-plugin`), not by appending to the pandas `conftest.py`, so repeated runs do not accumulate duplicate hook registrations.

Because the runner never refreshes an existing checkout, do not rely on its contents being current, and be aware that any edit you make under `pandas-testing/` persists into every subsequent run. Never modify the vendored pandas test files (the `tests/**.py` tree — see the "never modify the pandas test files" rule below); a stray edit there will silently follow you and can derail your investigation. The one sanctioned in-place edit is the temporary `xfail_strict` flip in `pandas-tests/pyproject.toml` described in Step 0 — revert it (or delete the checkout) once you are done so it does not linger. Delete the checkout whenever it may be stale. In particular, if you change `python/cudf/cudf/pandas/scripts/run-pandas-tests.sh` in a way that affects what it places in `pandas-testing/`, the existing checkout will not reflect that change — delete it before re-running so the runner rebuilds it. The runner uses two independent guards: it clones pandas into `pandas-testing/pandas/` only if that directory is missing, and copies the test tree into `pandas-testing/pandas-tests/` only if *that* is missing. So removing just `pandas-tests/` re-copies a clean test tree **from the existing clone**, while picking up a new pandas version (the clone is pinned to the tag matching the installed pandas) requires removing the whole `pandas-testing/` so the clone is refetched:

```bash
rm -rf pandas-testing/pandas-tests/   # re-copy a clean test tree from the existing clone
rm -rf pandas-testing/                # full reset, e.g. after a pandas version change or a change to run-pandas-tests.sh
```

The cudf Python package is almost entirely pure Python. For **inplace installs** (e.g. `pip install -e .`), changes to `.py` files take effect immediately — no rebuild is needed. For non-inplace installs (e.g. `./build.sh`), you must either reinstall or copy changed files to site-packages.

---

## Input Format

Input is one or more pytest node IDs, relative to the `pandas-tests/` directory inside the test harness:

```
tests/groupby/test_reductions.py::test_first_last_skipna[Float64-False-False-first]
tests/frame/methods/test_fillna.py::test_fillna_inplace
```

Node IDs with parameters like `[Float64-False-False-first]` target a specific parametrized case. When multiple node IDs are provided, check whether they share a root cause before attempting separate fixes.

---

## Step 0 — Update pandas-testing-plugin.py

The file `python/cudf/cudf/pandas/scripts/pandas-testing-plugin.py` contains three dictionaries that gate how tests are handled:

- **`NODEIDS_THAT_FAIL`** — tests marked `xfail` (expected to fail). Keys are alphabetically sorted.
- **`NODEIDS_TO_SKIP`** — tests marked `skip` (not run at all). Keys are alphabetically sorted.
- **`NODEIDS_PATHS_TO_SKIP`** — prefix-based path skips covering entire modules.

The pandas-tests harness runs with `xfail_strict = false` (set in the vendored `pandas-tests/pyproject.toml` to tolerate flaky XPASSes — [rapidsai/cudf#22681](https://github.com/rapidsai/cudf/issues/22681)). A test listed in `NODEIDS_THAT_FAIL` that now *passes* is therefore reported as `XPASS` **without failing the run**, so a stale entry will not flag itself. You must change the false to true yourself before testing your fix; otherwise the test reports `XPASS` instead of a genuine `PASSED` and the dead marker lingers silently. Do not commit this change in any commit.

Search for the node ID:

```bash
grep -n "tests/groupby/test_reductions.py::test_first_last_skipna" \
    python/cudf/cudf/pandas/scripts/pandas-testing-plugin.py
```

If found, remove the line. Keys must remain in alphabetical order after the edit. Then validate the file still parses:

```bash
python -c "exec(open('python/cudf/cudf/pandas/scripts/pandas-testing-plugin.py').read())"
```

If the node ID is *not* found in any dictionary, you are dealing with a new regression — proceed directly to Step 1.

---

## Step 1 — Reproduce the Failure

Run from the repo root:

```bash
bash python/cudf/cudf/pandas/scripts/run-pandas-tests.sh \
    "tests/groupby/test_reductions.py::test_first_last_skipna[Float64-False-False-first]" \
    -xvs
```

On the first run the script clones the matching pandas version and copies the test tree (subsequent runs reuse it). It runs pytest with both `-p cudf.pandas` and the marker plugin `-p cudf.pandas.scripts.pandas-testing-plugin`, and automatically applies `-m "not slow and not single_cpu and not db and not network"` and `--disable-warnings`. Substitute your actual node ID.

**If the test passes**: the xfail entry was stale. Commit only the `pandas-testing-plugin.py` change and stop.

**If the test fails**: read the failure output carefully — the assertion message tells you the exact behavioral difference. Proceed to Step 2.

---

## Step 2 — Understand the Test

After the first run, the test file exists at:

```
pandas-testing/pandas-tests/tests/<module>/<test_file>.py
```

Read the test. Identify:
- Which pandas API is being exercised (method name, class, module)
- What the test asserts — this is the exact behavior cudf must match
- Whether parameters narrow the scope (e.g. only `Float64` dtype, only `skipna=False`)

The assertion error output from Step 1 tells you what cudf produced vs. what was expected. Use both together to understand the gap.

---

## Step 3 — Diagnose Root Cause

**Always check cudf's direct behavior first.** Do not jump to proxy investigation.

### 3a. Direct cudf test (most important step)

Write a minimal script that exercises the API using cudf directly, without going through `cudf.pandas`. This is strictly an example — adapt the structure to match what the failing test actually does (comparisons, exception checks, dtype validation, etc.):

```python
import cudf
import pandas as pd

# Mirror the test setup
pdf = pd.DataFrame({"value": [1.0, None, 2.0, 3.0]}, dtype="Float64")
cdf = cudf.from_pandas(pdf)

pd_result = pdf.groupby(level=0).first(skipna=False)
cudf_result = cdf.groupby(level=0).first(skipna=False)

print("pandas:", pd_result)
print("cudf:  ", cudf_result)
print("match:", pd_result.equals(cudf_result.to_pandas()))
```

For tests that assert exceptions are raised, structure your script to verify both cudf and pandas raise the same exception type and message.

Save to a temporary file (e.g. `test_debug.py`) and run:

```bash
python test_debug.py
```

Results:
- **cudf result differs from pandas** → cudf implementation bug → go to Step 4a
- **cudf raises an exception** → missing feature or bug → evaluate scope; may need user input if the feature is large. Note: this may be OK if the test is verifying that an exception *should* be raised.
- **cudf result matches pandas** → proxy/dispatch bug → go to Step 4b
- **cudf result matches pandas AND the test still fails** → check if vanilla pandas (without cudf.pandas) also fails → go to Step 3d

**Classify the root cause before writing any fix.** Ask yourself: Is this a specific method/keyword handling bug? A broad dtype casting mismatch affecting many operations? A proxy/wrapping issue? A missing cudf capability? For broad issues, the fix should be applied at the shared/base layer, not patched per individual method. If the only apparent fix is test-shaped (i.e. it looks like it exists to make exactly these node IDs pass), step back and re-examine the general API contract.

### 3b. Environment variable diagnostics (run through cudf.pandas)

Use these env vars to trace what is happening at the proxy layer:

| Variable | Effect |
|----------|--------|
| `CUDF_PANDAS_FAIL_ON_FALLBACK=1` | Raises instead of silently falling back — shows exactly which operation triggers fallback |
| `LOG_FAST_FALLBACK=1` | Logs every fallback with function name and exception |
| `CUDF_PANDAS_DEBUGGING=1` | Runs both cudf and pandas paths in parallel, warns on result divergence |

```bash
CUDF_PANDAS_FAIL_ON_FALLBACK=1 bash python/cudf/cudf/pandas/scripts/run-pandas-tests.sh \
    "<node_id>" -xvs
```

```bash
LOG_FAST_FALLBACK=1 bash python/cudf/cudf/pandas/scripts/run-pandas-tests.sh \
    "<node_id>" -xvs 2>&1 | grep -i fallback
```

**Fallback is a diagnostic signal, not a fix.** If `CUDF_PANDAS_FAIL_ON_FALLBACK=1` causes the test to fail, routing the operation to pandas CPU is not acceptable as a final solution unless adding GPU support would require implementing large, entirely unsupported features. Bug fixes for cudf's behavior (e.g. to add support for a particular dtype to a function) are the expected path.

### 3c. Standalone instrumented script

For deeper investigation, copy the relevant test body into a temporary script (e.g. `test_debug.py`), add print statements or assertions at intermediate steps, then run through the proxy layer or try to write matching pandas code to see the differences in behaviors and fix them:

```bash
python -m cudf.pandas test_debug.py
```

This gives you full control to narrow down exactly where the divergence begins.

### 3d. Verify vanilla pandas behavior (critical sanity check)

Before implementing any proxy-layer fix, check whether the test passes under vanilla pandas in your environment:

```bash
python -m pytest pandas-testing/pandas-tests/tests/<path>::<test> -xvs
```

(Without `-p cudf.pandas` — just run it directly.)

If the test **also fails under vanilla pandas**, the issue is NOT a cudf bug. Common causes:
- **Missing dependency**: pandas CI has a package installed (e.g. `xlsxwriter`, `lxml`, `odfpy`) that changes code path selection. Check pandas' `ci/deps/` YAML files to see what they install.
- **Version mismatch**: the installed version of a third-party library differs from what pandas CI uses.
- **Pandas test bug**: the test itself is broken (e.g. relies on side effects of other packages being present).

Resolution for dependency gaps: add the missing package to `dependencies.yaml` under the `test_cudf_pandas_pandas_tests` section (for conda environments that don't use pip extras), then run `rapids-dependency-file-generator` to propagate. See Step 4c.

Resolution for pandas bugs: xfail the test with an explanation string that describes why it's a pandas/upstream issue, and optionally write up a bug report for upstream.

---

## Step 4a — Fix a cudf Implementation Bug

The fix must be **tightly scoped** and try to keep it minimal. Fixing one edge case can break another test. Do not refactor surrounding code while fixing, and do not add `mode.pandas_compatible` guards (ask the user first).

**Classify the failure mode** before writing a fix: is it a specific issue (e.g. one method handles a keyword incorrectly) or a broad failure mode (e.g. dtype casting inconsistency that affects many operations)? For broad issues, consider whether the fix should be applied at a shared/base level rather than patching individual methods.

Common fix locations:

| What fails | Where to look |
|---|---|
| DataFrame method | `python/cudf/cudf/core/dataframe.py` |
| Series method | `python/cudf/cudf/core/series.py` |
| Index operation | `python/cudf/cudf/core/index.py` |
| Shared DataFrame/Series method | `python/cudf/cudf/core/indexed_frame.py` |
| Column-level operation | `python/cudf/cudf/core/column/*.py` |
| GroupBy operation | `python/cudf/cudf/core/groupby/` |
| IO operation | `python/cudf/cudf/io/` |

Note: `mode.pandas_compatible` is automatically set to `True` when cudf.pandas is active. Account for this in any conditional logic, but do not add new guards for it without explicit user approval.

If the bug is not in cudf core, move to cudf.pandas-specific fixes.

**Prohibited in cudf implementation fixes:**
- Do not call `pandas._libs.*` or any other private/underscored pandas module. Use public pandas APIs (`pd.Timestamp`, `pd.to_datetime`, `pd.DateOffset`, etc.) and write equivalent local logic if needed.
- Do not convert to pyarrow and use `pyarrow.compute` as a substitute for libcudf behavior. If libcudf doesn't support the exact semantics, consult the user before adding a pyarrow fallback.
- Do not construct and return `pd.Series(...)`, `pd.Index(...)`, or `pd.DataFrame(...)` from cudf public methods. Use `_return_or_inplace`, `_from_column`, or other cudf container reconstruction helpers so that return types remain cudf-native.

---

## Step 4b — Fix a Proxy/Dispatch Bug

Only reach this step after Step 3a has confirmed that cudf itself is correct. Some of the common cases you should consider at this stage:

**Most common cause**: a pandas object in a particular state is not round-tripping correctly through cudf's conversion APIs. `cudf.from_pandas()` and `<object>.to_pandas()` are culprits here. To debug this case, set up the standalone instrumented script as described in step 3c. Then instrument the code to perform direct modifications and testing of the proxy state, for instance by accessing the `_fsproxy_fast` and `_fsproxy_slow` attributes of the relevant objects and calling `cudf.from_pandas` and `<object>.to_pandas` to see if information is being lost or corrupted in one of the conversions.

**Next most common cause**: a pandas return type has no registered cudf proxy. Check `python/cudf/cudf/pandas/_wrappers/pandas.py` — this file registers which pandas types map to which cudf types using `make_final_proxy_type()` and related functions. If a new pandas type needs wrapping, add the registration here.

**`fast_slow_proxy.py` and `module_accelerator.py`** are core infrastructure files. Fix them only if you believe the bug is in one of them.

### Important constraint for proxy fixes

**Never make cudf.pandas diverge from pandas to pass a test.** If your proposed proxy fix would cause cudf.pandas to behave *differently* from vanilla pandas (e.g. suppressing an exception that pandas raises, or returning a different value), that fix is wrong — even if it makes the test pass. The test may be broken, or the issue may be an environment/dependency gap rather than a proxy bug. Always verify vanilla pandas behavior first (Step 3d).

---

## Step 4c — Fix a Dependency or Environment Gap

Only reach this step if Step 3d confirmed the test also fails under vanilla pandas due to a missing package or version mismatch.

1. **Identify the missing dependency.** Check what pandas CI installs by examining their CI config files (available in `pandas-testing/pandas/ci/deps/`). Common culprits: `xlsxwriter`, `lxml`, `odfpy`, `python-calamine`, `pyxlsb`.

2. **Add to `dependencies.yaml`** under the `test_cudf_pandas_pandas_tests` section. This group provides packages that pandas CI has installed (via pip extras like `pandas[excel]`) but conda environments need listed explicitly:

```yaml
  # Additional dependencies for running the pandas test suite under cudf.pandas.
  # Unlike test_python_pandas_cudf (which uses pip extras like pandas[excel]),
  # conda environments need these listed explicitly.
  test_cudf_pandas_pandas_tests:
    common:
      - output_types: [conda]
        packages:
          - <new-package>
```

3. **Regenerate dependency files:**

```bash
rapids-dependency-file-generator
```

This propagates the change to `python/cudf/pyproject.toml` and any other generated files.

4. **If the test still fails even with the dependency present** (e.g. the test has a genuine pandas/upstream bug that happens regardless), xfail it with an explanation:

```python
"tests/io/excel/test_openpyxl.py::test_name": "<root cause explanation>",
```

The explanation string in the xfail dict should describe the *root cause* (e.g. "openpyxl limitation", "pandas test bug: assumes xlsxwriter present"), not just the error message.
---

## Step 5 — Verify the Fix

Three checks are required. Run them in order.

**a. Target test passes:**

```bash
bash python/cudf/cudf/pandas/scripts/run-pandas-tests.sh "<node_id>" -xvs
```

Expected: exit code 0, test shows `PASSED`.

**b. Fix runs on GPU (no silent fallback):**

```bash
CUDF_PANDAS_FAIL_ON_FALLBACK=1 bash python/cudf/cudf/pandas/scripts/run-pandas-tests.sh \
    "<node_id>" -xvs
```

Expected: test still passes. If this fails, the fix works by falling back to pandas rather than actually fixing cudf — that is not acceptable.

Exception: some tests intentionally validate fallback behavior. If `FAIL_ON_FALLBACK` causes this test to fail but the test logic requires fallback, skip this check for that specific test and note the justification.

**c. No regressions in the module:**

```bash
bash python/cudf/cudf/pandas/scripts/run-pandas-tests.sh \
    "tests/<module_directory>/" --tb=line -q
```

Replace `<module_directory>` with the directory containing your test (e.g. `tests/groupby/`). Any new failures that are not already listed in `pandas-testing-plugin.py` must be investigated before committing.

**d. Add unit tests (where appropriate):**

If your fix changes cudf behavior, add a unit test in `cudf` classic or `cudf.pandas` depending on where the fix lives. When adding tests:
- Prefer adding to existing test parameterizations if a related test already exists
- If no related test exists, write a test that compares cudf output to pandas (using `assert_eq` or `assert_exceptions_equal`)
- Place cudf classic tests alongside existing tests in `python/cudf/cudf/tests/`
- Place cudf.pandas-specific tests in `python/cudf/cudf_pandas_tests/`

---

## Step 6 — Commit

Stage only the intended files — never anything from `pandas-testing/`:

```bash
git add python/cudf/cudf/          # source fix (if applicable)
git add python/cudf/cudf/pandas/scripts/pandas-testing-plugin.py   # xfail removal
git status                          # verify nothing from pandas-testing/ is staged
git commit -m "fix(cudf.pandas): <short description>

Fixes the following failing pandas tests:
- tests/groupby/test_reductions.py::test_first_last_skipna[Float64-False-False-first]"
```

If many node IDs share the same root cause, summarize them in the commit message rather than listing every parametrized variant. Commit message should convey what was wrong and what was fixed.

---

## STOP Conditions

Stop immediately and report findings to the user when any of the following apply:

- The fix requires modifying `.pyx`, `.cu`, `.cuh`, or `CMakeLists.txt` files — this requires compilation, which is outside the scope of this skill
- Investigation reveals the behavioral divergence is **intentional** — cudf was deliberately designed to differ from pandas
- You have been through 3 or more investigation cycles without converging on a fix
- The fix would add a new `mode.pandas_compatible` conditional

For intentional divergence: stop and ask the user. In most cases, the goal is to make cudf match pandas. Only when the divergence has significant performance implications should `mode.pandas_compatible` be used — and even then, sparingly. cudf already overuses this flag; the default behavior should agree with pandas.

---

## Important Notes

- `mode.pandas_compatible` is automatically set to `True` when `cudf.pandas` is active. This is done at the end of `python/cudf/cudf/pandas/_wrappers/pandas.py`.
- cudf Python is almost entirely pure Python — for inplace installs, changes to `.py` files take effect immediately without rebuilding.
- The xfail/skip markers are applied by a pytest plugin (`-p cudf.pandas.scripts.pandas-testing-plugin`) loaded fresh on every run, not by appending to the pandas `conftest.py`. The runner sets up `pandas-testing/` only when it is missing and never refreshes it afterward, so if you modify `python/cudf/cudf/pandas/scripts/run-pandas-tests.sh` in a way that changes the `pandas-testing/` directory, delete it before re-running the script. Deleting only `pandas-tests/` re-copies the test tree from the existing clone; to pick up a new pandas version delete the whole `pandas-testing/`, since the `pandas/` clone is guarded separately and is not refetched by removing `pandas-tests/` alone.
- Keys in all three `pandas-testing-plugin.py` dictionaries must remain in alphabetical order.
- Never write comments that explain what old code was replaced — write comments about what the code does and why.
- Never modify the pandas test files themselves — fix cudf, not pandas.
- Never fix the testing APIs (like `assert_frame_equal`, `assert_series_equal`) — fix the actual APIs that produce wrong results.
- First see if the problem is in cudf classic and fix it there; if not, then move over to cudf.pandas.
- The pandas-tests harness runs with `xfail_strict = false` (vendored `pandas-tests/pyproject.toml`, to tolerate flaky XPASSes — issue #22681), so a stale `NODEIDS_THAT_FAIL` entry that now passes shows up as a non-failing `XPASS` and won't flag itself. While testing your fix, flip `xfail_strict` to `true` in that file so the `XPASS` surfaces (do not commit that change) and remove the stale entry so the test reports a genuine `PASSED`.
- When a fix requires adding a test dependency, update `dependencies.yaml` (under `test_cudf_pandas_pandas_tests` for conda environments) and run `rapids-dependency-file-generator` to propagate. Never manually edit the generated `pyproject.toml` entries marked as auto-generated.
- Always verify vanilla pandas behavior before implementing proxy-layer fixes. If the test also fails without cudf.pandas, the problem is upstream or environmental, not a cudf bug.
- xfail explanation strings should describe the root cause ("openpyxl limitation", "pandas test assumes xlsxwriter is installed"), not just the error type ("AssertionError", "IndexError").
