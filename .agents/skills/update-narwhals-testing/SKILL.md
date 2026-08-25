---
name: update-narwhals-testing
description: Use when updating Narwhals for cudf-polars and cudf testing in CI
---

# Narwhals testing for cudf-polars and cudf

This rule describes the procedure for updating Narwhals unit testing for cudf-polars
and cudf in CI. A successful update involves verifying locally that the variations of Narwhals unit test run in `ci/test_narwhals.sh` all pass.

## Prerequisites
1. Prompt the user for a suitable development environment that contains both cudf and cudf-polars.
2. Prompt the user for the target Narwhals version for the upgrade.

Notes on the development environment:
- The environment must also have `pytest-env` installed (it is a CI-only dependency and is often missing from local dev environments). Without it, the runs fail with `ImportError: Error importing plugin "env"`.
- The environment may contain extra packages that the `test_python_narwhals` CI job does not install (for example `dask`). Such packages can cause spurious local-only failures (for example `tests/tpch_q1_test.py::test_q1[dask]`, which `pytest.importorskip`s away in CI). Do not add these to a plugin; confirm a failure is reproducible in CI's dependency set before recording it.

## Step 1. Bump Narwhals in `dependencies.yaml`

In the `depends_on_narwhals` section in `dependencies.yaml`, change the `narwhals==` version pinning to the target Narwhals version

## Step 2. Install Narwhals from a local repository

Identify a directory that contains the Narwhals repository to install it into the development environment. If it is not provided by the user or one cannot be identified, clone it to a temporary location.

Ensure the Narwhals Git source tree is checked-out to the target upgrade version before installing.

[Recent](https://github.com/narwhals-dev/narwhals/issues/3811) Narwhals versions use the `uv_build` build backend, so the editable install requires `uv` and `uv-build` in the environment and must be run with `--no-build-isolation`.

For example, to clone Narwhals and install it in a fictional, conda development environment named "cudf-dev"
```bash
NARWHALS_VERSION="v2.24.0"
git clone https://github.com/narwhals-dev/narwhals.git --single-branch --branch $NARWHALS_VERSION /tmp/narwhals
cd /tmp/narwhals
conda run -n cudf-dev pip install uv uv-build
conda run -n cudf-dev pip install -U -e . --no-build-isolation
```

## Step 3. Iterate and classify test failures from `pytest` invocations in `ci/test_narwhals.sh`

Next, iterate through the various Narwhals unit test invocations in `ci/test_narwhals.sh` via `pytest` until there are no test failures.

Currently 3 variations of Narwhals unit tests are run:

1. With cudf through setting `--constructors=cudf`
2. With cudf-polars through setting the `NARWHALS_POLARS_GPU=1` environment variable and `--constructors=polars[lazy]`
3. With cudf.pandas through setting the `NARWHALS_DEFAULT_CONSTRUCTORS=pandas` environment variable and `-p cudf.pandas`

Each of these variations has an accompanying pytest plugin to mark tests to skip or as expected to fail before the tests are run.

1. The cudf test run uses `ci/narwhals_cudf_test_plugin.py` via `-p narwhals_cudf_test_plugin`
2. The cudf.polars test run uses `ci/narwhals_cudf_polars_test_plugin.py` via `-p narwhals_cudf_polars_test_plugin`
3. The cudf.pandas test run uses `ci/narwhals_cudf_pandas_test_plugin.py` via `-p narwhals_cudf_pandas_test_plugin`

The `TESTS_TO_SKIP` and `EXPECTED_FAILURES` dictionaries in each file contain keys that are valid pytest test identifiers and values that are the reasons for skipping or failure of the test respectively.

Before running Narwhals unit tests, review each `TESTS_TO_SKIP` and `EXPECTED_FAILURES` in the plugin files and remove entries where the pytest test identifier no longer points to a test in Narwhals, as tests may have been removed in a newer Narwhals version.

Next, for each variation of Narwhals unit tests:

1. Run the exact `pytest` command found in `ci/test_narwhals.sh` for the test variation and pipe the output to a file. Run it from within the cloned Narwhals repository (the CI script does `pushd narwhals` first) and set `PYTHONPATH` to the `ci/` directory so the `-p narwhals_*_test_plugin` plugins can be imported, e.g. `PYTHONPATH="<repo>/ci" ...`. Node ids in this working directory are of the form `tests/...py::<test>[...]`, which is exactly the dictionary key format.
2. Review the test output file and collect the failing tests.
3. For each failing test:
    * If a test is failing because it exercises behavior that cannot be supported in cudf or cudf-polars, add a key, value pair to the `TESTS_TO_SKIP` dictionary where the key is in the format `<path>::<pytest_node_id>` and the value is a short reason for skipping.
    * If a test is failing because of a bug in cudf or cudf-polars, or a bug or bad assumption in Narwhals, add a key, value pair to the `EXPECTED_FAILURES` dictionary where the key is in the format `<path>::<pytest_node_id>` and the value is a short reason for failure.
    * If a test reports `[XPASS(strict)]` (Narwhals marks it `xfail(strict=True)` for the constructor but it now passes), it **must** go in `TESTS_TO_SKIP`, not `EXPECTED_FAILURES`. Adding an `xfail` marker does not override Narwhals' own strict marker, so the strict XPASS would still fail; skipping is the equivalent of the old deselection.
4. Repeat the process until the test output file reports no failures.

Importantly when evaluating failing tests, do **not** attempt to fix cudf or cudf-polars to solve the failures. This is out of scope.

Once all variations of Narwhals unit tests pass, report the new test failures for each variation.
