---
name: update-pandas-testing
description: Use when updating pandas for cudf.pandas testing in CI, or when updating the skip and expected-failure lists in python/cudf/cudf/pandas/scripts/pandas-testing-plugin.py
---

# pandas testing for cudf.pandas

This rule describes the procedure for updating the pandas unit testing of cudf.pandas in CI. A successful update involves verifying locally that the pandas unit tests run by `ci/cudf_pandas_scripts/pandas-tests/run.sh` all pass.

## Prerequisites
1. Prompt the user for a development environment. Assume that the provided environment already contains cudf, the target version of pandas, and the pandas test dependencies.
2. If the user does not provide an environment, create one from the file in `conda/environments/` that matches the CUDA version and CPU architecture of the machine (`all_cuda-<CUDA major><CUDA minor>_arch-<arch>.yaml`). These files contain pandas and the `test_python_pandas_tests` dependencies from `dependencies.yaml`, but not cudf. Build and install cudf into the new environment in editable mode.

    For example, to create a fictional conda development environment named "cudf-dev" on an x86_64 machine with CUDA 13
    ```bash
    conda env create -n cudf-dev -f conda/environments/all_cuda-133_arch-x86_64.yaml
    conda run -n cudf-dev ./build.sh libcudf pylibcudf cudf --pydevelop
    ```

Notes on the development environment:
- cudf must be installed in editable mode from this repository. `run-pandas-tests.sh` loads the plugin as `-p cudf.pandas.scripts.pandas-testing-plugin`, so the plugin is imported from the installed cudf package. If cudf is not an editable install, edits to `python/cudf/cudf/pandas/scripts/pandas-testing-plugin.py` have no effect. Verify with `python -c "import cudf; print(cudf.__file__)"`.
- The environment may contain extra packages, or different versions of optional packages, than the `pandas-tests` CI job installs. These can cause spurious local-only failures. Do not add these to the plugin; confirm a failure is reproducible in CI's dependency set before recording it.
- `run-pandas-tests.sh` creates a `pandas-testing/` directory in the current working directory. It clones the `v<installed pandas version>` tag into `pandas-testing/pandas` and copies the tests into `pandas-testing/pandas-tests`, but only if those directories do not already exist. If they exist from an earlier pandas version, the old tests are silently reused. Always run the script from a fresh working directory (outside the cudf repository) for each pandas version.

## The `pandas-testing-plugin.py` dictionaries

`python/cudf/cudf/pandas/scripts/pandas-testing-plugin.py` marks tests to skip or as expected to fail before the tests are run. Each dictionary has keys that are pytest node ids of the form `tests/<path>.py::<test>[<params>]` and values that are a short reason. The node ids are relative to the `pandas-testing/pandas-tests` directory, which is where pytest runs. Keep the keys in each dictionary in alphabetical order.

| Dictionary | Marker applied | Use for |
|---|---|---|
| `NODEIDS_TO_SKIP` | `skip` | Tests of behavior that cudf.pandas cannot support, for example tests of private pandas APIs |
| `NODEIDS_PATHS_TO_SKIP` | `skip` | Keys are node id prefixes; skips a whole file, directory, or test name |
| `NODEIDS_TO_SKIP_WHEN_SHARDED` | `skip`, only with `--num-shards` > 1 | Tests that fail only because sharding changes the test order |
| `NODEIDS_THAT_MAY_FAIL` | `xfail(strict=False)` | Flaky or environment-sensitive tests |
| `NODEIDS_THAT_FAIL` | `xfail` (strict) | Tests that fail because of a bug in cudf.pandas, or a bug or bad assumption in pandas |

`pytest_collection_modifyitems` checks the dictionaries in this order and applies only the first match: `NODEIDS_TO_SKIP`, `NODEIDS_TO_SKIP_WHEN_SHARDED`, `NODEIDS_PATHS_TO_SKIP`, `NODEIDS_THAT_MAY_FAIL`, `NODEIDS_THAT_FAIL`.

The vendored `pyproject.toml` in `run-pandas-tests.sh` sets `xfail_strict = true`. So a test in `NODEIDS_THAT_FAIL` that now passes is reported as `[XPASS(strict)]` and fails the run.

## Skip tests of non-public pandas APIs

cudf.pandas only aims to support the public pandas API. A unit test that exercises a non-public pandas API belongs in `NODEIDS_TO_SKIP` or `NODEIDS_PATHS_TO_SKIP` with a reason that names one non-public API that the test uses, for example `"Tests private pandas APIs, for example pandas.core.apply.reconstruct_func"` or `"Tests private pandas APIs, for example the _mgr attribute"`, regardless of whether it currently passes or fails. The public pandas API is the set of objects documented in the `.rst` files in `doc/source/reference` of the pandas repository.

The scripts in `.agents/skills/update-pandas-testing/scripts/` make this determination. Run them with the plain `python` of the development environment, not with `-m cudf.pandas`.

1. Clone the pandas repository at the tag of the installed pandas version into `pandas-testing/pandas` of the fresh working directory. `run-pandas-tests.sh` reuses this clone, so the analyzed tests and the run tests are the same.
    ```bash
    PANDAS_VERSION=$(python -c "import pandas; print(pandas.__version__)")
    git clone https://github.com/pandas-dev/pandas --depth=1 -b "v${PANDAS_VERSION}" pandas-testing/pandas
    ```
2. Collect the node ids of the tests. This also creates `pandas-testing/pandas-tests`. The script exits with status 1 after a `--collect-only` run because it finds no report log to move; this is expected.
    ```bash
    bash <cudf repo>/python/cudf/cudf/pandas/scripts/run-pandas-tests.sh --collect-only -q > collected.txt
    ```
3. Collect the public API into a text file with one fully qualified name per line, for example `pandas.DataFrame.groupby`. Use this file to check by hand whether a name is public, for example with `grep -x "pandas.Series.str.lower" public_api.txt`.
    ```bash
    python <cudf repo>/.agents/skills/update-pandas-testing/scripts/collect_public_api.py pandas-testing/pandas > public_api.txt
    ```
4. Report the test functions that reference non-public pandas APIs. Pass test files or directories relative to `pandas-tests` to limit the report.
    ```bash
    python <cudf repo>/.agents/skills/update-pandas-testing/scripts/find_private_api_tests.py \
      public_api.txt pandas-testing/pandas-tests > private_api_report.txt
    ```
    The script statically analyzes each test function, its decorators, and the fixtures and helpers that it uses from the same file. It flags two kinds of references:
    * A pandas object, reached through an import or an attribute chain such as `pd.core.common.foo`, that is not in `public_api.txt` and is not the same object as an entry in it. For example, `pandas.core.arrays.DatetimeArray` is public because it is `pandas.arrays.DatetimeArray`.
    * An underscore attribute such as `df._mgr` or `idx._data` that is not a documented member name (for example `ExtensionArray._from_sequence`) and is not defined by the test code in the same directory.

    It does not flag test infrastructure such as `pandas._testing` and `pandas.compat` (`TEST_INFRASTRUCTURE` in the script), the `pandas.api` namespace, or the IPython display methods such as `_repr_html_`. Each finding that comes from a fixture or helper is marked `(via <name>)`.
5. Review every flagged test function in the report. Do not add a test only because the script flags it. Add it only if the non-public API is part of what the test checks, not only part of its setup. Findings marked `(via <name>)` are often setup, for example a fixture that uses `pandas.core.computation` to choose its parameters for tests of the public `pd.eval`. If a flagged name is public but missing from the reference documentation, add it to `UNDOCUMENTED_PUBLIC` in the script instead.
6. Print the candidate dictionary entries. Tests that the plugin already skips are omitted. Without `--include-indirect`, only tests with direct findings are included.
    ```bash
    python <cudf repo>/.agents/skills/update-pandas-testing/scripts/find_private_api_tests.py \
      public_api.txt pandas-testing/pandas-tests \
      --format plugin \
      --collected collected.txt \
      --plugin <cudf repo>/python/cudf/cudf/pandas/scripts/pandas-testing-plugin.py > private_api_entries.txt
    ```
    If every test function in a file is flagged, the file is printed as a `NODEIDS_PATHS_TO_SKIP` candidate. Otherwise, each collected node id of a flagged test function is printed as a `NODEIDS_TO_SKIP` candidate. The reason of each entry names the first finding of the test function in the report, preferring a direct finding over one marked `(via <name>)`; a file's reason names the first finding of its first test function. Add the entries that the review confirmed, and remove each of these tests from `NODEIDS_TO_SKIP_WHEN_SHARDED`, `NODEIDS_THAT_FAIL` and `NODEIDS_THAT_MAY_FAIL`. `NODEIDS_TO_SKIP` is checked first, so an entry for the same test in another dictionary never applies.
    If the review found that the named API is only setup, change the reason to name the non-public API that the test checks instead, in the same `"Tests private pandas APIs, for example <API>"` form. Write a fully qualified name such as `pandas.core.dtypes.cast.dict_compat` for an object, and `the <name> attribute` for an underscore attribute.

The static analysis does not see tests that a test class inherits from a base class in another file (for example the classes in `tests/extension/base/`), or fixtures defined in a `conftest.py`. Review these by hand if a failure in the next step points at a non-public API.

## Iterate and classify test failures from `run-pandas-tests.sh`

Next, run the pandas unit tests with cudf.pandas until there are no test failures.

### Remove stale entries

Before running the pandas unit tests, remove dictionary entries whose node id no longer points to a test in the new pandas version, as tests may have been renamed, re-parametrized, or removed. Use the `collected.txt` file from the previous step to list the stale keys:
```python
import runpy

plugin = runpy.run_path(
    "<cudf repo>/python/cudf/cudf/pandas/scripts/pandas-testing-plugin.py"
)
collected = {
    line.strip() for line in open("collected.txt") if "::" in line
}
for name in (
    "NODEIDS_TO_SKIP",
    "NODEIDS_TO_SKIP_WHEN_SHARDED",
    "NODEIDS_THAT_MAY_FAIL",
    "NODEIDS_THAT_FAIL",
):
    for key in sorted(set(plugin[name]) - collected):
        print(name, key)
for prefix in plugin["NODEIDS_PATHS_TO_SKIP"]:
    if not any(nodeid.startswith(prefix) for nodeid in collected):
        print("NODEIDS_PATHS_TO_SKIP", prefix)
```
The `-m "not slow and not single_cpu and not db and not network"` filter in `run-pandas-tests.sh` deselects some tests. Before removing a key, confirm that it is truly gone from the new pandas test suite (for example with `grep` in `pandas-testing/pandas-tests`) and not only deselected by a marker.

### Remove entries that never apply

`pytest_collection_modifyitems` applies only the first dictionary that matches a test, in the order given in "The `pandas-testing-plugin.py` dictionaries". List the entries that an earlier dictionary already covers:
```python
import runpy

plugin = runpy.run_path(
    "<cudf repo>/python/cudf/cudf/pandas/scripts/pandas-testing-plugin.py"
)
skip = set(plugin["NODEIDS_TO_SKIP"])
paths = tuple(plugin["NODEIDS_PATHS_TO_SKIP"])
may_fail = set(plugin["NODEIDS_THAT_MAY_FAIL"])
for name in (
    "NODEIDS_TO_SKIP_WHEN_SHARDED",
    "NODEIDS_THAT_MAY_FAIL",
    "NODEIDS_THAT_FAIL",
):
    for key in sorted(set(plugin[name]) & skip):
        print(name, key, "is in NODEIDS_TO_SKIP")
for name in ("NODEIDS_TO_SKIP", "NODEIDS_THAT_MAY_FAIL", "NODEIDS_THAT_FAIL"):
    for key in sorted(plugin[name]):
        if key.startswith(paths):
            print(name, key, "is under a NODEIDS_PATHS_TO_SKIP prefix")
for key in sorted(set(plugin["NODEIDS_THAT_FAIL"]) & may_fail):
    print("NODEIDS_THAT_FAIL", key, "is in NODEIDS_THAT_MAY_FAIL")
```
Remove each listed entry from the dictionary named first on its line. The one exception is a `NODEIDS_TO_SKIP` entry under a `NODEIDS_PATHS_TO_SKIP` key that is the node id of that single test. In that case, keep the `NODEIDS_TO_SKIP` entry and remove the `NODEIDS_PATHS_TO_SKIP` entry. A test in both `NODEIDS_TO_SKIP_WHEN_SHARDED` and `NODEIDS_THAT_MAY_FAIL` is intended: it is skipped when sharded and allowed to fail otherwise. Run this check again before sorting the dictionaries.

### Run the tests

1. Run the same `run-pandas-tests.sh` command that `ci/cudf_pandas_scripts/pandas-tests/run.sh` runs, from the fresh working directory, and pipe the output to a file. Omit `--junitxml` and the sharding arguments, so the full suite runs unsharded like the nightly job:
    ```bash
    bash <cudf repo>/python/cudf/cudf/pandas/scripts/run-pandas-tests.sh \
      --numprocesses 8 \
      --tb=line \
      --max-worker-restart=3 \
      --dist worksteal \
      --report-log=pr.json > pandas-tests-output.txt 2>&1
    ```
    The full suite takes a long time. To iterate faster on a subset, pass test paths relative to `pandas-tests` as extra arguments, for example `tests/groupby`.
2. Review the test output and collect the failing tests. The script moves the report log to `pandas-testing/pr.json`. Each line is a JSON test report, and the `nodeid` values are in the dictionary key format:
    ```bash
    jq -r 'select(.["$report_type"] == "TestReport" and .outcome == "failed") | .nodeid' pandas-testing/pr.json | sort -u
    ```
    `python <cudf repo>/python/cudf/cudf/pandas/scripts/analyze-test-failures.py pandas-testing/pr.json "<file-or-pattern>"` prints the most common failures for a group of tests.
3. For each failing test:
    * If a test is failing because it exercises behavior that cannot be supported in cudf.pandas, add a key, value pair to the `NODEIDS_TO_SKIP` dictionary where the key is the node id and the value is a short reason for skipping. If every test in a file or directory fails for the same reason, add the prefix to `NODEIDS_PATHS_TO_SKIP` instead.
    * If a test is failing because of a bug in cudf.pandas, or a bug or bad assumption in pandas, add a key, value pair to the `NODEIDS_THAT_FAIL` dictionary where the key is the node id and the value is `"TODO: Fails starting with pandas version <version>"`.
    * If a test fails only on some runs, or only because of the installed version of an optional dependency, add a key, value pair to `NODEIDS_THAT_MAY_FAIL` where the key is the node id and the value is `"TODO: Fails starting with pandas version <version>"`.
    * In both reasons, `<version>` is the pandas version installed in the development environment (`python -c "import pandas; print(pandas.__version__)"`), for example `"TODO: Fails starting with pandas version 3.0.6"`.
    * If a test in `NODEIDS_THAT_FAIL` reports `[XPASS(strict)]`, it now passes. Remove it from `NODEIDS_THAT_FAIL`.
    * If a test not in the plugin reports `[XPASS(strict)]` (pandas marks it `xfail(strict=True)` but it passes under cudf.pandas), it **must** go in `NODEIDS_TO_SKIP`, not `NODEIDS_THAT_FAIL`. Adding an `xfail` marker does not override pandas' own strict marker, so the strict XPASS would still fail.
4. Repeat the process until the test output reports no failures.

### Run the tests with sharding

Pull request CI splits the suite across shards (see `NUM_SHARDS` in the `pandas-tests-shards` job in `.github/workflows/pr.yaml`). Sharding changes the order the tests run in, which exposes order-dependent behavior in cudf.pandas. After the unsharded run passes, run each shard by adding `--shard-id <i> --num-shards <NUM_SHARDS>` to the command above for every `i` from `0` to `NUM_SHARDS - 1`. Add tests that fail only in a sharded run to `NODEIDS_TO_SKIP_WHEN_SHARDED`, and repeat until every shard reports no failures.

Importantly when evaluating failing tests, do **not** attempt to fix cudf.pandas to solve the failures. This is out of scope.

## Sort the dictionaries

Once the unsharded run and every shard pass, sort the keys of every dictionary in `pandas-testing-plugin.py` alphabetically:
```bash
python <cudf repo>/.agents/skills/update-pandas-testing/scripts/sort_plugin_dictionaries.py \
  <cudf repo>/python/cudf/cudf/pandas/scripts/pandas-testing-plugin.py
```
The script requires each entry to start on its own line, and exits with an error without changing the file if one does not. Comment lines directly above an entry move with that entry. If a comment describes several entries, check that it still reads correctly after the sort, and move or repeat it if it does not. Pass `--check` to only verify the order; the script then exits with status 1 if a dictionary is not sorted.

Finally, report the new test failures, the tests skipped as tests of non-public pandas APIs, and the removed stale entries for each dictionary.
