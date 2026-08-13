# Testing cuDF

## Tooling
Tests in cuDF are written using [`pytest`](https://docs.pytest.org/en/latest/).
Test coverage is measured using [`coverage.py`](https://coverage.readthedocs.io/en/latest/),
with the [`pytest-cov`](https://github.com/pytest-dev/pytest-cov) plugin.
Code coverage reports are uploaded to [Codecov](https://app.codecov.io/gh/rapidsai/cudf).
Each PR also indicates whether it increases or decreases test coverage.

### Configuring pytest

Pytest configurations live in `python/cudf/pyproject.toml` with 3 exceptions that use configurations in `pytest.ini` as an override:

- Pytest benchmarks in `python/cudf/benchmarks/pytest.ini`
- cudf.pandas third party tests in `python/cudf/cudf_pandas_tests/third_party_integration_tests/tests/pytest.ini`
- cudf.pandas unit tests in `python/cudf/cudf_pandas_tests/pytest.ini`

These overrides are needed for pytest-plugin or testing configuration-specific set up. Otherwise, ideally all configuration files
should reflect the same, strict pytest runtime configurations.

## Test organization

Generally, the directories under `cudf/tests` describe tests of a certain object (e.g. `series/`)
or a general topic (e.g. `reshape/`, `series/methods`), and the test files are named according to an APIs name
(e.g. `series/methods/test_astype.py`, `reshape/test_concat.py`). Sometimes, tested operations and
APIs do not have a singular name to correspond to the file name and are instead named by a topic as well.
Some common examples include:

- `test_constructors.py`: `__init__` and `@classmethod` constructors for objects
- `test_attributes.py`: `@property`s of objects
- `test_binops.py`: Binary methods (e.g. `+`, `%`)
- `test_reductions.py`: Reduction methods (e.g. `mean`, `quantile`)

The organization aims to have many, specific test files that target a particular operation for a particular object;
therefore, there may be test files with the same name but live in different directories e.g.

- `series/methods/test_astype.py`, `dataframe/methods/test_astype.py`, `indexes/index/methods/test_astype.py`
- `series/methods/test_reductions.py`, `dataframe/methods/test_reductions.py`, `groupby/test_reductions.py`

When adding new tests, make a best-effort to place them in a file according to the tested object and API.

## Test contents

### Writing tests

For an API, unit tests should exercise both standard and exceptional cases.

Standard use cases include:
- Supported input data types for the API
- Supported combination of input data structures, e.g. binary operations between `DataFrame` and `Series`
- Exercising interactions between all API parameters

Exceptional uses cases include:
1. `Series`/`DataFrame`/`Index` with zero rows
2. `DataFrame` with zero columns
3. All null data
4. For string or list APIs, empty strings/lists
5. For list APIs, lists containing all null elements or empty strings
6. For numeric data:
  1. All 0s.
  2. All 1s.
  3. Containing/all inf
  4. Containing/all nan
  5. `INT${PRECISION}_MAX` for a given precision (e.g. `2**32` for `int32`).

Additional exceptional use cases are also dependent on an API's operation e.g. join.

When writing a unit test for an API that mirrors pandas, construct the test body where pandas operations
are done independently of cuDF operations, and assert equality for both results in the end, performing little to
no conversions between pandas and cuDF. For example:

```python
def test_mean():
    data = {"a": [1, 2, None], "b": [2, 3, 4]}
    expected = pd.DataFrame(data).mean()
    result = cudf.DataFrame(data).mean()
    assert_frame_equal(result, expected)
```


Standard use cases may be covered using parametrization (using `pytest.mark.parametrize`).

In general, it is preferable to write separate tests for different exceptional cases.
Excessive parametrization and branching increases complexity and obfuscates the purpose of a test.
Typically, exception cases require specific assertions or other special logic, so they are best kept separate.
The main exception to this rule is tests based on comparison to pandas.
Such tests may test exceptional cases alongside more typical cases since the logic is generally identical.

(test_parametrization)=

### Parametrization: custom fixtures and `pytest.mark.parametrize`

To test multiple, standard and/or exceptional uses cases for the same API, leverage
[pytest fixtures](https://docs.pytest.org/en/latest/explanation/fixtures.html)
and [`@pytest.mark.parametrize`](https://docs.pytest.org/en/latest/how-to/parametrize.html#pytest-mark-parametrize).
when writing a unit test. Note that fixture results are constructed lazily when needed
whereas parametrizations are constructed eagerly at test collection time.

```{warning}
Parameters for `@pytest.mark.parametrize` should have a minimal memory footprint, ideally no GPU memory footprint,
and runtime as these parameters are executed during test collection and objects persist during the entire test duration.
Generally, as much work should be deferred to the test body as possible.

- Avoid constructing fully materialized test data objects if possible. Specify input parameters that eventually construct these objects in the test body instead.
- Avoid calling APIs that would require a long runtime. Defer these calls to the test body with a `lambda` instead.
```

Test parameterization using fixtures or `@pytest.mark.parametrize` should require little to no additional test body logic
to exercise an API and assert a result. If a test body requires branching or significant processing logic to handle a parameterized input,
a separate unit test should be written instead.


A pytest fixture should be used to parameterize inputs for:
- Concept applicable for multiple APIs, e.g. all support cuDF data types.
- Grouping APIs with a shared implementation or similar categorization, e.g. "reduction methods"
- An API parameter with a fixed amount of valid values that exists in multiple APIs, e.g. `nan_is_null`
- Input data that is applicable to multiple APIs

Otherwise use `pytest.mark.parametrize` on a specific test.

(xfailing_tests)=

### Tests with expected failures (`xfail`s)

For tests that are expected fail due to a known deficiency in cuDF or an external dependency
that has the potential to be fixed, use the
[`pytest.mark.xfail`](https://docs.pytest.org/en/stable/reference/reference.html#pytest.mark.xfail)
fixture on the test to note that test is expected to fail.

```{warning}
Do not use [`pytest.xfail`](https://docs.pytest.org/en/stable/reference/reference.html#pytest-xfail)
as it ceases test execution where used. Expected failing test should have an opportunity to
continually execute to see if changes cause the test to pass.
```

A `pytest.mark.xfail` should be scoped as specific as possible to the known failing scenario:
- If all assertions in a test fail for all test inputs, use the `@pytest.mark.xfail` decorator directly on the test
- If a specific parameterized input is expected to fail, use
[`pytest.param`](https://docs.pytest.org/en/stable/how-to/skipping.html#skip-xfail-with-parametrize) with `marks=pytest.mark.xfail`.

```python
@pytest.mark.parametrize(
    "value",
    [
        1,
        2,
        pytest.param(
            3, marks=pytest.mark.xfail(reason="code doesn't work for 3")
        ),
    ],
)
def test_value(value):
    assert value < 3
```
- If it is impractical to use `pytest.param` for a combination of test inputs or only certain assertions fail in the test body,
use `request.applymarker` to dynamically add a `pytest.mark.xfail`

```python
@pytest.mark.parametrize("v1", [1, 2, 3])
@pytest.mark.parametrize("v2", [1, 2, 3])
def test_sum_lt_6(request, v1, v2):
    assert v1 + v2 > 0
    request.applymarker(
        pytest.mark.xfail(
            condition=(v1 == 3 and v2 == 3),
            reason="Add comment linking to relevant issue",
        )
    )
    assert v1 + v2 < 6
```

Additionally, the `reason` passed to `pytest.mark.xfail` should provide a descriptive reason with external links
tracking or providing context to the expected failure.

Configuration files specify [`xfail_strict=true`](https://docs.pytest.org/en/latest/how-to/skipping.html#strict-parameter)
so any expected test failure that passes will fail the test suite run.


(testing_warnings)=

### Testing code that throws warnings

Pytest is invoked with `filterwarnings = ["error"]` so tests should validate that an API call throws the expected warning or address
the warning as appropriate.

- Warnings that are explicitly thrown in cuDF should be tested using the [`pytest.warns`](https://docs.pytest.org/en/latest/how-to/capture-warnings.html#assertwarnings) context manager.
- If a warning originates from a package or testing dependency:
    - If practically addressable, modify the test body or cuDF codebase in a backward compatible way to avoid the warning
    - If not addressable, use `pytest.mark.filterwarnings` if it only impacts a small subset of tests or add a warning filter to the appropriate configuration file if it impacts a large amount of the test suite.
        - A [warning filter](https://docs.python.org/3/library/warnings.html#the-warnings-filter) should include at minimum a `message` and `category` component.

Periodically revisit any warning filters added in configuration files and with `pytest.mark.filterwarnings` as code and
dependency updates may have made them obsolete.

```{warning}
Do not use the [`warnings.catch_warnings`](https://docs.python.org/3/library/warnings.html#warnings.catch_warnings)
context manager to supress warnings. Following the guidance above, explicit validation that a warning occurs is preferred.
```

### Testing utility functions

The `cudf.testing` and `cudf.testing._utils` both provides utilities for testing assertions:

- When comparing specific, known objects, use `assert_frame_equal`, `assert_series_equal` and `assert_index_equal` where applicable.
- When comparing scalar objects, use `assert` directly.
- When comparing objects whose types might not be known or dynamic in a test, use `assert_eq`
- When comparing two calls that should have the same exception behavior, use `assert_exceptions_equal`
- When comparing a singular exception call, use `pytest.raises` as a context manager.

### Version testing

It is recommended to have `cudf` pytests only work on the latest supported pandas version i.e., `PANDAS_CURRENT_SUPPORTED_VERSION`. Any anticipated failures should be either `skipped` or `xfailed`.

For example:

```python
@pytest.mark.skipif(PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION, reason="bug in older version of pandas")
def test_bug_from_older_pandas_versions(...):
    ...

@pytest.mark.xfail(PANDAS_VERSION >= PANDAS_CURRENT_SUPPORTED_VERSION, reason="bug in latest version of pandas")
def test_bug_in_current_and_maybe_future_versions(...):
    ...
```

If pandas makes a bugfix release and fixes this, then we'll see it in CI immediately, patch it, and bump `PANDAS_CURRENT_SUPPORTED_VERSION` which also usually happens during pandas upgrades.

### Parallelization

The majority of our tests are run in parallel using
[pytest-xdist](https://pytest-xdist.readthedocs.io/en/stable/). If a test needs
to be run serially, say because it requires a large fraction of the GPU memory
of our test node, mark with with `@pytest.mark.serial`. Tests marked with
`@serial` will be excluded from the main `pytest` run with `-m not serial` and
run subsequentlly with `-m serial`.
