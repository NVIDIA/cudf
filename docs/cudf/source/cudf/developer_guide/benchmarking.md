# Benchmarking cuDF

The goal of the benchmarks in this repository is to measure the performance of various cuDF APIs.
Benchmarks in cuDF are written using the
[`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/en/latest/index.html) plugin to the
[`pytest`](https://docs.pytest.org/en/latest/) Python testing framework.
Using `pytest-benchmark` provides a seamless experience for developers familiar with `pytest`.
We include benchmarks of both public APIs and internal functions.
The former give us a macro view of our performance, especially vis-à-vis pandas.
The latter help us quantify and minimize the overhead of our Python bindings.

```{note}
Our current benchmarks focus entirely on measuring run time.
However, minimizing memory footprint can be just as important for some cases.
In the future, we may update our benchmarks to also include memory usage measurements.
```

## Benchmark organization

At the top level benchmarks are divided into `internal` and `API` directories.
API benchmarks are for public features that we expect users to consume.
Internal benchmarks capture the performance of cuDF internals that have no stability guarantees.

Within each directory, benchmarks are organized based on the type of function.
Functions in cuDF generally fall into two groups:

1. Methods of classes like `DataFrame` or `Series`.
2. Free functions operating on the above classes like `cudf.merge`.

The former should be organized into files named `bench_class.py`.
For example, benchmarks of `DataFrame.eval` belong in `API/bench_dataframe.py`.
Benchmarks should be written at the highest level of generality possible with respect to the class hierarchy.
For instance, all classes support the `take` method, so those benchmarks belong in `API/bench_frame_or_index.py`.
If a method has a slightly different API for different classes, benchmarks should use a minimal common API,
_unless_ developers expect certain arguments to trigger code paths with very different performance characteristics.
One example, is `DataFrame.where`, which supports a wide range of inputs (like other `DataFrame`s) that other classes don't support.
Therefore, we have separate benchmarks for `DataFrame`, in addition to the general benchmarks for all `Frame` and `Index` classes.

```{note}
`pytest` does not support having two benchmark files with the same name, even if they are in separate directories.
Therefore, benchmarks of internal methods of _public_ classes go in files suffixed with `_internal`.
Benchmarks of `DataFrame._apply_boolean_mask`, for instance, belong in `internal/bench_dataframe_internal.py`.
```

Free functions have more flexibility.
Broadly speaking, they should be grouped into benchmark files containing similar functionality.
For example, I/O benchmarks can all live in `bench_io.py`.
For now those groupings are left to the discretion of developers.

## Running benchmarks

By default, pytest discovers test files and functions prefixed with `test_`.
For benchmarks, we configure `pytest` to instead search using the `bench_` prefix.
After installing `pytest-benchmark`, running benchmarks is as simple as just running `pytest`.

When benchmarks are run, the default behavior is to output the results in a table to the terminal.
A common requirement is to then compare the performance of benchmarks before and after a change.
We can generate these comparisons by saving the output using the `--benchmark-autosave` option to pytest.
When using this option, after the benchmarks are run the output will contain a line:
```
Saved benchmark data in: /path/to/XXXX_*.json
```

The `XXXX` is a four-digit number identifying the benchmark.
If preferred, a user may also use the `--benchmark-save=NAME` option,
which allows more control over the resulting filename.
Given two benchmark runs `XXXX` and `YYYY`, benchmarks can then be compared using
```
pytest-benchmark compare XXXX YYYY
```
Note that the comparison uses the `pytest-benchmark` command rather than the `pytest` command.
`pytest-benchmark` has a number of additional options that can be used to customize the output.
The next line contains one useful example, but developers should experiment to find a useful output
```
pytest-benchmark compare XXXX YYYY --sort="name" --columns=Mean --name=short --group-by=param
```

For more details, see the [`pytest-benchmark` documentation](https://pytest-benchmark.readthedocs.io/en/latest/comparing.html).

## Benchmark contents

### Benchmark configuration

Benchmarks must support [comparing to pandas](#comparing-to-pandas) and [being run as tests](#testing-benchmarks).
To satisfy these requirements, one must follow these rules when writing benchmarks:
1. Import `cudf` and `cupy` from the config module:
   ```python
       from ..common.config import cudf, cupy # Do this
       import cudf, cupy # Not this
   ```
   This enables swapping out for `pandas` and `numpy` respectively.
2. Avoid hard-coding benchmark dataset sizes, and instead use the sizes advertised by `config.py`.
   This enables running the benchmarks in "test" mode on small datasets, which will be much faster.


### Writing benchmarks

Just as benchmarks should be written in terms of the highest level classes in the hierarchy,
they should also assume as little as possible about the nature of the data.
For instance, unless there are meaningful functional differences,
benchmarks should not care about the dtype or nullability of the data.
Objects that differ in these ways should be interchangeable for most benchmarks.
The goal of writing benchmarks in this way is to then automatically benchmark objects with different properties.
We support this use case with the `benchmark_with_object` decorator.

The use of this decorator is best demonstrated by example:

```python
@benchmark_with_object(cls="dataframe", dtype="int", cols=6)
def bench_foo(benchmark, dataframe):
    benchmark(dataframe.foo)
```

In the example above `bench_foo` will be run for DataFrames containing six columns of integer data.
The decorator allows automatically parametrizing the following object properties:

- `cls`: Objects of a specific class, e.g. `DataFrame`.
- `dtype`: Objects of a specific dtype.
- `nulls`: Objects with and without null entries.
- `cols`: Objects with a certain number of columns.
- `rows`: Objects with a certain number of rows.

In the example, since we did not specify the number of rows or nullability,
it will be run once for each valid number of rows and for both nullable and non-nullable data.
The valid set of all parameters (e.g. the numbers of rows) is stored in the `common/config.py` file.
This decorator allows a developer to write a generic benchmark that works for many types of objects,
then have that benchmark automatically run for all objects of interest.

### Parametrizing tests

The `benchmark_with_object` decorator covers most use cases and automatically guarantees a baseline of benchmark coverage.
However, many benchmarks will require more customized objects.
In some cases those will be the primary targets whose methods are called.
For instance, a benchmark may require a `Series` with a specific data distribution.
In others, those objects will be arguments passed to other functions.
An example of this is `DataFrame.where`, which accepts many types of objects to filter with.

In the first case, fixtures should follow certain rules.
When writing fixtures, developers should make the data sizes dependent on the benchmarks configuration.
The `benchmarks/common/config.py` file defines standard data sizes to be used in benchmarks.
These data sizes can be tweaked for debugging purposes (see [](#testing-benchmarks) below).
Fixture sizes should be relative to the `NUM_ROWS` and/or `NUM_COLS` variables defined in the config module.
These rules ensure consistency between these fixtures and those provided by `benchmark_with_object`.


## Comparing to pandas

An important aspect of benchmarking cuDF is comparing it to pandas.
We often want to generate quantitative comparisons, so we need to make that as easy as possible.
Our benchmarks support this by setting the environment variable `CUDF_BENCHMARKS_USE_PANDAS`.
When this variable is detected, all benchmarks will automatically be run using pandas instead of cuDF.
Therefore, comparisons can easily be generated by simply running the benchmarks twice,
once with the variable set and once without.
Note that this variable only affects API benchmarks, not internal benchmarks,
since the latter are not even guaranteed to be valid pandas code.

```{note}
`CUDF_BENCHMARKS_USE_PANDAS` effectively remaps `cudf` to `pandas` and `cupy` to `numpy`.
It does so by aliasing these modules in `common.config.py`.
This aliasing is why it is critical for developers to import these packages from `config.py`.
```

## Testing benchmarks

Benchmarks need to be kept up to date with API changes in cuDF.
However, we cannot simply run benchmarks in CI.
Doing so would consume too many resources, and it would significantly slow down the development cycle

To balance these issues, our benchmarks also support running in "testing" mode.
To do so, developers can set the `CUDF_BENCHMARKS_DEBUG_ONLY` environment variable.
When benchmarks are run with this variable, all data sizes are set to a minimum and the number of sizes are reduced.
Our CI testing takes advantage of this to ensure that benchmarks remain valid code.

```{note}
The objects provided by `benchmark_with_object` respect the `NUM_ROWS` and `NUM_COLS` defined in `common/config.py`.
`CUDF_BENCHMARKS_DEBUG_ONLY` works by conditionally redefining these values.
This is why it is crucial for developers to use these variables when defining custom fixtures or cases.
```

## Profiling

Although not strictly part of our benchmarking suite, profiling is a common need so we provide some guidelines here.
Here are two easy ways (there may be others) to profile benchmarks:
1. The [`pytest-profiling`](https://github.com/man-group/pytest-plugins/tree/master/pytest-profiling) plugin.
2. The [`py-spy`](https://github.com/benfred/py-spy) package.

Using the former is as simple as adding the `--profile` (or `--profile-svg`) arguments to the `pytest` invocation.
The latter requires instead invoking pytest from py-spy, like so:
```
py-spy record -- pytest bench_foo.py
```
Each tool has different strengths and provides somewhat different information.
Developers should try both and see what works for a particular workflow.
Developers are also encouraged to share useful alternatives that they discover.

## Advanced Topics

This section discusses some underlying details of how cuDF benchmarks work.
They are not usually necessary for typical developers or benchmark writers.
This information is primarily for developers looking to extend the types of objects that can be easily benchmarked.

### Understanding `benchmark_with_object`

Under the hood, `benchmark_with_object` is made up of two critical pieces, fixture unions and some decorator magic.

#### Fixture unions

Fixture unions are a feature of `pytest_cases`.
A fixture union is a fixture that, when used as a test function parameter,
will trigger the test to run once for each fixture contained in the union.
Since most cuDF benchmarks can be run with the same relatively small set of objects,
our benchmarks generate the Cartesian product of possible fixtures and then create all possible unions.

This feature is critical to the design of our benchmarks.
For each of the relevant parameter combinations (size, nullability, etc) we programmatically generate a new fixture.
The resulting fixtures are unambiguously named according to the following scheme:
`{classname}_dtype_{dtype}[_nulls_{true|false}][[_cols_{num_cols}]_rows_{num_rows}]`.
If a fixture name does not contain a particular component, it represents a union of all values of that component.
As an example, consider the fixture `dataframe_dtype_int_rows_100`.
This fixture is a union of both nullable and non-nullable `DataFrame`s with different numbers of columns.

#### The `benchmark_with_object` decorator

The long names of the above unions are cumbersome when writing tests.
Moreover, having this information embedded in the name means that in order to change the parameters used,
the entire benchmark needs to have the fixture name replaced.
The `benchmark_with_object` decorator is the solution to this problem.
When used on a test function, it essentially replaces the function parameter name with the true fixture.
In our original example from above

```python
@benchmark_with_object(cls="dataframe", dtype="int", cols=6)
def bench_foo(benchmark, dataframe):
    benchmark(dataframe.foo)
```

is functionally equivalent to

```python
def bench_foo(benchmark, dataframe_dtype_int_cols_6):
    benchmark(dataframe_dtype_int_cols_6.foo)
```
