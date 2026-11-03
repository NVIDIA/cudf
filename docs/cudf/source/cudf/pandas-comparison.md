# Comparison of cuDF and pandas

cuDF is a DataFrame library that closely matches the pandas API but
behaviorally is *not* a full drop-in replacement for pandas.
This page documents the similarities and differences
between cuDF and pandas.

cuDF also provides a pandas
accelerator mode (`cudf.pandas`) that supports 100% of the pandas API
and accelerates pandas code on the GPU without requiring any code
change. Visit the [`cudf.pandas` documentation](../cudf_pandas/index)
if you have existing pandas code that you want to accelerate with cuDF.

For a detailed list of API-level behavioral differences, see the
[pandas Compatibility Notes](PandasCompat).

## Supported operations

cuDF supports many of the same data structures and operations as
pandas. This includes `Series`, `DataFrame`, `Index` and
operations on them such as unary and binary operations, indexing,
filtering, concatenating, joining, groupby and window operations -
among many others.

The best way to check if we support a particular pandas API is to search
our [API docs](/cudf/api_docs/index).

## Data types

cuDF supports all the data types in pandas except for `pandas.PeriodDtype`, `pandas.SparseDtype`
and third-party `ExtensionDtype`s from other libraries.
In addition, cuDF supports data types for decimal, list,
and "struct" values. See the section on [Data Types](data-types) for
details.

### No true `"object"` data type

In pandas and NumPy, the `"object"` data type can represent data
of arbitrary Python objects.

```{code} python
>>> import pandas as pd
>>> s = pd.Series(["a", 1, [1, 2, 3]])
0            a
1            1
2    [1, 2, 3]
dtype: object
```

cuDF can use `"object"` to represent string data but does *not* support storing or operating on
collections of arbitrary Python objects.

## Null (or "missing") values

Unlike pandas, missing values are represented by the same
missing value indicator, `cudf.NA`.

```{code} python
>>> s = cudf.Series([1, 2, cudf.NA])
>>> s
0       1
1       2
2    <NA>
dtype: int64
```

Nulls are not coerced to `NaN` in any situation;
compare the behavior of cuDF with pandas below:

```{code} python
>>> s = cudf.Series([1, 2, cudf.NA], dtype="category")
>>> s
0       1
1       2
2    <NA>
dtype: category
Categories (2, int64): [1, 2]

>>> s = pd.Series([1, 2, pd.NA], dtype="category")
>>> s
0      1
1      2
2    NaN
dtype: category
Categories (2, int64): [1, 2]
```

See the docs on [missing data](missing-data) for details.

(pandas-comparison/iteration)=

## Iteration

Iterating over a cuDF `Series`, `DataFrame` or `Index` is not
supported. This is because iterating over data that resides on the GPU
will yield *extremely* poor performance, as GPUs are optimized for
highly parallel operations rather than sequential operations.

In the vast majority of cases, it is possible to avoid iteration and
use an existing function or method to accomplish the same task. If you
absolutely must iterate, copy the data from GPU to CPU by using
`.to_arrow()` or `.to_pandas()`, then convert the result back to GPU
using a `Series`, `DataFrame` or `Index` constructor.

## Result ordering

In pandas, `join` (or `merge`), `value_counts` and `groupby` operations provide
certain guarantees about the order of rows in the result returned.  In a pandas
`join`, the order of join keys is (depending on the particular style of join
being performed) either preserved or sorted lexicographically by default.
`groupby` sorts the group keys, and preserves the order of rows within each
group. In some cases, disabling this option in pandas can yield better
performance.

By contrast, cuDF's default behavior is to return rows in a
non-deterministic order to maximize performance.  Compare the results
obtained from pandas and cuDF below:

```{code} python
>>> import cupy as cp
>>> cp.random.seed(0)
>>> import cudf
>>> df = cudf.DataFrame({'a': cp.random.randint(0, 1000, 1000), 'b': range(1000)})
>>> df.groupby("a").mean().head()
         b
a
29   193.0
803  915.0
5    138.0
583  300.0
418  613.0
>>> df.to_pandas().groupby("a").mean().head()
            b
a
0   70.000000
1  356.333333
2  770.000000
3  838.000000
4  342.000000
```

In most cases, the rows of a DataFrame are accessed by index labels
rather than by position, so the order in which rows are returned
doesn't matter. However, if you require that results be returned in a
predictable (sorted) order, you can pass the `sort=True` option
explicitly or enable the `mode.pandas_compatible` option when trying
to match pandas behavior with `sort=False`:

```{code} python
>>> df.groupby("a", sort=True).mean().head()
         b
a
0   70.000000
1  356.333333
2  770.000000
3  838.000000
4  342.000000

>>> cudf.set_option("mode.pandas_compatible", True)
>>> df.groupby("a").mean().head()
            b
a
0   70.000000
1  356.333333
2  770.000000
3  838.000000
4  342.000000
```

## Floating-point computation

cuDF leverages GPUs to execute operations in parallel.  This means the
order of operations is not always deterministic.  This impacts the
determinism of floating-point operations because floating-point
arithmetic is non-associative, that is, `(a + b) + c` is not necessarily equal to `a + (b + c)`.

For example, `s.sum()` is not guaranteed to produce identical results
to pandas or produce identical results from run to run, when `s` is a
Series of floats. If you need to compare floating point results, you
should typically do so using the functions provided in the
[`cudf.testing`](/cudf/api_docs/general_utilities)
module, which allow you to compare values up to a desired precision.

## Column names

Unlike pandas, cuDF does not support `DataFrame`s with duplicate column names.

## Writing a DataFrame to Parquet with non-string column names

When there is a DataFrame with non-string column names, pandas casts each
column name to `str` before writing to a Parquet file. `cudf` raises an
error by default if this is attempted. However, to achieve similar behavior
as pandas you can enable the `mode.pandas_compatible` option, which will
enable `cudf` to cast the column names to `str` just like pandas.

```python
>>> import cudf
>>> df = cudf.DataFrame({1: [1, 2, 3], "1": ["a", "b", "c"]})
>>> df.to_parquet("df.parquet")

Traceback (most recent call last):
ValueError: Writing a Parquet file requires string column names
>>> cudf.set_option("mode.pandas_compatible", True)
>>> df.to_parquet("df.parquet")

UserWarning: The DataFrame has column names of non-string type. They will be converted to strings on write.
```

## `.apply()` function limitations

The `.apply()` function in pandas accepts a user-defined function
(UDF) that can include arbitrary operations that are applied to each
value of a `Series`, `DataFrame`, or in the case of a groupby,
each group.  cuDF also supports `.apply()`, but it relies on Numba to
JIT compile the UDF and execute it on the GPU. This can be extremely
fast, but imposes a few limitations on what operations are allowed in
the UDF. See the docs on [UDFs](guide-to-udfs) for details.
