# Supported Data Types

cuDF largely uses the same [data type objects](https://pandas.pydata.org/docs/user_guide/basics.html#dtypes) supported by pandas, including
numeric, datetime, timedelta, and string data types and their nullable variants. cuDF also supports
data types from the [Arrow type system](https://arrow.apache.org/docs/format/CDataInterface.html#data-type-description-format-strings) such as decimals, list,
and struct types.

All data types in cuDF are [nullable](missing-data).

<div class="special-table">

| Kind of data         | Default data type(s)                                                                                                                     |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| Signed integer       | `'int8'`, `'int16'`, `'int32'`, `'int64'`                                                                                                |
| Unsigned integer     | `'uint8'`, `'uint16'`, `'uint32'`, `'uint64'`                                                                                            |
| Floating-point       | `'float32'`, `'float64'`                                                                                                                 |
| Datetime             | `'datetime64[s]'`, `'datetime64[ms]'`, `'datetime64[us]'`, `'datetime64[ns]'`                                                            |
| Datetime w/ timezone | `pandas.DatetimeTZDtype`                                                                                                                 |
| Timedelta (duration) | `'timedelta64[s]'`, `'timedelta64[ms]'`, `'timedelta64[us]'`, `'timedelta64[ns]'`                                                        |
| Category             | {py:class}`~cudf.core.dtypes.CategoricalDtype`                                                                                           |
| String               | `pandas.StringDtype`                                                                                                                     |
| Decimal              | {py:class}`~cudf.core.dtypes.Decimal32Dtype`, {py:class}`~cudf.core.dtypes.Decimal64Dtype`, {py:class}`~cudf.core.dtypes.Decimal128Dtype`|
| List                 | {py:class}`~cudf.core.dtypes.ListDtype`                                                                                                  |
| Struct               | {py:class}`~cudf.core.dtypes.StructDtype`                                                                                                |
| Interval             | {py:class}`~cudf.core.dtypes.IntervalDtype`                                                                                              |

</div>

```{note}
cuDF does not have an analogous type for `pandas.PeriodDtype` or `pandas.SparseDtype`.
```

## Specifying data types

cuDF APIs with a `dtype` parameter accept the same types of arguments as pandas,
including pandas [nullable types](https://pandas.pydata.org/docs/reference/arrays.html#nullable-integer),
e.g. `pandas.Int64Dtype()`, and [`pandas.ArrowDtype`](https://pandas.pydata.org/docs/reference/api/pandas.ArrowDtype.html)

```python
>>> import cudf
>>> import pandas as pd
>>> import pyarrow as pa
>>> s = cudf.Series([1, 2, 3], dtype="float32")
>>> s
0    1.0
1    2.0
2    3.0
dtype: float32

>>> s = cudf.Series([1, 2, 3], dtype=pd.Float64Dtype())
>>> s
0    1.0
1    2.0
2    3.0
dtype: Float32

>>> s = cudf.Series([1, 2, 3], dtype=pd.ArrowDtype(pa.float64()))
>>> s
0    1.0
1    2.0
2    3.0
dtype: double[pyarrow]
```

These data type objects are treated as metadata describing the type of data
and have no behavior differences when operations are performed on the GPU data.

## A note on `object`

In pandas, `"object"` can represent string or arbitrary Python object data.

```python
>>> import pandas as pd
>>> pd.Series([True, 1, "a", pd.Series([])], dtype=object)
0                         True
1                            1
2                            a
3    Series([], dtype: object)
dtype: object

>>> pd.Series(["a", "b", "C"], dtype=object)
0    a
1    b
2    C
dtype: object
```

In cuDF, `"object"` type can only be used as a representation for string data
as cuDF does not support storing arbitrary Python objects.

```python
>>> import cudf
>>> s = cudf.Series(["abc", "def", "ghi"], dtype="object")
>>> s.dtype
dtype("object")
```

## Decimal data types

{py:class}`~cudf.core.dtypes.Decimal32Dtype`,
{py:class}`~cudf.core.dtypes.Decimal64Dtype`, and
{py:class}`~cudf.core.dtypes.Decimal128Dtype` are data types for decimal data useful when you
need to store values with greater precision than allowed by floating-point
representation.

Decimal data types in cuDF are based on fixed-point representation. A
decimal data type is composed of a _precision_ and a _scale_.  The
precision represents the total number of digits in each value of this
dtype. For example, the precision associated with the decimal value
`1.023` is `4`. The scale is the total number of digits to the right
of the decimal point. The scale associated with the value `1.023` is
3.

Each decimal data type is associated with a maximum precision:

```python
>>> cudf.Decimal32Dtype.MAX_PRECISION
9.0
>>> cudf.Decimal64Dtype.MAX_PRECISION
18.0
>>> cudf.Decimal128Dtype.MAX_PRECISION
38
```

One way to create a decimal Series is from values of type [decimal.Decimal][python-decimal].

```python
>>> from decimal import Decimal
>>> s = cudf.Series([Decimal("1.01"), Decimal("4.23"), Decimal("0.5")])
>>> s
0    1.01
1    4.23
2    0.50
dtype: decimal128
>>> s.dtype
Decimal128Dtype(precision=3, scale=2)
```

Notice the data type of the result: `1.01`, `4.23`, `0.50` can all be
represented with a precision of at least 3 and a scale of at least 2.

However, the value `1.234` needs a precision of at least 4, and a
scale of at least 3, and cannot be fully represented using this data
type:

```python
>>> s[1] = Decimal("1.234")  # raises an error
```

## Nested data types (`List` and `Struct`)

{py:class}`~cudf.core.dtypes.ListDtype` and
{py:class}`~cudf.core.dtypes.StructDtype` are data types in cuDF for
working with list-like and dictionary-like data. These are referred to as
"nested" data types because they are specified by a "child" type of the elements
which also might be a list or struct type.

You can create lists and struct Series from existing pandas Series of
lists and dictionaries respectively:

```python
>>> psr = pd.Series([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])
>>> psr
0 {'a': 1, 'b': 2}
1 {'a': 3, 'b': 4}
dtype: object
>>> gsr = cudf.from_pandas(psr)
>>> gsr
0 {'a': 1, 'b': 2}
1 {'a': 3, 'b': 4}
dtype: struct
>>> gsr.dtype
StructDtype({'a': dtype('int64'), 'b': dtype('int64')})
```

Or by reading them from disk, using a [file format that supports nested data](/cudf/io/index.md).

```python
>>> pdf = pd.DataFrame({"a": [[1, 2], [3, 4, 5], [6, 7, 8]]})
>>> pdf.to_parquet("lists.pq")
>>> gdf = cudf.read_parquet("lists.pq")
>>> gdf
           a
0     [1, 2]
1  [3, 4, 5]
2  [6, 7, 8]
>>> gdf["a"].dtype
ListDtype(int64)
```

[python-decimal]: https://docs.python.org/3/library/decimal.html#decimal.Decimal
