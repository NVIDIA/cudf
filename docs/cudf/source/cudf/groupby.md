---
myst:
  substitutions:
    describe: '`describe`'
---

(basics-groupby)=

# GroupBy

cuDF supports a small (but important) subset of Pandas' [groupby
API](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html).

## Summary of supported operations

1. Grouping by one or more columns
2. Basic aggregations such as "sum", "mean", etc.
3. Quantile aggregation
4. A "collect" or `list` aggregation for collecting values in a group
   into lists
5. Automatic exclusion of columns with unsupported dtypes ("nuisance"
   columns) when aggregating
6. Iterating over the groups of a GroupBy object
7. `GroupBy.groups` API that returns a mapping of group keys to row
   labels
8. `GroupBy.apply` API for performing arbitrary operations on each
   group. Note that this has very limited functionality compared to the
   equivalent Pandas function. See the section on
   [apply](#groupby-apply) for more details.
9. `GroupBy.pipe` similar to
   [Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#piping-function-calls).

## Grouping

A GroupBy object is created by grouping the values of a `Series` or
`DataFrame` by one or more columns:

```python
>>> import cudf
>>> df = cudf.DataFrame({'a': [1, 1, 1, 2, 2], 'b': [1, 1, 2, 2, 3], 'c': [1, 2, 3, 4, 5]})
>>> df
   a  b  c
0  1  1  1
1  1  1  2
2  1  2  3
3  2  2  4
4  2  3  5
>>> gb1 = df.groupby('a')  # grouping by a single column
>>> gb2 = df.groupby(['a', 'b'])  # grouping by multiple columns
>>> gb3 = df.groupby(cudf.Series(['a', 'a', 'b', 'b', 'b']))  # grouping by an external column
```

````{warning}
Unlike Pandas, cuDF uses `sort=False` by default to achieve better
performance, which does not guarantee any particular group order in
the result.

For example:

```python
>>> df = cudf.DataFrame({'a' : [2, 2, 1], 'b' : [42, 21, 11]})
>>> df.groupby('a').sum()
   b
a
2  63
1  11
>>> df.to_pandas().groupby('a').sum()
   b
a
1  11
2  63
```

Setting `sort=True` will produce Pandas-like output, but with some performance penalty:

```python
>>> df.groupby('a', sort=True).sum()
   b
a
1  11
2  63
```
````

### Grouping by index levels

You can also group by one or more levels of a MultiIndex:

```python
>>> df = cudf.DataFrame(
...     {'a': [1, 1, 1, 2, 2], 'b': [1, 1, 2, 2, 3], 'c': [1, 2, 3, 4, 5]}
... ).set_index(['a', 'b'])
...
>>> df.groupby(level='a')
```

### The `Grouper` object

A `Grouper` can be used to disambiguate between columns and levels
when they have the same name:

```python
>>> df
   b  c
b
1  1  1
1  1  2
1  2  3
2  2  4
2  3  5
>>> df.groupby('b', level='b')  # ValueError: Cannot specify both by and level
>>> df.groupby([cudf.Grouper(key='b'), cudf.Grouper(level='b')])  # OK
```

## Aggregation

Aggregations on groups are supported via the `agg` method:

```python
>>> df
   a  b  c
0  1  1  1
1  1  1  2
2  1  2  3
3  2  2  4
4  2  3  5
>>> df.groupby('a').agg('sum')
   b  c
a
1  4  6
2  5  9
>>> df.groupby('a').agg({'b': ['sum', 'min'], 'c': 'mean'})
    b        c
  sum min mean
a
1   4   1  2.0
2   5   2  4.5
>>> df.groupby("a").corr(method="pearson")
          b          c
a
1 b  1.000000  0.866025
  c  0.866025  1.000000
2 b  1.000000  1.000000
  c  1.000000  1.000000
```

The following table summarizes the available aggregations and the types
that support them:

```{eval-rst}
.. table::
    :class: special-table

    +------------------------------------+-----------+------------+----------+---------------+--------+----------+------------+-----------+
    | Aggregations / dtypes              | Numeric   | Datetime   | String   | Categorical   | List   | Struct   | Interval   | Decimal   |
    +====================================+===========+============+==========+===============+========+==========+============+===========+
    | count                              | ✅        | ✅         | ✅       | ✅            |        |          |            | ✅        |
    +------------------------------------+-----------+------------+----------+---------------+--------+----------+------------+-----------+
    | size                               | ✅        | ✅         | ✅       | ✅            |        |          |            | ✅        |
    +------------------------------------+-----------+------------+----------+---------------+--------+----------+------------+-----------+
    | sum                                | ✅        | ✅         |          |               |        |          |            | ✅        |
    +------------------------------------+-----------+------------+----------+---------------+--------+----------+------------+-----------+
    | idxmin                             | ✅        | ✅         |          |               |        |          |            | ✅        |
    +------------------------------------+-----------+------------+----------+---------------+--------+----------+------------+-----------+
    | idxmax                             | ✅        | ✅         |          |               |        |          |            | ✅        |
    +------------------------------------+-----------+------------+----------+---------------+--------+----------+------------+-----------+
    | min                                | ✅        | ✅         | ✅       |               |        |          |            | ✅        |
    +------------------------------------+-----------+------------+----------+---------------+--------+----------+------------+-----------+
    | max                                | ✅        | ✅         | ✅       |               |        |          |            | ✅        |
    +------------------------------------+-----------+------------+----------+---------------+--------+----------+------------+-----------+
    | mean                               | ✅        | ✅         |          |               |        |          |            |           |
    +------------------------------------+-----------+------------+----------+---------------+--------+----------+------------+-----------+
    | var                                | ✅        | ✅         |          |               |        |          |            |           |
    +------------------------------------+-----------+------------+----------+---------------+--------+----------+------------+-----------+
    | std                                | ✅        | ✅         |          |               |        |          |            |           |
    +------------------------------------+-----------+------------+----------+---------------+--------+----------+------------+-----------+
    | quantile                           | ✅        | ✅         |          |               |        |          |            |           |
    +------------------------------------+-----------+------------+----------+---------------+--------+----------+------------+-----------+
    | median                             | ✅        | ✅         |          |               |        |          |            |           |
    +------------------------------------+-----------+------------+----------+---------------+--------+----------+------------+-----------+
    | nunique                            | ✅        | ✅         | ✅       | ✅            |        |          |            | ✅        |
    +------------------------------------+-----------+------------+----------+---------------+--------+----------+------------+-----------+
    | nth                                | ✅        | ✅         | ✅       |               |        |          |            | ✅        |
    +------------------------------------+-----------+------------+----------+---------------+--------+----------+------------+-----------+
    | collect                            | ✅        | ✅         | ✅       |               | ✅     |          |            | ✅        |
    +------------------------------------+-----------+------------+----------+---------------+--------+----------+------------+-----------+
    | unique                             | ✅        | ✅         | ✅       | ✅            |        |          |            |           |
    +------------------------------------+-----------+------------+----------+---------------+--------+----------+------------+-----------+
    | corr                               | ✅        |            |          |               |        |          |            | ✅        |
    +------------------------------------+-----------+------------+----------+---------------+--------+----------+------------+-----------+
    | cov                                | ✅        |            |          |               |        |          |            | ✅        |
    +------------------------------------+-----------+------------+----------+---------------+--------+----------+------------+-----------+
```

## GroupBy apply

To apply function on each group, use the `GroupBy.apply()` method:

```python
>>> df
   a  b  c
0  1  1  1
1  1  1  2
2  1  2  3
3  2  2  4
4  2  3  5
>>> df.groupby('a').apply(lambda x: x.max() - x.min())
   a  b  c
a
0  0  1  2
1  0  1  1
```

### Limitations

- `apply` works by applying the provided function to each group
  sequentially, and concatenating the results together. **This can be
  very slow**, especially for a large number of small groups. For a
  small number of large groups, it can give acceptable performance.
- The results may not always match Pandas exactly. For example, cuDF
  may return a `DataFrame` containing a single column where Pandas
  returns a `Series`. Some post-processing may be required to match
  Pandas behavior.
- cuDF does not support some of the exceptional cases that Pandas
  supports with `apply`, such as calling [describe] inside the
  callable.

## Transform

The `.transform()` method aggregates per group, and broadcasts the
result to the group size, resulting in a Series/DataFrame that is of
the same size as the input Series/DataFrame.

```python
>>> import cudf
>>> df = cudf.DataFrame({'a': [2, 1, 1, 2, 2], 'b': [1, 2, 3, 4, 5]})
>>> df.groupby('a').transform('max')
   b
0  5
1  3
2  3
3  5
4  5
```

## Rolling window calculations

Use the `GroupBy.rolling()` method to perform rolling window
calculations on each group:

```python
>>> df
   a  b  c
0  1  1  1
1  1  1  2
2  1  2  3
3  2  2  4
4  2  3  5
```

Rolling window sum on each group with a window size of 2:

```python
>>> df.groupby('a').rolling(2).sum()
        a     b     c
a
1 0  <NA>  <NA>  <NA>
  1     2     2     3
  2     2     3     5
2 3  <NA>  <NA>  <NA>
  4     4     5     9
```

[describe]: https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#flexible-apply
