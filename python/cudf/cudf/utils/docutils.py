# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for parameterized docstring
"""

import functools
import re
import string

_regex_whitespaces = re.compile(r"^\s+$")


def _only_spaces(s):
    return bool(_regex_whitespaces.match(s))


def docfmt(**kwargs):
    """Format docstring.

    Similar to saving the result of ``__doc__.format(**kwargs)`` as the
    function's docstring.
    """
    kwargs = {k: v.lstrip() for k, v in kwargs.items()}

    def outer(fn):
        buf = []
        if fn.__doc__ is None:
            return fn
        formatsiter = string.Formatter().parse(fn.__doc__)
        for literal, field, fmtspec, conv in formatsiter:
            assert conv is None
            assert not fmtspec
            buf.append(literal)
            if field is not None:
                # get indentation
                lines = literal.rsplit("\n", 1)
                if _only_spaces(lines[-1]):
                    indent = " " * len(lines[-1])
                    valuelines = kwargs[field].splitlines(True)
                    # first line
                    buf.append(valuelines[0])
                    # subsequent lines are indented
                    buf.extend([indent + ln for ln in valuelines[1:]])
                else:
                    buf.append(kwargs[field])
        fn.__doc__ = "".join(buf)
        return fn

    return outer


def docfmt_partial(**kwargs):
    return functools.partial(docfmt, **kwargs)


def copy_docstring(other):
    """
    Decorator that sets ``__doc__`` to ``other.__doc___``.
    """

    def wrapper(func):
        func.__doc__ = other.__doc__
        return func

    return wrapper


def doc_apply(doc):
    """Set `__doc__` attribute of `func` to `doc`."""

    def wrapper(func):
        func.__doc__ = doc
        return func

    return wrapper


doc_describe = docfmt_partial(
    docstring="""
        Generate descriptive statistics.

        Descriptive statistics include those that summarize the
        central tendency, dispersion and shape of a dataset's
        distribution, excluding ``NaN`` values.

        Analyzes both numeric and object series, as well as
        ``DataFrame`` column sets of mixed data types. The
        output will vary depending on what is provided.
        Refer to the notes below for more detail.

        Parameters
        ----------
        percentiles : list-like of numbers, optional
            The percentiles to include in the output.
            All should fall between 0 and 1. The default is
            ``[.25, .5, .75]``, which returns the 25th, 50th,
            and 75th percentiles.

        include : 'all', list-like of dtypes or None(default), optional
            A list of data types to include in the result.
            Ignored for ``Series``. Here are the options:

            - 'all' : All columns of the input will be included in the output.
            - A list-like of dtypes : Limits the results to the
              provided data types.
              To limit the result to numeric types submit
              ``numpy.number``. To limit it instead to object columns submit
              the ``numpy.object`` data type. Strings
              can also be used in the style of
              ``select_dtypes`` (e.g. ``df.describe(include=['O'])``). To
              select pandas categorical columns, use ``'category'``
            - None (default) : The result will include all numeric columns.

        exclude : list-like of dtypes or None (default), optional,
            A list of data types to omit from the result. Ignored
            for ``Series``. Here are the options:

            - A list-like of dtypes : Excludes the provided data types
              from the result. To exclude numeric types submit
              ``numpy.number``. To exclude object columns submit the data
              type ``numpy.object``. Strings can also be used in the style of
              ``select_dtypes`` (e.g. ``df.describe(include=['O'])``). To
              exclude pandas categorical columns, use ``'category'``
            - None (default) : The result will exclude nothing.

        Returns
        -------
        output_frame : Series or DataFrame
            Summary statistics of the Series or Dataframe provided.

        Notes
        -----
        For numeric data, the result's index will include ``count``,
        ``mean``, ``std``, ``min``, ``max`` as well as lower, ``50`` and
        upper percentiles. By default the lower percentile is ``25`` and the
        upper percentile is ``75``. The ``50`` percentile is the
        same as the median.

        For strings dtype or datetime dtype, the result's index
        will include ``count``, ``unique``, ``top``, and ``freq``. The ``top``
        is the most common value. The ``freq`` is the most common value's
        frequency. Timestamps also include the ``first`` and ``last`` items.

        If multiple object values have the highest count, then the
        ``count`` and ``top`` results will be arbitrarily chosen from
        among those with the highest count.

        For mixed data types provided via a ``DataFrame``, the default is to
        return only an analysis of numeric columns. If the dataframe consists
        only of object and categorical data without any numeric columns, the
        default is to return an analysis of both the object and categorical
        columns. If ``include='all'`` is provided as an option, the result
        will include a union of attributes of each type.

        The ``include`` and ``exclude`` parameters can be used to limit
        which columns in a ``DataFrame`` are analyzed for the output.
        The parameters are ignored when analyzing a ``Series``.

        Examples
        --------
        Describing a ``Series`` containing numeric values.

        >>> import cudf
        >>> s = cudf.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> s
        0     1
        1     2
        2     3
        3     4
        4     5
        5     6
        6     7
        7     8
        8     9
        9    10
        dtype: int64
        >>> s.describe()
        count    10.00000
        mean      5.50000
        std       3.02765
        min       1.00000
        25%       3.25000
        50%       5.50000
        75%       7.75000
        max      10.00000
        dtype: float64

        Describing a categorical ``Series``.

        >>> s = cudf.Series(['a', 'b', 'a', 'b', 'c', 'a'], dtype='category')
        >>> s
        0    a
        1    b
        2    a
        3    b
        4    c
        5    a
        dtype: category
        Categories (3, str): ['a', 'b', 'c']
        >>> s.describe()
        count     6
        unique    3
        top       a
        freq      3
        dtype: str

        Describing a timestamp ``Series``.

        >>> s = cudf.Series([
        ...   "2000-01-01",
        ...   "2010-01-01",
        ...   "2010-01-01"
        ... ], dtype="datetime64[s]")
        >>> s
        0   2000-01-01
        1   2010-01-01
        2   2010-01-01
        dtype: datetime64[s]
        >>> s.describe()
        count                     3
        mean    2006-09-01 08:00:00
        min     2000-01-01 00:00:00
        25%     2004-12-31 12:00:00
        50%     2010-01-01 00:00:00
        75%     2010-01-01 00:00:00
        max     2010-01-01 00:00:00
        dtype: str

        Describing a ``DataFrame``. By default only numeric fields are
        returned.

        >>> df = cudf.DataFrame({"categorical": cudf.Series(['d', 'e', 'f'],
        ...                         dtype='category'),
        ...                      "numeric": [1, 2, 3],
        ...                      "string": ['a', 'b', 'c']
        ... })
        >>> df
          categorical  numeric string
        0           d        1      a
        1           e        2      b
        2           f        3      c
        >>> df.describe()
               numeric
        count      3.0
        mean       2.0
        std        1.0
        min        1.0
        25%        1.5
        50%        2.0
        75%        2.5
        max        3.0

        Describing all columns of a ``DataFrame`` regardless of data type.

        >>> df.describe(include='all')
               categorical numeric string
        count            3     3.0      3
        unique           3     NaN      3
        top              d     NaN      a
        freq             1     NaN      1
        mean           NaN     2.0    NaN
        std            NaN     1.0    NaN
        min            NaN     1.0    NaN
        25%            NaN     1.5    NaN
        50%            NaN     2.0    NaN
        75%            NaN     2.5    NaN
        max            NaN     3.0    NaN

        Describing a column from a ``DataFrame`` by accessing it as an
        attribute.

        >>> df.numeric.describe()
        count    3.0
        mean     2.0
        std      1.0
        min      1.0
        25%      1.5
        50%      2.0
        75%      2.5
        max      3.0
        Name: numeric, dtype: float64

        Including only numeric columns in a ``DataFrame`` description.

        >>> df.describe(include=[np.number])
               numeric
        count      3.0
        mean       2.0
        std        1.0
        min        1.0
        25%        1.5
        50%        2.0
        75%        2.5
        max        3.0

        Including only string columns in a ``DataFrame`` description.

        >>> df.describe(include=[str])
               string
        count       3
        unique      3
        top         a
        freq        1

        Including only categorical columns from a ``DataFrame`` description.

        >>> df.describe(include=['category'])
               categorical
        count            3
        unique           3
        top              d
        freq             1

        Excluding numeric columns from a ``DataFrame`` description.

        >>> df.describe(exclude=[np.number])
               categorical string
        count            3      3
        unique           3      3
        top              d      a
        freq             1      1

        Excluding object columns from a ``DataFrame`` description.

        >>> df.describe(exclude=[str])
               categorical numeric
        count            3     3.0
        unique           3     NaN
        top              d     NaN
        freq             1     NaN
        mean           NaN     2.0
        std            NaN     1.0
        min            NaN     1.0
        25%            NaN     1.5
        50%            NaN     2.0
        75%            NaN     2.5
        max            NaN     3.0
"""
)
