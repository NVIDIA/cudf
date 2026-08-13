# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np
import pandas as pd

import pylibcudf as plc

import cudf
from cudf.api.extensions import no_default
from cudf.api.types import (
    is_integer,
    is_scalar,
    is_string_dtype,
)
from cudf.core.accessors.base_accessor import BaseAccessor
from cudf.core.accessors.lists import ListMethods
from cudf.core.column.column import ColumnBase, as_column, column_empty
from cudf.core.dtypes import ListDtype
from cudf.utils.dtypes import (
    DEFAULT_STRING_DTYPE,
    can_convert_to_column,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from cudf._typing import ColumnLike
    from cudf.core.column.string import StringColumn
    from cudf.core.index import Index
    from cudf.core.series import Series


def _is_supported_regex_flags(flags: int) -> bool:
    all_flags = re.MULTILINE | re.DOTALL | re.IGNORECASE
    return flags == 0 or (
        (flags & all_flags) != 0 and (flags & ~all_flags) == 0
    )


def _massage_string_arg(
    value, name, allow_col: bool = False
) -> StringColumn | plc.Scalar:
    if isinstance(value, str):
        return plc.Scalar.from_py(value, dtype=plc.DataType(plc.TypeId.STRING))

    allowed_types = ["Scalar"]

    if allow_col:
        if isinstance(value, list):
            return as_column(value, dtype=DEFAULT_STRING_DTYPE)  # type: ignore[return-value]

        from cudf.core.column.string import StringColumn

        if isinstance(value, StringColumn):
            return value

        allowed_types.append("Column")

    if len(allowed_types) == 1:
        expected = allowed_types[0]
    else:
        expected = ", ".join(allowed_types[:-1]) + ", or " + allowed_types[-1]

    raise ValueError(f"Expected {expected} for {name} but got {type(value)}")


class StringMethods(BaseAccessor):
    """
    Vectorized string functions for Series and Index.

    This mimics pandas ``df.str`` interface. nulls stay null
    unless handled otherwise by a particular method.
    Patterned after Python's string methods, with some
    inspiration from R's stringr package.
    """

    _column: StringColumn

    def __init__(self, parent: Series | Index):
        value_type = (
            parent.dtype.leaf_type
            if isinstance(parent.dtype, ListDtype)
            else parent.dtype
        )
        # Convert categorical with string categories to string dtype
        if isinstance(value_type, cudf.CategoricalDtype) and is_string_dtype(
            value_type.categories.dtype
        ):
            parent = parent.astype(value_type.categories.dtype)
            value_type = parent.dtype
            is_valid_string = True
        else:
            # Validate dtype is suitable for string operations
            is_valid_string = is_string_dtype(value_type)
        if not is_valid_string:
            # Mirror pandas' inferred-type naming in the error message
            # (see pandas lib.infer_dtype) so error messages match.
            kind_to_inferred = {
                "i": "integer",
                "u": "integer",
                "f": "floating",
                "b": "boolean",
                "c": "complex",
                "M": "datetime64",
                "m": "timedelta64",
            }
            inferred_dtype = kind_to_inferred.get(
                getattr(value_type, "kind", ""), str(value_type)
            )
            raise AttributeError(
                "Can only use .str accessor with string values, "
                f"not {inferred_dtype}"
            )
        super().__init__(parent=parent)

    def htoi(self) -> Series | Index:
        """
        Returns integer value represented by each hex string.
        String is interpreted to have hex (base-16) characters.

        Returns
        -------
        Series/Index of str dtype

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["1234", "ABCDEF", "1A2", "cafe"])
        >>> s.str.htoi()
        0        4660
        1    11259375
        2         418
        3       51966
        dtype: int64
        """
        return self._return_or_inplace(self._column.hex_to_integers())

    hex_to_int = htoi

    def ip2int(self) -> Series | Index:
        """
        This converts ip strings to integers

        Returns
        -------
        Series/Index of str dtype

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["12.168.1.1", "10.0.0.1"])
        >>> s.str.ip2int()
        0    212336897
        1    167772161
        dtype: uint32

        Returns 0's if any string is not an IP.

        >>> s = cudf.Series(["12.168.1.1", "10.0.0.1", "abc"])
        >>> s.str.ip2int()
        0    212336897
        1    167772161
        2            0
        dtype: uint32
        """
        return self._return_or_inplace(self._column.ipv4_to_integers())

    ip_to_int = ip2int

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.slice(start=key.start, stop=key.stop, step=key.step)
        else:
            return self.get(key)

    def len(self) -> Series | Index:
        r"""
        Computes the length of each element in the Series/Index.

        Returns
        -------
        Series or Index of int
            A Series or Index of integer values
            indicating the length of each element in the Series or Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["dog", "", "\n", None])
        >>> s.str.len()
        0       3
        1       0
        2       1
        3    <NA>
        dtype: int32
        """
        return self._return_or_inplace(self._column.count_characters())

    def byte_count(self) -> Series | Index:
        """
        Computes the number of bytes of each string in the Series/Index.

        Returns
        -------
        Series or Index of int
            A Series or Index of integer values
            indicating the number of bytes of each strings in the
            Series or Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["abc","d","ef"])
        >>> s.str.byte_count()
        0    3
        1    1
        2    2
        dtype: int32
        >>> s = cudf.Series(["Hello", "Bye", "Thanks 😊"])
        >>> s.str.byte_count()
        0     5
        1     3
        2    11
        dtype: int32
        """
        return self._return_or_inplace(self._column.count_bytes())

    @overload
    def cat(
        self, sep: str | None = None, na_rep: str | None = None
    ) -> str: ...

    @overload
    def cat(
        self, others, sep: str | None = None, na_rep: str | None = None
    ) -> Series | Index | StringColumn: ...

    def cat(self, others=None, sep=None, na_rep=None):
        """
        Concatenate strings in the Series/Index with given separator.

        If ``others`` is specified, this function concatenates the Series/Index
        and elements of others element-wise. If others is not passed, then all
        values in the Series/Index are concatenated into a single string with
        a given sep.

        Parameters
        ----------
        others : Series or List of str
            Strings to be appended.
            The number of strings must match ``size()`` of this instance.
            This must be either a Series of string dtype or a Python
            list of strings.

        sep : str
            If specified, this separator will be appended to each string
            before appending the others.

        na_rep : str
            This character will take the place of any null strings
            (not empty strings) in either list.

            -  If ``na_rep`` is ``None``, and ``others`` is ``None``,
               missing values in the Series/Index are
               omitted from the result.

            -  If ``na_rep`` is ``None``, and ``others`` is
               not ``None``, a row containing a missing value
               in any of the columns (before concatenation)
               will have a missing value in the result.

        Returns
        -------
        concat : str or Series/Index of str dtype
            If ``others`` is ``None``, ``str`` is returned,
            otherwise a ``Series/Index`` (same type as caller)
            of str dtype is returned.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['a', 'b', None, 'd'])
        >>> s.str.cat(sep=' ')
        'a b d'

        By default, NA values in the Series are ignored. Using na_rep, they
        can be given a representation:

        >>> s.str.cat(sep=' ', na_rep='?')
        'a b ? d'

        If others is specified, corresponding values are concatenated with
        the separator. Result will be a Series of strings.

        >>> s.str.cat(['A', 'B', 'C', 'D'], sep=',')
        0     a,A
        1     b,B
        2    NaN
        3     d,D
        dtype: str

        Missing values will remain missing in the result, but can again be
        represented using na_rep

        >>> s.str.cat(['A', 'B', 'C', 'D'], sep=',', na_rep='-')
        0    a,A
        1    b,B
        2    -,C
        3    d,D
        dtype: str

        If sep is not specified, the values are concatenated without
        separation.

        >>> s.str.cat(['A', 'B', 'C', 'D'], na_rep='-')
        0    aA
        1    bB
        2    -C
        3    dD
        dtype: str
        """
        if sep is None:
            sep = ""

        if others is None:
            data = self._column.join_strings(sep, na_rep)
            if len(data) == 0:
                return ""
        else:
            parent_index = (
                self._parent.index
                if isinstance(self._parent, cudf.Series)
                else self._parent
            )
            if (
                can_convert_to_column(others)
                and len(others) > 0
                and (
                    can_convert_to_column(
                        others.iloc[0]
                        if isinstance(others, cudf.Series)
                        else others[0]
                    )
                )
            ):
                other_cols = (
                    (
                        as_column(
                            frame.reindex(parent_index),
                            dtype=DEFAULT_STRING_DTYPE,
                        )
                        if (
                            parent_index is not None
                            and isinstance(frame, cudf.Series)
                            and not frame.index.equals(parent_index)
                        )
                        else as_column(frame, dtype=DEFAULT_STRING_DTYPE)
                    )
                    for frame in others
                )
            elif others is not None and not isinstance(others, StringMethods):
                if (
                    parent_index is not None
                    and isinstance(others, cudf.Series)
                    and not others.index.equals(parent_index)
                ):
                    others = others.reindex(parent_index)

                other_cols = [as_column(others, dtype=DEFAULT_STRING_DTYPE)]
            else:
                raise TypeError(
                    "others must be Series, Index, DataFrame, np.ndarray "
                    "or list-like (either containing only strings or "
                    "containing only objects of type Series/Index/"
                    "np.ndarray[1-dim])"
                )
            data = self._column.concatenate(other_cols, sep, na_rep)

        if len(data) == 1 and data.null_count == 1:
            data = as_column("", length=len(data))
        # We only want to keep the index if we are adding something to each
        # row, not if we are joining all the rows into a single string.
        out = self._return_or_inplace(data, retain_index=others is not None)
        if len(out) == 1 and others is None:
            if isinstance(out, cudf.Series):
                out = out.iloc[0]
            else:
                out = out[0]
        return out

    def join(
        self, sep=None, string_na_rep=None, sep_na_rep=None
    ) -> Series | Index:
        """
        Join lists contained as elements in the Series/Index with passed
        delimiter.

        If the elements of a Series are lists themselves, join the content of
        these lists using the delimiter passed to the function.
        This function is an equivalent to :meth:`str.join`.
        In the special case that the lists in the Series contain only ``None``,
        a `<NA>`/`None` value will always be returned.

        Parameters
        ----------
        sep : str or array-like
            If str, the delimiter is used between list entries.
            If array-like, the string at a position is used as a
            delimiter for corresponding row of the list entries.
        string_na_rep : str, default None
            This character will take the place of null strings
            (not empty strings) in the Series but will be considered
            only if the Series contains list elements and those lists have
            at least one non-null string. If ``string_na_rep`` is ``None``,
            it defaults to empty space "".
        sep_na_rep : str, default None
            This character will take the place of any null strings
            (not empty strings) in `sep`. This parameter can be used
            only if `sep` is array-like. If ``sep_na_rep`` is ``None``,
            it defaults to empty space "".

        Returns
        -------
        Series/Index: object
            The list entries concatenated by intervening occurrences of
            the delimiter.

        Raises
        ------
        ValueError
            - If ``sep_na_rep`` is supplied when ``sep`` is str.
            - If ``sep`` is array-like and not of equal length with Series/Index.
        TypeError
            - If ``string_na_rep`` or ``sep_na_rep`` are not scalar values.
            - If ``sep`` is not of following types: str or array-like.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([['a', 'b', 'c'], ['d', 'e'], ['f'], ['g', ' ', 'h']])
        >>> ser
        0    [a, b, c]
        1       [d, e]
        2          [f]
        3    [g,  , h]
        dtype: list
        >>> ser.str.join(sep='-')
        0    a-b-c
        1      d-e
        2        f
        3    g- -h
        dtype: str

        ``sep`` can an array-like input:

        >>> ser.str.join(sep=['-', '+', '.', '='])
        0    a-b-c
        1      d+e
        2        f
        3    g= =h
        dtype: str

        If the actual series doesn't have lists, each character is joined
        by `sep`:

        >>> ser = cudf.Series(['abc', 'def', 'ghi'])
        >>> ser
        0    abc
        1    def
        2    ghi
        dtype: str
        >>> ser.str.join(sep='_')
        0    a_b_c
        1    d_e_f
        2    g_h_i
        dtype: str

        We can replace `<NA>`/`None` values present in lists using
        ``string_na_rep`` if the lists contain at least one valid string
        (lists containing all `None` will result in a `<NA>`/`None` value):

        >>> ser = cudf.Series([['a', 'b', None], [None, None, None], None, ['c', 'd']])
        >>> ser
        0          [a, b, None]
        1    [None, None, None]
        2                  None
        3                [c, d]
        dtype: list
        >>> ser.str.join(sep='_', string_na_rep='k')
        0    a_b_k
        1     NaN
        2     NaN
        3      c_d
        dtype: str

        We can replace `<NA>`/`None` values present in lists of ``sep``
        using ``sep_na_rep``:

        >>> ser.str.join(sep=[None, '^', '.', '-'], sep_na_rep='+')
        0    a+b+
        1    NaN
        2    NaN
        3     c-d
        dtype: str
        """
        if sep is None:
            sep = ""

        if string_na_rep is None:
            string_na_rep = ""

        if is_scalar(sep) and sep_na_rep:
            raise ValueError(
                "sep_na_rep cannot be defined when `sep` is scalar."
            )

        if sep_na_rep is None:
            sep_na_rep = ""

        if not is_scalar(string_na_rep):
            raise TypeError(
                f"string_na_rep should be a string scalar, got {string_na_rep}"
                f" of type : {type(string_na_rep)}"
            )

        if isinstance(self._column.dtype, ListDtype):
            list_column = self._column
            result_dtype = cast("ListDtype", list_column.dtype).element_type
        else:
            # If self._column is not a ListColumn, we will have to
            # split each row by character and create a ListColumn out of it.
            list_column = self._column.fillna("").character_tokenize()
            # TODO: pd.Handle ArrowDtype
            result_dtype = cast("ListDtype", list_column.dtype).element_type
            if len(list_column) == 0:
                list_column = column_empty(  # type: ignore[assignment]
                    len(self._column), dtype=list_column.dtype
                )

        if is_scalar(sep):
            data = list_column.join_list_elements(  # type: ignore[attr-defined]
                sep, string_na_rep, "", result_dtype
            )
        elif can_convert_to_column(sep):
            sep_column = as_column(sep)
            if len(sep_column) != len(list_column):
                raise ValueError(
                    f"sep should be of similar size to the series, "
                    f"got: {len(sep_column)}, expected: {len(list_column)}"
                )
            if not is_scalar(sep_na_rep):
                raise TypeError(
                    f"sep_na_rep should be a string scalar, got {sep_na_rep} "
                    f"of type: {type(sep_na_rep)}"
                )
            data = list_column.join_list_elements(  # type: ignore[attr-defined]
                sep_column,
                sep_na_rep,
                string_na_rep,
                result_dtype,
            )
        else:
            raise TypeError(
                f"sep should be an str, array-like or Series object, "
                f"found {type(sep)}"
            )

        return self._return_or_inplace(data)

    def extract(
        self, pat: str, flags: int = 0, expand: bool = True
    ) -> Series | Index:
        r"""
        Extract capture groups in the regex `pat` as columns in a DataFrame.

        For each subject string in the Series, extract groups from the first
        match of regular expression `pat`.

        Parameters
        ----------
        pat : str
            Regular expression pattern with capturing groups.
        flags : int, default 0 (no flags)
            Flags to pass through to the regex engine (e.g. re.MULTILINE)
        expand : bool, default True
            If True, return DataFrame with one column per capture group.
            If False, return a Series/Index if there is one capture group or
            DataFrame if there are multiple capture groups.

        Returns
        -------
        DataFrame or Series/Index
            A DataFrame with one row for each subject string, and one column
            for each group. If `expand=False` and `pat` has only one capture
            group, then return a Series/Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['a1', 'b2', 'c3'])
        >>> s.str.extract(r'([ab])(\d)')
              0     1
        0     a     1
        1     b     2
        2   NaN   NaN

        A pattern with one group will return a DataFrame with one
        column if expand=True.

        >>> s.str.extract(r'[ab](\d)', expand=True)
              0
        0     1
        1     2
        2   NaN

        A pattern with one group will return a Series if expand=False.

        >>> s.str.extract(r'[ab](\d)', expand=False)
        0       1
        1       2
        2    NaN
        dtype: str

        .. pandas-compat::
            :meth:`pandas.Series.str.extract`

            The `flags` parameter currently only supports re.DOTALL,
            re.MULTILINE and re.IGNORECASE.
        """
        if not isinstance(expand, bool):
            raise ValueError("expand parameter must be True or False")
        if not _is_supported_regex_flags(flags):
            raise NotImplementedError(
                "unsupported value for `flags` parameter"
            )

        compiled = re.compile(pat)
        group_names = list(compiled.groupindex.keys())
        if len(group_names) > 0:
            pat = re.sub(r"\(\?P<([A-Za-z_][A-Za-z0-9_]*)>", "(", pat)
        data = self._column.extract(pat, flags)
        result_name = None
        if len(data) == 1 and expand is False:
            _, data = data.popitem()  # type: ignore[assignment]
            if len(group_names) > 0:
                result_name = group_names[0]
        elif expand is False and len(data) > 1:
            expand = True
        if len(group_names) == len(data):
            named_data = {}
            for key, value in data.items():
                named_data[group_names[key]] = value
            data = named_data  # type: ignore[assignment]
        return self._return_or_inplace(
            data, expand=expand, replace_name=result_name
        )

    def _remove_named_capture_groups(self, pat: str) -> str:
        r"""
        Removes any named capture groups from the given regex pattern.
        Named capture groups are expected in the format "(?P<name>)" only.
        These are unnecessary (and unexpected) by the pylibcudf for non-extract regex calls.
        """
        to_remove = "|".join(re.compile(pat).groupindex.keys())
        if to_remove:
            pat = re.sub(rf"\(\?P<(?:{to_remove})>", "(", pat)
        return pat

    def contains(
        self,
        pat: str | Sequence,
        case: bool = True,
        flags: int = 0,
        na=no_default,
        regex: bool = True,
    ) -> Series | Index:
        r"""
        Test if pattern or regex is contained within a string of a Series or
        Index.

        Return boolean Series or Index based on whether a given pattern or
        regex is contained within a string of a Series or Index.

        Parameters
        ----------
        pat : str or list-like
            Character sequence or regular expression.
            If ``pat`` is list-like then regular expressions are not
            accepted.
        flags : int, default 0 (no flags)
            Flags to pass through to the regex engine (e.g. re.MULTILINE)
        na : scalar, optional
            Fill value for missing values. The default depends on dtype of the
            array. For the ``"str"`` dtype, ``False`` is used. For object
            dtype, ``numpy.nan`` is used. For the nullable ``StringDtype``,
            ``pandas.NA`` is used.
        regex : bool, default True
            If True, assumes the pattern is a regular expression.
            If False, treats the pattern as a literal string.

        Returns
        -------
        Series/Index of bool dtype
            A Series/Index of boolean dtype indicating whether the given
            pattern is contained within the string of each element of the
            Series/Index.

        Examples
        --------
        >>> import cudf
        >>> s1 = cudf.Series(['Mouse', 'dog', 'house and parrot', '23', None])
        >>> s1
        0               Mouse
        1                 dog
        2    house and parrot
        3                  23
        4                 NaN
        dtype: str
        >>> s1.str.contains('og', regex=False)
        0    False
        1     True
        2    False
        3    False
        4    False
        dtype: bool

        Returning an Index of booleans using only a literal pattern.

        >>> data = ['Mouse', 'dog', 'house and parrot', '23.0', np.nan]
        >>> idx = cudf.Index(data)
        >>> idx
        Index(['Mouse', 'dog', 'house and parrot', '23.0', nan], dtype='str')
        >>> idx.str.contains('23', regex=False)
        Index([False, False, False, True, False], dtype='bool')

        Returning 'house' or 'dog' when either expression occurs in a string.

        >>> s1.str.contains('house|dog', regex=True)
        0    False
        1     True
        2     True
        3    False
        4    False
        dtype: bool

        Returning any digit using regular expression.

        >>> s1.str.contains('\\d', regex=True)
        0    False
        1    False
        2    False
        3     True
        4    False
        dtype: bool

        Ensure ``pat`` is a not a literal pattern when ``regex`` is set
        to True. Note in the following example one might expect
        only `s2[1]` and `s2[3]` to return True. However,
        '.0' as a regex matches any character followed by a 0.

        >>> s2 = cudf.Series(['40', '40.0', '41', '41.0', '35'])
        >>> s2.str.contains('.0', regex=True)
        0     True
        1     True
        2    False
        3     True
        4    False
        dtype: bool

        The ``pat`` may also be a sequence of strings in which case
        the individual strings are searched in corresponding rows.

        >>> s2 = cudf.Series(['house', 'dog', 'and', '', ''])
        >>> s1.str.contains(s2)
        0    False
        1     True
        2     True
        3     True
        4    False
        dtype: bool

        .. pandas-compat::
            :meth:`pandas.Series.str.contains`

            The parameter `case` is not yet supported and will
            raise a NotImplementedError if anything other than the default
            value is set.
            The `flags` parameter currently only supports re.DOTALL,
            re.MULTILINE and re.IGNORECASE.
        """
        if (
            na is not no_default
            and not pd.isna(na)
            and not isinstance(na, bool)
        ):
            # GH#59561
            raise ValueError(
                f"na must be None, pd.NA, np.nan, True, or False;got {na!r}"
            )
        if regex and isinstance(pat, re.Pattern):
            flags = pat.flags & ~re.U
            pat = pat.pattern
        if not _is_supported_regex_flags(flags):
            raise NotImplementedError(
                "unsupported value for `flags` parameter"
            )
        if regex and not case:
            raise NotImplementedError(
                "`case=False` only supported when `regex=False`"
            )

        if is_scalar(pat):
            if regex:
                pat = self._remove_named_capture_groups(pat)  # type: ignore[arg-type]
                result_col = self._column.contains_re(pat, flags)
            else:
                if case is False:
                    input_column = self._column.to_lower()
                    pat_normed = pat.lower()  # type: ignore[union-attr]
                else:
                    input_column = self._column
                    pat_normed = pat
                result_col = input_column.str_contains(pat_normed)
        else:
            # TODO: we silently ignore the `regex=` flag here
            col_pat = as_column(pat, dtype=DEFAULT_STRING_DTYPE)
            if case is False:
                input_column = self._column.to_lower()
                col_pat = col_pat.to_lower()  # type: ignore[attr-defined]
            else:
                input_column = self._column
            result_col = input_column.str_contains(col_pat)  # type: ignore[arg-type]
        if (
            na is no_default
            and self._column._PANDAS_NA_VALUE in {np.nan, None}
            and self._column.has_nulls()
        ):
            result_col = result_col.fillna(False)
        if na is not no_default:
            result_col = result_col.fillna(na)
        return self._return_or_inplace(result_col)

    def like(self, pat: str, esc: str | None = None) -> Series | Index:
        """
        Test if a like pattern matches a string of a Series or Index.

        Return boolean Series or Index based on whether a given pattern
        matches strings in a Series or Index.

        Parameters
        ----------
        pat : str
            Pattern for matching. Use '%' for any number of any character
            including no characters. Use '_' for any single character.

        esc : str
            Character to use if escape is necessary to match '%' or '_'
            literals.

        Returns
        -------
        Series/Index of bool dtype
            A Series/Index of boolean dtype indicating whether the given
            pattern matches the string of each element of the Series/Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['abc', 'a', 'b' ,'ddbc', '%bb'])
        >>> s.str.like('%b_')
        0   True
        1   False
        2   False
        3   True
        4   True
        dtype: bool

        Parameter `esc` can be used to match a wildcard literal.

        >>> s.str.like('%b_', esc='/' )
        0   True
        1   False
        2   False
        3   True
        4   True
        dtype: bool
        """
        if not isinstance(pat, str):
            raise TypeError(
                f"expected a string object, not {type(pat).__name__}"
            )

        if esc is None:
            esc = ""

        if not isinstance(esc, str):
            raise TypeError(
                f"expected a string object, not {type(esc).__name__}"
            )

        if len(esc) > 1:
            raise ValueError(
                "expected esc to contain less than or equal to 1 characters"
            )

        return self._return_or_inplace(self._column.like(pat, esc))

    def repeat(
        self,
        repeats: int | Sequence,
    ) -> Series | Index:
        """
        Duplicate each string in the Series or Index.
        Equivalent to `str.repeat()
        <https://pandas.pydata.org/docs/reference/api/pandas.Series.str.repeat.html>`_.

        Parameters
        ----------
        repeats : int or sequence of int
            Same value for all (int) or different value per (sequence).

        Returns
        -------
        Series or Index of object
            Series or Index of repeated string objects specified by
            input parameter repeats.

        Examples
        --------
        >>> s = cudf.Series(['a', 'b', 'c'])
        >>> s
        0    a
        1    b
        2    c
        dtype: str

        Single int repeats string in Series

        >>> s.str.repeat(repeats=2)
        0    aa
        1    bb
        2    cc
        dtype: str

        Sequence of int repeats corresponding string in Series

        >>> s.str.repeat(repeats=[1, 2, 3])
        0      a
        1     bb
        2    ccc
        dtype: str
        """
        if isinstance(repeats, int) and repeats < 0:
            raise ValueError("Negative repeats is not allowed.")
        if can_convert_to_column(repeats):
            repeats = as_column(repeats, dtype=np.dtype(np.int64))  # type: ignore[assignment]
        return self._return_or_inplace(self._column.repeat_strings(repeats))  # type: ignore[arg-type]

    def replace(
        self,
        pat: str | Sequence,
        repl: str | Sequence,
        n: int = -1,
        case=None,
        flags: int = 0,
        regex: bool = True,
    ) -> Series | Index:
        """
        Replace occurrences of pattern/regex in the Series/Index with some
        other string. Equivalent to `str.replace()
        <https://docs.python.org/3/library/stdtypes.html#str.replace>`_
        or `re.sub()
        <https://docs.python.org/3/library/re.html#re.sub>`_.

        Parameters
        ----------
        pat : str or list-like
            String(s) to be replaced as a character sequence or regular
            expression.
        repl : str or list-like
            String(s) to be used as replacement.
        n : int, default -1 (all)
            Number of replacements to make from the start.
        regex : bool, default True
            If True, assumes the pattern is a regular expression.
            Raises ValueError if pat is not a string.
            If False, treats the pattern as a literal string.

        Returns
        -------
        Series/Index of str dtype
            A copy of the object with all matching occurrences of pat replaced
            by repl.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['foo', 'fuz', None])
        >>> s
        0     foo
        1     fuz
        2    NaN
        dtype: str

        When pat is a string and regex is True (the default), the given pat
        is compiled as a regex. When repl is a string, it replaces matching
        regex patterns as with ``re.sub()``. NaN value(s) in the Series
        are left as is:

        >>> s.str.replace('f.', 'ba', regex=True)
        0     bao
        1     baz
        2    NaN
        dtype: str

        When pat is a string and `regex` is False, every pat is replaced
        with repl as with ``str.replace()``:

        >>> s.str.replace('f.', 'ba', regex=False)
        0     foo
        1     fuz
        2    NaN
        dtype: str

        .. pandas-compat::
            :meth:`pandas.Series.str.replace`

            The parameters `case` and `flags` are not yet supported and will
            raise a `NotImplementedError` if anything other than the default
            value is set.
        """
        if case is not None:
            raise NotImplementedError("`case` parameter is not yet supported")
        if flags != 0:
            raise NotImplementedError("`flags` parameter is not yet supported")

        if can_convert_to_column(pat) and can_convert_to_column(repl):
            if n != -1:
                warnings.warn(
                    "`n` parameter is not supported when "
                    "`pat` and `repl` are list-like inputs"
                )

            if regex:
                raise ValueError(
                    "multiple pattern replace with regex=True is not supported"
                )
            else:
                result = self._column.replace_multiple(
                    as_column(pat, dtype=DEFAULT_STRING_DTYPE),  # type: ignore[arg-type]
                    as_column(repl, dtype=DEFAULT_STRING_DTYPE),  # type: ignore[arg-type]
                )
            return self._return_or_inplace(result)

        # If 'pat' is re.Pattern then get the pattern string from it
        if regex and isinstance(pat, re.Pattern):
            pat = pat.pattern

        if not isinstance(repl, str):
            raise TypeError(f"repl must be a str, not {type(repl).__name__}.")

        # Pandas forces non-regex replace when pat is a single-character
        if regex is True and len(pat) > 0:
            result = self._column.replace_re(
                pat,  # type: ignore[arg-type]
                repl,
                n,
            )
        else:
            result = self._column.replace_str(
                pat,  # type: ignore[arg-type]
                repl,
                n,
            )
        return self._return_or_inplace(result)

    def replace_with_backrefs(self, pat: str, repl: str) -> Series | Index:
        r"""
        Use the ``repl`` back-ref template to create a new string
        with the extracted elements found using the ``pat`` expression.

        Parameters
        ----------
        pat : str or compiled regex
            Regex with groupings to identify extract sections.
            This should not be a compiled regex.
        repl : str
            String template containing back-reference indicators.

        Returns
        -------
        Series/Index of str dtype

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["A543","Z756"])
        >>> s.str.replace_with_backrefs('(\\d)(\\d)', 'V\\2\\1')
        0    AV453
        1    ZV576
        dtype: str
        """
        # If 'pat' is re.Pattern then get the pattern string from it
        if isinstance(pat, re.Pattern):
            pat = pat.pattern
        return self._return_or_inplace(
            self._column.replace_with_backrefs(pat, repl)
        )

    def slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> Series | Index:
        """
        Slice substrings from each element in the Series or Index.

        Parameters
        ----------
        start : int, optional
            Start position for slice operation.
        stop : int, optional
            Stop position for slice operation.
        step : int, optional
            Step size for slice operation.

        Returns
        -------
        Series/Index of str dtype
            Series or Index from sliced substring from
            original string object.

        See Also
        --------
        slice_replace
            Replace a slice with a string.

        get
            Return element at position. Equivalent
            to ``Series.str.slice(start=i, stop=i+1)``
            with ``i`` being the position.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["koala", "fox", "chameleon"])
        >>> s
        0        koala
        1          fox
        2    chameleon
        dtype: str
        >>> s.str.slice(start=1)
        0        oala
        1          ox
        2    hameleon
        dtype: str
        >>> s.str.slice(start=-1)
        0    a
        1    x
        2    n
        dtype: str
        >>> s.str.slice(stop=2)
        0    ko
        1    fo
        2    ch
        dtype: str
        >>> s.str.slice(step=2)
        0      kaa
        1       fx
        2    caeen
        dtype: str
        >>> s.str.slice(start=0, stop=5, step=3)
        0    kl
        1     f
        2    cm
        dtype: str
        """
        return self._return_or_inplace(
            self._column.slice_strings(start, stop, step)
        )

    def isinteger(self) -> Series | Index:
        """
        Check whether all characters in each string form integer.

        If a string has zero characters, False is returned for
        that check.

        Returns
        -------
        Series or Index of bool
            Series or Index of boolean values with the same
            length as the original Series/Index.

        See Also
        --------
        isalnum
            Check whether all characters are alphanumeric.

        isalpha
            Check whether all characters are alphabetic.

        isdecimal
            Check whether all characters are decimal.

        isdigit
            Check whether all characters are digits.

        isnumeric
            Check whether all characters are numeric.

        isfloat
            Check whether all characters are float.

        islower
            Check whether all characters are lowercase.

        isspace
            Check whether all characters are whitespace.

        isupper
            Check whether all characters are uppercase.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["1", "0.1", "+100", "-15", "abc"])
        >>> s.str.isinteger()
        0     True
        1    False
        2     True
        3     True
        4    False
        dtype: bool
        >>> s = cudf.Series(["this is plan text", "", "10 10"])
        >>> s.str.isinteger()
        0    False
        1    False
        2    False
        dtype: bool
        """
        return self._return_or_inplace(self._column.is_integer())

    def ishex(self) -> Series | Index:
        """
        Check whether all characters in each string form a hex integer.

        If a string has zero characters, False is returned for
        that check.

        Returns
        -------
        Series or Index of bool
            Series or Index of boolean values with the same
            length as the original Series/Index.

        See Also
        --------
        isdecimal
            Check whether all characters are decimal.

        isdigit
            Check whether all characters are digits.

        isnumeric
            Check whether all characters are numeric.

        isfloat
            Check whether all characters are float.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["", "123DEF", "2D3", "-15", "abc"])
        >>> s.str.ishex()
        0    False
        1     True
        2     True
        3    False
        4     True
        dtype: bool
        """
        return self._return_or_inplace(self._column.is_hex())

    def istimestamp(self, format: str) -> Series | Index:
        """
        Check whether all characters in each string can be converted to
        a timestamp using the given format.

        Returns
        -------
        Series or Index of bool
            Series or Index of boolean values with the same
            length as the original Series/Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["20201101", "192011", "18200111", "2120-11-01"])
        >>> s.str.istimestamp("%Y%m%d")
        0     True
        1    False
        2     True
        3    False
        dtype: bool
        """
        return self._return_or_inplace(self._column.is_timestamp(format))

    def isfloat(self) -> Series | Index:
        r"""
        Check whether all characters in each string form floating value.

        If a string has zero characters, False is returned for
        that check.

        Returns
        -------
        Series or Index of bool
            Series or Index of boolean values with the same
            length as the original Series/Index.

        See Also
        --------
        isalnum
            Check whether all characters are alphanumeric.

        isalpha
            Check whether all characters are alphabetic.

        isdecimal
            Check whether all characters are decimal.

        isdigit
            Check whether all characters are digits.

        isinteger
            Check whether all characters are integer.

        isnumeric
            Check whether all characters are numeric.

        islower
            Check whether all characters are lowercase.

        isspace
            Check whether all characters are whitespace.

        isupper
            Check whether all characters are uppercase.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["1.1", "0.123213", "+0.123", "-100.0001", "234",
        ... "3-"])
        >>> s.str.isfloat()
        0     True
        1     True
        2     True
        3     True
        4     True
        5    False
        dtype: bool
        >>> s = cudf.Series(["this is plain text", "\t\n", "9.9", "9.9.9"])
        >>> s.str.isfloat()
        0    False
        1    False
        2     True
        3    False
        dtype: bool
        """
        return self._return_or_inplace(self._column.is_float())

    def isdecimal(self) -> Series | Index:
        """
        Check whether all characters in each string are decimal.

        This is equivalent to running the Python string method
        `str.isdecimal()
        <https://docs.python.org/3/library/stdtypes.html#str.isdecimal>`_
        for each element of the Series/Index.
        If a string has zero characters, False is returned for
        that check.

        Returns
        -------
        Series or Index of bool
            Series or Index of boolean values with the same
            length as the original Series/Index.

        See Also
        --------
        isalnum
            Check whether all characters are alphanumeric.

        isalpha
            Check whether all characters are alphabetic.

        isdigit
            Check whether all characters are digits.

        isinteger
            Check whether all characters are integer.

        isnumeric
            Check whether all characters are numeric.

        isfloat
            Check whether all characters are float.

        islower
            Check whether all characters are lowercase.

        isspace
            Check whether all characters are whitespace.

        isupper
            Check whether all characters are uppercase.

        Examples
        --------
        >>> import cudf
        >>> s3 = cudf.Series(['23', '³', '⅕', ''])

        The s3.str.isdecimal method checks for characters used to form
        numbers in base 10.

        >>> s3.str.isdecimal()
        0     True
        1    False
        2    False
        3    False
        dtype: bool
        """
        return self._return_or_inplace(
            self._column.all_characters_of_type(
                plc.strings.char_types.StringCharacterTypes.DECIMAL
            )
        )

    def isalnum(self) -> Series | Index:
        """
        Check whether all characters in each string are alphanumeric.

        This is equivalent to running the Python string method
        `str.isalnum()
        <https://docs.python.org/3/library/stdtypes.html#str.isalnum>`_
        for each element of the Series/Index. If a string has zero
        characters, False is returned for that check.

        Equivalent to: ``isalpha() or isdigit() or isnumeric() or isdecimal()``

        Returns
        -------
        Series or Index of bool
            Series or Index of boolean values with the
            same length as the original Series/Index.

        See Also
        --------
        isalpha
            Check whether all characters are alphabetic.

        isdecimal
            Check whether all characters are decimal.

        isdigit
            Check whether all characters are digits.

        isinteger
            Check whether all characters are integer.

        isnumeric
            Check whether all characters are numeric.

        isfloat
            Check whether all characters are float.

        islower
            Check whether all characters are lowercase.

        isspace
            Check whether all characters are whitespace.

        isupper
            Check whether all characters are uppercase.

        Examples
        --------
        >>> import cudf
        >>> s1 = cudf.Series(['one', 'one1', '1', ''])
        >>> s1.str.isalnum()
        0     True
        1     True
        2     True
        3    False
        dtype: bool

        Note that checks against characters mixed with
        any additional punctuation or whitespace will
        evaluate to false for an alphanumeric check.

        >>> s2 = cudf.Series(['A B', '1.5', '3,000'])
        >>> s2.str.isalnum()
        0    False
        1    False
        2    False
        dtype: bool
        """
        return self._return_or_inplace(
            self._column.all_characters_of_type(
                plc.strings.char_types.StringCharacterTypes.ALPHANUM
            )
        )

    def isalpha(self) -> Series | Index:
        """
        Check whether all characters in each string are alphabetic.

        This is equivalent to running the Python string method
        `str.isalpha()
        <https://docs.python.org/3/library/stdtypes.html#str.isalpha>`_
        for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.

        Returns
        -------
        Series or Index of bool
            Series or Index of boolean values with the same length
            as the original Series/Index.

        See Also
        --------
        isalnum
            Check whether all characters are alphanumeric.

        isdecimal
            Check whether all characters are decimal.

        isdigit
            Check whether all characters are digits.

        isinteger
            Check whether all characters are integer.

        isnumeric
            Check whether all characters are numeric.

        isfloat
            Check whether all characters are float.

        islower
            Check whether all characters are lowercase.

        isspace
            Check whether all characters are whitespace.

        isupper
            Check whether all characters are uppercase.

        Examples
        --------
        >>> import cudf
        >>> s1 = cudf.Series(['one', 'one1', '1', ''])
        >>> s1.str.isalpha()
        0     True
        1    False
        2    False
        3    False
        dtype: bool
        """
        return self._return_or_inplace(
            self._column.all_characters_of_type(
                plc.strings.char_types.StringCharacterTypes.ALPHA
            )
        )

    def isdigit(self) -> Series | Index:
        """
        Check whether all characters in each string are digits.

        This is equivalent to running the Python string method
        `str.isdigit()
        <https://docs.python.org/3/library/stdtypes.html#str.isdigit>`_
        for each element of the Series/Index.
        If a string has zero characters, False is returned
        for that check.

        Returns
        -------
        Series or Index of bool
            Series or Index of boolean values with the same
            length as the original Series/Index.

        See Also
        --------
        isalnum
            Check whether all characters are alphanumeric.

        isalpha
            Check whether all characters are alphabetic.

        isdecimal
            Check whether all characters are decimal.

        isinteger
            Check whether all characters are integer.

        isnumeric
            Check whether all characters are numeric.

        isfloat
            Check whether all characters are float.

        islower
            Check whether all characters are lowercase.

        isspace
            Check whether all characters are whitespace.

        isupper
            Check whether all characters are uppercase.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['23', '³', '⅕', ''])

        The ``s.str.isdigit`` method is the same as ``s.str.isdecimal`` but
        also includes special digits, like superscripted and
        subscripted digits in unicode.

        >>> s.str.isdigit()
        0     True
        1     True
        2    False
        3    False
        dtype: bool
        """
        return self._return_or_inplace(
            self._column.all_characters_of_type(
                plc.strings.char_types.StringCharacterTypes.DIGIT
            )
        )

    def isnumeric(self) -> Series | Index:
        """
        Check whether all characters in each string are numeric.

        This is equivalent to running the Python string method
        `str.isnumeric()
        <https://docs.python.org/3/library/stdtypes.html#str.isnumeric>`_
        for each element of the Series/Index. If a
        string has zero characters, False is returned for that check.

        Returns
        -------
        Series or Index of bool
            Series or Index of boolean values with the same
            length as the original Series/Index.

        See Also
        --------
        isalnum
            Check whether all characters are alphanumeric.

        isalpha
            Check whether all characters are alphabetic.

        isdecimal
            Check whether all characters are decimal.

        isdigit
            Check whether all characters are digits.

        isinteger
            Check whether all characters are integer.

        isfloat
            Check whether all characters are float.

        islower
            Check whether all characters are lowercase.

        isspace
            Check whether all characters are whitespace.

        isupper
            Check whether all characters are uppercase.

        Examples
        --------
        >>> import cudf
        >>> s1 = cudf.Series(['one', 'one1', '1', ''])
        >>> s1.str.isnumeric()
        0    False
        1    False
        2     True
        3    False
        dtype: bool

        The ``s1.str.isnumeric`` method is the same as ``s2.str.isdigit`` but
        also includes other characters that can represent
        quantities such as unicode fractions.

        >>> s2 = cudf.Series(['23', '³', '⅕', ''], dtype='str')
        >>> s2.str.isnumeric()
        0     True
        1     True
        2     True
        3    False
        dtype: bool
        """
        return self._return_or_inplace(
            self._column.all_characters_of_type(
                plc.strings.char_types.StringCharacterTypes.NUMERIC
            )
        )

    def isupper(self) -> Series | Index:
        """
        Check whether all characters in each string are uppercase.

        This is equivalent to running the Python string method
        `str.isupper()
        <https://docs.python.org/3/library/stdtypes.html#str.isupper>`_
        for each element of the Series/Index.
        If a string has zero characters, False is returned
        for that check.

        Returns
        -------
        Series or Index of bool
            Series or Index of boolean values with the same
            length as the original Series/Index.

        See Also
        --------
        isalnum
            Check whether all characters are alphanumeric.

        isalpha
            Check whether all characters are alphabetic.

        isdecimal
            Check whether all characters are decimal.

        isdigit
            Check whether all characters are digits.

        isinteger
            Check whether all characters are integer.

        isnumeric
            Check whether all characters are numeric.

        isfloat
            Check whether all characters are float.

        islower
            Check whether all characters are lowercase.

        isspace
            Check whether all characters are whitespace.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['leopard', 'Golden Eagle', 'SNAKE', ''])
        >>> s.str.isupper()
        0    False
        1    False
        2     True
        3    False
        dtype: bool
        """
        return self._return_or_inplace(
            self._column.all_characters_of_type(
                plc.strings.char_types.StringCharacterTypes.UPPER,
                plc.strings.char_types.StringCharacterTypes.CASE_TYPES,
            )
        )

    def islower(self) -> Series | Index:
        """
        Check whether all characters in each string are lowercase.

        This is equivalent to running the Python string method
        `str.islower()
        <https://docs.python.org/3/library/stdtypes.html#str.islower>`_
        for each element of the Series/Index.
        If a string has zero characters, False is returned
        for that check.

        Returns
        -------
        Series or Index of bool
            Series or Index of boolean values with the same
            length as the original Series/Index.

        See Also
        --------
        isalnum
            Check whether all characters are alphanumeric.

        isalpha
            Check whether all characters are alphabetic.

        isdecimal
            Check whether all characters are decimal.

        isdigit
            Check whether all characters are digits.

        isinteger
            Check whether all characters are integer.

        isnumeric
            Check whether all characters are numeric.

        isfloat
            Check whether all characters are float.

        isspace
            Check whether all characters are whitespace.

        isupper
            Check whether all characters are uppercase.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['leopard', 'Golden Eagle', 'SNAKE', ''])
        >>> s.str.islower()
        0     True
        1    False
        2    False
        3    False
        dtype: bool
        """
        return self._return_or_inplace(
            self._column.all_characters_of_type(
                plc.strings.char_types.StringCharacterTypes.LOWER,
                plc.strings.char_types.StringCharacterTypes.CASE_TYPES,
            )
        )

    def isipv4(self) -> Series | Index:
        """
        Check whether all characters in each string form an IPv4 address.

        If a string has zero characters, False is returned for
        that check.

        Returns
        -------
        Series or Index of bool
            Series or Index of boolean values with the same
            length as the original Series/Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["", "127.0.0.1", "255.255.255.255", "123.456"])
        >>> s.str.isipv4()
        0    False
        1     True
        2     True
        3    False
        dtype: bool
        """
        return self._return_or_inplace(self._column.is_ipv4())

    def lower(self) -> Series | Index:
        """
        Converts all characters to lowercase.

        Equivalent to `str.lower()
        <https://docs.python.org/3/library/stdtypes.html#str.lower>`_.

        Returns
        -------
        Series or Index of object
            A copy of the object with all strings converted to lowercase.

        See Also
        --------
        upper
            Converts all characters to uppercase.

        title
            Converts first character of each word to uppercase and remaining
            to lowercase.

        capitalize
            Converts first character to uppercase and remaining to lowercase.

        swapcase
            Converts uppercase to lowercase and lowercase to uppercase.

        Examples
        --------
        >>> import cudf
        >>> data = ['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe']
        >>> s = cudf.Series(data)
        >>> s.str.lower()
        0                 lower
        1              capitals
        2    this is a sentence
        3              swapcase
        dtype: str
        """
        return self._return_or_inplace(self._column.to_lower())

    def upper(self) -> Series | Index:
        """
        Convert each string to uppercase.
        This only applies to ASCII characters at this time.

        Equivalent to `str.upper()
        <https://docs.python.org/3/library/stdtypes.html#str.upper>`_.

        Returns
        -------
        Series or Index of object

        See Also
        --------
        lower
            Converts all characters to lowercase.

        upper
            Converts all characters to uppercase.

        title
            Converts first character of each word to uppercase and
            remaining to lowercase.

        capitalize
            Converts first character to uppercase and remaining to
            lowercase.

        swapcase
            Converts uppercase to lowercase and lowercase to uppercase.

        Examples
        --------
        >>> import cudf
        >>> data = ['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe']
        >>> s = cudf.Series(data)
        >>> s
        0                 lower
        1              CAPITALS
        2    this is a sentence
        3              SwApCaSe
        dtype: str
        >>> s.str.upper()
        0                 LOWER
        1              CAPITALS
        2    THIS IS A SENTENCE
        3              SWAPCASE
        dtype: str
        """
        return self._return_or_inplace(self._column.to_upper())

    def capitalize(self) -> Series | Index:
        """
        Convert strings in the Series/Index to be capitalized.
        This only applies to ASCII characters at this time.

        Returns
        -------
        Series or Index of object

        Examples
        --------
        >>> import cudf
        >>> data = ['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe']
        >>> s = cudf.Series(data)
        >>> s.str.capitalize()
        0                 Lower
        1              Capitals
        2    This is a sentence
        3              Swapcase
        dtype: str
        >>> s = cudf.Series(["hello, friend","goodbye, friend"])
        >>> s.str.capitalize()
        0      Hello, friend
        1    Goodbye, friend
        dtype: str
        """
        return self._return_or_inplace(self._column.capitalize())

    def swapcase(self) -> Series | Index:
        """
        Change each lowercase character to uppercase and vice versa.
        This only applies to ASCII characters at this time.

        Equivalent to `str.swapcase()
        <https://docs.python.org/3/library/stdtypes.html#str.swapcase>`_.

        Returns
        -------
        Series or Index of object

        See Also
        --------
        lower
            Converts all characters to lowercase.

        upper
            Converts all characters to uppercase.

        title
            Converts first character of each word to uppercase and remaining
            to lowercase.

        capitalize
            Converts first character to uppercase and remaining to lowercase.

        Examples
        --------
        >>> import cudf
        >>> data = ['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe']
        >>> s = cudf.Series(data)
        >>> s
        0                 lower
        1              CAPITALS
        2    this is a sentence
        3              SwApCaSe
        dtype: str
        >>> s.str.swapcase()
        0                 LOWER
        1              capitals
        2    THIS IS A SENTENCE
        3              sWaPcAsE
        dtype: str
        """
        return self._return_or_inplace(self._column.swapcase())

    def title(self) -> Series | Index:
        """
        Uppercase the first letter of each letter after a space
        and lowercase the rest.
        This only applies to ASCII characters at this time.

        Equivalent to `str.title()
        <https://docs.python.org/3/library/stdtypes.html#str.title>`_.

        Returns
        -------
        Series or Index of object

        See Also
        --------
        lower
            Converts all characters to lowercase.

        upper
            Converts all characters to uppercase.

        capitalize
            Converts first character to uppercase and remaining to lowercase.

        swapcase
            Converts uppercase to lowercase and lowercase to uppercase.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["lower", "CAPITALS", "this is a sentence", "SwApCaSe"])
        >>> s
        0                 lower
        1              CAPITALS
        2    this is a sentence
        3              SwApCaSe
        dtype: str
        >>> s.str.title()
        0                 Lower
        1              Capitals
        2    This Is A Sentence
        3              Swapcase
        dtype: str
        """
        return self._return_or_inplace(self._column.title())

    def istitle(self) -> Series | Index:
        """
        Check whether each string is title formatted.
        The first letter of each word should be uppercase and the rest
        should be lowercase.

        Equivalent to :meth:`str.istitle`.

        Returns
        -------
        Series or Index of object

        Examples
        --------
        >>> import cudf
        >>> series = cudf.Series(["leopard", "Golden Eagle", "SNAKE", ""])
        >>> series.str.istitle()
        0    False
        1     True
        2    False
        3    False
        dtype: bool
        """
        return self._return_or_inplace(self._column.is_title())

    def filter_alphanum(
        self, repl: str | None = None, keep: bool = True
    ) -> Series | Index:
        """
        Remove non-alphanumeric characters from strings in this column.

        Parameters
        ----------
        repl : str
            Optional string to use in place of removed characters.
        keep : bool
            Set to False to remove all alphanumeric characters instead
            of keeping them.

        Returns
        -------
        Series/Index of str dtype
            Strings with only alphanumeric characters.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["pears £12", "plums $34", "Temp 72℉", "100K℧"])
        >>> s.str.filter_alphanum(" ")
        0    pears  12
        1    plums  34
        2     Temp 72
        3        100K
        dtype: str
        """
        if repl is None:
            repl = ""
        try:
            plc_repl = plc.Scalar.from_py(
                repl, dtype=plc.DataType(plc.TypeId.STRING)
            )
        except (NotImplementedError, TypeError) as err:
            raise TypeError(
                f"repl should be a string, but got {type(repl).__name__}"
            ) from err
        if keep:
            types_to_remove = (
                plc.strings.char_types.StringCharacterTypes.ALL_TYPES
            )
            types_to_keep = (
                plc.strings.char_types.StringCharacterTypes.ALPHANUM
            )
        else:
            types_to_remove = (
                plc.strings.char_types.StringCharacterTypes.ALPHANUM
            )
            types_to_keep = (
                plc.strings.char_types.StringCharacterTypes.ALL_TYPES
            )

        return self._return_or_inplace(
            self._column.filter_characters_of_type(
                types_to_remove,
                plc_repl,
                types_to_keep,
            )
        )

    def slice_from(self, starts: Series, stops: Series) -> Series | Index:
        """
        Return substring of each string using positions for each string.

        The starts and stops parameters are of Column type.

        Parameters
        ----------
        starts : Series
            Beginning position of each the string to extract.
            Default is beginning of the each string.
        stops : Series
            Ending position of the each string to extract.
            Default is end of each string.
            Use -1 to specify to the end of that string.

        Returns
        -------
        Series/Index of str dtype
            A substring of each string using positions for each string.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["hello","there"])
        >>> s
        0    hello
        1    there
        dtype: str
        >>> starts = cudf.Series([1, 3])
        >>> stops = cudf.Series([5, 5])
        >>> s.str.slice_from(starts, stops)
        0    ello
        1      re
        dtype: str
        """
        return self._return_or_inplace(
            self._column.slice_strings(starts._column, stops._column)
        )

    def slice_replace(
        self,
        start: int | None = None,
        stop: int | None = None,
        repl: str | None = None,
    ) -> Series | Index:
        """
        Replace the specified section of each string with a new string.

        Parameters
        ----------
        start : int, optional
            Beginning position of the string to replace.
            Default is beginning of the each string.
        stop : int, optional
            Ending position of the string to replace.
            Default is end of each string.
        repl : str, optional
            String to insert into the specified position values.

        Returns
        -------
        Series/Index of str dtype
            A new string with the specified section of the string
            replaced with `repl` string.

        See Also
        --------
        slice
            Just slicing without replacement.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['a', 'ab', 'abc', 'abdc', 'abcde'])
        >>> s
        0        a
        1       ab
        2      abc
        3     abdc
        4    abcde
        dtype: str

        Specify just `start`, meaning replace `start` until the `end` of
        the string with `repl`.

        >>> s.str.slice_replace(1, repl='X')
        0    aX
        1    aX
        2    aX
        3    aX
        4    aX
        dtype: str

        Specify just `stop`, meaning the `start` of the string to `stop`
        is replaced with `repl`, and the rest of the string is included.

        >>> s.str.slice_replace(stop=2, repl='X')
        0       X
        1       X
        2      Xc
        3     Xdc
        4    Xcde
        dtype: str

        Specify `start` and `stop`, meaning the slice from `start`
        to `stop` is replaced with `repl`. Everything before or
        after `start` and `stop` is included as is.

        >>> s.str.slice_replace(start=1, stop=3, repl='X')
        0      aX
        1      aX
        2      aX
        3     aXc
        4    aXde
        dtype: str
        """
        if start is None:
            start = 0

        if stop is None:
            stop = -1

        if repl is None:
            repl = ""

        try:
            plc_repl = plc.Scalar.from_py(
                repl, dtype=plc.DataType(plc.TypeId.STRING)
            )
        except (NotImplementedError, TypeError) as err:
            raise TypeError(
                f"repl should be a string, but got {type(repl).__name__}"
            ) from err
        return self._return_or_inplace(
            self._column.replace_slice(start, stop, plc_repl)
        )

    def insert(
        self, start: int = 0, repl: str | None = None
    ) -> Series | Index:
        """
        Insert the specified string into each string in the specified
        position.

        Parameters
        ----------
        start : int
            Beginning position of the string to replace.
            Default is beginning of the each string.
            Specify -1 to insert at the end of each string.
        repl : str
            String to insert into the specified position value.

        Returns
        -------
        Series/Index of str dtype
            A new string series with the specified string
            inserted at the specified position.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["abcdefghij", "0123456789"])
        >>> s.str.insert(2, '_')
        0    ab_cdefghij
        1    01_23456789
        dtype: str

        When no `repl` is passed, nothing is inserted.

        >>> s.str.insert(2)
        0    abcdefghij
        1    0123456789
        dtype: str

        Negative values are also supported for `start`.

        >>> s.str.insert(-1,'_')
        0    abcdefghij_
        1    0123456789_
        dtype: str
        """
        return self.slice_replace(start, start, repl)

    def get(self, i: int = 0) -> Series | Index:
        """
        Extract element from each component at specified position.

        Parameters
        ----------
        i : int
            Position of element to extract.

        Returns
        -------
        Series/Index of str dtype

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["hello world", "rapids", "cudf"])
        >>> s
        0    hello world
        1         rapids
        2           cudf
        dtype: str
        >>> s.str.get(10)
        0       d
        1    NaN
        2    NaN
        dtype: str
        >>> s.str.get(1)
        0    e
        1    a
        2    u
        dtype: str

        ``get`` also accepts negative index number.

        >>> s.str.get(-1)
        0    d
        1    s
        2    f
        dtype: str
        """
        if isinstance(self._column.dtype, ListDtype):
            return ListMethods(self._parent).get(i)
        str_lens = self.len()
        if i < 0:
            next_index = i - 1
            step = -1
            to_mask = str_lens < abs(i)
        else:
            next_index = i + 1
            step = 1
            to_mask = str_lens <= i
        result = self.slice(i, next_index, step)
        if to_mask.any():
            result[to_mask] = pd.NA  # type: ignore[index]
        return result

    def get_json_object(
        self,
        json_path: str,
        *,
        allow_single_quotes: bool = False,
        strip_quotes_from_single_strings: bool = True,
        missing_fields_as_nulls: bool = False,
    ) -> Series | Index:
        r"""
        Applies a JSONPath string to an input strings column
        where each row in the column is a valid json string

        Parameters
        ----------
        json_path : str
            The JSONPath string to be applied to each row
            of the input column
        allow_single_quotes : bool, default False
            If True, representing strings with single
            quotes is allowed.
            If False, strings must only be represented
            with double quotes.
        strip_quotes_from_single_strings : bool, default True
            If True, strip the quotes from the return value of
            a given row if it is a string.
            If False, values returned for a given row include
            quotes if they are strings.
        missing_fields_as_nulls : bool, default False
            If True, when an object is queried for a field
            it does not contain, "null" is returned.
            If False, when an object is queried for a field
            it does not contain, None is returned.

        Returns
        -------
        Column: New strings column containing the retrieved json object strings

        Examples
        --------
        >>> import cudf
        >>> series = cudf.Series(
        ...     [
        ...         '{"store":{"book":[{"category":"reference",'
        ...         '"author":"Nigel Rees","title":"Sayings of the Century",'
        ...         '"price":8.95},{"category":"fiction",'
        ...         '"author":"Evelyn Waugh","title":"Sword of Honour",'
        ...         '"price":12.99}]}}'
        ...     ]
        ... )
        >>> series.str.get_json_object("$.store.book")
        0    [{"category":"reference","author":"Nigel Rees"...
        dtype: str
        """
        return self._return_or_inplace(
            self._column.get_json_object(
                json_path,
                allow_single_quotes,
                strip_quotes_from_single_strings,
                missing_fields_as_nulls,
            )
        )

    def split(
        self,
        pat: str | None = None,
        n: int = -1,
        expand: bool = False,
        regex: bool | None = None,
    ) -> Series | Index:
        """
        Split strings around given separator/delimiter.

        Splits the string in the Series/Index from the beginning, at the
        specified delimiter string. Similar to `str.split()
        <https://docs.python.org/3/library/stdtypes.html#str.split>`_.

        Parameters
        ----------
        pat : str, default None
            String or regular expression to split on. If not specified, split
            on whitespace.
        n : int, default -1 (all)
            Limit number of splits in output. `None`, 0, and -1 will all be
            interpreted as "all splits".
        expand : bool, default False
            Expand the split strings into separate columns.

            * If ``True``, return DataFrame/MultiIndex expanding
              dimensionality.
            * If ``False``, return Series/Index, containing lists
              of strings.
        regex : bool, default None
            Determines if the passed-in pattern is a regular expression:

            * If ``True``, assumes the passed-in pattern is a regular
              expression
            * If ``False``, treats the pattern as a literal string.
            * If pat length is 1, treats pat as a literal string.

        Returns
        -------
        Series, Index, DataFrame or MultiIndex
            Type matches caller unless ``expand=True`` (see Notes).

        See Also
        --------
        rsplit
            Splits string around given separator/delimiter, starting from
            the right.

        str.split
            Standard library version for split.

        str.rsplit
            Standard library version for rsplit.

        Notes
        -----
        The handling of the n keyword depends on the number
        of found splits:

            - If found splits > n, make first n splits only
            - If found splits <= n, make all splits
            - If for a certain row the number of found
              splits < n, append None for padding up to n
              if ``expand=True``.

        If using ``expand=True``, Series and Index callers return
        DataFrame and MultiIndex objects, respectively.

        Examples
        --------
        >>> import cudf
        >>> data = ["this is a regular sentence",
        ...     "https://docs.python.org/index.html", None]
        >>> s = cudf.Series(data)
        >>> s
        0            this is a regular sentence
        1    https://docs.python.org/index.html
        2                                   NaN
        dtype: str

        In the default setting, the string is split by whitespace.

        >>> s.str.split()
        0        [this, is, a, regular, sentence]
        1    [https://docs.python.org/index.html]
        2                                    None
        dtype: list

        Without the ``n`` parameter, the outputs of ``rsplit``
        and ``split`` are identical.

        >>> s.str.rsplit()
        0        [this, is, a, regular, sentence]
        1    [https://docs.python.org/index.html]
        2                                    None
        dtype: list

        The `n` parameter can be used to limit the number of
        splits on the delimiter.

        >>> s.str.split(n=2)
        0          [this, is, a regular sentence]
        1    [https://docs.python.org/index.html]
        2                                    None
        dtype: list

        The `pat` parameter can be used to split by other characters.

        >>> s.str.split(pat="/")
        0               [this is a regular sentence]
        1    [https:, , docs.python.org, index.html]
        2                                       None
        dtype: list

        When using ``expand=True``, the split elements will expand out
        into separate columns. If ``<NA>`` value is present, it is propagated
        throughout the columns during the split.

        >>> s.str.split(expand=True)
                                            0     1     2        3         4
        0                                this    is     a  regular  sentence
        1  https://docs.python.org/index.html   NaN   NaN      NaN       NaN
        2                                 NaN   NaN   NaN      NaN       NaN
        """

        if expand not in (True, False):
            raise ValueError(
                f"expand parameter accepts only : [True, False], got {expand}"
            )

        # Pandas treats 0 as all
        if n is None or n == 0:
            n = -1

        if pat is None:
            pat = ""
            regex = False

        if regex and isinstance(pat, re.Pattern):
            pat = pat.pattern

        if regex is None:
            if len(str(pat)) <= 1:
                regex = False
            else:
                regex = True

        result_table: StringColumn | dict[int, StringColumn]
        if expand:
            if self._column.is_all_null:
                result_table = {0: self._column.copy()}
            else:
                if regex is True:
                    data = self._column.split_re(pat, n)
                else:
                    data = self._column.split(
                        plc.Scalar.from_py(
                            pat, dtype=plc.DataType(plc.TypeId.STRING)
                        ),
                        n,
                    )
                if len(data) == 1 and data[0].is_all_null:
                    result_table = {}
                else:
                    result_table = data
        else:
            if regex is True:
                result_table = self._column.split_record_re(pat, n)
            else:
                result_table = self._column.split_record(
                    plc.Scalar.from_py(
                        pat, dtype=plc.DataType(plc.TypeId.STRING)
                    ),
                    n,
                )

        return self._return_or_inplace(result_table, expand=expand)

    def rsplit(
        self,
        pat: str | None = None,
        n: int = -1,
        expand: bool = False,
        regex: bool | None = None,
    ) -> Series | Index:
        """
        Split strings around given separator/delimiter.

        Splits the string in the Series/Index from the end, at the
        specified delimiter string. Similar to `str.rsplit()
        <https://docs.python.org/3/library/stdtypes.html#str.rsplit>`_.

        Parameters
        ----------
        pat : str, default ' ' (space)
            String to split on, does not yet support regular expressions.
        n : int, default -1 (all)
            Limit number of splits in output. `None`, 0, and -1 will all be
            interpreted as "all splits".
        expand : bool, default False
            Expand the split strings into separate columns.

            * If ``True``, return DataFrame/MultiIndex expanding
              dimensionality.
            * If ``False``, return Series/Index, containing lists
              of strings.
        regex : bool, default None
            Determines if the passed-in pattern is a regular expression:

            * If ``True``, assumes the passed-in pattern is a regular
              expression
            * If ``False``, treats the pattern as a literal string.
            * If pat length is 1, treats pat as a literal string.

        Returns
        -------
        Series, Index, DataFrame or MultiIndex
            Type matches caller unless ``expand=True`` (see Notes).

        See Also
        --------
        split
            Split strings around given separator/delimiter.

        str.split
            Standard library version for split.

        str.rsplit
            Standard library version for rsplit.

        Notes
        -----
        The handling of the n keyword depends on the number of
        found splits:

        - If found splits > n, make first n splits only
        - If found splits <= n, make all splits
        - If for a certain row the number of found splits < n,
          append None for padding up to n if ``expand=True``.

        If using ``expand=True``, Series and Index callers return
        DataFrame and MultiIndex objects, respectively.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(
        ...     [
        ...         "this is a regular sentence",
        ...         "https://docs.python.org/3/tutorial/index.html",
        ...         None
        ...     ]
        ... )
        >>> s
        0                       this is a regular sentence
        1    https://docs.python.org/3/tutorial/index.html
        2                                              NaN
        dtype: str

        In the default setting, the string is split by whitespace.

        >>> s.str.rsplit()
        0                   [this, is, a, regular, sentence]
        1    [https://docs.python.org/3/tutorial/index.html]
        2                                               None
        dtype: list

        Without the ``n`` parameter, the outputs of ``rsplit``
        and ``split`` are identical.

        >>> s.str.split()
        0                   [this, is, a, regular, sentence]
        1    [https://docs.python.org/3/tutorial/index.html]
        2                                               None
        dtype: list

        The n parameter can be used to limit the number of
        splits on the delimiter. The outputs of split and rsplit are different.

        >>> s.str.rsplit(n=2)
        0                     [this is a, regular, sentence]
        1    [https://docs.python.org/3/tutorial/index.html]
        2                                               None
        dtype: list
        >>> s.str.split(n=2)
        0                     [this, is, a regular sentence]
        1    [https://docs.python.org/3/tutorial/index.html]
        2                                               None
        dtype: list

        When using ``expand=True``, the split elements will expand
        out into separate columns. If a null value is present,
        it is propagated throughout the columns during the split.

        >>> s.str.rsplit(n=2, expand=True)
                                                       0        1         2
        0                                      this is a  regular  sentence
        1  https://docs.python.org/3/tutorial/index.html      NaN       NaN
        2                                            NaN      NaN       NaN

        For slightly more complex use cases like splitting the
        html document name from a url, a combination of parameter
        settings can be used.

        >>> s.str.rsplit("/", n=1, expand=True)
                                            0           1
        0          this is a regular sentence         NaN
        1  https://docs.python.org/3/tutorial  index.html
        2                                 NaN         NaN
        """

        if expand not in (True, False):
            raise ValueError(
                f"expand parameter accepts only : [True, False], got {expand}"
            )

        # Pandas treats 0 as all
        if n == 0:
            n = -1

        if pat is None:
            pat = ""

        if regex and isinstance(pat, re.Pattern):
            pat = pat.pattern

        result_table: StringColumn | dict[int, StringColumn]
        if expand:
            if self._column.is_all_null:
                result_table = {0: self._column.copy()}
            else:
                if regex is True:
                    data = self._column.rsplit_re(pat, n)
                else:
                    data = self._column.rsplit(
                        plc.Scalar.from_py(
                            pat, dtype=plc.DataType(plc.TypeId.STRING)
                        ),
                        n,
                    )
                if len(data) == 1 and data[0].is_all_null:
                    result_table = {}
                else:
                    result_table = data
        else:
            if regex is True:
                result_table = self._column.rsplit_record_re(pat, n)
            else:
                result_table = self._column.rsplit_record(
                    plc.Scalar.from_py(
                        pat, dtype=plc.DataType(plc.TypeId.STRING)
                    ),
                    n,
                )

        return self._return_or_inplace(result_table, expand=expand)

    def split_part(
        self, delimiter: str | None = None, index: int = 0
    ) -> Series | Index:
        """
        Splits the string by delimiter and returns the token at the given index.

        Parameters
        ----------
        delimiter : str, default None
            The string to split on. If not specified, split on whitespace.
        index : int, default 0
            The index of the token to retrieve.

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["a_b_c", "d_e", "f"])
        >>> s.str.split_part(delimiter="_", index=1)
        0       b
        1       e
        2    NaN
        dtype: str
        """

        if delimiter is None:
            delimiter = ""
        delim_scalar = plc.Scalar.from_py(delimiter)
        return self._return_or_inplace(
            self._column.split_part(delim_scalar, index)
        )

    def partition(self, sep: str = " ", expand: bool = True) -> Series | Index:
        """
        Split the string at the first occurrence of sep.

        This method splits the string at the first occurrence
        of sep, and returns 3 elements containing the part
        before the separator, the separator itself, and the
        part after the separator. If the separator is not found,
        return 3 elements containing the string itself, followed
        by two empty strings.

        Parameters
        ----------
        sep : str, default ' ' (whitespace)
            String to split on.

        Returns
        -------
        DataFrame or MultiIndex
            Returns a DataFrame / MultiIndex

        See Also
        --------
        rpartition
            Split the string at the last occurrence of sep.

        split
            Split strings around given separators.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['Linda van der Berg', 'George Pitt-Rivers'])
        >>> s
        0    Linda van der Berg
        1    George Pitt-Rivers
        dtype: str

        >>> s.str.partition()
                0  1             2
        0   Linda     van der Berg
        1  George      Pitt-Rivers

        To partition by something different than a space:

        >>> s.str.partition('-')
                            0  1       2
        0  Linda van der Berg
        1         George Pitt  -  Rivers

        Also available on indices:

        >>> idx = cudf.Index(['X 123', 'Y 999'])
        >>> idx
        Index(['X 123', 'Y 999'], dtype='str')

        Which will create a MultiIndex:

        >>> idx.str.partition()
        MultiIndex([('X', ' ', '123'),
                    ('Y', ' ', '999')],
                   )

        .. pandas-compat::
            :meth:`pandas.Series.str.partition`

            The parameter `expand` is not yet supported and will raise a
            `NotImplementedError` if anything other than the default
            value is set.

        """
        if expand is not True:
            raise NotImplementedError(
                "`expand=False` is currently not supported"
            )

        if sep is None:
            sep = " "
        if not isinstance(sep, str):
            raise TypeError(
                f"sep should be a string, but got {type(sep).__name__}"
            )
        return self._return_or_inplace(
            self._column.partition(
                plc.Scalar.from_py(sep, dtype=plc.DataType(plc.TypeId.STRING)),
            ),
            expand=expand,
        )

    def rpartition(
        self, sep: str = " ", expand: bool = True
    ) -> Series | Index:
        """
        Split the string at the last occurrence of sep.

        This method splits the string at the last occurrence
        of sep, and returns 3 elements containing the part
        before the separator, the separator itself, and the
        part after the separator. If the separator is not
        found, return 3 elements containing two empty strings,
        followed by the string itself.

        Parameters
        ----------
        sep : str, default ' ' (whitespace)
            String to split on.

        Returns
        -------
        DataFrame or MultiIndex
            Returns a DataFrame / MultiIndex

        Notes
        -----
        The parameter `expand` is not yet supported and will raise a
        `NotImplementedError` if anything other than the default value is set.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['Linda van der Berg', 'George Pitt-Rivers'])
        >>> s
        0    Linda van der Berg
        1    George Pitt-Rivers
        dtype: str
        >>> s.str.rpartition()
                    0  1            2
        0  Linda van der            Berg
        1         George     Pitt-Rivers

        Also available on indices:

        >>> idx = cudf.Index(['X 123', 'Y 999'])
        >>> idx
        Index(['X 123', 'Y 999'], dtype='str')

        Which will create a MultiIndex:

        >>> idx.str.rpartition()
        MultiIndex([('X', ' ', '123'),
                    ('Y', ' ', '999')],
                   )
        """
        if expand is not True:
            raise NotImplementedError(
                "`expand=False` is currently not supported"
            )

        if sep is None:
            sep = " "
        if not isinstance(sep, str):
            raise TypeError(
                f"sep should be a string, but got {type(sep).__name__}"
            )
        return self._return_or_inplace(
            self._column.rpartition(
                plc.Scalar.from_py(sep, dtype=plc.DataType(plc.TypeId.STRING)),
            ),
            expand=expand,
        )

    def pad(
        self,
        width: int,
        side: Literal["left", "both", "right"] = "left",
        fillchar: str = " ",
    ) -> Series | Index:
        """
        Pad strings in the Series/Index up to width.

        Parameters
        ----------
        width : int
            Minimum width of resulting string;
            additional characters will be filled with
            character defined in fillchar.

        side : {'left', 'right', 'both'}, default 'left'
            Side from which to fill resulting string.

        fillchar : str,  default ' ' (whitespace)
            Additional character for filling, default is whitespace.

        Returns
        -------
        Series/Index of object
            Returns Series or Index with minimum number
            of char in object.

        See Also
        --------
        rjust
            Fills the left side of strings with an arbitrary character.
            Equivalent to ``Series.str.pad(side='left')``.

        ljust
            Fills the right side of strings with an arbitrary character.
            Equivalent to ``Series.str.pad(side='right')``.

        center
            Fills both sides of strings with an arbitrary character.
            Equivalent to ``Series.str.pad(side='both')``.

        zfill
            Pad strings in the Series/Index by prepending '0' character.
            Equivalent to ``Series.str.pad(side='left', fillchar='0')``.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["caribou", "tiger"])

        >>> s.str.pad(width=10)
        0       caribou
        1         tiger
        dtype: str

        >>> s.str.pad(width=10, side='right', fillchar='-')
        0    caribou---
        1    tiger-----
        dtype: str

        >>> s.str.pad(width=10, side='both', fillchar='-')
        0    -caribou--
        1    --tiger---
        dtype: str
        """
        if not isinstance(fillchar, str):
            msg = (
                f"fillchar must be a character, not {type(fillchar).__name__}"
            )
            raise TypeError(msg)

        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")

        if not is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)

        try:
            side_type = plc.strings.side_type.SideType[side.upper()]
        except KeyError:
            raise ValueError(
                "side has to be either one of {'left', 'right', 'both'}"
            )
        return self._return_or_inplace(
            self._column.pad(width, side_type, fillchar)
        )

    def zfill(self, width: int) -> Series | Index:
        """
        Pad strings in the Series/Index by prepending '0' characters.

        Strings in the Series/Index are padded with '0' characters
        on the left of the string to reach a total string length
        width. Strings in the Series/Index with length greater
        or equal to width are unchanged.

        The sign character is preserved if it appears in the first
        position of the string.

        Parameters
        ----------
        width : int
            Minimum length of resulting string;
            strings with length less than width
            be prepended with '0' characters.

        Returns
        -------
        Series/Index of str dtype
            Returns Series or Index with prepended '0' characters.

        See Also
        --------
        rjust
            Fills the left side of strings with an arbitrary character.

        ljust
            Fills the right side of strings with an arbitrary character.

        pad
            Fills the specified sides of strings with an arbitrary character.

        center
            Fills both sides of strings with an arbitrary character.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['-1', '1', '1000',  None])
        >>> s
        0      -1
        1       1
        2    1000
        3    NaN
        dtype: str

        Note that ``None`` is not string, therefore it is converted
        to ``None``. ``1000`` remains unchanged as
        it is longer than width.

        >>> s.str.zfill(3)
        0     -01
        1     001
        2    1000
        3    NaN
        dtype: str
        """
        if not is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)

        return self._return_or_inplace(self._column.zfill(width))

    def center(self, width: int, fillchar: str = " ") -> Series | Index:
        """
        Filling left and right side of strings in the Series/Index with an
        additional character.

        Parameters
        ----------
        width : int
            Minimum width of resulting string;
            additional characters will be filled
            with fillchar.

        fillchar : str, default is ' ' (whitespace)
            Additional character for filling.

        Returns
        -------
        Series/Index of str dtype
            Returns Series or Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['a', 'b', None, 'd'])
        >>> s.str.center(1)
        0       a
        1       b
        2    NaN
        3       d
        dtype: str
        >>> s.str.center(1, fillchar='-')
        0       a
        1       b
        2    NaN
        3       d
        dtype: str
        >>> s.str.center(2, fillchar='-')
        0      a-
        1      b-
        2    NaN
        3      d-
        dtype: str
        >>> s.str.center(5, fillchar='-')
        0    --a--
        1    --b--
        2     NaN
        3    --d--
        dtype: str
        >>> s.str.center(6, fillchar='-')
        0    --a---
        1    --b---
        2      NaN
        3    --d---
        dtype: str
        """
        return self.pad(width, "both", fillchar)

    def ljust(self, width: int, fillchar: str = " ") -> Series | Index:
        """
        Filling right side of strings in the Series/Index with an additional
        character. Equivalent to `str.ljust()
        <https://docs.python.org/3/library/stdtypes.html#str.ljust>`_.

        Parameters
        ----------
        width : int
            Minimum width of resulting string;
            additional characters will be filled
            with ``fillchar``.

        fillchar : str, default ' ' (whitespace)
            Additional character for filling, default is whitespace.

        Returns
        -------
        Series/Index of str dtype
            Returns Series or Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["hello world", "rapids ai"])
        >>> s.str.ljust(10, fillchar="_")
        0    hello world
        1     rapids ai_
        dtype: str
        >>> s = cudf.Series(["a", "",  "ab", "__"])
        >>> s.str.ljust(1, fillchar="-")
        0     a
        1     -
        2    ab
        3    __
        dtype: str
        """
        return self.pad(width, "right", fillchar)

    def rjust(self, width: int, fillchar: str = " ") -> Series | Index:
        """
        Filling left side of strings in the Series/Index with an additional
        character. Equivalent to `str.rjust()
        <https://docs.python.org/3/library/stdtypes.html#str.rjust>`_.

        Parameters
        ----------
        width : int
            Minimum width of resulting string;
            additional characters will be filled
            with fillchar.

        fillchar : str, default ' ' (whitespace)
            Additional character for filling, default is whitespace.

        Returns
        -------
        Series/Index of str dtype
            Returns Series or Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["hello world", "rapids ai"])
        >>> s.str.rjust(20, fillchar="_")
        0    _________hello world
        1    ___________rapids ai
        dtype: str
        >>> s = cudf.Series(["a", "",  "ab", "__"])
        >>> s.str.rjust(1, fillchar="-")
        0     a
        1     -
        2    ab
        3    __
        dtype: str
        """
        return self.pad(width, "left", fillchar)

    def strip(self, to_strip: str | None = None) -> Series | Index:
        r"""
        Remove leading and trailing characters.

        Strip whitespaces (including newlines) or a set of
        specified characters from each string in the Series/Index
        from left and right sides. Equivalent to `str.strip()
        <https://docs.python.org/3/library/stdtypes.html#str.strip>`_.

        Parameters
        ----------
        to_strip : str or None, default None
            Specifying the set of characters to be removed.
            All combinations of this set of characters
            will be stripped. If None then whitespaces are removed.

        Returns
        -------
        Series/Index of str dtype
            Returns Series or Index.

        See Also
        --------
        lstrip
            Remove leading characters in Series/Index.

        rstrip
            Remove trailing characters in Series/Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['1. Ant.  ', '2. Bee!\n', '3. Cat?\t', None])
        >>> s
        0    1. Ant.
        1    2. Bee!\n
        2    3. Cat?\t
        3          NaN
        dtype: str
        >>> s.str.strip()
        0    1. Ant.
        1    2. Bee!
        2    3. Cat?
        3        NaN
        dtype: str
        >>> s.str.strip('123.!? \n\t')
        0     Ant
        1     Bee
        2     Cat
        3    NaN
        dtype: str
        """
        if not (isinstance(to_strip, str) or to_strip is None):
            raise TypeError(
                f"to_strip should be a string or None, but got {type(to_strip).__name__}"
            )
        return self._return_or_inplace(
            self._column.strip(plc.strings.side_type.SideType.BOTH, to_strip)
        )

    def lstrip(self, to_strip: str | None = None) -> Series | Index:
        r"""
        Remove leading and trailing characters.

        Strip whitespaces (including newlines)
        or a set of specified characters from
        each string in the Series/Index from left side.
        Equivalent to `str.lstrip()
        <https://docs.python.org/3/library/stdtypes.html#str.lstrip>`_.

        Parameters
        ----------
        to_strip : str or None, default None
            Specifying the set of characters to be removed.
            All combinations of this set of characters will
            be stripped. If None then whitespaces are removed.

        Returns
        -------
            Series or Index of object

        See Also
        --------
        strip
            Remove leading and trailing characters in Series/Index.

        rstrip
            Remove trailing characters in Series/Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['1. Ant.  ', '2. Bee!\n', '3. Cat?\t', None])
        >>> s.str.lstrip('123.')
        0     Ant.
        1     Bee!\n
        2     Cat?\t
        3        NaN
        dtype: str
        """
        if not (isinstance(to_strip, str) or to_strip is None):
            raise TypeError(
                f"to_strip should be a string or None, but got {type(to_strip).__name__}"
            )
        return self._return_or_inplace(
            self._column.strip(plc.strings.side_type.SideType.LEFT, to_strip)
        )

    def rstrip(self, to_strip: str | None = None) -> Series | Index:
        r"""
        Remove leading and trailing characters.

        Strip whitespaces (including newlines)
        or a set of specified characters from each
        string in the Series/Index from right side.
        Equivalent to `str.rstrip()
        <https://docs.python.org/3/library/stdtypes.html#str.rstrip>`_.

        Parameters
        ----------
        to_strip : str or None, default None
            Specifying the set of characters to
            be removed. All combinations of this
            set of characters will be stripped.
            If None then whitespaces are removed.

        Returns
        -------
        Series/Index of str dtype
            Returns Series or Index.

        See Also
        --------
        strip
            Remove leading and trailing characters in Series/Index.

        lstrip
            Remove leading characters in Series/Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['1. Ant.  ', '2. Bee!\n', '3. Cat?\t', None])
        >>> s
        0    1. Ant.
        1    2. Bee!\n
        2    3. Cat?\t
        3          NaN
        dtype: str
        >>> s.str.rstrip('.!? \n\t')
        0    1. Ant
        1    2. Bee
        2    3. Cat
        3       NaN
        dtype: str
        """
        if not (isinstance(to_strip, str) or to_strip is None):
            raise TypeError(
                f"to_strip should be a string or None, but got {type(to_strip).__name__}"
            )
        return self._return_or_inplace(
            self._column.strip(plc.strings.side_type.SideType.RIGHT, to_strip)
        )

    def wrap(self, width: int, **kwargs) -> Series | Index:
        r"""
        Wrap long strings in the Series/Index to be formatted in
        paragraphs with length less than a given width.

        Parameters
        ----------
        width : int
            Maximum line width.

        Returns
        -------
        Series or Index

        Notes
        -----
        The parameters `expand_tabsbool`, `replace_whitespace`,
        `drop_whitespace`, `break_long_words`, `break_on_hyphens`,
        `expand_tabsbool` are not yet supported and will raise a
        NotImplementedError if they are set to any value.

        This method currently achieves behavior matching R's
        stringr library ``str_wrap`` function, the equivalent
        pandas implementation can be obtained using the
        following parameter setting:

            expand_tabs = False

            replace_whitespace = True

            drop_whitespace = True

            break_long_words = False

            break_on_hyphens = False

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["line to be wrapped", "another line to be wrapped"])
        >>> s.str.wrap(12, expand_tabs=False, break_long_words=False, break_on_hyphens=False)
        0             line to be\nwrapped
        1    another line\nto be\nwrapped
        dtype: str
        """
        if not is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)

        expand_tabs = kwargs.get("expand_tabs", None)
        if expand_tabs is True:
            raise NotImplementedError("`expand_tabs=True` is not supported")
        elif expand_tabs is None:
            if cudf.get_option("mode.pandas_compatible"):
                raise NotImplementedError("not implemented for pandas mode")
            warnings.warn(
                "wrap current implementation defaults to `expand_tabs`=False"
            )

        replace_whitespace = kwargs.get("replace_whitespace", True)
        if not replace_whitespace:
            raise NotImplementedError(
                "`replace_whitespace=False` is not supported"
            )

        drop_whitespace = kwargs.get("drop_whitespace", True)
        if not drop_whitespace:
            raise NotImplementedError(
                "`drop_whitespace=False` is not supported"
            )

        break_long_words = kwargs.get("break_long_words", None)
        if break_long_words is True:
            raise NotImplementedError(
                "`break_long_words=True` is not supported"
            )
        elif break_long_words is None:
            warnings.warn(
                "wrap current implementation defaults to "
                "`break_long_words`=False"
            )

        break_on_hyphens = kwargs.get("break_on_hyphens", None)
        if break_long_words is True:
            raise NotImplementedError(
                "`break_on_hyphens=True` is not supported"
            )
        elif break_on_hyphens is None:
            warnings.warn(
                "wrap current implementation defaults to "
                "`break_on_hyphens`=False"
            )
        return self._return_or_inplace(self._column.wrap(width))

    def count(self, pat: str, flags: int = 0) -> Series | Index:
        r"""
        Count occurrences of pattern in each string of the Series/Index.

        This function is used to count the number of times a particular
        regex pattern is repeated in each of the string elements of the Series.

        Parameters
        ----------
        pat : str or compiled regex
            Valid regular expression.
        flags : int, default 0 (no flags)
            Flags to pass through to the regex engine (e.g. re.MULTILINE)

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['A', 'B', 'Aaba', 'Baca', None, 'CABA', 'cat'])
        >>> s.str.count('a')
        0       0
        1       0
        2       2
        3       2
        4    <NA>
        5       0
        6       1
        dtype: int32

        Escape ``'$'`` to find the literal dollar sign.

        >>> s = cudf.Series(['$', 'B', 'Aab$', '$$ca', 'C$B$', 'cat'])
        >>> s.str.count('\\$')
        0    1
        1    0
        2    1
        3    2
        4    2
        5    0
        dtype: int32

        This is also available on Index.

        >>> index = cudf.Index(['A', 'A', 'Aaba', 'cat'])
        >>> index.str.count('a')
        Index([0, 0, 2, 1], dtype='int32')

        .. pandas-compat::
            :meth:`pandas.Series.str.count`

            -   `flags` parameter currently only supports re.DOTALL
                and re.MULTILINE and re.IGNORECASE.
            -   Some characters need to be escaped when passing
                in pat. e.g. ``'$'`` has a special meaning in regex
                and must be escaped when finding this literal character.
        """
        if isinstance(pat, re.Pattern):
            flags = pat.flags & ~re.U
            pat = pat.pattern
        if not _is_supported_regex_flags(flags):
            raise NotImplementedError(
                "unsupported value for `flags` parameter"
            )
        pat = self._remove_named_capture_groups(pat)
        return self._return_or_inplace(self._column.count_re(pat, flags))

    def _findall(
        self,
        method: Callable[
            [plc.Column, plc.strings.regex_program.RegexProgram], plc.Column
        ],
        pat: str | re.Pattern,
        flags: int = 0,
    ) -> Series | Index:
        if isinstance(pat, re.Pattern):
            flags = pat.flags & ~re.U
            pat = pat.pattern
        if not _is_supported_regex_flags(flags):
            raise NotImplementedError(
                "unsupported value for `flags` parameter"
            )
        pat = self._remove_named_capture_groups(pat)  # type: ignore[arg-type]
        return self._return_or_inplace(
            self._column.findall(method, pat, flags)
        )

    def findall(self, pat: str, flags: int = 0) -> Series | Index:
        """
        Find all occurrences of pattern or regular expression in the
        Series/Index.

        Parameters
        ----------
        pat : str
            Pattern or regular expression.
        flags : int, default 0 (no flags)
            Flags to pass through to the regex engine (e.g. re.MULTILINE)

        Returns
        -------
        DataFrame
            All non-overlapping matches of pattern or
            regular expression in each string of this Series/Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['Lion', 'Monkey', 'Rabbit'])

        The search for the pattern 'Monkey' returns one match:

        >>> s.str.findall('Monkey')
        0          []
        1    [Monkey]
        2          []
        dtype: list

        When the pattern matches more than one string
        in the Series, all matches are returned:

        >>> s.str.findall('on')
        0    [on]
        1    [on]
        2      []
        dtype: list

        Regular expressions are supported too. For instance,
        the search for all the strings ending with
        the word 'on' is shown next:

        >>> s.str.findall('on$')
        0    [on]
        1      []
        2      []
        dtype: list

        If the pattern is found more than once in the same
        string, then multiple strings are returned:

        >>> s.str.findall('b')
        0        []
        1        []
        2    [b, b]
        dtype: list

        .. pandas-compat::
            :meth:`pandas.Series.str.findall`

            The `flags` parameter currently only supports re.DOTALL,
            re.MULTILINE and re.IGNORECASE.
        """
        return self._findall(plc.strings.findall.findall, pat, flags)

    def find_re(self, pat: str, flags: int = 0) -> Series | Index:
        """
        Find first occurrence of pattern or regular expression in the
        Series/Index.

        Parameters
        ----------
        pat : str
            Pattern or regular expression.
        flags : int, default 0 (no flags)
            Flags to pass through to the regex engine (e.g. re.MULTILINE)

        Returns
        -------
        Series
            A Series of position values where the pattern first matches
            each string.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['Lion', 'Monkey', 'Rabbit', 'Cat'])
        >>> s.str.find_re('[ti]')
        0    1
        1   -1
        2    4
        3    2
        dtype: int32
        """
        return self._findall(plc.strings.findall.find_re, pat, flags)

    def find_multiple(self, patterns: Series | Index) -> Series:
        """
        Find all first occurrences of patterns in the Series/Index.

        Parameters
        ----------
        patterns : array-like, Sequence or Series
            Patterns to search for in the given Series/Index.

        Returns
        -------
        Series
            A Series with a list of indices of each pattern's first occurrence.
            If a pattern is not found, -1 is returned for that index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["strings", "to", "search", "in"])
        >>> s
        0    strings
        1         to
        2     search
        3         in
        dtype: str
        >>> t = cudf.Series(["a", "string", "g", "inn", "o", "r", "sea"])
        >>> t
        0         a
        1    string
        2         g
        3       inn
        4         o
        5         r
        6       sea
        dtype: str
        >>> s.str.find_multiple(t)
        0       [-1, 0, 5, -1, -1, 2, -1]
        1     [-1, -1, -1, -1, 1, -1, -1]
        2       [2, -1, -1, -1, -1, 3, 0]
        3    [-1, -1, -1, -1, -1, -1, -1]
        dtype: list
        """
        if can_convert_to_column(patterns):
            patterns_column = as_column(patterns)
        else:
            raise TypeError(
                "patterns should be an array-like or a Series object, "
                f"found {type(patterns)}"
            )

        if not is_string_dtype(patterns_column.dtype):
            raise TypeError(
                "patterns can only be of 'string' dtype, "
                f"got: {patterns_column.dtype}"
            )

        result = self._column.find_multiple(patterns_column)  # type: ignore[arg-type]

        return cudf.Series._from_column(
            result,
            name=self._parent.name,
            index=(
                self._parent.index
                if isinstance(self._parent, cudf.Series)
                else self._parent
            ),
        )

    def isempty(self) -> Series | Index:
        """
        Check whether each string is an empty string.

        Returns
        -------
        Series or Index of bool
            Series or Index of boolean values with the same length as
            the original Series/Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["1", "abc", "", " ", None])
        >>> s.str.isempty()
        0    False
        1    False
        2     True
        3    False
        4    False
        dtype: bool
        """
        return self._return_or_inplace(
            # mypy can't deduce that the return value of
            # StringColumn.__eq__ is ColumnBase because the binops are
            # dynamically added by a mixin class
            cast("ColumnBase", self._column == "").fillna(False)
        )

    def isspace(self) -> Series | Index:
        r"""
        Check whether all characters in each string are whitespace.

        This is equivalent to running the Python string method
        `str.isspace()
        <https://docs.python.org/3/library/stdtypes.html#str.isspace>`_
        for each element of the Series/Index.
        If a string has zero characters, False is returned
        for that check.

        Returns
        -------
        Series or Index of bool
            Series or Index of boolean values with the same length as
            the original Series/Index.

        See Also
        --------
        isalnum
            Check whether all characters are alphanumeric.

        isalpha
            Check whether all characters are alphabetic.

        isdecimal
            Check whether all characters are decimal.

        isdigit
            Check whether all characters are digits.

        isinteger
            Check whether all characters are integer.

        isnumeric
            Check whether all characters are numeric.

        isfloat
            Check whether all characters are float.

        islower
            Check whether all characters are lowercase.

        isupper
            Check whether all characters are uppercase.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([' ', '\t\r\n ', ''])
        >>> s.str.isspace()
        0     True
        1     True
        2    False
        dtype: bool
        """
        return self._return_or_inplace(
            self._column.all_characters_of_type(
                plc.strings.char_types.StringCharacterTypes.SPACE
            )
        )

    def _starts_ends_with(
        self,
        method: Callable[[plc.Column, plc.Column | plc.Scalar], plc.Column],
        pat: str | tuple[str, ...],
    ) -> Series | Index:
        return self._return_or_inplace(
            self._column.starts_ends_with(method, pat)
        )

    def endswith(self, pat: str | tuple[str, ...]) -> Series | Index:
        """
        Test if the end of each string element matches a pattern.

        Parameters
        ----------
        pat : str or tuple[str, ...]
            String pattern or tuple of patterns. Regular expressions are not accepted.

        Returns
        -------
        Series or Index of bool
            A Series of booleans indicating whether the given
            pattern matches the end of each string element.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['bat', 'bear', 'caT', None])
        >>> s
        0     bat
        1    bear
        2     caT
        3    NaN
        dtype: str
        >>> s.str.endswith('t')
        0     True
        1    False
        2    False
        3    False
        dtype: bool

        .. pandas-compat::
            :meth:`pandas.Series.str.endswith`

            `na` parameter is not yet supported, as cudf uses
            native strings instead of Python objects.
        """
        return self._starts_ends_with(plc.strings.find.ends_with, pat)

    def startswith(self, pat: str | tuple[str, ...]) -> Series | Index:
        """
        Test if the start of each string element matches a pattern.

        Equivalent to `str.startswith()
        <https://docs.python.org/3/library/stdtypes.html#str.startswith>`_.

        Parameters
        ----------
        pat : str or tuple[str, ...]
            String pattern or tuple of patterns. Regular expressions are not accepted.

        Returns
        -------
        Series or Index of bool
            A Series of booleans indicating whether the given
            pattern matches the start of each string element.

        See Also
        --------
        endswith
            Same as startswith, but tests the end of string.

        contains
            Tests if string element contains a pattern.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['bat', 'Bear', 'cat', None])
        >>> s
        0     bat
        1    Bear
        2     cat
        3    NaN
        dtype: str
        >>> s.str.startswith('b')
        0     True
        1    False
        2    False
        3    False
        dtype: bool
        """
        return self._starts_ends_with(plc.strings.find.starts_with, pat)

    def removesuffix(self, suffix: str) -> Series | Index:
        """
        Remove a suffix from an object series.

        If the suffix is not present, the original string will be returned.

        Parameters
        ----------
        suffix : str
            Remove the suffix of the string.

        Returns
        -------
        Series/Index: object
            The Series or Index with given suffix removed.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["foo_str", "bar_str", "no_suffix"])
        >>> s
        0    foo_str
        1    bar_str
        2    no_suffix
        dtype: str
        >>> s.str.removesuffix("_str")
        0    foo
        1    bar
        2    no_suffix
        dtype: str
        """
        if suffix is None or len(suffix) == 0:
            return self._return_or_inplace(self._column)
        ends_column = self.endswith(suffix)._column
        removed_column = self.slice(0, -len(suffix), None)._column

        result = removed_column.copy_if_else(self._column, ends_column)
        return self._return_or_inplace(result)

    def removeprefix(self, prefix: str) -> Series | Index:
        """
        Remove a prefix from an object series.

        If the prefix is not present, the original string will be returned.

        Parameters
        ----------
        prefix : str
            Remove the prefix of the string.

        Returns
        -------
        Series/Index: object
            The Series or Index with given prefix removed.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["str_foo", "str_bar", "no_prefix"])
        >>> s
        0    str_foo
        1    str_bar
        2    no_prefix
        dtype: str
        >>> s.str.removeprefix("str_")
        0    foo
        1    bar
        2    no_prefix
        dtype: str
        """
        if prefix is None or len(prefix) == 0:
            return self._return_or_inplace(self._column)
        starts_column = self.startswith(prefix)._column
        removed_column = self.slice(len(prefix), None, None)._column
        result = removed_column.copy_if_else(self._column, starts_column)
        return self._return_or_inplace(result)

    def _find(
        self,
        method: Callable[[plc.Column, plc.Scalar, int, int], plc.Column],
        sub: str,
        start: int = 0,
        end: int | None = None,
    ) -> Series | Index:
        if not isinstance(sub, str):
            raise TypeError(
                f"expected a string object, not {type(sub).__name__}"
            )

        if end is None:
            end = -1

        return self._return_or_inplace(
            self._column.find(method, sub, start, end)
        )

    def find(
        self, sub: str, start: int = 0, end: int | None = None
    ) -> Series | Index:
        """
        Return lowest indexes in each strings in the Series/Index
        where the substring is fully contained between ``[start:end]``.
        Return -1 on failure.

        Parameters
        ----------
        sub : str
            Substring being searched.

        start : int
            Left edge index.

        end : int
            Right edge index.

        Returns
        -------
        Series or Index of int

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['abc', 'a','b' ,'ddb'])
        >>> s.str.find('b')
        0    1
        1   -1
        2    0
        3    2
        dtype: int64

        Parameters such as `start` and `end` can also be used.

        >>> s.str.find('b', start=1, end=5)
        0    1
        1   -1
        2   -1
        3    2
        dtype: int64
        """
        return self._find(plc.strings.find.find, sub, start, end)

    def rfind(
        self, sub: str, start: int = 0, end: int | None = None
    ) -> Series | Index:
        """
        Return highest indexes in each strings in the Series/Index
        where the substring is fully contained between ``[start:end]``.
        Return -1 on failure. Equivalent to standard `str.rfind()
        <https://docs.python.org/3/library/stdtypes.html#str.rfind>`_.

        Parameters
        ----------
        sub : str
            Substring being searched.

        start : int
            Left edge index.

        end : int
            Right edge index.

        Returns
        -------
        Series or Index of int

        See Also
        --------
        find
            Return lowest indexes in each strings.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["abc", "hello world", "rapids ai"])
        >>> s.str.rfind('a')
        0    0
        1   -1
        2    7
        dtype: int64

        Using `start` and `end` parameters.

        >>> s.str.rfind('a', start=2, end=5)
        0   -1
        1   -1
        2   -1
        dtype: int64
        """
        return self._find(plc.strings.find.rfind, sub, start, end)

    def index(
        self, sub: str, start: int = 0, end: int | None = None
    ) -> Series | Index:
        """
        Return lowest indexes in each strings where the substring
        is fully contained between ``[start:end]``. This is the same
        as str.find except instead of returning -1, it raises a ValueError
        when the substring is not found.

        Parameters
        ----------
        sub : str
            Substring being searched.

        start : int
            Left edge index.

        end : int
            Right edge index.

        Returns
        -------
        Series or Index of object

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['abc', 'a','b' ,'ddb'])
        >>> s.str.index('b')
        Traceback (most recent call last):
        ...
        ValueError: substring not found

        Parameters such as `start` and `end` can also be used.

        >>> s = cudf.Series(['abc', 'abb','ab' ,'ddb'])
        >>> s.str.index('b', start=1, end=5)
        0    1
        1    1
        2    1
        3    2
        dtype: int64
        """
        if not isinstance(sub, str):
            raise TypeError(
                f"expected a string object, not {type(sub).__name__}"
            )

        if end is None:
            end = -1

        result_col = self.find(sub, start, end)._column

        result = self._return_or_inplace(result_col)

        if (result == -1).any():
            raise ValueError("substring not found")
        else:
            return result.astype(np.dtype(np.int64))

    def rindex(
        self, sub: str, start: int = 0, end: int | None = None
    ) -> Series | Index:
        """
        Return highest indexes in each strings where the substring
        is fully contained between ``[start:end]``. This is the same
        as ``str.rfind`` except instead of returning -1, it raises a
        ``ValueError`` when the substring is not found.

        Parameters
        ----------
        sub : str
            Substring being searched.

        start : int
            Left edge index.

        end : int
            Right edge index.

        Returns
        -------
        Series or Index of object

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['abc', 'a','b' ,'ddb'])
        >>> s.str.rindex('b')
        Traceback (most recent call last):
        ...
        ValueError: substring not found

        Parameters such as `start` and `end` can also be used.

        >>> s = cudf.Series(['abc', 'abb','ab' ,'ddb'])
        >>> s.str.rindex('b', start=1, end=5)
        0    1
        1    2
        2    1
        3    2
        dtype: int64
        """
        if not isinstance(sub, str):
            raise TypeError(
                f"expected a string object, not {type(sub).__name__}"
            )

        if end is None:
            end = -1

        result_col = self.rfind(sub, start, end)._column

        result = self._return_or_inplace(result_col)

        if (result == -1).any():
            raise ValueError("substring not found")
        else:
            return result.astype(np.dtype(np.int64))

    def match(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na=no_default,
    ) -> Series | Index:
        """
        Determine if each string matches a regular expression.

        Parameters
        ----------
        pat : str or compiled regex
            Character sequence or regular expression.
        flags : int, default 0 (no flags)
            Flags to pass through to the regex engine (e.g. re.MULTILINE)
        na : scalar, optional
            Fill value for missing values. The default depends on dtype of the
            array. For the ``"str"`` dtype, ``False`` is used. For object
            dtype, ``numpy.nan`` is used. For the nullable ``StringDtype``,
            ``pandas.NA`` is used.

        Returns
        -------
        Series or Index of boolean values.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["rapids", "ai", "cudf"])

        Checking for strings starting with `a`.

        >>> s.str.match('a')
        0    False
        1     True
        2    False
        dtype: bool

        Checking for strings starting with any of `a` or `c`.

        >>> s.str.match('[ac]')
        0    False
        1     True
        2     True
        dtype: bool

        .. pandas-compat::
            :meth:`pandas.Series.str.match`

            Parameter `case` is currently not supported.
            The `flags` parameter currently only supports re.DOTALL,
            re.MULTILINE and re.IGNORECASE.
        """
        if case is not True:
            raise NotImplementedError("`case` parameter is not yet supported")
        if isinstance(pat, re.Pattern):
            if flags:
                raise ValueError(
                    "cannot process flags argument with a compiled pattern"
                )
            flags = pat.flags & ~re.U
            pat = pat.pattern
        if not _is_supported_regex_flags(flags):
            raise NotImplementedError(
                "unsupported value for `flags` parameter"
            )
        pat = self._remove_named_capture_groups(pat)
        result = self._column.matches_re(pat, flags)
        if na is not no_default:
            result = result.fillna(na)
        elif self._column._PANDAS_NA_VALUE in {np.nan, None}:
            result = result.fillna(False)
        return self._return_or_inplace(result)

    def url_decode(self) -> Series | Index:
        """
        Returns a URL-decoded format of each string.
        No format checking is performed. All characters
        are expected to be encoded as UTF-8 hex values.

        Returns
        -------
        Series or Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['A%2FB-C%2FD', 'e%20f.g', '4-5%2C6'])
        >>> s.str.url_decode()
        0    A/B-C/D
        1      e f.g
        2      4-5,6
        dtype: str
        >>> data = ["https%3A%2F%2Frapids.ai%2Fstart.html",
        ...     "https%3A%2F%2Fmedium.com%2Frapids-ai"]
        >>> s = cudf.Series(data)
        >>> s.str.url_decode()
        0    https://rapids.ai/start.html
        1    https://medium.com/rapids-ai
        dtype: str
        """
        return self._return_or_inplace(self._column.url_decode())

    def url_encode(self) -> Series | Index:
        """
        Returns a URL-encoded format of each string.
        No format checking is performed.
        All characters are encoded except for ASCII letters,
        digits, and these characters: ``'.','_','-','~'``.
        Encoding converts to hex using UTF-8 encoded bytes.

        Returns
        -------
        Series or Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['A/B-C/D', 'e f.g', '4-5,6'])
        >>> s.str.url_encode()
        0    A%2FB-C%2FD
        1        e%20f.g
        2        4-5%2C6
        dtype: str
        >>> data = ["https://rapids.ai/start.html",
        ...     "https://medium.com/rapids-ai"]
        >>> s = cudf.Series(data)
        >>> s.str.url_encode()
        0    https%3A%2F%2Frapids.ai%2Fstart.html
        1    https%3A%2F%2Fmedium.com%2Frapids-ai
        dtype: str
        """
        return self._return_or_inplace(self._column.url_encode())

    def code_points(self) -> Series | Index:
        """
        Returns an array by filling it with the UTF-8 code point
        values for each character of each string.
        This function uses the ``len()`` method to determine
        the size of each sub-array of integers.

        Returns
        -------
        Series or Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["a","xyz", "éee"])
        >>> s.str.code_points()
        0       97
        1      120
        2      121
        3      122
        4    50089
        5      101
        6      101
        dtype: int32
        >>> s = cudf.Series(["abc"])
        >>> s.str.code_points()
        0    97
        1    98
        2    99
        dtype: int32
        """
        return self._return_or_inplace(
            self._column.code_points(), retain_index=False
        )

    def translate(self, table: dict) -> Series | Index:
        """
        Map all characters in the string through the given
        mapping table.

        Equivalent to standard `str.translate()
        <https://docs.python.org/3/library/stdtypes.html#str.translate>`_.

        Parameters
        ----------
        table : dict
            Table is a mapping of Unicode ordinals to Unicode
            ordinals, strings, or None.
            Unmapped characters are left untouched.
            `str.maketrans()
            <https://docs.python.org/3/library/stdtypes.html#str.maketrans>`_
            is a helper function for making translation tables.

        Returns
        -------
        Series or Index.

        Examples
        --------
        >>> import cudf
        >>> data = ['lower', 'CAPITALS', 'this is a sentence','SwApCaSe']
        >>> s = cudf.Series(data)
        >>> s.str.translate({'a': "1"})
        0                 lower
        1              CAPITALS
        2    this is 1 sentence
        3              SwApC1Se
        dtype: str
        >>> s.str.translate({'a': "1", "e":"#"})
        0                 low#r
        1              CAPITALS
        2    this is 1 s#nt#nc#
        3              SwApC1S#
        dtype: str
        """
        return self._return_or_inplace(self._column.translate(table))

    def filter_characters(
        self, table: dict, keep: bool = True, repl: str | None = None
    ) -> Series | Index:
        """
        Remove characters from each string using the character ranges
        in the given mapping table.

        Parameters
        ----------
        table : dict
            This table is a range of Unicode ordinals to filter.
            The minimum value is the key and the maximum value is the value.
            You can use `str.maketrans()
            <https://docs.python.org/3/library/stdtypes.html#str.maketrans>`_
            as a helper function for making the filter table.
            Overlapping ranges will cause undefined results.
            Range values are inclusive.
        keep : boolean
            If False, the character ranges in the ``table`` are removed.
            If True, the character ranges not in the ``table`` are removed.
            Default is True.
        repl : str
            Optional replacement string to use in place of removed characters.

        Returns
        -------
        Series or Index.

        Examples
        --------
        >>> import cudf
        >>> data = ['aeiou', 'AEIOU', '0123456789']
        >>> s = cudf.Series(data)
        >>> s.str.filter_characters({'a':'l', 'M':'Z', '4':'6'})
        0    aei
        1     OU
        2    456
        dtype: str
        >>> s.str.filter_characters({'a':'l', 'M':'Z', '4':'6'}, False, "_")
        0         ___ou
        1         AEI__
        2    0123___789
        dtype: str
        """
        if repl is None:
            repl = ""
        try:
            plc_repl = plc.Scalar.from_py(
                repl, dtype=plc.DataType(plc.TypeId.STRING)
            )
        except (NotImplementedError, TypeError) as err:
            raise TypeError(
                f"repl should be a string, but got {type(repl).__name__}"
            ) from err
        return self._return_or_inplace(
            self._column.filter_characters(table, keep, plc_repl)
        )

    def normalize_spaces(self) -> Series | Index:
        r"""
        Remove extra whitespace between tokens and trim whitespace
        from the beginning and the end of each string.

        Returns
        -------
        Series or Index of object.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series(["hello \t world"," test string  "])
        >>> ser.str.normalize_spaces()
        0    hello world
        1    test string
        dtype: str
        """
        return self._return_or_inplace(self._column.normalize_spaces())

    def tokenize(self, delimiter: str = " ") -> Series | Index:
        """
        Each string is split into tokens using the provided delimiter(s).
        The sequence returned contains the tokens in the order
        they were found.

        Parameters
        ----------
        delimiter : str or list of strs, Default is whitespace.
            The string used to locate the split points of each string.

        Returns
        -------
        Series or Index of object.

        Examples
        --------
        >>> import cudf
        >>> data = ["hello world", "goodbye world", "hello goodbye"]
        >>> ser = cudf.Series(data)
        >>> ser.str.tokenize()
        0      hello
        0      world
        1    goodbye
        1      world
        2      hello
        2    goodbye
        dtype: str
        """
        delim = _massage_string_arg(delimiter, "delimiter", allow_col=True)

        if isinstance(delim, ColumnBase):
            result = self._return_or_inplace(
                self._column.tokenize_column(delim),
                retain_index=False,
            )
        elif isinstance(delim, plc.Scalar):
            result = self._return_or_inplace(
                self._column.tokenize_scalar(delim),
                retain_index=False,
            )
        else:
            raise TypeError(
                f"Expected a Scalar or Column\
                for delimiters, but got {type(delimiter)}"
            )
        if isinstance(self._parent, cudf.Series):
            result.index = self._parent.index.repeat(  # type: ignore[union-attr]
                self.token_count(delimiter=delimiter)
            )
        return result

    def detokenize(
        self, indices: Series, separator: str = " "
    ) -> Series | Index:
        """
        Combines tokens into strings by concatenating them in the order
        in which they appear in the ``indices`` column. The ``separator`` is
        concatenated between each token.

        Parameters
        ----------
        indices : Series
            Each value identifies the output row for the corresponding token.
        separator : str
            The string concatenated between each token in an output row.
            Default is space.

        Returns
        -------
        Series or Index of object.

        Examples
        --------
        >>> import cudf
        >>> strs = cudf.Series(["hello", "world", "one", "two", "three"])
        >>> indices = cudf.Series([0, 0, 1, 1, 2])
        >>> strs.str.detokenize(indices)
        0    hello world
        1        one two
        2          three
        dtype: str
        """
        sep = _massage_string_arg(separator, "separator")
        return self._return_or_inplace(
            self._column.detokenize(indices._column, sep),  # type: ignore[arg-type]
            retain_index=False,
        )

    def character_tokenize(self) -> Series | Index:
        """
        Each string is split into individual characters.
        The sequence returned contains each character as an individual string.

        Returns
        -------
        Series or Index of object.

        Examples
        --------
        >>> import cudf
        >>> data = ["hello world", "goodbye, thank you."]
        >>> ser = cudf.Series(data)
        >>> ser.str.character_tokenize()
        0    h
        0    e
        0    l
        0    l
        0    o
        0
        0    w
        0    o
        0    r
        0    l
        0    d
        1    g
        1    o
        1    o
        1    d
        1    b
        1    y
        1    e
        1    ,
        1
        1    t
        1    h
        1    a
        1    n
        1    k
        1
        1    y
        1    o
        1    u
        1    .
        dtype: str
        """
        result_col = ColumnBase.create(
            self._column.character_tokenize().plc_column.children()[1],
            self._column.dtype,
        )
        if isinstance(self._parent, cudf.Series):
            lengths = self.len().fillna(0)
            index = self._parent.index.repeat(lengths)
            return type(self._parent)._from_column(
                result_col, name=self._parent.name, index=index
            )
        else:
            return self._return_or_inplace(result_col)

    def token_count(self, delimiter: str = " ") -> Series | Index:
        """
        Each string is split into tokens using the provided delimiter.
        The returned integer sequence is the number of tokens in each string.

        Parameters
        ----------
        delimiter : str or list of strs, Default is whitespace.
            The characters or strings used to locate the
            split points of each string.

        Returns
        -------
        Series or Index.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series(["hello world","goodbye",""])
        >>> ser.str.token_count()
        0    2
        1    1
        2    0
        dtype: int32
        """
        delim = _massage_string_arg(delimiter, "delimiter", allow_col=True)
        if isinstance(delim, ColumnBase):
            return self._return_or_inplace(
                self._column.count_tokens_column(delim)
            )

        elif isinstance(delim, plc.Scalar):
            return self._return_or_inplace(
                self._column.count_tokens_scalar(delim)
            )
        else:
            raise TypeError(
                f"Expected a Scalar or Column\
                for delimiters, but got {type(delimiter)}"
            )

    def ngrams(self, n: int = 2, separator: str = "_") -> Series | Index:
        """
        Generate the n-grams from a set of tokens, each record
        in series is treated a token.

        You can generate tokens from a Series instance using
        the ``Series.str.tokenize()`` function.

        Parameters
        ----------
        n : int
            The degree of the n-gram (number of consecutive tokens).
            Default of 2 for bigrams.
        separator : str
            The separator to use between within an n-gram.
            Default is '_'.

        Examples
        --------
        >>> import cudf
        >>> str_series = cudf.Series(['this is my', 'favorite book'])
        >>> str_series.str.ngrams(2, "_")
        0    this is my_favorite book
        dtype: str
        >>> str_series = cudf.Series(['abc','def','xyz','hhh'])
        >>> str_series.str.ngrams(2, "_")
        0    abc_def
        1    def_xyz
        2    xyz_hhh
        dtype: str
        """
        sep = _massage_string_arg(separator, "separator")
        return self._return_or_inplace(
            self._column.generate_ngrams(n, sep),  # type: ignore[arg-type]
            retain_index=False,
        )

    def character_ngrams(
        self, n: int = 2, as_list: bool = False
    ) -> Series | Index:
        """
        Generate the n-grams from characters in a column of strings.

        Parameters
        ----------
        n : int
            The degree of the n-gram (number of consecutive characters).
            Default of 2 for bigrams.
        as_list : bool
            Set to True to return ngrams in a list column where each
            list element is the ngrams for each string.

        Examples
        --------
        >>> import cudf
        >>> str_series = cudf.Series(['abcd','efgh','xyz'])
        >>> str_series.str.character_ngrams(2)
        0    ab
        0    bc
        0    cd
        1    ef
        1    fg
        1    gh
        2    xy
        2    yz
        dtype: str
        >>> str_series.str.character_ngrams(3)
        0    abc
        0    bcd
        1    efg
        1    fgh
        2    xyz
        dtype: str
        >>> str_series.str.character_ngrams(3,True)
        0    [abc, bcd]
        1    [efg, fgh]
        2         [xyz]
        dtype: list
        """
        result = self._return_or_inplace(
            self._column.generate_character_ngrams(n),
            retain_index=True,
        )
        if isinstance(result, cudf.Series) and not as_list:
            # before exploding, removes those lists which have 0 length
            result = result[result.list.len() > 0]
            return result.explode()
        return result

    def hash_character_ngrams(
        self, n: int = 5, as_list: bool = False, seed: int | np.uint32 = 0
    ) -> Series | Index:
        """
        Generate hashes of n-grams from characters in a column of strings.
        The MurmurHash32 algorithm is used to produce the hash results.

        Parameters
        ----------
        n : int
            The degree of the n-gram (number of consecutive characters).
            Default is 5.
        as_list : bool
            Set to True to return the hashes in a list column where each
            list element is the hashes for each string.
        seed: uint32
            The seed value for the hash algorithm.

        Examples
        --------
        >>> import cudf
        >>> str_series = cudf.Series(['abcdefg','stuvwxyz'])
        >>> str_series.str.hash_character_ngrams(n=5, as_list=True)
        0               [3902511862, 570445242, 4202475763]
        1    [556054766, 3166857694, 3760633458, 192452857]
        dtype: list
        >>> str_series.str.hash_character_ngrams(5)
        0    3902511862
        0     570445242
        0    4202475763
        1     556054766
        1    3166857694
        1    3760633458
        1     192452857
        dtype: uint32
        """

        result = self._return_or_inplace(
            self._column.hash_character_ngrams(n, seed),
            retain_index=True,
        )
        if isinstance(result, cudf.Series) and not as_list:
            return result.explode()
        return result

    def ngrams_tokenize(
        self, n: int = 2, delimiter: str = " ", separator: str = "_"
    ) -> Series | Index:
        """
        Generate the n-grams using tokens from each string.
        This will tokenize each string and then generate ngrams for each
        string.

        Parameters
        ----------
        n : int, Default 2.
            The degree of the n-gram (number of consecutive tokens).
        delimiter : str, Default is white-space.
            The character used to locate the split points of each string.
        sep : str, Default is '_'.
            The separator to use between tokens within an n-gram.

        Returns
        -------
        Series or Index of object.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series(['this is the', 'best book'])
        >>> ser.str.ngrams_tokenize(n=2, separator='_')
        0      this_is
        1       is_the
        2    best_book
        dtype: str
        """
        delim = _massage_string_arg(delimiter, "delimiter")
        sep = _massage_string_arg(separator, "separator")
        return self._return_or_inplace(
            self._column.ngrams_tokenize(n, delim, sep),  # type: ignore[arg-type]
            retain_index=False,
        )

    def replace_tokens(
        self, targets, replacements, delimiter: str | None = None
    ) -> Series | Index:
        """
        The targets tokens are searched for within each string in the series
        and replaced with the corresponding replacements if found.
        Tokens are identified by the delimiter character provided.

        Parameters
        ----------
        targets : array-like, Sequence or Series
            The tokens to search for inside each string.

        replacements : array-like, Sequence, Series or str
            The strings to replace for each found target token found.
            Alternately, this can be a single str instance and would be
            used as replacement for each string found.

        delimiter : str
            The character used to locate the tokens of each string.
            Default is whitespace.

        Returns
        -------
        Series or Index of object.

        Examples
        --------
        >>> import cudf
        >>> sr = cudf.Series(["this is me", "theme music", ""])
        >>> targets = cudf.Series(["is", "me"])
        >>> sr.str.replace_tokens(targets=targets, replacements="_")
        0       this _ _
        1    theme music
        2
        dtype: str
        >>> sr = cudf.Series(["this;is;me", "theme;music", ""])
        >>> sr.str.replace_tokens(targets=targets, replacements=":")
        0     this;is;me
        1    theme;music
        2
        dtype: str
        """
        if can_convert_to_column(targets):
            targets_column = as_column(targets)
        else:
            raise TypeError(
                f"targets should be an array-like or a Series object, "
                f"found {type(targets)}"
            )

        if is_scalar(replacements):
            replacements_column = as_column([replacements])
        elif can_convert_to_column(replacements):
            replacements_column = as_column(replacements)
            if len(targets_column) != len(replacements_column):
                raise ValueError(
                    "targets and replacements should be same size"
                    " sequences unless replacements is a string."
                )
        else:
            raise TypeError(
                f"replacements should be an str, array-like or Series object, "
                f"found {type(replacements)}"
            )

        if delimiter is None:
            delimiter = ""
        elif not is_scalar(delimiter):
            raise TypeError(
                f"Type of delimiter should be a string,"
                f" found {type(delimiter)}"
            )

        return self._return_or_inplace(
            self._column.replace_tokens(
                targets_column,  # type: ignore[arg-type]
                replacements_column,  # type: ignore[arg-type]
                plc.Scalar.from_py(
                    delimiter, dtype=plc.DataType(plc.TypeId.STRING)
                ),
            ),
        )

    def filter_tokens(
        self,
        min_token_length: int,
        replacement: str | None = None,
        delimiter: str | None = None,
    ) -> Series | Index:
        """
        Remove tokens from within each string in the series that are
        smaller than min_token_length and optionally replace them
        with the replacement string.
        Tokens are identified by the delimiter character provided.

        Parameters
        ----------
        min_token_length: int
            Minimum number of characters for a token to be retained
            in the output string.

        replacement : str
            String used in place of removed tokens.

        delimiter : str
            The character(s) used to locate the tokens of each string.
            Default is whitespace.

        Returns
        -------
        Series or Index of object.

        Examples
        --------
        >>> import cudf
        >>> sr = cudf.Series(["this is me", "theme music", ""])
        >>> sr.str.filter_tokens(3, replacement="_")
        0       this _ _
        1    theme music
        2
        dtype: str
        >>> sr = cudf.Series(["this;is;me", "theme;music", ""])
        >>> sr.str.filter_tokens(5,None,";")
        0             ;;
        1    theme;music
        2
        dtype: str
        """

        if replacement is None:
            replacement = ""
        elif not is_scalar(replacement):
            raise TypeError(
                f"Type of replacement should be a string,"
                f" found {type(replacement)}"
            )

        if delimiter is None:
            delimiter = ""
        elif not is_scalar(delimiter):
            raise TypeError(
                f"Type of delimiter should be a string,"
                f" found {type(delimiter)}"
            )

        return self._return_or_inplace(
            self._column.filter_tokens(
                min_token_length,
                plc.Scalar.from_py(
                    replacement, dtype=plc.DataType(plc.TypeId.STRING)
                ),
                plc.Scalar.from_py(
                    delimiter, dtype=plc.DataType(plc.TypeId.STRING)
                ),
            ),
        )

    def porter_stemmer_measure(self) -> Series | Index:
        """
        Compute the Porter Stemmer measure for each string.
        The Porter Stemmer algorithm is described `here
        <https://tartarus.org/martin/PorterStemmer/def.txt>`_.

        Returns
        -------
        Series or Index of object.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series(["hello", "super"])
        >>> ser.str.porter_stemmer_measure()
        0    1
        1    2
        dtype: int32
        """
        return self._return_or_inplace(self._column.porter_stemmer_measure())

    def is_consonant(self, position) -> Series | Index:
        """
        Return true for strings where the character at ``position`` is a
        consonant. The ``position`` parameter may also be a list of integers
        to check different characters per string.
        If the ``position`` is larger than the string length, False is
        returned for that string.

        Parameters
        ----------
        position: int or list-like
           The character position to check within each string.

        Returns
        -------
        Series or Index of bool dtype.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series(["toy", "trouble"])
        >>> ser.str.is_consonant(1)
        0    False
        1     True
        dtype: bool
        >>> positions = cudf.Series([2, 3])
        >>> ser.str.is_consonant(positions)
        0     True
        1    False
        dtype: bool
        """
        if can_convert_to_column(position):
            position = as_column(position)
        return self._return_or_inplace(self._column.is_letter(False, position))

    def is_vowel(self, position) -> Series | Index:
        """
        Return true for strings where the character at ``position`` is a
        vowel -- not a consonant. The ``position`` parameter may also be
        a list of integers to check different characters per string.
        If the ``position`` is larger than the string length, False is
        returned for that string.

        Parameters
        ----------
        position: int or list-like
           The character position to check within each string.

        Returns
        -------
        Series or Index of bool dtype.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series(["toy", "trouble"])
        >>> ser.str.is_vowel(1)
        0     True
        1    False
        dtype: bool
        >>> positions = cudf.Series([2, 3])
        >>> ser.str.is_vowel(positions)
        0    False
        1     True
        dtype: bool
        """
        if can_convert_to_column(position):
            position = as_column(position)
        return self._return_or_inplace(self._column.is_letter(True, position))

    def build_suffix_array(self, min_width: int) -> Series | Index:
        """
        Builds a suffix array for the input strings column.
        A suffix array is the indices of the sorted set of substrings
        of the input column as: [ input[0:], input[1:], ... input[bytes-1:] ]
        where bytes is the total number of bytes in input.
        The returned array represent the sorted strings such that
        result[i] = input[suffix_array[i]:]

        For details, see :cpp:func:`build_suffix_array`

        Parameters
        ----------
        min_width : int
            The minimum number of bytes to determine duplicates

        Returns
        -------
        Column
            New column of suffix array
        """
        return self._return_or_inplace(
            self._column.build_suffix_array(min_width),
            inplace=False,
            expand=False,
            retain_index=False,
        )

    def resolve_duplicates(self, sa, min_width: int) -> Series | Index:
        """
        Returns duplicate strings found in the input column
        with min_width minimum number of bytes.
        The indices are expected to be the suffix array previously created
        for input. Otherwise, the results are undefined.

        For details, see :cpp:func:`resolve_duplicates`

        Parameters
        ----------
        sa : Column
            Suffix array from build_suffix_array
        min_width : int
            Minimum number of bytes that must match

        Returns
        -------
        Column
            New column of duplicates
        """
        sa_column = sa._column
        return self._return_or_inplace(
            self._column.resolve_duplicates(sa_column, min_width),
            inplace=False,
            expand=False,
            retain_index=False,
        )

    def resolve_duplicates_pair(
        self, sa1, input2, sa2, min_width: int
    ) -> Series | Index:
        """
        Returns duplicate strings in input1 found in input2
        with min_width minimum number of bytes.
        The indices are expected to be the suffix array previously
        created for the inputs. Otherwise, the results are undefined.

        For details, see :cpp:func:`resolve_duplicates_pair`

        Parameters
        ----------
        sa1 : Column
            Suffix array from build_suffix_array for this column
        input2 : Column
            2nd strings column of text
        sa2 : Column
            Suffix array from build_suffix_array for input2
        min_width : int
            Minimum number of bytes that must match

        Returns
        -------
        Column
            New column of duplicates
        """
        sa1_col = sa1._column
        sa2_col = sa2._column
        input2_col = input2._column
        return self._return_or_inplace(
            self._column.resolve_duplicates_pair(
                sa1_col, input2_col, sa2_col, min_width
            ),
            inplace=False,
            expand=False,
            retain_index=False,
        )

    def edit_distance(self, targets) -> Series | Index:
        """
        The ``targets`` strings are measured against the strings in this
        instance using the Levenshtein edit distance algorithm.
        https://en.wikipedia.org/wiki/Levenshtein_distance

        The ``targets`` parameter may also be a single string in which
        case the edit distance is computed for all the strings against
        that single string.

        Parameters
        ----------
        targets : array-like, Sequence or Series or str
            The string(s) to measure against each string.

        Returns
        -------
        Series or Index of int32.

        Examples
        --------
        >>> import cudf
        >>> sr = cudf.Series(["puppy", "doggy", "kitty"])
        >>> targets = cudf.Series(["pup", "dogie", "kitten"])
        >>> sr.str.edit_distance(targets=targets)
        0    2
        1    2
        2    2
        dtype: int32
        >>> sr.str.edit_distance("puppy")
        0    0
        1    4
        2    4
        dtype: int32
        """
        if is_scalar(targets):
            targets_column = as_column([targets])
        elif can_convert_to_column(targets):
            targets_column = as_column(targets)
        else:
            raise TypeError(
                f"targets should be an str, array-like or Series object, "
                f"found {type(targets)}"
            )

        return self._return_or_inplace(
            self._column.edit_distance(targets_column)  # type: ignore[arg-type]
        )

    def minhash(
        self, seed: int, a: ColumnLike, b: ColumnLike, width: int
    ) -> Series | Index:
        """
        Compute the minhash of a strings column or a list strings column
        of terms.

        This uses the MurmurHash3_x86_32 algorithm for the hash function
        if a or b are of type np.uint32 or MurmurHash3_x86_128 if a and b
        are of type np.uint64.

        Calculation uses the formula (hv * a + b) % mersenne_prime
        where hv is the hash of a substring of width characters
        or ngrams of strings if a list column,
        a and b are provided values and mersenne_prime is 2^61-1.

        Parameters
        ----------
        seed : int
            The seed used for the hash algorithm.
        a : ColumnLike
            Values for minhash calculation.
            Must be of type uint32 or uint64.
        b : ColumnLike
            Values for minhash calculation.
            Must be of type uint32 or uint64.
        width : int
            The width of the substring to hash.
            Or the ngram number of strings to hash.

        Examples
        --------
        >>> import cudf
        >>> import numpy as np
        >>> s = cudf.Series(['this is my', 'favorite book'])
        >>> a = cudf.Series([1, 2, 3], dtype=np.uint32)
        >>> b = cudf.Series([4, 5, 6], dtype=np.uint32)
        >>> s.str.minhash(0, a=a, b=b, width=5)
        0    [1305480171, 462824409, 74608232]
        1       [32665388, 65330773, 97996158]
        dtype: list
        >>> sl = cudf.Series([['this', 'is', 'my'], ['favorite', 'book']])
        >>> sl.str.minhash(width=2, seed=0, a=a, b=b)
        0      [416367551, 832735099, 1249102647]
        1    [1906668704, 3813337405, 1425038810]
        dtype: list
        """
        a_column = as_column(a)
        b_column = as_column(b)
        if a_column.dtype != np.uint32 and a_column.dtype != np.uint64:
            raise ValueError(
                f"Expecting a and b Series as uint32 or unint64, got {type(a)}"
            )
        if a_column.dtype != b_column.dtype:
            raise ValueError(
                f"Expecting a and b Series dtype to match, got {type(a)}"
            )
        seed = a.dtype.type(seed)
        if a_column.dtype == np.uint32:
            if isinstance(self._parent.dtype, ListDtype):
                return self.minhash_ngrams(width, seed, a_column, b_column)
            else:
                return self._return_or_inplace(
                    self._column.minhash(seed, a_column, b_column, width)  # type: ignore[arg-type]
                )
        else:
            if isinstance(self._parent.dtype, ListDtype):
                return self.minhash64_ngrams(width, seed, a_column, b_column)
            else:
                return self.minhash64(seed, a_column, b_column, width)

    def minhash64(
        self, seed: int | np.uint64, a: ColumnLike, b: ColumnLike, width: int
    ) -> Series | Index:
        """
        Compute the minhash of a strings column.
        This uses the MurmurHash3_x64_128 algorithm for the hash function.

        Calculation uses the formula (hv * a + b) % mersenne_prime
        where hv is the hash of a substring of width characters,
        a and b are provided values and mersenne_prime is 2^61-1.

        Parameters
        ----------
        seed : uint64
            The seed used for the hash algorithm.
        a : ColumnLike
            Values for minhash calculation.
            Must be of type uint64.
        b : ColumnLike
            Values for minhash calculation.
            Must be of type uint64.
        width : int
            The width of the substring to hash.

        Examples
        --------
        >>> import cudf
        >>> import numpy as np
        >>> s = cudf.Series(['this is my', 'favorite book', 'to read'])
        >>> a = cudf.Series([2, 3], dtype=np.uint64)
        >>> b = cudf.Series([5, 6], dtype=np.uint64)
        >>> s.str.minhash64(0, a=a, b=b, width=5)
        0    [172452388517576012, 316595762085180527]
        1      [71427536958126239, 58787297728258215]
        2    [423885828176437114, 1140588505926961370]
        dtype: list
        """
        a_column = as_column(a)
        if a_column.dtype != np.uint64:
            raise ValueError(
                f"Expecting a Series with dtype uint64, got {type(a)}"
            )
        b_column = as_column(b)
        if b_column.dtype != np.uint64:
            raise ValueError(
                f"Expecting a Series with dtype uint64, got {type(b)}"
            )
        return self._return_or_inplace(
            self._column.minhash64(seed, a_column, b_column, width)  # type: ignore[arg-type]
        )

    def minhash_ngrams(
        self, ngrams: int, seed: int | np.uint32, a: ColumnLike, b: ColumnLike
    ) -> Series | Index:
        """
        Compute the minhash of a list column of strings.

        This uses the MurmurHash3_x86_32 algorithm for the hash function.

        Calculation uses the formula (hv * a + b) % mersenne_prime
        where hv is the hash of a ngrams of strings within each row,
        a and b are provided values and mersenne_prime is 2^61-1.

        Parameters
        ----------
        ngrams : int
            Number of strings to hash within each row.
        seed : uint32
            The seed used for the hash algorithm.
        a : ColumnLike
            Values for minhash calculation.
            Must be of type uint32.
        b : ColumnLike
            Values for minhash calculation.
            Must be of type uint32.

        Examples
        --------
        >>> import cudf
        >>> import numpy as np
        >>> s = cudf.Series([['this', 'is', 'my'], ['favorite', 'book']])
        >>> a = cudf.Series([1, 2, 3], dtype=np.uint32)
        >>> b = cudf.Series([4, 5, 6], dtype=np.uint32)
        >>> s.str.minhash_ngrams(ngrams=2, seed=0, a=a, b=b)
        0      [416367551, 832735099, 1249102647]
        1    [1906668704, 3813337405, 1425038810]
        dtype: list
        """
        a_column = as_column(a)
        if a_column.dtype != np.uint32:
            raise ValueError(
                f"Expecting a Series with dtype uint32, got {type(a)}"
            )
        b_column = as_column(b)
        if b_column.dtype != np.uint32:
            raise ValueError(
                f"Expecting a Series with dtype uint32, got {type(b)}"
            )
        return self._return_or_inplace(
            self._column.minhash_ngrams(ngrams, seed, a_column, b_column)  # type: ignore[attr-defined]
        )

    def minhash64_ngrams(
        self, ngrams: int, seed: int | np.uint64, a: ColumnLike, b: ColumnLike
    ) -> Series | Index:
        """
        Compute the minhash of a list column of strings.

        This uses the MurmurHash3_x64_128 algorithm for the hash function.

        Calculation uses the formula (hv * a + b) % mersenne_prime
        where hv is the hash of a ngrams of strings within each row,
        a and b are provided values and mersenne_prime is 2^61-1.

        Parameters
        ----------
        ngrams : int
            Number of strings to hash within each row.
        seed : uint64
            The seed used for the hash algorithm.
        a : ColumnLike
            Values for minhash calculation.
            Must be of type uint64.
        b : ColumnLike
            Values for minhash calculation.
            Must be of type uint64.

        Examples
        --------
        >>> import cudf
        >>> import numpy as np
        >>> s = cudf.Series([['this', 'is', 'my'], ['favorite', 'book']])
        >>> a = cudf.Series([2, 3], dtype=np.uint64)
        >>> b = cudf.Series([5, 6], dtype=np.uint64)
        >>> s.str.minhash64_ngrams(ngrams=2, seed=0, a=a, b=b)
        0    [1304293339825194559, 1956440009737791829]
        1     [472203876238918632, 1861227318965224922]
        dtype: list
        """
        a_column = as_column(a)
        if a_column.dtype != np.uint64:
            raise ValueError(
                f"Expecting a Series with dtype uint64, got {type(a)}"
            )
        b_column = as_column(b)
        if b_column.dtype != np.uint64:
            raise ValueError(
                f"Expecting a Series with dtype uint64, got {type(b)}"
            )
        return self._return_or_inplace(
            self._column.minhash64_ngrams(ngrams, seed, a_column, b_column)  # type: ignore[attr-defined]
        )

    def jaccard_index(self, input: Series, width: int) -> Series | Index:
        """
        Compute the Jaccard index between this column and the given
        input strings column.

        Parameters
        ----------
        input : Series
            The input strings column to compute the Jaccard index against.
            Must have the same number of strings as this column.
        width : int
            The number of characters for the sliding window calculation.

        Examples
        --------
        >>> import cudf
        >>> str1 = cudf.Series(["the brown dog", "jumped about"])
        >>> str2 = cudf.Series(["the black cat", "jumped around"])
        >>> str1.str.jaccard_index(str2, 5)
        0    0.058824
        1    0.307692
        dtype: float32
        """
        return self._return_or_inplace(
            self._column.jaccard_index(input._column, width)
        )
