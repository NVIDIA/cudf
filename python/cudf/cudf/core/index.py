# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import itertools
import operator
import warnings
from collections import Counter
from collections.abc import Hashable, MutableMapping
from functools import cache, cached_property
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import cupy
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.tools.times import to_time

import pylibcudf as plc

import cudf
from cudf.api.extensions import no_default
from cudf.api.types import (
    is_hashable,
    is_integer,
    is_list_like,
    is_scalar,
)
from cudf.core._internals import copying, sorting
from cudf.core.accessors import StringMethods
from cudf.core.column import (
    CategoricalColumn,
    ColumnBase,
    DatetimeColumn,
    DatetimeTZColumn,
    IntervalColumn,
    NumericalColumn,
    StringColumn,
    StructColumn,
    TimeDeltaColumn,
    access_columns,
)
from cudf.core.column.column import (
    _normalize_types_column,
    as_column,
    column_empty,
    concat_columns,
)
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.copy_types import GatherMap
from cudf.core.dtype.validators import (
    is_dtype_obj_numeric,
    is_dtype_obj_string,
)
from cudf.core.dtypes import (
    CategoricalDtype,
    IntervalDtype,
)
from cudf.core.join._join_helpers import _match_join_keys
from cudf.core.single_column_frame import SingleColumnFrame
from cudf.errors import MixedTypeError
from cudf.utils.docutils import copy_docstring
from cudf.utils.dtypes import (
    DEFAULT_STRING_DTYPE,
    SIZE_TYPE_DTYPE,
    _maybe_convert_to_default_type,
    cudf_dtype_from_pa_type,
    cudf_dtype_to_pa_type,
    dtype_from_pylibcudf_column,
    dtype_to_pylibcudf_type,
    find_common_type,
    get_dtype_of_same_kind,
    is_mixed_with_object_dtype,
    is_pandas_nullable_extension_dtype,
)
from cudf.utils.performance_tracking import _performance_tracking
from cudf.utils.scalar import pa_scalar_to_plc_scalar
from cudf.utils.utils import _is_same_name, _warn_no_dask_cudf

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
    from datetime import tzinfo

    from cudf._typing import ColumnLike, Dtype, DtypeObj
    from cudf.core.dataframe import DataFrame
    from cudf.core.multiindex import MultiIndex
    from cudf.core.series import Series
    from cudf.core.tools.datetimes import DateOffset


def ensure_index(index_like: Any, nan_as_null=no_default) -> Index:
    """
    Ensure an Index is returned.

    Avoids a shallow copy compared to calling cudf.Index(...)
    """
    if not isinstance(index_like, Index):
        return Index(index_like, nan_as_null=nan_as_null)
    return index_like


def _get_result_name(
    left_name: Hashable, right_name: Hashable
) -> Hashable | None:
    return left_name if _is_same_name(left_name, right_name) else None


def _tag_levels(names: list[Hashable]) -> list[tuple[Hashable, int]]:
    """Tag each level name with its occurrence index.

    Identifying a level by ``(name, occurrence)`` lets duplicate level names be
    matched by their position among equal names instead of collapsing onto a
    single join key. The tags are unique by construction.
    """
    seen: Counter = Counter()
    tags = []
    for name in names:
        tags.append((name, seen[name]))
        seen[name] += 1
    return tags


def _join_label(
    side: str,
    pos: int,
    tag: tuple[Hashable, int],
    overlap_index: dict[tuple[Hashable, int], int],
    key_labels: list[str],
) -> str:
    """Map a level to its merge-key label.

    Shared levels get a stable ``__join_key_<i>`` label so merge keys line up;
    levels unique to one side get a distinct placeholder so non-key columns
    never collide.
    """
    if tag in overlap_index:
        return key_labels[overlap_index[tag]]
    return f"__{side}_{pos}"


def _lexsorted_equal_range(
    idx: Index | MultiIndex,
    keys: list[ColumnBase],
    is_sorted: bool,
) -> tuple[int, int, ColumnBase | None]:
    """Get equal range for key in lexicographically sorted index. If index
    is not sorted when called, a sort will take place and `sort_inds` is
    returned. Otherwise `None` is returned in that position.
    """
    if not is_sorted:
        sort_inds = idx._get_sorted_inds()
        sort_vals = idx._gather(sort_inds)
    else:
        sort_inds = None
        sort_vals = idx
    sources = sort_vals._columns
    len_sources = len(sources)
    plc_lower_bound = sorting.search_sorted(
        sort_vals._columns,
        keys,
        side="left",
        ascending=itertools.repeat(
            sort_vals.is_monotonic_increasing, times=len_sources
        ),
        na_position=itertools.repeat("last", times=len_sources),
    )
    lower_bound = plc.copying.get_element(plc_lower_bound, 0).to_py()
    plc_upper_bound = sorting.search_sorted(
        sources,
        keys,
        side="right",
        ascending=itertools.repeat(
            sort_vals.is_monotonic_increasing, times=len_sources
        ),
        na_position=itertools.repeat("last", times=len_sources),
    )
    upper_bound = plc.copying.get_element(plc_upper_bound, 0).to_py()

    return cast("int", lower_bound), cast("int", upper_bound), sort_inds


def _index_from_data(data: MutableMapping, name: Any = no_default):
    """Construct an index of the appropriate type from some data."""
    if len(data) == 0:
        raise ValueError("Cannot construct Index from any empty Table")
    if len(data) == 1:
        values = next(iter(data.values()))

        if isinstance(values, NumericalColumn):
            index_class_type = Index
        elif isinstance(values, DatetimeColumn):
            index_class_type = DatetimeIndex
        elif isinstance(values, TimeDeltaColumn):
            index_class_type = TimedeltaIndex
        elif isinstance(values, StringColumn):
            index_class_type = Index
        elif isinstance(values, CategoricalColumn):
            index_class_type = CategoricalIndex
        elif isinstance(values, IntervalColumn):
            index_class_type = IntervalIndex
        elif isinstance(values, StructColumn):
            index_class_type = Index
        else:
            raise NotImplementedError(
                "Unsupported column type passed to "
                f"create an Index: {type(values)}"
            )
    else:
        index_class_type = cudf.MultiIndex
    return index_class_type._from_data(data, name)


def validate_range_arg(arg, arg_name: Literal["start", "stop", "step"]) -> int:
    """Validate start/stop/step argument in RangeIndex.__init__"""
    if not is_integer(arg):
        raise TypeError(
            f"{arg_name} must be an integer, not {type(arg).__name__}"
        )
    return int(arg)


class Index(SingleColumnFrame):
    """
    Immutable sequence used for indexing and alignment.

    The basic object storing axis labels for all pandas objects.

    Parameters
    ----------
    data : array-like (1-dimensional)
    dtype : str, numpy.dtype, or ExtensionDtype, optional
        Data type for the output Index. If not specified, this will be
        inferred from `data`.
    copy : bool, default False
        Copy input data.
    name : object
        Name to be stored in the index.
    tupleize_cols : bool (default: True)
        When True, attempt to create a MultiIndex if possible.
        Currently not supported.
    """

    _accessors: set[Any] = set()

    @_performance_tracking
    def __new__(cls, *args, **kwargs) -> Self:
        if cls is Index:
            return _as_index(*args, **kwargs)
        return object.__new__(cls)

    @_performance_tracking
    def __init__(
        self,
        data=None,
        dtype=None,
        copy: bool = False,
        name=None,
        tupleize_cols: bool = True,
        nan_as_null=no_default,
    ) -> None:
        # Constructed in __new__
        # Subclasses should initialize with
        # SingleColumnFrame.__init__(self, {name: column})
        return None

    @property
    def _constructor(self):
        return Index

    @property
    def _constructor_expanddim(self):
        return cudf.MultiIndex

    @_performance_tracking
    def __array_function__(self, func, types, args, kwargs):
        # check if the function is implemented for the current type
        cudf_index_module = type(self)
        for submodule in func.__module__.split(".")[1:]:
            # point cudf_index_module to the correct submodule
            if hasattr(cudf_index_module, submodule):
                cudf_index_module = getattr(cudf_index_module, submodule)
            else:
                return NotImplemented

        fname = func.__name__

        handled_types = [Index, cudf.Series]

        # check if  we don't handle any of the types (including sub-class)
        for t in types:
            if not any(
                issubclass(t, handled_type) for handled_type in handled_types
            ):
                return NotImplemented

        if hasattr(cudf_index_module, fname):
            cudf_func = getattr(cudf_index_module, fname)
            # Handle case if cudf_func is same as numpy function
            if cudf_func is func:
                return NotImplemented
            else:
                result = cudf_func(*args, **kwargs)
                if fname == "unique":
                    # NumPy expects a sorted result for `unique`, which is not
                    # guaranteed by cudf.Index.unique.
                    result = result.sort_values()
                return result

        else:
            return NotImplemented

    @_performance_tracking
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Defer to Series (and DataFrame) which have higher priority than
        # Index when participating in a numpy ufunc — pandas dispatches
        # ``np.ufunc(Index, Series)`` to ``Series`` so the result is a
        # ``Series``. Without this, cudf would handle it here and return an
        # ``Index``.
        if any(
            isinstance(inp, (cudf.Series, cudf.DataFrame))
            for inp in inputs
            if inp is not self
        ):
            return NotImplemented

        ret = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

        if ret is not None:
            return ret

        # Attempt to dispatch all other functions to cupy.
        cupy_func = getattr(cupy, ufunc.__name__)
        if cupy_func:
            if ufunc.nin == 2:
                other = inputs[self is inputs[0]]
                inputs = self._make_operands_for_binop(other)
            else:
                inputs = {
                    name: (col, None, False, None)
                    for name, col in self._column_labels_and_values
                }

            data = self._apply_cupy_ufunc_to_operands(
                ufunc, cupy_func, inputs, **kwargs
            )

            out = tuple(_index_from_data(out) for out in data)
            return out[0] if ufunc.nout == 1 else out

        return NotImplemented

    @classmethod
    @_performance_tracking
    def _from_column(
        cls, column: ColumnBase, *, name: Hashable = None
    ) -> Self:
        if cls is Index:
            ca = ColumnAccessor({name: column}, verify=False)
            return _index_from_data(ca)
        else:
            return super()._from_column(column, name=name)

    @classmethod
    @_performance_tracking
    def _from_data(cls, data: MutableMapping, name: Any = no_default) -> Self:
        out = super()._from_data(data=data)
        if name is not no_default:
            out.name = name
        return out

    @_performance_tracking
    def _from_data_like_self(self, data: MutableMapping) -> Self:
        return _index_from_data(data, self.name)

    @_performance_tracking
    def to_pylibcudf(self) -> tuple[plc.Column, dict]:
        """
        Convert this Index to a pylibcudf.Column.

        Returns
        -------
        pylibcudf.Column
            A pylibcudf.Column referencing the same data.
        dict
            Dict of metadata (includes name)

        Notes
        -----
        This is always a zero-copy operation. The result is a view of the
        existing data. Changes to the pylibcudf data will be reflected back
        to the cudf object and vice versa.
        """
        metadata = {"name": self.name}
        return self._column.to_pylibcudf(), metadata

    @classmethod
    @_performance_tracking
    def from_pylibcudf(
        cls, col: plc.Column, metadata: dict | None = None
    ) -> Self:
        """
        Create a Index from a pylibcudf.Column.

        Parameters
        ----------
        col : pylibcudf.Column
            The input Column.

        Returns
        -------
        pylibcudf.Column
            A new pylibcudf.Column referencing the same data.
        metadata : dict | None
            The Index metadata.

        Notes
        -----
        This function will generate an Index which contains a Column
        pointing to the provided pylibcudf Column.  It will directly access
        the data and mask buffers of the pylibcudf Column, so the newly created
        object is not tied to the lifetime of the original pylibcudf.Column.
        """
        name = None
        if metadata is not None:
            if not (
                isinstance(metadata, dict)
                and len(metadata) == 1
                and set(metadata) == {"name"}
            ):
                raise ValueError("Metadata dict must only contain a name")
            name = metadata.get("name")
        normalized = _normalize_types_column(col)
        return cls._from_column(
            ColumnBase.create(
                normalized, dtype=dtype_from_pylibcudf_column(normalized)
            ),
            name=name,
        )

    @classmethod
    @_performance_tracking
    def from_arrow(cls, obj: pa.Array) -> Index | MultiIndex:
        """Create from PyArrow Array/ChunkedArray.

        Parameters
        ----------
        array : PyArrow Array/ChunkedArray
            PyArrow Object which has to be converted.

        Raises
        ------
        TypeError for invalid input type.

        Returns
        -------
        SingleColumnFrame

        Examples
        --------
        >>> import cudf
        >>> import pyarrow as pa
        >>> cudf.Index.from_arrow(pa.array(["a", "b", None]))
        Index(['a', 'b', nan], dtype='str')
        """
        try:
            return cls._from_column(ColumnBase.from_arrow(obj))
        except TypeError:
            # Try interpreting object as a MultiIndex before failing.
            return cudf.MultiIndex.from_arrow(obj)

    @cached_property
    def is_monotonic_increasing(self) -> bool:
        return super().is_monotonic_increasing

    @cached_property
    def is_monotonic_decreasing(self) -> bool:
        return super().is_monotonic_decreasing

    @property
    def has_duplicates(self) -> bool:
        # .is_unique is already a cached_property
        return not self.is_unique

    @property
    def nlevels(self) -> int:
        """
        Number of levels.
        """
        return self._num_columns

    @property
    def names(self) -> pd.core.indexes.frozen.FrozenList:
        """
        Returns a FrozenList containing the name of the Index.
        """
        return pd.core.indexes.frozen.FrozenList([self.name])

    @names.setter
    def names(self, values: list[Hashable]) -> None:
        if not is_list_like(values):
            raise ValueError("Names must be a list-like")

        num_values = len(values)
        if num_values > 1:
            raise ValueError(
                f"Length of new names must be 1, got {num_values}"
            )

        self.name = values[0]

    def _set_names(self, names, inplace: bool = False) -> Self | None:
        if inplace:
            idx = self
        else:
            idx = self.copy(deep=False)

        idx.names = names
        if not inplace:
            return idx
        return None

    def set_names(
        self, names, level=None, inplace: bool = False
    ) -> Self | None:
        """
        Set Index or MultiIndex name.
        Able to set new names partially and by level.

        Parameters
        ----------
        names : label or list of label
            Name(s) to set.
        level : int, label or list of int or label, optional
            If the index is a MultiIndex, level(s) to set (None for all
            levels). Otherwise level must be None.
        inplace : bool, default False
            Modifies the object directly, instead of creating a new Index or
            MultiIndex.

        Returns
        -------
        Index
            The same type as the caller or None if inplace is True.

        See Also
        --------
        cudf.Index.rename : Able to set new names without level.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index([1, 2, 3, 4])
        >>> idx
        Index([1, 2, 3, 4], dtype='int64')
        >>> idx.set_names('quarter')
        Index([1, 2, 3, 4], dtype='int64', name='quarter')
        >>> idx = cudf.MultiIndex.from_product([['python', 'cobra'],
        ... [2018, 2019]])
        >>> idx
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   )
        >>> idx.names
        FrozenList([None, None])
        >>> idx.set_names(['kind', 'year'], inplace=True)
        >>> idx.names
        FrozenList(['kind', 'year'])
        >>> idx.set_names('species', level=0, inplace=True)
        >>> idx.names
        FrozenList(['species', 'year'])
        """
        if level is not None:
            raise ValueError("Level must be None for non-MultiIndex")

        if not is_list_like(names):
            names = [names]

        return self._set_names(names=names, inplace=inplace)

    def rename(self, name: Hashable, inplace: bool = False) -> Index | None:
        """
        Alter Index name.

        Defaults to returning new index.

        Parameters
        ----------
        name : label
            Name(s) to set.

        Returns
        -------
        Index

        Examples
        --------
        >>> import cudf
        >>> index = cudf.Index([1, 2, 3], name='one')
        >>> index
        Index([1, 2, 3], dtype='int64', name='one')
        >>> index.name
        'one'
        >>> renamed_index = index.rename('two')
        >>> renamed_index
        Index([1, 2, 3], dtype='int64', name='two')
        >>> renamed_index.name
        'two'
        """
        if inplace is True:
            self.name = name
            return None
        else:
            out = self.copy(deep=False)
            out.name = name
            return out

    def find_label_range(self, loc: slice) -> slice:
        """
        Translate a label-based slice to an index-based slice

        Parameters
        ----------
        loc
            slice to search for.

        Notes
        -----
        As with all label-based searches, the slice is right-closed.

        Returns
        -------
        New slice translated into integer indices of the index (right-open).
        """
        # pandas doesn't have this method, deprecate?
        start = loc.start
        stop = loc.stop
        step = 1 if loc.step is None else loc.step
        start_side: Literal["left", "right"]
        stop_side: Literal["left", "right"]
        if step < 0:
            start_side, stop_side = "right", "left"
        else:
            start_side, stop_side = "left", "right"
        istart = (
            None
            if start is None
            else self.get_slice_bound(start, side=start_side)
        )
        istop = (
            None
            if stop is None
            else self.get_slice_bound(stop, side=stop_side)
        )
        if step < 0:
            # Fencepost
            istart = None if istart is None else max(istart - 1, 0)
            istop = None if (istop is None or istop == 0) else istop - 1
        return slice(istart, istop, step)

    def get_slice_bound(
        self,
        label: Hashable,
        side: Literal["left", "right"],
    ) -> int:
        """
        Calculate slice bound that corresponds to given label.
        Returns leftmost (one-past-the-rightmost if ``side=='right'``) position
        of given label.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}

        Returns
        -------
        int
            Index of label.
        """
        if side not in {"left", "right"}:
            raise ValueError(f"Invalid side argument {side}")
        if self.is_monotonic_increasing or self.is_monotonic_decreasing:
            return self.searchsorted(
                label, side=side, ascending=self.is_monotonic_increasing
            )
        else:
            try:
                left, right = self._column._find_first_and_last(label)
            except ValueError:
                raise KeyError(f"{label=} not in index")
            if left != right:
                raise KeyError(
                    f"Cannot get slice bound for non-unique label {label=}"
                )
            if side == "left":
                return left
            else:
                return right + 1

    def shift(self, periods: int = 1, freq=None) -> Self:
        """
        Shift index by desired number of time frequency increments.
        """
        raise NotImplementedError(
            f"Shift is not implemented yet for {type(self).__name__}"
        )

    def _union(self, other, sort: bool | None = None) -> Index:
        # TODO: As a future optimization we should explore
        # not doing `to_frame`
        self_df = self.to_frame(index=False, name=0)
        other_df = other.to_frame(index=False, name=0)
        self_df["order"] = self_df.index
        other_df["order"] = other_df.index
        res = self_df.merge(other_df, on=[0], how="outer")
        res = res.sort_values(
            by=res._data.to_pandas_index[1:], ignore_index=True
        )
        union_result = _index_from_data({0: res._data[0]})

        if sort in {None, True} and len(other):
            return union_result.sort_values()
        return union_result

    def union(self, other, sort: bool | None = None) -> Index:
        """
        Form the union of two Index objects.

        Parameters
        ----------
        other : Index or array-like
        sort : bool or None, default None
            Whether to sort the resulting Index.

            * None : Sort the result, except when

              1. `self` and `other` are equal.
              2. `self` or `other` has length 0.

            * False : do not sort the result.
            * True : Sort the result (which may raise TypeError).

        Returns
        -------
        union : Index

        Examples
        --------
        Union of an Index
        >>> import cudf
        >>> import pandas as pd
        >>> idx1 = cudf.Index([1, 2, 3, 4])
        >>> idx2 = cudf.Index([3, 4, 5, 6])
        >>> idx1.union(idx2)
        Index([1, 2, 3, 4, 5, 6], dtype='int64')

        MultiIndex case

        >>> idx1 = cudf.from_pandas(
        ...    pd.MultiIndex.from_arrays(
        ...         [[1, 1, 2, 2], ["Red", "Blue", "Red", "Blue"]]
        ...    )
        ... )
        >>> idx1
        MultiIndex([(1,  'Red'),
                    (1, 'Blue'),
                    (2,  'Red'),
                    (2, 'Blue')],
                   )
        >>> idx2 = cudf.from_pandas(
        ...    pd.MultiIndex.from_arrays(
        ...         [[3, 3, 2, 2], ["Red", "Green", "Red", "Green"]]
        ...    )
        ... )
        >>> idx2
        MultiIndex([(3,   'Red'),
                    (3, 'Green'),
                    (2,   'Red'),
                    (2, 'Green')],
                   )
        >>> idx1.union(idx2)
        MultiIndex([(1,  'Blue'),
                    (1,   'Red'),
                    (2,  'Blue'),
                    (2, 'Green'),
                    (2,   'Red'),
                    (3, 'Green'),
                    (3,   'Red')],
                   )
        >>> idx1.union(idx2, sort=False)
        MultiIndex([(1,   'Red'),
                    (1,  'Blue'),
                    (2,   'Red'),
                    (2,  'Blue'),
                    (3,   'Red'),
                    (3, 'Green'),
                    (2, 'Green')],
                   )
        """
        if not isinstance(other, Index):
            other = Index(other, name=getattr(other, "name", self.name))

        if sort not in {None, False, True}:
            raise ValueError(
                f"The 'sort' keyword only takes the values of "
                f"[None, False, True]; {sort} was passed."
            )

        # pandas ignores the dtype of a zero-length RangeIndex or
        # object-dtype operand when reconciling dtypes for union
        # (pandas-dev/pandas#60797), so such operands must neither
        # trigger mixed-type errors nor dtype promotion.
        dtype_ignored = any(
            len(idx) == 0
            and (isinstance(idx, RangeIndex) or idx.dtype == np.dtype(object))
            for idx in (self, other)
        )

        if not dtype_ignored:
            # Cache dtype.kind to avoid repeated attribute access
            self_kind = self.dtype.kind
            other_kind = other.dtype.kind

            if (
                (self_kind in "Mm" or other_kind in "Mm")
                and self_kind != other_kind
                and not isinstance(self.dtype, CategoricalDtype)
                and not isinstance(other.dtype, CategoricalDtype)
            ):
                # datetime/timedelta + any other kind results in object
                # dtype for union in pandas, which cudf cannot represent;
                # the non-empty merge path already refuses these pairs.
                # Categorical operands are excluded because the merge
                # decategorizes them, matching pandas.
                raise MixedTypeError("Cannot perform union with mixed types")
            if (
                self_kind == "M"
                and other_kind == "M"
                and isinstance(self.dtype, pd.DatetimeTZDtype)
                != isinstance(other.dtype, pd.DatetimeTZDtype)
            ):
                # tz-naive + tz-aware results in object dtype in pandas.
                raise MixedTypeError("Cannot perform union with mixed types")

            if (self_kind == "b" and other_kind != "b") or (
                self_kind != "b" and other_kind == "b"
            ):
                # Bools + other types will result in mixed type.
                # This is not yet consistent in pandas and specific to
                # APIs.
                raise MixedTypeError("Cannot perform union with mixed types")
            if (self_kind == "i" and other_kind == "u") or (
                self_kind == "u" and other_kind == "i"
            ):
                # signed + unsigned types will result in mixed type for
                # union in pandas, which cudf cannot represent.
                raise MixedTypeError("Cannot perform union with mixed types")

        # pandas reconciles mismatched dtypes to their common type before
        # short-circuiting on an empty operand, so the result dtype must
        # not depend on which operand is empty.
        promote_dtype: DtypeObj | None = None
        if not dtype_ignored and self.dtype != other.dtype:
            if is_dtype_obj_numeric(
                self.dtype, include_decimal=False
            ) and is_dtype_obj_numeric(other.dtype, include_decimal=False):
                common_dtype = find_common_type([self.dtype, other.dtype])
                # Mixed-backend pairs can only reconcile to object; never
                # promote numeric values through a lossy object cast.
                if is_dtype_obj_numeric(common_dtype, include_decimal=False):
                    promote_dtype = common_dtype
            elif (
                isinstance(self.dtype, np.dtype)
                and isinstance(other.dtype, np.dtype)
                and self.dtype.kind == other.dtype.kind
                and self.dtype.kind in "Mm"
            ):
                # Same-kind datetime/timedelta dtypes only differ in
                # unit; reconcile to the common resolution.
                promote_dtype = find_common_type([self.dtype, other.dtype])

        if not len(other):
            res = self._get_reconciled_name_object(other)
            if promote_dtype is not None:
                res = res.astype(promote_dtype)
            if sort:
                return res.sort_values()  # type: ignore[return-value]
            return res
        elif self.equals(other):
            common_dtype = find_common_type([self.dtype, other.dtype])
            res = self._get_reconciled_name_object(other).astype(common_dtype)
            if sort:
                return res.sort_values()
            return res
        elif not len(self):
            res = other._get_reconciled_name_object(self)
            if promote_dtype is not None:
                res = res.astype(promote_dtype)
            if sort:
                return res.sort_values()
            return res

        result = self._union(other, sort=sort)
        result.name = _get_result_name(self.name, other.name)
        return result

    def _intersection(self, other, sort: bool | None = None) -> Index:
        lcol = self.unique()._column
        rcol = other.unique()._column
        if (
            is_dtype_obj_numeric(self.dtype, include_decimal=False)
            and is_dtype_obj_numeric(other.dtype, include_decimal=False)
            and self.dtype != other.dtype
        ):
            # pandas casts mismatched numeric dtypes to their common type
            # for set operations ("cast to float, not object"); cast up
            # front so the merge's key-dtype rules (which keep the left
            # operand's dtype for inner joins) don't leak the left dtype
            # into the result.
            common_dtype = find_common_type([self.dtype, other.dtype])
            lcol = lcol.astype(common_dtype)
            rcol = rcol.astype(common_dtype)
        intersection_result = _index_from_data(
            cudf.DataFrame._from_data({"None": lcol})
            .merge(
                cudf.DataFrame._from_data({"None": rcol}),
                how="inner",
                on="None",
            )
            ._data
        )

        if sort in {None, True} and len(other):
            return intersection_result.sort_values()
        return intersection_result

    def intersection(self, other, sort: bool | None = False) -> Index:
        """
        Form the intersection of two Index objects.

        This returns a new Index with elements common to the index and `other`.

        Parameters
        ----------
        other : Index or array-like
        sort : False or None, default False
            Whether to sort the resulting index.

            * False : do not sort the result.
            * None : sort the result, except when `self` and `other` are equal
              or when the values cannot be compared.
            * True : Sort the result (which may raise TypeError).

        Returns
        -------
        intersection : Index

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> idx1 = cudf.Index([1, 2, 3, 4])
        >>> idx2 = cudf.Index([3, 4, 5, 6])
        >>> idx1.intersection(idx2)
        Index([3, 4], dtype='int64')

        MultiIndex case

        >>> idx1 = cudf.from_pandas(
        ...    pd.MultiIndex.from_arrays(
        ...         [[1, 1, 3, 4], ["Red", "Blue", "Red", "Blue"]]
        ...    )
        ... )
        >>> idx2 = cudf.from_pandas(
        ...    pd.MultiIndex.from_arrays(
        ...         [[1, 1, 2, 2], ["Red", "Blue", "Red", "Blue"]]
        ...    )
        ... )
        >>> idx1
        MultiIndex([(1,  'Red'),
                    (1, 'Blue'),
                    (3,  'Red'),
                    (4, 'Blue')],
                )
        >>> idx2
        MultiIndex([(1,  'Red'),
                    (1, 'Blue'),
                    (2,  'Red'),
                    (2, 'Blue')],
                )
        >>> idx1.intersection(idx2)
        MultiIndex([(1,  'Red'),
                    (1, 'Blue')],
                )
        >>> idx1.intersection(idx2, sort=False)
        MultiIndex([(1,  'Red'),
                    (1, 'Blue')],
                )
        """

        if not isinstance(other, Index):
            other = Index(
                other,
                name=getattr(other, "name", self.name),
            )

        if sort not in {None, False, True}:
            raise ValueError(
                f"The 'sort' keyword only takes the values of "
                f"[None, False, True]; {sort} was passed."
            )

        if self.equals(other):
            if self.has_duplicates:
                result = self.unique()._get_reconciled_name_object(other)
            else:
                result = self._get_reconciled_name_object(other)
            if sort is True:
                result = result.sort_values()  # type: ignore[assignment]
            return result
        if not len(self) or not len(other):
            if not len(self) and isinstance(other.dtype, CategoricalDtype):
                # Preserve categorical dtype when self is empty,
                # matching pandas behavior.
                result = other[:0]
                result.name = _get_result_name(self.name, other.name)
                return result

            common_dtype = find_common_type([self.dtype, other.dtype])

            lhs = self.unique() if self.has_duplicates else self
            rhs = other
            if not len(other):
                lhs, rhs = rhs, lhs

            return lhs._get_reconciled_name_object(rhs).astype(common_dtype)

        res_name = _get_result_name(self.name, other.name)

        if (
            self.dtype.kind == "b"
            and is_dtype_obj_numeric(other.dtype, include_decimal=False)
            and other.dtype.kind != "b"
        ) or (
            is_dtype_obj_numeric(self.dtype, include_decimal=False)
            and self.dtype.kind != "b"
            and other.dtype.kind == "b"
        ):
            if isinstance(self, cudf.MultiIndex):
                return self[:0].rename(res_name)
            else:
                return Index([], name=res_name)

        if self.has_duplicates:
            lhs = self.unique()
        else:
            lhs = self
        if other.has_duplicates:
            rhs = other.unique()
        else:
            rhs = other
        result = lhs._intersection(rhs, sort=sort)
        result.name = res_name
        return result

    def difference(self, other, sort: bool | None = None) -> Index:
        """
        Return a new Index with elements from the index that are not in
        `other`.

        This is the set difference of two Index objects.

        Parameters
        ----------
        other : Index or array-like
        sort : False or None, default None
            Whether to sort the resulting index. By default, the
            values are attempted to be sorted, but any TypeError from
            incomparable elements is caught by cudf.

            * None : Attempt to sort the result, but catch any TypeErrors
              from comparing incomparable elements.
            * False : Do not sort the result.
            * True : Sort the result (which may raise TypeError).

        Returns
        -------
        difference : Index

        Examples
        --------
        >>> import cudf
        >>> idx1 = cudf.Index([2, 1, 3, 4])
        >>> idx1
        Index([2, 1, 3, 4], dtype='int64')
        >>> idx2 = cudf.Index([3, 4, 5, 6])
        >>> idx2
        Index([3, 4, 5, 6], dtype='int64')
        >>> idx1.difference(idx2)
        Index([1, 2], dtype='int64')
        >>> idx1.difference(idx2, sort=False)
        Index([2, 1], dtype='int64')
        """
        if not isinstance(other, Index):
            other = cudf.Index(
                other,
                name=getattr(other, "name", self.name),
            )

        if sort not in {None, False, True}:
            raise ValueError(
                f"The 'sort' keyword only takes the values "
                f"of [None, False, True]; {sort} was passed."
            )

        if not len(other):
            res = self._get_reconciled_name_object(other).unique()
            if sort:
                return res.sort_values()  # type: ignore[return-value]
            return res
        elif self.equals(other):
            res = self[:0]._get_reconciled_name_object(other).unique()
            if sort:
                return res.sort_values()
            return res

        res_name = _get_result_name(self.name, other.name)

        if is_mixed_with_object_dtype(self, other) or len(other) == 0:
            difference = self.unique()
            difference.name = res_name
            if sort is True:
                return difference.sort_values()  # type: ignore[return-value]
        else:
            other = other.copy(deep=False)
            difference = _index_from_data(
                cudf.DataFrame._from_data({"None": self._column.unique()})
                .merge(
                    cudf.DataFrame._from_data({"None": other._column}),
                    how="leftanti",
                    on="None",
                )
                ._data
            )
            difference.name = res_name

            if self.dtype != other.dtype:
                difference = difference.astype(self.dtype)

        if sort in {None, True} and len(other):
            return difference.sort_values()  # type: ignore[return-value]

        return difference

    def _get_reconciled_name_object(self, other: Index) -> Index:
        """
        If the result of a set operation will be self,
        return self, unless the name changes, in which
        case make a shallow copy of self.
        """
        name = _get_result_name(self.name, other.name)
        if not _is_same_name(self.name, name):
            return self.rename(name)  # type: ignore[return-value]
        return self

    def sort_values(
        self,
        return_indexer: bool = False,
        ascending: bool = True,
        na_position: Literal["first", "last"] = "last",
        key=None,
    ) -> Self | tuple[Self, cupy.ndarray]:
        """
        Return a sorted copy of the index, and optionally return the indices
        that sorted the index itself.

        Parameters
        ----------
        return_indexer : bool, default False
            Should the indices that would sort the index be returned.
        ascending : bool, default True
            Should the index values be sorted in an ascending order.
        na_position : {'first' or 'last'}, default 'last'
            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at
            the end.
        key : None, optional
            This parameter is NON-FUNCTIONAL.

        Returns
        -------
        sorted_index : Index
            Sorted copy of the index.
        indexer : cupy.ndarray, optional
            The indices that the index itself was sorted by.

        See Also
        --------
        cudf.Series.min : Sort values of a Series.
        cudf.DataFrame.sort_values : Sort values in a DataFrame.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index([10, 100, 1, 1000])
        >>> idx
        Index([10, 100, 1, 1000], dtype='int64')

        Sort values in ascending order (default behavior).

        >>> idx.sort_values()
        Index([1, 10, 100, 1000], dtype='int64')

        Sort values in descending order, and also get the indices `idx` was
        sorted by.

        >>> idx.sort_values(ascending=False, return_indexer=True)
        (Index([1000, 100, 10, 1], dtype='int64'), array([3, 1, 0, 2]))

        Sorting values in a MultiIndex:

        >>> midx = cudf.MultiIndex(
        ...      levels=[[1, 3, 4, -10], [1, 11, 5]],
        ...      codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        ...      names=["x", "y"],
        ... )
        >>> midx
        MultiIndex([(  1,  1),
                    (  1,  5),
                    (  3, 11),
                    (  4, 11),
                    (-10,  1)],
                   names=['x', 'y'])
        >>> midx.sort_values()
        MultiIndex([(-10,  1),
                    (  1,  1),
                    (  1,  5),
                    (  3, 11),
                    (  4, 11)],
                   names=['x', 'y'])
        >>> midx.sort_values(ascending=False)
        MultiIndex([(  4, 11),
                    (  3, 11),
                    (  1,  5),
                    (  1,  1),
                    (-10,  1)],
                   names=['x', 'y'])
        """
        if key is not None:
            raise NotImplementedError("key parameter is not yet implemented.")
        if na_position not in {"first", "last"}:
            raise ValueError(f"invalid na_position: {na_position}")

        indices = self.argsort(ascending=ascending, na_position=na_position)
        index_sorted = self.take(indices)

        if return_indexer:
            return index_sorted, indices
        else:
            return index_sorted

    def join(
        self,
        other,
        how: str = "left",
        level=None,
        return_indexers: bool = False,
        sort: bool = False,
    ) -> Index:
        """
        Compute join_index and indexers to conform data structures
        to the new index.

        Parameters
        ----------
        other : Index.
        how : {'left', 'right', 'inner', 'outer'}
        return_indexers : bool, default False
        sort : bool, default False
            Sort the join keys lexicographically in the result Index. If False,
            the order of the join keys depends on the join type (how keyword).

        Returns: index

        Examples
        --------
        >>> import cudf
        >>> lhs = cudf.DataFrame({
        ...     "a": [2, 3, 1],
        ...     "b": [3, 4, 2],
        ... }).set_index(['a', 'b']).index
        >>> lhs
        MultiIndex([(2, 3),
                    (3, 4),
                    (1, 2)],
                   names=['a', 'b'])
        >>> rhs = cudf.DataFrame({"a": [1, 4, 3]}).set_index('a').index
        >>> rhs
        Index([1, 4, 3], dtype='int64', name='a')
        >>> lhs.join(rhs, how='inner')
        MultiIndex([(3, 4),
                    (1, 2)],
                   names=['a', 'b'])
        """
        if return_indexers is not False:
            raise NotImplementedError("return_indexers is not implemented")
        self_is_multi = isinstance(self, cudf.MultiIndex)
        other_is_multi = isinstance(other, cudf.MultiIndex)
        if level is not None:
            if self_is_multi and other_is_multi:
                raise TypeError(
                    "Join on level between two MultiIndex objects is ambiguous"
                )

            if not is_scalar(level):
                raise ValueError("level should be an int or a label only")

        if level is None and self_is_multi and other_is_multi:
            # Joining two MultiIndexes joins on the levels they have in common
            # (mirroring pandas' Index._join_multi). When the level names match
            # exactly the join reduces to a set operation over the full tuples.
            if list(self.names) == list(other.names):
                return self._join_same_level_multi(other, how, sort)
            # pandas ignores ``sort`` for differing-name MultiIndex joins
            # (its Index._join_multi is invoked without it), sorting only outer
            # joins, so ``sort`` is intentionally not forwarded here.
            return self._join_multi(other, how)

        if other_is_multi:
            if how == "left":
                how = "right"
            elif how == "right":
                how = "left"
            rhs = self.copy(deep=False)
            lhs = other.copy(deep=False)
        else:
            lhs = self.copy(deep=False)
            rhs = other.copy(deep=False)
        same_names = lhs.names == rhs.names
        # There should be no `None` values in Joined indices,
        # so essentially it would be `left/right` or 'inner'
        # in case of MultiIndex
        if isinstance(lhs, cudf.MultiIndex):
            on = (
                lhs._data.get_labels_by_index(level)[0]
                if isinstance(level, int)
                else level
            )

            if on is not None:
                rhs.names = (on,)
            on = rhs.names[0]
            if how == "outer":
                how = "left"
            elif how == "right":
                how = "inner"
        else:
            # Both are normal indices
            on = lhs.names[0]
            rhs.names = lhs.names

        lhs = lhs.to_frame()
        rhs = rhs.to_frame()

        output = lhs.merge(rhs, how=how, on=on, sort=sort)

        # If both inputs were MultiIndexes, the output is a MultiIndex.
        # Otherwise, the output is only a MultiIndex if there are multiple
        # columns
        if self_is_multi and other_is_multi:
            return cudf.MultiIndex._from_data(output._data)
        else:
            idx = _index_from_data(output._data)
            idx.name = self.name if same_names else None
            return idx

    @_performance_tracking
    def _join_same_level_multi(self, other, how: str, sort: bool):
        """Join two MultiIndexes that share the same level names.

        With identical level names the join behaves like a set operation over
        the full tuples, matching pandas' ``Index.join`` fast path.
        """
        # pandas always sorts the result of an outer join, regardless of
        # ``sort``; for the other join types ``sort`` controls whether the
        # join keys are ordered lexicographically.
        sort = sort or how == "outer"
        if how == "inner":
            result = self.intersection(other, sort=False)
        elif how == "outer":
            return self.union(other).sort_values()
        elif how == "left":
            result = self.copy(deep=False)
        elif how == "right":
            result = other.copy(deep=False)
        else:
            raise ValueError(f"Invalid join type {how}")
        return result.sort_values() if sort else result

    @_performance_tracking
    def _join_multi(self, other, how: str):
        """Join two MultiIndexes with differing level names.

        The join is performed on the levels the two indexes have in common and
        the remaining levels are carried through, mirroring pandas'
        ``Index._join_multi``. The resulting level order is ``self``'s levels
        followed by ``other``'s unique levels (or the reverse for ``how`` of
        ``"right"``). Like pandas, the result preserves the driving side's row
        order except for ``how="outer"``, which is sorted lexicographically.
        """
        self_names = list(self.names)
        other_names = list(other.names)

        self_tags = _tag_levels(self_names)
        other_tags = _tag_levels(other_names)
        # ``self_tags`` are unique by construction, so this set only removes
        # nothing; ``overlap`` stays an ordered list because ``overlap_index``
        # below relies on self's level order.
        other_tag_set = set(other_tags)
        overlap = [tag for tag in self_tags if tag in other_tag_set]
        if not overlap:
            raise ValueError("cannot join with no overlapping index names")
        overlap_index = {tag: i for i, tag in enumerate(overlap)}

        key_labels = [f"__join_key_{i}" for i in range(len(overlap))]
        left_labels = [
            _join_label("l", i, tag, overlap_index, key_labels)
            for i, tag in enumerate(self_tags)
        ]
        right_labels = [
            _join_label("r", j, tag, overlap_index, key_labels)
            for j, tag in enumerate(other_tags)
        ]

        left_df = cudf.DataFrame._from_data(
            dict(zip(left_labels, self._columns, strict=True))
        )
        right_df = cudf.DataFrame._from_data(
            dict(zip(right_labels, other._columns, strict=True))
        )

        # Preserve the row order of the side that drives the join; pandas only
        # sorts the result for outer joins.
        if how != "outer":
            order_side = right_df if how == "right" else left_df
            order_side["__join_order"] = cudf.Series(range(len(order_side)))

        merged = left_df.merge(right_df, on=key_labels, how=how, sort=False)

        if how == "outer":
            merged = merged.sort_values(by=key_labels)
        else:
            merged = merged.sort_values(by="__join_order")

        # Reassemble levels: the driving side's levels in full, then the other
        # side's unique levels. The driving side is ``self`` except for
        # ``how="right"``, which leads with ``other``.
        if how == "right":
            first_labels, first_names = right_labels, other_names
            second_labels, second_names, second_tags = (
                left_labels,
                self_names,
                self_tags,
            )
        else:
            first_labels, first_names = left_labels, self_names
            second_labels, second_names, second_tags = (
                right_labels,
                other_names,
                other_tags,
            )
        ordered = list(zip(first_labels, first_names, strict=True))
        ordered += [
            (lbl, name)
            for lbl, name, tag in zip(
                second_labels, second_names, second_tags, strict=True
            )
            if tag not in overlap_index
        ]

        return cudf.MultiIndex.from_arrays(
            [merged[lbl] for lbl, _ in ordered],
            names=[name for _, name in ordered],
        )

    def drop_duplicates(
        self,
        keep: Literal["first", "last", False] = "first",
        nulls_are_equal: bool = True,
    ) -> Self:
        """
        Drop duplicate rows in index.

        keep : {"first", "last", False}, default "first"
            - 'first' : Drop duplicates except for the first occurrence.
            - 'last' : Drop duplicates except for the last occurrence.
            - ``False`` : Drop all duplicates.
        nulls_are_equal: bool, default True
            Null elements are considered equal to other null elements.
        """
        columns = list(self._columns)
        original_dtypes = [col.dtype for col in columns]
        result_columns = self._drop_duplicates_columns(
            columns,
            keys=list(range(len(columns))),
            keep=keep,
            nulls_are_equal=nulls_are_equal,
        )
        return self._from_columns_like_self(
            [
                ColumnBase.create(col, dtype)
                for col, dtype in zip(
                    result_columns, original_dtypes, strict=True
                )
            ],
            self._column_names,
        )

    def duplicated(
        self, keep: Literal["first", "last", False] = "first"
    ) -> cupy.ndarray:
        """
        Indicate duplicate index values.

        Duplicated values are indicated as ``True`` values in the resulting
        array. Either all duplicates, all except the first, or all except the
        last occurrence of duplicates can be indicated.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            The value or values in a set of duplicates to mark as missing.

            - ``'first'`` : Mark duplicates as ``True`` except for the first
              occurrence.
            - ``'last'`` : Mark duplicates as ``True`` except for the last
              occurrence.
            - ``False`` : Mark all duplicates as ``True``.

        Returns
        -------
        cupy.ndarray[bool]

        See Also
        --------
        Series.duplicated : Equivalent method on cudf.Series.
        DataFrame.duplicated : Equivalent method on cudf.DataFrame.
        Index.drop_duplicates : Remove duplicate values from Index.

        Examples
        --------
        By default, for each set of duplicated values, the first occurrence is
        set to False and all others to True:

        >>> import cudf
        >>> idx = cudf.Index(['lama', 'cow', 'lama', 'beetle', 'lama'])
        >>> idx.duplicated()
        array([False, False,  True, False,  True])

        which is equivalent to

        >>> idx.duplicated(keep='first')
        array([False, False,  True, False,  True])

        By using 'last', the last occurrence of each set of duplicated values
        is set to False and all others to True:

        >>> idx.duplicated(keep='last')
        array([ True, False,  True, False, False])

        By setting keep to ``False``, all duplicates are True:

        >>> idx.duplicated(keep=False)
        array([ True, False,  True, False,  True])
        """
        return self.to_series().duplicated(keep=keep).to_cupy()

    def dropna(self, how: Literal["any", "all"] = "any") -> Self:
        """
        Drop null rows from Index.

        how : {"any", "all"}, default "any"
            Specifies how to decide whether to drop a row.
            "any" (default) drops rows containing at least
            one null value. "all" drops only rows containing
            *all* null values.
        """
        if how not in {"any", "all"}:
            raise ValueError(f"{how=} must be 'any' or 'all'")
        if not self.hasnans:
            return self.copy(deep=False)

        # Convert nans to nulls to be consistent with IndexedFrame.dropna
        data_columns = [col.nans_to_nulls() for col in self._columns]
        original_dtypes = [col.dtype for col in self._columns]
        result_columns = self._drop_nulls_columns(
            data_columns,
            keys=list(range(len(data_columns))),
            how=how,
        )
        return self._from_columns_like_self(
            [
                ColumnBase.create(col, dtype)
                for col, dtype in zip(
                    result_columns, original_dtypes, strict=True
                )
            ],
            self._column_names,
        )

    def _gather(
        self, gather_map, nullify: bool = False, check_bounds: bool = True
    ) -> Self:
        """Gather rows of index specified by indices in `gather_map`.

        Skip bounds checking if check_bounds is False.
        Set rows to null for all out of bound indices if nullify is `True`.
        """
        gather_map = as_column(gather_map)

        # TODO: For performance, the check and conversion of gather map should
        # be done by the caller. This check will be removed in future release.
        if gather_map.dtype.kind not in "iu":
            gather_map = gather_map.astype(SIZE_TYPE_DTYPE)

        # TODO: This call is purely to validate the bounds if
        # check_bounds is True, require instead that the caller
        # provides a GatherMap.
        GatherMap(gather_map, len(self), nullify=not check_bounds or nullify)
        original_dtypes = [col.dtype for col in self._columns]
        return self._from_columns_like_self(
            [
                ColumnBase.create(col, dtype)
                for col, dtype in zip(
                    copying.gather(self._columns, gather_map, nullify=nullify),
                    original_dtypes,
                    strict=True,
                )
            ],
            self._column_names,
        )

    def take(
        self, indices, axis=0, allow_fill: bool = True, fill_value=None
    ) -> Self:
        """Return a new index containing the rows specified by *indices*

        Parameters
        ----------
        indices : array-like
            Array of ints indicating which positions to take.
        axis : int
            The axis over which to select values, always 0.
        allow_fill : Unsupported
        fill_value : Unsupported

        Returns
        -------
        out : Index
            New object with desired subset of rows.

        Examples
        --------
        >>> idx = cudf.Index(['a', 'b', 'c', 'd', 'e'])
        >>> idx.take([2, 0, 4, 3])
        Index(['c', 'a', 'e', 'd'], dtype='str')
        """

        if axis not in {0, "index"}:
            raise NotImplementedError(
                "Gather along column axis is not yet supported."
            )
        if not allow_fill or fill_value is not None:
            raise NotImplementedError(
                "`allow_fill` and `fill_value` are unsupported."
            )

        return self._gather(indices)

    def _new_index_for_reset_index(
        self, levels: tuple | None, name
    ) -> None | Index:
        """Return the new index after .reset_index"""
        # None is caught later to return RangeIndex
        return None

    def _columns_for_reset_index(
        self, levels: tuple | None
    ) -> Generator[tuple[Any, ColumnBase], None, None]:
        """Return the columns and column names for .reset_index"""
        yield (
            "index" if self.name is None else self.name,
            next(iter(self._columns)),
        )

    def _binaryop(
        self,
        other: Any,
        op: str,
        fill_value: Any = None,
        *args,
        **kwargs,
    ) -> SingleColumnFrame:
        reflect, op = self._check_reflected_op(op)
        operands = self._make_operands_for_binop(other, fill_value, reflect)
        if operands is NotImplemented:
            return NotImplemented
        binop_result = self._colwise_binop(operands, op)

        if isinstance(other, cudf.Series):
            ret = other._from_data_like_self(binop_result)
            other_name = other.name
        else:
            ret = _index_from_data(binop_result)
            other_name = getattr(other, "name", self.name)

        ret.name = self.name if _is_same_name(self.name, other_name) else None
        # pandas preserves the freq on a DatetimeIndex when shifting it by a
        # constant timedelta / Tick offset (vector shifts drop the freq).
        if (
            isinstance(self, DatetimeIndex)
            and self.freq is not None
            and op in {"__add__", "__sub__"}
            and isinstance(ret, DatetimeIndex)
            and is_scalar(other)
        ):
            ret._freq = self._freq

        # pandas returns numpy arrays when the outputs are boolean. We
        # explicitly _do not_ use isinstance here: we want only boolean
        # Indexes, not dtype-specific subclasses.
        if isinstance(ret, (Index, cudf.Series)) and ret.dtype.kind == "b":
            if ret._column.has_nulls():
                ret = ret.fillna(op == "__ne__")

            if isinstance(ret, Index):
                return ret.values
        return ret

    @classmethod
    @_performance_tracking
    def _concat(cls, objs):
        if all(isinstance(obj, RangeIndex) for obj in objs):
            result = _concat_range_index(objs)
        else:
            data = concat_columns([o._column for o in objs])
            if cls is IntervalIndex:
                data = ColumnBase.create(
                    data.plc_column, objs[0]._column.dtype
                )
            result = Index._from_column(data)

        names = {obj.name for obj in objs}
        if len(names) == 1:
            name = names.pop()
        else:
            name = None

        result.name = name
        return result

    @cached_property
    def inferred_type(self) -> str:
        """
        Return a string of the type inferred from the values.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.inferred_type
        'integer'
        """
        if is_dtype_obj_string(self.dtype):
            if len(self) == 0:
                return "empty"
            else:
                return "string"
        elif self.dtype.kind in "iu":
            return "integer"
        elif self.dtype.kind == "f":
            return "floating"
        elif self.dtype.kind == "b":
            return "boolean"
        raise NotImplementedError(
            f"inferred_type not implemented for dtype {self.dtype}"
        )

    @_performance_tracking
    def memory_usage(self, deep: bool = False) -> int:
        return self._column.memory_usage

    def __sizeof__(self):
        return self.memory_usage(deep=True)

    @cached_property
    @_performance_tracking
    def is_unique(self) -> bool:
        return self._column.is_unique

    def _maybe_check_unique(self) -> None:
        """Raise :class:`~pandas.errors.DuplicateLabelError` if non-unique.

        Mirrors ``pandas.Index._maybe_check_unique``. Used by
        ``NDFrame.flags.allows_duplicate_labels`` (``pandas.Flags``)
        when setting the flag to ``False``.
        """
        if not self.is_unique:
            from pandas.errors import DuplicateLabelError

            raise DuplicateLabelError(f"Index has duplicates.\n{self}")

    @_performance_tracking
    def equals(self, other) -> bool:
        if not isinstance(other, Index) or len(self) != len(other):
            return False

        check_dtypes = False

        self_is_categorical = isinstance(self, CategoricalIndex)
        other_is_categorical = isinstance(other, CategoricalIndex)
        if self_is_categorical and not other_is_categorical:
            other = other.astype(self.dtype)
            check_dtypes = True
        elif other_is_categorical and not self_is_categorical:
            self = self.astype(other.dtype)
            check_dtypes = True
        elif (
            not self_is_categorical
            and not other_is_categorical
            and not isinstance(other, RangeIndex)
            and not isinstance(self, type(other))
        ):
            # Can compare Index to CategoricalIndex or RangeIndex
            # Other comparisons are invalid
            return False
        if self_is_categorical and other_is_categorical:
            if self.dtype != other.dtype:
                return False
            return self._column._get_decategorized_column().equals(
                other._column._get_decategorized_column()
            )
        try:
            return self._column.equals(
                other._column, check_dtypes=check_dtypes
            )
        except TypeError:
            return False

    @_performance_tracking
    def copy(self, name: Hashable = None, deep: bool = False) -> Self:
        """
        Make a copy of this object.

        Parameters
        ----------
        name : object, default None
            Name of index, use original name when None
        deep : bool, default True
            Make a deep copy of the data.
            With ``deep=False`` the original data is used

        Returns
        -------
        New index instance.
        """
        name = self.name if name is None else name
        col = self._column.copy(deep=deep)
        return type(self)._from_column(col, name=name)

    @_performance_tracking
    def astype(self, dtype: Dtype, copy: bool = True) -> Index:
        return super().astype({self.name: cudf.dtype(dtype)}, copy)

    @staticmethod
    def _return_get_indexer_result(result: cupy.ndarray) -> cupy.ndarray:
        return result.astype(np.int64)

    @_performance_tracking
    def get_indexer(self, target, method=None, limit=None, tolerance=None):
        if is_scalar(target):
            raise TypeError("Should be a sequence")

        if method not in {
            None,
            "ffill",
            "bfill",
            "pad",
            "backfill",
            "nearest",
        }:
            raise ValueError(
                f"Invalid fill method. Expecting pad (ffill), backfill (bfill)"
                f" or nearest. Got {method}"
            )

        if not self.is_unique:
            raise ValueError("Cannot get index for a non-unique Index.")

        is_sorted = (
            self.is_monotonic_increasing or self.is_monotonic_decreasing
        )

        if not is_sorted and method is not None:
            raise ValueError(
                "index must be monotonic increasing or decreasing if `method`"
                "is specified."
            )

        needle = as_column(target)
        result = as_column(
            -1,
            length=len(needle),
            dtype=SIZE_TYPE_DTYPE,
        )

        if not len(self):
            return self._return_get_indexer_result(result.values)

        haystack = self._column
        if isinstance(haystack.dtype, CategoricalDtype):
            haystack = haystack._get_decategorized_column()
        try:
            lcol, rcol = _match_join_keys(needle, haystack, "inner")
        except ValueError:
            return self._return_get_indexer_result(result.values)

        with access_columns(lcol, rcol, mode="read", scope="internal") as (
            lcol,
            rcol,
        ):
            left_plc, right_plc = plc.join.inner_join(
                plc.Table([lcol.plc_column]),
                plc.Table([rcol.plc_column]),
                plc.types.NullEquality.EQUAL,
            )
            scatter_map = ColumnBase.create(
                left_plc, dtype=dtype_from_pylibcudf_column(left_plc)
            )
            indices = ColumnBase.create(
                right_plc, dtype=dtype_from_pylibcudf_column(right_plc)
            )
        result = result._scatter_by_column(scatter_map, indices)
        result_series = cudf.Series._from_column(result)

        if method in {"ffill", "bfill", "pad", "backfill"}:
            result_series = _get_indexer_basic(
                index=self,
                positions=result_series,
                method=method,
                target_col=cudf.Series._from_column(needle),
                tolerance=tolerance,
            )
        elif method == "nearest":
            result_series = _get_nearest_indexer(
                index=self,
                positions=result_series,
                target_col=cudf.Series._from_column(needle),
                tolerance=tolerance,
            )
        elif method is not None:
            raise ValueError(
                f"{method=} is unsupported, only supported values are: "
                "{['ffill'/'pad', 'bfill'/'backfill', 'nearest', None]}"
            )

        return self._return_get_indexer_result(result_series.to_cupy())

    @_performance_tracking
    def get_loc(self, key) -> int | slice | cupy.ndarray:
        if not is_scalar(key):
            raise TypeError("Should be a scalar-like")

        is_sorted = (
            self.is_monotonic_increasing or self.is_monotonic_decreasing
        )

        lower_bound, upper_bound, sort_inds = _lexsorted_equal_range(
            self, [as_column([key])], is_sorted
        )

        if lower_bound == upper_bound:
            raise KeyError(key)

        if lower_bound + 1 == upper_bound:
            # Search result is unique, return int.
            return (
                lower_bound
                if is_sorted
                else sort_inds.element_indexing(lower_bound)  # type: ignore[union-attr]
            )

        if is_sorted:
            # In monotonic index, lex search result is continuous. A slice for
            # the range is returned.
            return slice(lower_bound, upper_bound)

        # Not sorted and not unique. Return a boolean mask
        mask = cupy.full(len(self), False)
        true_inds = sort_inds.slice(lower_bound, upper_bound).values  # type: ignore[union-attr]
        mask[true_inds] = True
        return mask

    @_performance_tracking
    def __repr__(self) -> str:
        max_seq_items = pd.get_option("max_seq_items") or len(self)
        mr = 0
        if 2 * max_seq_items < len(self):
            mr = max_seq_items + 1

        if len(self) > mr and mr != 0:
            top = self[0:mr]
            bottom = self[-1 * mr :]
            preprocess = cudf.concat([top, bottom])
        else:
            preprocess = self

        # TODO: Change below usages accordingly to
        # utilize `Index.to_string` once it is implemented
        # related issue : https://github.com/pandas-dev/pandas/issues/35389
        if isinstance(preprocess, CategoricalIndex):
            if preprocess.categories.dtype.kind == "f":
                output = repr(
                    preprocess.astype(DEFAULT_STRING_DTYPE)
                    .to_pandas()
                    .astype(
                        dtype=pd.CategoricalDtype(
                            categories=preprocess.dtype.categories.astype(
                                DEFAULT_STRING_DTYPE
                            ).to_pandas(),
                            ordered=preprocess.dtype.ordered,
                        )
                    )
                )
                break_idx = output.find("ordered=")
                output = (
                    output[:break_idx].replace("'", "") + output[break_idx:]
                )
            else:
                # Too many non-unique categories will cause
                # the output to take too long. In this case, we
                # split the categories into data and categories
                # and generate the repr separately and
                # merge them.
                pd_cats = pd.Categorical(
                    preprocess._column._get_decategorized_column().to_pandas()
                )
                pd_preprocess = pd.CategoricalIndex(pd_cats)
                data_repr = repr(pd_preprocess).split("\n")
                pd_preprocess.dtype._categories = (
                    preprocess.categories.to_pandas()
                )
                pd_preprocess.dtype._ordered = preprocess.dtype.ordered
                cats_repr = repr(pd_preprocess).split("\n")
                output = "\n".join(data_repr[:-1] + cats_repr[-1:])

            output = output.replace("nan", str(cudf.NA))
        elif preprocess._column.nullable:
            if is_dtype_obj_string(self.dtype):
                output = repr(
                    self.to_pandas(
                        nullable=is_pandas_nullable_extension_dtype(self.dtype)
                    )
                )
            else:
                output = repr(self._pandas_repr_compatible().to_pandas())
                # We should remove all the single quotes
                # from the output due to the type-cast to
                # object dtype happening above.
                # Note : The replacing of single quotes has
                # to happen only in case of non-Index[string] types,
                # as we want to preserve single quotes in case
                # of Index[string] and it is valid to have them.
                output = output.replace("'", "")
        else:
            output = repr(preprocess.to_pandas())

        # Fix and correct the class name of the output
        # string by finding first occurrence of "(" in the output
        index_class_split_index = output.find("(")
        output = self.__class__.__name__ + output[index_class_split_index:]

        lines = output.split("\n")

        tmp_meta = lines[-1]
        dtype_index = tmp_meta.rfind(" dtype=")
        prior_to_dtype = tmp_meta[:dtype_index]
        lines = lines[:-1]
        keywords = [f"dtype='{self.dtype}'"]
        if self.name is not None:
            keywords.append(f"name={self.name!r}")
        if "length" in tmp_meta:
            keywords.append(f"length={len(self)}")
        if (
            "freq" in tmp_meta
            and isinstance(self, DatetimeIndex)
            and self._freq is not None
        ):
            keywords.append(f"freq={self._freq.freqstr!r}")
        joined_keywords = ", ".join(keywords)
        lines.append(f"{prior_to_dtype} {joined_keywords})")
        return "\n".join(lines)

    @_performance_tracking
    def __getitem__(self, index):
        res = self._get_elements_from_column(index)
        if isinstance(res, ColumnBase):
            res = Index._from_column(res, name=self.name)
        return res

    @_performance_tracking
    def isna(self) -> cupy.ndarray:
        return self._column.isnull().values

    isnull = isna

    @_performance_tracking
    def notna(self) -> cupy.ndarray:
        return self._column.notnull().values

    notnull = notna

    @cached_property
    @_performance_tracking
    def hasnans(self) -> bool:
        return self._column.has_nulls(include_nan=True)

    @_performance_tracking
    def argsort(
        self,
        axis=0,
        kind="quicksort",
        order=None,
        ascending=True,
        na_position="last",
    ) -> cupy.ndarray:
        """Return the integer indices that would sort the index.

        Parameters
        ----------
        axis : {0 or "index"}
            Has no effect but is accepted for compatibility with numpy.
        kind : {'mergesort', 'quicksort', 'heapsort', 'stable'}, default 'quicksort'
            Choice of sorting algorithm. See :func:`numpy.sort` for more
            information. 'mergesort' and 'stable' are the only stable
            algorithms. Only quicksort is supported in cuDF.
        order : None
            Has no effect but is accepted for compatibility with numpy.
        ascending : bool or list of bool, default True
            If True, sort values in ascending order, otherwise descending.
        na_position : {'first' or 'last'}, default 'last'
            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs
            at the end.

        Returns
        -------
        cupy.ndarray: The indices sorted based on input.
        """
        return super().argsort(
            axis=axis,
            kind=kind,
            order=order,
            ascending=ascending,
            na_position=na_position,
        )

    def repeat(self, repeats, axis=None) -> Self:
        result = self._repeat([self._column], repeats, axis)[0]
        result = ColumnBase.create(result.plc_column, self.dtype)
        return type(self)._from_column(result, name=self.name)

    def __contains__(self, item) -> bool:
        hash(item)
        return item in self._column

    def any(self) -> bool:
        return self._column.any()

    def to_pandas(
        self, *, nullable: bool = False, arrow_type: bool = False
    ) -> pd.Index:
        result = self._column.to_pandas(
            nullable=nullable, arrow_type=arrow_type
        )
        result.name = self.name
        return result

    def to_frame(
        self, index: bool = True, name: Hashable = no_default
    ) -> DataFrame:
        """Create a DataFrame with a column containing this Index

        Parameters
        ----------
        index : boolean, default True
            Set the index of the returned DataFrame as the original Index
        name : object, defaults to index.name
            The passed name should substitute for the index name (if it has
            one).

        Returns
        -------
        DataFrame
            DataFrame containing the original Index data.

        See Also
        --------
        Index.to_series : Convert an Index to a Series.
        Series.to_frame : Convert Series to DataFrame.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index(['Ant', 'Bear', 'Cow'], name='animal')
        >>> idx.to_frame()
               animal
        animal
        Ant       Ant
        Bear     Bear
        Cow       Cow

        By default, the original Index is reused. To enforce a new Index:

        >>> idx.to_frame(index=False)
            animal
        0   Ant
        1  Bear
        2   Cow

        To override the name of the resulting column, specify `name`:

        >>> idx.to_frame(index=False, name='zoo')
            zoo
        0   Ant
        1  Bear
        2   Cow
        """
        return self._to_frame(name=name, index=self if index else None)

    def to_series(
        self, index: Index | None = None, name: Hashable | None = None
    ) -> Series:
        """
        Create a Series with both index and values equal to the index keys.
        Useful with map for returning an indexer based on an index.

        Parameters
        ----------
        index : Index, optional
            Index of resulting Series. If None, defaults to original index.
        name : Hashable, optional
            Name of resulting Series. If None, defaults to name of original
            index.

        Returns
        -------
        Series
            The dtype will be based on the type of the Index values.
        """
        from cudf.core.series import Series

        return Series._from_column(
            self._column,
            index=self.copy(deep=False) if index is None else index,
            name=self.name if name is None else name,
        )

    @_performance_tracking
    def fillna(self, value) -> Self:
        return super()._fillna(value)

    def append(self, other):
        if is_list_like(other):
            to_concat = [self]
            for obj in other:
                if not isinstance(obj, Index):
                    raise TypeError("all inputs must be Index")
                to_concat.append(obj)
        else:
            this = self
            other = ensure_index(other)

            if len(this) == 0 or len(other) == 0:
                # we'll filter out empties later in ._concat
                to_concat = [this, other]
            else:
                if is_mixed_with_object_dtype(this, other):
                    got_dtype = (
                        other.dtype
                        if is_dtype_obj_string(this.dtype)
                        else this.dtype
                    )
                    raise TypeError(
                        f"cudf does not support appending an Index of "
                        f"dtype `{DEFAULT_STRING_DTYPE}` with an Index "
                        f"of dtype `{got_dtype}`, please type-cast "
                        f"either one of them to same dtypes."
                    )

                if (
                    is_dtype_obj_numeric(
                        self._column.dtype, include_decimal=False
                    )
                    and self.dtype != other.dtype
                ):
                    common_type = find_common_type((self.dtype, other.dtype))
                    this = this.astype(common_type)
                    other = other.astype(common_type)
                to_concat = [this, other]

        return self._concat(to_concat)

    @_performance_tracking
    def where(self, cond, other=None, inplace: bool = False) -> Self:
        if getattr(other, "ndim", 1) > 1:
            raise NotImplementedError(
                "Only 1 dimensional other is currently supported"
            )
        cond = as_column(cond)
        if len(cond) != len(self):
            raise ValueError(
                f"cond must be the same length as self ({len(self)})"
            )

        if not is_scalar(other):
            other = as_column(other)

        return self._mimic_inplace(
            self._from_column(
                self._column.where(cond, other, inplace), name=self.name
            ),
            inplace=inplace,
        )

    def _validate_index_level(self, level) -> None:
        """
        Validate index level.

        Same validation done as pandas. Use for methods that accept a level parameter.
        """
        if isinstance(level, int):
            if level < 0 and level != -1:
                raise IndexError(
                    "Too many levels: Index has only 1 level, "
                    f"{level} is not a valid level number"
                )
            if level > 0:
                raise IndexError(
                    f"Too many levels: Index has only 1 level, not {level + 1}"
                )
        elif level != self.name:
            raise KeyError(
                f"Requested level ({level}) does not match index name ({self.name})"
            )

    def unique(self, level: int | None = None) -> Self:
        if level is not None:
            self._validate_index_level(level)
        return type(self)._from_column(self._column.unique(), name=self.name)

    def isin(self, values, level=None) -> cupy.ndarray:
        if level is not None:
            self._validate_index_level(level)
        if is_scalar(values):
            raise TypeError(
                "only list-like objects are allowed to be passed "
                f"to isin(), you passed a {type(values).__name__}"
            )
        return self._column.isin(values).values

    def get_level_values(self, level: Hashable) -> Self:
        """
        Return an Index of values for requested level.

        This is primarily useful to get an individual level of values from a
        MultiIndex, but is provided on Index as well for compatibility.

        Parameters
        ----------
        level : int or str
            It is either the integer position or the name of the level.

        Returns
        -------
        Index
            Calling object, as there is only one level in the Index.

        See Also
        --------
        cudf.MultiIndex.get_level_values : Get values for
            a level of a MultiIndex.

        Notes
        -----
        For Index, level should be 0, since there are no multiple levels.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index(["a", "b", "c"])
        >>> idx.get_level_values(0)
        Index(['a', 'b', 'c'], dtype='str')
        """
        if level == self.name:
            return self
        elif is_integer(level):
            if level != 0:
                raise IndexError(
                    f"Cannot get level: {level} for index with 1 level"
                )
            return self
        else:
            raise KeyError(f"Requested level with name {level} not found")

    @property
    @copy_docstring(StringMethods)
    @_performance_tracking
    def str(self):
        return StringMethods(parent=self)

    @cache
    @_warn_no_dask_cudf
    def __dask_tokenize__(self):
        # We can use caching, because an index is immutable
        return super().__dask_tokenize__()


class RangeIndex(Index):
    """
    Immutable Index implementing a monotonic integer range.

    This is the default index type used by DataFrame and Series
    when no explicit index is provided by the user.

    Parameters
    ----------
    start : int (default: 0), or other range instance
    stop : int (default: 0)
    step : int (default: 1)
    name : object, optional
        Name to be stored in the index.
    dtype : numpy dtype
        Unused, accepted for homogeneity with other index types.
    copy : bool, default False
        Unused, accepted for homogeneity with other index types.

    Attributes
    ----------
    start
    stop
    step

    Methods
    -------
    to_numpy
    to_arrow

    Examples
    --------
    >>> import cudf
    >>> cudf.RangeIndex(0, 10, 1, name="a")
    RangeIndex(start=0, stop=10, step=1, name='a')

    >>> cudf.RangeIndex(range(1, 10, 1), name="a")
    RangeIndex(start=1, stop=10, step=1, name='a')
    """

    _range: range

    @_performance_tracking
    def __init__(
        self,
        start,
        stop=None,
        step=1,
        dtype=None,
        copy=False,
        name=None,
        nan_as_null=no_default,
    ):
        if not is_hashable(name):
            raise ValueError("Name must be a hashable value.")
        self._name = name
        if dtype is not None and cudf.dtype(dtype).kind != "i":
            raise ValueError(f"{dtype=} must be a signed integer type")

        if isinstance(start, range):
            self._range = start
        elif isinstance(start, (pd.RangeIndex, RangeIndex)):
            self._range = range(start.start, start.stop, start.step)
            if name is None:
                self._name = start.name
        else:
            if stop is None:
                start, stop = 0, start
            start = validate_range_arg(start, "start")
            stop = validate_range_arg(stop, "stop")
            if step is not None:
                step = validate_range_arg(step, "step")
            else:
                step = 1
            try:
                self._range = range(start, stop, step)
            except ValueError as err:
                if step == 0:
                    raise ValueError("Step must not be zero.") from err
                raise

    @property
    @_performance_tracking
    def name(self) -> Hashable:
        return self._name

    @name.setter
    @_performance_tracking
    def name(self, value: Hashable) -> None:
        self._name = value

    @property
    def _constructor(self):
        return RangeIndex

    @cached_property
    def inferred_type(self) -> str:
        return "integer"

    @property
    @_performance_tracking
    def _num_rows(self) -> int:
        return len(self)

    @property
    @_performance_tracking
    def _column_names(self) -> tuple[Hashable]:
        return (self.name,)

    @cached_property
    @_performance_tracking
    def _column(self) -> ColumnBase:
        if len(self) > 0:
            return ColumnBase.from_range(self._range).astype(self.dtype)
        else:
            return column_empty(0, dtype=self.dtype)

    @property
    @_performance_tracking
    def _columns(self) -> tuple[ColumnBase]:
        return (self._column,)

    @property
    @_performance_tracking
    def _data(self) -> ColumnAccessor:  # type: ignore[override]  # (RangeIndex materializes data lazily)
        return ColumnAccessor({self.name: self._column}, verify=False)

    @property
    def _column_labels_and_values(
        self,
    ) -> Iterable[tuple[Hashable, ColumnBase]]:
        return zip(self._column_names, self._columns, strict=True)

    @_performance_tracking
    def _as_int_index(self) -> Index:
        # Convert self to an integer index. This method is used to perform ops
        # that are not defined directly on RangeIndex.
        return Index._from_column(self._column, name=self.name)

    @property
    def nlevels(self) -> int:
        """
        Number of levels.
        """
        return 1

    @property
    @_performance_tracking
    def start(self) -> int:
        """
        The value of the `start` parameter (0 if this was not supplied).
        """
        return self._range.start

    @property
    @_performance_tracking
    def stop(self) -> int:
        """
        The value of the stop parameter.
        """
        return self._range.stop

    @property
    @_performance_tracking
    def step(self) -> int:
        """
        The value of the step parameter.
        """
        return self._range.step

    @property
    @_performance_tracking
    def hasnans(self) -> bool:
        return False

    def _pandas_repr_compatible(self, nan_rep=None) -> Self:
        return self

    @_performance_tracking
    def __contains__(self, item: Any) -> bool:
        hash(item)
        if not isinstance(item, (np.floating, np.integer, int, float)):
            return False
        elif isinstance(item, (np.timedelta64, np.datetime64, bool)):
            # Cases that would pass the above check
            return False
        try:
            int_item = int(item)
            return int_item == item and int_item in self._range
        except (ValueError, OverflowError):
            return False

    @_performance_tracking
    def copy(self, name: Hashable | None = None, deep: bool = False) -> Self:
        """
        Make a copy of this object.

        Parameters
        ----------
        name : object optional (default: None), name of index
        deep : Bool (default: False)
            Ignored for RangeIndex

        Returns
        -------
        New RangeIndex instance with same range
        """

        name = self.name if name is None else name

        return RangeIndex(self._range, name=name)

    @_performance_tracking
    def astype(self, dtype: Dtype, copy: bool = True) -> Self:
        dtype = cudf.dtype(dtype)
        if self.dtype == dtype:
            return self if not copy else self.copy()
        return self._as_int_index().astype(dtype, copy=copy)

    def fillna(self, value) -> Self:
        return self.copy()

    @_performance_tracking
    def drop_duplicates(
        self, keep: Literal["first", "last", False] = "first"
    ) -> Self:
        return self

    @_performance_tracking
    def duplicated(
        self, keep: Literal["first", "last", False] = "first"
    ) -> cupy.ndarray:
        return cupy.zeros(len(self), dtype=bool)

    @_performance_tracking
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(start={self.start}, stop={self.stop}"
            f", step={self.step}"
            + (
                f", name={pd.io.formats.printing.default_pprint(self.name)}"
                if self.name is not None
                else ""
            )
            + ")"
        )

    @property
    @_performance_tracking
    def size(self) -> int:
        return len(self)

    @_performance_tracking
    def __len__(self) -> int:
        return len(self._range)

    @_performance_tracking
    def __getitem__(self, index):
        if index is Ellipsis:
            index = slice(None)
        if isinstance(index, slice):
            return type(self)(self._range[index], name=self.name)
        elif is_integer(index):
            new_key = int(index)
            try:
                return self._range[new_key]
            except IndexError as err:
                raise IndexError(
                    f"index {index} is out of bounds for axis 0 with size {len(self)}"
                ) from err
        elif is_scalar(index):
            raise IndexError(
                "only integers, slices (`:`), "
                "ellipsis (`...`), numpy.newaxis (`None`) "
                "and integer or boolean "
                "arrays are valid indices"
            )
        return self._as_int_index()[index]

    def _get_columns_by_label(self, labels) -> Index:
        # used in .sort_values
        if isinstance(labels, Hashable):
            if labels == self.name:
                return self._as_int_index()
        elif is_list_like(labels):
            if list(self.names) == list(labels):
                return self._as_int_index()
        raise KeyError(labels)

    @_performance_tracking
    def equals(self, other) -> bool:
        if isinstance(other, RangeIndex):
            return self._range == other._range
        elif not isinstance(other, Index) or len(self) != len(other):
            return False
        elif not (
            is_dtype_obj_numeric(other.dtype)
            or (
                isinstance(other, CategoricalIndex)
                and is_dtype_obj_numeric(other.categories.dtype)
            )
        ):
            return False
        return self._as_int_index().equals(other)

    @_performance_tracking
    def serialize(self):
        header = {}
        header["index_column"] = {}

        # store metadata values of index separately
        # We don't need to store the GPU buffer for RangeIndexes
        # cuDF only needs to store start/stop and rehydrate
        # during de-serialization
        header["index_column"]["start"] = self.start
        header["index_column"]["stop"] = self.stop
        header["index_column"]["step"] = self.step
        frames = []

        header["name"] = self.name
        header["dtype"] = self.dtype.str
        header["frame_count"] = 0
        return header, frames

    @classmethod
    @_performance_tracking
    def deserialize(cls, header, frames):
        h = header["index_column"]
        name = header["name"]
        start = h["start"]
        stop = h["stop"]
        step = h.get("step", 1)
        dtype = np.dtype(header["dtype"])
        return RangeIndex(
            start=start, stop=stop, step=step, dtype=dtype, name=name
        )

    @property
    @_performance_tracking
    def dtype(self) -> np.dtype:
        """
        `dtype` of the range of values in RangeIndex.

        By default the dtype is 64 bit signed integer. This is configurable
        via `default_integer_bitwidth` as 32 bit in `cudf.options`
        """
        dtype: np.dtype = np.dtype(np.int64)
        return _maybe_convert_to_default_type(dtype)

    @property
    def _dtypes(self) -> Generator[tuple[Hashable, np.dtype], None, None]:
        yield self.name, self.dtype

    @cached_property
    @_performance_tracking
    def values(self) -> cupy.ndarray:
        return cupy.arange(self.start, self.stop, self.step)

    @_performance_tracking
    def to_numpy(
        self,
        dtype: Dtype | None = None,
        copy: bool = False,
        na_value=None,
    ) -> np.ndarray:
        """Convert the RangeIndex to a NumPy array.

        Parameters
        ----------
        dtype : str or :class:`numpy.dtype`, optional
            The dtype to cast the result to. Defaults to the RangeIndex dtype.
        copy : bool, default False
            Whether to ensure that the returned value is not a view on
            another array. Note that ``copy=False`` does not ensure that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensures that
            a copy is made, even if not strictly necessary.
        na_value : Any, default None
            Value to use for missing values. Since ``RangeIndex`` cannot
            contain missing values, this parameter has no effect.

        Returns
        -------
        numpy.ndarray
        """
        return (
            self._to_numpy(dtype, na_value).copy()
            if copy
            else self._to_numpy(dtype, na_value)
        )

    @cache
    def _to_numpy(self, dtype=None, na_value=None) -> np.ndarray:
        return self.to_pandas().to_numpy(dtype=dtype, na_value=na_value)

    @_performance_tracking
    def to_cupy(self) -> cupy.ndarray:
        return self.values

    @_performance_tracking
    def to_pandas(
        self, *, nullable: bool = False, arrow_type: bool = False
    ) -> pd.RangeIndex:
        if nullable:
            raise NotImplementedError(f"{nullable=} is not implemented.")
        elif arrow_type:
            raise NotImplementedError(f"{arrow_type=} is not implemented.")
        return pd.RangeIndex(
            start=self.start,
            stop=self.stop,
            step=self.step,
            dtype=self.dtype,
            name=self.name,
        )

    @_performance_tracking
    def to_arrow(self) -> pa.Array:
        return pa.array(self._range, type=pa.from_numpy_dtype(self.dtype))

    def to_frame(
        self, index: bool = True, name: Hashable = no_default
    ) -> DataFrame:
        return self._as_int_index().to_frame(index=index, name=name)

    @property
    def is_unique(self) -> bool:
        return True

    @cached_property
    @_performance_tracking
    def is_monotonic_increasing(self) -> bool:
        return self.step > 0 or len(self) <= 1

    @cached_property
    @_performance_tracking
    def is_monotonic_decreasing(self):
        return self.step < 0 or len(self) <= 1

    @_performance_tracking
    def memory_usage(self, deep: bool = False) -> int:
        if deep:
            warnings.warn(
                "The deep parameter is ignored and is only included "
                "for pandas compatibility."
            )
        return 0

    def unique(self, level: int | None = None) -> Self:
        # RangeIndex always has unique values
        if level is not None:
            self._validate_index_level(level)
        return self.copy()

    _UFUNC_DUNDER_OPS = {
        "add": ("__add__", "__radd__"),
        "subtract": ("__sub__", "__rsub__"),
        "multiply": ("__mul__", "__rmul__"),
        "floor_divide": ("__floordiv__", "__rfloordiv__"),
    }

    @_performance_tracking
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # numpy operands enter through the ufunc protocol rather than the
        # (reflected) dunder, e.g. ``np.int32(2) * RangeIndex``. Like pandas
        # (``maybe_dispatch_ufunc_to_dunder_op``), route the shift/rescale
        # ufuncs on a raw int through the regular binop path first so they
        # preserve a range where possible.
        if (
            method == "__call__"
            and not kwargs
            and len(inputs) == 2
            and (ops := self._UFUNC_DUNDER_OPS.get(ufunc.__name__)) is not None
        ):
            reflected = inputs[0] is not self
            other = inputs[0] if reflected else inputs[1]
            if (
                isinstance(other, (int, np.integer))
                # np.timedelta64 subclasses np.integer but is not a raw int
                and not isinstance(other, np.timedelta64)
            ) or (
                isinstance(other, np.ndarray)
                and other.ndim == 0
                and other.dtype.kind in "iu"
            ):
                return self._binaryop(other, ops[reflected])
        return self._as_int_index().__array_ufunc__(
            ufunc, method, *inputs, **kwargs
        )

    @_performance_tracking
    def get_indexer(self, target, limit=None, method=None, tolerance=None):
        target_col = as_column(target)
        if method is not None or not is_dtype_obj_numeric(
            target_col.dtype, include_decimal=False
        ):
            # TODO: See if we can implement this without converting to
            # Integer index.
            return self._as_int_index().get_indexer(
                target=target, limit=limit, method=method, tolerance=tolerance
            )

        if self.step > 0:
            start, stop, step = self.start, self.stop, self.step
        else:
            # Reversed
            reverse = self._range[::-1]
            start, stop, step = reverse.start, reverse.stop, reverse.step

        target_array = target_col.values
        locs = target_array - start
        valid = (locs % step == 0) & (locs >= 0) & (target_array < stop)
        locs[~valid] = -1
        locs[valid] = locs[valid] / step

        if step != self.step:
            # Reversed
            locs[valid] = len(self) - 1 - locs[valid]
        return locs

    @_performance_tracking
    def get_loc(self, key) -> int:
        if not is_scalar(key):
            raise TypeError("Should be a scalar-like")
        idx = (key - self.start) / self.step
        idx_int_upper_bound = (self.stop - self.start) // self.step
        if idx > idx_int_upper_bound or idx < 0:
            raise KeyError(key)

        idx_int = (key - self.start) // self.step
        if idx_int != idx:
            raise KeyError(key)
        return idx_int

    @_performance_tracking
    def _union(self, other, sort=None):
        if isinstance(other, RangeIndex):
            # Variable suffixes are of the
            # following notation: *_o -> other, *_s -> self,
            # and *_r -> result
            start_s, step_s = self.start, self.step
            end_s = self.start + self.step * (len(self) - 1)
            start_o, step_o = other.start, other.step
            end_o = other.start + other.step * (len(other) - 1)
            if self.step < 0:
                start_s, step_s, end_s = end_s, -step_s, start_s
            if other.step < 0:
                start_o, step_o, end_o = end_o, -step_o, start_o
            if len(self) == 1 and len(other) == 1:
                step_s = step_o = abs(self.start - other.start)
            elif len(self) == 1:
                step_s = step_o
            elif len(other) == 1:
                step_o = step_s

            # Determine minimum start value of the result.
            start_r = min(start_s, start_o)
            # Determine maximum end value of the result.
            end_r = max(end_s, end_o)
            result = None
            min_step = min(step_o, step_s)

            if ((start_s - start_o) % min_step) == 0:
                # Checking to determine other is a subset of self with
                # equal step size.
                if (
                    step_o == step_s
                    and (start_s - end_o) <= step_s
                    and (start_o - end_s) <= step_s
                ):
                    result = type(self)(start_r, end_r + step_s, step_s)
                # Checking if self is a subset of other with unequal
                # step sizes.
                elif (
                    step_o % step_s == 0
                    and (start_o + step_s >= start_s)
                    and (end_o - step_s <= end_s)
                ):
                    result = type(self)(start_r, end_r + step_s, step_s)
                # Checking if other is a subset of self with unequal
                # step sizes.
                elif (
                    step_s % step_o == 0
                    and (start_s + step_o >= start_o)
                    and (end_s - step_o <= end_o)
                ):
                    result = type(self)(start_r, end_r + step_o, step_o)
            # Checking to determine when the steps are even but one of
            # the inputs spans across is near half or less then half
            # the other input. This case needs manipulation to step
            # size.
            elif (
                step_o == step_s
                and (step_s % 2 == 0)
                and (abs(start_s - start_o) <= step_s / 2)
                and (abs(end_s - end_o) <= step_s / 2)
            ):
                result = type(self)(start_r, end_r + step_s / 2, step_s / 2)
            if result is not None:
                if sort in {None, True} and not result.is_monotonic_increasing:
                    return result.sort_values()
                else:
                    return result

        # If all the above optimizations don't cater to the inputs,
        # we materialize RangeIndexes into integer indexes and
        # then perform `union`.
        return self._try_reconstruct_range_index(
            self._as_int_index()._union(other, sort=sort)
        )

    @_performance_tracking
    def _intersection(self, other, sort=None):
        if not isinstance(other, RangeIndex):
            return self._try_reconstruct_range_index(
                super()._intersection(other, sort=sort)
            )

        if not len(self) or not len(other):
            return RangeIndex(0)

        first = self._range[::-1] if self.step < 0 else self._range
        second = other._range[::-1] if other.step < 0 else other._range

        # check whether intervals intersect
        # deals with in- and decreasing ranges
        int_low = max(first.start, second.start)
        int_high = min(first.stop, second.stop)
        if int_high <= int_low:
            return RangeIndex(0)

        # Method hint: linear Diophantine equation
        # solve intersection problem
        # performance hint: for identical step sizes, could use
        # cheaper alternative
        gcd, s, _ = _extended_gcd(first.step, second.step)

        # check whether element sets intersect
        if (first.start - second.start) % gcd:
            return RangeIndex(0)

        # calculate parameters for the RangeIndex describing the
        # intersection disregarding the lower bounds
        tmp_start = (
            first.start + (second.start - first.start) * first.step // gcd * s
        )
        new_step = first.step * second.step // gcd
        no_steps = -(-(int_low - tmp_start) // abs(new_step))
        new_start = tmp_start + abs(new_step) * no_steps
        new_range = range(new_start, int_high, new_step)
        new_index = RangeIndex(new_range)

        if (self.step < 0 and other.step < 0) is not (new_index.step < 0):
            new_index = new_index[::-1]
        if sort in {None, True}:
            new_index = new_index.sort_values()

        return self._try_reconstruct_range_index(new_index)

    @_performance_tracking
    def difference(self, other, sort=None):
        if isinstance(other, RangeIndex) and self.equals(other):
            return self[:0]._get_reconciled_name_object(other)

        return self._try_reconstruct_range_index(
            super().difference(other, sort=sort)
        )

    def _try_reconstruct_range_index(self, index: Index) -> Self | Index:
        if isinstance(index, RangeIndex) or index.dtype.kind not in "iu":
            return index
        # Evenly spaced values can return a
        # RangeIndex instead of a materialized Index.
        if not index._column.has_nulls():
            if len(index) > 1:
                uniques = cupy.unique(cupy.diff(index.values))
                if len(uniques) == 1 and (diff := uniques[0].get()) != 0:
                    new_range = range(index[0], index[-1] + diff, diff)
                    return type(self)(new_range, name=index.name)
            elif index.dtype == np.dtype(np.int64):
                # 0- and 1-element results are always representable as a
                # range, but only when int64: a narrower dtype means the
                # result was deliberately materialized (e.g. under
                # default_integer_bitwidth) and must be preserved.
                if len(index) == 0:
                    return type(self)(range(0), name=index.name)
                start = int(index[0])
                if start < np.iinfo(np.int64).max:
                    return type(self)(range(start, start + 1), name=index.name)
        return index

    def sort_values(
        self,
        return_indexer=False,
        ascending=True,
        na_position="last",
        key=None,
    ):
        if key is not None:
            raise NotImplementedError("key parameter is not yet implemented.")
        if na_position not in {"first", "last"}:
            raise ValueError(f"invalid na_position: {na_position}")

        sorted_index = self
        if (ascending and self.step < 0) or (not ascending and self.step > 0):
            sorted_index = self[::-1]

        if return_indexer:
            return sorted_index, self.argsort(
                ascending=ascending, na_position=na_position
            )
        else:
            return sorted_index

    @_performance_tracking
    def _gather(self, gather_map, nullify=False, check_bounds=True):
        gather_map = as_column(gather_map)
        return Index._from_column(
            self._column.take(
                gather_map, nullify=nullify, check_bounds=check_bounds
            ),
            name=self.name,
        )

    def repeat(self, repeats, axis=None):
        return self._as_int_index().repeat(repeats, axis)

    def _split(self, splits: list[int]) -> list[Index]:
        return self._as_int_index()._split(splits)

    def _binaryop(self, other, op: str):  # type: ignore[override]
        # Ops with a raw int that shift or exactly rescale the range return
        # a RangeIndex to match pandas (RangeIndex._arith_method and
        # RangeIndex.__floordiv__); everything else materializes.
        if (
            isinstance(other, (np.ndarray, cupy.ndarray))
            and other.ndim == 0
            and other.dtype.kind in "iu"
        ):
            other = other.item()
        if isinstance(other, (int, np.integer)) and not isinstance(
            # np.timedelta64 subclasses np.integer but is not a raw int
            other,
            np.timedelta64,
        ):
            # The shift/rescale ops accept bools as 0/1, like pandas (whose
            # fastpath only checks that the op *result* is integral, which a
            # bool operand satisfies); ``__floordiv__`` matches pandas'
            # ``is_integer`` check in rejecting them.
            is_bool = isinstance(other, (bool, np.bool_))
            other_int = int(other)
            if op in {"__add__", "__radd__"}:
                return RangeIndex(
                    self.start + other_int,
                    self.stop + other_int,
                    self.step,
                    name=self.name,
                )
            elif op == "__sub__":
                return RangeIndex(
                    self.start - other_int,
                    self.stop - other_int,
                    self.step,
                    name=self.name,
                )
            elif op == "__rsub__":
                return RangeIndex(
                    other_int - self.start,
                    other_int - self.stop,
                    -self.step,
                    name=self.name,
                )
            elif op in {"__mul__", "__rmul__"}:
                if other_int != 0:
                    return RangeIndex(
                        self.start * other_int,
                        self.stop * other_int,
                        self.step * other_int,
                        name=self.name,
                    )
                # Rescaling by 0 (or False) would collapse the step; match
                # pandas by materializing to an int64 index of zeros.
                other = other_int
            elif op == "__floordiv__" and other_int != 0 and not is_bool:
                if len(self) == 0 or (
                    self.start % other_int == 0 and self.step % other_int == 0
                ):
                    start = self.start // other_int
                    step = self.step // other_int
                    stop = start + len(self) * step
                    return RangeIndex(start, stop, step or 1, name=self.name)
                elif len(self) == 1:
                    start = self.start // other_int
                    return RangeIndex(start, start + 1, 1, name=self.name)
        return self._as_int_index()._binaryop(other, op=op)

    def join(
        self, other, how="left", level=None, return_indexers=False, sort=False
    ):
        if how in {"left", "right"} or self.equals(other):
            # pandas supports directly merging RangeIndex objects and can
            # intelligently create RangeIndex outputs depending on the type of
            # join. Hence falling back to performing a merge on pd.RangeIndex
            # since the conversion is cheap.
            if isinstance(other, RangeIndex):
                result = self.to_pandas().join(
                    other.to_pandas(),
                    how=how,
                    level=level,
                    return_indexers=return_indexers,
                    sort=sort,
                )
                if return_indexers:
                    return tuple(
                        cudf.from_pandas(result[0]), result[1], result[2]
                    )
                else:
                    return cudf.from_pandas(result)
        return self._as_int_index().join(
            other, how, level, return_indexers, sort
        )

    @_performance_tracking
    def argsort(
        self,
        ascending=True,
        na_position="last",
    ) -> cupy.ndarray:
        if na_position not in {"first", "last"}:
            raise ValueError(f"invalid na_position: {na_position}")
        if (ascending and self.step < 0) or (not ascending and self.step > 0):
            return cupy.arange(len(self) - 1, -1, -1)
        else:
            return cupy.arange(len(self))

    @_performance_tracking
    def searchsorted(
        self,
        value: int,
        side: Literal["left", "right"] = "left",
        ascending: bool = True,
        na_position: Literal["first", "last"] = "last",
    ):
        assert (len(self) <= 1) or (ascending == (self.step > 0)), (
            "Invalid ascending flag"
        )
        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")
        ri = self._range
        if flip := (ri.step < 0):
            ri = ri[::-1]
            shift = int(side == "right")
        else:
            shift = int(side == "left")

        offset = (value - ri.start - shift) // ri.step + 1
        if flip:
            offset = len(ri) - offset
        return max(min(len(ri), offset), 0)

    @_performance_tracking
    def _factorize(
        self, sort: bool, use_na_sentinel: bool
    ) -> tuple[cupy.ndarray, Self]:
        # Specialized: the uniques of a RangeIndex are the RangeIndex
        # itself (or its reverse when ``sort`` flips a decreasing range),
        # not a generic Index built from a materialized column.
        if sort and self.step < 0:
            codes = cupy.arange(len(self) - 1, -1, -1)
            uniques = self[::-1]
        else:
            codes = cupy.arange(len(self), dtype=np.intp)
            uniques = self
        return codes, uniques

    @_performance_tracking
    def where(self, cond, other=None, inplace: bool = False) -> Self | None:
        return self._as_int_index().where(cond, other, inplace)

    @_performance_tracking
    def nunique(self, dropna: bool = True) -> int:
        return len(self)

    @_performance_tracking
    def isna(self) -> cupy.ndarray:
        return cupy.zeros(len(self), dtype=bool)

    isnull = isna

    @_performance_tracking
    def notna(self) -> cupy.ndarray:
        return cupy.ones(len(self), dtype=bool)

    notnull = isna

    @_performance_tracking
    def _minmax(self, meth: str) -> int | float:
        no_steps = len(self) - 1
        if no_steps == -1:
            return np.nan
        elif (meth == "min" and self.step > 0) or (
            meth == "max" and self.step < 0
        ):
            return self.start

        return self.start + self.step * no_steps

    def min(self) -> int | float:
        return self._minmax("min")

    def max(self) -> int | float:
        return self._minmax("max")

    def any(self) -> bool:
        return any(self._range)

    def all(self) -> bool:
        return 0 not in self._range

    def append(self, other):
        if len(other) == 0:
            return self.copy()
        result = self._as_int_index().append(other)
        return self._try_reconstruct_range_index(result)

    def _indices_of(self, value) -> NumericalColumn:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                f"Cannot use {type(value).__name__} to get an index of a "
                f"{type(self).__name__}."
            )
        try:
            return as_column(
                self._range.index(value), length=1, dtype=SIZE_TYPE_DTYPE
            )  # type: ignore[return-value]
        except ValueError:
            return column_empty(0, dtype=SIZE_TYPE_DTYPE)  # type: ignore[return-value]

    @_performance_tracking
    def nans_to_nulls(self) -> Self:
        return self.copy()

    def __pos__(self) -> Self:
        return self.copy()

    def __neg__(self) -> Self:
        rng = range(-self.start, -self.stop, -self.step)
        return type(self)(rng, name=self.name)

    def __abs__(self) -> Self | Index:
        if len(self) == 0 or self.min() >= 0:
            return self.copy()
        elif self.max() <= 0:
            return -self
        else:
            return abs(self._as_int_index())

    def __invert__(self) -> Self:
        if len(self) == 0:
            return self.copy()
        rng = range(~self.start, ~self.stop, -self.step)
        return type(self)(rng, name=self.name)

    def _columns_for_reset_index(
        self, levels: tuple | None
    ) -> Generator[tuple[Any, ColumnBase], None, None]:
        """Return the columns and column names for .reset_index"""
        # We need to explicitly materialize the RangeIndex to a column
        yield "index" if self.name is None else self.name, self._column

    @_warn_no_dask_cudf
    def __dask_tokenize__(self):
        return (type(self), self.start, self.stop, self.step)


_PERIODS_PER_DAY = {
    "s": 86_400,
    "ms": 86_400_000,
    "us": 86_400_000_000,
    "ns": 86_400_000_000_000,
}


def _time_to_micros(value: datetime.time) -> int:
    """Return the number of microseconds since midnight for ``value``."""
    return (
        value.hour * 3600 + value.minute * 60 + value.second
    ) * 1_000_000 + value.microsecond


class DatetimeIndex(Index):
    """
    Immutable , ordered and sliceable sequence of datetime64 data,
    represented internally as int64.

    Parameters
    ----------
    data : array-like (1-dimensional), optional
        Optional datetime-like data to construct index with.
    copy : bool
        Make a copy of input.
    freq : str, optional
        Frequency of the DatetimeIndex
    tz : pytz.timezone or dateutil.tz.tzfile
        This is not yet supported
    ambiguous : 'infer', bool-ndarray, 'NaT', default 'raise'
        This is not yet supported
    name : object
        Name to be stored in the index.
    dayfirst : bool, default False
        If True, parse dates in data with the day first order.
        This is not yet supported
    yearfirst : bool, default False
        If True parse dates in data with the year first order.
        This is not yet supported

    Attributes
    ----------
    year
    month
    day
    hour
    minute
    second
    microsecond
    nanosecond
    date
    time
    dayofyear
    day_of_year
    weekday
    quarter
    freq

    Methods
    -------
    ceil
    floor
    round
    tz_convert
    tz_localize

    Returns
    -------
    DatetimeIndex

    Examples
    --------
    >>> import cudf
    >>> cudf.DatetimeIndex([1, 2, 3, 4], name="a")
    DatetimeIndex(['1970-01-01 00:00:00.000000001',
                   '1970-01-01 00:00:00.000000002',
                   '1970-01-01 00:00:00.000000003',
                   '1970-01-01 00:00:00.000000004'],
                  dtype='datetime64[ns]', name='a')
    """

    timestamp_to_timedelta = {
        plc.DataType(plc.TypeId.TIMESTAMP_NANOSECONDS): plc.DataType(
            plc.TypeId.DURATION_NANOSECONDS
        ),
        plc.DataType(plc.TypeId.TIMESTAMP_MICROSECONDS): plc.DataType(
            plc.TypeId.DURATION_MICROSECONDS
        ),
        plc.DataType(plc.TypeId.TIMESTAMP_SECONDS): plc.DataType(
            plc.TypeId.DURATION_SECONDS
        ),
        plc.DataType(plc.TypeId.TIMESTAMP_MILLISECONDS): plc.DataType(
            plc.TypeId.DURATION_MILLISECONDS
        ),
    }

    _allowed = (
        "days",
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
        "nanoseconds",
    )

    MONTHLY_PERIODS = {
        pd.Timedelta("28 days"),
        pd.Timedelta("29 days"),
        pd.Timedelta("30 days"),
        pd.Timedelta("31 days"),
    }
    YEARLY_PERIODS = {
        pd.Timedelta("365 days"),
        pd.Timedelta("366 days"),
    }

    @_performance_tracking
    def __init__(
        self,
        data=None,
        freq=None,
        tz=None,
        ambiguous: Literal["raise"] = "raise",
        dayfirst: bool = False,
        yearfirst: bool = False,
        dtype=None,
        copy: bool = False,
        name=None,
        nan_as_null=no_default,
    ):
        self._freq = None
        # we should be more strict on what we accept here but
        # we'd have to go and figure out all the semantics around
        # pandas dtindex creation first which.  For now
        # just make sure we handle np.datetime64 arrays
        # and then just dispatch upstream
        was_pd_index = isinstance(data, pd.DatetimeIndex)

        if tz is not None:
            raise NotImplementedError("tz is not yet supported")
        if ambiguous != "raise":
            raise NotImplementedError("ambiguous is not yet supported")
        if dayfirst is not False:
            raise NotImplementedError("dayfirst == True is not yet supported")
        if yearfirst is not False:
            raise NotImplementedError("yearfirst == True is not yet supported")

        if freq is None and isinstance(data, (type(self), pd.DatetimeIndex)):
            freq = data.freq

        name = _getdefault_name(data, name=name)

        data = as_column(data)
        if data.dtype.kind == "b":
            raise ValueError(
                "Boolean data cannot be converted to a DatetimeIndex"
            )
        if dtype is not None:
            dtype = cudf.dtype(dtype)
            if dtype.kind != "M":
                raise TypeError("dtype must be a datetime type")
            elif not isinstance(data.dtype, pd.DatetimeTZDtype):
                if (
                    isinstance(dtype, pd.DatetimeTZDtype)
                    and data.dtype.kind not in "iuf"
                ):
                    # pandas interprets timezone-naive strings/datetimes as
                    # wall time in the target timezone (numeric data stays
                    # interpreted as UTC epoch values)
                    data = data.astype(
                        np.dtype(f"datetime64[{dtype.unit}]")
                    ).tz_localize(str(dtype.tz))
                else:
                    data = data.astype(dtype)
        elif data.dtype.kind != "M":
            if is_dtype_obj_string(data.dtype):
                # Pandas's array_to_datetime falls back to [s] when no
                # concrete (non-NaT) datetime is observed — empty input or
                # an all-NaT/None array (pandas-dev/pandas#55901). Otherwise
                # parsed strings land on [us].
                if len(data) == 0 or data.null_count == len(data):
                    target_unit = "s"
                else:
                    target_unit = "us"
                data = data.astype(np.dtype(f"datetime64[{target_unit}]"))
            else:
                data = data.astype(np.dtype("datetime64[ns]"))

        if copy:
            data = data.copy()

        SingleColumnFrame.__init__(
            self, ColumnAccessor({name: data}, verify=False)
        )
        self._freq = _validate_freq(freq)
        # existing pandas index needs no additional validation
        if not was_pd_index:
            self._validate_freq_against_data(self._freq)

    @_performance_tracking
    def serialize(self):
        header, frames = super().serialize()
        if self._freq is None:
            header["freq"] = None
        elif type(self._freq) is pd.DateOffset:
            # generic offsets have no parseable freqstr; store the kwds
            # plus n/normalize, which .kwds omits
            header["freq"] = {
                "kwds": self._freq.kwds,
                "n": self._freq.n,
                "normalize": self._freq.normalize,
            }
        else:
            header["freq"] = self._freq.freqstr
        return header, frames

    @classmethod
    @_performance_tracking
    def deserialize(cls, header, frames):
        obj = super().deserialize(header, frames)
        if isinstance(header_payload := header.get("freq"), dict):
            freq = pd.DateOffset(
                n=header_payload.get("n", 1),
                normalize=header_payload.get("normalize", False),
                **header_payload["kwds"],
            )
        else:
            freq = header_payload

        obj._freq = _validate_freq(freq)
        return obj

    @_performance_tracking
    def _from_columns_like_self(
        self,
        columns: list[ColumnBase],
        column_names: Iterable[str] | None = None,
    ):
        # Callers (e.g. _gather, sort_values) reorder or filter values, so we
        # cannot assume self._freq still applies to the result. Leave _freq
        # unset (None) and let callers that know the freq is preserved set it
        # explicitly. Matches pandas, which drops freq on these operations.
        return super()._from_columns_like_self(columns, column_names)

    @classmethod
    def _from_data(
        cls, data: MutableMapping, name: Any = no_default, freq: Any = None
    ):
        result = super()._from_data(data, name)
        result._freq = _validate_freq(freq)
        return result

    @classmethod
    @_performance_tracking
    def _from_column(
        cls, column: ColumnBase, *, name: Hashable = None, freq: Any = None
    ) -> Self:
        if column.dtype.kind != "M":
            raise ValueError("column must have a datetime type.")
        result = super()._from_column(column, name=name)
        result._freq = _validate_freq(freq)
        return result

    def __getitem__(self, index):
        value = super().__getitem__(index)
        if isinstance(value, np.datetime64):
            return pd.Timestamp(value)
        return value

    def find_label_range(self, loc: slice) -> slice:
        # For indexing, try to interpret slice arguments as datetime-convertible
        if any(
            not (val is None or isinstance(val, (str, datetime.datetime)))
            for val in (loc.start, loc.stop)
        ):
            raise TypeError(
                "Can only slice DatetimeIndex with a string or datetime objects"
            )
        if isinstance(loc.stop, str) and not (
            self.is_monotonic_increasing or self.is_monotonic_decreasing
        ):
            raise KeyError(
                "Value based partial slicing on non-monotonic DatetimeIndexes "
                "with non-existing keys is not allowed."
            )
        new_slice = slice(
            pd.to_datetime(loc.start) if loc.start is not None else None,
            pd.to_datetime(loc.stop) if loc.stop is not None else None,
            loc.step,
        )
        return super().find_label_range(new_slice)

    @_performance_tracking
    def copy(self, name=None, deep=False):
        idx_copy = super().copy(name=name, deep=deep)
        idx_copy._freq = self._freq
        return idx_copy

    def as_unit(self, unit: str, round_ok: bool = True) -> Self:
        """
        Convert to a dtype with the given unit resolution.

        Currently not implemented.

        Parameters
        ----------
        unit : {'s', 'ms', 'us', 'ns'}
        round_ok : bool, default True
            If False and the conversion requires rounding, raise ValueError.
        """
        raise NotImplementedError("as_unit is currently not implemented")

    def mean(self, *, skipna: bool = True, axis: int | None = 0):
        return self._column.mean(skipna=skipna)

    def std(self, *, skipna: bool = True, axis: int | None = 0, ddof: int = 1):
        return self._column.std(skipna=skipna, ddof=ddof)

    def strftime(self, date_format: str) -> Index:
        """
        Convert to Index using specified date_format.

        Return an Index of formatted strings specified by date_format, which
        supports the same string format as the python standard library.

        Parameters
        ----------
        date_format : str
            Date format string (e.g. "%Y-%m-%d").
        """
        return Index._from_column(
            self._column.strftime(
                date_format,
                dtype=get_dtype_of_same_kind(self.dtype, DEFAULT_STRING_DTYPE),
            ),
            name=self.name,
        )

    @_performance_tracking
    def indexer_between_time(
        self,
        start_time,
        end_time,
        include_start: bool = True,
        include_end: bool = True,
    ) -> cupy.ndarray:
        """
        Return index positions of values between particular times of day.

        Parameters
        ----------
        start_time, end_time : datetime.time, str
            Time passed either as object (datetime.time) or as string in
            appropriate format ("%H:%M", "%H%M", "%I:%M%p", "%I%M%p",
            "%H:%M:%S", "%H%M%S", "%I:%M:%S%p", "%I%M%S%p").
        include_start : bool, default True
        include_end : bool, default True

        Returns
        -------
        cupy.ndarray
            Integer positions of values between ``start_time`` and
            ``end_time``. Returned as a device array, following cuDF
            convention (pandas returns a host ``numpy.ndarray``).
        """
        start_time = to_time(start_time)
        end_time = to_time(end_time)

        # Microseconds since midnight for each value, computed the way pandas'
        # DatetimeIndex._get_time_micros does: take the integer view of the
        # local wall-clock timestamps, reduce it modulo one day, and scale the
        # remainder to microseconds (truncating any finer resolution). Using
        # the local time makes tz-aware indexes match pandas. NaT entries
        # become nulls, filled with the -1 sentinel pandas also uses; it sits
        # below every real time-of-day, so NaT is excluded from an ordinary
        # range and (as in pandas) included by a midnight-wrapping one.
        column = self._column
        if isinstance(column, DatetimeTZColumn):
            column = column._local_time
        unit = column.time_unit
        time_of_day = (
            Index._from_column(column.astype(np.dtype(np.int64)))
            % _PERIODS_PER_DAY[unit]
        )
        if unit == "ns":
            time_micros = time_of_day // 1_000
        elif unit == "ms":
            time_micros = time_of_day * 1_000
        elif unit == "s":
            time_micros = time_of_day * 1_000_000
        else:  # microsecond resolution
            time_micros = time_of_day
        time_micros = time_micros.fillna(-1)

        start_micros = _time_to_micros(start_time)
        end_micros = _time_to_micros(end_time)

        after_start = (
            time_micros >= start_micros
            if include_start
            else time_micros > start_micros
        )
        before_end = (
            time_micros <= end_micros
            if include_end
            else time_micros < end_micros
        )

        if start_micros <= end_micros:
            mask = after_start & before_end
        else:
            # the interval wraps around midnight
            mask = after_start | before_end

        return cupy.flatnonzero(mask)

    @cached_property
    def _constructor(self):
        return DatetimeIndex

    @cached_property
    def inferred_type(self) -> str:
        return "datetime64"

    @cached_property
    def asi8(self) -> cupy.ndarray:
        return self._column.astype(np.dtype(np.int64)).values

    @property
    def inferred_freq(self):
        if self._freq is not None:
            return self._freq

        plc_col = self._column.plc_column
        shifted = plc.copying.shift(
            plc_col, 1, plc.Scalar.from_py(None, dtype=plc_col.type())
        )
        diff = plc.binaryop.binary_operation(
            plc_col,
            shifted,
            plc.binaryop.BinaryOperator.SUB,
            self.timestamp_to_timedelta[plc_col.type()],
        )
        offset = plc.Column(
            diff.type(),
            diff.size() - 1,
            diff.data(),
            diff.null_mask(),
            diff.null_count(),
            1,
            diff.children(),
        )

        uniques = plc.stream_compaction.stable_distinct(
            plc.Table([offset]),
            [0],
            plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
            plc.types.NullEquality.EQUAL,
            plc.types.NanEquality.ALL_EQUAL,
        ).columns()[0]
        if uniques.size() <= 4:
            # inspect a small host copy for special cases
            uniques_host = uniques.to_arrow().to_pylist()
        if uniques.size() == 1:
            # base case of a fixed frequency
            # Arrow to_pylist() returns datetime.timedelta; wrap in pd.Timedelta
            # so .components and comparisons work correctly.
            freq = pd.Timedelta(uniques_host[0])

            # special case of YS-JAN, YS-FEB, etc
            # 365 days is allowable, but if it's the first of the month, pandas
            # has a special freq for it, which would take more work to determine
            # same with specifically 7 day intervals, where pandas has a unique
            # frequency depending on the day of the week corresponding to the days
            if freq == pd.Timedelta("365 days") and not self.is_year_end.all():
                raise NotImplementedError("Can't infer anchored year start")
            elif freq == pd.Timedelta("7 days"):
                raise NotImplementedError("Can't infer anchored week")

            cmps = freq.components

            kwds = {}
            for component in self._allowed:
                if (c := getattr(cmps, component)) != 0:
                    kwds[component] = c

            # not pd.DateOffset(**kwds): a generic pd.DateOffset never
            # compares equal to the fast offsets pandas infers (e.g.
            # pd.DateOffset(days=1) != pd.offsets.Day()) and its freqstr
            # is not parseable, so single-unit offsets must be converted
            # to their fast pandas equivalents
            return cudf.DateOffset(**kwds)._maybe_as_fast_pandas_offset()

        # maximum unique count supported is months with 4 unique lengths
        # bail above that for now
        elif 1 < uniques.size() <= 4:
            # length between 1 and 4, small host copy
            if all(x in self.YEARLY_PERIODS for x in uniques_host):
                # Could be year end or could be an anchored year end
                if self.is_year_end.all():
                    return pd.tseries.frequencies.to_offset("YE-DEC")
                else:
                    raise NotImplementedError()
            elif all(x in self.MONTHLY_PERIODS for x in uniques_host):
                if self.is_month_end.all():
                    return pd.tseries.frequencies.to_offset("ME")
            else:
                raise NotImplementedError
        else:
            return None
        return None

    def _get_slice_frequency(self, slc=None):
        if self._freq is None:
            # match pandas: slicing an index without a cached freq
            # produces a result with freq=None
            return None
        if slc.step in (1, None):
            # no change in freq
            return self._freq
        if slc == slice(None, None, None):
            return self._freq
        else:
            if slc:
                # fastpath: dont introspect
                return slc.step * self._freq
            else:
                return self.inferred_freq

    def _validate_freq_against_data(self, freq) -> None:
        if freq is None:
            return
        unique_vals = cudf.Series._from_column(
            self.to_series().diff()._column.unique()
        )
        if freq == pd.DateOffset(months=1):
            possible = pd.Series(list(self.MONTHLY_PERIODS | {pd.NaT}))
            if unique_vals.isin(possible).sum() != len(unique_vals):
                raise ValueError(
                    f"Inferred frequency from passed values does not "
                    f"conform to passed frequency {freq.freqstr}"
                )
        elif freq == pd.DateOffset(years=1):
            possible = pd.Series(list(self.YEARLY_PERIODS | {pd.NaT}))
            if unique_vals.isin(possible).sum() != len(unique_vals):
                raise ValueError(
                    f"Inferred frequency from passed values does not "
                    f"conform to passed frequency {freq.freqstr}"
                )
        else:
            if len(unique_vals) > 2 or (
                len(unique_vals) == 2 and unique_vals[1].value != freq.nanos
            ):
                raise ValueError(
                    f"Inferred frequency from passed values does not "
                    f"conform to passed frequency {freq.freqstr}"
                )

    @property
    def freq(self) -> pd.tseries.offsets.BaseOffset | None:
        return self._freq

    @freq.setter
    def freq(self, value) -> None:
        new_freq = _validate_freq(value)
        self._validate_freq_against_data(new_freq)
        self._freq = new_freq

    @property
    def freqstr(self) -> str:
        raise NotImplementedError("freqstr is currently not implemented")

    @property
    def resolution(self) -> str:
        """
        Returns day, hour, minute, second, millisecond or microsecond
        """
        raise NotImplementedError("resolution is currently not implemented")

    @property
    def unit(self) -> str:
        return self._column.time_unit

    @property
    def tz(self) -> tzinfo | None:
        """
        Return the timezone.

        Returns
        -------
        datetime.tzinfo or None
            Returns None when the array is tz-naive.
        """
        return self._column.tz

    @property
    def tzinfo(self) -> tzinfo | None:
        """
        Alias for tz attribute
        """
        return self.tz

    def to_pydatetime(self) -> np.ndarray:
        """
        Return an ndarray of ``datetime.datetime`` objects.

        Returns
        -------
        numpy.ndarray
            An ndarray of ``datetime.datetime`` objects.
        """
        return self.to_pandas().to_pydatetime()

    def to_julian_date(self) -> Index:
        return Index._from_column(
            self._column.to_julian_date(), name=self.name
        )

    def to_period(self, freq=None) -> pd.PeriodIndex:
        return self.to_pandas().to_period(freq=freq)

    def normalize(self) -> Self:
        """
        Convert times to midnight.

        Currently not implemented.
        """
        return type(self)._from_column(
            self._column.normalize(), name=self.name
        )

    @cached_property
    def time(self) -> np.ndarray:
        """
        Returns numpy array of ``datetime.time`` objects.

        The time part of the Timestamps.
        """
        return self.to_pandas().time

    @cached_property
    def timetz(self) -> np.ndarray:
        """
        Returns numpy array of ``datetime.time`` objects with timezones.

        The time part of the Timestamps.
        """
        return self.to_pandas().timetz

    @cached_property
    def date(self) -> np.ndarray:
        """
        Returns numpy array of python ``datetime.date`` objects.

        Namely, the date part of Timestamps without time and
        timezone information.
        """
        return self.to_pandas().date

    @property
    def is_month_start(self) -> cupy.ndarray:
        """
        Booleans indicating if dates are the first day of the month.
        """
        # .is_month_start is already a cached_property
        return self._column.is_month_start.values

    @property
    def is_month_end(self) -> cupy.ndarray:
        """
        Booleans indicating if dates are the last day of the month.
        """
        # .is_month_end is already a cached_property
        return self._column.is_month_end.values

    @property
    def is_quarter_end(self) -> cupy.ndarray:
        """
        Booleans indicating if dates are the last day of the quarter.
        """
        # .is_quarter_end is already a cached_property
        return self._column.is_quarter_end.values

    @property
    def is_quarter_start(self) -> cupy.ndarray:
        """
        Booleans indicating if dates are the start day of the quarter.
        """
        # .is_quarter_start is already a cached_property
        return self._column.is_quarter_start.values

    @property
    def is_year_end(self) -> cupy.ndarray:
        """
        Booleans indicating if dates are the last day of the year.
        """
        # .is_year_end is already a cached_property
        return self._column.is_year_end.values

    @property
    def is_year_start(self) -> cupy.ndarray:
        """
        Booleans indicating if dates are the first day of the year.
        """
        # .is_year_start is already a cached_property
        return self._column.is_year_start.values

    @property
    def is_normalized(self) -> bool:
        """
        Returns True if all of the dates are at midnight ("no time")
        """
        # .is_normalized is already a cached_property
        return self._column.is_normalized

    @property
    def days_in_month(self) -> Index:
        """
        Get the total number of days in the month that the date falls on.
        """
        # .days_in_month is already a cached_property
        return Index._from_column(self._column.days_in_month, name=self.name)

    daysinmonth = days_in_month

    @property
    def day_of_week(self) -> Index:
        """
        Get the day of week that the date falls on.
        """
        # .day_of_week is already a cached_property
        return Index._from_column(self._column.day_of_week, name=self.name)

    @property
    @_performance_tracking
    def year(self) -> Index:
        """
        The year of the datetime.

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> datetime_index = cudf.Index(pd.date_range("2000-01-01",
        ...             periods=3, freq="YE"))
        >>> datetime_index
        DatetimeIndex(['2000-12-31', '2001-12-31', '2002-12-31'], dtype='datetime64[us]', freq='YE-DEC')
        >>> datetime_index.year
        Index([2000, 2001, 2002], dtype='int32')
        """
        # .year is already a cached_property
        return Index._from_column(self._column.year, name=self.name)

    @property
    @_performance_tracking
    def month(self) -> Index:
        """
        The month as January=1, December=12.

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> datetime_index = cudf.Index(pd.date_range("2000-01-01",
        ...             periods=3, freq="ME"))
        >>> datetime_index
        DatetimeIndex(['2000-01-31', '2000-02-29', '2000-03-31'], dtype='datetime64[us]', freq='ME')
        >>> datetime_index.month
        Index([1, 2, 3], dtype='int32')
        """
        # .month is already a cached_property
        return Index._from_column(self._column.month, name=self.name)

    @property
    @_performance_tracking
    def day(self) -> Index:
        """
        The day of the datetime.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2000-01-01",
        ...             periods=3, freq="D"))
        >>> datetime_index
        DatetimeIndex(['2000-01-01', '2000-01-02', '2000-01-03'], dtype='datetime64[us]', freq='D')
        >>> datetime_index.day
        Index([1, 2, 3], dtype='int32')
        """
        # .day is already a cached_property
        return Index._from_column(self._column.day, name=self.name)

    @property
    @_performance_tracking
    def hour(self) -> Index:
        """
        The hours of the datetime.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2000-01-01",
        ...             periods=3, freq="h"))
        >>> datetime_index
        DatetimeIndex(['2000-01-01 00:00:00', '2000-01-01 01:00:00',
                    '2000-01-01 02:00:00'],
                    dtype='datetime64[us]', freq='h')
        >>> datetime_index.hour
        Index([0, 1, 2], dtype='int32')
        """
        # .hour is already a cached_property
        return Index._from_column(self._column.hour, name=self.name)

    @property
    @_performance_tracking
    def minute(self) -> Index:
        """
        The minutes of the datetime.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2000-01-01",
        ...             periods=3, freq="min"))
        >>> datetime_index
        DatetimeIndex(['2000-01-01 00:00:00', '2000-01-01 00:01:00',
                    '2000-01-01 00:02:00'],
                    dtype='datetime64[us]', freq='min')
        >>> datetime_index.minute
        Index([0, 1, 2], dtype='int32')
        """
        # .minute is already a cached_property
        return Index._from_column(self._column.minute, name=self.name)

    @property
    @_performance_tracking
    def second(self) -> Index:
        """
        The seconds of the datetime.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2000-01-01",
        ...             periods=3, freq="s"))
        >>> datetime_index
        DatetimeIndex(['2000-01-01 00:00:00', '2000-01-01 00:00:01',
                    '2000-01-01 00:00:02'],
                    dtype='datetime64[us]', freq='s')
        >>> datetime_index.second
        Index([0, 1, 2], dtype='int32')
        """
        # .second is already a cached_property
        return Index._from_column(self._column.second, name=self.name)

    @property
    @_performance_tracking
    def microsecond(self) -> Index:
        """
        The microseconds of the datetime.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2000-01-01",
        ...             periods=3, freq="us"))
        >>> datetime_index
        DatetimeIndex([       '2000-01-01 00:00:00', '2000-01-01 00:00:00.000001',
               '2000-01-01 00:00:00.000002'],
              dtype='datetime64[us]', freq='us')
        >>> datetime_index.microsecond
        Index([0, 1, 2], dtype='int32')
        """
        # libcudf's MICROSECOND component excludes the millisecond
        # component, but pandas folds milliseconds into microsecond. Match
        # pandas (and Series.dt.microsecond) by adding millisecond * 1000.
        micro = self._column.microsecond
        extra = self._column.millisecond.astype(np.dtype(np.int32)) * np.int32(
            1000
        )
        return Index._from_column(micro + extra, name=self.name)

    @property
    @_performance_tracking
    def nanosecond(self) -> Index:
        """
        The nanoseconds of the datetime.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2000-01-01",
        ...             periods=3, freq="ns"))
        >>> datetime_index
        DatetimeIndex([          '2000-01-01 00:00:00',
                       '2000-01-01 00:00:00.000000001',
                       '2000-01-01 00:00:00.000000002'],
                      dtype='datetime64[ns]', freq='ns')
        >>> datetime_index.nanosecond
        Index([0, 1, 2], dtype='int32')
        """
        # .nanosecond is already a cached_property
        return Index._from_column(self._column.nanosecond, name=self.name)

    @property
    @_performance_tracking
    def weekday(self) -> Index:
        """
        The day of the week with Monday=0, Sunday=6.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2016-12-31",
        ...     "2017-01-08", freq="D"))
        >>> datetime_index
        DatetimeIndex(['2016-12-31', '2017-01-01', '2017-01-02', '2017-01-03',
                    '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07',
                    '2017-01-08'],
                    dtype='datetime64[us]', freq='D')
        >>> datetime_index.weekday
        Index([5, 6, 0, 1, 2, 3, 4, 5, 6], dtype='int32')
        """
        # .weekday is already a cached_property
        return Index._from_column(self._column.weekday, name=self.name)

    @property
    @_performance_tracking
    def dayofweek(self) -> Index:
        """
        The day of the week with Monday=0, Sunday=6.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2016-12-31",
        ...     "2017-01-08", freq="D"))
        >>> datetime_index
        DatetimeIndex(['2016-12-31', '2017-01-01', '2017-01-02', '2017-01-03',
                    '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07',
                    '2017-01-08'],
                    dtype='datetime64[us]', freq='D')
        >>> datetime_index.dayofweek
        Index([5, 6, 0, 1, 2, 3, 4, 5, 6], dtype='int32')
        """
        # .weekday is already a cached_property
        return Index._from_column(self._column.weekday, name=self.name)

    @property
    @_performance_tracking
    def dayofyear(self) -> Index:
        """
        The day of the year, from 1-365 in non-leap years and
        from 1-366 in leap years.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2016-12-31",
        ...     "2017-01-08", freq="D"))
        >>> datetime_index
        DatetimeIndex(['2016-12-31', '2017-01-01', '2017-01-02', '2017-01-03',
                    '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07',
                    '2017-01-08'],
                    dtype='datetime64[us]', freq='D')
        >>> datetime_index.dayofyear
        Index([366, 1, 2, 3, 4, 5, 6, 7, 8], dtype='int16')
        """
        # .day_of_year is already a cached_property
        return Index._from_column(self._column.day_of_year, name=self.name)

    @property
    @_performance_tracking
    def day_of_year(self) -> Index:
        """
        The day of the year, from 1-365 in non-leap years and
        from 1-366 in leap years.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2016-12-31",
        ...     "2017-01-08", freq="D"))
        >>> datetime_index
        DatetimeIndex(['2016-12-31', '2017-01-01', '2017-01-02', '2017-01-03',
                    '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07',
                    '2017-01-08'],
                    dtype='datetime64[us]', freq='D')
        >>> datetime_index.day_of_year
        Index([366, 1, 2, 3, 4, 5, 6, 7, 8], dtype='int16')
        """
        # .day_of_year is already a cached_property
        return Index._from_column(self._column.day_of_year, name=self.name)

    @property
    @_performance_tracking
    def is_leap_year(self) -> cupy.ndarray:
        """
        Boolean indicator if the date belongs to a leap year.

        A leap year is a year, which has 366 days (instead of 365) including
        29th of February as an intercalary day. Leap years are years which are
        multiples of four with the exception of years divisible by 100 but not
        by 400.

        Returns
        -------
        ndarray
        Booleans indicating if dates belong to a leap year.
        """
        # .is_leap_year is already a cached_property
        res = self._column.is_leap_year.fillna(False)
        return cupy.asarray(res)

    @property
    @_performance_tracking
    def quarter(self) -> Index:
        """
        Integer indicator for which quarter of the year the date belongs in.

        There are 4 quarters in a year. With the first quarter being from
        January - March, second quarter being April - June, third quarter
        being July - September and fourth quarter being October - December.

        Returns
        -------
        Index
        Integer indicating which quarter the date belongs to.

        Examples
        --------
        >>> import cudf
        >>> gIndex = cudf.DatetimeIndex(["2020-05-31 08:00:00",
        ...    "1999-12-31 18:40:00"])
        >>> gIndex.quarter
        Index([2, 4], dtype='int8')
        """
        # .quarter is already a cached_property
        return Index._from_column(
            self._column.quarter.astype(np.dtype(np.int8))
        )

    @_performance_tracking
    def day_name(self, locale: str | None = None) -> Index:
        """
        Return the day names. Currently supports English locale only.

        Examples
        --------
        >>> import cudf
        >>> datetime_index = cudf.date_range("2016-12-31", "2017-01-08", freq="24h")
        >>> datetime_index
        DatetimeIndex(['2016-12-31', '2017-01-01', '2017-01-02', '2017-01-03',
                       '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07',
                       '2017-01-08'],
                      dtype='datetime64[us]', freq='24h')
        >>> datetime_index.day_name()
        Index(['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday',
               'Friday', 'Saturday', 'Sunday'],
              dtype='str')
        """
        day_names = self._column.get_day_names(locale)
        return Index._from_column(day_names, name=self.name)

    @_performance_tracking
    def month_name(self, locale: str | None = None) -> Index:
        """
        Return the month names. Currently supports English locale only.

        Examples
        --------
        >>> import cudf
        >>> datetime_index = cudf.date_range("2017-12-30", periods=6, freq='W')
        >>> datetime_index
        DatetimeIndex(['2017-12-30', '2018-01-06', '2018-01-13', '2018-01-20',
                    '2018-01-27', '2018-02-03'],
                      dtype='datetime64[us]', freq='W')
        >>> datetime_index.month_name()
        Index(['December', 'January', 'January', 'January', 'January', 'February'], dtype='str')
        """
        month_names = self._column.get_month_names(locale)
        return Index._from_column(month_names, name=self.name)

    @_performance_tracking
    def isocalendar(self) -> DataFrame:
        """
        Returns a DataFrame with the year, week, and day
        calculated according to the ISO 8601 standard.

        Returns
        -------
        DataFrame
        with columns year, week and day

        Examples
        --------
        >>> gIndex = cudf.DatetimeIndex(["2020-05-31 08:00:00",
        ...    "1999-12-31 18:40:00"])
        >>> gIndex.isocalendar()
                             year  week  day
        2020-05-31 08:00:00  2020    22    7
        1999-12-31 18:40:00  1999    52    5
        """
        ca = ColumnAccessor(self._column.isocalendar(), verify=False)
        return cudf.DataFrame._from_data(ca, index=self)

    @_performance_tracking
    def to_pandas(
        self, *, nullable: bool = False, arrow_type: bool = False
    ) -> pd.DatetimeIndex:
        result = super().to_pandas(nullable=nullable, arrow_type=arrow_type)
        if not arrow_type and self._freq is not None:
            # Prefer pandas's inferred_freq because the cached self._freq may
            # not conform (e.g. after deserialization or external assignment)
            # and pandas validates the assignment against the index values.
            # Fall back to the cached freq when inference is impossible
            # (empty / single-element indexes), so resample round-trips
            # preserve `freq` to match pandas.
            inferred = result.inferred_freq
            if inferred is None:
                try:
                    result.freq = self._freq
                except ValueError:
                    pass
            else:
                result.freq = inferred
        return result

    @_performance_tracking
    def ceil(self, freq: str) -> Self:
        """
        Perform ceil operation on the data to the specified freq.

        Parameters
        ----------
        freq : str
            One of ["D", "h", "min", "S", "L", "ms", "U", "us", "N"].
            Must be a fixed frequency like 'S' (second) not 'ME' (month end).
            See `frequency aliases <https://pandas.pydata.org/docs/\
                user_guide/timeseries.html#timeseries-offset-aliases>`__
            for more details on these aliases.

        Returns
        -------
        DatetimeIndex
            Index of the same type for a DatetimeIndex

        Examples
        --------
        >>> import cudf
        >>> gIndex = cudf.DatetimeIndex([
        ...     "2020-05-31 08:05:42",
        ...     "1999-12-31 18:40:30",
        ... ])
        >>> gIndex.ceil("min")
        DatetimeIndex(['2020-05-31 08:06:00', '1999-12-31 18:41:00'], dtype='datetime64[us]')
        """
        return type(self)._from_column(self._column.ceil(freq), name=self.name)

    @_performance_tracking
    def floor(self, freq: str) -> Self:
        """
        Perform floor operation on the data to the specified freq.

        Parameters
        ----------
        freq : str
            One of ["D", "h", "min", "S", "L", "ms", "U", "us", "N"].
            Must be a fixed frequency like 'S' (second) not 'ME' (month end).
            See `frequency aliases <https://pandas.pydata.org/docs/\
                user_guide/timeseries.html#timeseries-offset-aliases>`__
            for more details on these aliases.

        Returns
        -------
        DatetimeIndex
            Index of the same type for a DatetimeIndex

        Examples
        --------
        >>> import cudf
        >>> gIndex = cudf.DatetimeIndex([
        ...     "2020-05-31 08:59:59",
        ...     "1999-12-31 18:44:59",
        ... ])
        >>> gIndex.floor("min")
        DatetimeIndex(['2020-05-31 08:59:00', '1999-12-31 18:44:00'], dtype='datetime64[us]')
        """
        return type(self)._from_column(
            self._column.floor(freq), name=self.name
        )

    @_performance_tracking
    def round(self, freq: str) -> Self:
        """
        Perform round operation on the data to the specified freq.

        Parameters
        ----------
        freq : str
            One of ["D", "h", "min", "S", "L", "ms", "U", "us", "N"].
            Must be a fixed frequency like 'S' (second) not 'ME' (month end).
            See `frequency aliases <https://pandas.pydata.org/docs/\
                user_guide/timeseries.html#timeseries-offset-aliases>`__
            for more details on these aliases.

        Returns
        -------
        DatetimeIndex
            Index containing rounded datetimes.

        Examples
        --------
        >>> import cudf
        >>> dt_idx = cudf.Index([
        ...     "2001-01-01 00:04:45",
        ...     "2001-01-01 00:04:58",
        ...     "2001-01-01 00:05:04",
        ... ], dtype="datetime64[ns]")
        >>> dt_idx
        DatetimeIndex(['2001-01-01 00:04:45', '2001-01-01 00:04:58',
                       '2001-01-01 00:05:04'],
                      dtype='datetime64[ns]')
        >>> dt_idx.round('h')
        DatetimeIndex(['2001-01-01', '2001-01-01', '2001-01-01'], dtype='datetime64[ns]')
        >>> dt_idx.round('min')
        DatetimeIndex(['2001-01-01 00:05:00', '2001-01-01 00:05:00', '2001-01-01 00:05:00'], dtype='datetime64[ns]')
        """
        return type(self)._from_column(
            self._column.round(freq), name=self.name
        )

    def tz_localize(
        self,
        tz: str | None,
        ambiguous: Literal["NaT"] = "NaT",
        nonexistent: Literal["NaT"] = "NaT",
    ) -> Self:
        """
        Localize timezone-naive data to timezone-aware data.

        Parameters
        ----------
        tz : str
            Timezone to convert timestamps to.

        Returns
        -------
        DatetimeIndex containing timezone aware timestamps.

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> tz_naive = cudf.date_range('2018-03-01 09:00', periods=3, freq='24h')
        >>> tz_aware = tz_naive.tz_localize("America/New_York")
        >>> tz_aware
        DatetimeIndex(['2018-03-01 09:00:00-05:00', '2018-03-02 09:00:00-05:00',
                       '2018-03-03 09:00:00-05:00'],
                      dtype='datetime64[us, America/New_York]', freq='24h')

        Ambiguous or nonexistent datetimes are converted to NaT.

        >>> s = cudf.to_datetime(cudf.Series(['2018-10-28 01:20:00',
        ...                                   '2018-10-28 02:36:00',
        ...                                   '2018-10-28 03:46:00']))
        >>> s.dt.tz_localize("CET")
        0    2018-10-28 01:20:00.000000
        1                           NaT
        2    2018-10-28 03:46:00.000000
        dtype: datetime64[us, CET]

        Notes
        -----
        'NaT' is currently the only supported option for the
        ``ambiguous`` and ``nonexistent`` arguments. Any
        ambiguous or nonexistent timestamps are converted
        to 'NaT'.
        """
        result_col = self._column.tz_localize(tz, ambiguous, nonexistent)
        return DatetimeIndex._from_column(
            result_col, name=self.name, freq=self._freq
        )

    def tz_convert(self, tz: str | None) -> Self:
        """
        Convert tz-aware datetimes from one time zone to another.

        Parameters
        ----------
        tz : str
            Time zone for time. Corresponding timestamps would be converted
            to this time zone of the Datetime Array/Index.
            A `tz` of None will convert to UTC and remove the timezone
            information.

        Returns
        -------
        DatetimeIndex containing timestamps corresponding to the timezone
        `tz`.

        Examples
        --------
        >>> import cudf
        >>> dti = cudf.date_range('2018-03-01 09:00', periods=3, freq='24h')
        >>> dti = dti.tz_localize("America/New_York")
        >>> dti
        DatetimeIndex(['2018-03-01 09:00:00-05:00', '2018-03-02 09:00:00-05:00',
                       '2018-03-03 09:00:00-05:00'],
                      dtype='datetime64[us, America/New_York]', freq='24h')
        >>> dti.tz_convert("Europe/London")
        DatetimeIndex(['2018-03-01 14:00:00+00:00',
                       '2018-03-02 14:00:00+00:00',
                       '2018-03-03 14:00:00+00:00'],
                      dtype='datetime64[us, Europe/London]')
        """
        result_col = self._column.tz_convert(tz)
        return DatetimeIndex._from_column(result_col, name=self.name)


class TimedeltaIndex(Index):
    """
    Immutable, ordered and sliceable sequence of timedelta64 data,
    represented internally as int64.

    Parameters
    ----------
    data : array-like (1-dimensional), optional
        Optional datetime-like data to construct index with.
    copy : bool
        Make a copy of input.
    freq : str, optional
        This is not yet supported
    dtype : str or :class:`numpy.dtype`, optional
        Data type for the output Index. If not specified, the
        default dtype will be ``timedelta64[ns]``.
    name : object
        Name to be stored in the index.

    Attributes
    ----------
    days
    seconds
    microseconds
    nanoseconds
    components
    inferred_freq

    Methods
    -------
    None

    Returns
    -------
    TimedeltaIndex

    Examples
    --------
    >>> import cudf
    >>> cudf.TimedeltaIndex([1132223, 2023232, 342234324, 4234324],
    ...     dtype="timedelta64[ns]")
    TimedeltaIndex(['0 days 00:00:00.001132223', '0 days 00:00:00.002023232',
                    '0 days 00:00:00.342234324', '0 days 00:00:00.004234324'],
                  dtype='timedelta64[ns]')
    >>> cudf.TimedeltaIndex([1, 2, 3, 4], dtype="timedelta64[s]",
    ...     name="delta-index")
    TimedeltaIndex(['0 days 00:00:01', '0 days 00:00:02', '0 days 00:00:03',
                    '0 days 00:00:04'],
                  dtype='timedelta64[s]', name='delta-index')
    """

    @_performance_tracking
    def __init__(
        self,
        data=None,
        freq=None,
        dtype=None,
        copy: bool = False,
        name=None,
        nan_as_null=no_default,
    ):
        if freq is not None:
            raise NotImplementedError("freq is not yet supported")

        name = _getdefault_name(data, name=name)
        col = as_column(data)
        if is_dtype_obj_string(col.dtype):
            # String -> Timedelta parsing via astype isn't rigorous enough yet
            # to cover cudf.pandas test cases, go through pandas instead.
            col = as_column(pd.to_timedelta(data))

        if dtype is not None:
            dtype = cudf.dtype(dtype)
            if dtype.kind != "m":
                raise TypeError("dtype must be a timedelta type")
            col = col.astype(dtype)
        elif col.dtype.kind != "m":
            if is_dtype_obj_string(col.dtype):
                col = col.astype(np.dtype("timedelta64[us]"))
            else:
                col = col.astype(np.dtype("timedelta64[ns]"))

        if copy:
            col = col.copy()

        SingleColumnFrame.__init__(
            self, ColumnAccessor({name: col}, verify=False)
        )

    @classmethod
    @_performance_tracking
    def _from_column(
        cls, column: ColumnBase, *, name: Hashable = None, freq: Any = None
    ) -> Self:
        if column.dtype.kind != "m":
            raise ValueError("column must have a timedelta type.")
        return super()._from_column(column, name=name)

    def __getitem__(self, index):
        value = super().__getitem__(index)
        if isinstance(value, np.timedelta64):
            return pd.Timedelta(value)
        return value

    def as_unit(self, unit: str, round_ok: bool = True) -> Self:
        """
        Convert to a dtype with the given unit resolution.

        Currently not implemented.

        Parameters
        ----------
        unit : {'s', 'ms', 'us', 'ns'}
        round_ok : bool, default True
            If False and the conversion requires rounding, raise ValueError.
        """
        raise NotImplementedError("as_unit is currently not implemented")

    @cached_property
    def _constructor(self):
        return TimedeltaIndex

    @cached_property
    def inferred_type(self) -> str:
        return "timedelta64"

    @property
    def freq(self) -> DateOffset | None:
        raise NotImplementedError("freq is currently not implemented")

    @property
    def freqstr(self) -> str:
        raise NotImplementedError("freqstr is currently not implemented")

    @property
    def resolution(self) -> str:
        """
        Returns day, hour, minute, second, millisecond or microsecond
        """
        raise NotImplementedError("resolution is currently not implemented")

    @property
    def unit(self) -> str:
        return self._column.time_unit

    def to_pytimedelta(self) -> np.ndarray:
        """
        Return an ndarray of ``datetime.timedelta`` objects.

        Returns
        -------
        numpy.ndarray
            An ndarray of ``datetime.timedelta`` objects.
        """
        return self.to_pandas().to_pytimedelta()

    @cached_property
    def asi8(self) -> cupy.ndarray:
        return self._column.astype(np.dtype(np.int64)).values

    def sum(self, *, skipna: bool = True, axis: int | None = 0):
        return self._column.sum(skipna=skipna)

    def mean(self, *, skipna: bool = True, axis: int | None = 0):
        return self._column.mean(skipna=skipna)

    def median(self, *, skipna: bool = True, axis: int | None = 0):
        return self._column.median(skipna=skipna)

    def std(self, *, skipna: bool = True, axis: int | None = 0, ddof: int = 1):
        return self._column.std(skipna=skipna, ddof=ddof)

    def total_seconds(self) -> Index:
        """
        Return total duration of each element expressed in seconds.

        This method is currently not implemented.
        """
        return Index._from_column(self._column.total_seconds(), name=self.name)

    def ceil(self, freq: str) -> Self:
        """
        Ceil to the specified resolution.

        This method is currently not implemented.
        """
        return type(self)._from_column(self._column.ceil(freq), name=self.name)

    def floor(self, freq: str) -> Self:
        """
        Floor to the specified resolution.

        This method is currently not implemented.
        """
        return type(self)._from_column(
            self._column.floor(freq), name=self.name
        )

    def round(self, freq: str) -> Self:
        """
        Round to the specified resolution.

        This method is currently not implemented.
        """
        return type(self)._from_column(
            self._column.round(freq), name=self.name
        )

    @property
    @_performance_tracking
    def days(self) -> Index:
        """
        Number of days for each element.
        """
        # .days is already a cached_property
        # Need to specifically return `int64` to avoid overflow.
        return Index._from_column(
            self._column.days.astype(np.dtype(np.int64)), name=self.name
        )

    @property
    @_performance_tracking
    def seconds(self) -> Index:
        """
        Number of seconds (>= 0 and less than 1 day) for each element.
        """
        # .seconds is already a cached_property
        return Index._from_column(
            self._column.seconds.astype(np.dtype(np.int32)), name=self.name
        )

    @property
    @_performance_tracking
    def microseconds(self) -> Index:
        """
        Number of microseconds (>= 0 and less than 1 second) for each element.
        """
        # .microseconds is already a cached_property
        return Index._from_column(
            self._column.microseconds.astype(np.dtype(np.int32)),
            name=self.name,
        )

    @property
    @_performance_tracking
    def nanoseconds(self) -> Index:
        """
        Number of nanoseconds (>= 0 and less than 1 microsecond) for each
        element.
        """
        # .nanoseconds is already a cached_property
        return Index._from_column(
            self._column.nanoseconds.astype(np.dtype(np.int32)), name=self.name
        )

    @property
    @_performance_tracking
    def components(self) -> DataFrame:
        """
        Return a dataframe of the components (days, hours, minutes,
        seconds, milliseconds, microseconds, nanoseconds) of the Timedeltas.
        """
        # .components is already a cached_property
        ca = ColumnAccessor(self._column.components, verify=False)
        return cudf.DataFrame._from_data(ca)

    @property
    def inferred_freq(self):
        """
        Infers frequency of TimedeltaIndex.

        Notes
        -----
        This property is currently not supported.
        """
        raise NotImplementedError("inferred_freq is not yet supported")


class CategoricalIndex(Index):
    """
    A categorical of orderable values that represent the indices of another
    Column

    Parameters
    ----------
    data : array-like (1-dimensional)
        The values of the categorical. If categories are given,
        values not in categories will be replaced with None/NaN.
    categories : list-like, optional
        The categories for the categorical. Items need to be unique.
        If the categories are not given here (and also not in dtype),
        they will be inferred from the data.
    ordered : bool, optional
        Whether or not this categorical is treated as an ordered categorical.
        If not given here or in dtype, the resulting categorical will be
        unordered.
    dtype : CategoricalDtype or "category", optional
        If CategoricalDtype, cannot be used together with categories or
        ordered.
    copy : bool, default False
        Make a copy of input.
    name : object, optional
        Name to be stored in the index.

    Attributes
    ----------
    codes
    categories

    Methods
    -------
    equals

    Returns
    -------
    CategoricalIndex

    Examples
    --------
    >>> import cudf
    >>> import pandas as pd
    >>> cudf.CategoricalIndex(
    ... data=[1, 2, 3, 4], categories=[1, 2], ordered=False, name="a")
    CategoricalIndex([1, 2, <NA>, <NA>], categories=[1, 2], ordered=False, dtype='category', name='a')

    >>> cudf.CategoricalIndex(
    ... data=[1, 2, 3, 4], dtype=pd.CategoricalDtype([1, 2, 3]), name="a")
    CategoricalIndex([1, 2, 3, <NA>], categories=[1, 2, 3], ordered=False, dtype='category', name='a')
    """

    @_performance_tracking
    def __init__(
        self,
        data=None,
        categories=None,
        ordered=None,
        dtype=None,
        copy=False,
        name=None,
        nan_as_null=no_default,
    ):
        if dtype is not None:
            dtype = cudf.dtype(dtype)
            if not isinstance(dtype, CategoricalDtype):
                raise ValueError(
                    f"dtype must be a CategoricalDtype, got {dtype}"
                )

        if isinstance(dtype, CategoricalDtype):
            if categories is not None or ordered is not None:
                raise ValueError(
                    "Cannot specify `categories` or "
                    "`ordered` together with `dtype`."
                )
        if copy:
            data = as_column(data, dtype=dtype).copy(deep=True)
        name = _getdefault_name(data, name=name)
        if isinstance(data, CategoricalColumn):
            data = data
        elif isinstance(getattr(data, "dtype", None), pd.CategoricalDtype):
            data = as_column(data)
        elif isinstance(data, (cudf.Series, Index)) and isinstance(
            data.dtype, CategoricalDtype
        ):
            data = data._column
            if (
                isinstance(dtype, CategoricalDtype)
                and dtype._categories is None
            ):
                # dtype="category" was passed, but data is already categorical
                dtype = None
        else:
            data = as_column(
                data, dtype=dtype if dtype is not None else CategoricalDtype()
            )
            if dtype is not None:
                # dtype has already been taken care
                dtype = None

        if categories is not None:
            data = data.set_categories(categories, ordered=ordered)
        elif isinstance(dtype, CategoricalDtype):
            data = data.set_categories(dtype.categories, ordered=ordered)
        elif ordered is True and data.ordered is False:
            data = data.as_ordered(ordered=True)
        elif ordered is False and data.ordered is True:
            data = data.as_ordered(ordered=False)
        SingleColumnFrame.__init__(
            self, ColumnAccessor({name: data}, verify=False)
        )

    @classmethod
    @_performance_tracking
    def _from_column(
        cls, column: ColumnBase, *, name: Hashable = None, freq: Any = None
    ) -> Self:
        if not isinstance(column.dtype, cudf.CategoricalDtype):
            raise ValueError("column must have a categorial type.")
        return super()._from_column(column, name=name)

    @classmethod
    def from_codes(
        cls,
        codes: ColumnLike,
        categories: ColumnLike,
        ordered: bool,
        name: Hashable = None,
    ) -> Self:
        """
        Construct a CategoricalIndex from codes and categories.

        More performant that using the CategoricalIndex constructor.

        Parameters
        ----------
        codes : array-like
            The integer codes of the CategoricalIndex.
        categories : array-like
            The category labels of the CategoricalIndex.
        ordered : bool
            Whether the categories are ordered.
        name : Hashable, optional
            The name of the CategoricalIndex.
        """
        categories = as_column(categories)
        dtype = cudf.CategoricalDtype(categories=categories, ordered=ordered)
        codes = as_column(codes, dtype=dtype._codes_dtype)
        cat_col = ColumnBase.create(
            codes.plc_column,
            dtype,
        )
        return cls._from_column(cat_col, name=name)

    @property
    def ordered(self) -> bool:
        return self._column.ordered

    @cached_property
    def _constructor(self):
        return CategoricalIndex

    @cached_property
    def inferred_type(self) -> str:
        return "categorical"

    @property
    @_performance_tracking
    def codes(self) -> Index:
        """
        The category codes of this categorical.
        """
        return Index._from_column(self._column.codes)

    @property
    @_performance_tracking
    def categories(self) -> Index:
        """
        The categories of this categorical.
        """
        return self.dtype.categories

    def add_categories(self, new_categories) -> Self:
        """
        Add new categories.

        `new_categories` will be included at the last/highest place in the
        categories and will be unused directly after this call.
        """
        return type(self)._from_column(
            self._column.add_categories(new_categories), name=self.name
        )

    def as_ordered(self) -> Self:
        """
        Set the Categorical to be ordered.
        """
        return type(self)._from_column(
            self._column.as_ordered(ordered=True), name=self.name
        )

    def as_unordered(self) -> Self:
        """
        Set the Categorical to be unordered.
        """
        return type(self)._from_column(
            self._column.as_ordered(ordered=False), name=self.name
        )

    def remove_categories(self, removals) -> Self:
        """
        Remove the specified categories.

        `removals` must be included in the old categories.

        Parameters
        ----------
        removals : category or list of categories
           The categories which should be removed.
        """
        return type(self)._from_column(
            self._column.remove_categories(removals), name=self.name
        )

    def remove_unused_categories(self) -> Self:
        """
        Remove categories which are not used.

        This method is currently not supported.
        """
        return type(self)._from_column(
            self._column.remove_unused_categories(), name=self.name
        )

    def rename_categories(self, new_categories) -> Self:
        """
        Rename categories.

        This method is currently not supported.
        """
        return type(self)._from_column(
            self._column.rename_categories(new_categories), name=self.name
        )

    def reorder_categories(self, new_categories, ordered=None) -> Self:
        """
        Reorder categories as specified in new_categories.

        ``new_categories`` need to include all old categories and no new category
        items.

        Parameters
        ----------
        new_categories : Index-like
           The categories in new order.
        ordered : bool, optional
           Whether or not the categorical is treated as a ordered categorical.
           If not given, do not change the ordered information.
        """
        return type(self)._from_column(
            self._column.reorder_categories(new_categories, ordered=ordered),
            name=self.name,
        )

    def set_categories(
        self, new_categories, ordered=None, rename: bool = False
    ) -> Self:
        """
        Set the categories to the specified new_categories.

        Parameters
        ----------
        new_categories : list-like
            The categories in new order.
        ordered : bool, default None
            Whether or not the categorical is treated as
            a ordered categorical. If not given, do
            not change the ordered information.
        rename : bool, default False
            Whether or not the `new_categories` should be
            considered as a rename of the old categories
            or as reordered categories.
        """
        return type(self)._from_column(
            self._column.set_categories(
                new_categories, ordered=ordered, rename=rename
            ),
            name=self.name,
        )


@_performance_tracking
def interval_range(
    start=None,
    end=None,
    periods=None,
    freq=None,
    name=None,
    closed="right",
) -> IntervalIndex:
    """
    Returns a fixed frequency IntervalIndex.

    Parameters
    ----------
    start : numeric, default None
        Left bound for generating intervals.
    end : numeric , default None
        Right bound for generating intervals.
    periods : int, default None
        Number of periods to generate
    freq : numeric, default None
        The length of each interval. Must be consistent
        with the type of start and end
    name : str, default None
        Name of the resulting IntervalIndex.
    closed : {"left", "right", "both", "neither"}, default "right"
        Whether the intervals are closed on the left-side, right-side,
        both or neither.

    Returns
    -------
    IntervalIndex

    Examples
    --------
    >>> import cudf
    >>> import pandas as pd
    >>> cudf.interval_range(start=0,end=5)
    IntervalIndex([(0, 1], (1, 2], (2, 3], (3, 4], (4, 5]], dtype='interval[int64, right]')
    >>> cudf.interval_range(start=0,end=10, freq=2,closed='left')
    IntervalIndex([[0, 2), [2, 4), [4, 6), [6, 8), [8, 10)], dtype='interval[int64, left]')
    >>> cudf.interval_range(start=0,end=10, periods=3,closed='left')
    IntervalIndex([              [0.0, 3.3333333333333335),
                   [3.3333333333333335, 6.666666666666667),
                                 [6.666666666666667, 10.0)],
                  dtype='interval[float64, left]')
    """
    nargs = sum(_ is not None for _ in (start, end, periods, freq))

    # we need at least three of (start, end, periods, freq)
    if nargs == 2 and freq is None:
        freq = 1
        nargs += 1

    if nargs != 3:
        raise ValueError(
            "Of the four parameters: start, end, periods, and "
            "freq, exactly three must be specified"
        )

    if periods is not None and not is_integer(periods):
        raise TypeError(f"periods must be an integer, got {periods}")

    # Track whether all three of start/end/freq were provided before
    # computing the missing parameter, since the dtype logic differs.
    have_all_sef = all(x is not None for x in (start, end, freq))

    if start is None:
        start = end - freq * periods
    elif freq is None:
        quotient, remainder = divmod(end - start, periods)
        if remainder:
            freq = (end - start) / periods
        else:
            freq = int(quotient)
    elif periods is None:
        periods = int((end - start) / freq)
    elif end is None:
        end = start + periods * freq

    pa_start = pa.scalar(start)
    pa_end = pa.scalar(end)
    pa_freq = pa.scalar(freq)

    if any(
        not is_dtype_obj_numeric(
            cudf_dtype_from_pa_type(x.type), include_decimal=False
        )
        for x in (pa_start, pa.scalar(periods), pa_freq, pa_end)
    ):
        raise ValueError("start, end, periods, freq must be numeric values.")

    # Determine output dtype matching pandas behaviour.
    common_dtype: np.dtype = np.dtype("int64")
    if have_all_sef:
        if (
            isinstance(start, (np.integer, np.floating))
            and isinstance(end, (np.integer, np.floating))
            and start.dtype == end.dtype
        ):
            common_dtype = start.dtype
        elif any(
            isinstance(x, (float, np.floating)) for x in (start, end, freq)
        ):
            common_dtype = np.dtype("float64")
    else:
        common_dtype = find_common_type(
            [x for x in (start, end) if x is not None]
        )
        if all(is_integer(x) for x in (start, end) if x is not None):
            common_dtype = np.dtype("int64")

    # If freq has a fractional part but common_dtype is integer,
    # promote to float64 so the cast doesn't truncate.
    if common_dtype.kind == "i" and isinstance(freq, (float, np.floating)):
        if freq != int(freq):
            common_dtype = np.dtype("float64")

    pa_start = pa_start.cast(cudf_dtype_to_pa_type(common_dtype))
    pa_freq = pa_freq.cast(cudf_dtype_to_pa_type(common_dtype))

    plc_result = plc.filling.sequence(
        size=periods + 1,
        init=pa_scalar_to_plc_scalar(pa_start),
        step=pa_scalar_to_plc_scalar(pa_freq),
    )
    bin_edges = ColumnBase.create(plc_result, dtype=common_dtype)
    return IntervalIndex.from_breaks(bin_edges, closed=closed, name=name)


class IntervalIndex(Index):
    """
    Immutable index of intervals that are closed on the same side.

    Parameters
    ----------
    data : array-like (1-dimensional)
        Array-like containing Interval objects from which to build the
        IntervalIndex.
    closed : {"left", "right", "both", "neither"}, default "right"
        Whether the intervals are closed on the left-side, right-side,
        both or neither.
    dtype : dtype or None, default None
        If None, dtype will be inferred.
    copy : bool, default False
        Copy the input data.
    name : object, optional
        Name to be stored in the index.

    Attributes
    ----------
    values

    Methods
    -------
    from_breaks
    get_loc

    Returns
    -------
    IntervalIndex
    """

    @_performance_tracking
    def __init__(
        self,
        data,
        closed: Literal["left", "right", "neither", "both"] | None = None,
        dtype=None,
        copy: bool = False,
        name=None,
        verify_integrity: bool = True,
        nan_as_null=no_default,
    ):
        name = _getdefault_name(data, name=name)

        if dtype is not None:
            dtype = cudf.dtype(dtype)
            if not isinstance(dtype, IntervalDtype):
                raise TypeError("dtype must be an IntervalDtype")
            if closed is not None and closed != dtype.closed:
                raise ValueError("closed keyword does not match dtype.closed")
            closed = dtype.closed

        if closed is None:
            if isinstance(dtype, IntervalDtype):
                closed = dtype.closed
            elif hasattr(data, "dtype") and isinstance(
                data.dtype, (pd.IntervalDtype, IntervalDtype)
            ):
                closed = data.dtype.closed

        closed = closed or "right"

        if len(data) == 0:
            if not hasattr(data, "dtype"):
                child_type: Dtype = np.dtype(np.int64)
            elif isinstance(data.dtype, (pd.IntervalDtype, IntervalDtype)):
                child_type = data.dtype.subtype
            else:
                child_type = data.dtype
            child_plc_type = dtype_to_pylibcudf_type(child_type)
            left = plc.column_factories.make_empty_column(child_plc_type)
            right = plc.column_factories.make_empty_column(child_plc_type)
            plc_column = plc.Column(
                plc.DataType(plc.TypeId.STRUCT),
                0,
                None,
                None,
                0,
                0,
                [left, right],
            )
            interval_col = ColumnBase.create(
                plc_column, IntervalDtype(child_type, closed)
            )
        else:
            col = as_column(data)
            if not isinstance(col, IntervalColumn):
                raise TypeError("data must be an iterable of Interval data")
            if copy:
                col = col.copy()
            interval_col = ColumnBase.create(
                col.plc_column,
                IntervalDtype(col.dtype.subtype, closed),  # type: ignore[union-attr]
            )

        if dtype:
            interval_col = interval_col.astype(dtype)

        SingleColumnFrame.__init__(
            self, ColumnAccessor({name: interval_col}, verify=False)
        )

    @property
    def closed(self) -> Literal["left", "right", "neither", "both"]:
        return self._column.closed

    @property
    def closed_left(self) -> bool:
        return self.closed in ("left", "both")

    @property
    def closed_right(self) -> bool:
        return self.closed in ("right", "both")

    @classmethod
    @_performance_tracking
    def _from_column(
        cls, column: ColumnBase, *, name: Hashable = None, freq: Any = None
    ) -> Self:
        if not isinstance(column.dtype, cudf.IntervalDtype):
            raise ValueError("column must have a interval type.")
        return super()._from_column(column, name=name)

    @classmethod
    @_performance_tracking
    def from_breaks(
        cls,
        breaks,
        closed: Literal["left", "right", "neither", "both"] = "right",
        name=None,
        copy: bool = False,
        dtype=None,
    ) -> Self:
        """
        Construct an IntervalIndex from an array of splits.

        Parameters
        ----------
        breaks : array-like (1-dimensional)
            Left and right bounds for each interval.
        closed : {"left", "right", "both", "neither"}, default "right"
            Whether the intervals are closed on the left-side, right-side,
            both or neither.
        copy : bool, default False
            Copy the input data.
        name : object, optional
            Name to be stored in the index.
        dtype : dtype or None, default None
            If None, dtype will be inferred.

        Returns
        -------
        IntervalIndex

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> cudf.IntervalIndex.from_breaks([0, 1, 2, 3])
        IntervalIndex([(0, 1], (1, 2], (2, 3]], dtype='interval[int64, right]')
        """
        breaks = as_column(breaks, dtype=dtype)
        if (
            len(breaks) == 0
            and dtype is None
            and is_dtype_obj_string(breaks.dtype)
        ):
            breaks = breaks.astype(np.dtype(np.int64))
        if copy:
            breaks = breaks.copy()
        left_col = breaks.slice(0, len(breaks) - 1).plc_column
        right_col = breaks.slice(1, len(breaks)).copy().plc_column
        # For indexing, children should both have 0 offset
        right_col = plc.Column(
            right_col.type(),
            right_col.size(),
            right_col.data(),
            right_col.null_mask(),
            right_col.null_count(),
            0,
            right_col.children(),
        )
        plc_column = plc.Column(
            plc.DataType(plc.TypeId.STRUCT),
            left_col.size(),
            None,
            None,
            0,
            0,
            [left_col, right_col],
        )
        dtype = IntervalDtype(breaks.dtype, closed)
        interval_col = ColumnBase.create(plc_column, dtype)
        return IntervalIndex._from_column(interval_col, name=name)

    @cached_property
    def _constructor(self):
        return IntervalIndex

    @cached_property
    def inferred_type(self) -> str:
        return "interval"

    @classmethod
    def from_arrays(
        cls,
        left,
        right,
        closed: Literal["left", "right", "both", "neither"] = "right",
        copy: bool = False,
        dtype=None,
    ) -> Self:
        raise NotImplementedError("from_arrays is currently not supported.")

    @classmethod
    def from_tuples(
        cls,
        data,
        closed: Literal["left", "right", "both", "neither"] = "right",
        name=None,
        copy: bool = False,
        dtype=None,
    ) -> Self:
        pidx = pd.IntervalIndex.from_tuples(
            data, closed=closed, name=name, copy=copy, dtype=dtype
        )
        return cls(pidx, name=name)

    def __getitem__(self, index):
        raise NotImplementedError(
            "Getting a scalar from an IntervalIndex is not yet supported"
        )

    def _pandas_repr_compatible(self, nan_rep=None) -> Self:
        return self

    @property
    def is_empty(self) -> cupy.ndarray:
        """
        Indicates if an interval is empty, meaning it contains no points.
        """
        # .is_empty is already a cached_property
        return self._column.is_empty.values

    @property
    def is_non_overlapping_monotonic(self) -> bool:
        """
        Return a True if the IntervalIndex is non-overlapping and monotonic.
        """
        # .is_non_overlapping_monotonic is already a cached_property
        return self._column.is_non_overlapping_monotonic

    @property
    def is_overlapping(self) -> bool:
        """
        Return True if the IntervalIndex has overlapping intervals, else False.

        Currently not implemented
        """
        # .is_overlapping is already a cached_property
        return self._column.is_overlapping

    @property
    def length(self) -> Index:
        """
        Return an Index with entries denoting the length of each Interval.
        """
        # .length is already a cached_property
        return _index_from_data({None: self._column.length})

    @property
    def left(self) -> Index:
        """
        Return left bounds of the intervals in the IntervalIndex.

        The left bounds of each interval in the IntervalIndex are
        returned as an Index. The datatype of the left bounds is the
        same as the datatype of the endpoints of the intervals.
        """
        return _index_from_data({None: self._column.left})

    @property
    def mid(self) -> Index:
        """
        Return the midpoint of each interval in the IntervalIndex as an Index.

        Each midpoint is calculated as the average of the left and right bounds
        of each interval.
        """
        # .mid is already a cached_property
        return _index_from_data({None: self._column.mid})

    @property
    def right(self) -> Index:
        """
        Return right bounds of the intervals in the IntervalIndex.

        The right bounds of each interval in the IntervalIndex are
        returned as an Index. The datatype of the right bounds is the
        same as the datatype of the endpoints of the intervals.
        """
        return _index_from_data({None: self._column.right})

    def overlaps(self, other) -> cupy.ndarray:
        """
        Check elementwise if an Interval overlaps the values in the IntervalIndex.

        Currently not supported.
        """
        return self._column.overlaps(other).values

    def set_closed(
        self, closed: Literal["left", "right", "both", "neither"]
    ) -> Self:
        """
        Return an identical IntervalArray closed on the specified side.

        Parameters
        ----------
        closed : {'left', 'right', 'both', 'neither'}
            Whether the intervals are closed on the left-side, right-side, both
            or neither.
        """
        return type(self)._from_column(
            self._column.set_closed(closed), name=self.name
        )

    def to_tuples(self, na_tuple: bool = True) -> pd.Index:
        """
        Return an Index of tuples of the form (left, right).

        Parameters
        ----------
        na_tuple : bool, default True
            If ``True``, return ``NA`` as a tuple ``(nan, nan)``. If ``False``,
            just return ``NA`` as ``nan``.
        """
        return self.to_pandas().to_tuples(na_tuple=na_tuple)


@_performance_tracking
def _as_index(
    data=None,
    dtype=None,
    copy: bool = False,
    name=no_default,
    tupleize_cols: bool = True,
    nan_as_null=no_default,
    **kwargs,
) -> Index:
    """Create an Index from an arbitrary object

    Parameters
    ----------
    data : object
        Object to construct the Index from. See *Notes*.
    dtype : optional
        Optionally typecast the constructed Index to the given
        dtype.
    copy : bool, default False
        If True, Make copies of `arbitrary` if possible and create an
        Index out of it.
        If False, `arbitrary` will be shallow-copied if it is a
        device-object to construct an Index.
    name : object, optional
        Name of the index being created, by default it is `None`.
    nan_as_null : bool, optional, default None
        If None (default), treats NaN values in arbitrary as null.
        If True, combines the mask and NaNs to
        form a new validity mask. If False, leaves NaN values as is.

    Returns
    -------
    result : subclass of Index
        - CategoricalIndex for Categorical input.
        - DatetimeIndex for Datetime input.
        - Index for all other inputs.

    Notes
    -----
    Currently supported inputs are:

    * ``Column``
    * ``Buffer``
    * ``Series``
    * ``Index``
    * numba device array
    * numpy array
    * pyarrow array
    * pandas.Categorical

    Returns
    -------
    result : subclass of Index
        - CategoricalIndex for Categorical input.
        - DatetimeIndex for Datetime input.
        - Index for all other inputs.
    """
    if tupleize_cols is not True:
        raise NotImplementedError("tupleize_cols is currently not supported.")
    if nan_as_null is no_default:
        nan_as_null = (
            False if cudf.get_option("mode.pandas_compatible") else None
        )

    if name is no_default:
        name = getattr(data, "name", None)
    if dtype is not None:
        dtype = cudf.dtype(dtype)

    if isinstance(data, cudf.MultiIndex):
        if dtype is not None:
            raise TypeError(
                "dtype must be `None` for inputs of type: "
                f"{type(data).__name__}, found {dtype=} "
            )
        return data.copy(deep=copy)
    elif isinstance(data, Index):
        if not isinstance(data, cudf.RangeIndex):
            idx = type(data)._from_column(
                data._column.copy(deep=copy) if copy else data._column,
                name=name,
            )
        else:
            idx = data.copy(deep=copy).rename(name)
    elif isinstance(data, ColumnBase):
        raise ValueError("Use cudf.Index._from_column instead.")
    elif isinstance(data, (pd.RangeIndex, range)):
        idx = RangeIndex(
            start=data.start,
            stop=data.stop,
            step=data.step,
            name=name,
        )
    elif isinstance(data, pd.DatetimeIndex):
        return cudf.DatetimeIndex._from_data(
            {None: as_column(data, nan_as_null=nan_as_null)},
            name=data.name,
            freq=data.freq.freqstr if data.freq else None,
        )
    elif isinstance(data, pd.MultiIndex):
        if dtype is not None:
            raise TypeError(
                "dtype must be `None` for inputs of type: "
                f"{type(data).__name__}, found {dtype=} "
            )
        return cudf.MultiIndex(
            levels=data.levels,
            codes=data.codes,
            names=data.names,
            nan_as_null=nan_as_null,
        )
    elif isinstance(data, cudf.DataFrame) or is_scalar(data):
        raise TypeError("Index data must be 1-dimensional and list-like")
    else:
        return Index._from_column(
            as_column(data, dtype=dtype, nan_as_null=nan_as_null),
            name=name,
        )
    if dtype is not None:
        idx = idx.astype(dtype)
    return idx


def _getdefault_name(values, name):
    if name is None:
        return getattr(values, "name", None)
    return name


@_performance_tracking
def _concat_range_index(indexes: list[RangeIndex]) -> Index:
    """
    An internal Utility function to concat RangeIndex objects.
    """
    start = step = next_ = None

    # Filter the empty indexes
    non_empty_indexes = [obj for obj in indexes if len(obj)]

    if not non_empty_indexes:
        # Here all "indexes" had 0 length, i.e. were empty.
        # In this case return an empty range index.
        return RangeIndex(0, 0)

    for obj in non_empty_indexes:
        if start is None:
            # This is set by the first non-empty index
            start = obj.start
            if step is None and len(obj) > 1:
                step = obj.step
        elif step is None:
            # First non-empty index had only one element
            if obj.start == start:
                result = Index._from_column(
                    concat_columns([x._column for x in indexes])
                )
                return result
            step = obj.start - start

        non_consecutive = (step != obj.step and len(obj) > 1) or (
            next_ is not None and obj.start != next_
        )
        if non_consecutive:
            result = Index._from_column(
                concat_columns([x._column for x in indexes])
            )
            return result
        if step is not None:
            next_ = obj[-1] + step

    stop = non_empty_indexes[-1].stop if next_ is None else next_
    return RangeIndex(start, stop, step)


@_performance_tracking
def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """
    Extended Euclidean algorithms to solve Bezout's identity:
       a*x + b*y = gcd(x, y)
    Finds one particular solution for x, y: s, t
    Returns: gcd, s, t
    """
    s, old_s = 0, 1
    t, old_t = 1, 0
    r, old_r = b, a
    while r:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    return old_r, old_s, old_t


def _get_indexer_basic(index, positions, method, target_col, tolerance):
    # `positions` will be modified in-place, so it is the
    # responsibility of the caller to decide whether or not
    # to make a copy of it before passing it to this method.
    nonexact = positions == -1
    positions[nonexact] = index.searchsorted(
        target_col[nonexact],
        side="left" if method in {"pad", "ffill"} else "right",
    )
    if method in {"pad", "ffill"}:
        # searchsorted returns "indices into a sorted array such that,
        # if the corresponding elements in v were inserted before the
        # indices, the order of a would be preserved".
        # Thus, we need to subtract 1 to find values to the left.
        positions[nonexact] -= 1
        # This also mapped not found values (values of 0 from
        # np.searchsorted) to -1, which conveniently is also our
        # sentinel for missing values
    else:
        # Mark indices to the right of the largest value as not found
        positions[positions == len(index)] = np.int32(-1)

    if tolerance is not None:
        distance = abs(index[positions] - target_col)
        return positions.where(distance <= tolerance, -1)
    return positions


def _get_nearest_indexer(
    index: Index,
    positions: Series,
    target_col: ColumnBase,
    tolerance: int | float,
):
    """
    Get the indexer for the nearest index labels; requires an index with
    values that can be subtracted from each other.
    """
    left_indexer = _get_indexer_basic(
        index=index,
        positions=positions.copy(deep=True),
        method="pad",
        target_col=target_col,
        tolerance=tolerance,
    )
    right_indexer = _get_indexer_basic(
        index=index,
        # positions no longer used so don't copy
        positions=positions,
        method="backfill",
        target_col=target_col,
        tolerance=tolerance,
    )

    left_distances = abs(index[left_indexer] - target_col)
    right_distances = abs(index[right_indexer] - target_col)

    op = operator.lt if index.is_monotonic_increasing else operator.le
    indexer = left_indexer.where(
        op(left_distances, right_distances) | (right_indexer == -1),
        right_indexer,
    )

    if tolerance is not None:
        distance = abs(index[indexer] - target_col)
        return indexer.where(distance <= tolerance, -1)
    return indexer


def _validate_freq(freq: Any) -> pd.tseries.offsets.BaseOffset | None:
    """Normalize ``freq`` to a canonical pandas offset.

    Accepts freq strings, pandas offsets and cudf's own offset classes,
    rejecting frequencies that cudf cannot represent.
    """
    from cudf.core.tools.datetimes import DateOffset, MonthEnd, YearEnd

    if freq is None:
        return None
    if isinstance(freq, (DateOffset, MonthEnd, YearEnd)):
        return freq._maybe_as_fast_pandas_offset()
    if isinstance(freq, str):
        # cudf supports a subset of pandas frequency strings; parse with
        # cudf's DateOffset (which raises for unsupported ones) and store
        # the equivalent pandas offset.
        return DateOffset._from_freqstr(freq)._maybe_as_fast_pandas_offset()
    if not isinstance(freq, pd.tseries.offsets.BaseOffset):
        raise ValueError(f"Invalid frequency: {freq}")
    # gate pandas offsets on the same supported subset
    if type(freq) is pd.DateOffset:
        DateOffset(**freq.kwds)
    else:
        DateOffset._from_freqstr(freq.freqstr)
    return freq
