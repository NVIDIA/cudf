# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import operator
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Self

import cupy
import numpy as np
import pandas as pd
import pyarrow as pa

import pylibcudf as plc

import cudf
from cudf.api.extensions import no_default
from cudf.api.types import is_dtype_equal, is_scalar, is_string_dtype
from cudf.core._internals import copying, sorting
from cudf.core.abc import Serializable
from cudf.core.column import (
    ColumnBase,
    access_columns,
    as_column,
    deserialize_columns,
    serialize_columns,
)
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.dtype.validators import (
    is_dtype_obj_numeric,
    is_dtype_obj_string,
)
from cudf.core.mixins import BinaryOperand, Scannable
from cudf.utils.dtypes import (
    dtype_from_pylibcudf_column,
    find_common_type,
    is_pandas_nullable_extension_dtype,
)
from cudf.utils.performance_tracking import _performance_tracking
from cudf.utils.utils import _array_ufunc, _warn_no_dask_cudf

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
        Hashable,
        Iterable,
        MutableMapping,
    )
    from types import ModuleType

    from cudf._typing import Axis, Dtype, DtypeObj, ScalarLike
    from cudf.core.buffer import Buffer
    from cudf.core.series import Series


class Frame(BinaryOperand, Scannable, Serializable):
    """A collection of Column objects.

    Parameters
    ----------
    data : dict
        An dict mapping column names to Columns
    """

    _VALID_BINARY_OPERATIONS = BinaryOperand._SUPPORTED_BINARY_OPERATIONS

    def __init__(self, data: ColumnAccessor | MutableMapping[Any, ColumnBase]):
        self._data = ColumnAccessor(data)

    @property
    def _num_columns(self) -> int:
        return len(self._data)

    @property
    def _num_rows(self) -> int:
        return self._data.nrows

    @property
    def _column_names(self) -> tuple[Any, ...]:
        return self._data.names

    @property
    def _columns(self) -> tuple[ColumnBase, ...]:
        return self._data.columns

    @property
    def _column_labels_and_values(
        self,
    ) -> Iterable[tuple[Hashable, ColumnBase]]:
        return zip(self._column_names, self._columns, strict=True)

    @property
    def _dtypes(self) -> Generator[tuple[Hashable, DtypeObj], None, None]:
        for label, col in self._column_labels_and_values:
            yield label, col.dtype

    @property
    def ndim(self) -> int:
        raise NotImplementedError

    @_performance_tracking
    def serialize(self):
        # TODO: See if self._data can be serialized outright
        frames = []
        header = {
            "column_label_dtype": None,
            "dtype-is-cudf-serialized": False,
        }
        if (label_dtype := self._data.label_dtype) is not None:
            try:
                header["column_label_dtype"], frames = (
                    label_dtype.device_serialize()
                )
                header["dtype-is-cudf-serialized"] = True
            except AttributeError:
                if isinstance(label_dtype, pd.StringDtype):
                    header["column_label_dtype"] = (
                        label_dtype.storage,
                        label_dtype.na_value,
                    )
                else:
                    header["column_label_dtype"] = label_dtype.str

        header["columns"], column_frames = serialize_columns(self._columns)
        column_names, column_names_numpy_type = (
            zip(
                *[
                    (cname.item(), type(cname).__name__)
                    if isinstance(cname, np.generic)
                    else (cname, "")
                    for cname in self._column_names
                ],
                strict=True,
            )
            if self._column_names
            else ((), ())
        )
        header |= {
            "column_names": column_names,
            "column_names_numpy_type": column_names_numpy_type,
            "column_rangeindex": self._data.rangeindex,
            "column_multiindex": self._data.multiindex,
            "column_level_names": self._data._level_names,
        }
        frames.extend(column_frames)

        return header, frames

    @classmethod
    @_performance_tracking
    def deserialize(cls, header, frames):
        kwargs = {}
        dtype_header = header["column_label_dtype"]
        if header["dtype-is-cudf-serialized"]:
            count = dtype_header["frame_count"]
            kwargs["label_dtype"] = cls.device_deserialize(
                header, frames[:count]
            )
            frames = frames[count:]
        elif isinstance(dtype_header, tuple):
            kwargs["label_dtype"] = pd.StringDtype(*dtype_header)
        elif isinstance(dtype_header, str):
            kwargs["label_dtype"] = np.dtype(dtype_header)
        else:
            kwargs["label_dtype"] = dtype_header

        columns = deserialize_columns(header["columns"], frames)
        for metadata in [
            "rangeindex",
            "multiindex",
            "level_names",
        ]:
            key = f"column_{metadata}"
            if key in header:
                kwargs[metadata] = header[key]

        column_names = [
            getattr(np, cntype)(cname) if cntype != "" else cname
            for cname, cntype in zip(
                header["column_names"],
                header["column_names_numpy_type"],
                strict=True,
            )
        ]
        col_accessor = ColumnAccessor(
            data=dict(zip(column_names, columns, strict=True)), **kwargs
        )
        return cls._from_data(col_accessor)

    @_performance_tracking
    def nans_to_nulls(self) -> Self:
        result = []
        for col in self._columns:
            converted = col.nans_to_nulls()
            if converted is col:
                converted = converted.copy()
            result.append(converted)
        return self._from_data_like_self(
            self._data._from_columns_like_self(result)
        )

    @classmethod
    @_performance_tracking
    def _from_data(cls, data: MutableMapping) -> Self:
        """
        Construct cls from a ColumnAccessor-like mapping.
        """
        from cudf.core.index import Index

        if cls is Index:
            # Because Index.__new__ is overridden as a constructor,
            # to copy pandas...
            obj = object.__new__(cls)
        else:
            obj = cls.__new__(cls)
        Frame.__init__(obj, data)
        return obj

    @_performance_tracking
    def _from_data_like_self(self, data: MutableMapping) -> Self:
        """
        Return type(self) from a ColumnAccessor-like mapping but
        with the external properties, e.g. .index, .name, of self.
        """
        return self._from_data(data)

    @_performance_tracking
    def _from_columns_like_self(
        self,
        columns: list[ColumnBase],
        column_names: Iterable[str] | None = None,
    ):
        """Construct a Frame from a list of columns with metadata from self.

        If `column_names` is None, use column names from self.
        """
        if column_names is None:
            column_names = self._column_names
        data = dict(zip(column_names, columns, strict=True))
        return self.__class__._from_data(data)

    def _drop_duplicates_columns(
        self,
        columns: list[ColumnBase],
        keys: list[int],
        keep: Literal["first", "last", False],
        nulls_are_equal: bool,
    ) -> tuple[plc.Column, ...]:
        """Core stable_distinct implementation shared by Index and IndexedFrame."""
        _keep_options = {
            "first": plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
            "last": plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
            False: plc.stream_compaction.DuplicateKeepOption.KEEP_NONE,
        }
        if (keep_option := _keep_options.get(keep)) is None:
            raise ValueError('keep must be either "first", "last" or False')

        with access_columns(*columns, mode="read", scope="internal") as cols:
            plc_table = plc.stream_compaction.stable_distinct(
                plc.Table([col.plc_column for col in cols]),
                keys,
                keep_option,
                plc.types.NullEquality.EQUAL
                if nulls_are_equal
                else plc.types.NullEquality.UNEQUAL,
                plc.types.NanEquality.ALL_EQUAL,
            )
            return plc_table.columns()

    def _drop_nulls_columns(
        self,
        columns: list[ColumnBase],
        keys: list[int],
        how: Literal["any", "all"],
        thresh: int | None = None,
    ) -> tuple[plc.Column, ...]:
        """Core drop_nulls implementation shared by Index and IndexedFrame."""
        if how not in {"any", "all"}:
            raise ValueError("how must be 'any' or 'all'")

        if thresh is not None:
            keep_threshold = thresh
        elif how == "all":
            keep_threshold = 1
        else:
            keep_threshold = len(keys)

        with access_columns(*columns, mode="read", scope="internal") as cols:
            plc_table = plc.stream_compaction.drop_nulls(
                plc.Table([col.plc_column for col in cols]),
                keys,
                keep_threshold,
            )
            return plc_table.columns()

    @_performance_tracking
    def _mimic_inplace(
        self, result: Self, inplace: bool = False
    ) -> Self | None:
        if not isinstance(inplace, bool):
            raise TypeError("inplace must be a boolean")
        if inplace:
            for col in self._column_names:
                if col in result._data:
                    self._data[col]._mimic_inplace(
                        result._data[col], inplace=True
                    )
            self._data = result._data
            return None
        else:
            return result

    @property
    @_performance_tracking
    def size(self) -> int:
        """
        Return the number of elements in the underlying data.

        Returns
        -------
        size : Size of the DataFrame / Index / Series / MultiIndex

        Examples
        --------
        Size of an empty dataframe is 0.

        >>> import cudf
        >>> df = cudf.DataFrame()
        >>> df
        Empty DataFrame
        Columns: []
        Index: []
        >>> df.size
        0
        >>> df = cudf.DataFrame(index=[1, 2, 3])
        >>> df
        Empty DataFrame
        Columns: []
        Index: [1, 2, 3]
        >>> df.size
        0

        DataFrame with values

        >>> df = cudf.DataFrame({'a': [10, 11, 12],
        ...         'b': ['hello', 'rapids', 'ai']})
        >>> df
            a       b
        0  10   hello
        1  11  rapids
        2  12      ai
        >>> df.size
        6
        >>> df.index
        RangeIndex(start=0, stop=3)
        >>> df.index.size
        3

        Size of an Index

        >>> index = cudf.Index([])
        >>> index
        Index([], dtype='float64')
        >>> index.size
        0
        >>> index = cudf.Index([1, 2, 3, 10])
        >>> index
        Index([1, 2, 3, 10], dtype='int64')
        >>> index.size
        4

        Size of a MultiIndex

        >>> midx = cudf.MultiIndex(
        ...                 levels=[["a", "b", "c", None], ["1", None, "5"]],
        ...                 codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        ...                 names=["x", "y"],
        ...             )
        >>> midx
        MultiIndex([( 'a',  '1'),
                    ( 'a',  '5'),
                    ( 'b', <NA>),
                    ( 'c', <NA>),
                    (<NA>,  '1')],
                   names=['x', 'y'])
        >>> midx.size
        5
        """
        return self._num_columns * self._num_rows

    @property
    @_performance_tracking
    def empty(self) -> bool:
        """
        Indicator whether DataFrame or Series is empty.

        True if DataFrame/Series is entirely empty (no items),
        meaning any of the axes are of length 0.

        Returns
        -------
        out : bool
            If DataFrame/Series is empty, return True, if not return False.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'A' : []})
        >>> df
        Empty DataFrame
        Columns: [A]
        Index: []
        >>> df.empty
        True

        If we only have `null` values in our DataFrame, it is
        not considered empty! We will need to drop
        the `null`'s to make the DataFrame empty:

        >>> df = cudf.DataFrame({'A' : [None, None]})
        >>> df
              A
        0  <NA>
        1  <NA>
        >>> df.empty
        False
        >>> df.dropna().empty
        True

        Non-empty and empty Series example:

        >>> s = cudf.Series([1, 2, None])
        >>> s
        0       1
        1       2
        2    <NA>
        dtype: int64
        >>> s.empty
        False
        >>> s = cudf.Series([])
        >>> s
        Series([], dtype: float64)
        >>> s.empty
        True

        .. pandas-compat::
            :attr:`pandas.DataFrame.empty`, :attr:`pandas.Series.empty`

            If DataFrame/Series contains only `null` values, it is still not
            considered empty. See the example above.
        """
        return self.size == 0

    def memory_usage(self, deep: bool = False) -> int:
        """Return the memory usage of an object.

        Parameters
        ----------
        deep : bool
            The deep parameter is ignored and is only included for pandas
            compatibility.

        Returns
        -------
        The total bytes used.
        """
        raise NotImplementedError

    def __sizeof__(self):
        if cudf.get_option("mode.pandas_compatible"):
            return self.memory_usage(deep=True, index=True)
        return object.__sizeof__(self)

    @_performance_tracking
    def __len__(self) -> int:
        return self._num_rows

    @_performance_tracking
    def astype(
        self, dtype: dict[Hashable, DtypeObj], copy: bool | None = None
    ) -> Self:
        if copy is None:
            copy = True
        if not copy:
            if all(
                dtype.get(col_name, col.dtype) == col.dtype
                for col_name, col in self._column_labels_and_values
            ):
                return self.copy(deep=False)
        casted = (
            col.astype(dtype.get(col_name, col.dtype), copy=copy)
            for col_name, col in self._column_labels_and_values
        )
        ca = self._data._from_columns_like_self(casted, verify=False)
        return self._from_data_like_self(ca)

    @_performance_tracking
    def equals(self, other) -> bool:
        """
        Test whether two objects contain the same elements.

        This function allows two objects to be compared against
        each other to see if they have the same shape and elements. NaNs in
        the same location are considered equal. The column headers do not
        need to have the same type.

        Parameters
        ----------
        other : Index, Series, DataFrame
            The other object to be compared with.

        Returns
        -------
        bool
            True if all elements are the same in both objects, False
            otherwise.

        Examples
        --------
        >>> import cudf

        Comparing Series with `equals`:

        >>> s = cudf.Series([1, 2, 3])
        >>> other = cudf.Series([1, 2, 3])
        >>> s.equals(other)
        True
        >>> different = cudf.Series([1.5, 2, 3])
        >>> s.equals(different)
        False

        Comparing DataFrames with `equals`:

        >>> df = cudf.DataFrame({1: [10], 2: [20]})
        >>> df
            1   2
        0  10  20
        >>> exactly_equal = cudf.DataFrame({1: [10], 2: [20]})
        >>> exactly_equal
            1   2
        0  10  20
        >>> df.equals(exactly_equal)
        True

        For two DataFrames to compare equal, the types of column
        values must be equal, but the types of column labels
        need not:

        >>> different_column_type = cudf.DataFrame({1.0: [10], 2.0: [20]})
        >>> different_column_type
           1.0  2.0
        0   10   20
        >>> df.equals(different_column_type)
        True
        """
        if not isinstance(other, type(self)):
            return False
        elif self is other:
            return True
        elif (
            self._num_columns != other._num_columns
            or self._num_rows != other._num_rows
        ):
            return False

        return all(
            self_col.equals(other_col, check_dtypes=True)
            for self_col, other_col in zip(
                self._columns, other._columns, strict=True
            )
        )

    @_performance_tracking
    def _get_columns_by_label(self, labels) -> Self:
        """
        Returns columns of the Frame specified by `labels`.

        Akin to cudf.DataFrame(...).loc[:, labels]
        """
        return self._from_data(self._data.select_by_label(labels))

    @property
    @_performance_tracking
    def values(self) -> cupy.ndarray:
        """
        Return a CuPy representation of the DataFrame.

        Only the values in the DataFrame will be returned, the axes labels will
        be removed.

        Returns
        -------
        cupy.ndarray
            The values of the DataFrame.
        """
        return self.to_cupy()

    @_performance_tracking
    def __array__(self, dtype=None, copy=None):
        raise TypeError(
            "Implicit conversion to a host NumPy array via __array__ is not "
            "allowed, To explicitly construct a GPU matrix, consider using "
            ".to_cupy()\nTo explicitly construct a host matrix, consider "
            "using .to_numpy()."
        )

    @_performance_tracking
    def __arrow_array__(self, type=None):
        raise TypeError(
            "Implicit conversion to a host PyArrow object via __arrow_array__ "
            "is not allowed. Consider using .to_arrow()"
        )

    @_performance_tracking
    def _to_array(
        self,
        get_array: Callable,
        module: ModuleType,
        copy: bool,
        dtype: Dtype | None = None,
        na_value=no_default,
    ) -> cupy.ndarray | np.ndarray:
        # Internal function to implement to_cupy and to_numpy, which are nearly
        # identical except for the attribute they access to generate values.

        def is_numpy_object_dtype(dtype: Dtype | None) -> bool:
            try:
                return np.dtype(dtype) == np.dtype("O")
            except TypeError:
                return False

        def to_array(
            col: ColumnBase, to_dtype: Dtype | None
        ) -> cupy.ndarray | np.ndarray:
            if (
                col.has_nulls()
                and dtype is not None
                and not is_string_dtype(dtype)
                and na_value is no_default
            ):
                raise ValueError(
                    f"cannot convert to '{dtype}'-dtype NumPy array "
                    "with missing values. Specify an appropriate 'na_value' "
                    "for this dtype."
                )
            if na_value is not no_default:
                col = col.fillna(na_value)

            if isinstance(col.dtype, cudf.CategoricalDtype):
                col = col._get_decategorized_column()  # type: ignore[attr-defined]

            if (
                to_dtype is not None
                and cudf.dtype(to_dtype).kind != "O"
                and col.dtype != cudf.dtype(to_dtype)
            ):
                # Skip the in-cudf cast when the target is numpy object:
                # in cudf cudf.dtype(object) is the string dtype, so
                # casting would stringify values. module.asarray(..., dtype=object)
                # below correctly boxes the host values as Python objects.
                col = col.astype(cudf.dtype(to_dtype))

            array = get_array(col)

            if (
                is_pandas_nullable_extension_dtype(col.dtype)
                and col.dtype.kind in "iuf"
                and to_dtype is None
            ):
                to_dtype = array.dtype
            if (
                to_dtype != array.dtype
                and dtype is None
                and array.dtype.kind in "f"
                and col.has_nulls()
            ):
                to_dtype = None
            casted_array = module.asarray(array, dtype=to_dtype)
            if (
                col.has_nulls()
                and dtype is not None
                and is_string_dtype(dtype)
                and not is_numpy_object_dtype(dtype)
            ):
                casted_array[col.isnull().to_numpy()] = (
                    cudf.NA if na_value is no_default else na_value
                )
            if copy and casted_array is array:
                # Don't double copy after asarray
                casted_array = casted_array.copy()
            return casted_array

        ncol = self._num_columns
        if ncol == 0:
            return module.empty(
                shape=(len(self), ncol),
                dtype=np.dtype("float64"),
                order="F",
            )

        if dtype is None:
            if ncol == 1:
                to_dtype = next(self._dtypes)[1]
                if (na_value is no_default or na_value is None) and (
                    isinstance(to_dtype, np.dtype)
                    and to_dtype.kind in "iu"
                    and self._columns[0].has_nulls()
                ):
                    # A single nullable integer column must be promoted to
                    # float64 so its nulls can be represented as NaN, matching
                    # Series.to_numpy()/.values and the multi-column path
                    # below. ``na_value`` is left unset (``no_default`` for
                    # ``to_numpy`` / ``None`` for ``to_cupy``), so the nulls
                    # are not being filled and the integer output buffer would
                    # otherwise store the null sentinel instead.
                    to_dtype = np.dtype("float64")
            else:
                to_dtype = find_common_type(
                    [dtype for _, dtype in self._dtypes]
                )
                if to_dtype is not None and any(
                    col.has_nulls() for col in self._columns
                ):
                    if to_dtype.kind == "b" or any(
                        dtype.kind == "b" for _, dtype in self._dtypes
                    ):
                        if module == cupy:
                            raise ValueError(
                                "Cannot convert to cupy bool array with nulls."
                            )
                        else:
                            to_dtype = np.dtype("object")
                    elif to_dtype.kind in "ui":
                        to_dtype = np.dtype("float64")

            if is_pandas_nullable_extension_dtype(
                to_dtype
            ) or is_dtype_obj_string(to_dtype):
                to_dtype = getattr(to_dtype, "numpy_dtype", to_dtype)
                if getattr(to_dtype, "kind", None) in {"U", "O"}:
                    to_dtype = np.dtype(object)
            if isinstance(to_dtype, cudf.CategoricalDtype):
                to_dtype = to_dtype.categories.dtype
                # In pandas 3.0+ string categories have StringDtype, which is
                # not a numpy dtype. Map it to object so numpy can hold it.
                if is_dtype_obj_string(to_dtype):
                    to_dtype = np.dtype(object)

            if not isinstance(to_dtype, np.dtype):
                raise NotImplementedError(
                    f"{to_dtype} cannot be exposed as an array"
                )

        if self.ndim == 1:
            return to_array(
                self._columns[0], to_dtype if dtype is None else dtype
            )
        else:
            matrix = module.empty(
                shape=(len(self), ncol),
                dtype=to_dtype if dtype is None else dtype,
                order="F",
            )
            for i, col in enumerate(self._columns):
                # TODO: col.values may fail if there is nullable data or an
                # unsupported dtype. We may want to catch and provide a more
                # suitable error.
                matrix[:, i] = to_array(
                    col, to_dtype if dtype is None else dtype
                )
            return matrix

    @_performance_tracking
    def to_pylibcudf(self) -> tuple[plc.Table, dict[str, Any]]:
        """
        Converts Frame to a pylibcudf.Table.
        Note: This method should not be called directly on a Frame object
        Instead, it should be called on subclasses like DataFrame/Series.
        """
        raise NotImplementedError(f"{type(self)} must implement to_pylibcudf")

    @_performance_tracking
    def to_pandas(self, *, nullable: bool = False, arrow_type: bool = False):
        raise NotImplementedError(f"{type(self)} must implement to_pandas")

    @_performance_tracking
    def to_dlpack(self):
        """
        Converts a cuDF object to a DLPack tensor.
        DLPack is an open-source memory tensor structure:
        `dmlc/dlpack <https://github.com/dmlc/dlpack>`_.

        Returns
        -------
        PyCapsule
            A DLPack tensor pointer which is encapsulated in a PyCapsule object.

        Notes
        -----
        The result is in column-major (Fortran order) format. If the
        output tensor needs to be row major, transpose the output of this function.
        """
        dtypes = []
        for _, dtype in self._dtypes:
            if not is_dtype_obj_numeric(dtype, include_decimal=False):
                raise NotImplementedError(
                    "non-numeric data is not yet supported"
                )
            dtypes.append(dtype)
        dtype = find_common_type(dtypes)
        casted = self.astype(dtype)
        return plc.interop.to_dlpack(
            plc.Table([col.plc_column for col in casted._columns])
        )

    @_performance_tracking
    def to_cupy(
        self,
        dtype: Dtype | None = None,
        copy: bool = False,
        na_value=None,
    ) -> cupy.ndarray:
        """Convert the Frame to a CuPy array.

        Parameters
        ----------
        dtype : str or :class:`numpy.dtype`, optional
            The dtype to pass to :func:`numpy.asarray`.
        copy : bool, default False
            Whether to ensure that the returned value is not a view on
            another array. Note that ``copy=False`` does not *ensure* that
            ``to_cupy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary.
        na_value : Any, default None
            The value to use for missing values. The default value depends on
            dtype and the dtypes of the DataFrame columns.

        Returns
        -------
        cupy.ndarray
        """
        if (
            self._num_columns > 1
            and na_value is None
            and self._columns[0].dtype.kind in {"i", "u", "f", "b"}
            and (dtype is None or dtype == self._columns[0].dtype)
            and all(
                not col.nullable and col.dtype == self._columns[0].dtype
                for col in self._columns
            )
        ):
            if dtype is None:
                dtype = self._columns[0].dtype

            shape = (len(self), self._num_columns)
            out = cupy.empty(shape, dtype=dtype, order="F")

            table = plc.Table([col.plc_column for col in self._columns])
            plc.reshape.table_to_array(
                table,
                out.data.ptr,
                out.nbytes,
            )
            return out
        elif self._num_columns == 1:
            col = self._columns[0]
            final_dtype = col.dtype if dtype is None else dtype

            if (
                not copy
                and col.dtype.kind in {"i", "u", "f", "b"}
                and cupy.can_cast(col.dtype, final_dtype)
            ):
                if col.has_nulls():
                    if na_value is not None:
                        col = col.fillna(na_value)
                    else:
                        return self._to_array(
                            lambda col: col.values,
                            cupy,
                            copy,
                            dtype,
                            na_value,
                        )
                return cupy.asarray(col, dtype=final_dtype).reshape((-1, 1))
        return self._to_array(
            lambda col: col.values,
            cupy,
            copy,
            dtype,
            na_value,
        )

    @_performance_tracking
    def to_numpy(
        self,
        dtype: Dtype | None = None,
        copy: bool = True,
        na_value=no_default,
    ) -> np.ndarray:
        """Convert the Frame to a NumPy array.

        Parameters
        ----------
        dtype : str or :class:`numpy.dtype`, optional
            The dtype to pass to :func:`numpy.asarray`.
        copy : bool, default True
            Whether to ensure that the returned value is not a view on
            another array. This parameter must be ``True`` since cuDF must copy
            device memory to host to provide a numpy array.
        na_value : Any, default None
            The value to use for missing values. The default value depends on
            dtype and the dtypes of the DataFrame columns.

        Returns
        -------
        numpy.ndarray
        """
        if not copy:
            raise ValueError(
                "copy=False is not supported because conversion to a numpy "
                "array always copies the data."
            )

        return self._to_array(
            lambda col: col.to_numpy(), np, copy, dtype, na_value
        )

    @_performance_tracking
    def where(self, cond, other=None, inplace: bool = False) -> Self | None:
        """
        Replace values where the condition is False.

        Parameters
        ----------
        cond : bool Series/DataFrame, array-like
            Where cond is True, keep the original value.
            Where False, replace with corresponding value from other.
            Callables are not supported.
        other: scalar, list of scalars, Series/DataFrame
            Entries where cond is False are replaced with
            corresponding value from other. Callables are not
            supported. Default is None.

            DataFrame expects only Scalar or array like with scalars or
            dataframe with same dimension as self.

            Series expects only scalar or series like with same length
        inplace : bool, default False
            Whether to perform the operation in place on the data.

        Returns
        -------
        Same type as caller

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"A":[1, 4, 5], "B":[3, 5, 8]})
        >>> df.where(df % 2 == 0, [-1, -1])
           A  B
        0 -1 -1
        1  4 -1
        2 -1  8

        >>> ser = cudf.Series([4, 3, 2, 1, 0])
        >>> ser.where(ser > 2, 10)
        0     4
        1     3
        2    10
        3    10
        4    10
        dtype: int64
        >>> ser.where(ser > 2)
        0       4
        1       3
        2    <NA>
        3    <NA>
        4    <NA>
        dtype: int64

        .. pandas-compat::
            :meth:`pandas.DataFrame.where`, :meth:`pandas.Series.where`

            Note that ``where`` treats missing values as falsy,
            in parallel with pandas treatment of nullable data:

            >>> gsr = cudf.Series([1, 2, 3])
            >>> gsr.where([True, False, cudf.NA])
            0       1
            1    <NA>
            2    <NA>
            dtype: int64
            >>> gsr.where([True, False, False])
            0       1
            1    <NA>
            2    <NA>
            dtype: int64
        """
        raise NotImplementedError

    @_performance_tracking
    def _fillna(
        self,
        value: None | ScalarLike | Series = None,
        *,
        method: None | plc.replace.ReplacePolicy = None,
        axis: Axis | None = None,
        inplace: bool = False,
        limit: int | None = None,
        limit_area: Literal["inside", "outside", None] = None,
    ) -> Self | None:
        """Fill null values with ``value`` or specified ``method``.

        Parameters
        ----------
        value : scalar, Series-like or dict
            Value to use to fill nulls. If Series-like, null values
            are filled with values in corresponding indices.
            A dict can be used to provide different values to fill nulls
            in different columns. Cannot be used with ``method``.
        method : {'ffill', 'bfill'}, default None
            Method to use for filling null values in the dataframe or series.
            `ffill` propagates the last non-null values forward to the next
            non-null value. `bfill` propagates backward with the next non-null
            value. Cannot be used with ``value``.

            .. deprecated:: 24.04
                `method` is deprecated.

        Returns
        -------
        result : DataFrame, Series, or Index
            Copy with nulls filled.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, None], 'b': [3, None, 5]})
        >>> df
              a     b
        0     1     3
        1     2  <NA>
        2  <NA>     5
        >>> df.fillna(4)
           a  b
        0  1  3
        1  2  4
        2  4  5
        >>> df.fillna({'a': 3, 'b': 4})
           a  b
        0  1  3
        1  2  4
        2  3  5

        ``fillna`` on a Series object:

        >>> ser = cudf.Series(['a', 'b', None, 'c'])
        >>> ser
        0       a
        1       b
        2    None
        3       c
        dtype: object
        >>> ser.fillna('z')
        0    a
        1    b
        2    z
        3    c
        dtype: object

        ``fillna`` can also supports inplace operation:

        >>> ser.fillna('z', inplace=True)
        >>> ser
        0    a
        1    b
        2    z
        3    c
        dtype: object
        >>> df.fillna({'a': 3, 'b': 4}, inplace=True)
        >>> df
           a  b
        0  1  3
        1  2  4
        2  3  5

        ``fillna`` specified with fill ``method``

        >>> ser = cudf.Series([1, None, None, 2, 3, None, None])
        >>> ser.fillna(method='ffill')
        0    1
        1    1
        2    1
        3    2
        4    3
        5    3
        6    3
        dtype: int64
        >>> ser.fillna(method='bfill')
        0       1
        1       2
        2       2
        3       2
        4       3
        5    <NA>
        6    <NA>
        dtype: int64
        """
        if limit is not None:
            raise NotImplementedError("The limit keyword is not supported")
        if limit_area is not None:
            raise NotImplementedError("limit_area is currently not supported.")
        if axis:
            raise NotImplementedError("The axis keyword is not supported")

        if value is not None and method is not None:
            raise ValueError("Cannot specify both 'value' and 'method'.")

        if is_scalar(value):
            value = {name: value for name in self._column_names}
        elif not isinstance(value, (Mapping, cudf.Series)):
            raise TypeError(
                f'"value" parameter must be a scalar, dict '
                f"or Series, but you passed a "
                f'"{type(value).__name__}"'
            )

        filled_columns = [
            col.fillna(value[name], method) if name in value else col.copy()
            for name, col in self._column_labels_and_values
        ]

        return self._mimic_inplace(
            self._from_data_like_self(
                self._data._from_columns_like_self(
                    filled_columns, verify=False
                )
            ),
            inplace=inplace,
        )

    def _pandas_repr_compatible(self, nan_rep=None) -> Self:
        """Return Self but with columns prepared for a pandas-like repr."""
        columns = (
            col._prep_pandas_compat_repr(nan_rep) for col in self._columns
        )
        return self._from_data_like_self(
            self._data._from_columns_like_self(columns, verify=False)
        )

    @classmethod
    @_performance_tracking
    def from_arrow(cls, data: pa.Table) -> Self:
        """Convert from PyArrow Table to Frame

        Parameters
        ----------
        data : PyArrow Table

        Raises
        ------
        TypeError for invalid input type.

        Examples
        --------
        >>> import cudf
        >>> import pyarrow as pa
        >>> data = pa.table({"a":[1, 2, 3], "b":[4, 5, 6]})
        >>> cudf.core.frame.Frame.from_arrow(data)
           a  b
        0  1  4
        1  2  5
        2  3  6
        """
        if not isinstance(data, pa.Table):
            raise TypeError(
                f"data must be a pyarrow.Table, not {type(data).__name__}"
            )
        ca = ColumnAccessor(
            {
                name: ColumnBase.from_arrow(array)
                for name, array in zip(
                    data.column_names, data.itercolumns(), strict=True
                )
            },
            verify=False,
        )
        return cls._from_data(ca)

    @_performance_tracking
    def to_arrow(self) -> pa.Table:
        """
        Convert to arrow Table

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame(
        ...     {"a":[1, 2, 3], "b":[4, 5, 6]}, index=[1, 2, 3])
        >>> df.to_arrow()
        pyarrow.Table
        a: int64
        b: int64
        index: int64
        ----
        a: [[1,2,3]]
        b: [[4,5,6]]
        index: [[1,2,3]]
        """
        return pa.Table.from_pydict(
            {
                str(name): col.to_arrow()
                for name, col in self._column_labels_and_values
            }
        )

    @_performance_tracking
    def isna(self) -> Self:
        """
        Identify missing values.

        Return a boolean same-sized object indicating if
        the values are ``<NA>``. ``<NA>`` values gets mapped to
        ``True`` values. Everything else gets mapped to
        ``False`` values. ``<NA>`` values include:

        * Values where null mask is set.
        * ``NaN`` in float dtype.
        * ``NaT`` in datetime64 and timedelta64 types.

        Characters such as empty strings ``''`` or
        ``inf`` in case of float are not
        considered ``<NA>`` values.

        Returns
        -------
        DataFrame/Series/Index
            Mask of bool values for each element in
            the object that indicates whether an element is an NA value.

        Examples
        --------
        Show which entries in a DataFrame are NA.

        >>> import cudf
        >>> import numpy as np
        >>> import pandas as pd
        >>> df = cudf.DataFrame({'age': [5, 6, np.nan],
        ...                    'born': [pd.NaT, pd.Timestamp('1939-05-27'),
        ...                             pd.Timestamp('1940-04-25')],
        ...                    'name': ['Alfred', 'Batman', ''],
        ...                    'toy': [None, 'Batmobile', 'Joker']})
        >>> df
            age                           born    name        toy
        0     5                        NaT  Alfred        NaN
        1     6  1939-05-27 00:00:00.000000  Batman  Batmobile
        2  <NA>  1940-04-25 00:00:00.000000              Joker
        >>> df.isna()
             age   born   name    toy
        0  False   True  False   True
        1  False  False  False  False
        2   True  False  False  False

        Show which entries in a Series are NA.

        >>> ser = cudf.Series([5, 6, np.nan, np.inf, -np.inf])
        >>> ser
        0     5.0
        1     6.0
        2     NaN
        3     Inf
        4    -Inf
        dtype: float64
        >>> ser.isna()
        0    False
        1    False
        2     True
        3    False
        4    False
        dtype: bool

        Show which entries in an Index are NA.

        >>> idx = cudf.Index([1, 2, None, np.nan, 0.32, np.inf])
        >>> idx
        Index([1.0, 2.0, NaN, NaN, 0.32, Inf], dtype='float64')
        >>> idx.isna()
        array([False, False,  True,  True, False, False])
        """
        data_columns = (col.isnull() for col in self._columns)
        return self._from_data_like_self(
            self._data._from_columns_like_self(data_columns)
        )

    # Alias for isna
    isnull = isna

    @_performance_tracking
    def notna(self) -> Self:
        """
        Identify non-missing values.

        Return a boolean same-sized object indicating if
        the values are not ``<NA>``. Non-missing values get
        mapped to ``True``. ``<NA>`` values get mapped to
        ``False`` values. ``<NA>`` values include:

        * Values where null mask is set.
        * ``NaN`` in float dtype.
        * ``NaT`` in datetime64 and timedelta64 types.

        Characters such as empty strings ``''`` or
        ``inf`` in case of float are not
        considered ``<NA>`` values.

        Returns
        -------
        DataFrame/Series/Index
            Mask of bool values for each element in
            the object that indicates whether an element is not an NA value.

        Examples
        --------
        Show which entries in a DataFrame are NA.

        >>> import cudf
        >>> import numpy as np
        >>> import pandas as pd
        >>> df = cudf.DataFrame({'age': [5, 6, np.nan],
        ...                    'born': [pd.NaT, pd.Timestamp('1939-05-27'),
        ...                             pd.Timestamp('1940-04-25')],
        ...                    'name': ['Alfred', 'Batman', ''],
        ...                    'toy': [None, 'Batmobile', 'Joker']})
        >>> df
            age                           born    name        toy
        0     5                        NaT  Alfred        NaN
        1     6  1939-05-27 00:00:00.000000  Batman  Batmobile
        2  <NA>  1940-04-25 00:00:00.000000              Joker
        >>> df.notna()
             age   born  name    toy
        0   True  False  True  False
        1   True   True  True   True
        2  False   True  True   True

        Show which entries in a Series are NA.

        >>> ser = cudf.Series([5, 6, np.nan, np.inf, -np.inf])
        >>> ser
        0     5.0
        1     6.0
        2     NaN
        3     Inf
        4    -Inf
        dtype: float64
        >>> ser.notna()
        0     True
        1     True
        2    False
        3     True
        4     True
        dtype: bool

        Show which entries in an Index are NA.

        >>> idx = cudf.Index([1, 2, None, np.nan, 0.32, np.inf])
        >>> idx
        Index([1.0, 2.0, NaN, NaN, 0.32, Inf], dtype='float64')
        >>> idx.notna()
        array([ True,  True, False, False,  True,  True])
        """
        data_columns = (col.notnull() for col in self._columns)
        return self._from_data_like_self(
            self._data._from_columns_like_self(data_columns)
        )

    # Alias for notna
    notnull = notna

    @_performance_tracking
    def searchsorted(
        self,
        values,
        side: Literal["left", "right"] = "left",
        sorter=None,
        ascending: bool = True,
        na_position: Literal["first", "last"] = "last",
    ) -> ScalarLike | cupy.ndarray:
        """Find indices where elements should be inserted to maintain order

        Parameters
        ----------
        value : Frame (Shape must be consistent with self)
            Values to be hypothetically inserted into Self
        side : str {'left', 'right'} optional, default 'left'
            If 'left', the index of the first suitable location found is given
            If 'right', return the last such index
        sorter : 1-D array-like, optional
            Optional array of integer indices that sort `self` into ascending
            order. They are typically the result of ``np.argsort``.
            Currently not supported.
        ascending : bool optional, default True
            Sorted Frame is in ascending order (otherwise descending)
        na_position : str {'last', 'first'} optional, default 'last'
            Position of null values in sorted order

        Returns
        -------
        1-D cupy array of insertion points

        Examples
        --------
        >>> s = cudf.Series([1, 2, 3])
        >>> s.searchsorted(4)
        3
        >>> s.searchsorted([0, 4])
        array([0, 3])
        >>> s.searchsorted([1, 3], side='left')
        array([0, 2])
        >>> s.searchsorted([1, 3], side='right')
        array([1, 3])

        If the values are not monotonically sorted, wrong
        locations may be returned:

        >>> s = cudf.Series([2, 1, 3])
        >>> s.searchsorted(1)
        0

        >>> df = cudf.DataFrame({'a': [1, 3, 5, 7], 'b': [10, 12, 14, 16]})
        >>> df
           a   b
        0  1  10
        1  3  12
        2  5  14
        3  7  16
        >>> values_df = cudf.DataFrame({'a': [0, 2, 5, 6],
        ... 'b': [10, 11, 13, 15]})
        >>> values_df
           a   b
        0  0  10
        1  2  11
        2  5  13
        3  6  15
        >>> df.searchsorted(values_df, ascending=False)
        array([4, 4, 4, 0])
        """
        # Note: pandas.DataFrame does not support searchsorted

        if na_position not in {"first", "last"}:
            raise ValueError(f"invalid na_position: {na_position}")
        elif sorter is not None:
            raise NotImplementedError("sorter is currently not supported.")

        scalar_flag = None
        if is_scalar(values):
            scalar_flag = True

        if not isinstance(values, Frame):
            values = [as_column(values)]
        else:
            values = [*values._columns]
        if len(values) != self._num_columns:
            raise ValueError("Mismatch number of columns to search for.")

        if cudf.get_option("mode.pandas_compatible"):
            if any(
                is_pandas_nullable_extension_dtype(col.dtype)
                and col.has_nulls(include_nan=True)
                for col in self._columns
            ):
                raise ValueError(
                    "searchsorted requires array to be sorted, which is impossible "
                    "with NAs present."
                )
        # TODO: Change behavior based on the decision in
        # https://github.com/pandas-dev/pandas/issues/54668
        common_dtype_list = [
            find_common_type([col.dtype, val.dtype])
            for col, val in zip(self._columns, values, strict=True)
        ]
        sources = [
            col
            if is_dtype_equal(col.dtype, common_dtype)
            else col.astype(common_dtype)
            for col, common_dtype in zip(
                self._columns, common_dtype_list, strict=True
            )
        ]
        values = [
            val
            if is_dtype_equal(val.dtype, common_dtype)
            else val.astype(common_dtype)
            for val, common_dtype in zip(
                values, common_dtype_list, strict=True
            )
        ]

        plc_outcol = sorting.search_sorted(
            sources,
            values,
            side,
            ascending=itertools.repeat(ascending, times=len(sources)),
            na_position=itertools.repeat(na_position, times=len(sources)),
        )
        outcol = ColumnBase.create(
            plc.unary.cast(plc_outcol, plc.DataType(plc.TypeId.INT64)),
            np.dtype("int64"),
        )

        # Return result as cupy array if the values is non-scalar
        # If values is scalar, result is expected to be scalar.
        result = outcol.values
        if scalar_flag:
            return result[0].item()
        else:
            return result

    @_performance_tracking
    def argsort(
        self,
        by=None,
        axis=0,
        kind="quicksort",
        order=None,
        ascending=True,
        na_position="last",
    ) -> cupy.ndarray:
        """Return the integer indices that would sort the Series values.

        Parameters
        ----------
        by : str or list of str, default None
            Name or list of names to sort by. If None, sort by all columns.
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

        Examples
        --------
        **Series**

        >>> import cudf
        >>> s = cudf.Series([3, 1, 2])
        >>> s
        0    3
        1    1
        2    2
        dtype: int64
        >>> s.argsort()
        0    1
        1    2
        2    0
        dtype: int64
        >>> s[s.argsort()]
        1    1
        2    2
        0    3
        dtype: int64

        **DataFrame**
        >>> import cudf
        >>> df = cudf.DataFrame({'foo': [3, 1, 2]})
        >>> df.argsort()
        array([1, 2, 0])

        **Index**
        >>> import cudf
        >>> idx = cudf.Index([3, 1, 2])
        >>> idx.argsort()
        array([1, 2, 0])
        """
        if na_position not in {"first", "last"}:
            raise ValueError(f"invalid na_position: {na_position}")
        if kind != "quicksort":
            if kind not in {"mergesort", "heapsort", "stable"}:
                raise AttributeError(
                    f"{kind} is not a valid sorting algorithm for "
                    f"'DataFrame' object"
                )
            warnings.warn(
                f"GPU-accelerated {kind} is currently not supported, "
                "defaulting to quicksort."
            )

        if isinstance(by, str):
            by = [by]
        # numpy and pandas return ``np.intp`` (int64) positional indexers;
        # return the same so cuDF's ``argsort`` output dtype matches.
        return self._get_sorted_inds(
            by=by, ascending=ascending, na_position=na_position
        ).values.astype(np.intp)

    @_performance_tracking
    def _get_sorted_inds(
        self,
        by=None,
        ascending: bool | Iterable[bool] = True,
        na_position: Literal["first", "last"] = "last",
    ) -> ColumnBase:
        """
        Get the indices required to sort self according to the columns
        specified in by.
        """
        if by is None:
            to_sort = self._columns
        else:
            to_sort = self._get_columns_by_label(list(by))._columns

        if isinstance(ascending, bool):
            ascending_iter: Iterable[bool] = itertools.repeat(
                ascending, times=len(to_sort)
            )
        else:
            ascending_iter = ascending
        plc_result = sorting.order_by(
            to_sort,
            ascending_iter,
            itertools.repeat(na_position, times=len(to_sort)),
            stable=True,
        )
        return ColumnBase.create(
            plc_result, dtype_from_pylibcudf_column(plc_result)
        )

    @_performance_tracking
    def _split(self, splits: list[int]) -> list[Self]:
        """
        Split a frame with split points in ``splits``.

        Returns
        -------
        list[Self]
            Returns a list of Self of length len(splits) + 1.
        """
        if self._num_rows == 0:
            return []
        return [
            self._from_columns_like_self(
                [
                    ColumnBase.create(col, dtype)
                    for col, (_, dtype) in zip(
                        split, self._dtypes, strict=True
                    )
                ],
                self._column_names,
            )
            for split in copying.columns_split(self._columns, splits)
        ]

    @_performance_tracking
    def _encode(self) -> tuple[Self, ColumnBase]:
        plc_table, plc_column = plc.transform.encode(
            plc.Table([col.plc_column for col in self._columns])
        )
        columns = [
            ColumnBase.create(col, dtype)
            for col, (_, dtype) in zip(
                plc_table.columns(), self._dtypes, strict=True
            )
        ]
        indices = ColumnBase.create(
            plc_column, dtype_from_pylibcudf_column(plc_column)
        )
        keys = self._from_columns_like_self(columns)
        return keys, indices

    @_performance_tracking
    def _unaryop(self, op: str) -> Self:
        data_columns = (col.unary_operator(op) for col in self._columns)
        return self._from_data_like_self(
            self._data._from_columns_like_self(data_columns)
        )

    @staticmethod
    @_performance_tracking
    def _colwise_binop(
        operands: dict[str | None, tuple[ColumnBase, Any, bool, Any]],
        fn: str,
    ):
        """Implement binary ops between two frame-like objects.

        Binary operations for Frames can be reduced to a sequence of binary
        operations between column-like objects. Different types of frames need
        to preprocess different inputs, so subclasses should implement binary
        operations as a preprocessing step that calls this method.

        Parameters
        ----------
        operands : Dict[Optional[str], Tuple[ColumnBase, Any, bool, Any]]
            A mapping from column names to a tuple containing left and right
            operands as well as a boolean indicating whether or not to reflect
            an operation and fill value for nulls.
        fn : str
            The operation to perform.

        Returns
        -------
        Dict[ColumnBase]
            A dict of columns constructed from the result of performing the
            requested operation on the operands.
        """
        # Now actually perform the binop on the columns in left and right.
        output = {}
        for (
            col,
            (left_column, right_column, reflect, fill_value),
        ) in operands.items():
            output_mask: Buffer | None = None
            output_null_count: int | None = None
            left_is_column = isinstance(left_column, ColumnBase)
            right_is_column = isinstance(right_column, ColumnBase)
            if fill_value is not None:
                if left_is_column and right_is_column:
                    # ``has_nulls(include_nan=True)`` covers both validity-mask
                    # nulls and NaN-as-null floats (the latter is how columns
                    # are represented in ``mode.pandas_compatible``).
                    left_has_nulls = left_column.has_nulls(include_nan=True)
                    right_has_nulls = right_column.has_nulls(include_nan=True)
                    if left_has_nulls and right_has_nulls:
                        # Pandas semantics: nulls that are present in both
                        # left and right at the same position remain null in
                        # the output even when ``fill_value`` is given.
                        output_validity = (
                            left_column.notnull() | right_column.notnull()
                        )
                        output_mask, output_null_count = (
                            output_validity.as_mask()
                        )
                        left_column = left_column.fillna(fill_value)
                        right_column = right_column.fillna(fill_value)
                    elif left_has_nulls:
                        left_column = left_column.fillna(fill_value)
                    elif right_has_nulls:
                        right_column = right_column.fillna(fill_value)
                elif left_is_column:
                    if left_column.has_nulls(include_nan=True):
                        left_column = left_column.fillna(fill_value)
                elif right_is_column:
                    if right_column.has_nulls(include_nan=True):
                        right_column = right_column.fillna(fill_value)
                else:
                    assert False, "At least one operand must be a column."

            # TODO: Disable logical and binary operators between columns that
            # are not numerical using the new binops mixin.

            outcol = (
                getattr(operator, fn)(right_column, left_column)
                if reflect
                else getattr(operator, fn)(left_column, right_column)
            )
            if (
                isinstance(outcol, bool)
                and fn in {"__eq__", "__ne__"}
                and right_column is not pd.NA
            ):
                # Both columns returned NotImplemented, Python compared using is/is not
                # TODO: A better solution is to ensure each Column._binaryop
                # implementation accounts for this case.
                outcol = left_column._all_bools_with_nulls(
                    right_column, fn != "__eq__"
                )

            if output_mask is not None:
                outcol = outcol.set_mask(output_mask, output_null_count)

            output[col] = outcol

        return output

    @_performance_tracking
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _array_ufunc(self, ufunc, method, inputs, kwargs)

    @_performance_tracking
    def _apply_cupy_ufunc_to_operands(
        self, ufunc, cupy_func, operands, **kwargs
    ) -> list[dict[Any, ColumnBase]]:
        # Note: There are some operations that may be supported by libcudf but
        # are not supported by pandas APIs. In particular, libcudf binary
        # operations support logical and/or operations as well as
        # trigonometric, but those operations are not defined on
        # pd.Series/DataFrame. For now those operations will dispatch to cupy,
        # but if ufuncs are ever a bottleneck we could add special handling to
        # dispatch those (or any other) functions that we could implement
        # without cupy.

        from cudf.utils.dtypes import get_dtype_of_same_kind

        with access_columns(
            *(
                col
                for left, right, _, _ in operands.values()
                for col in (left, right)
                if isinstance(col, ColumnBase)
            ),
            mode="read",
            scope="internal",
        ):
            mask = None
            data: list[dict[Any, ColumnBase]] = [{} for _ in range(ufunc.nout)]
            for name, (left, right, _, _) in operands.items():
                cupy_inputs = []
                source_dtype = None
                for inp in (left, right) if ufunc.nin == 2 else (left,):
                    if (
                        isinstance(inp, ColumnBase)
                        and source_dtype is None
                        and is_pandas_nullable_extension_dtype(inp.dtype)
                    ):
                        source_dtype = inp.dtype
                    if isinstance(inp, ColumnBase) and inp.has_nulls():
                        new_mask = inp._get_mask_as_column()
                        mask = new_mask if mask is None else mask & new_mask

                        # Arbitrarily fill with zeros. For ufuncs, we assume
                        # that the end result propagates nulls via a bitwise
                        # and, so these elements are irrelevant.
                        inp = inp.fillna(0)
                    cupy_inputs.append(cupy.asarray(inp))

                cp_output = cupy_func(*cupy_inputs, **kwargs)
                if ufunc.nout == 1:
                    cp_output = (cp_output,)
                if mask is None:
                    mask_buff = None
                    null_count = 0
                else:
                    mask_buff, null_count = mask.as_mask()
                for i, out in enumerate(cp_output):
                    target_dtype = (
                        get_dtype_of_same_kind(source_dtype, out.dtype)
                        if source_dtype is not None
                        else None
                    )
                    data[i][name] = as_column(
                        out, dtype=target_dtype
                    ).set_mask(mask_buff, null_count)
            return data

    # Unary logical operators
    @_performance_tracking
    def __neg__(self) -> Self:
        """Negate for integral dtypes, logical NOT for bools."""
        return self._from_data_like_self(
            self._data._from_columns_like_self(
                (
                    col.unary_operator("not")
                    if col.dtype.kind == "b"
                    else col.unary_operator("negate")
                    for col in self._columns
                )
            )
        )

    @_performance_tracking
    def __pos__(self) -> Self:
        return self.copy(deep=True)

    @_performance_tracking
    def __abs__(self) -> Self:
        return self._unaryop("abs")

    def __bool__(self) -> None:
        raise ValueError(
            f"The truth value of a {type(self).__name__} is ambiguous. Use "
            "a.empty, a.bool(), a.item(), a.any() or a.all()."
        )

    @_performance_tracking
    def _reduce(
        self,
        op: str,
        axis: Axis | None = 0,
        numeric_only: bool = False,
        **kwargs,
    ) -> ScalarLike:
        raise NotImplementedError(
            f"Reductions are not supported for objects of type {type(self).__name__}."
        )

    @_performance_tracking
    def _scan(
        self,
        op: str,
        axis: Axis | None = None,
        skipna: bool = True,
        *args,
        **kwargs,
    ) -> Self:
        raise NotImplementedError(
            f"Scans are not supported for objects of type {type(self).__name__}."
        )

    @_performance_tracking
    def min(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ):
        """
        Return the minimum of the values in the DataFrame.

        Parameters
        ----------
        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values when computing the result.
        numeric_only: bool, default False
            If True, includes only float, int, boolean columns.
            If False, will raise error in-case there are
            non-numeric columns.

        Returns
        -------
        Series

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> min_series = df.min()
        >>> min_series
        a    1
        b    7
        dtype: int64
        >>> min_series.min()
        np.int64(1)

        .. pandas-compat::
            :meth:`pandas.DataFrame.min`, :meth:`pandas.Series.min`

            Parameters currently not supported are `level`, `numeric_only`.
        """
        return self._reduce(
            "min",
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            **kwargs,
        )

    @_performance_tracking
    def max(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ):
        """
        Return the maximum of the values in the DataFrame.

        Parameters
        ----------
        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values when computing the result.
        numeric_only: bool, default False
            If True, includes only float, int, boolean columns.
            If False, will raise error in-case there are
            non-numeric columns.

        Returns
        -------
        Series

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.max()
        a     4
        b    10
        dtype: int64

        .. pandas-compat::
            :meth:`pandas.DataFrame.max`, :meth:`pandas.Series.max`

            Parameters currently not supported are `level`, `numeric_only`.
        """
        return self._reduce(
            "max",
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            **kwargs,
        )

    @_performance_tracking
    def all(self, axis: Axis | None = 0, skipna: bool = True, **kwargs):
        """
        Return whether all elements are True in DataFrame.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Indicate which axis or axes should be reduced. For `Series`
            this parameter is unused and defaults to `0`.

            - 0 or 'index' : reduce the index, return a Series
                whose index is the original column labels.
            - 1 or 'columns' : reduce the columns, return a Series
                whose index is the original index.
            - None : reduce all axes, return a scalar.
        skipna: bool, default True
            Exclude NA/null values. If the entire row/column is NA and
            skipna is True, then the result will be True, as for an
            empty row/column.
            If skipna is False, then NA are treated as True, because
            these are not equal to zero.

        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported are `bool_only`.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [3, 2, 3, 4], 'b': [7, 0, 10, 10]})
        >>> df.all()
        a     True
        b    False
        dtype: bool

        .. pandas-compat::
            :meth:`pandas.DataFrame.all`, :meth:`pandas.Series.all`

            Parameters currently not supported are `axis`, `bool_only`,
            `level`.
        """
        return self._reduce(
            "all",
            axis=axis,
            skipna=skipna,
            **kwargs,
        )

    @_performance_tracking
    def any(self, axis: Axis | None = 0, skipna: bool = True, **kwargs):
        """
        Return whether any elements is True in DataFrame.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Indicate which axis or axes should be reduced. For `Series`
            this parameter is unused and defaults to `0`.

            - 0 or 'index' : reduce the index, return a Series
                whose index is the original column labels.
            - 1 or 'columns' : reduce the columns, return a Series
                whose index is the original index.
            - None : reduce all axes, return a scalar.
        skipna: bool, default True
            Exclude NA/null values. If the entire row/column is NA and
            skipna is True, then the result will be False, as for an
            empty row/column.
            If skipna is False, then NA are treated as True, because
            these are not equal to zero.

        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported are `bool_only`.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [3, 2, 3, 4], 'b': [7, 0, 10, 10]})
        >>> df.any()
        a    True
        b    True
        dtype: bool

        .. pandas-compat::
            :meth:`pandas.DataFrame.any`, :meth:`pandas.Series.any`

            Parameters currently not supported are `axis`, `bool_only`,
            `level`.
        """
        return self._reduce(
            "any",
            axis=axis,
            skipna=skipna,
            **kwargs,
        )

    @_performance_tracking
    def __str__(self) -> str:
        return repr(self)

    @_performance_tracking
    def __deepcopy__(self, memo) -> Self:
        return self.copy(deep=True)

    @_performance_tracking
    def __copy__(self) -> Self:
        return self.copy(deep=False)

    def copy(self, deep: bool = True) -> Self:
        raise NotImplementedError

    @_performance_tracking
    def __invert__(self) -> Self:
        """Bitwise invert (~) for integral dtypes, logical NOT for bools."""
        return self._from_data_like_self(
            self._data._from_columns_like_self((~col for col in self._columns))
        )

    @staticmethod
    @_performance_tracking
    def _repeat(
        columns: list[ColumnBase], repeats, axis=None
    ) -> list[ColumnBase]:
        if axis is not None:
            raise NotImplementedError(
                "Only axis=`None` supported at this time."
            )

        if not is_scalar(repeats):
            repeats = as_column(repeats)

        with access_columns(
            *columns, repeats, mode="read", scope="internal"
        ) as (*columns, repeats):
            plc_table = plc.Table([col.plc_column for col in columns])
            if isinstance(repeats, ColumnBase):
                repeats_plc = repeats.plc_column
            else:
                repeats_plc = repeats
            return [
                ColumnBase.create(col, reference_col.dtype)
                for col, reference_col in zip(
                    plc.filling.repeat(plc_table, repeats_plc).columns(),
                    columns,
                    strict=True,
                )
            ]

    @_performance_tracking
    @_warn_no_dask_cudf
    def __dask_tokenize__(self):
        from dask.base import normalize_token

        return [
            type(self),
            str(dict(self._dtypes)),
            normalize_token(self.to_pandas()),
        ]
