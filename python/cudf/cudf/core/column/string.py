# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
import pandas as pd
import pyarrow as pa

import pylibcudf as plc

import cudf
from cudf.api.types import is_scalar
from cudf.core._internals import binaryop
from cudf.core.column.column import (
    ColumnBase,
    ColumnList,
    PylibcudfFunction,
    as_column,
    bool_same_kind_policy,
    column_empty,
    list_dtype_policy,
    pandas_string_int32_policy,
    pylibcudf_result_dtype_policy,
    same_dtype_policy,
)
from cudf.core.dtype.validators import is_dtype_obj_string
from cudf.core.mixins import Scannable
from cudf.errors import MixedTypeError
from cudf.utils.dtypes import (
    cudf_dtype_to_pa_type,
    dtype_from_pylibcudf_column,
    dtype_to_pylibcudf_type,
    get_dtype_of_same_kind,
    is_pandas_nullable_extension_dtype,
)
from cudf.utils.scalar import pa_scalar_to_plc_scalar
from cudf.utils.temporal import (
    infer_format,
    raise_if_datetime_seconds_out_of_bounds,
)
from cudf.utils.utils import is_na_like

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    import cupy as cp

    from cudf._typing import (
        ColumnBinaryOperand,
        ColumnLike,
        DtypeObj,
        ScalarLike,
    )
    from cudf.core.column.datetime import DatetimeColumn
    from cudf.core.column.decimal import DecimalColumn
    from cudf.core.column.lists import ListColumn
    from cudf.core.column.numerical import NumericalColumn
    from cudf.core.column.timedelta import TimeDeltaColumn
    from cudf.core.dtypes import DecimalDtype


# For now all supported re flags have matching names in libcudf. If that ever changes
# this construction will need to be updated with more explicit mapping.
_FLAG_MAP = {
    getattr(re, flag): getattr(plc.strings.regex_flags.RegexFlags, flag)
    for flag in ("MULTILINE", "DOTALL", "IGNORECASE")
}


@lru_cache
def plc_flags_from_re_flags(
    flags: re.RegexFlag,
) -> plc.strings.regex_flags.RegexFlags:
    # Convert Python re flags to pylibcudf RegexFlags
    plc_flags = plc.strings.regex_flags.RegexFlags(0)
    for re_flag, plc_flag in _FLAG_MAP.items():
        if flags & re_flag:
            plc_flags |= plc_flag
            flags &= ~re_flag
    if flags:
        raise ValueError(f"Unsupported re flags: {flags}")
    return plc_flags


def _infer_timedelta_format(col: StringColumn) -> str:
    """Infer strptime format from the first non-null timedelta string."""
    nonnull = col.dropna()
    if len(nonnull) == 0 or ":" in nonnull.element_indexing(0):
        return "%D days %H:%M:%S"
    return "%D days"


class StringColumn(ColumnBase, Scannable):
    """Implements operations for Columns of String type"""

    _VALID_BINARY_OPERATIONS = {
        "__eq__",
        "__ne__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__add__",
        "__radd__",
        # These operators aren't actually supported, they only exist to allow
        # empty column binops with scalars of arbitrary other dtypes. See
        # the _binaryop method for more information.
        "__sub__",
        "__mul__",
        "__mod__",
        "__pow__",
        "__truediv__",
        "__floordiv__",
    }
    _VALID_SCANS = {
        "cummin",
        "cummax",
    }

    @property
    def _PANDAS_NA_VALUE(self) -> ScalarLike:
        """Pandas NA value, mostly for a repr"""
        if isinstance(self.dtype, (pd.StringDtype, pd.ArrowDtype)):
            return self.dtype.na_value
        # np.nan is also a valid NA value if dtype=object
        # https://github.com/pandas-dev/pandas/pull/63900#discussion_r2740653899
        return None

    @property
    def __cuda_array_interface__(self) -> Mapping[str, Any]:
        raise NotImplementedError(
            f"dtype {self.dtype} is not yet supported via "
            "`__cuda_array_interface__`"
        )

    def _validate_fillna_value(
        self, fill_value: ScalarLike | ColumnLike
    ) -> plc.Scalar | ColumnBase:
        """Align fill_value for .fillna based on column type."""
        if is_scalar(fill_value) and fill_value is np.nan:
            raise MixedTypeError("Cannot fill `np.nan` in string column")
        return super()._validate_fillna_value(fill_value)

    def _cast_setitem_value(self, value: Any) -> plc.Scalar | ColumnBase:
        if cudf.get_option("mode.pandas_compatible"):
            # pandas string dtype rejects non-string values (int/float/bool).
            if is_scalar(value):
                if (
                    value is not None
                    and not is_na_like(value)
                    and not isinstance(value, (str, np.str_))
                ):
                    raise TypeError(
                        f"Invalid value '{value}' for dtype '{self.dtype}'"
                    )
            elif isinstance(value, ColumnBase):
                if value.dtype.kind in {"i", "u", "f", "b"}:
                    raise TypeError(
                        f"Invalid value '{value}' for dtype '{self.dtype}'"
                    )
            elif getattr(getattr(value, "dtype", None), "kind", None) in {
                "i",
                "u",
                "f",
                "b",
            }:
                raise TypeError(
                    f"Invalid value '{value}' for dtype '{self.dtype}'"
                )
            else:
                # Plain Python sequences without a dtype attribute (e.g.
                # list/tuple): inspect element types.
                try:
                    iterator = iter(value)
                except TypeError:
                    iterator = None
                if iterator is not None:
                    for v in iterator:
                        if (
                            v is None
                            or is_na_like(v)
                            or isinstance(v, (str, np.str_))
                        ):
                            continue
                        raise TypeError(
                            f"Invalid value '{value}' for dtype '{self.dtype}'"
                        )
        return super()._cast_setitem_value(value)

    def sum(
        self,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs: Any,
    ) -> ScalarLike:
        col = self.nans_to_nulls() if skipna else self
        if not skipna and col.has_nulls():
            return pd.NA
        if skipna:
            col = col.dropna()

        if min_count > 0 and col.valid_count < min_count:
            return pd.NA

        if len(col) == 0:
            # pandas sums an empty/all-null object series to the numeric
            # identity 0; genuine string dtypes concatenate to "".
            return 0 if self.dtype == np.dtype("object") else ""
        return col.join_strings("", None).element_indexing(0)

    def any(
        self, skipna: bool = True, min_count: int = 0, **kwargs: Any
    ) -> ScalarLike:
        """Check if any string value is truthy (non-empty)."""
        if not skipna and self.has_nulls():
            # pandas 3 treats the NaN null sentinel as truthy, matching
            # numpy semantics, so any(skipna=False) returns True when nulls
            # are present.
            return True
        elif skipna and self.null_count == self.size:
            return False
        raise NotImplementedError("`any` not implemented for `StringColumn`")

    def all(
        self, skipna: bool = True, min_count: int = 0, **kwargs: Any
    ) -> ScalarLike:
        """Check if all string values are truthy (non-empty)."""
        if self.null_count == self.size:
            # With skipna=True nulls are dropped, so all-null is vacuously
            # True. pandas 3 treats the NaN null sentinel as truthy,
            # matching numpy semantics, so all(skipna=False) is True too.
            # A partially-null column must NOT short-circuit here: the
            # result depends on the truthiness of the non-null strings
            # (e.g. all([NaN, ""], skipna=False) is False).
            return True
        raise NotImplementedError("`all` not implemented for `StringColumn`")

    def _reduce(
        self,
        op: str,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs: Any,
    ) -> ScalarLike:
        """Validate reduction operations for StringColumn."""
        if op in {"min", "max"}:
            return super()._reduce(op, skipna, min_count, **kwargs)

        # For empty columns with statistical operations, return NaN (pandas behavior)
        if len(self) == 0 and op in {"var", "std", "mean", "sum_of_squares"}:
            return np.nan

        raise TypeError(f"Series.{op} does not support StringColumn")

    def __contains__(self, item: ScalarLike) -> bool:
        other = [item] if is_scalar(item) else item
        return self.contains(as_column(other, dtype=self.dtype)).any()

    def as_numerical_column(self, dtype: np.dtype) -> NumericalColumn:
        if dtype.kind == "b":
            result = self.count_characters() > np.int8(0)
            if not is_pandas_nullable_extension_dtype(dtype):
                result = result.fillna(False)
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(result.plc_column, dtype),
            )

        cast_func: Callable[[plc.Column, plc.DataType], plc.Column]
        data = self
        if dtype.kind in {"i", "u"}:
            if data.str_contains("_").any():
                # Python (PEP 515) allows underscores between digits in
                # integer literals (e.g. "123_1" == 1231) but libcudf does
                # not. Strip only the valid underscores so invalid ones
                # (e.g. "_12", "12_", "1__2") still fail like pandas.
                # Two passes: the first can skip an underscore whose
                # neighboring digit was consumed by a previous match.
                data = data.replace_with_backrefs(r"(\d)_(\d)", r"\1\2")
                data = data.replace_with_backrefs(r"(\d)_(\d)", r"\1\2")
            if not data.is_all_integer():
                raise ValueError(
                    "Could not convert strings to integer "
                    "type due to presence of non-integer values."
                )
            cast_func = plc.strings.convert.convert_integers.to_integers
        elif dtype.kind == "f":
            if not self.is_all_float():
                raise ValueError(
                    "Could not convert strings to float "
                    "type due to presence of non-floating values."
                )
            cast_func = plc.strings.convert.convert_floats.to_floats
        else:
            raise ValueError(f"dtype must be a numerical type, not {dtype}")
        plc_dtype = dtype_to_pylibcudf_type(dtype)
        with data.access(mode="read", scope="internal"):
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(
                    cast_func(data.plc_column, plc_dtype), dtype
                ),
            )

    def strptime(
        self, dtype: DtypeObj, format: str
    ) -> DatetimeColumn | TimeDeltaColumn:
        if dtype.kind not in "Mm":
            raise ValueError(
                f"dtype must be datetime or timedelta type, not {dtype}"
            )
        elif self.is_all_null:
            return column_empty(len(self), dtype=dtype)  # type: ignore[return-value]
        elif (self == "None").any():
            raise ValueError(
                "Cannot convert `None` value to datetime or timedelta."
            )

        casting_func: Callable[[plc.Column, plc.DataType, str], plc.Column]
        if dtype.kind == "M":
            if format.endswith("%z"):
                raise NotImplementedError(
                    "cuDF does not yet support timezone-aware datetimes"
                )
            is_nat = self == "NaT"
            without_nat = self.apply_boolean_mask(is_nat.unary_operator("not"))
            char_counts = without_nat.count_characters()  # type: ignore[attr-defined]
            if char_counts.distinct_count(dropna=True) != 1:
                # Unfortunately disables OK cases like:
                # ["2020-01-01", "2020-01-01 00:00:00"]
                # But currently incorrect for cases like (drops 10):
                # ["2020-01-01", "2020-01-01 10:00:00"]
                raise NotImplementedError(
                    "Cannot parse date-like strings with different formats"
                )
            valid_ts = self.is_timestamp(format)
            valid = valid_ts | is_nat
            if not valid.all():
                raise ValueError(f"Column contains invalid data for {format=}")

            if isinstance(dtype, np.dtype):
                target_unit = np.datetime_data(dtype)[0]
            elif isinstance(dtype, pd.DatetimeTZDtype):
                target_unit = dtype.unit
            else:
                target_unit = dtype.pyarrow_dtype.unit
            if target_unit != "s" and len(without_nat):
                # libcudf parses directly into int64 values of the
                # target unit and silently wraps on overflow
                # (see https://github.com/rapidsai/cudf/issues/23247).
                # Parse to seconds first (which cannot realistically
                # overflow) and reject values whose whole-second part
                # falls outside the target unit's range, like pandas
                # does. This double parse can be removed once libcudf
                # detects the overflow itself.
                with without_nat.access(mode="read", scope="internal"):
                    seconds = ColumnBase.create(
                        plc.strings.convert.convert_datetime.to_timestamps(
                            without_nat.plc_column,
                            dtype_to_pylibcudf_type(np.dtype("datetime64[s]")),
                            format,
                        ),
                        np.dtype("datetime64[s]"),
                    )
                lo, hi = seconds.minmax()
                raise_if_datetime_seconds_out_of_bounds(lo, hi, target_unit)

            casting_func = plc.strings.convert.convert_datetime.to_timestamps
            add_back_nat = is_nat.any()
        elif dtype.kind == "m":
            casting_func = plc.strings.convert.convert_durations.to_durations
            is_nat = self == "NaT"
            add_back_nat = is_nat.any()

        with self.access(mode="read", scope="internal"):
            plc_dtype = dtype_to_pylibcudf_type(dtype)
            result_col = cast(
                "cudf.core.column.datetime.DatetimeColumn | cudf.core.column.timedelta.TimeDeltaColumn",
                ColumnBase.create(
                    casting_func(self.plc_column, plc_dtype, format), dtype
                ),
            )

        if add_back_nat:
            result_col[is_nat] = None

        return result_col

    def as_datetime_column(self, dtype: np.dtype) -> DatetimeColumn:
        not_null = self.apply_boolean_mask(self.notnull())
        if len(not_null) == 0:
            # We should hit the self.null_count == len(self) condition
            # so format doesn't matter
            format = ""
        else:
            # infer on host from the first not na element
            format = infer_format(not_null.element_indexing(0))
        return self.strptime(dtype, format)  # type: ignore[return-value]

    def as_timedelta_column(self, dtype: np.dtype) -> TimeDeltaColumn:
        format = _infer_timedelta_format(self)
        return self.strptime(dtype, format)  # type: ignore[return-value]

    def as_decimal_column(self, dtype: DecimalDtype) -> DecimalColumn:
        with self.access(mode="read", scope="internal"):
            plc_column = (
                plc.strings.convert.convert_fixed_point.to_fixed_point(
                    self.plc_column,
                    dtype_to_pylibcudf_type(dtype),
                )
            )
            return cast(
                "cudf.core.column.decimal.DecimalColumn",
                ColumnBase.create(plc_column, dtype),
            )

    def as_string_column(self, dtype: DtypeObj) -> Self:
        if isinstance(dtype, np.dtype) and dtype.kind == "U":
            dtype = np.dtype("object")
        if dtype != self.dtype:
            if (
                cudf.get_option("mode.pandas_compatible")
                and self.null_count != 0
                and isinstance(dtype, np.dtype)
                and dtype == np.dtype("O")
                and isinstance(self.dtype, (pd.StringDtype, pd.ArrowDtype))
                and self.dtype.na_value is pd.NA
            ):
                raise NotImplementedError(
                    "Casting nullable string columns with pd.NA to object "
                    "is not supported."
                )
            return cast("Self", ColumnBase.create(self.plc_column, dtype))
        return self

    def _process_values_for_isin(
        self, values: Sequence | ColumnBase
    ) -> tuple[ColumnBase, ColumnBase]:
        lhs, rhs = super()._process_values_for_isin(values)
        if lhs.dtype != rhs.dtype and is_dtype_obj_string(rhs.dtype):
            # Strings of any dtype flavor (object, pd.StringDtype with
            # python/pyarrow storage and NaN/NA na_value, pd.ArrowDtype
            # string) hold comparable values; align rhs with lhs's dtype
            # so ColumnBase.isin does not treat them as disjoint types.
            rhs = rhs.astype(lhs.dtype)
        return lhs, rhs

    @property
    def values(self) -> cp.ndarray:
        """
        Return a CuPy representation of the Column.
        """
        # dask checks for a TypeError instead of NotImplementedError
        raise TypeError(f"cupy does not support {self.dtype}")

    @property
    def _should_use_large_string(self) -> bool:
        """Whether this column's dtype maps to arrow large_string."""
        dtype = self.dtype
        if isinstance(dtype, pd.StringDtype):
            return dtype.storage == "pyarrow"
        if isinstance(dtype, pd.ArrowDtype):
            return pa.types.is_large_string(dtype.pyarrow_dtype)
        return False

    def to_arrow(self) -> pa.Array:
        result = super().to_arrow()
        # libcudf produces arrow string (32-bit offsets), but pandas 3.0
        # uses large_string (64-bit offsets) for pyarrow-backed string
        # dtypes. Cast to match the offset width declared by self.dtype so
        # that the physical arrow type is consistent with the cudf dtype
        # regardless of the offset width preserved from the source.
        if self._should_use_large_string:
            if pa.types.is_string(result.type):
                result = result.cast(pa.large_string())
        elif pa.types.is_large_string(result.type):
            result = result.cast(pa.string())
        return result

    def to_pandas(
        self,
        *,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Index:
        if arrow_type and nullable:
            raise ValueError(
                f"{arrow_type=} and {nullable=} cannot both be set."
            )
        if arrow_type:
            # When the dtype is an explicit ArrowDtype, honor the offset width
            # (string vs large_string) it declares via self.to_arrow().
            # Otherwise (e.g. a default StringDtype) preserve the source's raw
            # arrow offset width so round-trips do not silently widen.
            if isinstance(self.dtype, pd.ArrowDtype):
                pa_array = self.to_arrow()
            else:
                pa_array = ColumnBase.to_arrow(self)
            return pd.Index(
                pd.arrays.ArrowExtensionArray(pa_array), copy=False
            )
        if not nullable and isinstance(self.dtype, pd.StringDtype):
            if self.dtype.storage == "pyarrow":
                pa_array = self.to_arrow()
                pandas_array = self.dtype.__from_arrow__(pa_array)
            elif self.dtype.na_value is np.nan:
                pandas_array = pd.array(
                    self.to_arrow().to_pandas(), dtype=self.dtype
                )
            else:
                return super().to_pandas(
                    nullable=nullable, arrow_type=arrow_type
                )
            return pd.Index(pandas_array, copy=False)
        return super().to_pandas(nullable=nullable, arrow_type=arrow_type)

    def can_cast_safely(self, to_dtype: DtypeObj) -> bool:
        if self.dtype == to_dtype:
            return True
        elif to_dtype.kind in {"i", "u"} and self.is_all_integer():
            return True
        elif to_dtype.kind == "f" and self.is_all_float():
            return True
        else:
            return False

    def find_and_replace(
        self,
        to_replace: ColumnBase | list,
        replacement: ColumnBase | list,
        all_nan: bool = False,
    ) -> Self:
        """Return col with *to_replace* replaced with *value*"""
        to_replace_col, replacement_col = (
            ColumnBase._prepare_find_and_replace_columns(
                to_replace,
                replacement,
            )
        )

        if (
            to_replace_col.dtype != self.dtype
            and replacement_col.dtype != self.dtype
        ):
            return self.copy()

        result = self._find_and_replace_with_dedup(
            to_replace_col,
            replacement_col,
            self.dtype,
        )
        return cast("Self", result)

    def _binaryop(self, other: ColumnBinaryOperand, op: str) -> ColumnBase:
        reflect, op = self._check_reflected_op(op)
        # Due to https://github.com/pandas-dev/pandas/issues/46332 we need to
        # support binary operations between empty or all null string columns
        # and columns of other dtypes, even if those operations would otherwise
        # be invalid. For example, you cannot divide strings, but pandas allows
        # division between an empty string column and a (nonempty) integer
        # column. Ideally we would disable these operators entirely, but until
        # the above issue is resolved we cannot avoid this problem.
        if self.is_all_null:
            if op in {
                "__add__",
                "__sub__",
                "__mul__",
                "__mod__",
                "__pow__",
                "__truediv__",
                "__floordiv__",
            }:
                return self
            elif op in {"__eq__", "__lt__", "__le__", "__gt__", "__ge__"}:
                return self.notnull()
            elif op == "__ne__":
                return self.isnull()

        if is_scalar(other):
            if is_na_like(other):
                other = pa.scalar(None, type=cudf_dtype_to_pa_type(self.dtype))
            else:
                other = pa.scalar(other)  # type: ignore[arg-type]
        elif not isinstance(other, type(self)):
            return NotImplemented

        if isinstance(other, (StringColumn, pa.Scalar)):
            if isinstance(other, pa.Scalar) and not pa.types.is_string(
                other.type
            ):
                if op in {"__eq__", "__ne__"}:
                    return as_column(
                        op == "__ne__",
                        length=len(self),
                        dtype=get_dtype_of_same_kind(
                            self.dtype, np.dtype(np.bool_)
                        ),
                    ).set_mask(self.mask, self.null_count)
                else:
                    return NotImplemented

            if op == "__add__":
                if isinstance(other, pa.Scalar):
                    other = cast(
                        "cudf.core.column.string.StringColumn",
                        as_column(other, length=len(self)),
                    )
                lhs, rhs = (other, self) if reflect else (self, other)
                concatenated = lhs.concatenate([rhs], "", None)
                return ColumnBase.create(concatenated.plc_column, self.dtype)
            elif op in {
                "__eq__",
                "__ne__",
                "__gt__",
                "__lt__",
                "__ge__",
                "__le__",
                "NULL_EQUALS",
                "NULL_NOT_EQUALS",
            }:
                if isinstance(other, pa.Scalar):
                    other = pa_scalar_to_plc_scalar(other)
                lhs_op, rhs_op = (other, self) if reflect else (self, other)
                return binaryop.binaryop(
                    lhs=lhs_op,
                    rhs=rhs_op,
                    op=op,
                    dtype=get_dtype_of_same_kind(
                        self.dtype, np.dtype(np.bool_)
                    ),
                )
        return NotImplemented

    def minhash(
        self,
        seed: int | np.uint32,
        a: NumericalColumn,
        b: NumericalColumn,
        width: int,
    ) -> ListColumn:
        with self.access(mode="read", scope="internal"):
            # Convert int to np.uint32 with validation
            if isinstance(seed, int):
                if seed < 0 or seed > np.iinfo(np.uint32).max:
                    raise ValueError(
                        f"seed must be in range [0, {np.iinfo(np.uint32).max}]"
                    )
                seed = np.uint32(seed)
            result = plc.nvtext.minhash.minhash(
                self.plc_column,
                seed,
                a.plc_column,
                b.plc_column,
                width,
            )
            return cast(
                "cudf.core.column.lists.ListColumn",
                ColumnBase.create(
                    result,
                    cudf.ListDtype(
                        get_dtype_of_same_kind(self.dtype, np.dtype(np.uint32))
                    ),
                ),
            )

    def minhash64(
        self,
        seed: int | np.uint64,
        a: NumericalColumn,
        b: NumericalColumn,
        width: int,
    ) -> ListColumn:
        with self.access(mode="read", scope="internal"):
            # Convert int to np.uint64 with validation
            if isinstance(seed, int):
                if seed < 0 or seed > np.iinfo(np.uint64).max:
                    raise ValueError(
                        f"seed must be in range [0, {np.iinfo(np.uint64).max}]"
                    )
                seed = np.uint64(seed)
            result = plc.nvtext.minhash.minhash64(
                self.plc_column,
                seed,
                a.plc_column,
                b.plc_column,
                width,
            )
            return cast(
                "cudf.core.column.lists.ListColumn",
                ColumnBase.create(
                    result,
                    cudf.ListDtype(
                        get_dtype_of_same_kind(self.dtype, np.dtype(np.uint64))
                    ),
                ),
            )

    def jaccard_index(self, other: Self, width: int) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            result = plc.nvtext.jaccard.jaccard_index(
                self.plc_column,
                other.plc_column,
                width,
            )
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(
                    result,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.float32)),
                ),
            )

    def generate_ngrams(self, ngrams: int, separator: plc.Scalar) -> Self:
        with self.access(mode="read", scope="internal"):
            result = plc.nvtext.generate_ngrams.generate_ngrams(
                self.plc_column,
                ngrams,
                separator,
            )
            return cast(
                "Self",
                ColumnBase.create(result, self.dtype),
            )

    def generate_character_ngrams(self, ngrams: int) -> ListColumn:
        with self.access(mode="read", scope="internal"):
            result = plc.nvtext.generate_ngrams.generate_character_ngrams(
                self.plc_column, ngrams
            )
            return cast(
                "cudf.core.column.lists.ListColumn",
                ColumnBase.create(result, cudf.ListDtype(self.dtype)),
            )

    def hash_character_ngrams(
        self, ngrams: int, seed: int | np.uint32
    ) -> ListColumn:
        with self.access(mode="read", scope="internal"):
            # Convert int to np.uint32 with validation
            if isinstance(seed, int):
                if seed < 0 or seed > np.iinfo(np.uint32).max:
                    raise ValueError(
                        f"seed must be in range [0, {np.iinfo(np.uint32).max}]"
                    )
                seed = np.uint32(seed)
            result = plc.nvtext.generate_ngrams.hash_character_ngrams(
                self.plc_column, ngrams, seed
            )
            return cast(
                "cudf.core.column.lists.ListColumn",
                ColumnBase.create(
                    result,
                    cudf.ListDtype(
                        get_dtype_of_same_kind(self.dtype, np.dtype(np.uint32))
                    ),
                ),
            )

    def build_suffix_array(self, min_width: int) -> Self:
        with self.access(mode="read", scope="internal"):
            result = plc.nvtext.deduplicate.build_suffix_array(
                self.plc_column, min_width
            )
            return cast(
                "Self",
                ColumnBase.create(result, np.dtype(np.int32)),
            )

    def resolve_duplicates(self, sa: Self, min_width: int) -> Self:
        with self.access(mode="read", scope="internal"):
            result = plc.nvtext.deduplicate.resolve_duplicates(
                self.plc_column,
                sa.plc_column,
                min_width,
            )
            return cast(
                "Self",
                ColumnBase.create(result, self.dtype),
            )

    def resolve_duplicates_pair(
        self, sa1: Self, input2: Self, sa2: Self, min_width: int
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            result = plc.nvtext.deduplicate.resolve_duplicates_pair(
                self.plc_column,
                sa1.plc_column,
                input2.plc_column,
                sa2.plc_column,
                min_width,
            )
            return cast(
                "Self",
                ColumnBase.create(result, self.dtype),
            )

    def edit_distance(self, targets: Self) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            result = plc.nvtext.edit_distance.edit_distance(
                self.plc_column, targets.plc_column
            )
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(
                    result,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.int32)),
                ),
            )

    def byte_pair_encoding(
        self,
        merge_pairs: plc.nvtext.byte_pair_encode.BPEMergePairs,
        separator: str,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.byte_pair_encode.byte_pair_encoding(
                self.plc_column,
                merge_pairs,
                plc.Scalar.from_py(
                    separator, dtype=plc.DataType(plc.TypeId.STRING)
                ),
            )
            return cast(
                "Self",
                ColumnBase.create(plc_column, self.dtype),
            )

    def ngrams_tokenize(
        self,
        ngrams: int,
        delimiter: plc.Scalar,
        separator: plc.Scalar,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.ngrams_tokenize.ngrams_tokenize(
                self.plc_column,
                ngrams,
                delimiter,
                separator,
            )
            return cast(
                "Self",
                ColumnBase.create(plc_column, self.dtype),
            )

    def normalize_spaces(self) -> Self:
        return cast(
            "Self",
            PylibcudfFunction(
                plc.nvtext.normalize.normalize_spaces,
                same_dtype_policy,
            ).execute_with_args(self),
        )

    def normalize_characters(
        self, normalizer: plc.nvtext.normalize.CharacterNormalizer
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.normalize.normalize_characters(
                self.plc_column,
                normalizer,
            )
            return cast(
                "Self",
                ColumnBase.create(plc_column, self.dtype),
            )

    def replace_tokens(
        self, targets: Self, replacements: Self, delimiter: plc.Scalar
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.replace.replace_tokens(
                self.plc_column,
                targets.plc_column,
                replacements.plc_column,
                delimiter,
            )
            return cast(
                "Self",
                ColumnBase.create(plc_column, self.dtype),
            )

    def filter_tokens(
        self,
        min_token_length: int,
        replacement: plc.Scalar,
        delimiter: plc.Scalar,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.replace.filter_tokens(
                self.plc_column,
                min_token_length,
                replacement,
                delimiter,
            )
            return cast(
                "Self",
                ColumnBase.create(plc_column, self.dtype),
            )

    def porter_stemmer_measure(self) -> NumericalColumn:
        return cast(
            "cudf.core.column.numerical.NumericalColumn",
            PylibcudfFunction(
                plc.nvtext.stemmer.porter_stemmer_measure,
                pylibcudf_result_dtype_policy,
            ).execute_with_args(self),
        )

    def is_letter(
        self, is_vowel: bool, index: int | NumericalColumn
    ) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.stemmer.is_letter(
                self.plc_column,
                is_vowel,
                index if isinstance(index, int) else index.plc_column,
            )
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(
                    plc_column,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_)),
                ),
            )

    def tokenize_scalar(self, delimiter: plc.Scalar) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.tokenize.tokenize_scalar(
                self.plc_column, delimiter
            )
            return cast(
                "Self",
                ColumnBase.create(plc_column, self.dtype),
            )

    def tokenize_column(self, delimiters: Self) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.tokenize.tokenize_column(
                self.plc_column,
                delimiters.plc_column,
            )
            return cast(
                "Self",
                ColumnBase.create(plc_column, self.dtype),
            )

    def count_tokens_scalar(self, delimiter: plc.Scalar) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.tokenize.count_tokens_scalar(
                self.plc_column, delimiter
            )
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(
                    plc_column,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.int32)),
                ),
            )

    def count_tokens_column(self, delimiters: Self) -> NumericalColumn:
        return cast(
            "cudf.core.column.numerical.NumericalColumn",
            PylibcudfFunction(
                plc.nvtext.tokenize.count_tokens_column,
                pylibcudf_result_dtype_policy,
            ).execute_with_args(self, delimiters),
        )

    def character_tokenize(self) -> Self:
        return cast(
            "Self",
            PylibcudfFunction(
                plc.nvtext.tokenize.character_tokenize,
                list_dtype_policy,
            ).execute_with_args(self),
        )

    def tokenize_with_vocabulary(
        self,
        vocabulary: plc.nvtext.tokenize.TokenizeVocabulary,
        delimiter: str,
        default_id: int,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            return cast(
                "Self",
                ColumnBase.create(
                    plc.nvtext.tokenize.tokenize_with_vocabulary(
                        self.plc_column,
                        vocabulary,
                        plc.Scalar.from_py(
                            delimiter, dtype=plc.DataType(plc.TypeId.STRING)
                        ),
                        default_id,
                    ),
                    cudf.ListDtype(
                        get_dtype_of_same_kind(self.dtype, np.dtype(np.int32))
                    ),
                ),
            )

    def wordpiece_tokenize(
        self,
        vocabulary: plc.nvtext.wordpiece_tokenize.WordPieceVocabulary,
        max_words_per_row: int,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            return cast(
                "Self",
                ColumnBase.create(
                    plc.nvtext.wordpiece_tokenize.wordpiece_tokenize(
                        self.plc_column,
                        vocabulary,
                        max_words_per_row,
                    ),
                    cudf.ListDtype(
                        get_dtype_of_same_kind(self.dtype, np.dtype(np.int32))
                    ),
                ),
            )

    def detokenize(self, indices: ColumnBase, separator: plc.Scalar) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.tokenize.detokenize(
                self.plc_column,
                indices.plc_column,
                separator,
            )
            return cast(
                "Self",
                ColumnBase.create(plc_column, self.dtype),
            )

    def to_lower(self) -> Self:
        result = cast(
            "Self",
            PylibcudfFunction(
                plc.strings.case.to_lower,
                same_dtype_policy,
            ).execute_with_args(self),
        )

        # Handle Greek final sigma (ς) special case
        # Greek capital sigma (Σ) lowercases to regular sigma (σ) at libcudf level,  # noqa: RUF003
        # but should become final sigma (ς) when at the end of a word.
        # Replace σ with ς when followed by end-of-string or non-letter character.  # noqa: RUF003
        has_sigma = result.str_contains("σ")
        if has_sigma.any():
            result = result.replace_with_backrefs(
                r"σ($|[^a-zA-Zα-ωΑ-Ωά-ώΆ-Ώ])", r"ς\1"
            )

        return result

    def to_upper(self) -> Self:
        return cast(
            "Self",
            PylibcudfFunction(
                plc.strings.case.to_upper,
                same_dtype_policy,
            ).execute_with_args(self),
        )

    def capitalize(self) -> Self:
        return cast(
            "Self",
            PylibcudfFunction(
                plc.strings.capitalize.capitalize,
                same_dtype_policy,
            ).execute_with_args(self),
        )

    def swapcase(self) -> Self:
        return cast(
            "Self",
            PylibcudfFunction(
                plc.strings.case.swapcase,
                same_dtype_policy,
            ).execute_with_args(self),
        )

    def title(self) -> Self:
        return cast(
            "Self",
            PylibcudfFunction(
                plc.strings.capitalize.title,
                same_dtype_policy,
            ).execute_with_args(self),
        )

    def is_title(self) -> Self:
        return cast(
            "Self",
            PylibcudfFunction(
                plc.strings.capitalize.is_title,
                bool_same_kind_policy,
            ).execute_with_args(self),
        )

    def replace_multiple(self, pattern: Self, replacements: Self) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.replace.replace_multiple(
                self.plc_column,
                pattern.plc_column,
                replacements.plc_column,
            )
            return cast(
                "Self",
                ColumnBase.create(plc_result, self.dtype),
            )

    def is_hex(self) -> NumericalColumn:
        return cast(
            "cudf.core.column.numerical.NumericalColumn",
            PylibcudfFunction(
                plc.strings.convert.convert_integers.is_hex,
                pylibcudf_result_dtype_policy,
            ).execute_with_args(self),
        )

    def hex_to_integers(self) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.convert.convert_integers.hex_to_integers(
                self.plc_column, plc.DataType(plc.TypeId.INT64)
            )
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(
                    plc_column,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.int64)),
                ),
            )

    def is_ipv4(self) -> NumericalColumn:
        return cast(
            "cudf.core.column.numerical.NumericalColumn",
            PylibcudfFunction(
                plc.strings.convert.convert_ipv4.is_ipv4,
                pylibcudf_result_dtype_policy,
            ).execute_with_args(self),
        )

    def ipv4_to_integers(self) -> NumericalColumn:
        return cast(
            "cudf.core.column.numerical.NumericalColumn",
            PylibcudfFunction(
                plc.strings.convert.convert_ipv4.ipv4_to_integers,
                pylibcudf_result_dtype_policy,
            ).execute_with_args(self),
        )

    def is_timestamp(self, format: str) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.convert.convert_datetime.is_timestamp(
                self.plc_column, format
            )
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(
                    plc_column,
                    get_dtype_of_same_kind(self.dtype, np.dtype("bool")),
                ),
            )

    def _split_record_re(
        self,
        pattern: str,
        maxsplit: int,
        method: Callable[
            [plc.Column, plc.strings.regex_program.RegexProgram, int],
            plc.Column,
        ],
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = method(
                self.plc_column,
                plc.strings.regex_program.RegexProgram.create(
                    pattern,
                    plc.strings.regex_flags.RegexFlags.DEFAULT,
                ),
                maxsplit,
            )
            result_dtype = get_dtype_of_same_kind(
                self.dtype, cudf.ListDtype(self.dtype)
            )
            return cast(
                "Self",
                ColumnBase.create(plc_column, result_dtype),
            )

    def split_record_re(self, pattern: str, maxsplit: int) -> Self:
        return self._split_record_re(
            pattern, maxsplit, plc.strings.split.split.split_record_re
        )

    def rsplit_record_re(self, pattern: str, maxsplit: int) -> Self:
        return self._split_record_re(
            pattern, maxsplit, plc.strings.split.split.rsplit_record_re
        )

    def _split_re(
        self,
        pattern: str,
        maxsplit: int,
        method: Callable[
            [plc.Column, plc.strings.regex_program.RegexProgram, int],
            plc.Table,
        ],
    ) -> dict[int, Self]:
        with self.access(mode="read", scope="internal"):
            plc_table = method(
                self.plc_column,
                plc.strings.regex_program.RegexProgram.create(
                    pattern,
                    plc.strings.regex_flags.RegexFlags.DEFAULT,
                ),
                maxsplit,
            )
            return dict(
                enumerate(
                    cast(
                        "Self",
                        ColumnBase.create(col, self.dtype),
                    )
                    for col in plc_table.columns()
                )
            )

    def split_re(self, pattern: str, maxsplit: int) -> dict[int, Self]:
        return self._split_re(
            pattern, maxsplit, plc.strings.split.split.split_re
        )

    def rsplit_re(self, pattern: str, maxsplit: int) -> dict[int, Self]:
        return self._split_re(
            pattern, maxsplit, plc.strings.split.split.rsplit_re
        )

    def _split_record(
        self,
        delimiter: plc.Scalar,
        maxsplit: int,
        method: Callable[[plc.Column, plc.Scalar, int], plc.Column],
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = method(
                self.plc_column,
                delimiter,
                maxsplit,
            )
            result_dtype = get_dtype_of_same_kind(
                self.dtype, cudf.ListDtype(self.dtype)
            )
            return cast(
                "Self",
                ColumnBase.create(plc_column, result_dtype),
            )

    def split_record(self, delimiter: plc.Scalar, maxsplit: int) -> Self:
        return self._split_record(
            delimiter, maxsplit, plc.strings.split.split.split_record
        )

    def rsplit_record(self, delimiter: plc.Scalar, maxsplit: int) -> Self:
        return self._split_record(
            delimiter, maxsplit, plc.strings.split.split.rsplit_record
        )

    def _split(
        self,
        delimiter: plc.Scalar,
        maxsplit: int,
        method: Callable[[plc.Column, plc.Scalar, int], plc.Table],
    ) -> dict[int, Self]:
        with self.access(mode="read", scope="internal"):
            plc_table = method(
                self.plc_column,
                delimiter,
                maxsplit,
            )
            return dict(
                enumerate(
                    cast(
                        "Self",
                        ColumnBase.create(col, self.dtype),
                    )
                    for col in plc_table.columns()
                )
            )

    def split(self, delimiter: plc.Scalar, maxsplit: int) -> dict[int, Self]:
        return self._split(delimiter, maxsplit, plc.strings.split.split.split)

    def rsplit(self, delimiter: plc.Scalar, maxsplit: int) -> dict[int, Self]:
        return self._split(delimiter, maxsplit, plc.strings.split.split.rsplit)

    def split_part(self, delimiter: plc.Scalar, index: int) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.split.split.split_part(
                self.plc_column,
                delimiter,
                index,
            )
            return cast(
                "Self",
                ColumnBase.create(plc_column, self.dtype),
            )

    def _partition(
        self,
        delimiter: plc.Scalar,
        method: Callable[[plc.Column, plc.Scalar], plc.Table],
    ) -> dict[int, Self]:
        with self.access(mode="read", scope="internal"):
            plc_table = method(
                self.plc_column,
                delimiter,
            )
            return dict(
                enumerate(
                    cast(
                        "Self",
                        ColumnBase.create(col, self.dtype),
                    )
                    for col in plc_table.columns()
                )
            )

    def partition(self, delimiter: plc.Scalar) -> dict[int, Self]:
        return self._partition(
            delimiter, plc.strings.split.partition.partition
        )

    def rpartition(self, delimiter: plc.Scalar) -> dict[int, Self]:
        return self._partition(
            delimiter, plc.strings.split.partition.rpartition
        )

    def url_decode(self) -> Self:
        return cast(
            "Self",
            PylibcudfFunction(
                plc.strings.convert.convert_urls.url_decode,
                same_dtype_policy,
            ).execute_with_args(self),
        )

    def url_encode(self) -> Self:
        return cast(
            "Self",
            PylibcudfFunction(
                plc.strings.convert.convert_urls.url_encode,
                same_dtype_policy,
            ).execute_with_args(self),
        )

    def is_integer(self) -> NumericalColumn:
        return cast(
            "cudf.core.column.numerical.NumericalColumn",
            PylibcudfFunction(
                plc.strings.convert.convert_integers.is_integer,
                pylibcudf_result_dtype_policy,
            ).execute_with_args(self),
        )

    def is_float(self) -> NumericalColumn:
        return cast(
            "cudf.core.column.numerical.NumericalColumn",
            PylibcudfFunction(
                plc.strings.convert.convert_floats.is_float,
                pylibcudf_result_dtype_policy,
            ).execute_with_args(self),
        )

    def is_all_integer(self, int_type: DtypeObj | None = None) -> bool:
        """Check if all non-null strings in the column are integers.

        This is an optimized version of `is_integer().all()` that avoids
        creating an intermediate boolean column.

        Parameters
        ----------
        int_type : dtype, optional
            If provided, additionally checks that every string represents a
            value that fits within ``int_type`` (i.e. no underflow/overflow).
            By default no range checking is performed.

        Returns
        -------
        bool
            True if all non-null strings are valid integers, False otherwise.
        """
        plc_int_type = (
            dtype_to_pylibcudf_type(int_type) if int_type is not None else None
        )
        with self.access(mode="read", scope="internal"):
            bool_plc = plc.strings.convert.convert_integers.is_integer(
                self.plc_column, plc_int_type
            )
            return self._reduce_bool_column(bool_plc)

    def is_all_float(self) -> bool:
        """Check if all non-null strings in the column are floats.

        This is an optimized version of `is_float().all()` that avoids
        creating an intermediate boolean column.

        Returns
        -------
        bool
            True if all non-null strings are valid floats, False otherwise.
        """
        with self.access(mode="read", scope="internal"):
            bool_plc = plc.strings.convert.convert_floats.is_float(
                self.plc_column
            )
            return self._reduce_bool_column(bool_plc)

    @staticmethod
    def _reduce_bool_column(bool_plc: plc.Column) -> bool:
        """Reduce a boolean column to a single bool using all()."""
        result_scalar = plc.reduce.reduce(
            bool_plc,
            plc.aggregation.all(),
            plc.types.DataType(plc.types.TypeId.BOOL8),
        )
        result = result_scalar.to_py()
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"
        return result

    def count_characters(self) -> NumericalColumn:
        return cast(
            "cudf.core.column.numerical.NumericalColumn",
            PylibcudfFunction(
                plc.strings.attributes.count_characters,
                pandas_string_int32_policy,
            ).execute_with_args(self),
        )

    def count_bytes(self) -> NumericalColumn:
        return cast(
            "cudf.core.column.numerical.NumericalColumn",
            PylibcudfFunction(
                plc.strings.attributes.count_bytes,
                pylibcudf_result_dtype_policy,
            ).execute_with_args(self),
        )

    def join_strings(self, separator: str, na_rep: str | None) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.combine.join_strings(
                self.plc_column,
                plc.Scalar.from_py(
                    separator, dtype=plc.DataType(plc.TypeId.STRING)
                ),
                plc.Scalar.from_py(
                    na_rep, dtype=plc.DataType(plc.TypeId.STRING)
                ),
            )
            return cast(
                "Self",
                ColumnBase.create(plc_column, self.dtype),
            )

    def concatenate(
        self, others: Iterable[Self], sep: str, na_rep: str | None
    ) -> Self:
        return cast(
            "Self",
            PylibcudfFunction(
                plc.strings.combine.concatenate,
                same_dtype_policy,
            ).execute_with_args(
                ColumnList(*itertools.chain([self], others)),
                plc.Scalar.from_py(sep, dtype=plc.DataType(plc.TypeId.STRING)),
                plc.Scalar.from_py(
                    na_rep, dtype=plc.DataType(plc.TypeId.STRING)
                ),
            ),
        )

    def extract(self, pattern: str, flags: int) -> dict[int, Self]:
        with self.access(mode="read", scope="internal"):
            plc_table = plc.strings.extract.extract(
                self.plc_column,
                plc.strings.regex_program.RegexProgram.create(
                    pattern,
                    plc_flags_from_re_flags(flags),
                ),
            )
            return dict(
                enumerate(
                    cast("Self", ColumnBase.create(col, self.dtype))
                    for col in plc_table.columns()
                )
            )

    def contains_re(self, pattern: str, flags: int) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.contains.contains_re(
                self.plc_column,
                plc.strings.regex_program.RegexProgram.create(
                    pattern,
                    plc_flags_from_re_flags(flags),
                ),
            )
            dtype = get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_))
            return cast("Self", ColumnBase.create(plc_column, dtype))

    def str_contains(self, pattern: str | Self) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_pattern = (
                plc.Scalar.from_py(
                    pattern, dtype=plc.DataType(plc.TypeId.STRING)
                )
                if isinstance(pattern, str)
                else pattern.plc_column
            )
            plc_column = plc.strings.find.contains(
                self.plc_column,
                plc_pattern,
            )
            return cast(
                "Self",
                ColumnBase.create(
                    plc_column,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_)),
                ),
            )

    def like(self, pattern: str, escape: str) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.contains.like(
                self.plc_column,
                pattern,
                escape,
            )
            return cast(
                "Self",
                ColumnBase.create(
                    plc_column,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_)),
                ),
            )

    def repeat_strings(self, repeats: int | ColumnBase) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_repeats = (
                repeats.plc_column
                if isinstance(repeats, ColumnBase)
                else repeats
            )
            plc_column = plc.strings.repeat.repeat_strings(
                self.plc_column,
                plc_repeats,
            )
            return cast(
                "Self",
                ColumnBase.create(plc_column, self.dtype),
            )

    def replace_re(
        self,
        pattern: str,
        replacement: str,
        max_replace_count: int = -1,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.replace_re.replace_re(
                self.plc_column,
                plc.strings.regex_program.RegexProgram.create(
                    pattern,
                    plc.strings.regex_flags.RegexFlags.DEFAULT,
                ),
                plc.Scalar.from_py(
                    replacement, dtype=plc.DataType(plc.TypeId.STRING)
                ),
                max_replace_count,
            )
            return cast(
                "Self",
                ColumnBase.create(plc_column, self.dtype),
            )

    def replace_str(
        self, pattern: str, replacement: str, max_replace_count: int = -1
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.replace.replace(
                self.plc_column,
                plc.Scalar.from_py(
                    pattern, dtype=plc.DataType(plc.TypeId.STRING)
                ),
                plc.Scalar.from_py(
                    replacement, dtype=plc.DataType(plc.TypeId.STRING)
                ),
                max_replace_count,
            )
            return cast(
                "Self",
                ColumnBase.create(plc_result, self.dtype),
            )

    def replace_with_backrefs(self, pattern: str, replacement: str) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.replace_re.replace_with_backrefs(
                self.plc_column,
                plc.strings.regex_program.RegexProgram.create(
                    pattern, plc.strings.regex_flags.RegexFlags.DEFAULT
                ),
                replacement,
            )
            return cast(
                "Self",
                ColumnBase.create(plc_result, self.dtype),
            )

    def slice_strings(
        self,
        start: int | None | NumericalColumn,
        stop: int | None | NumericalColumn,
        step: int | None = None,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            if isinstance(start, ColumnBase) and isinstance(stop, ColumnBase):
                plc_start: plc.Column | plc.Scalar = start.plc_column
                plc_stop: plc.Column | plc.Scalar = stop.plc_column
                plc_step: plc.Scalar | None = None
            elif all(isinstance(x, int) or x is None for x in (start, stop)):
                plc_start = plc.Scalar.from_py(
                    start, dtype=plc.DataType(plc.TypeId.INT32)
                )
                plc_stop = plc.Scalar.from_py(
                    stop, dtype=plc.DataType(plc.TypeId.INT32)
                )
                plc_step = plc.Scalar.from_py(
                    step, dtype=plc.DataType(plc.TypeId.INT32)
                )
            else:
                raise ValueError("Invalid start and stop types")
            plc_result = plc.strings.slice.slice_strings(
                self.plc_column, plc_start, plc_stop, plc_step
            )
            return cast(
                "Self",
                ColumnBase.create(plc_result, self.dtype),
            )

    def all_characters_of_type(
        self,
        char_type: plc.strings.char_types.StringCharacterTypes,
        case_type: plc.strings.char_types.StringCharacterTypes = plc.strings.char_types.StringCharacterTypes.ALL_TYPES,
    ) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.char_types.all_characters_of_type(
                self.plc_column, char_type, case_type
            )
            dtype = get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_))
            result = cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(plc_result, dtype),
            )
            if (
                isinstance(self.dtype, pd.StringDtype)
                and self.dtype.na_value is np.nan
            ):
                result = result.fillna(False)
            return result

    def filter_characters_of_type(
        self,
        types_to_remove: plc.strings.char_types.StringCharacterTypes,
        replacement: plc.Scalar,
        types_to_keep: plc.strings.char_types.StringCharacterTypes,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.char_types.filter_characters_of_type(
                self.plc_column,
                types_to_remove,
                replacement,
                types_to_keep,
            )
            return cast(
                "Self",
                ColumnBase.create(plc_column, self.dtype),
            )

    def replace_slice(self, start: int, stop: int, repl: plc.Scalar) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.replace.replace_slice(
                self.plc_column,
                repl,
                start,
                stop,
            )
            return cast(
                "Self",
                ColumnBase.create(plc_result, self.dtype),
            )

    def get_json_object(
        self,
        json_path: str,
        allow_single_quotes: bool,
        strip_quotes_from_single_strings: bool,
        missing_fields_as_nulls: bool,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            options = plc.json.GetJsonObjectOptions(
                allow_single_quotes=allow_single_quotes,
                strip_quotes_from_single_strings=(
                    strip_quotes_from_single_strings
                ),
                missing_fields_as_nulls=missing_fields_as_nulls,
            )
            plc_result = plc.json.get_json_object(
                self.plc_column,
                plc.Scalar.from_py(
                    json_path, dtype=plc.DataType(plc.TypeId.STRING)
                ),
                options,
            )
            return cast(
                "Self",
                ColumnBase.create(plc_result, self.dtype),
            )

    def pad(
        self, width: int, side: plc.strings.side_type.SideType, fillchar: str
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.padding.pad(
                self.plc_column,
                width,
                side,
                fillchar,
            )
            return cast(
                "Self",
                ColumnBase.create(plc_result, self.dtype),
            )

    def zfill(self, width: int) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.padding.zfill(
                self.plc_column,
                width,
            )
            return cast(
                "Self",
                ColumnBase.create(plc_result, self.dtype),
            )

    def strip(
        self, side: plc.strings.side_type.SideType, to_strip: str | None = None
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.strip.strip(
                self.plc_column,
                side,
                plc.Scalar.from_py(
                    to_strip or "", dtype=plc.DataType(plc.TypeId.STRING)
                ),
            )
            return cast(
                "Self",
                ColumnBase.create(plc_result, self.dtype),
            )

    def wrap(self, width: int) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.wrap.wrap(
                self.plc_column,
                width,
            )
            return cast(
                "Self",
                ColumnBase.create(plc_result, self.dtype),
            )

    def count_re(self, pattern: str, flags: int) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.contains.count_re(
                self.plc_column,
                plc.strings.regex_program.RegexProgram.create(
                    pattern, plc_flags_from_re_flags(flags)
                ),
            )
            res = ColumnBase.create(
                plc_result,
                get_dtype_of_same_kind(self.dtype, np.dtype(np.int32)),
            )
            return cast("cudf.core.column.numerical.NumericalColumn", res)

    def findall(
        self,
        method: Callable[
            [plc.Column, plc.strings.regex_program.RegexProgram], plc.Column
        ],
        pat: str,
        flags: int = 0,
    ) -> Self:
        # Return type depends on method parameter at runtime:
        # - plc.strings.findall.findall -> LIST<STRING>
        # - plc.strings.findall.find_re -> INT32
        with self.access(mode="read", scope="internal"):
            if len(self) == 0:
                return cast(
                    "Self",
                    column_empty(0, dtype=self.dtype),
                )
            plc_result = method(
                self.plc_column,
                plc.strings.regex_program.RegexProgram.create(
                    pat, plc_flags_from_re_flags(flags)
                ),
            )
            res = ColumnBase.create(
                plc_result,
                get_dtype_of_same_kind(
                    self.dtype, dtype_from_pylibcudf_column(plc_result)
                ),
            )
            return cast("Self", res)

    def find_multiple(self, patterns: Self) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.find_multiple.find_multiple(
                self.plc_column,
                patterns.plc_column,
            )
            return cast(
                "Self",
                ColumnBase.create(
                    plc_result,
                    cudf.ListDtype(
                        get_dtype_of_same_kind(self.dtype, np.dtype(np.int32))
                    ),
                ),
            )

    def starts_ends_with(
        self,
        method: Callable[[plc.Column, plc.Column | plc.Scalar], plc.Column],
        pat: str | tuple[str, ...],
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            if isinstance(pat, str):
                plc_pat = plc.Scalar.from_py(
                    pat, dtype=plc.DataType(plc.TypeId.STRING)
                )
                plc_result = method(self.plc_column, plc_pat)
            elif isinstance(pat, tuple) and all(
                isinstance(p, str) for p in pat
            ):
                plc_self = self.plc_column
                plc_pat = plc.Scalar.from_py(
                    pat[0], dtype=plc.DataType(plc.TypeId.STRING)
                )
                plc_result = method(plc_self, plc_pat)
                for next_pat in pat[1:]:
                    plc_pat = plc.Scalar.from_py(
                        next_pat, dtype=plc.DataType(plc.TypeId.STRING)
                    )
                    plc_next_result = method(plc_self, plc_pat)
                    plc_result = plc.binaryop.binary_operation(
                        plc_result,
                        plc_next_result,
                        plc.binaryop.BinaryOperator.BITWISE_OR,
                        plc.DataType(plc.TypeId.BOOL8),
                    )
            else:
                raise TypeError(
                    f"expected a str or tuple[str, ...], not {type(pat).__name__}"
                )
            dtype = get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_))
            result = cast("Self", ColumnBase.create(plc_result, dtype))
        if self._PANDAS_NA_VALUE in {np.nan, None}:
            result = result.fillna(False)
        return result

    def find(
        self,
        method: Callable[[plc.Column, plc.Scalar, int, int], plc.Column],
        sub: str,
        start: int,
        end: int,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = method(
                self.plc_column,
                plc.Scalar.from_py(sub, dtype=plc.DataType(plc.TypeId.STRING)),
                start,
                end,
            )
            plc_result = plc.unary.cast(
                plc_result, plc.DataType(plc.TypeId.INT64)
            )
            return cast(
                "Self",
                ColumnBase.create(
                    plc_result,
                    get_dtype_of_same_kind(self.dtype, np.dtype("int64")),
                ),
            )

    def matches_re(self, pattern: str, flags: int) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.contains.matches_re(
                self.plc_column,
                plc.strings.regex_program.RegexProgram.create(
                    pattern, plc_flags_from_re_flags(flags)
                ),
            )
            dtype = get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_))
            return cast("Self", ColumnBase.create(plc_result, dtype))

    def code_points(self) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.attributes.code_points(
                self.plc_column,
            )
            return cast(
                "Self",
                ColumnBase.create(
                    plc_result,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.int32)),
                ),
            )

    def translate(self, table: dict) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.translate.translate(
                self.plc_column,
                str.maketrans(table),  # type: ignore[arg-type]
            )
            return cast(
                "Self",
                ColumnBase.create(plc_result, self.dtype),
            )

    def filter_characters(
        self,
        table: dict,
        keep: bool,
        repl: plc.Scalar,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.translate.filter_characters(
                self.plc_column,
                str.maketrans(table),  # type: ignore[arg-type]
                plc.strings.translate.FilterType.KEEP
                if keep
                else plc.strings.translate.FilterType.REMOVE,
                repl,
            )
            return cast(
                "Self",
                ColumnBase.create(plc_result, self.dtype),
            )
