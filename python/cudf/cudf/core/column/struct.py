# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Self, cast

import pandas as pd
import pyarrow as pa

import cudf
from cudf.core.column.column import ColumnBase
from cudf.core.dtype.conversions import fields_from_struct_dtype
from cudf.core.dtypes import StructDtype
from cudf.utils.scalar import pa_scalar_to_plc_scalar
from cudf.utils.utils import is_na_like

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pylibcudf as plc

    from cudf._typing import DtypeObj
    from cudf.core.column.string import StringColumn


def _maybe_na_to_none(value: Any) -> Any:
    """
    Convert NA-like values to None for pyarrow.
    """
    if is_na_like(value):
        return None
    else:
        return value


class StructColumn(ColumnBase):
    """
    Column that stores fields of values.

    Every column has n children, where n is
    the number of fields in the Struct Dtype.
    """

    def _plc_memory_usage(self, col: plc.Column) -> int:
        n = self._plc_memory_usage_buffers(col)
        if col.size() == 0:
            return n

        struct_view = col.struct_view()
        for child_index, child_dtype in enumerate(self.fields.values()):
            sliced_child = struct_view.get_sliced_child(child_index)
            child = ColumnBase.create(
                sliced_child, child_dtype, validate=False
            )
            n += child._plc_memory_usage(sliced_child)
        return n

    @functools.cached_property
    def fields(self) -> dict[str, DtypeObj]:
        return fields_from_struct_dtype(self.dtype)

    def _get_sliced_child(self, loc: int | str) -> tuple[ColumnBase, str]:
        """
        Get a child column properly sliced to match the parent's view.

        Parameters
        ----------
        loc : int | str
            If int, the positional index of the child column to get.
            If str, the label of the child column to get.

        Returns
        -------
        tuple[ColumnBase, str]
            The child column at the specified location with the associated field name.
        """
        if isinstance(loc, int):
            if loc < 0 or loc >= self.plc_column.num_children():
                raise IndexError(
                    f"Index {loc} out of range for {self.plc_column.num_children()} children"
                )
            int_loc = loc
            field_label = list(self.fields)[loc]
        elif isinstance(loc, str):
            if loc not in self.fields:
                raise KeyError(
                    f"Field {loc} not found in {self.fields.keys()}"
                )
            int_loc = list(self.fields).index(loc)
            field_label = loc
        else:
            raise ValueError(
                f"loc must be an integer location or string label, not {loc}"
            )

        sliced_plc_col = self.plc_column.struct_view().get_sliced_child(
            int_loc
        )
        sub_type = self.fields[field_label]
        return ColumnBase.create(sliced_plc_col, sub_type), field_label

    def _prep_pandas_compat_repr(
        self, nan_rep: str | None = None
    ) -> StringColumn | Self:
        """
        Preprocess Column to be compatible with pandas repr, namely handling nulls.

        * null (datetime/timedelta) = str(pd.NaT)
        * null (other types)= str(pd.NA)
        """
        # TODO: handle if self.has_nulls(): case
        return self

    def to_pandas(
        self,
        *,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Index:
        if arrow_type or isinstance(self.dtype, pd.ArrowDtype):
            return super().to_pandas(nullable=nullable, arrow_type=arrow_type)
        elif nullable and not arrow_type:
            raise NotImplementedError(
                f"pandas does not have a native nullable type for {self.dtype}."
            )
        else:
            # We cannot go via Arrow's `to_pandas` because of the following issue:
            # https://github.com/apache/arrow/issues/28428
            return pd.Index(self.to_arrow().tolist(), dtype="object")

    def _cast_setitem_value(self, value: Any) -> plc.Scalar:
        pa_type = (
            self.dtype.to_arrow()
            if isinstance(self.dtype, StructDtype)
            else cast("pd.ArrowDtype", self.dtype).pyarrow_dtype
        )
        if isinstance(value, dict):
            new_value = {
                field: _maybe_na_to_none(value.get(field, None))
                for field in self.fields
            }
            return pa_scalar_to_plc_scalar(pa.scalar(new_value, type=pa_type))
        elif value is None or value is cudf.NA:
            return pa_scalar_to_plc_scalar(pa.scalar(None, type=pa_type))
        else:
            raise ValueError(
                f"Can not set {type(value).__name__} into StructColumn"
            )

    @property
    def __cuda_array_interface__(self) -> Mapping[str, Any]:
        raise NotImplementedError(
            "Structs are not yet supported via `__cuda_array_interface__`"
        )
