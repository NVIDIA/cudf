# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
MLIR (numba_cuda_mlir) typing for GroupBy JIT apply: GroupByJITDataFrame
and _group_slot. Importing this module registers typing with numba_cuda_mlir.

GroupType / Block* shim declarations have been removed — all cooperative
reductions are now compiled as pure MLIR by numba_cuda_mlir's
cooperative_device_funcs module.
"""

from __future__ import annotations

from numba_cuda_mlir import models, types
from numba_cuda_mlir.extending import typing_registry
from numba_cuda_mlir.models import register_model
from numba_cuda_mlir.numba_cuda.core.errors import TypingError
from numba_cuda_mlir.numba_cuda.np import numpy_support
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
)
from numba_cuda_mlir.typing import signature as nb_signature

from cudf.core.udf.utils import Row, UDFError

SUPPORTED_GROUPBY_NUMBA_TYPES = [
    types.int32,
    types.int64,
    types.float32,
    types.float64,
]
SUPPORTED_GROUPBY_NUMPY_TYPES = [
    numpy_support.as_dtype(dt) for dt in SUPPORTED_GROUPBY_NUMBA_TYPES
]


class GroupByJITDataFrame(Row):
    """Row type for groupby apply UDF. Raises TypingError for missing keys so
    can_be_jitted can fall back to non-JIT (same as numba-cuda NumbaKeyError).
    """

    def typeof(self, key):
        if key not in self.fields:
            raise TypingError(
                f"Column {key!r} does not exist in group. "
                f"Valid columns: {list(self.fields.keys())}"
            )
        return super().typeof(key)

    def offset(self, key):
        if key not in self.fields:
            raise TypingError(
                f"Column {key!r} does not exist in group. "
                f"Valid columns: {list(self.fields.keys())}"
            )
        return super().offset(key)


register_model(GroupByJITDataFrame)(models.RecordModel)


@typing_registry.register_attr
class GroupByJITDataFrameAttr(AttributeTemplate):
    key = GroupByJITDataFrame

    def generic_resolve(self, typ, attr):
        if attr in typ.fields:
            return typ.typeof(attr)
        raise UDFError(
            f"JIT GroupBy.apply() does not support DataFrame.{attr}(). "
            "Only column names (e.g. group['col']) are valid."
        )


# --- _group_slot: device-only slot for group record (like _row_slot) ---
def _group_slot():
    """Device-only: return a slot for the group record. Do not call from Python."""
    raise NotImplementedError("_group_slot is only for JIT")


_group_slot_cases: list = []


class GroupSlotTemplate(AbstractTemplate):
    key = _group_slot

    def generic(self, args, kws):
        if len(args) != 0 or kws:
            return None
        for group_type in _group_slot_cases:
            return nb_signature(group_type)
        return None


def register_group_slot(dataframe_group_type):
    """Register typing/lowering for _group_slot() for this kernel's group type."""
    global _group_slot_cases
    _group_slot_cases = [dataframe_group_type]


def _register_group_slot_typing():
    typing_registry.register_global(
        _group_slot, types.Function(GroupSlotTemplate)
    )


_register_group_slot_typing()
