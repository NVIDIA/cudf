# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Lowerings for the groupby _group_slot helper.

All GroupType / Block* shim lowerings have been removed — cooperative
reductions are now handled by numba_cuda_mlir's MLIR cooperative module.
"""

from numba_cuda_mlir.extending import lowering_registry
from numba_cuda_mlir.numba_cuda import types

from cudf.core.udf.mlir_backend.groupby_typing import _group_slot

lower = lowering_registry.lower


def _lower_group_slot_impl(builder, target, group_type):
    """Allocate record_size bytes and store the pointer in target."""
    record_size = group_type.size
    i8 = builder.get_mlir_type(types.uint8)
    ptr = builder.alloca(i8, count=record_size)
    builder.store_var(target, ptr)


def _lower_group_slot(builder, target, args, kwargs):
    """Lowering for ``_group_slot()``.

    The group record type is the target's numba type. Registered once at
    import time: released numba-cuda-mlir snapshots the lowering registry into
    the target context at setup, so per-kernel late registration is never
    installed.
    """
    group_type = builder.get_numba_type(target.name)
    _lower_group_slot_impl(builder, target, group_type)


def register_group_slot_lowering(dataframe_group_type):
    """No-op retained for API compatibility; the _group_slot lowering is
    registered once at import (typing records the group type separately).
    """


lower(_group_slot)(_lower_group_slot)
