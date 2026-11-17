# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lowering for the groupby ``_group_slot`` helper."""

from numba_cuda_mlir.extending import lowering_registry
from numba_cuda_mlir.numba_cuda import types

from cudf.core.udf.mlir_backend.groupby_typing import _group_slot

lower = lowering_registry.lower


def _lower_group_slot(builder, target, args, kwargs):
    """Lowering for ``_group_slot()``: allocate record_size bytes and store the
    pointer in target.

    The group record type is the target's numba type. Registered once at
    import time: released numba-cuda-mlir snapshots the lowering registry into
    the target context at setup, so per-kernel late registration is never
    installed.
    """
    group_type = builder.get_numba_type(target.name)
    i8 = builder.get_mlir_type(types.uint8)
    ptr = builder.alloca(i8, count=group_type.size)
    builder.store_var(target, ptr)


lower(_group_slot)(_lower_group_slot)
