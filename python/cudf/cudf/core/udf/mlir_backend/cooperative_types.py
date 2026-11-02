# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import llvm
from numba_cuda_mlir.models import PrimitiveModel, register_model
from numba_cuda_mlir.numba_cuda.types import Type


class CooperativeArrayType(Type):
    """Block-cooperative array: {ptr meminfo, ptr data, i64 size, ptr index}.

    Represents an array that a CUDA block of threads processes together.
    Input views (from column data) have meminfo=null. In this cuDF build only
    the read-only reduction surface is exposed, so intermediates are not
    NRT-allocated here. ``index`` points to the group's int64 index slice
    (parallel to ``data``); it is null when no index is available and is only
    consumed by ``idxmin``/``idxmax``.

    ABI size: 8 (meminfo) + 8 (data) + 8 (size) + 8 (index) = 32 bytes.
    """

    _extensionty_size = 32
    np_dtype: np.dtype[np.object_] = np.dtype("object")

    def __init__(self, dtype):
        from numba_cuda_mlir.numba_cuda import types as nb_types

        if isinstance(dtype, str):
            dtype = getattr(nb_types, dtype)
        self.dtype = dtype
        super().__init__(name=f"cooperative_array({dtype})")

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return (
            isinstance(other, CooperativeArrayType)
            and self.dtype == other.dtype
        )

    @property
    def return_as(self):
        return self


@register_model(CooperativeArrayType)
class CooperativeArrayModel(PrimitiveModel):
    """Block-cooperative array: {ptr meminfo, ptr data, i64 size}.

    Field 0 (meminfo) is the NRT_MemInfo pointer (null for input views).
    Field 1 (data) points to the element buffer.
    Field 2 (size) is the number of elements.
    """

    _fields = ("meminfo", "data", "size", "index")

    def __init__(self, dmm, fe_type):
        be_type = llvm.StructType.get_literal(
            [
                llvm.PointerType.get(),  # meminfo
                llvm.PointerType.get(),  # data
                ir.IntegerType.get_signless(64),  # size
                llvm.PointerType.get(),  # index (int64 slice; null if absent)
            ]
        )
        super().__init__(dmm, fe_type, be_type)

    def has_nrt_meminfo(self):
        return True

    def get_nrt_meminfo(self, value):
        return llvm.extractvalue(llvm.PointerType.get(), value, [0])

    def get_field_position(self, name):
        return self._fields.index(name)
