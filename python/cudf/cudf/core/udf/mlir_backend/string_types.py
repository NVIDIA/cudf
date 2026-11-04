# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import llvm
from numba_cuda_mlir.models import PrimitiveModel, register_model
from numba_cuda_mlir.numba_cuda.types import Type


class MLIRStringType(Type):
    """NRT-managed owned string: {ptr meminfo, ptr data, i64 nbytes}.

    All memory is allocated via NRT_Allocate (arena-backed when enabled).
    No C++ udf_string class -- just raw bytes managed by NRT.

    ABI size: 8 (meminfo) + 8 (data) + 8 (nbytes) = 24 bytes.
    """

    _extensionty_size = 24
    np_dtype: np.dtype[np.object_] = np.dtype("object")

    def __init__(self):
        super().__init__(name="mlir_string")

    @property
    def is_internal(self):
        # mlir_string is an extension type that must be passed to kernels as a
        # CPointer (not an Array/memref), so it is never "internal".
        return False

    @property
    def return_as(self):
        return self


@register_model(MLIRStringType)
class MLIRStringModel(PrimitiveModel):
    """NRT-managed owned string: {ptr meminfo, ptr data, i64 nbytes}.

    Field 0 (meminfo) is the NRT_MemInfo pointer for lifetime management.
    Field 1 (data) points to the raw UTF-8 byte buffer.
    Field 2 (nbytes) is the byte length of the string.
    """

    _fields = ("meminfo", "data", "nbytes")

    def __init__(self, dmm, fe_type):
        be_type = llvm.StructType.get_literal(
            [
                llvm.PointerType.get(),  # meminfo
                llvm.PointerType.get(),  # data (char*)
                ir.IntegerType.get_signless(64),  # nbytes
            ]
        )
        super().__init__(dmm, fe_type, be_type)

    def has_nrt_meminfo(self):
        return True

    def get_nrt_meminfo(self, value):
        return llvm.extractvalue(llvm.PointerType.get(), value, [0])

    def get_field_position(self, name):
        return self._fields.index(name)
