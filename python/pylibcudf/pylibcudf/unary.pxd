# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from pylibcudf.libcudf.unary cimport unary_operator
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .types cimport DataType


cpdef Column unary_operation(
    Column input, unary_operator op, object stream = *, DeviceMemoryResource mr = *
)

cpdef Column is_null(Column input, object stream = *, DeviceMemoryResource mr = *)

cpdef Column is_valid(Column input, object stream = *, DeviceMemoryResource mr = *)

cpdef Column cast(
    Column input, DataType data_type, object stream = *, DeviceMemoryResource mr = *
)

cpdef Column is_nan(Column input, object stream = *, DeviceMemoryResource mr = *)

cpdef Column is_not_nan(Column input, object stream = *, DeviceMemoryResource mr = *)

cpdef bool is_supported_cast(DataType from_, DataType to)
