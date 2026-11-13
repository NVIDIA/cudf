# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings.cyruntime cimport cudaStream_t
from libcpp.memory cimport unique_ptr
from rmm.librmm.memory_resource cimport device_async_resource_ref
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table_view cimport table_view

cdef extern from "cudf/join/asof_join.hpp" namespace "cudf::experimental" nogil:
    cdef cppclass asof_join:
        asof_join(
            const table_view& right_by,
            const column_view& right_on,
            cudaStream_t stream,
            device_async_resource_ref mr,
        ) except +libcudf_exception_handler


cdef extern from * namespace "pylibcudf" nogil:
    """
    #include <cudf/join/asof_join.hpp>
    #include <cudf/column/column.hpp>
    #include <memory>

    namespace pylibcudf {
    inline std::unique_ptr<cudf::column> asof_join_indices(
      cudf::experimental::asof_join const& joiner,
      cudf::table_view const& left_by, cudf::column_view const& left_on,
      cudaStream_t stream, rmm::device_async_resource_ref mr)
    {
      auto indices = joiner.join(
        left_by, left_on, cudf::experimental::asof_join_strategy::BACKWARD,
        cudf::experimental::asof_join_exact_matches::YES, stream, mr);
      // cudf::column stores an rmm::device_buffer, so copy the returned CUDA buffer.
      cudf::column_view view{cudf::data_type{cudf::type_id::INT32},
                             left_on.size(), indices.data(), nullptr, 0};
      return std::make_unique<cudf::column>(view, stream, mr);
    }
    }  // namespace pylibcudf
    """
    cdef unique_ptr[column] asof_join_indices(
        const asof_join& joiner,
        const table_view& left_by,
        const column_view& left_on,
        cudaStream_t stream,
        device_async_resource_ref mr,
    ) except +libcudf_exception_handler
