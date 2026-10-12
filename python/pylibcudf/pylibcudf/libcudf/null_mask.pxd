# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t
from libcpp.pair cimport pair
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport bitmask_type, mask_state, size_type
from pylibcudf.libcudf.utilities.device_buffer cimport byte, device_buffer

from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/null_mask.hpp" namespace "cudf" nogil:
    cdef device_buffer[byte] copy_bitmask (
        column_view view,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef device_buffer[byte] copy_bitmask (
        const bitmask_type* null_mask,
        size_type begin_bit,
        size_type end_bit,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef size_t bitmask_allocation_size_bytes (
        size_type number_of_bits,
        size_t padding_boundary
    ) except +libcudf_exception_handler

    cdef device_buffer[byte] create_null_mask (
        size_type size,
        mask_state state,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef pair[device_buffer[byte], size_type] bitmask_and(
        table_view view,
        cudaStream_t stream,
        device_async_resource_ref mr
    )

    cdef pair[device_buffer[byte], size_type] bitmask_or(
        table_view view,
        cudaStream_t stream,
        device_async_resource_ref mr
    )

    cdef size_type null_count(
        const bitmask_type * bitmask,
        size_type start,
        size_type stop,
        cudaStream_t stream
    )

    cdef size_type index_of_first_set_bit(
        const bitmask_type * bitmask,
        size_type start,
        size_type stop,
        cudaStream_t stream
    )


cdef extern from * namespace "pylibcudf" nogil:
    """
    #include <cudf/null_mask.hpp>
    #include <memory>

    namespace pylibcudf {
    inline auto copy_bitmask_to_unique_ptr(
      cudf::column_view view, cudaStream_t stream,
      rmm::device_async_resource_ref mr)
    {
      return std::make_unique<cuda::device_buffer<std::byte>>(
        cudf::copy_bitmask(view, stream, mr));
    }

    inline auto copy_bitmask_to_unique_ptr(
      cudf::bitmask_type const* mask, cudf::size_type begin_bit,
      cudf::size_type end_bit, cudaStream_t stream,
      rmm::device_async_resource_ref mr)
    {
      return std::make_unique<cuda::device_buffer<std::byte>>(
        cudf::copy_bitmask(mask, begin_bit, end_bit, stream, mr));
    }

    inline auto create_null_mask_unique_ptr(
      cudf::size_type size, cudf::mask_state state, cudaStream_t stream,
      rmm::device_async_resource_ref mr)
    {
      return std::make_unique<cuda::device_buffer<std::byte>>(
        cudf::create_null_mask(size, state, stream, mr));
    }

    inline auto bitmask_and_unique_ptr(
      cudf::table_view view, cudaStream_t stream,
      rmm::device_async_resource_ref mr)
    {
      auto [mask, count] = cudf::bitmask_and(view, stream, mr);
      return std::pair{
        std::make_unique<cuda::device_buffer<std::byte>>(std::move(mask)), count};
    }

    inline auto bitmask_or_unique_ptr(
      cudf::table_view view, cudaStream_t stream,
      rmm::device_async_resource_ref mr)
    {
      auto [mask, count] = cudf::bitmask_or(view, stream, mr);
      return std::pair{
        std::make_unique<cuda::device_buffer<std::byte>>(std::move(mask)), count};
    }
    }  // namespace pylibcudf
    """
    cdef unique_ptr[device_buffer[byte]] copy_bitmask_to_unique_ptr(
        column_view view, cudaStream_t stream, device_async_resource_ref mr
    ) except +libcudf_exception_handler
    cdef unique_ptr[device_buffer[byte]] copy_bitmask_to_unique_ptr(
        const bitmask_type* mask, size_type begin_bit, size_type end_bit,
        cudaStream_t stream, device_async_resource_ref mr
    ) except +libcudf_exception_handler
    cdef unique_ptr[device_buffer[byte]] create_null_mask_unique_ptr(
        size_type size, mask_state state, cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
    cdef pair[unique_ptr[device_buffer[byte]], size_type] bitmask_and_unique_ptr(
        table_view view, cudaStream_t stream, device_async_resource_ref mr
    ) except +libcudf_exception_handler
    cdef pair[unique_ptr[device_buffer[byte]], size_type] bitmask_or_unique_ptr(
        table_view view, cudaStream_t stream, device_async_resource_ref mr
    ) except +libcudf_exception_handler
