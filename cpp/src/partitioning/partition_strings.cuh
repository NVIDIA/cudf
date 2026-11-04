/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "partition_string_copy.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/utilities.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cub/device/device_scan.cuh>
#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/cstddef>
#include <cuda/stream>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace cudf::detail {

CUDF_KERNEL void invert_partition_map(size_type const* gather_map,
                                      size_type* scatter_map,
                                      size_type num_rows)
{
  auto const row = static_cast<cuda::std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < num_rows) { scatter_map[gather_map[row]] = static_cast<size_type>(row); }
}

CUDF_KERNEL void partition_string_sizes(partition_string_column const* columns,
                                        size_type num_columns,
                                        size_type num_rows,
                                        size_type* sizes)
{
  auto const column = static_cast<cuda::std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column < num_columns) {
    sizes[column] = columns[column].input_offsets[num_rows] - columns[column].input_offsets[0];
  }
}

/**
 * @brief Materialize a complete permutation of non-nullable INT32-offset string columns.
 *
 * Batch the column-size readback and copy all character buffers in one kernel. The gather map
 * must be a permutation, as guaranteed by partition's rank/scan construction. Repeated or missing
 * indices are not supported. Other schemas return nullptr for the caller's generic gather path.
 */
[[nodiscard]] inline std::unique_ptr<table> gather_partition_strings(
  table_view const& input,
  device_span<size_type const> gather_map,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  if (input.num_columns() < 2 || input.num_columns() > 65'535 ||
      !std::all_of(input.begin(), input.end(), [](auto const& column) {
        return column.type().id() == type_id::STRING && column.null_count() == 0 &&
               strings_column_view(column).offsets().type().id() == type_id::INT32;
      })) {
    return nullptr;
  }

  auto const rows    = input.num_rows();
  auto const columns = input.num_columns();
  auto const temp_mr = cudf::get_current_device_resource_ref();
  std::vector<partition_string_column> descriptors;
  std::vector<std::unique_ptr<column>> offsets;
  descriptors.reserve(columns);
  offsets.reserve(columns);

  // A complete permutation preserves each sliced column's total character count.
  // Read the input boundaries in one batch before allocating or scanning the output columns.
  for (auto const& column : input) {
    auto const strings = strings_column_view(column);
    descriptors.push_back({strings.offsets().data<size_type>() + column.offset(),
                           strings.chars_begin(stream),
                           nullptr,
                           nullptr});
  }
  auto device_columns = cudf::detail::make_device_buffer(descriptors, stream, temp_mr);
  cuda::device_buffer<size_type> device_sizes(stream, temp_mr, columns, cuda::no_init);
  partition_string_sizes<<<(columns + 255) / 256, 256, 0, stream.get()>>>(
    device_columns.data(), columns, rows, device_sizes.data());
  auto const char_sizes         = cudf::detail::make_pinned_vector(device_sizes, stream);
  auto const max_chars          = *std::max_element(char_sizes.begin(), char_sizes.end());
  auto const offset64_threshold = cudf::strings::get_offset64_threshold();
  if (max_chars >= offset64_threshold) { return nullptr; }

  auto const sizes = [&](size_type const* input_offsets) {
    return cuda::transform_iterator(
      cuda::counting_iterator<size_type>{0},
      [map = gather_map.data(), input_offsets, rows] __device__(size_type i) -> size_type {
        if (i == rows) { return size_type{0}; }
        auto const row = map[i];
        return input_offsets[row + 1] - input_offsets[row];
      });
  };
  std::size_t temp_bytes = 0;
  if (max_chars > 0) {
    CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr,
                                                temp_bytes,
                                                sizes(nullptr),
                                                static_cast<size_type*>(nullptr),
                                                rows + 1,
                                                stream.get()));
  }
  cuda::device_buffer<std::byte> scan_storage(stream, temp_mr, temp_bytes, cuda::no_init);
  std::vector<rmm::device_buffer> chars;
  std::vector<partition_string_column> copy_columns;
  chars.reserve(columns);
  copy_columns.reserve(columns);
  for (size_type i = 0; i < columns; ++i) {
    offsets.push_back(cudf::make_numeric_column(
      data_type{type_id::INT32}, rows + 1, mask_state::UNALLOCATED, stream, mr));
    chars.emplace_back(char_sizes[i], stream, mr);
    descriptors[i].output_offsets = offsets.back()->mutable_view().data<size_type>();
    descriptors[i].output_chars   = static_cast<char*>(chars.back().data());
    if (char_sizes[i] == 0) {
      CUDF_CUDA_TRY(cudaMemsetAsync(descriptors[i].output_offsets,
                                    0,
                                    (static_cast<std::size_t>(rows) + 1) * sizeof(size_type),
                                    stream.get()));
    } else {
      CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(scan_storage.data(),
                                                  temp_bytes,
                                                  sizes(descriptors[i].input_offsets),
                                                  descriptors[i].output_offsets,
                                                  rows + 1,
                                                  stream.get()));
      copy_columns.push_back(descriptors[i]);
    }
  }

  if (!copy_columns.empty()) {
    // Complete this upload before releasing or reusing its pageable host source on every path.
    device_columns = cudf::detail::make_device_buffer(copy_columns, stream, temp_mr);
    cuda::device_buffer<size_type> scatter_map(stream, temp_mr, rows, cuda::no_init);
    invert_partition_map<<<(static_cast<std::size_t>(rows) + 255) / 256, 256, 0, stream.get()>>>(
      gather_map.data(), scatter_map.data(), rows);
    auto const blocks = std::min<std::size_t>(65'535, (static_cast<std::size_t>(rows) + 63) / 64);
    partition_strings_plan::copy_characters<true>
      <<<dim3(blocks, copy_columns.size()), 256, 0, stream.get()>>>(
        device_columns.data(), scatter_map.data(), rows);
    CUDF_CUDA_TRY(cudaPeekAtLastError());
  }

  std::vector<std::unique_ptr<column>> output;
  output.reserve(columns);
  for (size_type i = 0; i < columns; ++i) {
    output.push_back(
      cudf::make_strings_column(rows,
                                std::move(offsets[i]),
                                std::move(chars[i]),
                                0,
                                cudf::create_null_mask(0, mask_state::UNALLOCATED, stream, mr)));
  }
  return std::make_unique<table>(std::move(output));
}

}  // namespace cudf::detail
