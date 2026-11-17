/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * SPDX-FileCopyrightText: Copyright (c) 2'026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "partition_string_copy.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/detail/utilities/execution_env.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/utilities.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/span.hpp>

#include <cub/device/device_scan.cuh>
#include <cuda/buffer>
#include <cuda/iterator>

#include <algorithm>

namespace cudf::detail {

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

// Materialize a complete permutation of non-nullable INT32-offset STRING columns.
// The caller checks the input schema and column-count limits before calling this helper.
[[nodiscard]] inline std::unique_ptr<table> gather_partition_strings(
  table_view const& input,
  device_span<size_type const> gather_map,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  auto const rows      = input.num_rows();
  auto const columns   = input.num_columns();
  auto const temp_mr   = mr.get_temporary_mr();
  auto const output_mr = mr.get_output_mr();
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
  std::vector<rmm::device_buffer> chars;
  std::vector<partition_string_column> copy_columns;
  chars.reserve(columns);
  copy_columns.reserve(columns);
  for (size_type i = 0; i < columns; ++i) {
    offsets.push_back(cudf::make_numeric_column(
      data_type{type_id::INT32}, rows + 1, mask_state::UNALLOCATED, stream, output_mr));
    chars.emplace_back(char_sizes[i], stream, output_mr);
    descriptors[i].output_offsets = offsets.back()->mutable_view().data<size_type>();
    descriptors[i].output_chars   = static_cast<char*>(chars.back().data());
    if (char_sizes[i] == 0) {
      CUDF_CUDA_TRY(cudaMemsetAsync(descriptors[i].output_offsets,
                                    0,
                                    (static_cast<std::size_t>(rows) + 1) * sizeof(size_type),
                                    stream.get()));
    } else {
      CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(sizes(descriptors[i].input_offsets),
                                                  descriptors[i].output_offsets,
                                                  rows + 1,
                                                  cudf::detail::make_env(stream, mr)));
      copy_columns.push_back(descriptors[i]);
    }
  }

  if (!copy_columns.empty()) {
    // Complete this upload before releasing or reusing its pageable host source on every path.
    device_columns    = cudf::detail::make_device_buffer(copy_columns, stream, temp_mr);
    auto const blocks = std::min<std::size_t>(65'535, (static_cast<std::size_t>(rows) + 63) / 64);
    partition_strings_plan::copy_characters<true>
      <<<dim3(blocks, copy_columns.size()), 256, 0, stream.get()>>>(
        device_columns.data(), gather_map.data(), rows);
    CUDF_CUDA_TRY(cudaPeekAtLastError());
  }

  std::vector<std::unique_ptr<column>> output;
  output.reserve(columns);
  for (size_type i = 0; i < columns; ++i) {
    output.push_back(cudf::make_strings_column(
      rows,
      std::move(offsets[i]),
      std::move(chars[i]),
      0,
      cudf::create_null_mask(0, mask_state::UNALLOCATED, stream, output_mr)));
  }
  return std::make_unique<table>(std::move(output));
}

}  // namespace cudf::detail
