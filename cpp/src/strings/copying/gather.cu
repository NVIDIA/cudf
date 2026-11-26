/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/detail/gather.cuh>
#include <cudf/strings/detail/gather.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/cstddef>
#include <cuda/std/utility>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace cudf::strings::detail {
namespace {

struct gather_string_column {
  size_type const* input_offsets;
  char const* input_chars;
  size_type* output_offsets;
  char* output_chars;
};
namespace gather_strings_plan {
constexpr int threads_per_block = 256;
constexpr int lanes_per_row     = 4;
static_assert(threads_per_block % lanes_per_row == 0);
constexpr std::size_t rows_per_copy_block = threads_per_block / lanes_per_row;
constexpr std::size_t max_copy_blocks     = 65'535;

/** All warp lanes participate, including groups beyond the last row. */
__device__ inline void copy_group(gather_string_column col,
                                  cuda::std::size_t source_row,
                                  size_type destination,
                                  size_type rows)
{
  auto const lane = threadIdx.x % lanes_per_row;
  size_type begin = 0, length = 0, target = 0;
  if (source_row < static_cast<cuda::std::size_t>(rows)) {
    begin  = col.input_offsets[source_row];
    length = col.input_offsets[source_row + 1] - begin;
    target = col.output_offsets[destination];
  }
  // Balance individual strings without a separate size/skew sampling pass.
  auto long_rows   = __ballot_sync(0xffff'ffffu, lane == 0 && length > 128);
  auto medium_rows = __ballot_sync(0xffff'ffffu, lane == 0 && length > 8 && length <= 128);
  if (length <= 8) {
    for (size_type byte = lane; byte < length; byte += lanes_per_row) {
      col.output_chars[int64_t{target} + byte] = col.input_chars[int64_t{begin} + byte];
    }
  }
  // Two 16-lane groups copy medium strings; every lane also participates
  // when the last queue entry occupies only one half warp.
  auto const warp_lane = threadIdx.x % 32;
  while (medium_rows) {
    auto const first = __ffs(medium_rows) - 1;
    medium_rows &= medium_rows - 1;
    bool const has_second = medium_rows != 0;
    auto const second     = has_second ? __ffs(medium_rows) - 1 : first;
    if (has_second) { medium_rows &= medium_rows - 1; }
    auto const owner  = warp_lane < 16 ? first : second;
    auto const source = __shfl_sync(0xffff'ffffu, begin, owner);
    auto const output = __shfl_sync(0xffff'ffffu, target, owner);
    auto count        = __shfl_sync(0xffff'ffffu, length, owner);
    if (warp_lane >= 16 && !has_second) { count = 0; }
    for (size_type byte = warp_lane % 16; byte < count; byte += 16) {
      col.output_chars[int64_t{output} + byte] = col.input_chars[int64_t{source} + byte];
    }
  }
  while (long_rows) {
    auto const leader = __ffs(long_rows) - 1;
    auto const source = __shfl_sync(0xffff'ffffu, begin, leader);
    auto const output = __shfl_sync(0xffff'ffffu, target, leader);
    auto const count  = __shfl_sync(0xffff'ffffu, length, leader);
    copy_string_warp(col.input_chars + source,
                     col.output_chars,
                     output,
                     int64_t{output} + count,
                     0,
                     threadIdx.x % 32);
    long_rows &= long_rows - 1;
  }
}

CUDF_KERNEL void copy_characters(gather_string_column const* columns,
                                 size_type const* gather_map,
                                 size_type rows)
{
  auto const col = columns[blockIdx.y];
  for (auto base = cuda::std::size_t{blockIdx.x} * rows_per_copy_block;
       base < static_cast<cuda::std::size_t>(rows);
       base += cuda::std::size_t{gridDim.x} * rows_per_copy_block) {
    auto const output_row = base + threadIdx.x / lanes_per_row;
    auto const in_bounds  = output_row < static_cast<cuda::std::size_t>(rows);
    auto const source     = in_bounds ? gather_map[output_row] : rows;
    auto const target     = in_bounds ? static_cast<size_type>(output_row) : 0;
    copy_group(col, source, target, rows);
  }
}

}  // namespace gather_strings_plan

CUDF_KERNEL void gather_string_sizes(gather_string_column const* columns,
                                     size_type num_columns,
                                     size_type num_rows,
                                     size_type* sizes)
{
  auto const column = cudf::detail::grid_1d::global_thread_id();
  if (column < num_columns) {
    sizes[column] = columns[column].input_offsets[num_rows] - columns[column].input_offsets[0];
  }
}

}  // namespace

std::unique_ptr<table> gather_non_null_permutation(table_view const& input,
                                                   device_span<size_type const> gather_map,
                                                   cuda::stream_ref stream,
                                                   cudf::memory_resources mr)
{
  auto const rows    = input.num_rows();
  auto const columns = input.num_columns();
  CUDF_EXPECTS(gather_map.size() == static_cast<std::size_t>(rows),
               "Permutation size must match the input row count");
  if (columns == 0) {
    return std::make_unique<table>(std::vector<std::unique_ptr<column>>{}, rows);
  }
  if (rows == 0) { return empty_like(input); }
  auto const temp_mr   = mr.get_temporary_mr();
  auto const output_mr = mr.get_output_mr();
  std::vector<gather_string_column> descriptors;
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
  cuda::device_buffer<gather_string_column> device_columns(stream, temp_mr, columns, cuda::no_init);
  cuda::device_buffer<size_type> device_sizes(stream, temp_mr, columns, cuda::no_init);
  auto char_sizes = cudf::detail::make_pinned_vector_async<size_type>(columns, stream);
  try {
    cudf::detail::cuda_memcpy_async<gather_string_column>(
      device_columns, host_span<gather_string_column const>{descriptors}, stream);
    gather_string_sizes<<<cudf::detail::grid_1d{columns, 256}.num_blocks, 256, 0, stream.get()>>>(
      device_columns.data(), columns, rows, device_sizes.data());
    CUDF_CUDA_TRY(cudaPeekAtLastError());
    cudf::detail::cuda_memcpy<size_type>(char_sizes, device_sizes, stream);
  } catch (...) {
    // CUDA 13 can defer reading descriptors until the stream reaches its upload. If a later
    // operation throws, wait before unwinding so descriptors and the pinned readback buffer
    // remain alive until all queued copies complete.
    cudf::detail::sync_stream(stream);
    throw;
  }

  // Let the offsets utility construct the scan iterator to avoid nested index arithmetic.
  auto const sizes = [&](size_type const* input_offsets) {
    return [map = gather_map.data(), input_offsets] __device__(size_type i) -> size_type {
      auto const row = map[i];
      return input_offsets[row + 1] - input_offsets[row];
    };
  };
  std::vector<rmm::device_buffer> chars;
  std::vector<gather_string_column> copy_columns;
  chars.reserve(columns);
  copy_columns.reserve(columns);
  for (size_type i = 0; i < columns; ++i) {
    auto const size_at = sizes(descriptors[i].input_offsets);
    offsets.push_back(
      cudf::strings::detail::make_offsets_child_column(size_at, rows, char_sizes[i], stream, mr));
    if (offsets.back()->type().id() == type_id::INT64) {
      auto const pairs = cuda::transform_iterator(
        cuda::counting_iterator<size_type>{0},
        [map = gather_map.data(), column = descriptors[i]] __device__(
          size_type row) -> cuda::std::pair<char const*, size_type> {
          auto const source = map[row];
          auto const offset = column.input_offsets[source];
          return {column.input_chars + offset, column.input_offsets[source + 1] - offset};
        });
      chars.push_back(cudf::strings::detail::make_chars_buffer(
                        offsets.back()->view(), char_sizes[i], pairs, rows, stream, mr)
                        .release());
    } else {
      chars.emplace_back(char_sizes[i], stream, output_mr);
      descriptors[i].output_offsets = offsets.back()->mutable_view().data<size_type>();
      descriptors[i].output_chars   = static_cast<char*>(chars.back().data());
      if (char_sizes[i] != 0) { copy_columns.push_back(descriptors[i]); }
    }
  }

  if (!copy_columns.empty()) {
    // Complete this upload before releasing or reusing its pageable host source on every path.
    cudf::detail::cuda_memcpy<gather_string_column>(
      device_span<gather_string_column>{device_columns.data(), copy_columns.size()},
      host_span<gather_string_column const>{copy_columns.data(), copy_columns.size()},
      stream);
    auto const blocks = std::min<std::size_t>(
      gather_strings_plan::max_copy_blocks,
      cudf::detail::div_rounding_up_safe(static_cast<std::size_t>(rows),
                                         gather_strings_plan::rows_per_copy_block));
    gather_strings_plan::copy_characters<<<dim3(blocks, copy_columns.size()),
                                           gather_strings_plan::threads_per_block,
                                           0,
                                           stream.get()>>>(
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
      cuda::device_buffer<bitmask_type>(stream, output_mr, 0, cuda::no_init)));
  }
  return std::make_unique<table>(std::move(output));
}

}  // namespace cudf::strings::detail
