/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

#include <cub/device/device_memcpy.cuh>
#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <cuda/std/cstddef>
#include <cuda/std/execution>
#include <cuda/std/iterator>
#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <vector>

namespace cudf {
namespace strings {
namespace detail {

// Helper function for loading 16B from a potentially unaligned memory location to registers.
__forceinline__ __device__ uint4 load_uint4(char const* ptr)
{
  auto const offset       = reinterpret_cast<std::uintptr_t>(ptr) % 4;
  auto const* aligned_ptr = reinterpret_cast<unsigned int const*>(ptr - offset);
  auto const shift        = offset * 8;

  uint4 regs = {aligned_ptr[0], aligned_ptr[1], aligned_ptr[2], aligned_ptr[3]};
  uint tail  = 0;
  if (shift) tail = aligned_ptr[4];

  regs.x = __funnelshift_r(regs.x, regs.y, shift);
  regs.y = __funnelshift_r(regs.y, regs.z, shift);
  regs.z = __funnelshift_r(regs.z, regs.w, shift);
  regs.w = __funnelshift_r(regs.w, tail, shift);

  return regs;
}

/**
 * @brief Cooperatively copy one string with a full warp, using aligned 16-byte stores.
 *
 * The byte copies at either end keep unaligned loads within the input string. All warp lanes
 * participate, including when the string is too short for vectorized copies.
 *
 * @param in_start First character of the input string
 * @param out_chars Output character buffer
 * @param out_start First output byte for this string
 * @param out_end One past the last output byte for this string
 * @param alignment_offset Output buffer address modulo `sizeof(uint4)`
 * @param warp_lane Lane within the warp
 */
__forceinline__ __device__ void copy_string_warp(char const* in_start,
                                                 char* out_chars,
                                                 int64_t out_start,
                                                 int64_t out_end,
                                                 std::size_t alignment_offset,
                                                 std::size_t warp_lane)
{
  constexpr size_t out_datatype_size = sizeof(uint4);
  constexpr size_t in_datatype_size  = sizeof(uint);

  // Reserve four bytes at each end so load_uint4 cannot read beyond the input string.
  int64_t const out_start_aligned =
    (out_start + in_datatype_size + alignment_offset + out_datatype_size - 1) / out_datatype_size *
      out_datatype_size -
    alignment_offset;
  int64_t const out_end_aligned =
    (out_end - in_datatype_size + alignment_offset) / out_datatype_size * out_datatype_size -
    alignment_offset;

  for (int64_t ichar = out_start_aligned + warp_lane * out_datatype_size; ichar < out_end_aligned;
       ichar += cudf::detail::warp_size * out_datatype_size) {
    // The bounds already align out_chars + ichar; avoid converting back to a uint4 index.
    *cuda::std::bit_cast<uint4*>(out_chars + ichar) = load_uint4(in_start + (ichar - out_start));
  }

  if (out_end_aligned <= out_start_aligned) {
    for (auto ichar = out_start + warp_lane; ichar < out_end; ichar += cudf::detail::warp_size) {
      out_chars[ichar] = in_start[ichar - out_start];
    }
  } else {
    if (out_start + warp_lane < out_start_aligned) {
      out_chars[out_start + warp_lane] = in_start[warp_lane];
    }
    auto const ichar = out_end_aligned + warp_lane;
    if (ichar < out_end) { out_chars[ichar] = in_start[ichar - out_start]; }
  }
}

/**
 * @brief Gather characters from the input iterator, with string parallel strategy.
 *
 * This strategy assigns strings to warps so that each warp can cooperatively copy from the input
 * location of the string to the corresponding output location. Large datatype (uint4) is used for
 * stores. This strategy is best suited for large strings.
 *
 * @tparam StringIterator Iterator should produce `string_view` objects.
 * @tparam MapIterator Iterator for retrieving integer indices of the `StringIterator`.
 *
 * @param strings_begin Start of the iterator to retrieve `string_view` instances.
 * @param out_chars Output buffer for gathered characters.
 * @param out_offsets The offset values associated with the output buffer.
 * @param string_indices Start of index iterator.
 * @param total_out_strings Number of output strings to be gathered.
 */
template <typename StringIterator, typename MapIterator>
CUDF_KERNEL void gather_chars_fn_string_parallel(StringIterator strings_begin,
                                                 char* out_chars,
                                                 cudf::detail::input_offsetalator const out_offsets,
                                                 MapIterator string_indices,
                                                 size_type total_out_strings)
{
  auto const global_thread_id = cudf::detail::grid_1d::global_thread_id();
  auto const global_warp_id   = global_thread_id / cudf::detail::warp_size;
  auto const warp_lane        = global_thread_id % cudf::detail::warp_size;
  auto const nwarps           = cudf::detail::grid_1d::grid_stride() / cudf::detail::warp_size;

  auto const alignment_offset = cuda::std::bit_cast<std::uintptr_t>(out_chars) % sizeof(uint4);

  for (auto istring = global_warp_id; istring < total_out_strings; istring += nwarps) {
    auto const out_start = out_offsets[istring];
    auto const out_end   = out_offsets[istring + 1];

    // This check is necessary because string_indices[istring] may be out of bound.
    if (out_start == out_end) continue;

    copy_string_warp(strings_begin[string_indices[istring]].data(),
                     out_chars,
                     out_start,
                     out_end,
                     alignment_offset,
                     warp_lane);
  }
}

/**
 * @brief Gather characters from the input iterator, with char parallel strategy.
 *
 * This strategy assigns characters to threads, and uses binary search for getting the string
 * index. To improve the binary search performance, fixed number of strings per threadblock is
 * used. This strategy is best suited for small strings.
 *
 * @tparam StringIterator Iterator should produce `string_view` objects.
 * @tparam MapIterator Iterator for retrieving integer indices of the `StringIterator`.
 *
 * @param strings_begin Start of the iterator to retrieve `string_view` instances.
 * @param out_chars Output buffer for gathered characters.
 * @param out_offsets The offset values associated with the output buffer.
 * @param string_indices Start of index iterator.
 * @param total_out_strings Number of output strings to be gathered.
 */
template <int strings_per_threadblock, typename StringIterator, typename MapIterator>
CUDF_KERNEL void gather_chars_fn_char_parallel(StringIterator strings_begin,
                                               char* out_chars,
                                               cudf::detail::input_offsetalator const out_offsets,
                                               MapIterator string_indices,
                                               size_type total_out_strings)
{
  __shared__ int64_t out_offsets_threadblock[strings_per_threadblock + 1];

  // Current thread block will process output strings starting at `begin_out_string_idx`.
  size_type begin_out_string_idx = blockIdx.x * strings_per_threadblock;

  // Number of strings to be processed by the current threadblock.
  size_type strings_current_threadblock =
    min(strings_per_threadblock, total_out_strings - begin_out_string_idx);

  if (strings_current_threadblock <= 0) return;

  // Collectively load offsets of strings processed by the current thread block.
  for (size_type idx = threadIdx.x; idx <= strings_current_threadblock; idx += blockDim.x) {
    out_offsets_threadblock[idx] = out_offsets[idx + begin_out_string_idx];
  }
  __syncthreads();

  for (int64_t out_ibyte = threadIdx.x + out_offsets_threadblock[0];
       out_ibyte < out_offsets_threadblock[strings_current_threadblock];
       out_ibyte += blockDim.x) {
    // binary search for the string index corresponding to out_ibyte
    auto const string_idx_iter =
      cuda::std::prev(thrust::upper_bound(thrust::seq,
                                          out_offsets_threadblock,
                                          out_offsets_threadblock + strings_current_threadblock,
                                          out_ibyte));
    size_type string_idx = cuda::std::distance(out_offsets_threadblock, string_idx_iter);

    // calculate which character to load within the string
    auto const icharacter = out_ibyte - out_offsets_threadblock[string_idx];

    size_type in_string_idx = string_indices[begin_out_string_idx + string_idx];
    out_chars[out_ibyte]    = strings_begin[in_string_idx].data()[icharacter];
  }
}

/**
 * @brief Returns a new strings column using the specified indices to select
 * elements from the `strings` column.
 *
 * Caller must update the validity mask in the output column.
 *
 * ```
 * s1 = ["a", "b", "c", "d", "e", "f"]
 * map = [0, 2]
 * s2 = gather<true>( s1, map.begin(), map.end() )
 * s2 is ["a", "c"]
 * ```
 *
 * @tparam NullifyOutOfBounds If true, indices outside the column's range are nullified.
 * @tparam MapIterator Iterator for retrieving integer indices of the column.
 *
 * @param strings Strings instance for this operation.
 * @param begin Start of index iterator.
 * @param end End of index iterator.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Memory resources used for temporary allocations and the returned column.
 * @return New strings column containing the gathered strings.
 */
template <bool NullifyOutOfBounds, typename MapIterator>
std::unique_ptr<cudf::column> gather(strings_column_view const& strings,
                                     MapIterator begin,
                                     MapIterator end,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr)
{
  auto const output_mr = mr.get_output_mr();
  auto const temp_mr   = mr.get_temporary_mr();

  auto const output_count = std::distance(begin, end);
  if (output_count == 0) return make_empty_column(type_id::STRING);

  // build offsets column
  auto const d_strings    = column_device_view::create(strings.parent(), stream, temp_mr);
  auto const d_in_offsets = cudf::detail::offsetalator_factory::make_input_iterator(
    strings.is_empty() ? make_empty_column(type_id::INT32)->view() : strings.offsets(),
    strings.offset());

  auto sizes_itr = cuda::transform_iterator(
    begin,
    cuda::proclaim_return_type<size_type>(
      [d_strings = *d_strings, d_in_offsets] __device__(size_type idx) {
        if (NullifyOutOfBounds && (idx < 0 || idx >= d_strings.size())) { return 0; }
        if (not d_strings.is_valid(idx)) { return 0; }
        return static_cast<size_type>(d_in_offsets[idx + 1] - d_in_offsets[idx]);
      }));

  auto [out_offsets_column, out_char_bytes] = cudf::strings::detail::make_offsets_child_column(
    sizes_itr, sizes_itr + output_count, stream, mr);

  // build out offset view
  auto const offsets_view =
    cudf::detail::offsetalator_factory::make_input_iterator(out_offsets_column->view());
  cudf::prefetch::detail::prefetch(strings.chars_begin(stream), strings.chars_size(stream), stream);

  // build output char column
  auto out_chars_data = rmm::device_uvector<char>(out_char_bytes, stream, output_mr);
  cudf::prefetch::detail::prefetch(out_chars_data, stream);
  auto d_out_chars = out_chars_data.data();

  constexpr int warps_per_threadblock = 4;
  // String parallel strategy will be used if average string length is above this threshold.
  // Otherwise, char parallel strategy will be used.
  constexpr int64_t string_parallel_threshold = 32;

  int64_t const average_string_length = out_char_bytes / output_count;

  if (average_string_length > string_parallel_threshold) {
    constexpr int max_threadblocks = 65536;
    auto const grid_size =
      min(cudf::detail::div_rounding_up_safe(static_cast<int64_t>(output_count),
                                             static_cast<int64_t>(warps_per_threadblock)),
          static_cast<int64_t>(max_threadblocks));
    gather_chars_fn_string_parallel<<<grid_size,
                                      warps_per_threadblock * cudf::detail::warp_size,
                                      0,
                                      stream.get()>>>(
      d_strings->begin<string_view>(), d_out_chars, offsets_view, begin, output_count);
  } else {
    // Threshold is based on empirical data on H100.
    // If row count is above this threshold we use the cub::DeviceMemcpy::Batched API, otherwise we
    // use the custom cuDF kernel.
    constexpr int64_t cub_batch_copy_threshold = 1024 * 1024 * 0.5;

    if (output_count < cub_batch_copy_threshold) {
      constexpr int strings_per_threadblock = 32;
      auto const grid_size                  = cudf::detail::div_rounding_up_safe(
        static_cast<int64_t>(output_count), static_cast<int64_t>(strings_per_threadblock));
      gather_chars_fn_char_parallel<strings_per_threadblock>
        <<<grid_size, warps_per_threadblock * cudf::detail::warp_size, 0, stream.get()>>>(
          d_strings->begin<string_view>(), d_out_chars, offsets_view, begin, output_count);
    } else {
      // Iterator over the character column of input strings to gather
      auto in_chars_itr = cuda::transform_iterator(
        begin,
        cuda::proclaim_return_type<char const*>([d_strings = *d_strings] __device__(size_type idx) {
          if (NullifyOutOfBounds && (idx < 0 || idx >= d_strings.size())) {
            return static_cast<char const*>(nullptr);
          }
          if (not d_strings.is_valid(idx)) { return static_cast<char const*>(nullptr); }
          return d_strings.element<string_view>(idx).data();
        }));

      // Iterator over the output locations to write the output
      auto out_chars_itr = cudf::detail::make_counting_transform_iterator(
        0,
        cuda::proclaim_return_type<char*>(
          [d_strings = *d_strings, offsets_view, d_out_chars] __device__(size_type idx) {
            return d_out_chars + offsets_view[idx];
          }));

      auto env = cuda::std::execution::env{
        cuda::std::execution::prop{cuda::get_stream_t{}, stream},
        cuda::std::execution::prop{cuda::mr::get_memory_resource_t{}, temp_mr}};
      CUDF_CUDA_TRY(
        cub::DeviceMemcpy::Batched(in_chars_itr, out_chars_itr, sizes_itr, output_count, env));
    }
  }

  return make_strings_column(output_count,
                             std::move(out_offsets_column),
                             out_chars_data.release(),
                             0,  // caller sets these
                             cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
}

/**
 * @brief Returns a new strings column using the specified indices to select
 * elements from the `strings` column.
 *
 * Caller must update the validity mask in the output column.
 *
 * ```
 * s1 = ["a", "b", "c", "d", "e", "f"]
 * map = [0, 2]
 * s2 = gather( s1, map.begin(), map.end(), true )
 * s2 is ["a", "c"]
 * ```
 *
 * @tparam MapIterator Iterator for retrieving integer indices of the column.
 *
 * @param strings Strings instance for this operation.
 * @param begin Start of index iterator.
 * @param end End of index iterator.
 * @param nullify_out_of_bounds If true, indices outside the column's range are nullified.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Memory resources used for temporary allocations and the returned column.
 * @return New strings column containing the gathered strings.
 */
template <typename MapIterator>
std::unique_ptr<cudf::column> gather(strings_column_view const& strings,
                                     MapIterator begin,
                                     MapIterator end,
                                     bool nullify_out_of_bounds,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr)
{
  if (nullify_out_of_bounds) return gather<true>(strings, begin, end, stream, mr);
  return gather<false>(strings, begin, end, stream, mr);
}

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

template <typename MapIterator>
CUDF_KERNEL void copy_characters(gather_string_column const* columns,
                                 MapIterator gather_map,
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

template <typename Column>
CUDF_KERNEL void gather_string_sizes(Column const* columns,
                                     size_type num_columns,
                                     size_type num_rows,
                                     size_type* sizes)
{
  auto const column = cudf::detail::grid_1d::global_thread_id();
  if (column < num_columns) {
    sizes[column] = columns[column].input_offsets[num_rows] - columns[column].input_offsets[0];
  }
}

/**
 * @brief Gather a complete permutation of non-null STRING columns in one batch.
 *
 * The input must have no nulls and use INT32 offsets, with at most 65,535 columns.
 * Slices and all-valid masks are supported. The output has no null mask and follows
 * the large-strings policy, including promotion to INT64 offsets when needed.
 * The map must be a permutation of `[0, input.num_rows())`, with no repeated or omitted rows.
 * Map values are not checked.
 * This preserves each column's character count, allowing a single batched size readback.
 *
 * @note Batching can reduce synchronization and kernel-launch overhead, but this function is not
 * guaranteed to outperform other gather or materialization paths for every workload. Performance
 * depends on row and column counts, string lengths, and permutation locality. Callers should use
 * workload-specific heuristics to select this function or an alternative path; this utility does
 * not make that selection.
 *
 * @tparam MapIterator Random-access device iterator over row indices
 * @param input Table of non-null STRING columns with INT32 offsets
 * @param begin First index of the complete permutation
 * @param end One past the last index
 * @param stream CUDA stream used for device operations
 * @param mr Memory resources for temporary allocations and the returned table
 * @return Table containing the permuted STRING columns
 * @throw cudf::logic_error if the permutation size differs from the input row count
 */
template <typename MapIterator>
std::unique_ptr<table> gather_non_null_permutation(table_view const& input,
                                                   MapIterator begin,
                                                   MapIterator end,
                                                   cuda::stream_ref stream,
                                                   cudf::memory_resources mr)
{
  auto const rows    = input.num_rows();
  auto const columns = input.num_columns();
  CUDF_EXPECTS((end - begin) == rows, "Permutation size must match the input row count");
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
    return [map = begin, input_offsets] __device__(size_type i) -> size_type {
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
        [map = begin, column = descriptors[i]] __device__(
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
                                           stream.get()>>>(device_columns.data(), begin, rows);
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

}  // namespace detail
}  // namespace strings
}  // namespace cudf
