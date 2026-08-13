/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/detail/concatenate.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

namespace cudf {
namespace strings {
namespace detail {
// Benchmark data, shared at https://github.com/rapidsai/cudf/pull/4703, shows
// that the single kernel optimization generally performs better, but when the
// number of chars/col is beyond a certain threshold memcpy performs better.
// This heuristic estimates which strategy will give better performance by
// comparing the mean chars/col with values from the above table.
constexpr bool use_fused_kernel_heuristic(bool const has_nulls,
                                          size_t const total_bytes,
                                          size_t const num_columns)
{
  return has_nulls ? total_bytes < num_columns * 1572864  // midpoint of 1048576 and 2097152
                   : total_bytes < num_columns * 393216;  // midpoint of 262144 and 524288
}

// Using a functor instead of a lambda as a workaround for:
// error: The enclosing parent function ("create_strings_device_views") for an
// extended __device__ lambda must not have deduced return type
struct chars_size_transform {
  __device__ size_t operator()(column_device_view const& col) const
  {
    if (col.size() > 0) {
      auto const offsets   = col.child(strings_column_view::offsets_column_index);
      auto const d_offsets = cudf::detail::input_offsetalator(offsets.head(), offsets.type());
      return d_offsets[col.size() + col.offset()] - d_offsets[col.offset()];
    } else {
      return 0;
    }
  }
};

auto create_strings_device_views(host_span<column_view const> views, rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  // Assemble contiguous array of device views
  auto [device_view_owners, device_views_ptr] =
    create_column_device_views<column_device_view>(views, stream);

  // Compute the partition offsets and size of offset column
  // Note: Using 64-bit size_t so we can detect overflow of 32-bit size_type
  auto input_offsets = cudf::detail::make_host_vector<size_t>(views.size() + 1, stream);
  auto offset_it     = std::next(input_offsets.begin());
  thrust::transform(
    thrust::host, views.begin(), views.end(), offset_it, [](auto const& col) -> size_t {
      return static_cast<size_t>(col.size());
    });
  thrust::inclusive_scan(thrust::host, offset_it, input_offsets.end(), offset_it);
  auto d_input_offsets = cudf::detail::make_device_uvector_async(
    input_offsets, stream, cudf::get_current_device_resource_ref());
  auto const output_size = input_offsets.back();

  // Compute the partition offsets and size of chars column
  // Note: Using 64-bit size_t so we can detect overflow of 32-bit size_type
  auto d_partition_offsets = rmm::device_uvector<size_t>(views.size() + 1, stream);
  d_partition_offsets.set_element_to_zero_async(0, stream);  // zero first element

  thrust::transform_inclusive_scan(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    device_views_ptr,
    device_views_ptr + views.size(),
    std::next(d_partition_offsets.begin()),
    chars_size_transform{},
    cuda::std::plus{});
  auto const output_chars_size = d_partition_offsets.back_element(stream);
  stream.synchronize();  // ensure copy of output_chars_size is complete before returning

  return std::make_tuple(std::move(device_view_owners),
                         device_views_ptr,
                         std::move(d_input_offsets),
                         std::move(d_partition_offsets),
                         output_size,
                         output_chars_size);
}

template <size_type block_size, bool Nullable>
CUDF_KERNEL void fused_concatenate_string_offset_kernel(
  column_device_view const* input_views,
  size_t const* input_offsets,
  size_t const* partition_offsets,
  size_type const num_input_views,
  size_type const output_size,
  cudf::detail::output_offsetalator output_data,
  bitmask_type* output_mask,
  size_type* out_valid_count)
{
  auto output_index          = cudf::detail::grid_1d::global_thread_id();
  size_type warp_valid_count = 0;

  unsigned active_mask;
  if (Nullable) { active_mask = __ballot_sync(0xFFFF'FFFFu, output_index < output_size); }
  while (output_index < output_size) {
    // Lookup input index by searching for output index in offsets
    auto const offset_it            = cuda::std::prev(thrust::upper_bound(
      thrust::seq, input_offsets, input_offsets + num_input_views, output_index));
    size_type const partition_index = offset_it - input_offsets;

    auto const offset_index  = output_index - *offset_it;
    auto const& input_view   = input_views[partition_index];
    auto const offsets_child = input_view.child(strings_column_view::offsets_column_index);
    auto const input_data =
      cudf::detail::input_offsetalator(offsets_child.head(), offsets_child.type());
    output_data[output_index] =
      input_data[offset_index + input_view.offset()]  // handle parent offset
      - input_data[input_view.offset()]               // subtract first offset if non-zero
      + partition_offsets[partition_index];           // add offset of source column

    if (Nullable) {
      bool const bit_is_set       = input_view.is_valid(offset_index);
      bitmask_type const new_word = __ballot_sync(active_mask, bit_is_set);

      // First thread writes bitmask word
      if (threadIdx.x % cudf::detail::warp_size == 0) {
        output_mask[word_index(output_index)] = new_word;
      }

      warp_valid_count += __popc(new_word);
    }

    output_index += cudf::detail::grid_1d::grid_stride();
    if (Nullable) { active_mask = __ballot_sync(active_mask, output_index < output_size); }
  }

  // Fill final offsets index with total size of char data
  if (output_index == output_size) {
    output_data[output_size] = partition_offsets[num_input_views];
  }

  if (Nullable) {
    using cudf::detail::single_lane_block_sum_reduce;
    auto block_valid_count = single_lane_block_sum_reduce<block_size, 0>(warp_valid_count);
    if (threadIdx.x == 0) { atomicAdd(out_valid_count, block_valid_count); }
  }
}

CUDF_KERNEL void fused_concatenate_string_chars_kernel(column_device_view const* input_views,
                                                       size_t const* partition_offsets,
                                                       size_type const num_input_views,
                                                       size_type const output_size,
                                                       char* output_data)
{
  auto output_index = cudf::detail::grid_1d::global_thread_id();

  while (output_index < output_size) {
    // Lookup input index by searching for output index in offsets
    auto const offset_it            = cuda::std::prev(thrust::upper_bound(
      thrust::seq, partition_offsets, partition_offsets + num_input_views, output_index));
    size_type const partition_index = offset_it - partition_offsets;

    auto const offset_index = output_index - *offset_it;
    auto const& input_view  = input_views[partition_index];

    auto const offsets_child = input_view.child(strings_column_view::offsets_column_index);
    auto const input_offsets_data =
      cudf::detail::input_offsetalator(offsets_child.head(), offsets_child.type());

    auto const* input_chars_data = input_view.head<char>();

    auto const first_char     = input_offsets_data[input_view.offset()];
    output_data[output_index] = input_chars_data[offset_index + first_char];

    output_index += cudf::detail::grid_1d::grid_stride();
  }
}

std::unique_ptr<column> concatenate(host_span<column_view const> columns,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  // Compute output sizes
  auto const device_views         = create_strings_device_views(columns, stream);
  auto const& d_views             = std::get<1>(device_views);
  auto const& d_input_offsets     = std::get<2>(device_views);
  auto const& d_partition_offsets = std::get<3>(device_views);
  auto const strings_count        = std::get<4>(device_views);
  auto const total_bytes          = std::get<5>(device_views);
  auto const offsets_count        = strings_count + 1;

  if (strings_count == 0) { return make_empty_column(type_id::STRING); }

  CUDF_EXPECTS(offsets_count <= static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
               "total number of strings exceeds the column size limit",
               std::overflow_error);

  bool const has_nulls =
    std::any_of(columns.begin(), columns.end(), [](auto const& col) { return col.has_nulls(); });

  // create output chars column
  rmm::device_uvector<char> output_chars(total_bytes, stream, mr);
  auto d_new_chars = output_chars.data();

  // create output offsets column
  auto offsets_column = create_offsets_child_column(total_bytes, offsets_count, stream, mr);
  auto itr_new_offsets =
    cudf::detail::offsetalator_factory::make_output_iterator(offsets_column->mutable_view());

  rmm::device_buffer null_mask{0, stream, mr};
  size_type null_count{};
  if (has_nulls) {
    null_mask =
      cudf::detail::create_null_mask(strings_count, mask_state::UNINITIALIZED, stream, mr);
  }

  {  // Copy offsets columns with single kernel launch
    cudf::detail::device_scalar<size_type> d_valid_count(
      0, stream, cudf::get_current_device_resource_ref());

    constexpr size_type block_size{256};
    cudf::detail::grid_1d config(offsets_count, block_size);
    auto const kernel = has_nulls ? fused_concatenate_string_offset_kernel<block_size, true>
                                  : fused_concatenate_string_offset_kernel<block_size, false>;
    kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
      d_views,
      d_input_offsets.data(),
      d_partition_offsets.data(),
      static_cast<size_type>(columns.size()),
      strings_count,
      itr_new_offsets,
      reinterpret_cast<bitmask_type*>(null_mask.data()),
      d_valid_count.data());
    CUDF_CUDA_TRY(cudaGetLastError());

    if (has_nulls) { null_count = strings_count - d_valid_count.value(stream); }
  }

  if (total_bytes > 0) {
    // Use a heuristic to guess when the fused kernel will be faster than memcpy
    if (use_fused_kernel_heuristic(has_nulls, total_bytes, columns.size())) {
      // Use single kernel launch to copy chars columns
      constexpr size_t block_size{256};
      // cudf::detail::grid_1d limited to size_type elements
      auto const num_blocks = util::div_rounding_up_safe(total_bytes, block_size);
      auto const kernel     = fused_concatenate_string_chars_kernel;
      kernel<<<num_blocks, block_size, 0, stream.value()>>>(d_views,
                                                            d_partition_offsets.data(),
                                                            static_cast<size_type>(columns.size()),
                                                            total_bytes,
                                                            d_new_chars);
      CUDF_CUDA_TRY(cudaGetLastError());
    } else {
      // Memcpy each input chars column (more efficient for very large strings)
      for (auto column = columns.begin(); column != columns.end(); ++column) {
        size_type column_size = column->size();
        if (column_size == 0)  // nothing to do
          continue;            // empty column may not have children
        size_type column_offset   = column->offset();
        column_view offsets_child = column->child(strings_column_view::offsets_column_index);

        auto const bytes_offset = get_offset_value(offsets_child, column_offset, stream);
        auto const bytes_end = get_offset_value(offsets_child, column_size + column_offset, stream);
        // copy the chars column data
        auto d_chars     = column->head<char>() + bytes_offset;
        auto const bytes = bytes_end - bytes_offset;

        CUDF_CUDA_TRY(cudf::detail::memcpy_async(d_new_chars, d_chars, bytes, stream));

        // get ready for the next column
        d_new_chars += bytes;
      }
    }
  }

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             output_chars.release(),
                             null_count,
                             std::move(null_mask));
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
