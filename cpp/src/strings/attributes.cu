/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/detail/attributes.hpp>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/warp/warp_reduce.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Threshold to decide on using string or warp parallel functions.
 *
 * If the average byte length of a string in a column exceeds this value then
 * the warp-parallel function is used.
 * Otherwise, a regular string-parallel function is used.
 *
 * This value was found using the strings_lengths benchmark results.
 */
constexpr size_type AVG_CHAR_BYTES_THRESHOLD = 64;

/**
 * @brief Returns a numeric column containing lengths of each string in
 * based on the provided unary function
 *
 * Any null string will result in a null entry for that row in the output column.
 *
 * @tparam UnaryFunction Device function that returns an integer given a string_view
 * @param strings Strings instance for this operation
 * @param ufn Function returns an integer for each string
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column with lengths for each string
 */
template <typename UnaryFunction>
std::unique_ptr<column> counts_fn(strings_column_view const& strings,
                                  UnaryFunction& ufn,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  // create output column
  auto results   = make_numeric_column(data_type{type_to_id<size_type>()},
                                     strings.size(),
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto d_lengths = results->mutable_view().data<int32_t>();
  // input column device view
  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  // fill in the lengths
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator<cudf::size_type>{strings.size()},
                    d_lengths,
                    cuda::proclaim_return_type<int32_t>([d_strings, ufn] __device__(size_type idx) {
                      return d_strings.is_null(idx)
                               ? 0
                               : static_cast<int32_t>(ufn(d_strings.element<string_view>(idx)));
                    }));
  results->set_null_count(strings.null_count());  // reset null count
  return results;
}

/**
 * @brief Count characters using a warp per string
 *
 * @param d_strings Column with strings to count
 * @param d_lengths Results of the counts per string
 */
CUDF_KERNEL void count_characters_parallel_fn(column_device_view const d_strings,
                                              size_type* d_lengths)
{
  auto const idx    = cudf::detail::grid_1d::global_thread_id();
  using warp_reduce = cub::WarpReduce<size_type>;
  __shared__ typename warp_reduce::TempStorage temp_storage;

  if (idx >= (d_strings.size() * cudf::detail::warp_size)) { return; }

  auto const str_idx  = static_cast<size_type>(idx / cudf::detail::warp_size);
  auto const lane_idx = static_cast<size_type>(idx % cudf::detail::warp_size);
  if (d_strings.is_null(str_idx)) {
    d_lengths[str_idx] = 0;
    return;
  }
  auto const d_str   = d_strings.element<string_view>(str_idx);
  auto const str_ptr = d_str.data();

  size_type count = 0;
  for (auto i = lane_idx; i < d_str.size_bytes(); i += cudf::detail::warp_size) {
    count += static_cast<size_type>(is_begin_utf8_char(str_ptr[i]));
  }
  auto const char_count = warp_reduce(temp_storage).Sum(count);
  if (lane_idx == 0) { d_lengths[str_idx] = char_count; }
}

std::unique_ptr<column> count_characters_parallel(strings_column_view const& input,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  // create output column
  auto results = make_numeric_column(data_type{type_to_id<size_type>()},
                                     input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);

  auto const d_lengths = results->mutable_view().data<size_type>();
  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);

  // fill in the lengths
  constexpr thread_index_type block_size = 256;
  constexpr thread_index_type warp_size  = cudf::detail::warp_size;
  cudf::detail::grid_1d grid{input.size() * warp_size, block_size};
  count_characters_parallel_fn<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
    *d_strings, d_lengths);
  CUDF_CUDA_TRY(cudaGetLastError());

  // reset null count after call to mutable_view()
  results->set_null_count(input.null_count());

  return results;
}

}  // namespace

std::unique_ptr<column> count_characters(strings_column_view const& input,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  if ((input.size() == input.null_count()) ||
      ((input.chars_size(stream) / (input.size() - input.null_count())) <
       AVG_CHAR_BYTES_THRESHOLD)) {
    auto ufn = cuda::proclaim_return_type<size_type>(
      [] __device__(string_view const& d_str) { return d_str.length(); });
    return counts_fn(input, ufn, stream, mr);
  }

  return count_characters_parallel(input, stream, mr);
}

std::unique_ptr<column> count_bytes(strings_column_view const& input,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  auto ufn = cuda::proclaim_return_type<size_type>(
    [] __device__(string_view const& d_str) { return d_str.size_bytes(); });
  return counts_fn(input, ufn, stream, mr);
}

}  // namespace detail

namespace {
/**
 * @brief Sets the code-point values for each character in the output
 * integer memory for each string in the strings column.
 *
 * For each string, there is a sub-array in d_results with length equal
 * to the number of characters in that string. The function here will
 * write code-point values to that section as pointed to by the
 * corresponding d_offsets value calculated for that string.
 */
struct code_points_fn {
  column_device_view d_strings;
  size_type* d_offsets;  // offset within d_results to fill with each string's code-point values
  int32_t* d_results;    // base integer array output

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) return;
    auto d_str  = d_strings.element<string_view>(idx);
    auto result = d_results + d_offsets[idx];
    thrust::copy(thrust::seq, d_str.begin(), d_str.end(), result);
  }
};

}  // namespace

namespace detail {
//
std::unique_ptr<column> code_points(strings_column_view const& input,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  auto strings_column = column_device_view::create(input.parent(), stream);
  auto d_column       = *strings_column;

  // create offsets vector to account for each string's character length
  rmm::device_uvector<size_type> offsets(input.size() + 1, stream);
  thrust::transform_inclusive_scan(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{input.size()},
    offsets.begin() + 1,
    cuda::proclaim_return_type<size_type>([d_column] __device__(size_type idx) {
      size_type length = 0;
      if (!d_column.is_null(idx)) length = d_column.element<string_view>(idx).length();
      return length;
    }),
    cuda::std::plus<size_type>());

  offsets.set_element_to_zero_async(0, stream);

  // the total size is the number of characters in the entire column
  size_type num_characters = offsets.back_element(stream);
  // create output column with no nulls
  auto results = make_numeric_column(
    data_type{type_id::INT32}, num_characters, mask_state::UNALLOCATED, stream, mr);
  auto results_view = results->mutable_view();
  // fill column with character code-point values
  auto d_results = results_view.data<int32_t>();
  // now set the ranges from each strings' character values
  thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<size_type>{0},
                     input.size(),
                     code_points_fn{d_column, offsets.data(), d_results});

  results->set_null_count(0);
  return results;
}

}  // namespace detail

// external APIS

std::unique_ptr<column> count_characters(strings_column_view const& input,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::count_characters(input, stream, mr);
}

std::unique_ptr<column> count_bytes(strings_column_view const& input,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::count_bytes(input, stream, mr);
}

std::unique_ptr<column> code_points(strings_column_view const& input,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::code_points(input, stream, mr);
}

}  // namespace strings
}  // namespace cudf
