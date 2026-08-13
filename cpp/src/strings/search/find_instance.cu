/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief String per warp function for find_instance
 */
CUDF_KERNEL void find_instance_warp_parallel_fn(column_device_view const d_strings,
                                                string_view const d_target,
                                                size_type const instance,
                                                size_type* d_results)
{
  auto const tid     = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx = tid / cudf::detail::warp_size;
  if (str_idx >= d_strings.size() or d_strings.is_null(str_idx)) { return; }

  namespace cg        = cooperative_groups;
  auto const warp     = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto const lane_idx = warp.thread_rank();

  auto const d_str = d_strings.element<string_view>(str_idx);
  auto const begin = d_str.data();
  auto const end   = begin + d_str.size_bytes();

  // each thread compares the target with the thread's individual starting byte for its string
  auto const max_pos   = d_str.size_bytes();
  size_type char_pos   = max_pos;
  size_type char_count = 0;
  size_type byte_count = 0;
  size_type offset     = 0;
  auto itr             = begin + lane_idx;
  while (byte_count + d_target.size_bytes() <= d_str.size_bytes()) {
    size_type const is_char =
      (itr + d_target.size_bytes() <= end) && !is_utf8_continuation_char(*itr);
    size_type const found = is_char && (d_target.compare(itr, d_target.size_bytes()) == 0);
    // count of threads that matched in this warp and produce an offset in each thread
    auto const found_count = cg::reduce(warp, found, cg::plus<size_type>());
    auto const found_scan  = cg::inclusive_scan(warp, found);
    // handy character counter for threads in this warp
    auto const chars_scan = cg::exclusive_scan(warp, is_char);
    // activate the thread where we hit the desired find instance
    auto const found_pos = (found_scan + offset) == (instance + 1) ? chars_scan : char_pos;
    // copy the position value for that thread into all warp threads
    char_pos = cg::reduce(warp, found_pos, cg::less<size_type>());
    if (char_pos < max_pos) { break; }  // all threads will stop
    offset += found_count;              // otherwise continue with the next set
    char_count += cg::reduce(warp, is_char, cg::plus<size_type>());
    itr += cudf::detail::warp_size;
    byte_count += cudf::detail::warp_size;
  }

  // output the position if an instance match has been found
  if (lane_idx == 0) { d_results[str_idx] = char_pos == max_pos ? -1 : char_pos + char_count; }
}

}  // namespace

std::unique_ptr<column> find_instance(strings_column_view const& input,
                                      string_scalar const& target,
                                      size_type instance,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
    instance >= 0, "Parameter instance must be positive integer or zero.", std::invalid_argument);
  CUDF_EXPECTS(target.is_valid(stream), "Parameter target must be valid.", std::invalid_argument);

  // create output column
  auto results = make_numeric_column(data_type{type_to_id<size_type>()},
                                     input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  // if input is empty or all-null then we are done
  if (input.size() == input.null_count()) { return results; }

  auto d_target  = target.value(stream);
  auto d_strings = column_device_view::create(input.parent(), stream);
  auto d_results = results->mutable_view().data<size_type>();

  constexpr thread_index_type block_size = 256;
  constexpr thread_index_type warp_size  = cudf::detail::warp_size;
  static_assert(block_size % warp_size == 0, "block size must be a multiple of warp size");
  cudf::detail::grid_1d grid{input.size() * warp_size, block_size};
  find_instance_warp_parallel_fn<<<grid.num_blocks,
                                   grid.num_threads_per_block,
                                   0,
                                   stream.value()>>>(*d_strings, d_target, instance, d_results);
  CUDF_CUDA_TRY(cudaGetLastError());

  return results;
}

}  // namespace detail

std::unique_ptr<column> find_instance(strings_column_view const& input,
                                      string_scalar const& target,
                                      size_type instance,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::find_instance(input, target, instance, stream, mr);
}

}  // namespace strings
}  // namespace cudf
