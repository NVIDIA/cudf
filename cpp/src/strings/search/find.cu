/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/iterator>
#include <cuda/std/limits>
#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Threshold to decide on using string or warp parallel functions.
 *
 * If the average byte length of a string in a column exceeds this value then
 * a warp-parallel function is used.
 *
 * Note that this value is shared by the find and rfind functions.
 */
constexpr size_type AVG_CHAR_BYTES_THRESHOLD = 64;

/**
 * @brief Find function handles a string per thread
 */
template <typename TargetIterator, bool forward = true>
struct finder_fn {
  column_device_view const d_strings;
  TargetIterator const d_targets;
  size_type const start;
  size_type const stop;

  __device__ size_type operator()(size_type idx) const
  {
    if (d_strings.is_null(idx)) { return -1; }
    auto const d_str = d_strings.element<string_view>(idx);
    if (d_str.empty() && (start > 0)) { return -1; }
    if (stop >= 0 && start > stop) { return -1; }
    auto const d_target = d_targets[idx];

    auto const count = (stop < 0) ? stop : (stop - start);
    return forward ? d_str.find(d_target, start, count) : d_str.rfind(d_target, start, count);
  }
};

/**
 * @brief Special logic handles an empty target for find/rfind
 *
 * where length = number of characters in the input string
 * if forward = true:
 *   return start iff (start <= length), otherwise return -1
 * if forward = false:
 *   return stop iff (0 <= stop <= length), otherwise return length
 */
template <bool forward = true>
struct empty_target_fn {
  column_device_view const d_strings;
  size_type const start;
  size_type const stop;

  __device__ size_type operator()(size_type idx) const
  {
    if (d_strings.is_null(idx)) { return -1; }
    auto d_str = d_strings.element<string_view>(idx);

    // common case shortcut
    if (forward && start == 0) { return 0; }

    auto const length = d_str.length();
    if (start > length) { return -1; }
    if constexpr (forward) { return start; }

    return (stop < 0) || (stop > length) ? length : stop;
  }
};

/**
 * @brief String per warp function for find/rfind
 */
template <typename TargetIterator, bool forward = true>
CUDF_KERNEL void finder_warp_parallel_fn(column_device_view const d_strings,
                                         TargetIterator const d_targets,
                                         size_type const start,
                                         size_type const stop,
                                         size_type* d_results)
{
  namespace cg        = cooperative_groups;
  auto const warp     = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto const lane_idx = warp.thread_rank();

  auto const tid     = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx = tid / cudf::detail::warp_size;
  if (str_idx >= d_strings.size() or d_strings.is_null(str_idx)) { return; }

  auto const d_str    = d_strings.element<string_view>(str_idx);
  auto const d_target = d_targets[str_idx];

  auto const [begin, left_over] = bytes_to_character_position(d_str, start);
  auto const start_char_pos     = start - left_over;  // keep track of character position

  auto const end = [d_str, start, stop, begin = begin] {
    if (stop < 0) { return d_str.size_bytes(); }
    if (stop <= start) { return begin; }
    // we count from `begin` instead of recounting from the beginning of the string
    return begin + cuda::std::get<0>(bytes_to_character_position(
                     string_view(d_str.data() + begin, d_str.size_bytes() - begin), stop - start));
  }();

  // each thread compares the target with the thread's individual starting byte
  size_type position = forward ? cuda::std::numeric_limits<size_type>::max() : -1;
  for (auto itr = begin + lane_idx; itr + d_target.size_bytes() <= end;
       itr += cudf::detail::warp_size) {
    if (d_target.compare(d_str.data() + itr, d_target.size_bytes()) == 0) {
      position = itr;
      if (forward) break;
    }
  }

  // find stores the minimum position while rfind stores the maximum position
  auto const result = forward ? cg::reduce(warp, position, cg::less<size_type>())
                              : cg::reduce(warp, position, cg::greater<size_type>());

  if (lane_idx == 0) {
    // the final result needs to be fixed up convert max() to -1
    // and a byte position to a character position
    d_results[str_idx] =
      ((result < cuda::std::numeric_limits<size_type>::max()) && (result >= begin))
        ? start_char_pos + characters_in_string(d_str.data() + begin, result - begin)
        : -1;
  }
}

template <typename TargetIterator, bool forward = true>
void find_utility(strings_column_view const& input,
                  TargetIterator const& target_itr,
                  column& output,
                  size_type start,
                  size_type stop,
                  cuda::stream_ref stream)
{
  auto d_strings = column_device_view::create(input.parent(), stream);
  auto d_results = output.mutable_view().data<size_type>();
  if ((input.chars_size(stream) / (input.size() - input.null_count())) > AVG_CHAR_BYTES_THRESHOLD) {
    // warp-per-string runs faster for longer strings (but not shorter ones)
    constexpr auto block_size             = 256;
    constexpr thread_index_type warp_size = cudf::detail::warp_size;
    cudf::detail::grid_1d grid{input.size() * warp_size, block_size};
    finder_warp_parallel_fn<TargetIterator, forward>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
        *d_strings, target_itr, start, stop, d_results);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    // string-per-thread function
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{input.size()},
                      d_results,
                      finder_fn<TargetIterator, forward>{*d_strings, target_itr, start, stop});
  }
}

template <bool forward = true>
std::unique_ptr<column> find_fn(strings_column_view const& input,
                                string_scalar const& target,
                                size_type start,
                                size_type stop,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(target.is_valid(stream), "Parameter target must be valid.");
  CUDF_EXPECTS(start >= 0, "Parameter start must be positive integer or zero.");
  CUDF_EXPECTS(stop <= 0 or start <= stop, "Parameter start must be less than stop.");

  // create output column
  auto results = make_numeric_column(data_type{type_to_id<size_type>()},
                                     input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  // if input is empty or all-null then we are done
  if (input.size() == input.null_count()) { return results; }

  auto d_target = string_view(target.data(), target.size());

  // special logic for empty target results
  if (d_target.empty()) {
    auto d_strings = column_device_view::create(input.parent(), stream);
    auto d_results = results->mutable_view().data<size_type>();
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{input.size()},
                      d_results,
                      empty_target_fn<forward>{*d_strings, start, stop});
    return results;
  }

  // find-utility function fills in the results column
  auto target_itr      = cuda::make_constant_iterator(d_target);
  using TargetIterator = decltype(target_itr);
  find_utility<TargetIterator, forward>(input, target_itr, *results, start, stop, stream);
  results->set_null_count(input.null_count());
  return results;
}
}  // namespace

std::unique_ptr<column> find(strings_column_view const& input,
                             string_scalar const& target,
                             size_type start,
                             size_type stop,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
{
  return find_fn<true>(input, target, start, stop, stream, mr);
}

std::unique_ptr<column> rfind(strings_column_view const& input,
                              string_scalar const& target,
                              size_type start,
                              size_type stop,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  return find_fn<false>(input, target, start, stop, stream, mr);
}

template <bool forward = true>
std::unique_ptr<column> find(strings_column_view const& input,
                             strings_column_view const& target,
                             size_type start,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(start >= 0, "Parameter start must be positive integer or zero.");
  CUDF_EXPECTS(input.size() == target.size(), "input and target columns must be the same size");

  // create output column
  auto results = make_numeric_column(data_type{type_to_id<size_type>()},
                                     input.size(),
                                     cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                     0,
                                     stream,
                                     mr);
  // if input is empty or all-null then we are done
  if (input.size() == input.null_count()) { return results; }

  // call find utility with target iterator
  auto d_targets  = column_device_view::create(target.parent(), stream);
  auto target_itr = cudf::detail::make_null_replacement_iterator<string_view>(
    *d_targets, string_view{}, target.has_nulls());
  find_utility<decltype(target_itr), forward>(input, target_itr, *results, start, -1, stream);

  // AND the bitmasks from input and target
  auto [null_mask, null_count] =
    cudf::detail::bitmask_and(table_view({input.parent(), target.parent()}), stream, mr);
  results->set_null_mask(std::move(null_mask), null_count);
  return results;
}

}  // namespace detail

// external APIs

std::unique_ptr<column> find(strings_column_view const& strings,
                             string_scalar const& target,
                             size_type start,
                             size_type stop,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::find(strings, target, start, stop, stream, mr);
}

std::unique_ptr<column> rfind(strings_column_view const& strings,
                              string_scalar const& target,
                              size_type start,
                              size_type stop,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::rfind(strings, target, start, stop, stream, mr);
}

std::unique_ptr<column> find(strings_column_view const& input,
                             strings_column_view const& target,
                             size_type start,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::find<true>(input, target, start, stream, mr);
}

}  // namespace strings
}  // namespace cudf
