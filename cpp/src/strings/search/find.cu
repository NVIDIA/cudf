/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/atomic>
#include <cuda/iterator>
#include <cuda/std/limits>
#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
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
 * Note that this value is shared by find, rfind, and contains functions.
 */
constexpr size_type AVG_CHAR_BYTES_THRESHOLD = 64;

/**
 * @brief Per-row byte-length threshold for heterogeneous contains.
 *
 * Rows with `size_bytes <= HETERO_LENGTH_THRESHOLD` are searched thread-per-row;
 * rows with `size_bytes > HETERO_LENGTH_THRESHOLD` are deferred to warp-per-row.
 */
constexpr size_type HETERO_LENGTH_THRESHOLD = 96;

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
  auto results = make_numeric_column(
    data_type{type_to_id<size_type>()}, input.size(), rmm::device_buffer{}, 0, stream, mr);
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

namespace detail {
namespace {

/**
 * @brief Search short rows and collect long-row indices in a single pass.
 *
 * Executes one thread per string/row, fusing the length-based partitioning with the thread-per-row
 * search of the short rows:
 *  - null rows: write `false` (the output null mask already marks these rows invalid)
 *  - short rows (`size_bytes <= length_threshold`): run the thread-per-row contains scan and write
 *    the result directly
 *  - long rows (`size_bytes > length_threshold`): append the row index to `d_long_indices` for the
 *    subsequent warp-parallel pass
 *
 * The long-index append uses a warp-aggregated atomic so each group of coalesced long-row threads
 * performs a single `atomicAdd` on `d_long_count`, minimizing global-atomic contention.
 *
 * @param d_strings Column of input strings
 * @param d_target String to search for in each row of `d_strings`
 * @param d_results Indicates which rows contain `d_target`
 * @param length_threshold Rows with more bytes than this are deferred to the warp-parallel pass
 * @param d_long_indices Output list of deferred long-row indices (capacity == d_strings.size())
 * @param d_long_count Output count of entries written to `d_long_indices`
 */
CUDF_KERNEL void contains_string_per_thread_heterogeneous(column_device_view const d_strings,
                                                          string_view const d_target,
                                                          bool* d_results,
                                                          size_type const length_threshold,
                                                          size_type* d_long_indices,
                                                          size_type* d_long_count)
{
  auto const tidx = cudf::detail::grid_1d::global_thread_id();
  if (tidx >= static_cast<cudf::thread_index_type>(d_strings.size())) { return; }
  auto const str_idx = static_cast<size_type>(tidx);

  if (d_strings.is_null(str_idx)) {
    d_results[str_idx] = false;
    return;
  }

  auto const d_str = d_strings.element<string_view>(str_idx);

  if (d_str.size_bytes() <= length_threshold) {
    auto const target_bytes = d_target.size_bytes();
    auto found              = false;
    for (size_type i = 0; !found && ((i + target_bytes) <= d_str.size_bytes()); ++i) {
      found = d_target.compare(d_str.data() + i, target_bytes) == 0;
    }
    d_results[str_idx] = found;
    return;
  }

  // Long row: defer to the warp-parallel pass. Coalesced long-row threads aggregate their appends
  // into a single atomicAdd, then each writes its index at a distinct offset.
  namespace cg      = cooperative_groups;
  auto const active = cg::coalesced_threads();
  size_type offset  = 0;
  if (active.thread_rank() == 0) {
    offset = atomicAdd(d_long_count, static_cast<size_type>(active.num_threads()));
  }
  offset                                                                = active.shfl(offset, 0);
  d_long_indices[offset + static_cast<size_type>(active.thread_rank())] = str_idx;
}

/**
 * @brief Check if `d_target` appears in each deferred long row of `d_strings`.
 *
 * This executes as a warp per string/row and performs well for longer strings. Only the (non-null)
 * long-row indices collected by ::contains_string_per_thread_heterogeneous are processed.
 * @see AVG_CHAR_BYTES_THRESHOLD
 *
 * The row count is read from `d_long_count` on the device so the two passes launch back-to-back on
 * the stream with no intervening host synchronization. A fixed (grid-stride) launch is used: each
 * warp strides over the deferred rows, so the second pass works for any `*d_long_count` without the
 * host needing to know the value to size the grid.
 *
 * @param d_strings Column of input strings
 * @param d_target String to search for in each row of `d_strings`
 * @param d_results Indicates which rows contain `d_target`
 * @param d_long_indices Indices of the long (non-null) rows to process
 * @param d_long_count Number of entries in `d_long_indices` (read on the device)
 */
CUDF_KERNEL void contains_warp_parallel_fn_heterogeneous(column_device_view const d_strings,
                                                         string_view const d_target,
                                                         bool* d_results,
                                                         size_type const* d_long_indices,
                                                         size_type const* d_long_count)
{
  auto const long_count = *d_long_count;

  namespace cg        = cooperative_groups;
  auto const warp     = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto const lane_idx = warp.thread_rank();

  // Grid-stride over the deferred long rows, one warp per row.
  auto const first_warp = cudf::detail::grid_1d::global_thread_id() / cudf::detail::warp_size;
  auto const warp_stride =
    (static_cast<cudf::thread_index_type>(gridDim.x) * blockDim.x) / cudf::detail::warp_size;

  for (auto str_idx = first_warp; str_idx < static_cast<cudf::thread_index_type>(long_count);
       str_idx += warp_stride) {
    // Long indices are guaranteed non-null by the partitioning kernel.
    auto const real_str_idx = d_long_indices[str_idx];
    auto const d_str        = d_strings.element<string_view>(real_str_idx);

    auto constexpr bytes_per_warp = 4;
    auto found                    = false;
    // All lanes must participate in every warp.any() call. Using `!found` or a per-lane bounds
    // check as the loop condition would cause lanes to exit at different iterations, leaving the
    // remaining warp.any() calls with only a partial tile. Instead, keep all lanes in lockstep
    // via an unconditional while(true) and guard per-lane work with in-loop predicates.
    auto i = static_cast<cudf::size_type>(lane_idx * bytes_per_warp);
    while (true) {
      bool const in_bounds = (i + d_target.size_bytes()) <= d_str.size_bytes();
      // All lanes converge: stop when no lane has unchecked positions.
      if (!warp.any(in_bounds && !found)) break;
      if (in_bounds && !found) {
        for (auto j = 0; !found && (j < bytes_per_warp); j++) {
          if (((i + j + d_target.size_bytes()) <= d_str.size_bytes()) &&
              d_target.compare(d_str.data() + i + j, d_target.size_bytes()) == 0) {
            found = true;
          }
        }
      }
      // All lanes converge: early-exit as soon as any lane finds the target.
      if (warp.any(found)) break;
      i += cudf::detail::warp_size * bytes_per_warp;
    }

    auto const result = warp.any(found);
    if (lane_idx == 0) { d_results[real_str_idx] = result; }
  }
}

std::unique_ptr<column> contains_heterogeneous(strings_column_view const& input,
                                               string_scalar const& target,
                                               cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr)
{
  auto strings_count = input.size();
  if (strings_count == 0) return make_empty_column(type_id::BOOL8);

  CUDF_EXPECTS(target.is_valid(stream), "Parameter target must be valid.");

  auto d_target = string_view(target.data(), target.size());

  // create output column
  auto results      = make_numeric_column(data_type{type_id::BOOL8},
                                     strings_count,
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  auto results_view = results->mutable_view();
  auto d_results    = results_view.data<bool>();

  // Handle empty target
  if (d_target.empty()) {
    thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                 results_view.begin<bool>(),
                 results_view.end<bool>(),
                 true);
    results->set_null_count(input.null_count());
    return results;
  }

  auto strings_column = column_device_view::create(input.parent(), stream);
  auto d_strings      = *strings_column;

  constexpr int block_size = 256;

  // First pass (thread-per-string): search the short rows in place and collect the indices of the
  // long rows (> length_threshold bytes) for the warp-parallel pass. `long_indices` is sized for
  // the worst case (all rows long); only the first `long_count` entries are populated.
  size_type const length_threshold = HETERO_LENGTH_THRESHOLD;
  rmm::device_uvector<size_type> long_indices(strings_count, stream);
  cudf::detail::device_scalar<size_type> d_long_count(0, stream);

  auto const grid = cudf::detail::grid_1d{strings_count, block_size};
  contains_string_per_thread_heterogeneous<<<grid.num_blocks,
                                             grid.num_threads_per_block,
                                             0,
                                             stream.get()>>>(
    d_strings, d_target, d_results, length_threshold, long_indices.data(), d_long_count.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  // Second pass (warp-per-string): search the deferred long rows. The deferred count is read on the
  // device inside the kernel, so this launches back-to-back with the first pass and never
  // synchronizes the stream (a host readback here dominates the runtime on small columns).
  //
  // A fixed grid-stride launch is used: size it for full occupancy but never exceed the worst case
  // of one warp per row, so short-heavy columns launch only a handful of blocks that read
  // `*d_long_count == 0` and exit immediately.
  auto constexpr warps_per_block = block_size / cudf::detail::warp_size;
  int max_blocks_per_sm          = 0;
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_blocks_per_sm, contains_warp_parallel_fn_heterogeneous, block_size, 0));
  auto const persistent_blocks =
    static_cast<int64_t>(max_blocks_per_sm) * cudf::detail::num_multiprocessors();
  auto const worst_case_blocks =
    cudf::util::div_rounding_up_safe<int64_t>(strings_count, warps_per_block);
  auto const num_warp_blocks = std::max<int64_t>(1, std::min(persistent_blocks, worst_case_blocks));

  contains_warp_parallel_fn_heterogeneous<<<num_warp_blocks, block_size, 0, stream.get()>>>(
    d_strings, d_target, d_results, long_indices.data(), d_long_count.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  results->set_null_count(input.null_count());
  return results;
}

/**
 * @brief Utility to return a bool column indicating the presence of
 * a given target string in a strings column.
 *
 * Null string entries return corresponding null output column entries.
 *
 * @tparam BoolFunction Return bool value given two strings.
 *
 * @param strings Column of strings to check for target.
 * @param target UTF-8 encoded string to check in strings column.
 * @param pfn Returns bool value if target is found in the given string.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New BOOL column.
 */
template <typename BoolFunction>
std::unique_ptr<column> contains_fn(strings_column_view const& strings,
                                    string_scalar const& target,
                                    BoolFunction pfn,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  auto strings_count = strings.size();
  if (strings_count == 0) return make_empty_column(type_id::BOOL8);

  CUDF_EXPECTS(target.is_valid(stream), "Parameter target must be valid.");
  if (target.size() == 0)  // empty target string returns true
  {
    auto const true_scalar =
      make_fixed_width_scalar<bool>(true, stream, cudf::get_current_device_resource_ref());
    auto results = make_column_from_scalar(*true_scalar, strings.size(), stream, mr);
    results->set_null_mask(cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                           strings.null_count());
    return results;
  }

  auto d_target       = string_view(target.data(), target.size());
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  // create output column
  auto results      = make_numeric_column(data_type{type_id::BOOL8},
                                     strings_count,
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto results_view = results->mutable_view();
  auto d_results    = results_view.data<bool>();
  // set the bool values by evaluating the passed function
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{strings_count},
                    d_results,
                    [d_strings, pfn, d_target] __device__(size_type idx) {
                      return !d_strings.is_null(idx) &&
                             bool{pfn(d_strings.element<string_view>(idx), d_target)};
                    });
  results->set_null_count(strings.null_count());
  return results;
}

/**
 * @brief Utility to return a bool column indicating the presence of
 * a string targets[i] in strings[i].
 *
 * Null string entries return corresponding null output column entries.
 *
 * @tparam BoolFunction Return bool value given two strings.
 *
 * @param strings Column of strings to check for `targets[i]`.
 * @param targets Column of strings to be checked in `strings[i]``.
 * @param pfn Returns bool value if target is found in the given string.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New BOOL column.
 */
template <typename BoolFunction>
std::unique_ptr<column> contains_fn(strings_column_view const& strings,
                                    strings_column_view const& targets,
                                    BoolFunction pfn,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  if (strings.is_empty()) return make_empty_column(type_id::BOOL8);

  CUDF_EXPECTS(targets.size() == strings.size(),
               "strings and targets column must be the same size");

  auto targets_column = column_device_view::create(targets.parent(), stream);
  auto d_targets      = *targets_column;
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  // create output column
  auto results      = make_numeric_column(data_type{type_id::BOOL8},
                                     strings.size(),
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto results_view = results->mutable_view();
  auto d_results    = results_view.data<bool>();
  // set the bool values by evaluating the passed function
  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{strings.size()},
    d_results,
    [d_strings, pfn, d_targets] __device__(size_type idx) {
      // empty target string returns true
      if (d_targets.is_valid(idx) && d_targets.element<string_view>(idx).length() == 0) {
        return true;
      } else if (!d_strings.is_null(idx) && !d_targets.is_null(idx)) {
        return bool{pfn(d_strings.element<string_view>(idx), d_targets.element<string_view>(idx))};
      } else {
        return false;
      }
    });
  results->set_null_count(strings.null_count());
  return results;
}
/**
 * @brief Warp-per-string contains check; fastest when most strings are long (avg >= threshold).
 */
CUDF_KERNEL void contains_warp_parallel_fn(column_device_view const d_strings,
                                           string_view const d_target,
                                           bool* d_results)
{
  auto const idx     = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx = idx / cudf::detail::warp_size;
  if (str_idx >= d_strings.size()) { return; }

  namespace cg        = cooperative_groups;
  auto const warp     = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto const lane_idx = warp.thread_rank();

  if (d_strings.is_null(str_idx)) { return; }
  auto const d_str              = d_strings.element<string_view>(str_idx);
  auto constexpr bytes_per_warp = 4;
  auto found                    = false;
  for (auto i = lane_idx * bytes_per_warp;
       !found && ((i + d_target.size_bytes()) <= d_str.size_bytes());
       i += cudf::detail::warp_size * bytes_per_warp) {
    for (auto j = 0; !found && (j < bytes_per_warp); j++) {
      if (((i + j + d_target.size_bytes()) <= d_str.size_bytes()) &&
          d_target.compare(d_str.data() + i + j, d_target.size_bytes()) == 0)
        found = true;
    }
  }
  auto const result = warp.any(found);
  if (lane_idx == 0) { d_results[str_idx] = result; }
}

std::unique_ptr<column> contains_warp_parallel(strings_column_view const& input,
                                               string_scalar const& target,
                                               cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(target.is_valid(stream), "Parameter target must be valid.");
  auto d_target     = string_view(target.data(), target.size());
  auto results      = make_numeric_column(data_type{type_id::BOOL8},
                                     input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  auto results_view = results->mutable_view();
  if (d_target.empty()) {
    thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                 results_view.begin<bool>(),
                 results_view.end<bool>(),
                 true);
  } else {
    auto const d_strings                   = column_device_view::create(input.parent(), stream);
    constexpr thread_index_type block_size = 256;
    constexpr thread_index_type warp_size  = cudf::detail::warp_size;
    cudf::detail::grid_1d grid{input.size() * warp_size, block_size};
    contains_warp_parallel_fn<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
      *d_strings, d_target, results_view.data<bool>());
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  results->set_null_count(input.null_count());
  return results;
}

}  // namespace

std::unique_ptr<column> contains(strings_column_view const& input,
                                 string_scalar const& target,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) return make_empty_column(type_id::BOOL8);

  // Compute avg_bytes to dispatch. The baseline (main branch) already pays this O(1) D2H cost.
  auto const avg_bytes = static_cast<size_type>(input.chars_size(stream) / input.size());

  // Warp-per-string: best when most strings are long (avg >= threshold).
  if (avg_bytes >= HETERO_LENGTH_THRESHOLD) {
    return contains_warp_parallel(input, target, stream, mr);
  }

  // Heterogeneous: thread-per-short + warp-per-long. Gives large speedups vs the old
  // warp-per-all path for mixed-width columns. Selected on average width only; the two-pass
  // overhead is bounded because the second launch reads `*d_long_count == 0` and exits.
  if (avg_bytes > AVG_CHAR_BYTES_THRESHOLD) {
    return contains_heterogeneous(input, target, stream, mr);
  }

  // Thread-per-string: matches baseline for mostly-short columns (avg <= 64).
  auto pfn = [] __device__(string_view d_string, string_view d_target) {
    for (size_type i = 0; i <= (d_string.size_bytes() - d_target.size_bytes()); ++i) {
      if (d_target.compare(d_string.data() + i, d_target.size_bytes()) == 0) { return true; }
    }
    return false;
  };
  return contains_fn(input, target, pfn, stream, mr);
}

std::unique_ptr<column> contains(strings_column_view const& strings,
                                 strings_column_view const& targets,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  auto pfn = [] __device__(string_view d_string, string_view d_target) {
    for (size_type i = 0; i <= (d_string.size_bytes() - d_target.size_bytes()); ++i) {
      if (d_target.compare(d_string.data() + i, d_target.size_bytes()) == 0) { return true; }
    }
    return false;
  };
  return contains_fn(strings, targets, pfn, stream, mr);
}

std::unique_ptr<column> starts_with(strings_column_view const& strings,
                                    string_scalar const& target,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  auto pfn = [] __device__(string_view d_string, string_view d_target) {
    return (d_target.size_bytes() <= d_string.size_bytes()) &&
           (d_target.compare(d_string.data(), d_target.size_bytes()) == 0);
  };
  return contains_fn(strings, target, pfn, stream, mr);
}

std::unique_ptr<column> starts_with(strings_column_view const& strings,
                                    strings_column_view const& targets,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  auto pfn = [] __device__(string_view d_string, string_view d_target) {
    return (d_target.size_bytes() <= d_string.size_bytes()) &&
           (d_target.compare(d_string.data(), d_target.size_bytes()) == 0);
  };
  return contains_fn(strings, targets, pfn, stream, mr);
}

std::unique_ptr<column> ends_with(strings_column_view const& strings,
                                  string_scalar const& target,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  auto pfn = [] __device__(string_view d_string, string_view d_target) {
    auto const str_size = d_string.size_bytes();
    auto const tgt_size = d_target.size_bytes();
    return (tgt_size <= str_size) &&
           (d_target.compare(d_string.data() + str_size - tgt_size, tgt_size) == 0);
  };

  return contains_fn(strings, target, pfn, stream, mr);
}

std::unique_ptr<column> ends_with(strings_column_view const& strings,
                                  strings_column_view const& targets,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  auto pfn = [] __device__(string_view d_string, string_view d_target) {
    auto const str_size = d_string.size_bytes();
    auto const tgt_size = d_target.size_bytes();
    return (tgt_size <= str_size) &&
           (d_target.compare(d_string.data() + str_size - tgt_size, tgt_size) == 0);
  };

  return contains_fn(strings, targets, pfn, stream, mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<column> contains(strings_column_view const& strings,
                                 string_scalar const& target,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(strings, target, stream, mr);
}

std::unique_ptr<column> contains(strings_column_view const& strings,
                                 strings_column_view const& targets,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(strings, targets, stream, mr);
}

std::unique_ptr<column> starts_with(strings_column_view const& strings,
                                    string_scalar const& target,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::starts_with(strings, target, stream, mr);
}

std::unique_ptr<column> starts_with(strings_column_view const& strings,
                                    strings_column_view const& targets,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::starts_with(strings, targets, stream, mr);
}

std::unique_ptr<column> ends_with(strings_column_view const& strings,
                                  string_scalar const& target,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::ends_with(strings, target, stream, mr);
}

std::unique_ptr<column> ends_with(strings_column_view const& strings,
                                  strings_column_view const& targets,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::ends_with(strings, targets, stream, mr);
}

}  // namespace strings
}  // namespace cudf
