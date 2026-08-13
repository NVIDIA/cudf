/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/limits>
#include <cuda/std/utility>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

constexpr size_type AVG_CHAR_BYTES_THRESHOLD = 128;

/**
 * @brief Function logic for compute_substrings_from_fn API
 *
 * This computes the output size and resolves the substring
 */
template <typename IndexIterator>
struct substring_from_fn {
  column_device_view const d_column;
  IndexIterator const starts;
  IndexIterator const stops;

  __device__ string_index_pair operator()(size_type idx) const
  {
    if (d_column.is_null(idx)) { return string_index_pair{nullptr, 0}; }
    auto const d_str  = d_column.template element<string_view>(idx);
    auto const length = d_str.length();
    auto const start  = cuda::std::max(starts[idx], 0);
    if (start >= length) { return string_index_pair{"", 0}; }

    auto const stop    = stops[idx];
    auto const end     = (((stop < 0) || (stop > length)) ? length : stop);
    auto const sub_str = start < end ? d_str.substr(start, end - start) : string_view{};
    return sub_str.empty() ? string_index_pair{"", 0}
                           : string_index_pair{sub_str.data(), sub_str.size_bytes()};
  }

  substring_from_fn(column_device_view const& d_column, IndexIterator starts, IndexIterator stops)
    : d_column(d_column), starts(starts), stops(stops)
  {
  }
};

template <typename IndexIterator>
CUDF_KERNEL void substring_from_kernel(column_device_view const d_strings,
                                       IndexIterator starts,
                                       IndexIterator stops,
                                       string_index_pair* d_output)
{
  auto const idx     = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx = idx / cudf::detail::warp_size;
  if (str_idx >= d_strings.size()) { return; }

  namespace cg    = cooperative_groups;
  auto const warp = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());

  if (d_strings.is_null(str_idx)) {
    if (warp.thread_rank() == 0) { d_output[str_idx] = string_index_pair{nullptr, 0}; }
    return;
  }
  auto const d_str = d_strings.element<cudf::string_view>(str_idx);
  if (d_str.empty()) {
    if (warp.thread_rank() == 0) { d_output[str_idx] = string_index_pair{"", 0}; }
    return;
  }

  auto const start = max(starts[str_idx], 0);
  auto stop        = [stop = stops[str_idx]] {
    return (stop < 0) ? cuda::std::numeric_limits<size_type>::max() : stop;
  }();
  auto const end = d_str.data() + d_str.size_bytes();

  auto start_counts = cuda::std::make_pair(0, 0);
  auto stop_counts  = cuda::std::make_pair(0, 0);

  auto itr = d_str.data() + warp.thread_rank();

  size_type char_count = 0;
  size_type byte_count = 0;
  while (byte_count < d_str.size_bytes()) {
    if (char_count <= start) { start_counts = {char_count, byte_count}; }
    if (char_count <= stop) {
      stop_counts = {char_count, byte_count};
    } else {
      break;
    }
    size_type const cc = (itr < end) && is_begin_utf8_char(*itr);
    size_type const bc = (itr < end) ? bytes_in_utf8_byte(*itr) : 0;
    char_count += cg::reduce(warp, cc, cg::plus<int>());
    byte_count += cg::reduce(warp, bc, cg::plus<int>());
    itr += cudf::detail::warp_size;
  }

  warp.sync();

  if (warp.thread_rank() == 0) {
    if (start >= char_count) {
      d_output[str_idx] = string_index_pair{"", 0};
      return;
    }

    // we are just below start/stop and must now increment up to them from here
    auto first_byte = start_counts.second;
    if (start_counts.first < start) {
      auto const sub_str = string_view(d_str.data() + first_byte, d_str.size_bytes() - first_byte);
      first_byte +=
        cuda::std::get<0>(bytes_to_character_position(sub_str, start - start_counts.first));
    }

    stop           = min(stop, char_count);
    auto last_byte = stop_counts.second;
    if (stop_counts.first < stop) {
      auto const sub_str = string_view(d_str.data() + last_byte, d_str.size_bytes() - last_byte);
      last_byte +=
        cuda::std::get<0>(bytes_to_character_position(sub_str, stop - stop_counts.first));
    }

    d_output[str_idx] = (first_byte < last_byte)
                          ? string_index_pair{d_str.data() + first_byte, last_byte - first_byte}
                          : string_index_pair{"", 0};
  }
}

/**
 * @brief Function logic for the substring API.
 *
 * This will perform a substring operation on each string
 * using the provided start, stop, and step parameters.
 */
struct substring_fn {
  column_device_view const d_column;
  cuda::std::optional<size_type> const d_start;
  cuda::std::optional<size_type> const d_stop;
  size_type const d_step;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(size_type idx)
  {
    if (d_column.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    auto const d_str  = d_column.template element<string_view>(idx);
    auto const length = d_str.length();
    if (length == 0) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    auto const step  = d_step;
    auto const begin = [&] {  // always inclusive
      // when invalid, default depends on step
      if (!d_start.has_value()) { return (step > 0) ? d_str.begin() : (d_str.end() - 1); }
      // normal positive position logic
      auto start = d_start.value();
      if (start >= 0) {
        if (start < length) return d_str.begin() + start;
        return d_str.end() + (step < 0 ? -1 : 0);
      }
      // handle negative position here
      auto adjust = length + start;
      if (adjust >= 0) return d_str.begin() + adjust;
      return d_str.begin() + (step < 0 ? -1 : 0);
    }();
    auto const end = [&] {  // always exclusive
      // when invalid, default depends on step
      if (!d_stop.has_value()) { return step > 0 ? d_str.end() : (d_str.begin() - 1); }
      // normal positive position logic
      auto stop = d_stop.value();
      if (stop >= 0) return (stop < length) ? (d_str.begin() + stop) : d_str.end();
      // handle negative position here
      auto adjust = length + stop;
      return d_str.begin() + (adjust >= 0 ? adjust : -1);
    }();

    size_type bytes = 0;
    char* d_buffer  = d_chars ? d_chars + d_offsets[idx] : nullptr;
    auto itr        = begin;
    while (step > 0 ? itr < end : end < itr) {
      if (d_buffer) {
        d_buffer += from_char_utf8(*itr, d_buffer);
      } else {
        bytes += bytes_in_char_utf8(*itr);
      }
      itr += step;
    }
    if (!d_chars) { d_sizes[idx] = bytes; }
  }
};

/**
 * @brief Common utility function for the slice_strings APIs
 *
 * It wraps calling the functors appropriately to build the output strings column.
 *
 * The input iterators may have unique position values per string in `d_column`.
 * This can also be called with constant value iterators to handle special
 * slice functions if possible.
 *
 * @tparam IndexIterator Iterator type for character position values
 *
 * @param input Input strings column to substring
 * @param starts Start positions index iterator
 * @param stops Stop positions index iterator
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
template <typename IndexIterator>
std::unique_ptr<column> compute_substrings_from_fn(strings_column_view const& input,
                                                   IndexIterator starts,
                                                   IndexIterator stops,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  auto results = rmm::device_uvector<string_index_pair>(input.size(), stream);

  auto const d_column = column_device_view::create(input.parent(), stream);

  if ((input.chars_size(stream) / (input.size() - input.null_count())) < AVG_CHAR_BYTES_THRESHOLD) {
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{input.size()},
                      results.begin(),
                      substring_from_fn{*d_column, starts, stops});
  } else {
    constexpr thread_index_type block_size = 512;
    auto const threads =
      static_cast<cudf::thread_index_type>(input.size()) * cudf::detail::warp_size;
    auto const num_blocks = util::div_rounding_up_safe(threads, block_size);
    substring_from_kernel<IndexIterator>
      <<<num_blocks, block_size, 0, stream.value()>>>(*d_column, starts, stops, results.data());
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  return make_strings_column(results.begin(), results.end(), stream, mr);
}

}  // namespace

std::unique_ptr<column> slice_strings(strings_column_view const& input,
                                      numeric_scalar<size_type> const& start,
                                      numeric_scalar<size_type> const& stop,
                                      numeric_scalar<size_type> const& step,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  if (input.size() == input.null_count()) {
    return std::make_unique<column>(input.parent(), stream, mr);
  }

  auto const step_valid = step.is_valid(stream);
  auto const step_value = step_valid ? step.value(stream) : 1;
  if (step_valid) { CUDF_EXPECTS(step_value != 0, "Step parameter must not be 0"); }

  // optimization for (step==1 and start < stop) -- expect this to be most common
  if (step_value == 1) {
    auto const start_value = start.is_valid(stream) ? start.value(stream) : 0;
    auto const stop_value =
      stop.is_valid(stream) ? stop.value(stream) : std::numeric_limits<size_type>::max();
    // note that any negative values here must use the alternate function below
    if ((start_value >= 0) && (start_value < stop_value)) {
      // this is about 2x faster on long strings for this common case
      return compute_substrings_from_fn(input,
                                        cuda::constant_iterator<size_type>(start_value),
                                        cuda::constant_iterator<size_type>(stop_value),
                                        stream,
                                        mr);
    }
  }

  auto const d_column = column_device_view::create(input.parent(), stream);

  auto const d_start = start.is_valid(stream) ? cuda::std::optional<size_type>{start.value(stream)}
                                              : cuda::std::nullopt;
  auto const d_stop =
    stop.is_valid(stream) ? cuda::std::optional<size_type>{stop.value(stream)} : cuda::std::nullopt;

  auto [offsets, chars] = make_strings_children(
    substring_fn{*d_column, d_start, d_stop, step_value}, input.size(), stream, mr);

  return make_strings_column(input.size(),
                             std::move(offsets),
                             chars.release(),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

std::unique_ptr<column> slice_strings(strings_column_view const& input,
                                      std::optional<size_type> start,
                                      std::optional<size_type> stop,
                                      std::optional<size_type> step,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  if (input.size() == input.null_count()) {
    return std::make_unique<column>(input.parent(), stream, mr);
  }

  auto const step_valid = step.has_value();
  auto const step_value = step_valid ? step.value() : 1;
  if (step_valid) { CUDF_EXPECTS(step_value != 0, "Step parameter must not be 0"); }

  // optimization for (step==1 and start < stop) -- expect this to be most common
  if (step_value == 1) {
    auto const start_value = start.has_value() ? start.value() : 0;
    auto const stop_value = stop.has_value() ? stop.value() : std::numeric_limits<size_type>::max();
    // note that any negative values here must use the alternate function below
    if ((start_value >= 0) && (start_value < stop_value)) {
      // this is about 2x faster on long strings for this common case
      return compute_substrings_from_fn(input,
                                        cuda::constant_iterator<size_type>(start_value),
                                        cuda::constant_iterator<size_type>(stop_value),
                                        stream,
                                        mr);
    }
  }

  auto const d_column = column_device_view::create(input.parent(), stream);

  auto const d_start =
    start.has_value() ? cuda::std::optional<size_type>{start.value()} : cuda::std::nullopt;
  auto const d_stop =
    stop.has_value() ? cuda::std::optional<size_type>{stop.value()} : cuda::std::nullopt;

  auto [offsets, chars] = make_strings_children(
    substring_fn{*d_column, d_start, d_stop, step_value}, input.size(), stream, mr);

  return make_strings_column(input.size(),
                             std::move(offsets),
                             chars.release(),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

std::unique_ptr<column> slice_strings(strings_column_view const& input,
                                      column_view const& starts_column,
                                      column_view const& stops_column,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  if (input.size() == input.null_count()) {
    return std::make_unique<column>(input.parent(), stream, mr);
  }

  CUDF_EXPECTS(starts_column.size() == input.size(),
               "Parameter starts must have the same number of rows as strings.");
  CUDF_EXPECTS(stops_column.size() == input.size(),
               "Parameter stops must have the same number of rows as strings.");
  CUDF_EXPECTS(starts_column.null_count() == 0, "Parameter starts must not contain nulls.");
  CUDF_EXPECTS(stops_column.null_count() == 0, "Parameter stops must not contain nulls.");

  auto starts_iter = cudf::detail::indexalator_factory::make_input_iterator(starts_column);
  auto stops_iter  = cudf::detail::indexalator_factory::make_input_iterator(stops_column);
  return compute_substrings_from_fn(input, starts_iter, stops_iter, stream, mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> slice_strings(strings_column_view const& input,
                                      numeric_scalar<size_type> const& start,
                                      numeric_scalar<size_type> const& stop,
                                      numeric_scalar<size_type> const& step,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::slice_strings(input, start, stop, step, stream, mr);
}

std::unique_ptr<column> slice_strings(strings_column_view const& input,
                                      std::optional<size_type> start,
                                      std::optional<size_type> stop,
                                      std::optional<size_type> step,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::slice_strings(input, start, stop, step, stream, mr);
}

std::unique_ptr<column> slice_strings(strings_column_view const& input,
                                      column_view const& starts_column,
                                      column_view const& stops_column,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::slice_strings(input, starts_column, stops_column, stream, mr);
}

}  // namespace strings
}  // namespace cudf
