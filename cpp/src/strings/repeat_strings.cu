/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/stream>
#include <thrust/for_each.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

namespace cudf {
namespace strings {
namespace detail {
std::unique_ptr<string_scalar> repeat_string(string_scalar const& input,
                                             size_type repeat_times,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  if (!input.is_valid(stream)) { return std::make_unique<string_scalar>("", false, stream, mr); }
  if (input.size() == 0 || repeat_times <= 0) {
    return std::make_unique<string_scalar>("", true, stream, mr);
  }
  if (repeat_times == 1) { return std::make_unique<string_scalar>(input, stream, mr); }

  CUDF_EXPECTS(input.size() <= std::numeric_limits<size_type>::max() / repeat_times,
               "The output size exceeds the column size limit",
               std::overflow_error);

  auto const str_size = input.size();
  auto const iter     = cuda::counting_iterator<cudf::size_type>{0};
  auto buff           = rmm::device_buffer(repeat_times * input.size(), stream, mr);

  // Pull data from the input string into each byte of the output string.
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    iter,
                    iter + repeat_times * str_size,
                    static_cast<char*>(buff.data()),
                    [in_ptr = input.data(), str_size] __device__(auto const idx) {
                      return in_ptr[idx % str_size];
                    });

  return std::make_unique<string_scalar>(std::move(buff), true, stream, mr);
}

namespace {
/**
 * @brief Generate a strings column in which each row is an empty string or a null.
 *
 * The output strings column has the same bitmask as the input column.
 */
auto generate_empty_output(strings_column_view const& input,
                           size_type strings_count,
                           cuda::stream_ref stream,
                           rmm::device_async_resource_ref mr)
{
  auto offsets_column = make_numeric_column(
    data_type{type_id::INT32}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(offsets_column->mutable_view().template data<int32_t>(),
                                0,
                                offsets_column->size() * sizeof(int32_t),
                                stream.get()));

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             rmm::device_buffer{},
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

/**
 * @brief Functor to compute output string sizes and repeat the input strings.
 *
 * This functor is called only when `repeat_times > 0`. In addition, the total number of threads
 * running this functor is `repeat_times * strings_count` (instead of `string_count`) for maximizing
 * parallelism and better load-balancing.
 */
struct compute_size_and_repeat_fn {
  column_device_view const strings_dv;
  size_type const repeat_times;
  bool const has_nulls;
  size_type* d_sizes;
  char* d_chars;
  cudf::detail::input_offsetalator d_offsets;

  /**
   * @brief Called by make_strings_children to build output
   *
   * @param idx Thread index in the range [0,repeat_times * strings_count)
   * @param d_sizes Return output size here in 1st call (d_chars==nullptr)
   * @param d_chars Write output here in 2nd call
   * @param d_offsets Offsets to address output row within d_chars
   */
  __device__ void operator()(size_type idx) const noexcept
  {
    auto const str_idx    = idx / repeat_times;  // value cycles in [0, string_count)
    auto const repeat_idx = idx % repeat_times;  // value cycles in [0, repeat_times)
    auto const is_valid   = !has_nulls || strings_dv.is_valid_nocheck(str_idx);

    if (!d_chars && repeat_idx == 0) {
      d_sizes[str_idx] =
        is_valid ? repeat_times * strings_dv.element<string_view>(str_idx).size_bytes() : 0;
    }

    // Each input string will be copied by `repeat_times` threads into the output string.
    if (d_chars && is_valid) {
      auto const d_str    = strings_dv.element<string_view>(str_idx);
      auto const str_size = d_str.size_bytes();
      if (str_size > 0) {
        auto const input_ptr  = d_str.data();
        auto const output_ptr = d_chars + d_offsets[str_idx] + repeat_idx * str_size;
        std::memcpy(output_ptr, input_ptr, str_size);
      }
    }
  }
};

}  // namespace

std::unique_ptr<column> repeat_strings(strings_column_view const& input,
                                       size_type repeat_times,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  auto const strings_count = input.size();
  if (strings_count == 0) { return make_empty_column(type_id::STRING); }

  if (repeat_times <= 0) {
    // If the number of repetitions is not positive, each row of the output strings column will be
    // either an empty string (if the input row is not null), or a null (if the input row is null).
    return generate_empty_output(input, strings_count, stream, mr);
  }

  // If `repeat_times == 1`, just make a copy of the input.
  if (repeat_times == 1) { return std::make_unique<column>(input.parent(), stream, mr); }

  auto const strings_dv_ptr = column_device_view::create(input.parent(), stream);
  auto const fn = compute_size_and_repeat_fn{*strings_dv_ptr, repeat_times, input.has_nulls()};

  auto [offsets_column, chars] =
    make_strings_children(fn, strings_count * repeat_times, strings_count, stream, mr);
  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             chars.release(),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

namespace {
/**
 * @brief Functor to compute string sizes and repeat the input strings, each string is repeated by a
 * separate number of times.
 */
template <class Iterator>
struct compute_sizes_and_repeat_fn {
  column_device_view const strings_dv;
  column_device_view const repeat_times_dv;
  Iterator const repeat_times_iter;
  bool const strings_has_nulls;
  bool const rtimes_has_nulls;
  size_type* d_sizes;
  char* d_chars;
  cudf::detail::input_offsetalator d_offsets;

  /**
   * @brief Called by make_strings_children to build output
   *
   * @param idx Row index
   * @param d_sizes Return output size here in 1st call (d_chars==nullptr)
   * @param d_chars Write output here in 2nd call
   * @param d_offsets Offsets to address output row within d_chars
   */
  __device__ void operator()(size_type idx) const noexcept
  {
    auto const string_is_valid = !strings_has_nulls || strings_dv.is_valid_nocheck(idx);
    auto const rtimes_is_valid = !rtimes_has_nulls || repeat_times_dv.is_valid_nocheck(idx);

    // Any null input (either string or repeat_times value) will result in a null output.
    auto const is_valid = string_is_valid && rtimes_is_valid;
    if (!is_valid) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }

    auto repeat_times = repeat_times_iter[idx];
    auto const d_str  = strings_dv.element<string_view>(idx);

    if (!d_chars) {
      // repeat_times could be negative
      d_sizes[idx] = cuda::std::max(repeat_times, 0) * d_str.size_bytes();
    } else {
      auto output_ptr = d_chars + d_offsets[idx];
      while (repeat_times-- > 0) {
        output_ptr = copy_and_increment(output_ptr, d_str.data(), d_str.size_bytes());
      }
    }
  }
};

}  // namespace

std::unique_ptr<column> repeat_strings(strings_column_view const& input,
                                       column_view const& repeat_times,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.size() == repeat_times.size(), "The input columns must have the same size.");
  CUDF_EXPECTS(cudf::is_index_type(repeat_times.type()),
               "repeat_strings expects an integer type for the `repeat_times` input column.");

  auto const strings_count = input.size();
  if (strings_count == 0) { return make_empty_column(type_id::STRING); }

  auto const strings_dv_ptr      = column_device_view::create(input.parent(), stream);
  auto const repeat_times_dv_ptr = column_device_view::create(repeat_times, stream);
  auto const repeat_times_iter =
    cudf::detail::indexalator_factory::make_input_iterator(repeat_times);
  auto const fn =
    compute_sizes_and_repeat_fn<decltype(repeat_times_iter)>{*strings_dv_ptr,
                                                             *repeat_times_dv_ptr,
                                                             repeat_times_iter,
                                                             input.has_nulls(),
                                                             repeat_times.has_nulls()};

  auto [offsets_column, chars] = make_strings_children(fn, strings_count, stream, mr);

  // We generate new bitmask by AND of the two input columns' bitmasks.
  // Note that if either of the input columns are nullable, the output column will also be nullable
  // but may not have nulls.
  auto [null_mask, null_count] =
    cudf::detail::bitmask_and(table_view{{input.parent(), repeat_times}}, stream, mr);

  return make_strings_column(
    strings_count, std::move(offsets_column), chars.release(), null_count, std::move(null_mask));
}
}  // namespace detail

std::unique_ptr<string_scalar> repeat_string(string_scalar const& input,
                                             size_type repeat_times,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::repeat_string(input, repeat_times, stream, mr);
}

std::unique_ptr<column> repeat_strings(strings_column_view const& input,
                                       size_type repeat_times,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::repeat_strings(input, repeat_times, stream, mr);
}

std::unique_ptr<column> repeat_strings(strings_column_view const& input,
                                       column_view const& repeat_times,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::repeat_strings(input, repeat_times, stream, mr);
}

}  // namespace strings
}  // namespace cudf
