/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/detail/convert/int_to_string.cuh>
#include <cudf/strings/detail/convert/string_to_int.cuh>
#include <cudf/strings/detail/converters.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {

namespace detail {
namespace {

/**
 * @brief This only checks if a string is a valid integer within the bounds of its storage type.
 */
template <typename IntegerType>
struct string_to_integer_check_fn {
  __device__ bool operator()(cuda::std::pair<string_view, bool> const& p) const
  {
    if (!p.second || p.first.empty()) { return false; }

    auto const d_str = p.first.data();
    if (d_str[0] == '-' && cuda::std::is_unsigned_v<IntegerType>) { return false; }

    auto iter           = d_str + static_cast<int>((d_str[0] == '-' || d_str[0] == '+'));
    auto const iter_end = d_str + p.first.size_bytes();
    if (iter == iter_end) { return false; }

    auto const sign      = d_str[0] == '-' ? IntegerType{-1} : IntegerType{1};
    auto const bound_val = sign > 0 ? cuda::std::numeric_limits<IntegerType>::max()
                                    : cuda::std::numeric_limits<IntegerType>::min();

    IntegerType value = 0;      // parse the string to integer and check for overflow along the way
    while (iter != iter_end) {  // check all bytes for valid characters
      auto const chr = *iter++;
      // Check for valid character
      if (chr < '0' || chr > '9') { return false; }

      // Check for underflow and overflow:
      auto const digit       = static_cast<IntegerType>(chr - '0');
      auto const bound_check = (bound_val - sign * digit) / IntegerType{10} * sign;
      if (value > bound_check) return false;
      value = value * IntegerType{10} + digit;
    }

    return true;
  }
};

/**
 * @brief Returns `true` if all characters in the string
 * are valid for conversion to an integer.
 *
 * Valid characters are in [-+0-9]. The sign character (+/-)
 * is optional but if present must be the first character.
 * An empty string returns `false`.
 * No bounds checking is performed to verify if the integer will fit
 * within a specific integer type.
 *
 * @param d_str String to check.
 * @return true if string has valid integer characters
 */
inline __device__ bool is_integer(string_view const& d_str)
{
  if (d_str.empty()) return false;
  auto const end = d_str.end();
  auto begin     = d_str.begin();
  if (*begin == '+' || *begin == '-') ++begin;
  return (begin < end) && thrust::all_of(thrust::seq, begin, end, [] __device__(auto chr) {
           return chr >= '0' && chr <= '9';
         });
}

/**
 * @brief The dispatch functions for checking if strings are valid integers.
 */
struct dispatch_is_integer_fn {
  template <typename T>
  std::unique_ptr<column> operator()(strings_column_view const& input,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr) const
    requires(cudf::is_integral_not_bool<T>())
  {
    auto const d_column = column_device_view::create(input.parent(), stream);
    auto results        = make_numeric_column(data_type{type_id::BOOL8},
                                       input.size(),
                                       cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                       input.null_count(),
                                       stream,
                                       mr);

    auto d_results = results->mutable_view().data<bool>();
    if (input.has_nulls()) {
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        d_column->pair_begin<string_view, true>(),
                        d_column->pair_end<string_view, true>(),
                        d_results,
                        string_to_integer_check_fn<T>{});
    } else {
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        d_column->pair_begin<string_view, false>(),
                        d_column->pair_end<string_view, false>(),
                        d_results,
                        string_to_integer_check_fn<T>{});
    }

    // Calling mutable_view() on a column invalidates it's null count so we need to set it back
    results->set_null_count(input.null_count());

    return results;
  }

  template <typename T>
  std::unique_ptr<column> operator()(strings_column_view const&,
                                     cuda::stream_ref,
                                     rmm::device_async_resource_ref) const
    requires(not cudf::is_integral_not_bool<T>())
  {
    CUDF_FAIL("is_integer is expecting an integer type");
  }
};

}  // namespace

std::unique_ptr<column> is_integer(strings_column_view const& input,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  auto const d_column = column_device_view::create(input.parent(), stream);
  auto results        = make_numeric_column(data_type{type_id::BOOL8},
                                     input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);

  auto d_results = results->mutable_view().data<bool>();
  if (input.has_nulls()) {
    thrust::transform(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      d_column->pair_begin<string_view, true>(),
      d_column->pair_end<string_view, true>(),
      d_results,
      [] __device__(auto const& p) { return p.second ? is_integer(p.first) : false; });
  } else {
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      d_column->pair_begin<string_view, false>(),
                      d_column->pair_end<string_view, false>(),
                      d_results,
                      [] __device__(auto const& p) { return is_integer(p.first); });
  }

  // Calling mutable_view() on a column invalidates it's null count so we need to set it back
  results->set_null_count(input.null_count());

  return results;
}

std::unique_ptr<column> is_integer(strings_column_view const& input,
                                   data_type int_type,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return cudf::make_empty_column(type_id::BOOL8); }
  return type_dispatcher(int_type, dispatch_is_integer_fn{}, input, stream, mr);
}

}  // namespace detail

// external APIs
std::unique_ptr<column> is_integer(strings_column_view const& input,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_integer(input, stream, mr);
}

std::unique_ptr<column> is_integer(strings_column_view const& input,
                                   data_type int_type,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_integer(input, int_type, stream, mr);
}

namespace detail {
namespace {
/**
 * @brief Converts strings into an integers.
 *
 * Used by the dispatch method to convert to different integer types.
 */
template <typename IntegerType>
struct string_to_integer_fn {
  column_device_view const strings_column;  // strings to convert

  __device__ IntegerType operator()(size_type idx)
  {
    if (strings_column.is_null(idx)) return static_cast<IntegerType>(0);
    // the cast to IntegerType will create predictable results
    // for integers that are larger than the IntegerType can hold
    return static_cast<IntegerType>(string_to_integer(strings_column.element<string_view>(idx)));
  }
};

/**
 * @brief The dispatch functions for converting strings to integers.
 *
 * The output_column is expected to be one of the integer types only.
 */
struct dispatch_to_integers_fn {
  template <typename IntegerType>
  void operator()(column_device_view const& strings_column,
                  mutable_column_view& output_column,
                  cuda::stream_ref stream) const
    requires(cudf::is_integral_not_bool<IntegerType>())
  {
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{strings_column.size()},
                      output_column.data<IntegerType>(),
                      string_to_integer_fn<IntegerType>{strings_column});
  }
  // non-integer types throw an exception
  template <typename T>
  void operator()(column_device_view const&, mutable_column_view&, cuda::stream_ref) const
    requires(not cudf::is_integral_not_bool<T>())
  {
    CUDF_FAIL("Output for to_integers must be an integer type.");
  }
};

}  // namespace

// This will convert a strings column into any integer column type.
std::unique_ptr<column> to_integers(strings_column_view const& input,
                                    data_type output_type,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  size_type strings_count = input.size();
  if (strings_count == 0) {
    return make_numeric_column(output_type, 0, mask_state::UNALLOCATED, stream, mr);
  }

  // Create integer output column copying the strings null-mask
  auto results = make_numeric_column(output_type,
                                     strings_count,
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  // Fill output column with integers
  auto const strings_dev_view = column_device_view::create(input.parent(), stream);
  auto results_view           = results->mutable_view();
  type_dispatcher(output_type, dispatch_to_integers_fn{}, *strings_dev_view, results_view, stream);

  // Calling mutable_view() on a column invalidates it's null count so we need to set it back
  results->set_null_count(input.null_count());

  return results;
}

}  // namespace detail

// external API
std::unique_ptr<column> to_integers(strings_column_view const& input,
                                    data_type output_type,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_integers(input, output_type, stream, mr);
}

namespace detail {
namespace {
template <typename IntegerType>
struct from_integers_fn {
  column_device_view d_integers;
  size_type* d_sizes;
  char* d_chars;
  cudf::detail::input_offsetalator d_offsets;

  /**
   * @brief Converts an integer element into a string.
   *
   * The integer is converted into base-10 using only characters [0-9].
   * No formatting is done for the string other than prepending the '-'
   * character for negative values.
   */
  __device__ void integer_element_to_string(size_type idx)
  {
    IntegerType value = d_integers.element<IntegerType>(idx);
    char* d_buffer    = d_chars + d_offsets[idx];
    integer_to_string(value, d_buffer);
  }

  __device__ void operator()(size_type idx)
  {
    if (d_integers.is_null(idx)) {
      if (d_chars == nullptr) { d_sizes[idx] = 0; }
      return;
    }
    if (d_chars != nullptr) {
      integer_element_to_string(idx);
    } else {
      d_sizes[idx] = count_digits(d_integers.element<IntegerType>(idx));
    }
  }
};

/**
 * @brief This dispatch method is for converting integers into strings.
 * The template function declaration ensures only integer types are used.
 */
struct dispatch_from_integers_fn {
  template <typename IntegerType>
  std::unique_ptr<column> operator()(column_view const& integers,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr) const
    requires(cudf::is_integral_not_bool<IntegerType>())
  {
    size_type strings_count = integers.size();
    auto column             = column_device_view::create(integers, stream);
    auto d_column           = *column;

    // copy the null mask
    rmm::device_buffer null_mask = cudf::detail::copy_bitmask(integers, stream, mr);

    auto [offsets, chars] =
      make_strings_children(from_integers_fn<IntegerType>{d_column}, strings_count, stream, mr);

    return make_strings_column(strings_count,
                               std::move(offsets),
                               chars.release(),
                               integers.null_count(),
                               std::move(null_mask));
  }

  // non-integer types throw an exception
  template <typename T>
  std::unique_ptr<column> operator()(column_view const&,
                                     cuda::stream_ref,
                                     rmm::device_async_resource_ref) const
    requires(not cudf::is_integral_not_bool<T>())
  {
    CUDF_FAIL("Values for from_integers function must be an integer type.");
  }
};
}  // namespace

// This will convert all integer column types into a strings column.
std::unique_ptr<column> from_integers(column_view const& integers,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  size_type strings_count = integers.size();
  if (strings_count == 0) return make_empty_column(type_id::STRING);

  return type_dispatcher(integers.type(), dispatch_from_integers_fn{}, integers, stream, mr);
}

}  // namespace detail

// external API
std::unique_ptr<column> from_integers(column_view const& integers,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::from_integers(integers, stream, mr);
}

}  // namespace strings
}  // namespace cudf
