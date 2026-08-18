/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/convert/convert_fixed_point.hpp>
#include <cudf/strings/detail/convert/fixed_point.cuh>
#include <cudf/strings/detail/convert/fixed_point_to_string.cuh>
#include <cudf/strings/detail/converters.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/climits>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/stream>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Converts strings into an integers and records decimal places.
 *
 * The conversion uses the provided scale to build the resulting
 * integer. This can prevent overflow for strings with many digits.
 */
template <typename DecimalType>
struct string_to_decimal_fn {
  column_device_view const d_strings;
  int32_t const scale;

  string_to_decimal_fn(column_device_view const& d_strings, int32_t scale)
    : d_strings(d_strings), scale(scale)
  {
  }

  __device__ DecimalType operator()(size_type idx) const
  {
    if (d_strings.is_null(idx)) { return 0; }
    auto const d_str = d_strings.element<string_view>(idx);
    if (d_str.empty()) { return 0; }

    auto iter           = d_str.data();
    auto const iter_end = d_str.data() + d_str.size_bytes();

    return parse_decimal<DecimalType>(iter, iter_end, scale);
  }
};

/**
 * @brief This only checks the string format for valid decimal characters.
 *
 * This follows closely the logic above but just ensures there are valid
 * characters for conversion and the integer component does not overflow.
 */
template <typename DecimalType>
struct string_to_decimal_check_fn {
  column_device_view const d_strings;
  int32_t const scale;

  string_to_decimal_check_fn(column_device_view const& d_strings, int32_t scale)
    : d_strings{d_strings}, scale{scale}
  {
  }

  __device__ bool operator()(size_type idx) const
  {
    if (d_strings.is_null(idx)) { return false; }
    auto const d_str = d_strings.element<string_view>(idx);
    if (d_str.empty()) { return false; }

    auto iter = d_str.data() + static_cast<int>((d_str.data()[0] == '-' || d_str.data()[0] == '+'));

    auto const iter_end = d_str.data() + d_str.size_bytes();

    using UnsignedDecimalType = cuda::std::make_unsigned_t<DecimalType>;
    auto [value, exp_offset]  = parse_integer<UnsignedDecimalType>(iter, iter_end);

    // only exponent notation is expected here
    if ((iter < iter_end) && (*iter != 'e' && *iter != 'E')) { return false; }
    ++iter;

    int32_t exp_ten = 0;  // check exponent overflow
    if (iter < iter_end) {
      auto exp_result = parse_exponent<true>(iter, iter_end);
      if (!exp_result) { return false; }
      exp_ten = exp_result.value();
    }
    exp_ten += exp_offset;

    // finally, check for overflow based on the exp_ten and scale values
    return (exp_ten < scale) or
           value <= static_cast<UnsignedDecimalType>(
                      cuda::std::numeric_limits<DecimalType>::max() /
                      static_cast<DecimalType>(exp10(static_cast<double>(exp_ten - scale))));
  }
};

/**
 * @brief The dispatch function for converting strings column to fixed-point column.
 */
struct dispatch_to_fixed_point_fn {
  template <typename T>
  std::unique_ptr<column> operator()(strings_column_view const& input,
                                     data_type output_type,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr) const
    requires(cudf::is_fixed_point<T>())
  {
    using DecimalType = device_storage_type_t<T>;

    auto const d_column = column_device_view::create(input.parent(), stream);

    // create output column
    auto results   = make_fixed_point_column(output_type,
                                           input.size(),
                                           cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                           input.null_count(),
                                           stream,
                                           mr);
    auto d_results = results->mutable_view().data<DecimalType>();

    // convert strings into decimal values
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{input.size()},
                      d_results,
                      string_to_decimal_fn<DecimalType>{*d_column, output_type.scale()});
    results->set_null_count(input.null_count());
    return results;
  }

  template <typename T>
  std::unique_ptr<column> operator()(strings_column_view const&,
                                     data_type,
                                     cuda::stream_ref,
                                     rmm::device_async_resource_ref) const
    requires(not cudf::is_fixed_point<T>())
  {
    CUDF_FAIL("Output for to_fixed_point must be a decimal type.");
  }
};

}  // namespace

// This will convert a strings column into any integer column type.
std::unique_ptr<column> to_fixed_point(strings_column_view const& input,
                                       data_type output_type,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) return make_empty_column(output_type);
  return type_dispatcher(output_type, dispatch_to_fixed_point_fn{}, input, output_type, stream, mr);
}

}  // namespace detail

// external API
std::unique_ptr<column> to_fixed_point(strings_column_view const& input,
                                       data_type output_type,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_fixed_point(input, output_type, stream, mr);
}

namespace detail {
namespace {
template <typename DecimalType>
struct from_fixed_point_fn {
  column_device_view d_decimals;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  /**
   * @brief Converts a decimal element into a string.
   *
   * The value is converted into base-10 digits [0-9]
   * plus the decimal point and a negative sign prefix.
   */
  __device__ void fixed_point_element_to_string(size_type idx)
  {
    auto const value = d_decimals.element<DecimalType>(idx);
    auto const scale = d_decimals.type().scale();
    char* d_buffer   = d_chars + d_offsets[idx];

    fixed_point_to_string(value, scale, d_buffer);
  }

  __device__ void operator()(size_type idx)
  {
    if (d_decimals.is_null(idx)) {
      if (d_chars == nullptr) { d_sizes[idx] = 0; }
      return;
    }
    if (d_chars != nullptr) {
      fixed_point_element_to_string(idx);
    } else {
      d_sizes[idx] =
        fixed_point_string_size(d_decimals.element<DecimalType>(idx), d_decimals.type().scale());
    }
  }
};

/**
 * @brief The dispatcher functor for converting fixed-point values into strings.
 */
struct dispatch_from_fixed_point_fn {
  template <typename T>
  std::unique_ptr<column> operator()(column_view const& input,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr) const
    requires(cudf::is_fixed_point<T>())
  {
    using DecimalType = device_storage_type_t<T>;  // underlying value type

    auto const d_column = column_device_view::create(input, stream);

    auto [offsets, chars] =
      make_strings_children(from_fixed_point_fn<DecimalType>{*d_column}, input.size(), stream, mr);

    return make_strings_column(input.size(),
                               std::move(offsets),
                               chars.release(),
                               input.null_count(),
                               cudf::detail::copy_bitmask(input, stream, mr));
  }

  template <typename T>
  std::unique_ptr<column> operator()(column_view const&,
                                     cuda::stream_ref,
                                     rmm::device_async_resource_ref) const
    requires(not cudf::is_fixed_point<T>())
  {
    CUDF_FAIL("Values for from_fixed_point function must be a decimal type.");
  }
};

}  // namespace

std::unique_ptr<column> from_fixed_point(column_view const& input,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) return make_empty_column(type_id::STRING);
  return type_dispatcher(input.type(), dispatch_from_fixed_point_fn{}, input, stream, mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> from_fixed_point(column_view const& input,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::from_fixed_point(input, stream, mr);
}

namespace detail {
namespace {

struct dispatch_is_fixed_point_fn {
  template <typename T>
  std::unique_ptr<column> operator()(strings_column_view const& input,
                                     data_type decimal_type,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr) const
    requires(cudf::is_fixed_point<T>())
  {
    using DecimalType = device_storage_type_t<T>;

    auto const d_column = column_device_view::create(input.parent(), stream);

    // create output column
    auto results   = make_numeric_column(data_type{type_id::BOOL8},
                                       input.size(),
                                       cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                       input.null_count(),
                                       stream,
                                       mr);
    auto d_results = results->mutable_view().data<bool>();

    // check strings for valid fixed-point chars
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{input.size()},
                      d_results,
                      string_to_decimal_check_fn<DecimalType>{*d_column, decimal_type.scale()});
    results->set_null_count(input.null_count());
    return results;
  }

  template <typename T>
  std::unique_ptr<column> operator()(strings_column_view const&,
                                     data_type,
                                     cuda::stream_ref,
                                     rmm::device_async_resource_ref) const
    requires(not cudf::is_fixed_point<T>())
  {
    CUDF_FAIL("is_fixed_point is expecting a decimal type");
  }
};

}  // namespace

std::unique_ptr<column> is_fixed_point(strings_column_view const& input,
                                       data_type decimal_type,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) return cudf::make_empty_column(type_id::BOOL8);
  return type_dispatcher(
    decimal_type, dispatch_is_fixed_point_fn{}, input, decimal_type, stream, mr);
}
}  // namespace detail

std::unique_ptr<column> is_fixed_point(strings_column_view const& input,
                                       data_type decimal_type,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_fixed_point(input, decimal_type, stream, mr);
}

}  // namespace strings
}  // namespace cudf
