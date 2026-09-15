/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/detail/convert/string_to_float.cuh>
#include <cudf/strings/detail/converters.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/transform.h>

#include <cmath>
#include <limits>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Converts strings column entries into floats.
 *
 * Used by the dispatch method to convert to different float types.
 */
template <typename FloatType>
struct string_to_float_fn {
  column_device_view const strings_column;  // strings to convert

  __device__ FloatType operator()(size_type idx)
  {
    if (strings_column.is_null(idx)) return static_cast<FloatType>(0);
    // The cast to FloatType will create predictable results for floats that are larger than the
    // FloatType can hold
    return static_cast<FloatType>(stod(strings_column.element<string_view>(idx)));
  }
};

/**
 * @brief The dispatch functions for converting strings to floats.
 *
 * The output_column is expected to be one of the float types only.
 */
struct dispatch_to_floats_fn {
  template <typename FloatType>
  void operator()(column_device_view const& strings_column,
                  mutable_column_view& output_column,
                  cuda::stream_ref stream) const
    requires(std::is_floating_point_v<FloatType>)
  {
    auto d_results = output_column.data<FloatType>();
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{strings_column.size()},
                      d_results,
                      string_to_float_fn<FloatType>{strings_column});
  }
  // non-integral types throw an exception
  template <typename T>
  void operator()(column_device_view const&, mutable_column_view&, cuda::stream_ref) const
    requires(not std::is_floating_point_v<T>)
  {
    CUDF_FAIL("Output for to_floats must be a float type.");
  }
};

}  // namespace

// This will convert a strings column into any float column type.
std::unique_ptr<column> to_floats(strings_column_view const& input,
                                  data_type output_type,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  size_type strings_count = input.size();
  if (strings_count == 0) {
    return make_numeric_column(
      output_type, 0, mask_state::UNALLOCATED, stream, cudf::get_current_device_resource_ref());
  }
  auto strings_column = column_device_view::create(input.parent(), stream);
  auto d_strings      = *strings_column;
  // create float output column copying the strings null-mask
  auto results      = make_numeric_column(output_type,
                                     strings_count,
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  auto results_view = results->mutable_view();
  // fill output column with floats
  type_dispatcher(output_type, dispatch_to_floats_fn{}, d_strings, results_view, stream);
  results->set_null_count(input.null_count());
  return results;
}

}  // namespace detail

// external API

std::unique_ptr<column> to_floats(strings_column_view const& input,
                                  data_type output_type,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_floats(input, output_type, stream, mr);
}

namespace detail {
namespace {
/**
 * @brief Code logic for converting float value into a string.
 *
 * The floating point components are dissected and used to fill an
 * existing output char array.
 */
struct ftos_converter {
  // significant digits is independent of scientific notation range
  // digits more than this may require using long values instead of ints
  static constexpr unsigned int significant_digits = 10;
  // maximum power-of-10 that will fit in 32-bits
  static constexpr unsigned int nine_digits = 1000000000;  // 1x10^9
  // Range of numbers here is for normalizing the value.
  // If the value is above or below the following limits, the output is converted to
  // scientific notation in order to show (at most) the number of significant digits.
  static constexpr double upper_limit = 1000000000;  // max is 1x10^9
  static constexpr double lower_limit = 0.0001;      // printf uses scientific notation below this
  // Tables for doing normalization: converting to exponent form
  // IEEE double float has maximum exponent of 305 so these should cover everything
  double const upper10[9]  = {10, 100, 10000, 1e8, 1e16, 1e32, 1e64, 1e128, 1e256};
  double const lower10[9]  = {.1, .01, .0001, 1e-8, 1e-16, 1e-32, 1e-64, 1e-128, 1e-256};
  double const blower10[9] = {1.0, .1, .001, 1e-7, 1e-15, 1e-31, 1e-63, 1e-127, 1e-255};

  // utility for quickly converting known integer range to character array
  __device__ char* int2str(int value, char* output)
  {
    if (value == 0) {
      *output++ = '0';
      return output;
    }
    char buffer[significant_digits];  // should be big-enough for significant digits
    char* ptr = buffer;
    while (value > 0) {
      *ptr++ = (char)('0' + (value % 10));
      value /= 10;
    }
    while (ptr != buffer)
      *output++ = *--ptr;  // 54321 -> 12345
    return output;
  }

  /**
   * @brief Dissect a float value into integer, decimal, and exponent components.
   *
   * @return The number of decimal places.
   */
  __device__ int dissect_value(double value,
                               unsigned int& integer,
                               unsigned int& decimal,
                               int& exp10)
  {
    int decimal_places = significant_digits - 1;
    // normalize step puts value between lower-limit and upper-limit
    // by adjusting the exponent up or down
    exp10 = 0;
    if (value > upper_limit) {
      int fx = 256;
      for (int idx = 8; idx >= 0; --idx) {
        if (value >= upper10[idx]) {
          value *= lower10[idx];
          exp10 += fx;
        }
        fx = fx >> 1;
      }
    } else if ((value > 0.0) && (value < lower_limit)) {
      int fx = 256;
      for (int idx = 8; idx >= 0; --idx) {
        if (value < blower10[idx]) {
          value *= upper10[idx];
          exp10 -= fx;
        }
        fx = fx >> 1;
      }
    }
    //
    unsigned int max_digits = nine_digits;
    integer                 = (unsigned int)value;
    for (unsigned int i = integer; i >= 10; i /= 10) {
      --decimal_places;
      max_digits /= 10;
    }
    double remainder = (value - (double)integer) * (double)max_digits;
    decimal          = (unsigned int)remainder;
    remainder -= (double)decimal;
    decimal += (unsigned int)(2.0 * remainder);
    if (decimal >= max_digits) {
      decimal = 0;
      ++integer;
      if (exp10 && (integer >= 10)) {
        ++exp10;
        integer = 1;
      }
    }
    //
    while ((decimal % 10) == 0 && (decimal_places > 0)) {
      decimal /= 10;
      --decimal_places;
    }
    return decimal_places;
  }

  /**
   * @brief Main kernel method for converting float value to char output array.
   *
   * Output need not be more than (significant_digits + 7) bytes:
   * 7 = 1 sign, 1 decimal point, 1 exponent ('e'), 1 exponent-sign, 3 digits for exponent
   *
   * @param value Float value to convert.
   * @param output Memory to write output characters.
   * @return Number of bytes written.
   */
  __device__ int float_to_string(double value, char* output)
  {
    // check for valid value
    if (std::isnan(value)) {
      memcpy(output, "NaN", 3);
      return 3;
    }
    bool bneg = false;
    if (signbit(value)) {  // handles -0.0 too
      value = -value;
      bneg  = true;
    }
    if (std::isinf(value)) {
      if (bneg)
        memcpy(output, "-Inf", 4);
      else
        memcpy(output, "Inf", 3);
      return bneg ? 4 : 3;
    }

    // dissect value into components
    unsigned int integer = 0, decimal = 0;
    int exp10          = 0;
    int decimal_places = dissect_value(value, integer, decimal, exp10);
    //
    // now build the string from the
    // components: sign, integer, decimal, exp10, decimal_places
    //
    // sign
    char* ptr = output;
    if (bneg) *ptr++ = '-';
    // integer
    ptr = int2str(integer, ptr);
    // decimal
    *ptr++ = '.';
    if (decimal_places) {
      char buffer[10];
      char* pb = buffer;
      while (decimal_places--) {
        *pb++ = (char)('0' + (decimal % 10));
        decimal /= 10;
      }
      while (pb != buffer)  // reverses the digits
        *ptr++ = *--pb;     // e.g. 54321 -> 12345
    } else
      *ptr++ = '0';  // always include at least .0
    // exponent
    if (exp10) {
      *ptr++ = 'e';
      if (exp10 < 0) {
        *ptr++ = '-';
        exp10  = -exp10;
      } else
        *ptr++ = '+';
      if (exp10 < 10) *ptr++ = '0';  // extra zero-pad
      ptr = int2str(exp10, ptr);
    }
    // done
    return (int)(ptr - output);  // number of bytes written
  }

  /**
   * @brief Compute how man bytes are needed to hold the output string.
   *
   * @param value Float value to convert.
   * @return Number of bytes required.
   */
  __device__ int compute_ftos_size(double value)
  {
    if (std::isnan(value)) return 3;  // NaN
    bool bneg = false;
    if (signbit(value)) {  // handles -0.0 too
      value = -value;
      bneg  = true;
    }
    if (std::isinf(value)) return 3 + (int)bneg;  // Inf

    // dissect float into parts
    unsigned int integer = 0, decimal = 0;
    int exp10          = 0;
    int decimal_places = dissect_value(value, integer, decimal, exp10);
    // now count up the components
    // sign
    int count = (int)bneg;
    // integer
    count += (int)(integer == 0);
    while (integer > 0) {
      integer /= 10;
      ++count;
    }  // log10(integer)
    // decimal
    ++count;  // decimal point
    if (decimal_places)
      count += decimal_places;
    else
      ++count;  // always include .0
    // exponent
    if (exp10) {
      count += 2;  // 'e±'
      if (exp10 < 0) exp10 = -exp10;
      count += (int)(exp10 < 10);  // padding
      while (exp10 > 0) {
        exp10 /= 10;
        ++count;
      }  // log10(exp10)
    }
    return count;
  }
};

template <typename FloatType>
struct from_floats_fn {
  column_device_view d_floats;
  size_type* d_sizes;
  char* d_chars;
  cudf::detail::input_offsetalator d_offsets;

  __device__ size_type compute_output_size(FloatType value)
  {
    ftos_converter fts;
    return static_cast<size_type>(fts.compute_ftos_size(static_cast<double>(value)));
  }

  __device__ void float_to_string(size_type idx)
  {
    FloatType value = d_floats.element<FloatType>(idx);
    ftos_converter fts;
    fts.float_to_string(static_cast<double>(value), d_chars + d_offsets[idx]);
  }

  __device__ void operator()(size_type idx)
  {
    if (d_floats.is_null(idx)) {
      if (d_chars == nullptr) { d_sizes[idx] = 0; }
      return;
    }
    if (d_chars != nullptr) {
      float_to_string(idx);
    } else {
      d_sizes[idx] = compute_output_size(d_floats.element<FloatType>(idx));
    }
  }
};

/**
 * @brief This dispatch method is for converting floats into strings.
 *
 * The template function declaration ensures only float types are allowed.
 */
struct dispatch_from_floats_fn {
  template <typename FloatType>
  std::unique_ptr<column> operator()(column_view const& floats,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr) const
    requires(std::is_floating_point_v<FloatType>)
  {
    size_type strings_count = floats.size();
    auto column             = column_device_view::create(floats, stream);
    auto d_column           = *column;

    // copy the null mask
    rmm::device_buffer null_mask = cudf::detail::copy_bitmask(floats, stream, mr);

    auto [offsets, chars] =
      make_strings_children(from_floats_fn<FloatType>{d_column}, strings_count, stream, mr);

    return make_strings_column(strings_count,
                               std::move(offsets),
                               chars.release(),
                               floats.null_count(),
                               std::move(null_mask));
  }

  // non-float types throw an exception
  template <typename T>
  std::unique_ptr<column> operator()(column_view const&,
                                     cuda::stream_ref,
                                     rmm::device_async_resource_ref) const
    requires(not std::is_floating_point_v<T>)
  {
    CUDF_FAIL("Values for from_floats function must be a float type.");
  }
};

}  // namespace

// This will convert all float column types into a strings column.
std::unique_ptr<column> from_floats(column_view const& floats,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  size_type strings_count = floats.size();
  if (strings_count == 0) return make_empty_column(type_id::STRING);

  return type_dispatcher(floats.type(), dispatch_from_floats_fn{}, floats, stream, mr);
}

}  // namespace detail

// external API
std::unique_ptr<column> from_floats(column_view const& floats,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::from_floats(floats, stream, mr);
}

namespace detail {
std::unique_ptr<column> is_float(strings_column_view const& input,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  auto strings_column = column_device_view::create(input.parent(), stream);
  auto d_column       = *strings_column;
  // create output column
  auto results   = make_numeric_column(data_type{type_id::BOOL8},
                                     input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  auto d_results = results->mutable_view().data<bool>();
  // check strings for valid float chars
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{input.size()},
                    d_results,
                    [d_column] __device__(size_type idx) {
                      if (d_column.is_null(idx)) return false;
                      return is_float(d_column.element<string_view>(idx));
                    });
  results->set_null_count(input.null_count());
  return results;
}

}  // namespace detail

// external API
std::unique_ptr<column> is_float(strings_column_view const& input,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_float(input, stream, mr);
}

}  // namespace strings
}  // namespace cudf
