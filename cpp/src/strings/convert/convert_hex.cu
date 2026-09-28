/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/stream>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {
/**
 * @brief Converts hex strings into an integers.
 *
 * Used by the dispatch method to convert to different integer types.
 */
template <typename IntegerType>
struct hex_to_integer_fn {
  column_device_view const strings_column;

  /**
   * @brief Converts a single hex string into an integer.
   *
   * Non-hexadecimal characters are ignored.
   * This means it can handle "0x01A23" and "1a23".
   *
   * Overflow of the int64 type is not detected.
   */
  __device__ int64_t string_to_integer(string_view const& d_str)
  {
    int64_t result = 0, base = 1;
    char const* str = d_str.data();
    size_type index = d_str.size_bytes();
    while (index-- > 0) {
      char ch = str[index];
      if (ch >= '0' && ch <= '9') {
        result += static_cast<int64_t>(ch - 48) * base;
        base *= 16;
      } else if (ch >= 'A' && ch <= 'F') {
        result += static_cast<int64_t>(ch - 55) * base;
        base *= 16;
      } else if (ch >= 'a' && ch <= 'f') {
        result += static_cast<int64_t>(ch - 87) * base;
        base *= 16;
      }
    }
    return result;
  }

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
struct dispatch_hex_to_integers_fn {
  template <typename IntegerType>
  void operator()(column_device_view const& strings_column,
                  mutable_column_view& output_column,
                  cuda::stream_ref stream) const
    requires(cudf::is_integral_not_bool<IntegerType>())
  {
    auto d_results = output_column.data<IntegerType>();
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{strings_column.size()},
                      d_results,
                      hex_to_integer_fn<IntegerType>{strings_column});
  }
  // non-integer types throw an exception
  template <typename T, typename... Args>
  void operator()(Args&&...) const
    requires(not cudf::is_integral_not_bool<T>())
  {
    CUDF_FAIL("Output for hex_to_integers must be an integer type.");
  }
};

/**
 * @brief Functor to convert integers to hexadecimal strings
 *
 * @tparam IntegerType The specific integer type to convert from.
 */
template <typename IntegerType>
struct integer_to_hex_fn {
  column_device_view const d_column;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  __device__ void byte_to_hex(uint8_t byte, char* hex)
  {
    hex[0] = [&] {
      if (byte < 16) { return '0'; }
      uint8_t const nibble = byte / 16;

      byte = byte - (nibble * 16);
      return static_cast<char>(nibble < 10 ? '0' + nibble : 'A' + (nibble - 10));
    }();
    hex[1] = byte < 10 ? '0' + byte : 'A' + (byte - 10);
  }

  __device__ void operator()(size_type idx)
  {
    if (d_column.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }

    // Reinterpret an integer value as a little-endian byte sequence.
    // For example, 123456 becomes 0x40E2'0100
    auto const value = d_column.element<IntegerType>(idx);
    auto value_bytes = reinterpret_cast<uint8_t const*>(&value);

    // compute the number of output bytes
    int bytes      = sizeof(IntegerType);
    int byte_index = sizeof(IntegerType);
    while ((--byte_index > 0) && (value_bytes[byte_index] & 0xFF) == 0) {
      --bytes;
    }

    // create output
    byte_index = bytes - 1;
    if (d_chars) {
      auto d_buffer = d_chars + d_offsets[idx];
      while (byte_index >= 0) {
        byte_to_hex(value_bytes[byte_index], d_buffer);
        d_buffer += 2;
        --byte_index;
      }
    } else {
      d_sizes[idx] = static_cast<size_type>(bytes) * 2;  // 2 hex characters per byte
    }
  }
};

struct dispatch_integers_to_hex_fn {
  template <typename IntegerType>
  std::unique_ptr<column> operator()(column_view const& input,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr) const
    requires(cudf::is_integral_not_bool<IntegerType>())
  {
    auto const d_column = column_device_view::create(input, stream);

    auto [offsets_column, chars] =
      make_strings_children(integer_to_hex_fn<IntegerType>{*d_column}, input.size(), stream, mr);

    return make_strings_column(input.size(),
                               std::move(offsets_column),
                               chars.release(),
                               input.null_count(),
                               cudf::detail::copy_bitmask(input, stream, mr));
  }
  // non-integer types throw an exception
  template <typename T, typename... Args>
  std::unique_ptr<column> operator()(Args...) const
    requires(not cudf::is_integral_not_bool<T>())
  {
    CUDF_FAIL("integers_to_hex only supports integer type columns");
  }
};

}  // namespace

// This will convert a strings column into any integer column type.
std::unique_ptr<column> hex_to_integers(strings_column_view const& strings,
                                        data_type output_type,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  size_type strings_count = strings.size();
  if (strings_count == 0) return make_empty_column(output_type);
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  // create integer output column copying the strings null-mask
  auto results      = make_numeric_column(output_type,
                                     strings_count,
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto results_view = results->mutable_view();
  // fill output column with integers
  type_dispatcher(output_type, dispatch_hex_to_integers_fn{}, d_strings, results_view, stream);
  results->set_null_count(strings.null_count());
  return results;
}

std::unique_ptr<column> is_hex(strings_column_view const& strings,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_column       = *strings_column;
  // create output column
  auto results   = make_numeric_column(data_type{type_id::BOOL8},
                                     strings.size(),
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto d_results = results->mutable_view().data<bool>();
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{strings.size()},
                    d_results,
                    [d_column] __device__(size_type idx) {
                      if (d_column.is_null(idx)) return false;
                      auto const d_str = d_column.element<string_view>(idx);
                      if (d_str.empty()) return false;
                      auto const starts_with_0x = [](auto const& sv) {
                        return sv.length() > 1 && (sv.substr(0, 2) == string_view("0x", 2) ||
                                                   sv.substr(0, 2) == string_view("0X", 2));
                      };
                      auto begin = d_str.data() + (starts_with_0x(d_str) ? 2 : 0);
                      auto end   = begin + d_str.size_bytes();
                      return (cuda::std::distance(begin, end) > 0) &&
                             thrust::all_of(thrust::seq, begin, end, [] __device__(auto chr) {
                               return (chr >= '0' && chr <= '9') || (chr >= 'A' && chr <= 'F') ||
                                      (chr >= 'a' && chr <= 'f');
                             });
                    });
  results->set_null_count(strings.null_count());
  return results;
}

std::unique_ptr<column> integers_to_hex(column_view const& input,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return cudf::make_empty_column(type_id::STRING); }
  return type_dispatcher(input.type(), dispatch_integers_to_hex_fn{}, input, stream, mr);
}

}  // namespace detail

// external API
std::unique_ptr<column> hex_to_integers(strings_column_view const& strings,
                                        data_type output_type,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::hex_to_integers(strings, output_type, stream, mr);
}

std::unique_ptr<column> is_hex(strings_column_view const& strings,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_hex(strings, stream, mr);
}

std::unique_ptr<column> integers_to_hex(column_view const& input,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::integers_to_hex(input, stream, mr);
}

}  // namespace strings
}  // namespace cudf
