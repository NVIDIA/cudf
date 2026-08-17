/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/convert/int_cast.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/functional>
#include <cuda/stream>
#include <thrust/for_each.h>
#include <thrust/transform_reduce.h>

namespace cudf {
namespace strings {
namespace detail {

namespace {

struct dispatch_set_int_fn {
  template <typename T>
    requires(cudf::is_integral_not_bool<T>())
  __device__ void operator()(mutable_column_device_view& d_results,
                             size_type idx,
                             uint64_t value) const
  {
    d_results.element<T>(idx) = static_cast<T>(value);
  }
  template <typename T>
    requires(not cudf::is_integral_not_bool<T>())
  __device__ void operator()(mutable_column_device_view&, size_type, uint64_t) const
  {
    CUDF_UNREACHABLE("invalid type");
  }
};

struct cast_to_integer_fn {
  column_device_view const d_strings;
  mutable_column_device_view d_results;
  endian swap;
  size_type output_type_size;

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) { return; }
    auto const d_str = d_strings.element<string_view>(idx);
    auto const size  = cuda::std::min(d_str.size_bytes(), output_type_size);

    auto value = uint64_t{0};
    auto data  = reinterpret_cast<u_char const*>(d_str.data());
    if (swap == endian::LITTLE) {
      for (size_type i = 0; i < size; i++) {
        value = (value << CHAR_BIT) | data[i];
      }
    } else {
      memcpy(&value, data, size);
    }
    type_dispatcher(d_results.type(), dispatch_set_int_fn{}, d_results, idx, value);
  }
};
}  // namespace

std::unique_ptr<column> cast_to_integer(strings_column_view const& input,
                                        data_type output_type,
                                        endian swap,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(cudf::is_integral_not_bool(output_type),
               "Output type must be an integer type",
               cudf::data_type_error);

  auto results   = make_numeric_column(output_type,
                                     input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  auto d_strings = column_device_view::create(input.parent(), stream);
  auto d_results = mutable_column_device_view::create(*results, stream);

  auto const type_size = static_cast<size_type>(cudf::size_of(output_type));
  thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<size_type>{0},
                     input.size(),
                     cast_to_integer_fn{*d_strings, *d_results, swap, type_size});

  return results;
}

}  // namespace detail

// external API
std::unique_ptr<column> cast_to_integer(strings_column_view const& input,
                                        data_type output_type,
                                        endian swap,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::cast_to_integer(input, output_type, swap, stream, mr);
}

namespace detail {

struct dispatch_get_int_fn {
  template <typename T>
    requires(cudf::is_integral_not_bool<T>())
  __device__ uint64_t operator()(column_device_view const& d_results, size_type idx) const
  {
    return static_cast<uint64_t>(d_results.element<T>(idx));
  }
  template <typename T>
    requires(not cudf::is_integral_not_bool<T>())
  __device__ uint64_t operator()(column_device_view const&, size_type) const
  {
    CUDF_UNREACHABLE("invalid type");
  }
};

namespace {
struct from_integers_fn {
  column_device_view const d_column;
  endian swap;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(size_type idx) const
  {
    if (d_column.is_null(idx)) {
      if (d_chars == nullptr) { d_sizes[idx] = 0; }
      return;
    }

    auto value = type_dispatcher(d_column.type(), dispatch_get_int_fn{}, d_column, idx);
    // compute the number of UTF-8 bytes needed
    auto const size = static_cast<size_type>(sizeof(uint64_t) - (__clzll(value) / CHAR_BIT));
    if (d_chars == nullptr) {
      d_sizes[idx] = size;
      return;
    }

    // byte swap the bytes to the output buffer
    auto output = d_chars + d_offsets[idx];
    if (swap == endian::LITTLE) {
      for (size_type i = 0; i < size; i++) {
        output[i] = static_cast<u_char>(value >> ((size - 1 - i) * CHAR_BIT));
      }
    } else {
      memcpy(output, &value, size);
    }
  };
};
}  // namespace

// Convert boolean column to strings column
std::unique_ptr<column> cast_from_integer(column_view const& integers,
                                          endian swap,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(cudf::is_integral_not_bool(integers.type()),
               "Input type must be an integer type",
               cudf::data_type_error);
  if (integers.size() == 0) { return make_empty_column(type_id::STRING); }

  auto d_column = column_device_view::create(integers, stream);

  auto [offsets, chars] =
    make_strings_children(from_integers_fn{*d_column, swap}, integers.size(), stream, mr);

  return make_strings_column(integers.size(),
                             std::move(offsets),
                             chars.release(),
                             integers.null_count(),
                             cudf::detail::copy_bitmask(integers, stream, mr));
}

}  // namespace detail

// external API

std::unique_ptr<column> cast_from_integer(column_view const& integers,
                                          endian swap,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::cast_from_integer(integers, swap, stream, mr);
}

namespace detail {

std::optional<cudf::data_type> integer_cast_type(strings_column_view const& input,
                                                 cuda::stream_ref stream)
{
  if (input.size() == 0) { return std::nullopt; }
  auto d_strings = column_device_view::create(input.parent(), stream);

  auto bits_size = thrust::transform_reduce(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{input.size()},
    cuda::proclaim_return_type<size_type>(
      [d_strings = *d_strings] __device__(size_type idx) -> size_type {
        if (d_strings.is_null(idx)) { return 0; }
        auto const d_str  = d_strings.element<string_view>(idx);
        auto const bits   = d_str.size_bytes() * CHAR_BIT;
        u_char first_byte = bits > 0 ? d_str.data()[0] : 0;
        return bits - ((first_byte & 0x80) == 0);
      }),
    size_type{0},
    cuda::maximum<size_type>{});

  if (bits_size <= 8) { return data_type{type_id::INT8}; }
  if (bits_size <= 16) {
    return bits_size == 16 ? data_type{type_id::UINT16} : data_type{type_id::INT16};
  }
  if (bits_size <= 32) {
    return bits_size == 32 ? data_type{type_id::UINT32} : data_type{type_id::INT32};
  }
  if (bits_size <= 64) {
    return bits_size == 64 ? data_type{type_id::UINT64} : data_type{type_id::INT64};
  }
  return std::nullopt;
}
}  // namespace detail

std::optional<cudf::data_type> integer_cast_type(strings_column_view const& input,
                                                 cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  return detail::integer_cast_type(input, stream);
}
}  // namespace strings
}  // namespace cudf
