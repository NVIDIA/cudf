/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/convert/convert_ipv4.hpp>
#include <cudf/strings/detail/convert/int_to_string.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {
/**
 * @brief Converts IPv4 strings into integers.
 *
 * Only single-byte characters are expected.
 * No checking is done on the format of individual strings.
 * Any character that is not [0-9] is considered a delimiter.
 * This means "128-34-56-709" will parse successfully.
 */
struct ipv4_to_integers_fn {
  column_device_view const d_strings;

  __device__ uint32_t operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) return 0;
    string_view d_str  = d_strings.element<string_view>(idx);
    uint32_t ipvals[4] = {0};  // IPV4 format: xxx.xxx.xxx.xxx
    int32_t ipv_idx    = 0;
    int32_t factor     = 1;
    char const* in_ptr = d_str.data();
    char const* end    = in_ptr + d_str.size_bytes();
    while ((in_ptr < end) && (ipv_idx < 4)) {
      char ch = *in_ptr++;
      if (ch < '0' || ch > '9') {
        ++ipv_idx;
        factor = 1;
      } else {
        ipvals[ipv_idx] = (ipvals[ipv_idx] * factor) + static_cast<uint32_t>(ch - '0');
        factor          = 10;
      }
    }
    uint32_t result = (ipvals[0] << 24) + (ipvals[1] << 16) + (ipvals[2] << 8) + ipvals[3];
    return result;
  }
};

}  // namespace

// Convert strings column of IPv4 addresses to integers column
std::unique_ptr<column> ipv4_to_integers(strings_column_view const& input,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  size_type strings_count = input.size();
  if (strings_count == 0) {
    return make_numeric_column(data_type{type_id::UINT32}, 0, mask_state::UNALLOCATED, stream, mr);
  }

  auto strings_column = column_device_view::create(input.parent(), stream);
  // create output column copying the strings' null-mask
  auto results   = make_numeric_column(data_type{type_id::UINT32},
                                     strings_count,
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  auto d_results = results->mutable_view().data<uint32_t>();
  // fill output column with ipv4 integers
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{strings_count},
                    d_results,
                    ipv4_to_integers_fn{*strings_column});
  // done
  results->set_null_count(input.null_count());
  return results;
}

}  // namespace detail

// external API
std::unique_ptr<column> ipv4_to_integers(strings_column_view const& input,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::ipv4_to_integers(input, stream, mr);
}

namespace detail {
namespace {
/**
 * @brief Converts integers into IPv4 addresses.
 *
 * Each integer is divided into 8-bit sub-integers.
 * The sub-integers are converted into 1-3 character digits.
 * These are placed appropriately between '.' character.
 */
struct integers_to_ipv4_fn {
  column_device_view const d_column;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(size_type idx)
  {
    if (d_column.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }

    auto const ip_number = d_column.element<uint32_t>(idx);

    char* out_ptr   = d_chars ? d_chars + d_offsets[idx] : nullptr;
    int shift_bits  = 24;
    size_type bytes = 3;  // at least 3 dots: xxx.xxx.xxx.xxx
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      uint8_t value = static_cast<uint8_t>((ip_number >> shift_bits) & 0x00FF);
      if (out_ptr) {
        out_ptr += integer_to_string(value, out_ptr);
        if ((n + 1) < 4) *out_ptr++ = '.';
      } else {
        bytes += count_digits(value);
      }
      shift_bits -= 8;
    }

    if (!d_chars) { d_sizes[idx] = bytes; }
  }
};

}  // namespace

// Convert integers into IPv4 addresses
std::unique_ptr<column> integers_to_ipv4(column_view const& integers,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  if (integers.is_empty()) return make_empty_column(type_id::STRING);

  CUDF_EXPECTS(integers.type().id() == type_id::UINT32, "Input column must be UINT32 type");

  auto d_column = column_device_view::create(integers, stream);
  auto [offsets_column, chars] =
    make_strings_children(integers_to_ipv4_fn{*d_column}, integers.size(), stream, mr);

  return make_strings_column(integers.size(),
                             std::move(offsets_column),
                             chars.release(),
                             integers.null_count(),
                             cudf::detail::copy_bitmask(integers, stream, mr));
}

std::unique_ptr<column> is_ipv4(strings_column_view const& input,
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
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{input.size()},
                    d_results,
                    [d_column] __device__(size_type idx) {
                      if (d_column.is_null(idx)) return false;
                      auto const d_str = d_column.element<string_view>(idx);
                      if (d_str.empty()) return false;
                      constexpr int max_ip = 255;  // values must be in [0,255]
                      int ip_vals[4]       = {-1, -1, -1, -1};
                      int ipv_idx          = 0;  // index into ip_vals
                      for (auto const ch : d_str) {
                        if ((ch >= '0') && (ch <= '9')) {
                          auto const ip_val    = ip_vals[ipv_idx];
                          int const new_ip_val = static_cast<int>(ch - '0') +  // compute new value
                                                 (ip_val < 0 ? 0 : (10 * ip_val));
                          if (new_ip_val > max_ip) return false;
                          ip_vals[ipv_idx] = new_ip_val;
                        }
                        // here ipv_idx is incremented only when ch=='.'
                        else if (ch != '.' || (++ipv_idx > 3))
                          return false;
                      }
                      // final check for any missing values
                      return ip_vals[0] >= 0 && ip_vals[1] >= 0 && ip_vals[2] >= 0 &&
                             ip_vals[3] >= 0;
                    });
  results->set_null_count(input.null_count());
  return results;
}

}  // namespace detail

// external API

std::unique_ptr<column> integers_to_ipv4(column_view const& integers,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::integers_to_ipv4(integers, stream, mr);
}

std::unique_ptr<column> is_ipv4(strings_column_view const& input,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_ipv4(input, stream, mr);
}

}  // namespace strings
}  // namespace cudf
