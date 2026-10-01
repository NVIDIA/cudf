/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/error.hpp>

#include <limits>

namespace cudf {
namespace io {
namespace text {

byte_range_info::byte_range_info(int64_t offset, int64_t size) : _offset(offset), _size(size)
{
  CUDF_EXPECTS(offset >= 0, "offset must be non-negative");
  CUDF_EXPECTS(size >= 0, "size must be non-negative");
}

byte_range_info create_byte_range_info_max() { return {0, std::numeric_limits<int64_t>::max()}; }

std::vector<byte_range_info> create_byte_range_infos_consecutive(int64_t total_bytes,
                                                                 int64_t range_count)
{
  auto range_size = cudf::detail::div_rounding_up_safe(total_bytes, range_count);
  auto ranges     = std::vector<byte_range_info>();

  ranges.reserve(range_count);

  for (int64_t i = 0; i < range_count; i++) {
    auto offset = i * range_size;
    auto size   = std::min(range_size, total_bytes - offset);
    ranges.emplace_back(offset, size);
  }

  return ranges;
}

}  // namespace text
}  // namespace io
}  // namespace cudf
