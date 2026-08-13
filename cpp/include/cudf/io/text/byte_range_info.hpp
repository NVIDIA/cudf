/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <cstdint>
#include <vector>

/**
 * @file
 * @brief Class and helper functions for specifying a byte range within a data source.
 */

namespace CUDF_EXPORT cudf {
namespace io {
namespace text {
/**
 * @addtogroup io_readers
 * @{
 */

/**
 * @brief stores offset and size used to indicate a byte range
 */
class byte_range_info {
 private:
  int64_t _offset{};  ///< offset in bytes
  int64_t _size{};    ///< size in bytes

 public:
  byte_range_info() = default;
  /**
   * @brief Constructs a byte_range_info object
   *
   * @param offset offset in bytes
   * @param size size in bytes
   */
  byte_range_info(int64_t offset, int64_t size);

  /**
   * @brief Copy constructor
   *
   * @param other byte_range_info object to copy
   */
  byte_range_info(byte_range_info const& other) noexcept = default;
  /**
   * @brief  Copy assignment operator
   *
   * @param other byte_range_info object to copy
   * @return this object after copying
   */
  byte_range_info& operator=(byte_range_info const& other) noexcept = default;

  /**
   * @brief Get the offset in bytes
   *
   * @return Offset in bytes
   */
  [[nodiscard]] int64_t offset() const { return _offset; }

  /**
   * @brief Get the size in bytes
   *
   * @return Size in bytes
   */
  [[nodiscard]] int64_t size() const { return _size; }

  /**
   * @brief Returns whether the span is empty.
   *
   * @return true iff the range is empty, i.e. `size() == 0`
   */
  [[nodiscard]] bool is_empty() const { return size() == 0; }
};

/**
 * @brief Create a collection of consecutive ranges between [0, total_bytes).
 *
 * Each range wil be the same size except if `total_bytes` is not evenly divisible by
 * `range_count`, in which case the last range size will be the remainder.
 *
 * @param total_bytes total number of bytes in all ranges
 * @param range_count total number of ranges in which to divide bytes
 * @return Vector of range objects
 */
std::vector<byte_range_info> create_byte_range_infos_consecutive(int64_t total_bytes,
                                                                 int64_t range_count);

/**
 * @brief Create a byte_range_info which represents as much of a file as possible. Specifically,
 * ``[0, numeric_limits<int64_t>:\:max())``.
 *
 * @return Byte range info of size ``[0, numeric_limits<int64_t>:\:max())``
 */
byte_range_info create_byte_range_info_max();

/** @} */  // end of group

}  // namespace text
}  // namespace io
}  // namespace CUDF_EXPORT cudf
