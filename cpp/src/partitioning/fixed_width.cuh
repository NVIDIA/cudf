/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/bit>

#include <algorithm>
#include <cstdint>

namespace cudf::detail::partition_metadata {

/**
 * @brief Storage layout used for per-row partition metadata.
 */
enum class layout {
  PACKED32,  ///< Store each row's metadata in a packed 32-bit word
  DEFAULT    ///< Store partition identifiers and offsets separately
};

/**
 * @brief Computes the ceiling of the base-2 logarithm of a value.
 *
 * @param value Value whose ceiling base-2 logarithm is requested
 * @return `ceil(log2(value))`, or zero when `value` is zero or one
 */
constexpr int ceil_log2(std::uint64_t value) noexcept
{
  return value < 2 ? 0 : cuda::std::bit_width(value - 1);
}

/**
 * @brief Selects whether partition metadata can use packed 32-bit storage.
 *
 * @param num_partitions Number of possible partition identifiers
 * @param rows_per_block Maximum number of rows processed by one CTA
 * @return The packed layout when both fields need at most 32 bits in total, otherwise the default
 * layout
 */
constexpr layout pick_layout(size_type num_partitions, size_type rows_per_block) noexcept
{
  auto const partition_bits = ceil_log2(static_cast<std::uint64_t>(num_partitions));
  auto const offset_bits    = ceil_log2(static_cast<std::uint64_t>(rows_per_block));
  auto const total_bits     = partition_bits + offset_bits;
  return total_bits <= 32 ? layout::PACKED32 : layout::DEFAULT;
}

/**
 * @brief Device-accessible view of partition metadata packed into 32-bit words.
 *
 * Each word stores the partition identifier in its low-order `partition_bits` bits and the
 * CTA-local partition offset in its remaining high-order bits.
 */
struct packed_view {
  device_span<std::uint32_t> values;  ///< Packed metadata values, one for each input row
  int partition_bits;  ///< Number of low-order bits reserved for the partition identifier

  /**
   * @brief Returns the number of rows represented by this view.
   *
   * @return Number of packed metadata values
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr size_type size() const
  {
    return static_cast<size_type>(values.size());
  }

  /**
   * @brief Packs and stores metadata for one input row.
   *
   * @param row_index Input row whose metadata is stored
   * @param partition Partition identifier for the row
   * @param offset CTA-local offset of the row within its partition
   */
  CUDF_HOST_DEVICE constexpr void store(size_type row_index,
                                        size_type partition,
                                        size_type offset) const
  {
    values[row_index] = (static_cast<std::uint32_t>(offset) << partition_bits) |
                        static_cast<std::uint32_t>(partition);
  }

  /**
   * @brief Loads and decodes the metadata for an input row.
   *
   * @param row_index Input row whose metadata is requested
   * @param[out] partition Partition identifier for the row
   * @param[out] offset CTA-local offset of the row within its partition
   */
  CUDF_HOST_DEVICE constexpr void load(size_type row_index,
                                       size_type& partition,
                                       size_type& offset) const
  {
    auto const value = values[row_index];
    partition        = static_cast<size_type>(value & partition_mask());
    offset           = static_cast<size_type>(value >> partition_bits);
  }

  /**
   * @brief Returns the partition identifier for an input row.
   *
   * @param row_index Input row whose partition identifier is requested
   * @return Partition identifier for the row
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr size_type partition(size_type row_index) const
  {
    return static_cast<size_type>(values[row_index] & partition_mask());
  }

 private:
  /**
   * @brief Returns a mask covering the packed partition-identifier bits.
   *
   * @return Partition-identifier bit mask
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr std::uint32_t partition_mask() const
  {
    return partition_bits == 0 ? std::uint32_t{0}
                               : (std::uint32_t{1} << partition_bits) - std::uint32_t{1};
  }
};

/**
 * @brief Device-accessible view of partition metadata stored in separate arrays.
 */
struct default_view {
  device_span<size_type> partitions;  ///< Partition identifier for each input row
  device_span<size_type> offsets;     ///< CTA-local partition offset for each input row

  /**
   * @brief Returns the number of rows represented by this view.
   *
   * @return Number of partition metadata entries
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr size_type size() const
  {
    return static_cast<size_type>(partitions.size());
  }

  /**
   * @brief Stores metadata for one input row.
   *
   * @param row_index Input row whose metadata is stored
   * @param partition Partition identifier for the row
   * @param offset CTA-local offset of the row within its partition
   */
  CUDF_HOST_DEVICE constexpr void store(size_type row_index,
                                        size_type partition,
                                        size_type offset) const
  {
    partitions[row_index] = partition;
    offsets[row_index]    = offset;
  }

  /**
   * @brief Loads the metadata for an input row.
   *
   * @param row_index Input row whose metadata is requested
   * @param[out] partition Partition identifier for the row
   * @param[out] offset CTA-local offset of the row within its partition
   */
  CUDF_HOST_DEVICE constexpr void load(size_type row_index,
                                       size_type& partition,
                                       size_type& offset) const
  {
    partition = partitions[row_index];
    offset    = offsets[row_index];
  }

  /**
   * @brief Returns the partition identifier for an input row.
   *
   * @param row_index Input row whose partition identifier is requested
   * @return Partition identifier for the row
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr size_type partition(size_type row_index) const
  {
    return partitions[row_index];
  }
};

}  // namespace cudf::detail::partition_metadata

namespace cudf::detail {

/**
 * @brief Checks if a table is compatible with fixed-width row operations
 *
 * A table is compatible with fixed-width row operations if all of its columns contain fixed-width
 * data types.
 *
 * @param table The table to check for compatibility
 * @return Boolean indicating if the table is compatible with fixed-width row operations
 */
inline bool is_fixed_width_partition_compatible(table_view const& table)
{
  return std::all_of(table.begin(), table.end(), [](column_view const& column) {
    return cudf::is_fixed_width(column.type());
  });
}

}  // namespace cudf::detail
