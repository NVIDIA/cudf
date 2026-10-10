/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/packed_types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace cudf::detail {

/**
 * @brief Physical representation of frame-of-reference offsets.
 */
enum class fused_for_layout : std::uint8_t {
  byte_aligned,  ///< Store each offset in the smallest supported whole-byte width
  bitpacked      ///< Store each offset in its exact unsigned bit width
};

/**
 * @brief Describes one independently encoded segment of a fused
 * frame-of-reference payload.
 *
 * Byte-aligned segments store `wire_width` bytes per unsigned offset.
 * Bitpacked segments store `wire_width` bits per unsigned offset. Each logical
 * value is reconstructed as `reference + offset`, modulo the logical width.
 * Raw segments use equal logical and wire byte widths, a zero reference, and
 * `fused_for_layout::byte_aligned`. All offsets are byte offsets.
 */
struct alignas(8) fused_for_segment {
  std::uint64_t wire_offset;     ///< Byte offset of the segment in the wire buffer
  std::uint64_t logical_offset;  ///< Byte offset of the segment in the reconstructed buffer
  std::uint64_t reference;       ///< Frame-of-reference base, stored as unsigned bits
  std::uint32_t element_count;   ///< Number of logical values in the segment
  std::uint8_t logical_width;    ///< Width in bytes of one reconstructed value
  std::uint8_t wire_width;       ///< Width in bytes, or bits when `is_bitpacked()` is true
  fused_for_layout layout;       ///< Physical representation of the offsets
  std::uint8_t reserved{};       ///< Reserved; must be zero

  [[nodiscard]] CUDF_HOST_DEVICE constexpr bool is_bitpacked() const noexcept
  { return layout == fused_for_layout::bitpacked; }
  [[nodiscard]] CUDF_HOST_DEVICE constexpr bool is_encoded() const noexcept
  { return is_bitpacked() || wire_width < logical_width; }
};

/**
 * @brief A wire payload whose eligible numeric segments were narrowed while
 * the input table was being packed.
 *
 * `metadata` describes the fully reconstructed logical packed table.
 * `wire_data` begins with `segment_count` device-resident segment descriptors,
 * followed by the encoded bytes. Keeping descriptors in the transmitted
 * device buffer avoids a host readback per encoded tile.
 * `logical_data_size` is the output allocation size needed by
 * `decode_fused_for`.
 */
struct fused_for_packed_columns {
  std::unique_ptr<std::vector<std::uint8_t>> metadata;  ///< Packed-table metadata
  std::unique_ptr<rmm::device_buffer> wire_data;        ///< Descriptors and encoded segments
  std::size_t segment_count;                            ///< Number of device descriptors
  std::size_t logical_data_size;  ///< Reconstructed packed-table allocation size
};

/**
 * @brief Controls the physical representation produced by fused FOR packing.
 */
struct fused_for_options {
  /// Default logical bytes considered per tile.
  static constexpr std::size_t default_tile_bytes = 8 * 1024;

  std::size_t tile_bytes{default_tile_bytes};               ///< Logical bytes considered per tile
  fused_for_layout layout{fused_for_layout::byte_aligned};  ///< Offset representation
};

/**
 * @brief Split and pack a table while applying tiled frame-of-reference
 * narrowing to eligible top-level integral, fixed-point, timestamp, and
 * duration columns.
 *
 * This is intentionally a separate API from `contiguous_split`. The ordinary
 * path remains byte-for-byte unchanged when this function is not called.
 *
 * The transform is fused into the copy that materializes each packed
 * partition. Unsupported columns and tiles for which the selected layout does
 * not narrow the values are copied raw. Each eligible tile is represented by
 * one `fused_for_segment` descriptor.
 *
 * @param input Input table
 * @param splits Row indices at which to split `input`
 * @param stream CUDA stream used for all work
 * @param mr Device memory resource for returned packed allocations
 * @param options Tile size and physical layout
 * @return One compressed wire payload per partition
 */
CUDF_EXPORT std::vector<fused_for_packed_columns> contiguous_split_fused_for(
  table_view const& input,
  std::vector<size_type> const& splits,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  fused_for_options const& options = {});

/**
 * @brief Pack a table with the same fused frame-of-reference transform.
 *
 * @param input Input table
 * @param stream CUDA stream used for all work
 * @param mr Device memory resource for the returned packed allocation
 * @param options Tile size and physical layout
 * @return One compressed wire payload
 */
CUDF_EXPORT fused_for_packed_columns pack_fused_for(table_view const& input,
                                                    cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr,
                                                    fused_for_options const& options = {});

/**
 * @brief Reconstruct ordinary packed columns from a fused
 * frame-of-reference wire payload.
 *
 * The output is materialized exactly once. Raw bytes, null masks, and string
 * offsets are copied into their final packed locations while encoded numeric
 * values are reconstructed directly into their final logical-width locations.
 *
 * @param input Payload produced by `contiguous_split_fused_for` or
 * `pack_fused_for`
 * @param stream CUDA stream used for all work
 * @param mr Device memory resource for the reconstructed packed allocation
 * @return Ordinary packed columns containing the reconstructed values
 */
CUDF_EXPORT packed_columns decode_fused_for(fused_for_packed_columns&& input,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr);

}  // namespace cudf::detail
