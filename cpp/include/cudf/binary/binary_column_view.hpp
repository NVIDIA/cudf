/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>

/**
 * @file
 * @brief Class definition for `cudf::binary_column_view`.
 */

namespace CUDF_EXPORT cudf {

/**
 * @brief A wrapper providing access to a `BINARY` column's offsets and payload.
 */
class binary_column_view : private column_view {
 public:
  static constexpr size_type offsets_column_index{0};

  explicit binary_column_view(column_view binary_column);
  binary_column_view()                          = default;
  binary_column_view(binary_column_view const&) = default;
  binary_column_view(binary_column_view&&)      = default;
  ~binary_column_view() override                = default;
  binary_column_view& operator=(binary_column_view const&) = default;
  binary_column_view& operator=(binary_column_view&&) = default;

  using column_view::has_nulls;
  using column_view::is_empty;
  using column_view::null_count;
  using column_view::null_mask;
  using column_view::offset;
  using column_view::size;

  [[nodiscard]] column_view parent() const;
  [[nodiscard]] column_view offsets() const;

  /**
   * @brief Returns the total number of bytes in the underlying payload buffer.
   *
   * This reports the unsliced parent's payload size.
   */
  [[nodiscard]] int64_t bytes_size(rmm::cuda_stream_view stream) const;

  [[nodiscard]] uint8_t const* bytes_begin() const noexcept;
  [[nodiscard]] uint8_t const* bytes_end(rmm::cuda_stream_view stream) const;
};

}  // namespace CUDF_EXPORT cudf
