/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/column/column_view.hpp>

namespace CUDF_EXPORT cudf {

/**
 * @brief A non-owning, immutable view of device data as a column of elements,
 * some of which may be null as indicated by a bitmask.
 *
 * @ingroup column_classes
 *
 * A `scalar_column_view` can be constructed explicitly from a `cudf::column_view`.
 *
 * Unless otherwise noted, the memory layout of the `scalar_column_view`'s data and
 * bitmask is expected to adhere to the Arrow Physical Memory Layout
 * Specification: https://arrow.apache.org/docs/format/Columnar.html#format-columnar
 *
 * Because `scalar_column_view` is non-owning, no device memory is allocated nor freed
 * when `scalar_column_view` objects are created or destroyed.
 *
 */
struct scalar_column_view : private column_view {
  /**
   * @brief Construct a `scalar_column_view` from a `column_view`.
   * @throws cudf::logic_error if the column view does not have exactly one element.
   * @param view The column view to construct from
   */
  explicit scalar_column_view(column_view view) : column_view(std::move(view))
  {
    CUDF_EXPECTS(
      this->size() == 1, "A scalar column view must have exactly one element.", std::logic_error);
  }

  using column_view::data;
  using column_view::has_nulls;
  using column_view::head;
  using column_view::is_empty;
  using column_view::null_count;
  using column_view::null_mask;
  using column_view::nullable;
  using column_view::offset;
  using column_view::size;
  using column_view::type;

  /**
   * @brief Returns a reference to the underlying `column_view` of this `scalar_column_view`.
   * @return A reference to the underlying `column_view`
   */
  [[nodiscard]] column_view const& as_column_view() const noexcept { return *this; }
};

/**
 * @brief A non-owning, mutable view of one row of device column storage.
 *
 * The owning scalar is responsible for keeping its host null-count metadata consistent with any
 * device-side changes made through this view.
 *
 * @ingroup column_classes
 */
struct mutable_scalar_column_view : private mutable_column_view {
  /**
   * @brief Construct a `mutable_scalar_column_view` from a `mutable_column_view`.
   *
   * @throws cudf::logic_error if the column view does not have exactly one element.
   *
   * @param view The mutable column view to construct from
   */
  explicit mutable_scalar_column_view(mutable_column_view view)
    : mutable_column_view(std::move(view))
  {
    CUDF_EXPECTS(
      this->size() == 1, "A scalar column view must have exactly one element.", std::logic_error);
  }

  using mutable_column_view::data;
  using mutable_column_view::head;
  using mutable_column_view::is_empty;
  using mutable_column_view::null_count;
  using mutable_column_view::null_mask;
  using mutable_column_view::nullable;
  using mutable_column_view::offset;
  using mutable_column_view::size;
  using mutable_column_view::type;

  /**
   * @brief Returns the underlying `mutable_column_view`.
   *
   * @return A reference to the underlying mutable view
   */
  [[nodiscard]] mutable_column_view const& as_mutable_column_view() const noexcept { return *this; }
};

}  // namespace CUDF_EXPORT cudf
