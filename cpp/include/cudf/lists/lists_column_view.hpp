/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>

/**
 * @file
 * @brief Class definition for cudf::lists_column_view
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup lists_classes
 * @{
 */

/**
 * @brief Given a column-view of lists type, an instance of this class
 * provides a wrapper on this compound column for list operations.
 */
class lists_column_view : private column_view {
 public:
  lists_column_view() = default;
  /**
   * @brief Construct a new lists column view object from a column view.
   *
   * @param lists_column The column view to wrap
   */
  lists_column_view(column_view const& lists_column);
  lists_column_view(lists_column_view&&)      = default;  ///< Move constructor
  lists_column_view(lists_column_view const&) = default;  ///< Copy constructor
  ~lists_column_view() override               = default;
  /**
   * @brief Copy assignment operator
   *
   * @return The reference to this lists column
   */
  lists_column_view& operator=(lists_column_view const&) = default;
  /**
   * @brief Move assignment operator
   *
   * @return The reference to this lists column
   */
  lists_column_view& operator=(lists_column_view&&) = default;

  static constexpr size_type offsets_column_index{0};  ///< The index of the offsets column
  static constexpr size_type child_column_index{1};    ///< The index of the child column

  using column_view::child_begin;
  using column_view::child_end;
  using column_view::has_nulls;
  using column_view::is_empty;
  using column_view::null_count;
  using column_view::null_mask;
  using column_view::offset;
  using column_view::size;
  using offset_iterator = int32_t const*;  ///< Iterator type for offsets

  /**
   * @brief Returns the parent column.
   *
   * @return The parent column
   */
  [[nodiscard]] column_view parent() const;

  /**
   * @brief Returns the internal column of offsets
   *
   * @throw cudf::logic_error if this is an empty column
   * @return The internal column of offsets
   */
  [[nodiscard]] column_view offsets() const;

  /**
   * @brief Returns the internal child column
   *
   * @throw cudf::logic_error if this is an empty column
   * @return The internal child column
   */
  [[nodiscard]] column_view child() const;

  /**
   * @brief Returns the internal child column, applying any offset from the root.
   *
   * Slice/split offset values are only stored at the root level of a list column.
   * So when doing computations on them, we need to apply that offset to
   * the child columns when recursing.  Most functions operating in a recursive manner
   * on lists columns should be using `get_sliced_child()` instead of `child()`.
   *
   * @throw cudf::logic_error if this is an empty column
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return A sliced child column view
   */
  [[nodiscard]] column_view get_sliced_child(rmm::cuda_stream_view stream) const;

  /**
   * @brief Return first offset (accounting for column offset)
   *
   * @return Pointer to the first offset
   */
  [[nodiscard]] offset_iterator offsets_begin() const noexcept
  {
    return offsets().begin<int32_t>() + offset();
  }

  /**
   * @brief Return pointer to the position that is one past the last offset
   *
   * This function return the position that is one past the last offset of the lists column.
   * Since the current lists column may be a sliced column, this offsets_end() iterator should not
   * be computed using the size of the offsets() child column, which is also the offsets of the
   * entire original (non-sliced) lists column.
   *
   * @return Pointer to one past the last offset
   */
  [[nodiscard]] offset_iterator offsets_end() const noexcept
  {
    return offsets_begin() + size() + 1;
  }
};
/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
