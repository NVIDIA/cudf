/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>

/**
 * @file
 * @brief Class definition for cudf::strings_column_view
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup strings_classes
 * @{
 */

/**
 * @brief Given a column-view of strings type, an instance of this class
 * provides a wrapper on this compound column for strings operations.
 */
class strings_column_view : private column_view {
 public:
  /**
   * @brief Construct a new strings column view object from a column view.s
   *
   * @param strings_column The column view to wrap.
   */
  strings_column_view(column_view strings_column);
  // So we can use this from cython.
  strings_column_view()                           = default;
  strings_column_view(strings_column_view&&)      = default;  ///< Move constructor
  strings_column_view(strings_column_view const&) = default;  ///< Copy constructor
  ~strings_column_view() override                 = default;
  /**
   * @brief Copy assignment operator
   *
   * @return Reference to this instance
   */
  strings_column_view& operator=(strings_column_view const&) = default;
  /**
   * @brief Move assignment operator
   *
   * @return Reference to this instance (after transferring ownership)
   */
  strings_column_view& operator=(strings_column_view&&) = default;

  static constexpr size_type offsets_column_index{0};  ///< Child index of the offsets column

  using column_view::has_nulls;
  using column_view::is_empty;
  using column_view::null_count;
  using column_view::null_mask;
  using column_view::offset;
  using column_view::size;

  using offset_iterator = size_type const*;  ///< offsets iterator type
  using chars_iterator  = char const*;       ///< character iterator type

  /**
   * @brief Returns the parent column.
   *
   * @return The parents column
   */
  [[nodiscard]] column_view parent() const;

  /**
   * @brief Returns the internal column of offsets
   *
   * @throw cudf::logic_error if this is an empty column
   * @return The offsets column
   */
  [[nodiscard]] column_view offsets() const;

  /**
   * @brief Returns the number of bytes in the chars child column.
   *
   * This accounts for empty columns but does not reflect a sliced parent column
   * view  (i.e.: non-zero offset or reduced row count).
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Number of bytes in the chars child column
   */
  [[nodiscard]] int64_t chars_size(cuda::stream_ref stream) const;

  /**
   * @brief Return an iterator for the chars child column.
   *
   * This does not apply the offset of the parent.
   * The offsets child must be used to properly address the char bytes.
   *
   * For example, to access the first character of string `i` (accounting for
   * a sliced column offset) use: `chars_begin(stream)[offsets_begin()[i]]`.
   *
   * @return Iterator pointing to the first char byte.
   */
  [[nodiscard]] chars_iterator chars_begin(cuda::stream_ref) const noexcept;

  /**
   * @brief Return an end iterator for the offsets child column.
   *
   * This does not apply the offset of the parent.
   * The offsets child must be used to properly address the char bytes.
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Iterator pointing 1 past the last char byte.
   */
  [[nodiscard]] chars_iterator chars_end(cuda::stream_ref stream) const;
};

//! Strings column APIs.
namespace strings {
}  // namespace strings

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
