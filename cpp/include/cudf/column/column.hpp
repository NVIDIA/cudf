/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @file
 * @brief Class definition for cudf::column
 */

namespace CUDF_EXPORT cudf {

/**
 * @brief A container of nullable device data as a column of elements.
 *
 * @ingroup column_classes Column
 * @{
 */

class column {
 public:
  column()                               = default;
  ~column()                              = default;
  column& operator=(column const& other) = delete;
  column& operator=(column&& other)      = delete;

  /**
   * @brief Construct a new column object by deep copying the contents of
   *`other`.
   *
   * Uses the specified `stream` and device memory resource for all allocations
   * and copies.
   *
   * @param other The `column` to copy
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for all device memory allocations
   */
  column(column const& other,
         cuda::stream_ref stream           = cudf::get_default_stream(),
         rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Move the contents from `other` to create a new column.
   *
   * After the move, `other.size() == 0` and `other.type() = {EMPTY}`
   *
   * @param other The column whose contents will be moved into the new column
   */
  column(column&& other) noexcept;

  /**
   * @brief Construct a new column by taking ownership of the contents of a device_uvector.
   *
   * @param other The device_uvector whose contents will be moved into the new column.
   * @param null_mask Column's null value indicator bitmask. May be empty if `null_count` is 0.
   * @param null_count The count of null elements.
   */
  template <typename T, CUDF_ENABLE_IF(cudf::is_numeric<T>() or cudf::is_chrono<T>())>
  column(rmm::device_uvector<T>&& other, rmm::device_buffer&& null_mask, size_type null_count)
    : _type{cudf::data_type{cudf::type_to_id<T>()}},
      _size{[&]() {
        CUDF_EXPECTS(
          other.size() <= static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
          "The device_uvector size exceeds the column size limit",
          std::overflow_error);
        return static_cast<size_type>(other.size());
      }()},
      _data{other.release()},
      _null_mask{std::move(null_mask)},
      _null_count{null_count}
  {
  }

  /**
   * @brief Construct a new column from existing device memory.
   *
   * @note This constructor is primarily intended for use in column factory
   * functions.
   *
   * @throws cudf::logic_error if `size < 0`
   *
   * @param dtype The element type
   * @param size The number of elements in the column
   * @param data The column's data
   * @param null_mask Column's null value indicator bitmask. May be empty if `null_count` is 0.
   * @param null_count Optional, the count of null elements.
   * @param children Optional, vector of child columns
   */
  template <typename B1, typename B2 = rmm::device_buffer>
  column(data_type dtype,
         size_type size,
         B1&& data,
         B2&& null_mask,
         size_type null_count,
         std::vector<std::unique_ptr<column>>&& children = {})
    : _type{dtype},
      _size{size},
      _data{std::forward<B1>(data)},
      _null_mask{std::forward<B2>(null_mask)},
      _null_count{null_count},
      _children{std::move(children)}
  {
    CUDF_EXPECTS(size >= 0, "Column size cannot be negative.");
  }

  /**
   * @brief Construct a new column by deep copying the contents of a
   * `column_view`.
   *
   * This accounts for the `column_view`'s offset.
   *
   * @param view The view to copy
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for all device memory allocations
   */
  explicit column(column_view view,
                  cuda::stream_ref stream           = cudf::get_default_stream(),
                  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Returns the column's logical element type
   *
   * @return The column's logical element type
   */
  [[nodiscard]] data_type type() const noexcept { return _type; }

  /**
   * @brief Returns the number of elements
   *
   * @return The number of elements
   */
  [[nodiscard]] size_type size() const noexcept { return _size; }

  /**
   * @brief Returns the count of null elements.
   *
   * @return The number of null elements
   */
  [[nodiscard]] size_type null_count() const { return _null_count; }

  /**
   * @brief Sets the column's null value indicator bitmask to `new_null_mask`.
   *
   * @throws cudf::logic_error if new_null_count is larger than 0 and the size
   * of `new_null_mask` does not match the size of this column.
   *
   * @param new_null_mask New null value indicator bitmask (rvalue overload &
   * moved) to set the column's null value indicator mask. May be empty if
   * `new_null_count` is 0.
   * @param new_null_count The count of null elements.
   */
  void set_null_mask(rmm::device_buffer&& new_null_mask, size_type new_null_count);

  /**
   * @brief Sets the column's null value indicator bitmask to `new_null_mask`.
   *
   * @throws cudf::logic_error if new_null_count is larger than 0 and the size of `new_null_mask`
   * does not match the size of this column.
   *
   * @param new_null_mask New null value indicator bitmask (lvalue overload & copied) to set the
   * column's null value indicator mask. May be empty if `new_null_count` is 0.
   * @param new_null_count The count of null elements
   * @param stream The stream on which to perform the allocation and copy. Uses the default CUDF
   * stream if none is specified.
   */
  void set_null_mask(rmm::device_buffer const& new_null_mask,
                     size_type new_null_count,
                     cuda::stream_ref stream = cudf::get_default_stream());

  /**
   * @brief Updates the count of null elements.
   *
   * @throws cudf::logic_error if `new_null_count > 0 and nullable() == false`
   *
   * @param new_null_count The new null count.
   */
  void set_null_count(size_type new_null_count);

  /**
   * @brief Indicates whether it is possible for the column to contain null
   * values, i.e., it has an allocated null mask.
   *
   * This may return `false` iff `null_count() == 0`.
   *
   * May return true even if `null_count() == 0`. This function simply indicates
   * whether the column has an allocated null mask.
   *
   * @return true The column can hold null values
   * @return false The column cannot hold null values
   */
  [[nodiscard]] bool nullable() const noexcept { return (_null_mask.size() > 0); }

  /**
   * @brief Indicates whether the column contains null elements.
   *
   * @return true One or more elements are null
   * @return false Zero elements are null
   */
  [[nodiscard]] bool has_nulls() const noexcept { return (null_count() > 0); }

  /**
   * @brief Returns the number of child columns
   *
   * @return The number of child columns
   */
  [[nodiscard]] size_type num_children() const noexcept { return _children.size(); }

  /**
   * @brief Returns a reference to the specified child
   *
   * @param child_index Index of the desired child
   * @return Reference to the desired child
   */
  column& child(size_type child_index) noexcept { return *_children[child_index]; };

  /**
   * @brief Returns a const reference to the specified child
   *
   * @param child_index Index of the desired child
   * @return Const reference to the desired child
   */
  [[nodiscard]] column const& child(size_type child_index) const noexcept
  {
    return *_children[child_index];
  };

  /**
   * @brief Wrapper for the contents of a column.
   *
   * Returned by `column::release()`.
   */
  struct contents {
    std::unique_ptr<rmm::device_buffer> data;       ///< data device memory buffer
    std::unique_ptr<rmm::device_buffer> null_mask;  ///< null mask device memory buffer
    std::vector<std::unique_ptr<column>> children;  ///< child columns
  };

  /**
   * @brief Releases ownership of the column's contents.
   *
   * It is the caller's responsibility to query the `size(), null_count(),
   * type()` before invoking `release()`.
   *
   * After calling `release()` on a column it will be empty, i.e.:
   * - `type() == data_type{EMPTY}`
   * - `size() == 0`
   * - `null_count() == 0`
   * - `num_children() == 0`
   *
   * @return A `contents` struct containing the data, null mask, and children of
   * the column.
   */
  contents release() noexcept;

  /**
   * @brief Returns the total device allocation size of the column in bytes
   *
   * This includes the size of the data buffer, null mask, and any child columns.
   * It also includes any padding bytes and is at least as large as
   * `size()*sizeof(T)` for the column's logical type.
   *
   * @return The total allocation size in bytes
   */
  [[nodiscard]] std::size_t alloc_size() const;

  /**
   * @brief Creates an immutable, non-owning view of the column's data and
   * children.
   *
   * @return The immutable, non-owning view
   */
  [[nodiscard]] column_view view() const;

  /**
   * @brief Implicit conversion operator to a `column_view`.
   *
   * This allows passing a `column` object directly into a function that
   * requires a `column_view`. The conversion is automatic.
   *
   * @return Immutable, non-owning `column_view`
   */
  operator column_view() const { return this->view(); };

  /**
   * @brief Creates a mutable, non-owning view of the column's data, null mask,
   * and children
   *
   * @return The mutable, non-owning view
   */
  mutable_column_view mutable_view();

  /**
   * @brief Implicit conversion operator to a `mutable_column_view`.
   *
   * This allows passing a `column` object into a function that accepts a
   * `mutable_column_view`. The conversion is automatic.
   *
   * The caller is expected to update the null count appropriately if the null mask
   * is modified.
   *
   * @return Mutable, non-owning `mutable_column_view`
   */
  operator mutable_column_view() { return this->mutable_view(); };

 private:
  cudf::data_type _type{type_id::EMPTY};  ///< Logical type of elements in the column
  cudf::size_type _size{};                ///< The number of elements in the column
  rmm::device_buffer _data{};             ///< Dense, contiguous, type erased device memory
                                          ///< buffer containing the column elements
  rmm::device_buffer _null_mask{};        ///< Bitmask used to represent null values.
                                          ///< May be empty if `null_count() == 0`
  mutable cudf::size_type _null_count{};  ///< The number of null elements
  std::vector<std::unique_ptr<column>> _children{};  ///< Depending on element type, child
                                                     ///< columns may contain additional data
};

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
