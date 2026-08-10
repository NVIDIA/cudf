/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cassert>
#include <memory>

/**
 * @file table_device_view.cuh
 * @brief Table device view class definitions
 */

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Base class for a device table of `ColumnDeviceView`s
 *
 * This class should not be used directly. Instead:
 * - `table_device_view` should be used for a table of columns of type `column_device_view`
 * - `mutable_table_device_view` should be used for a table of columns of type
 * `mutable_column_device_view`
 *
 * All public constructors and member functions of `table_device_view_base` are
 * available in both `table_device_view` and `mutable_table_device_view`.
 *
 * @tparam ColumnDeviceView The type of column device view the table contains:
 *                          expects column_device_view or mutable_column_device_view
 * @tparam HostTableView The type of table view used to create the table device view:
 *                       expects table_view or mutable_table_view
 */
template <typename ColumnDeviceView, typename HostTableView>
class table_device_view_base {
 public:
  table_device_view_base()                              = delete;
  ~table_device_view_base()                             = default;
  table_device_view_base(table_device_view_base const&) = default;  ///< Copy constructor
  table_device_view_base(table_device_view_base&&)      = default;  ///< Move constructor
  /**
   * @brief Copy assignment operator
   *
   * @return Reference to this object
   */
  table_device_view_base& operator=(table_device_view_base const&) = default;
  /**
   * @brief Move assignment operator
   *
   * @return Reference to this object (after transferring ownership)
   */
  table_device_view_base& operator=(table_device_view_base&&) = default;

  /**
   * @brief Returns an iterator to the first view in the `table`.
   *
   * @return An iterator to the first view in the `table`
   */
  __device__ ColumnDeviceView* begin() const noexcept { return _columns; }

  /**
   * @brief Returns an iterator one past the last column view in the `table`.
   *
   * `end()` acts as a place holder. Attempting to dereference it results in
   * undefined behavior.
   *
   * @return An iterator to one past the last column view in the `table`
   */
  __device__ ColumnDeviceView* end() const noexcept { return _columns + _num_columns; }

  /**
   * @brief Returns a reference to the view of the specified column
   *
   * @param column_index The index of the desired column
   * @return A reference to the desired column
   */
  __device__ ColumnDeviceView const& column(size_type column_index) const noexcept
  {
    assert(column_index >= 0);
    assert(column_index < _num_columns);
    return _columns[column_index];
  }

  /**
   * @brief Returns a reference to the view of the specified column
   *
   * @param column_index The index of the desired column
   * @return A reference to the desired column
   */
  __device__ ColumnDeviceView& column(size_type column_index) noexcept
  {
    assert(column_index >= 0);
    assert(column_index < _num_columns);
    return _columns[column_index];
  }

  /**
   * @brief Returns the number of columns
   *
   * @return The number of columns
   */
  [[nodiscard]] __host__ __device__ size_type num_columns() const noexcept { return _num_columns; }

  /**
   * @brief Returns the number of rows
   *
   * @return The number of rows
   */
  [[nodiscard]] __host__ __device__ size_type num_rows() const noexcept { return _num_rows; }

  /**
   * @brief Destroy the `table_device_view` object.
   *
   * @note Does not free the table data, simply frees the device memory
   * allocated to hold the constituent column views.
   */
  void destroy();

 private:
  ColumnDeviceView* _columns{};  ///< Array of view objects in device memory
  size_type _num_rows{};
  size_type _num_columns{};

 protected:
  /**
   * @brief Constructor for subclasses to create a table device view from a host table view
   *
   * @param source_view The host table_view to create table device view from
   * @param columns Pointer to the array of column device views in device memory corresponding to
   * the columns of `source_view`
   */
  table_device_view_base(HostTableView source_view, ColumnDeviceView* columns);
};
}  // namespace detail

/**
 * @brief Table device view that is usable in device memory
 */
class table_device_view : public detail::table_device_view_base<column_device_view, table_view> {
 public:
  /**
   * @brief Factory to construct a table device view that is usable in device memory.
   *
   * Allocates and copies views of `source_view`'s children to device memory to
   * make them accessible in device code.
   *
   * Returns a `std::unique_ptr<table_device_view>` with a custom deleter to
   * free the device memory allocated for the children.
   *
   * @param source_view The table view whose contents will be copied to create a new table
   * @param stream CUDA stream used for device memory operations
   * @param mr Device memory resource used to allocate the returned column-view storage
   * @return A `unique_ptr` to a `table_device_view` that makes the data from `source_view`
   * available in device memory
   */
  static std::unique_ptr<table_device_view, std::function<void(table_device_view*)>> create(
    table_view source_view,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

 private:
  table_device_view(table_view source_view, column_device_view* columns);
};

/**
 * @brief Mutable table device view that is usable in device memory
 *
 * Elements of the table can be modified in device memory.
 */
class mutable_table_device_view
  : public detail::table_device_view_base<mutable_column_device_view, mutable_table_view> {
 public:
  /**
   * @brief Factory to construct a mutable table device view that is usable in device memory.
   *
   * Allocates and copies views of `source_view`'s children to device memory to
   * make them accessible in device code.
   *
   * Returns a `std::unique_ptr<mutable_table_device_view>` with a custom deleter to
   * free the device memory allocated for the children.
   *
   * @param source_view The table view whose contents will be copied to create a new table
   * @param stream CUDA stream used for device memory operations
   * @param mr Device memory resource used to allocate the returned column-view storage
   * @return A `unique_ptr` to a `mutable_table_device_view` that makes the data from `source_view`
   * available in device memory
   */
  static std::unique_ptr<mutable_table_device_view, std::function<void(mutable_table_device_view*)>>
  create(mutable_table_view source_view,
         rmm::cuda_stream_view stream      = cudf::get_default_stream(),
         rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

 private:
  mutable_table_device_view(mutable_table_view source_view, mutable_column_device_view* columns);
};

/**
 * @brief Copies the contents of a table_view to a column device view in contiguous device memory
 *
 * @tparam ColumnDeviceView The column device view type to copy to
 * @tparam HostTableView The type of the table_view to copy from
 * @param source_view The table_view to copy from
 * @param stream The stream to use for device memory allocation
 * @param mr Device memory resource used to allocate the returned device buffer
 * @return tuple of device_buffer and @p ColumnDeviceView device pointer
 */
template <typename ColumnDeviceView, typename HostTableView>
std::pair<std::unique_ptr<rmm::device_buffer>, ColumnDeviceView*> create_column_device_views(
  HostTableView source_view,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace CUDF_EXPORT cudf
