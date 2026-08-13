/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <memory>

namespace cudf {
namespace lists {
namespace detail {

/**
 * @brief Holder for a list row's positional information, without
 *        also holding a reference to the list column.
 *
 * Analogous to the list_view, this class is default constructable,
 * and can thus be stored in rmm::device_uvector. It is used to represent
 * the results of a `scatter()` operation; a device_uvector may hold
 * several instances of unbound_list_view, each with a flag indicating
 * whether it came from the scatter source or target. Each instance
 * may later be "bound" to the appropriate source/target column, to
 * reconstruct the list_view.
 */
struct unbound_list_view {
  /**
   * @brief Flag type, indicating whether this list row originated from
   *        the source or target column, in `scatter()`.
   */
  enum class label_type : bool { SOURCE, TARGET };

  using lists_column_device_view = cudf::lists_column_device_view;
  using list_device_view         = cudf::list_device_view;

  unbound_list_view()                                    = default;
  unbound_list_view(unbound_list_view const&)            = default;
  unbound_list_view(unbound_list_view&&)                 = default;
  unbound_list_view& operator=(unbound_list_view const&) = default;
  unbound_list_view& operator=(unbound_list_view&&)      = default;

  /**
   * @brief __device__ Constructor, for use from `scatter()`.
   *
   * @param scatter_source_label Whether the row came from source or target
   * @param lists_column The actual source/target lists column
   * @param row_index Index of the row in lists_column that this instance represents
   */
  __device__ inline unbound_list_view(label_type scatter_source_label,
                                      cudf::lists_column_device_view const& lists_column,
                                      size_type const& row_index)
    : _label{scatter_source_label}, _row_index{row_index}
  {
    _size = list_device_view{lists_column, row_index}.size();
  }

  /**
   * @brief __device__ Constructor, for use when constructing the child column
   *        of a scattered list column
   *
   * @param scatter_source_label Whether the row came from source or target
   * @param row_index Index of the row that this instance represents in the source/target column
   * @param size The number of elements in this list row
   */
  __device__ inline unbound_list_view(label_type scatter_source_label,
                                      size_type const& row_index,
                                      size_type const& size)
    : _label{scatter_source_label}, _row_index{row_index}, _size{size}
  {
  }

  /**
   * @brief Returns number of elements in this list row.
   */
  [[nodiscard]] __device__ inline size_type size() const { return _size; }

  /**
   * @brief Returns whether this row came from the `scatter()` source or target
   */
  [[nodiscard]] __device__ inline label_type label() const { return _label; }

  /**
   * @brief Returns the index in the source/target column
   */
  [[nodiscard]] __device__ inline size_type row_index() const { return _row_index; }

  /**
   * @brief Binds to source/target column (depending on SOURCE/TARGET labels),
   *        to produce a bound list_view.
   *
   * @param scatter_source Source column for the scatter operation
   * @param scatter_target Target column for the scatter operation
   * @return A (bound) list_view for the row that this object represents
   */
  [[nodiscard]] __device__ inline list_device_view bind_to_column(
    lists_column_device_view const& scatter_source,
    lists_column_device_view const& scatter_target) const
  {
    return list_device_view(_label == label_type::SOURCE ? scatter_source : scatter_target,
                            _row_index);
  }

 private:
  // Note: Cannot store reference to list column, because of storage in device_uvector.
  // Only keep track of whether this list row came from the source or target of scatter.

  label_type _label{
    label_type::SOURCE};   // Whether this list row came from the scatter source or target.
  size_type _row_index{};  // Row index in the Lists column.
  size_type _size{};       // Number of elements in *this* list row.
};

std::unique_ptr<column> build_lists_child_column_recursive(
  data_type child_column_type,
  rmm::device_uvector<unbound_list_view> const& list_vector,
  cudf::column_view const& list_offsets,
  cudf::lists_column_view const& source_lists_column_view,
  cudf::lists_column_view const& target_lists_column_view,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace lists
}  // namespace cudf
