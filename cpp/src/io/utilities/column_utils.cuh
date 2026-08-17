/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/statistics/statistics.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/for_each.h>

namespace cudf {
namespace io {

/**
 * @brief Create column_device_view pointers from leaf columns
 *
 * A device_uvector is created to store the leaves of parent columns. The
 * column descriptor array is updated to point to these leaf columns.
 *
 * @tparam ColumnDescriptor Struct describing properties of columns with
 * pointers to leaf and parent columns
 *
 * @param col_desc Column description array
 * @param parent_table_device_view Table device view containing parent columns
 * @param stream CUDA stream to use
 *
 * @return Device array containing leaf column device views
 */
template <typename ColumnDescriptor>
rmm::device_uvector<column_device_view> create_leaf_column_device_views(
  typename cudf::device_span<ColumnDescriptor> col_desc,
  table_device_view const& parent_table_device_view,
  cuda::stream_ref stream)
{
  rmm::device_uvector<column_device_view> leaf_column_views(parent_table_device_view.num_columns(),
                                                            stream);
  auto leaf_columns = cudf::device_span<column_device_view>{leaf_column_views};

  auto iter = cuda::counting_iterator<size_type>{0};
  thrust::for_each(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    iter,
    iter + parent_table_device_view.num_columns(),
    [col_desc, parent_col_view = parent_table_device_view, leaf_columns] __device__(
      size_type index) {
      col_desc[index].parent_column = parent_col_view.begin() + index;
      column_device_view col        = parent_col_view.column(index);
      // traverse till leaf column
      while (col.type().id() == type_id::LIST || col.type().id() == type_id::STRUCT) {
        auto const child = (col.type().id() == type_id::LIST)
                             ? col.child(lists_column_view::child_column_index)
                             : col.child(0);
        // stop early if writing a byte array
        if (col_desc[index].stats_dtype == dtype_byte_array &&
            child.type().id() == type_id::UINT8) {
          break;
        }
        col = child;
      }
      // Store leaf_column to device storage
      column_device_view* leaf_col_ptr = leaf_columns.data() + index;
      *leaf_col_ptr                    = col;
      col_desc[index].leaf_column      = leaf_col_ptr;
    });

  return leaf_column_views;
}

}  // namespace io
}  // namespace cudf
