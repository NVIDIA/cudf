/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/device_buffer.hpp>

#include <memory>
#include <vector>

/**
 * @file
 * @brief Packed table and column types for serialization
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup copy_split
 * @{
 */

/**
 * @brief Column data in a serialized format
 *
 * Contains data from an array of columns in two contiguous buffers: one on host, which contains
 * table metadata and one on device which contains the table data.
 */
struct packed_columns {
  packed_columns()
    : metadata(std::make_unique<std::vector<uint8_t>>()),
      gpu_data(std::make_unique<rmm::device_buffer>())
  {
  }

  /**
   * @brief Construct a new packed columns object
   *
   * @param md Host-side metadata buffer
   * @param gd Device-side data buffer
   */
  packed_columns(std::unique_ptr<std::vector<uint8_t>>&& md,
                 std::unique_ptr<rmm::device_buffer>&& gd)
    : metadata(std::move(md)), gpu_data(std::move(gd))
  {
  }

  std::unique_ptr<std::vector<uint8_t>> metadata;  ///< Host-side metadata buffer
  std::unique_ptr<rmm::device_buffer> gpu_data;    ///< Device-side data buffer
};

/**
 * @brief A table with its data stored in a contiguous, serialized format
 *
 * Contains a `table_view` that points into the underlying `packed_columns` data. The table_view
 * and internal column_views in this struct are not owned by a top level cudf::table or
 * cudf::column. The backing memory and metadata is instead owned by the `data` field and is in
 * one contiguous block.
 *
 * The user is responsible for assuring that the `table` or any derived table_views do not outlive
 * the memory owned by `data`.
 */
struct packed_table {
  cudf::table_view table;  ///< Table view pointing into the packed data
  packed_columns data;     ///< Column data owned
};

/** @} */
}  // namespace CUDF_EXPORT cudf
