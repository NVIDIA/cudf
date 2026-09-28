/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <unordered_set>

/**
 * @file hybrid_scan_composer.hpp
 * @brief Hybrid scan composer function(s)
 */

/**
 * @brief Enum to represent the currently used hybrid scan filters
 */
enum class hybrid_scan_filter_type : uint8_t {
  ROW_GROUPS_WITH_STATS         = 0,
  ROW_GROUPS_WITH_DICT_PAGES    = 1,
  ROW_GROUPS_WITH_BLOOM_FILTERS = 2,
  // FILTER_COLUMN_PAGES_WITH_PAGE_INDEX = 3,
  // PAYLOAD_COLUMN_PAGES_WITH_ROW_MASK  = 4,
};

/**
 * @brief Read parquet file with the hybrid scan reader
 *
 * @param io_source Parquet reader options
 * @param filters Set of hybrid scan filters to apply
 * @param stream CUDA stream
 * @param mr Device memory resource
 *
 * @return Unique pointer to the read table
 */
std::unique_ptr<cudf::table> hybrid_scan(cudf::io::parquet_reader_options const& options,
                                         std::unordered_set<hybrid_scan_filter_type> const& filters,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr);
