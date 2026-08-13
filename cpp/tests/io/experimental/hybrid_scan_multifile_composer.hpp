/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

/**
 * @brief Read parquet sources with the hybrid scan multifile reader
 *
 * @param source_info Input source info containing one or more Parquet sources
 * @param filter_expression Filter expression
 * @param payload_column_names List of paths of select payload column names, if any
 * @param case_sensitive_names Whether column names are case sensitive
 * @param stream CUDA stream for hybrid scan reader
 * @param mr Device memory resource
 *
 * @return Tuple of filter and payload tables
 */
std::tuple<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>> hybrid_scan_multifile(
  cudf::io::source_info const& source_info,
  cudf::ast::operation const& filter_expression,
  std::optional<std::vector<std::string>> const& payload_column_names,
  bool case_sensitive_names,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Read parquet sources with the hybrid scan multifile reader in a single step
 *
 * @param source_info Input source info containing one or more Parquet sources
 * @param filter_expression Filter expression
 * @param column_names List of column names to read, if any
 * @param case_sensitive_names Whether column names are case sensitive
 * @param stream CUDA stream
 * @param mr Device memory resource
 *
 * @return Materialized table
 */
std::unique_ptr<cudf::table> hybrid_scan_multifile_single_step(
  cudf::io::source_info const& source_info,
  cudf::ast::operation const& filter_expression,
  std::optional<std::vector<std::string>> const& column_names,
  bool case_sensitive_names,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Read parquet sources with the hybrid scan multifile reader using chunked materialization
 *
 * Splits the filtered row groups into read passes via `construct_row_group_passes` and chunks the
 * filter and payload column materialization within each pass.
 *
 * @param source_info Input source info containing one or more Parquet sources
 * @param filter_expression Filter expression
 * @param payload_column_names List of paths of select payload column names, if any
 * @param case_sensitive_names Whether column names are case sensitive
 * @param stream CUDA stream for hybrid scan reader
 * @param mr Device memory resource
 *
 * @return Tuple of filter and payload tables
 */
std::tuple<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>>
chunked_hybrid_scan_multifile(cudf::io::source_info const& source_info,
                              cudf::ast::operation const& filter_expression,
                              std::optional<std::vector<std::string>> const& payload_column_names,
                              bool case_sensitive_names,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr);

/**
 * @brief Read parquet sources with the hybrid scan multifile reader in a single step using chunked
 * materialization
 *
 * @param source_info Input source info containing one or more Parquet sources
 * @param filter_expression Filter expression
 * @param column_names List of column names to read, if any
 * @param case_sensitive_names Whether column names are case sensitive
 * @param stream CUDA stream
 * @param mr Device memory resource
 *
 * @return Materialized table
 */
std::unique_ptr<cudf::table> chunked_hybrid_scan_multifile_single_step(
  cudf::io::source_info const& source_info,
  cudf::ast::operation const& filter_expression,
  std::optional<std::vector<std::string>> const& column_names,
  bool case_sensitive_names,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);
