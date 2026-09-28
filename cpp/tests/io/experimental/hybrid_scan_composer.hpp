/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/aligned_resource_adaptor.hpp>

auto constexpr bloom_filter_alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;

/**
 * @brief Read parquet file with the hybrid scan reader
 *
 * @param datasource Input datasource
 * @param filter_expression Filter expression
 * @param payload_column_names List of paths of select payload column names, if any
 * @param case_sensitive_names Whether column names are case sensitive
 * @param stream CUDA stream for hybrid scan reader
 * @param mr Device memory resource
 * @param aligned_mr Device memory resource to allocate aligned memory for bloom filters
 *
 * @return Tuple of filter and payload tables
 */
std::tuple<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>> hybrid_scan(
  cudf::io::datasource& datasource,
  cudf::ast::operation const& filter_expression,
  std::optional<std::vector<std::string>> const& payload_column_names,
  bool case_sensitive_names,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  rmm::mr::aligned_resource_adaptor& aligned_mr);

/**
 * @brief Read parquet file with the hybrid scan reader
 *
 * @param datasource Input datasource
 * @param filter_expression Filter expression
 * @param payload_column_names List of paths of select payload column names, if any
 * @param case_sensitive_names Whether column names are case sensitive
 * @param stream CUDA stream for hybrid scan reader
 * @param mr Device memory resource
 * @param aligned_mr Device memory resource to allocate aligned memory for bloom filters
 *
 * @return Tuple of filter and payload tables
 */
std::tuple<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>> chunked_hybrid_scan(
  cudf::io::datasource& datasource,
  cudf::ast::operation const& filter_expression,
  std::optional<std::vector<std::string>> const& payload_column_names,
  bool case_sensitive_names,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  rmm::mr::aligned_resource_adaptor& aligned_mr);

/**
 * @brief Read parquet file with the hybrid scan reader in a single step
 *
 * @param datasource Input datasource
 * @param filter_expression Filter expression
 * @param column_names List of column names to read, if any
 * @param case_sensitive_names Whether column names are case sensitive
 * @param stream CUDA stream
 * @param mr Device memory resource
 *
 * @return Materialized table
 */
std::unique_ptr<cudf::table> hybrid_scan_single_step(
  cudf::io::datasource& datasource,
  cudf::ast::operation const& filter_expression,
  std::optional<std::vector<std::string>> const& column_names,
  bool case_sensitive_names,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Read parquet file with the hybrid scan reader in a single step using chunked reading
 *
 * @param datasource Input datasource
 * @param filter_expression Filter expression
 * @param column_names List of column names to read, if any
 * @param case_sensitive_names Whether column names are case sensitive
 * @param stream CUDA stream
 * @param mr Device memory resource
 *
 * @return Materialized table
 */
std::unique_ptr<cudf::table> chunked_hybrid_scan_single_step(
  cudf::io::datasource& datasource,
  cudf::ast::operation const& filter_expression,
  std::optional<std::vector<std::string>> const& column_names,
  bool case_sensitive_names,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);
