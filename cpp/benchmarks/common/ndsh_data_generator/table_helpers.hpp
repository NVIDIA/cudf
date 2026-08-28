/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <vector>

namespace cudf::datagen {

/**
 * @brief Add a column of days to a column of timestamp_days
 *
 * @param timestamp_days The column of timestamp_days
 * @param days The column of days to add
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::column> add_calendrical_days(
  cudf::column_view const& timestamp_days,
  cudf::column_view const& days,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Perform a left join operation between two tables
 *
 * @param left_input The left table
 * @param right_input The right table
 * @param left_on The indices of the columns to join on in the left table
 * @param right_on The indices of the columns to join on in the right table
 * @param compare_nulls The null equality comparison
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 */
std::unique_ptr<cudf::table> perform_left_join(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Generate the `p_retailprice` column of the `part` table
 *
 * @param p_partkey The `p_partkey` column of the `part` table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
[[nodiscard]] std::unique_ptr<cudf::column> calculate_p_retailprice(
  cudf::column_view const& p_partkey,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Generate the `l_suppkey` column of the `lineitem` table
 *
 * @param l_partkey The `l_partkey` column of the `lineitem` table
 * @param scale_factor The scale factor to use
 * @param num_rows The number of rows in the `lineitem` table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
[[nodiscard]] std::unique_ptr<cudf::column> calculate_l_suppkey(
  cudf::column_view const& l_partkey,
  cudf::size_type scale_factor,
  cudf::size_type num_rows,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Generate the `ps_suppkey` column of the `partsupp` table
 *
 * @param ps_partkey The `ps_partkey` column of the `partsupp` table
 * @param scale_factor The scale factor to use
 * @param num_rows The number of rows in the `partsupp` table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
[[nodiscard]] std::unique_ptr<cudf::column> calculate_ps_suppkey(
  cudf::column_view const& ps_partkey,
  cudf::size_type scale_factor,
  cudf::size_type num_rows,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
/**
 * @brief Calculate the cardinality of the `lineitem` table
 *
 * @param o_rep_freqs The frequency of each `o_orderkey` value in the `lineitem` table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
[[nodiscard]] cudf::size_type calculate_l_cardinality(
  cudf::column_view const& o_rep_freqs,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
/**
 * @brief Calculate the charge column for the `lineitem` table
 *
 * @param extendedprice The `l_extendedprice` column
 * @param tax The `l_tax` column
 * @param discount The `l_discount` column
 * @param stream The CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
[[nodiscard]] std::unique_ptr<cudf::column> calculate_charge(
  cudf::column_view const& extendedprice,
  cudf::column_view const& tax,
  cudf::column_view const& discount,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Generate a column of random addresses according to TPC-H specification clause 4.2.2.7
 *
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
[[nodiscard]] std::unique_ptr<cudf::column> generate_address_column(
  cudf::size_type num_rows,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Generate a phone number column according to TPC-H specification clause 4.2.2.9
 *
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
[[nodiscard]] std::unique_ptr<cudf::column> generate_phone_column(
  cudf::size_type num_rows,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace cudf::datagen
