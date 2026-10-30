/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/cuio_common.hpp"

#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>

#include <rmm/device_uvector.hpp>

/**
 * @brief A class to represent a table with column names attached
 */
class table_with_names {
 public:
  table_with_names(std::unique_ptr<cudf::table> tbl, std::vector<std::string> col_names)
    : tbl(std::move(tbl)), col_names(col_names) {};
  /**
   * @brief Return the table view
   */
  [[nodiscard]] cudf::table_view table() const;
  /**
   * @brief Return the column view for a given column name
   *
   * @param col_name The name of the column
   */
  [[nodiscard]] cudf::column_view column(std::string const& col_name) const;
  /**
   * @param Return the column names of the table
   */
  [[nodiscard]] std::vector<std::string> const& column_names() const;
  /**
   * @brief Translate a column name to a column index
   *
   * @param col_name The name of the column
   */
  [[nodiscard]] cudf::size_type column_id(std::string const& col_name) const;
  /**
   * @brief Append a column to the table
   *
   * @param col The column to append
   * @param col_name The name of the appended column
   */
  table_with_names& append(std::unique_ptr<cudf::column>& col, std::string const& col_name);
  /**
   * @brief Select a subset of columns from the table
   *
   * @param col_names The names of the columns to select
   */
  [[nodiscard]] cudf::table_view select(std::vector<std::string> const& col_names) const;
  /**
   * @brief Write the table to a parquet file
   *
   * @param filepath The path to the parquet file
   */
  void to_parquet(std::string const& filepath) const;

 private:
  std::unique_ptr<cudf::table> tbl;
  std::vector<std::string> col_names;
};

enum class query_mode : int32_t { END_TO_END = 0, COMPUTE_ONLY = 1 };

[[nodiscard]] query_mode query_mode_from_string(std::string const& str);

/**
 * @brief Select and copy named columns from a table
 *
 * @param table The input table
 * @param columns The columns to copy
 */
[[nodiscard]] std::unique_ptr<table_with_names> apply_projection(
  std::unique_ptr<table_with_names> const& table, std::vector<std::string> const& columns);

/**
 * @brief Inner join two tables and gather the result
 *
 * @param left_input The left input table
 * @param right_input The right input table
 * @param left_on The columns to join on in the left table
 * @param right_on The columns to join on in the right table
 * @param compare_nulls The null equality policy
 */
[[nodiscard]] std::unique_ptr<cudf::table> join_and_gather(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls);

/**
 * @brief Apply an inner join operation to two tables
 *
 * @param left_input The left input table
 * @param right_input The right input table
 * @param left_on The columns to join on in the left table
 * @param right_on The columns to join on in the right table
 * @param compare_nulls The null equality policy
 */
[[nodiscard]] std::unique_ptr<table_with_names> apply_inner_join(
  std::unique_ptr<table_with_names> const& left_input,
  std::unique_ptr<table_with_names> const& right_input,
  std::vector<std::string> const& left_on,
  std::vector<std::string> const& right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL);

/**
 * @brief Apply a left join operation to two tables
 *
 * @param left_input The left input table
 * @param right_input The right input table
 * @param left_on The columns to join on in the left table
 * @param right_on The columns to join on in the right table
 * @param compare_nulls The null equality policy
 */
[[nodiscard]] std::unique_ptr<table_with_names> apply_left_join(
  std::unique_ptr<table_with_names> const& left_input,
  std::unique_ptr<table_with_names> const& right_input,
  std::vector<std::string> const& left_on,
  std::vector<std::string> const& right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL);

/**
 * @brief Apply a left semi join operation to two tables
 *
 * @param left_input The left input table
 * @param right_input The right input table
 * @param left_on The columns to join on in the left table
 * @param right_on The columns to join on in the right table
 * @param compare_nulls The null equality policy
 */
[[nodiscard]] std::unique_ptr<table_with_names> apply_left_semi_join(
  std::unique_ptr<table_with_names> const& left_input,
  std::unique_ptr<table_with_names> const& right_input,
  std::vector<std::string> const& left_on,
  std::vector<std::string> const& right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL);

/**
 * @brief Apply a left anti join operation to two tables
 *
 * @param left_input The left input table
 * @param right_input The right input table
 * @param left_on The columns to join on in the left table
 * @param right_on The columns to join on in the right table
 * @param compare_nulls The null equality policy
 */
[[nodiscard]] std::unique_ptr<table_with_names> apply_left_anti_join(
  std::unique_ptr<table_with_names> const& left_input,
  std::unique_ptr<table_with_names> const& right_input,
  std::vector<std::string> const& left_on,
  std::vector<std::string> const& right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL);

/**
 * @brief Apply a filter predicate to a table
 *
 * @param table The input table
 * @param predicate The filter predicate
 */
[[nodiscard]] std::unique_ptr<table_with_names> apply_filter(
  std::unique_ptr<table_with_names> const& table, cudf::ast::operation const& predicate);

/**
 * @brief Apply a boolean mask to a table
 *
 * @param table The input table
 * @param mask The boolean mask
 */
[[nodiscard]] std::unique_ptr<table_with_names> apply_mask(
  std::unique_ptr<table_with_names> const& table, std::unique_ptr<cudf::column> const& mask);

/**
 * @brief Remove duplicate rows from a table
 *
 * @param table The input table
 */
[[nodiscard]] std::unique_ptr<table_with_names> apply_distinct(
  std::unique_ptr<table_with_names> const& table);

/**
 * @brief Copy a range of rows from a table
 *
 * @param table The input table
 * @param begin The first row to copy
 * @param end One past the last row to copy
 */
[[nodiscard]] std::unique_ptr<table_with_names> apply_slice(
  std::unique_ptr<table_with_names> const& table, cudf::size_type begin, cudf::size_type end);

/**
 * Struct representing group by key columns, value columns, and the type of aggregations to perform
 * on the value columns
 */
struct groupby_context_t {
  std::vector<std::string> keys;
  std::unordered_map<std::string, std::vector<std::pair<cudf::aggregation::Kind, std::string>>>
    values;
};

/**
 * @brief Apply a groupby operation to a table
 *
 * @param table The input table
 * @param ctx The groupby context
 */
[[nodiscard]] std::unique_ptr<table_with_names> apply_groupby(
  std::unique_ptr<table_with_names> const& table, groupby_context_t const& ctx);

/**
 * @brief Apply an order by operation to a table
 *
 * @param table The input table
 * @param sort_keys The sort keys
 * @param sort_key_orders The sort key orders
 */
[[nodiscard]] std::unique_ptr<table_with_names> apply_orderby(
  std::unique_ptr<table_with_names> const& table,
  std::vector<std::string> const& sort_keys,
  std::vector<cudf::order> const& sort_key_orders);

/**
 * @brief Apply a reduction operation to a column
 *
 * @param column The input column
 * @param agg_kind The aggregation kind
 * @param col_name The name of the output column
 */
[[nodiscard]] std::unique_ptr<table_with_names> apply_reduction(
  cudf::column_view const& column,
  cudf::aggregation::Kind const& agg_kind,
  std::string const& col_name);

/**
 * @brief Read a parquet file into a table
 *
 * @param source_info The source of the parquet file
 * @param columns The columns to read
 * @param predicate The filter predicate to pushdown
 */
[[nodiscard]] std::unique_ptr<table_with_names> read_parquet(
  cudf::io::source_info const& source_info,
  std::vector<std::string> const& columns                = {},
  std::unique_ptr<cudf::ast::operation> const& predicate = nullptr);

/**
 * @brief Generate the `std::tm` structure from year, month, and day
 *
 * @param year The year
 * @param month The month
 * @param day The day
 */
std::tm make_tm(int year, int month, int day);

/**
 * @brief Calculate the number of days since the UNIX epoch
 *
 * @param year The year
 * @param month The month
 * @param day The day
 */
int32_t days_since_epoch(int year, int month, int day);

/**
 * @brief Write a `cudf::table` to a parquet cuio sink
 *
 * @param table The `cudf::table` to write
 * @param col_names The column names of the table
 * @param source The source sink pair to write the table to
 */
void write_to_parquet_device_buffer(std::unique_ptr<cudf::table> const& table,
                                    std::vector<std::string> const& col_names,
                                    cuio_source_sink_pair& source);

/**
 * @brief Generate NDS-H tables and write to parquet device buffers
 *
 * @param scale_factor The scale factor of NDS-H tables to generate
 * @param table_names The names of the tables to generate
 * @param sources The parquet data sources to populate
 */
void generate_parquet_data_sources(double scale_factor,
                                   std::vector<std::string> const& table_names,
                                   std::unordered_map<std::string, cuio_source_sink_pair>& sources);

[[nodiscard]] bool write_ndsh_results();

void write_ndsh_result(table_with_names const& result, std::string const& query_name);
