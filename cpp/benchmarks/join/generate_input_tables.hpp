/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <utility>
#include <vector>

/**
 * @brief Generates build and probe tables for join benchmarking with specific key columns
 * and payload columns.
 *
 * The function first creates a base table with distinct rows of size build_table_numrows /
 * multiplicity + 1 by passing cardinality as zero to the random table generator's profile. In order
 * to populate the build and probe tables, random row index gather maps are created which are used
 * to index the base table.
 *
 * The build table gather map has indices in [0 ... build_table_numrows / multiplicity - 1], with
 * some indices repeated according to the multiplicity specified.
 *
 * The probe table gather map is created based on selectivity fraction, 's' passed. This results
 * in 's' fraction of the probe table gather map having entries in [0 ... build_table_numrows /
 * multiplicity - 1] (keys that exist in the build table) and the remaining (1-s) fraction having
 * entries outside this range (keys that don't exist in the build table).
 *
 * After the key columns are created, payload columns are added to both tables. These payload
 * columns are simple sequences starting from 0.
 *
 * @param key_types Vector of cuDF data types used for key columns in both tables
 * @param build_table_numrows Number of rows in the build table (hash table source)
 * @param probe_table_numrows Number of rows in the probe table
 * @param num_payload_cols Number of non-key columns to add to each table
 * @param multiplicity Number of times each unique key appears in the build table
 * @param selectivity Fraction of keys in the probe table that match keys in the build table
 *
 * @tparam Nullable If true, columns have 30% probability of null values; if false, all values are
 * valid
 *
 * @return A pair of unique pointers to build and probe tables
 */
template <bool Nullable>
std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>> generate_input_tables(
  std::vector<cudf::type_id> const& key_types,
  cudf::size_type build_table_numrows,
  cudf::size_type probe_table_numrows,
  cudf::size_type num_payload_cols,
  int multiplicity,
  double selectivity);
