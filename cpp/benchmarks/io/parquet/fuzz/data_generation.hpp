/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common/generate_input.hpp"

#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <cstddef>
#include <exception>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief Draws one of `values`, each with the same probability
 *
 * @throws std::invalid_argument if `values` is empty
 */
template <typename T>
T pick(std::mt19937& rng, std::vector<T> const& values)
{
  CUDF_EXPECTS(not values.empty(), "Cannot pick from an empty list", std::invalid_argument);
  return values[std::uniform_int_distribution<std::size_t>(0, values.size() - 1)(rng)];
}

/**
 * @brief Draws true with `probability`
 */
inline bool coin(std::mt19937& rng, double probability)
{
  return std::bernoulli_distribution(probability)(rng);
}

/**
 * @brief Values that random tables draw their settings from
 *
 * Each setting with a list of values is drawn from it with `pick`, so repeating a value makes it
 * more likely. The number of columns and the list and struct depths are drawn uniformly from one
 * to their maxima.
 */
struct table_recipe {
  /// Row counts, which must be positive
  std::vector<cudf::size_type> num_rows{1, 7, 1'000, 20'000, 100'000, 300'000};
  int max_columns = 6;
  /// Types of the columns, and of the first member of struct columns
  std::vector<cudf::type_id> column_types{cudf::type_id::BOOL8,
                                          cudf::type_id::INT8,
                                          cudf::type_id::INT16,
                                          cudf::type_id::INT32,
                                          cudf::type_id::INT64,
                                          cudf::type_id::UINT8,
                                          cudf::type_id::UINT16,
                                          cudf::type_id::UINT32,
                                          cudf::type_id::UINT64,
                                          cudf::type_id::FLOAT32,
                                          cudf::type_id::FLOAT64,
                                          cudf::type_id::TIMESTAMP_MILLISECONDS,
                                          cudf::type_id::TIMESTAMP_MICROSECONDS,
                                          cudf::type_id::DECIMAL32,
                                          cudf::type_id::DECIMAL64,
                                          cudf::type_id::DECIMAL128,
                                          cudf::type_id::STRING,
                                          cudf::type_id::STRING,
                                          cudf::type_id::STRING,
                                          cudf::type_id::LIST,
                                          cudf::type_id::STRUCT};
  /// Probabilities of a null row, where `std::nullopt` means no null masks
  std::vector<std::optional<double>> null_probabilities{std::nullopt, 0.0, 0.01, 0.3, 0.95};
  /// Upper bounds on the number of distinct values of each column, where 0 means no bound
  std::vector<cudf::size_type> cardinalities{0, 50, 5'000};
  std::vector<cudf::size_type> run_lengths{1, 4, 32};  ///< Average lengths of runs of a value
  std::vector<int> string_lengths{4, 32, 200};         ///< Upper bounds of the string lengths
  int max_list_depth = 3;
  /// Types of the leaves of list columns
  std::vector<cudf::type_id> list_types{
    cudf::type_id::INT32, cudf::type_id::STRING, cudf::type_id::FLOAT64, cudf::type_id::STRUCT};
  std::vector<int> list_lengths{4, 64, 2'000};  ///< Upper bounds of the list lengths
  /// A table whose lists would have more leaf elements than this uses shorter `list_lengths`
  double max_list_elements = 100'000'000;
  int max_struct_depth     = 3;
  /// Probability of nesting structs `deep_struct_depth` deep instead
  double deep_struct_probability = 0.15;
  int deep_struct_depth          = 11;
  /// Members of struct columns after the first one, which has a type from `column_types` (and is
  /// left out when that type is nested)
  std::vector<cudf::type_id> struct_member_types{cudf::type_id::INT32, cudf::type_id::STRING};
};

/**
 * @brief Settings of a random table, for `create_random_table`
 */
struct table_settings {
  cudf::size_type num_rows;
  std::vector<cudf::type_id> types;
  data_profile profile;     ///< Distributions of the values
  std::string description;  ///< All the settings, for failure messages
};

/**
 * @brief Draws the settings of a random table from `recipe`
 *
 * @throws std::invalid_argument if a list of `recipe` is empty, a row count is not positive or a
 * maximum is less than one
 */
table_settings pick_table_settings(table_recipe const& recipe, std::mt19937& rng);

/**
 * @brief Encodings that the Parquet writer supports for top-level columns of each type, including
 * the default encoding
 */
std::map<cudf::type_id, std::vector<cudf::io::column_encoding>> supported_column_encodings();

/**
 * @brief Values that random Parquet writer settings draw from, the same way as in `table_recipe`
 */
struct writer_recipe {
  std::vector<cudf::io::dictionary_policy> dictionary_policies{
    cudf::io::dictionary_policy::NEVER,
    cudf::io::dictionary_policy::ADAPTIVE,
    cudf::io::dictionary_policy::ALWAYS};
  double v2_headers_probability = 0.5;  ///< Probability of V2 page headers
  std::vector<cudf::io::compression_type> compressions{cudf::io::compression_type::NONE,
                                                       cudf::io::compression_type::SNAPPY,
                                                       cudf::io::compression_type::ZSTD,
                                                       cudf::io::compression_type::GZIP,
                                                       cudf::io::compression_type::LZ4};
  std::vector<std::size_t> max_page_bytes{4 * 1024, 32 * 1024, 256 * 1024, 1024 * 1024};
  std::vector<cudf::size_type> max_page_rows{500, 5'000, 20'000};
  std::vector<cudf::size_type> max_page_fragment_rows{128, 1'000, 5'000};
  std::vector<cudf::size_type> row_group_rows{5'000, 50'000, 1'000'000};
  std::vector<std::size_t> row_group_bytes{1 << 20, 128 << 20};
  std::vector<std::size_t> max_dictionary_bytes{4 * 1024, 1024 * 1024};
  std::vector<cudf::io::statistics_freq> statistics{cudf::io::statistics_freq::STATISTICS_NONE,
                                                    cudf::io::statistics_freq::STATISTICS_ROWGROUP,
                                                    cudf::io::statistics_freq::STATISTICS_PAGE,
                                                    cudf::io::statistics_freq::STATISTICS_COLUMN};
  /// Encodings requested for top-level columns of each type. Other columns use the default.
  std::map<cudf::type_id, std::vector<cudf::io::column_encoding>> column_encodings =
    supported_column_encodings();
  /// Probability of splitting a table of at least `min_split_rows` rows between several files
  double split_probability       = 0.25;
  cudf::size_type min_split_rows = 1'000;
  std::vector<int> split_file_counts{2, 3};  ///< Numbers of files of a split table
};

/**
 * @brief Settings of random Parquet files
 */
struct writer_settings {
  cudf::io::table_input_metadata metadata;  ///< Requested column encodings
  cudf::io::dictionary_policy dictionary_policy;
  bool v2_headers;
  cudf::io::compression_type compression;
  std::size_t max_page_bytes;
  cudf::size_type max_page_rows;
  cudf::size_type max_page_fragment_rows;
  cudf::size_type row_group_rows;
  std::size_t row_group_bytes;
  std::size_t max_dictionary_bytes;
  cudf::io::statistics_freq statistics;
  int num_files;            ///< Number of files to split the rows between
  std::string description;  ///< All the settings, for failure messages
};

/**
 * @brief Draws the settings of random Parquet files of `table` from `recipe`
 *
 * @throws std::invalid_argument if a list of `recipe` is empty
 */
writer_settings pick_writer_settings(writer_recipe const& recipe,
                                     cudf::table_view const& table,
                                     std::mt19937& rng);

/**
 * @brief Writes `table` to `paths` with `settings`, splitting its rows evenly between the files
 */
void write_parquet_files(cudf::table_view const& table,
                         writer_settings const& settings,
                         std::vector<std::string> const& paths);

/**
 * @brief Whether `e`, thrown by the Parquet writer, is one of its deliberate limits rather than a
 * bug
 */
bool is_writer_limit(std::exception const& e);
