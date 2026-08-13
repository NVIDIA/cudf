/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/parquet_schema.hpp>

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace cudf::io::parquet::detail {

/**
 * @brief Gets the dot-separated path for a schema element.
 *
 * @param schema_tree The schema tree describing the file structure
 * @param schema_idx Index of the schema element
 * @return Dot-separated schema path from the root child to the schema element
 */
[[nodiscard]] std::string column_path_from_index(std::span<SchemaElement const> schema_tree,
                                                 int schema_idx);

/**
 * @brief Returns a normalized (lowercased) column name or path when case-insensitive matching is
 * enabled
 *
 * @param col_path The column name or path to normalize
 * @param case_sensitive_names Whether to normalize the column path case-insensitively
 *
 * @return The normalized column path
 */
[[nodiscard]] std::string normalize_column_path(std::string_view col_path,
                                                bool case_sensitive_names);

/**
 * @brief Compares two column paths with specified case sensitivity
 *
 * @param lhs The left-hand side column path
 * @param rhs The right-hand side column path
 * @param case_sensitive Whether to compare the column paths case-sensitively
 *
 * @return Boolean indicating if the column paths are equal
 */
[[nodiscard]] bool are_column_paths_equal(std::string_view lhs,
                                          std::string_view rhs,
                                          bool case_sensitive);

/**
 * @brief Transparent hash for column paths that honors a case-sensitivity policy.
 *
 * Hashes the column path according to `case_sensitive_names`, so that two paths that compare equal
 * under `are_column_paths_equal` also hash equally.
 */
struct column_path_hash {
  using is_transparent = void;
  bool case_sensitive_names{true};

  std::size_t operator()(std::string_view path) const;
};

/**
 * @brief Transparent equality for column paths that honors a case-sensitivity policy.
 *
 * Delegates to `are_column_paths_equal`, keeping the comparison consistent with `column_path_hash`.
 */
struct column_path_equal {
  using is_transparent = void;
  bool case_sensitive_names{true};

  bool operator()(std::string_view lhs, std::string_view rhs) const;
};

/**
 * @brief A set of column paths matched with a configurable case-sensitivity policy.
 */
using column_path_set = std::unordered_set<std::string, column_path_hash, column_path_equal>;

/**
 * @brief Constructs an empty `column_path_set` whose hash/equality use the given policy.
 *
 * @param case_sensitive_names Whether column-path matching is case-sensitive
 * @param bucket_hint Optional initial bucket count
 * @return An empty `column_path_set` using the requested policy
 */
[[nodiscard]] column_path_set make_column_path_set(bool case_sensitive_names,
                                                   std::size_t bucket_hint = 0);

/**
 * @brief A map keyed by column path matched with a configurable case-sensitivity policy.
 */
template <typename Value>
using column_path_map = std::unordered_map<std::string, Value, column_path_hash, column_path_equal>;

/**
 * @brief Constructs an empty `column_path_map` whose hash/equality use the given policy.
 *
 * @tparam Value Mapped value type
 * @param case_sensitive_names Whether column-path matching is case-sensitive
 * @param bucket_hint Optional initial bucket count
 * @return An empty `column_path_map` using the requested policy
 */
template <typename Value>
[[nodiscard]] column_path_map<Value> make_column_path_map(bool case_sensitive_names,
                                                          std::size_t bucket_hint = 0)
{
  return column_path_map<Value>(
    bucket_hint, column_path_hash{case_sensitive_names}, column_path_equal{case_sensitive_names});
}

}  // namespace cudf::io::parquet::detail
