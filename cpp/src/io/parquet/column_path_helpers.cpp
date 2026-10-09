/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "column_path_helpers.hpp"

#include <cudf/io/parquet_schema.hpp>
#include <cudf/logger.hpp>

#include <algorithm>
#include <cstddef>
#include <cwchar>
#include <functional>
#include <locale>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cudf::io::parquet::detail {

namespace {

/**
 * @brief Checks if a string is ASCII-only
 *
 * @param input The string to check
 * @return True if the string is ASCII-only, false otherwise
 */
bool is_ascii(std::string_view input)
{
  return std::ranges::all_of(input, [](unsigned char char_value) { return char_value < 0x80; });
}

/**
 * @brief Lowercases an ASCII character, leaving all other bytes unchanged
 *
 * Locale-independent so that it agrees with the fixed `C.UTF-8` mapping used for UTF-8 paths.
 *
 * @param char_value The character to lowercase
 * @return The lowercase character
 */
constexpr char to_lower_ascii(char char_value)
{
  return (char_value >= 'A' and char_value <= 'Z') ? static_cast<char>(char_value - 'A' + 'a')
                                                   : char_value;
}

/**
 * @brief Lowercases the ASCII characters of a string, leaving all other bytes unchanged
 *
 * @param input The string to lowercase
 * @return The lowercase string
 */
std::string to_lower_ascii(std::string_view input)
{
  std::string result(input.size(), '\0');
  std::ranges::transform(
    input, result.begin(), [](char char_value) { return to_lower_ascii(char_value); });
  return result;
}

/**
 * @brief Lowercases a UTF-8 string using the `C.UTF-8` locale (simple per-codepoint mapping)
 *
 * Falls back to the classic locale if `C.UTF-8` is unavailable. Returns the input unchanged if it
 * is not valid UTF-8. Output byte length may differ from input.
 *
 * @param input The string to lowercase
 * @return The lowercase string
 */
std::string to_lower_utf8(std::string_view input)
{
  static auto const locale = [] {
    try {
      return std::locale("C.UTF-8");
    } catch (std::runtime_error const&) {
      CUDF_LOG_WARN("C.UTF-8 locale not available, falling back to classic locale");
      return std::locale::classic();
    }
  }();
  auto const& ctype = std::use_facet<std::ctype<wchar_t>>(locale);

  // UTF-8 <-> UTF-32 converter available in every locale
  auto const& utf8 =
    std::use_facet<std::codecvt<char32_t, char8_t, std::mbstate_t>>(std::locale::classic());

  auto const* in_begin = reinterpret_cast<char8_t const*>(input.data());
  auto const* in_end   = in_begin + input.size();
  char8_t const* in_next{};
  std::u32string wide(input.size(), U'\0');
  char32_t* wide_next{};
  std::mbstate_t state{};
  if (utf8.in(
        state, in_begin, in_end, in_next, wide.data(), wide.data() + wide.size(), wide_next) !=
      std::codecvt_base::ok) {
    CUDF_LOG_WARN("Encountered invalid UTF-8 in column name or path, returning input unchanged");
    return std::string{input};
  }
  wide.resize(wide_next - wide.data());

  std::ranges::transform(wide, wide.begin(), [&](char32_t chr) {
    return static_cast<char32_t>(ctype.tolower(static_cast<wchar_t>(chr)));
  });

  std::string result(wide.size() * 4, '\0');
  char32_t const* wide_out_next{};
  char8_t* out_next{};
  auto* out_begin   = reinterpret_cast<char8_t*>(result.data());
  state             = {};
  auto const status = utf8.out(state,
                               wide.data(),
                               wide.data() + wide.size(),
                               wide_out_next,
                               out_begin,
                               out_begin + result.size(),
                               out_next);
  if (status != std::codecvt_base::ok) {
    CUDF_LOG_WARN("Failed to convert UTF-8 string to lowercase, returning input unchanged");
    return std::string{input};
  }
  result.resize(out_next - out_begin);
  return result;
}

}  // namespace

std::string column_path_from_index(std::span<SchemaElement const> schema_tree, int schema_idx)
{
  std::vector<std::string> path;
  for (auto idx = schema_idx; idx > 0; idx = schema_tree[idx].parent_idx) {
    path.push_back(schema_tree[idx].name);
  }

  return std::accumulate(
    path.rbegin() + 1, path.rend(), path.back(), [](auto path_so_far, auto const& elem_name) {
      return std::move(path_so_far) + "." + elem_name;
    });
}

std::string normalize_column_path(std::string_view col_path, bool case_sensitive_names)
{
  if (case_sensitive_names) { return std::string{col_path}; }
  return is_ascii(col_path) ? to_lower_ascii(col_path) : to_lower_utf8(col_path);
}

bool are_column_paths_equal(std::string_view lhs, std::string_view rhs, bool case_sensitive_names)
{
  if (case_sensitive_names) { return lhs == rhs; }
  // ASCII-only paths
  if (is_ascii(lhs) and is_ascii(rhs)) {
    return lhs.size() == rhs.size() and
           std::ranges::equal(lhs, rhs, [](char lhs_char, char rhs_char) {
             return to_lower_ascii(lhs_char) == to_lower_ascii(rhs_char);
           });
  }
  // UTF-8 paths
  return to_lower_utf8(lhs) == to_lower_utf8(rhs);
}

std::size_t column_path_hash::operator()(std::string_view path) const
{
  return std::hash<std::string>{}(normalize_column_path(path, case_sensitive_names));
}

bool column_path_equal::operator()(std::string_view lhs, std::string_view rhs) const
{
  return are_column_paths_equal(lhs, rhs, case_sensitive_names);
}

column_path_set make_column_path_set(bool case_sensitive_names, std::size_t bucket_hint)
{
  return column_path_set(
    bucket_hint, column_path_hash{case_sensitive_names}, column_path_equal{case_sensitive_names});
}

}  // namespace cudf::io::parquet::detail
