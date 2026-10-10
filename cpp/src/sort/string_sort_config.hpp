/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <charconv>
#include <cstdlib>
#include <string_view>

namespace cudf::detail {

/**
 * @brief Experimental single-column string sorting implementations.
 */
enum class string_sort_algorithm { PREFIX, SEGMENTED, SEGMENTED_RLE };

struct segmented_string_sort_config {
  int lexic_precision{1};
  int radix_run_min{512};
  bool eliminate_exact_duplicates{false};
  bool trace{false};
};

[[nodiscard]] inline int parse_integer_setting(char const* value,
                                               int minimum,
                                               int maximum,
                                               int fallback)
{
  if (value == nullptr) { return fallback; }
  auto const text = std::string_view{value};
  if (text.empty()) { return fallback; }
  auto parsed             = int{};
  auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), parsed);
  return error == std::errc{} && end == text.data() + text.size() && parsed >= minimum &&
             parsed <= maximum
           ? parsed
           : fallback;
}

[[nodiscard]] inline segmented_string_sort_config parse_segmented_string_sort_config(
  string_sort_algorithm algorithm,
  char const* lexic_precision,
  char const* radix_run_min,
  char const* trace)
{
  auto config                       = segmented_string_sort_config{};
  config.lexic_precision            = parse_integer_setting(lexic_precision, 1, 255, 1);
  config.radix_run_min              = parse_integer_setting(radix_run_min, 2, 1 << 20, 512);
  config.eliminate_exact_duplicates = algorithm == string_sort_algorithm::SEGMENTED_RLE;
  config.trace                      = parse_integer_setting(trace, 0, 1, 0) != 0;
  return config;
}

/**
 * @brief Parses `LIBCUDF_STRING_SORT_ALGORITHM`.
 *
 * Unset and invalid values select the production prefix implementation.
 */
[[nodiscard]] inline string_sort_algorithm parse_string_sort_algorithm(char const* value)
{
  if (value == nullptr) { return string_sort_algorithm::PREFIX; }
  auto const setting = std::string_view{value};
  if (setting == "1") { return string_sort_algorithm::SEGMENTED; }
  if (setting == "2") { return string_sort_algorithm::SEGMENTED_RLE; }
  return string_sort_algorithm::PREFIX;
}

/**
 * @brief Returns the process-configured experimental string-sort implementation.
 *
 * The environment is read once. Applications must set it before the first libcudf sort call.
 */
[[nodiscard]] inline string_sort_algorithm configured_string_sort_algorithm()
{
  static auto const selected =
    parse_string_sort_algorithm(std::getenv("LIBCUDF_STRING_SORT_ALGORITHM"));
  return selected;
}

[[nodiscard]] inline segmented_string_sort_config const& configured_segmented_string_sort()
{
  static auto const config =
    parse_segmented_string_sort_config(configured_string_sort_algorithm(),
                                       std::getenv("LIBCUDF_SEGMENTED_STRING_SORT_LEXIC_PRECISION"),
                                       std::getenv("LIBCUDF_SEGMENTED_STRING_SORT_RADIX_RUN_MIN"),
                                       std::getenv("LIBCUDF_SEGMENTED_STRING_SORT_TRACE"));
  return config;
}

}  // namespace cudf::detail
