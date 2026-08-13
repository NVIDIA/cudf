/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "variant_path.hpp"

#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <charconv>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace cudf::io::parquet::experimental::detail {

namespace {

// Dot-notation field names accept any byte except the structural characters '.' and '['.
[[nodiscard]] constexpr bool is_name_char(char c) { return c != '.' && c != '['; }

// Reads a maximal run of name characters from the front of `tail`.
[[nodiscard]] std::string read_unquoted_name(std::string_view tail)
{
  std::size_t n = 0;
  while (n < tail.size() && is_name_char(tail[n])) {
    ++n;
  }
  return std::string{tail.substr(0, n)};
}

// Reads a bracket step "[<non-negative integer>]" from the front of `tail`.
// The returned token keeps its brackets (e.g. "[42]").
[[nodiscard]] std::string read_bracket_step(std::string_view tail)
{
  CUDF_EXPECTS(!tail.empty() && tail.front() == '[',
               "expected '[' to open variant path index",
               std::invalid_argument);

  // Consume the maximal run of decimal digits.
  std::size_t n = 1;
  while (n < tail.size() && tail[n] >= '0' && tail[n] <= '9') {
    ++n;
  }
  CUDF_EXPECTS(
    n != 1, "expected non-negative integer after '[' in variant path", std::invalid_argument);

  // Reject indices that cannot be a valid array position (don't fit in cudf::size_type)
  cudf::size_type index = 0;
  auto const result     = std::from_chars(tail.data() + 1, tail.data() + n, index);
  CUDF_EXPECTS(
    result.ec == std::errc{}, "variant path index is out of range", std::invalid_argument);

  CUDF_EXPECTS(n < tail.size() && tail[n] == ']',
               "expected ']' to close variant path index",
               std::invalid_argument);
  return std::string{tail.substr(0, n + 1)};  // include the closing ']'
}

}  // namespace

std::vector<std::string> parse_variant_path(std::string_view path)
{
  std::vector<std::string> steps;
  auto const len  = path.size();
  std::size_t pos = 0;

  // Optional leading '$'
  if (pos < len && path[pos] == '$') { ++pos; }

  bool first = true;
  while (pos < len) {
    char const c = path[pos];
    if (c == '[') {
      steps.emplace_back(read_bracket_step(path.substr(pos)));
    } else {
      if (c == '.') {
        ++pos;
        CUDF_EXPECTS(pos < len && is_name_char(path[pos]),
                     "trailing '.' with no field name",
                     std::invalid_argument);
      } else {
        // Neither a '.'/'[' step nor a valid leading name (e.g. a stray ']' or a name after a step)
        CUDF_EXPECTS(
          first && is_name_char(c), "unexpected character in variant path", std::invalid_argument);
      }
      steps.emplace_back(read_unquoted_name(path.substr(pos)));
    }
    pos += steps.back().size();
    first = false;
  }

  CUDF_EXPECTS(!steps.empty(), "variant path is empty", std::invalid_argument);

  return steps;
}

}  // namespace cudf::io::parquet::experimental::detail
