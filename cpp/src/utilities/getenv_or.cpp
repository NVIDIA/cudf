/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/getenv_or.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/numeric>

#include <charconv>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace cudf::detail {

std::uint64_t parse_size(std::string_view input, std::string_view identifier, bool non_zero)
{
  CUDF_EXPECTS(
    !input.empty(), std::format("{} must not be empty", identifier), std::invalid_argument);

  std::uint64_t size{};
  auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), size);

  CUDF_EXPECTS(error == std::errc{},
               std::format("{} must start with an unsigned integer representable as uint64. "
                           "Received '{}'",
                           identifier,
                           input),
               std::invalid_argument);
  CUDF_EXPECTS(!non_zero || size != 0,
               std::format("{} must be nonzero. Received '{}'", identifier, input),
               std::invalid_argument);

  constexpr struct {
    std::string_view name;
    std::uint64_t bytes;
  } units[] = {{"", 1},
               {"B", 1},
               {"kB", 1'000},
               {"KB", 1'000},
               {"MB", 1'000'000},
               {"GB", 1'000'000'000},
               {"TB", 1'000'000'000'000},
               {"PB", 1'000'000'000'000'000},
               {"EB", 1'000'000'000'000'000'000},
               {"KiB", std::uint64_t{1} << 10},
               {"MiB", std::uint64_t{1} << 20},
               {"GiB", std::uint64_t{1} << 30},
               {"TiB", std::uint64_t{1} << 40},
               {"PiB", std::uint64_t{1} << 50},
               {"EiB", std::uint64_t{1} << 60}};

  auto suffix = input.substr(end - input.data());

  for (auto& unit : units) {
    if (suffix != unit.name) { continue; }
    auto size_bytes = cuda::mul_overflow(size, unit.bytes);
    CUDF_EXPECTS(
      !size_bytes.overflow,
      std::format(
        "{} overflows uint64_t when converted to bytes. Received '{}'", identifier, input),
      std::invalid_argument);
    return size_bytes.value;
  }

  CUDF_FAIL(std::format("{} has unknown byte-size unit '{}' in '{}'", identifier, suffix, input),
            std::invalid_argument);
}

bool get_bool_env_or(std::string_view env_var_name, bool default_val)
{
  auto val = getenv_or(env_var_name, default_val ? std::string{"ON"} : std::string{"OFF"});
  return val == "ON" || val == "on" || val == "1" || val == "true" || val == "TRUE" ||
         val == "True";
}

}  // namespace cudf::detail
