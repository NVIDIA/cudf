/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/logger.hpp>
#include <cudf/utilities/export.hpp>

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

namespace cudf::detail {

/**
 * @brief Parses an unsigned decimal byte size with an optional SI or binary unit suffix.
 *
 * Bare integers and the suffix B mean bytes. Decimal suffixes kB (or KB), MB, GB, TB, PB, and EB
 * use powers of 1000. Binary suffixes KiB, MiB, GiB, TiB, PiB, and EiB use powers of 1024.
 * Suffixes are case-sensitive. Signs, whitespace, and fractional values are not accepted.
 *
 * @param input Size to parse
 * @param identifier Name of the option or environment variable used in error messages
 * @param non_zero Whether zero is rejected
 * @return Size in bytes
 * @throws std::invalid_argument if input is empty, malformed, has an unknown unit, overflows
 * uint64_t, or is zero when non_zero is true
 */
CUDF_EXPORT std::uint64_t parse_size(std::string_view input,
                                     std::string_view identifier,
                                     bool non_zero);

/**
 * @brief Returns the value of the environment variable, or a default value if the variable is not
 * present.
 */
template <typename T>
T getenv_or(std::string_view env_var_name, T default_val)
{
  auto const env_val = std::getenv(env_var_name.data());
  if (env_val != nullptr) {
    CUDF_LOG_INFO("Environment variable %.*s read as %s",
                  static_cast<int>(env_var_name.length()),
                  env_var_name.data(),
                  env_val);
  } else {
    std::stringstream ss;
    ss << default_val;
    CUDF_LOG_INFO("Environment variable %.*s is not set, using default value %s",
                  static_cast<int>(env_var_name.length()),
                  env_var_name.data(),
                  ss.str());
  }

  if (env_val == nullptr) { return std::move(default_val); }

  std::stringstream sstream(env_val);
  T converted_val;
  sstream >> converted_val;
  return converted_val;
}

/**
 * @brief Specialization of getenv_or for bool, to allow common "ON"/"OFF" string values.
 */
bool get_bool_env_or(std::string_view env_var_name, bool default_val);

}  // namespace cudf::detail
