/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/logger.hpp>

#include <cstdlib>
#include <optional>
#include <sstream>
#include <string>

namespace cudf::detail {

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

template <typename T>
std::optional<T> getenv_optional(std::string_view env_var_name)
{
  auto const env_val = std::getenv(env_var_name.data());
  if (env_val == nullptr) { return std::nullopt; }

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
