/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/getenv_or.hpp>

#include <string>
#include <string_view>

namespace cudf::detail {

bool get_bool_env_or(std::string_view env_var_name, bool default_val)
{
  auto val = getenv_or(env_var_name, default_val ? std::string{"ON"} : std::string{"OFF"});
  return val == "ON" || val == "on" || val == "1" || val == "true" || val == "TRUE" ||
         val == "True";
}

}  // namespace cudf::detail
