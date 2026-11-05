/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <ranges>
#include <string>
#include <string_view>

namespace cudf::detail {

/**
 * @brief Maps each element to a string and joins the results with a separator.
 *
 * Empty ranges produce an empty string. The mapper is called once per element, in range order.
 */
template <std::ranges::input_range Range, typename Mapper = std::identity>
std::string string_join(Range&& values, Mapper map = {}, std::string_view separator = "")
{
  std::string result;
  bool first = true;
  for (auto&& value : values) {
    if (!first) { result += separator; }
    first = false;
    result += std::invoke(map, value);
  }
  return result;
}

}  // namespace cudf::detail
