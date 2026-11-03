/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <string>
#include <string_view>

namespace cudf::detail {

/**
 * @brief Joins an iterator range, optionally mapping each element before appending it.
 *
 * The mapper is invoked exactly once per element. Its result must be appendable to a
 * `std::string`. Empty ranges produce an empty string; separators appear only between elements,
 * including elements that map to empty strings. Supports single-pass iterators.
 *
 * @param first Iterator to the first element
 * @param last Sentinel marking the end of the range
 * @param separator Text inserted between adjacent elements
 * @param mapper Optional callable or member pointer; defaults to the identity function
 * @return The joined string
 */
template <typename Iterator, typename Sentinel, typename Mapper = std::identity>
std::string join_strings(Iterator first,
                         Sentinel last,
                         std::string_view separator,
                         Mapper mapper = {})
{
  std::string result;
  if (first == last) { return result; }
  result += std::invoke(mapper, *first);
  while (++first != last) {
    result += separator;
    result += std::invoke(mapper, *first);
  }
  return result;
}

}  // namespace cudf::detail
