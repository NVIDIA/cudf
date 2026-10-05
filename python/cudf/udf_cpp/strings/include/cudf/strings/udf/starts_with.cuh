/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/strings/string_view.cuh>

namespace cudf {
namespace strings {
namespace udf {

/**
 * @brief Returns true if the beginning of the specified string
 * matches the given character array.
 *
 * @param dstr String to check
 * @param tgt Character array encoded in UTF-8
 * @param bytes Number of bytes to read from `tgt`
 * @return true if `tgt` matches the beginning of `dstr`
 */
__device__ inline bool starts_with(cudf::string_view const dstr,
                                   char const* tgt,
                                   cudf::size_type bytes)
{
  if (bytes > dstr.size_bytes()) { return false; }
  auto const start_str = cudf::string_view{dstr.data(), bytes};
  return start_str.compare(tgt, bytes) == 0;
}

/**
 * @brief Returns true if the beginning of the specified string
 * matches the given target string.
 *
 * @param dstr String to check
 * @param tgt String to match
 * @return true if `tgt` matches the beginning of `dstr`
 */
__device__ inline bool starts_with(cudf::string_view const dstr, cudf::string_view const& tgt)
{ return starts_with(dstr, tgt.data(), tgt.size_bytes()); }

/**
 * @brief Returns true if the end of the specified string
 * matches the given character array.
 *
 * @param dstr String to check
 * @param tgt Character array encoded in UTF-8
 * @param bytes Number of bytes to read from `tgt`
 * @return true if `tgt` matches the end of `dstr`
 */
__device__ inline bool ends_with(cudf::string_view const dstr,
                                 char const* tgt,
                                 cudf::size_type bytes)
{
  if (bytes > dstr.size_bytes()) { return false; }
  auto const end_str = cudf::string_view{dstr.data() + dstr.size_bytes() - bytes, bytes};
  return end_str.compare(tgt, bytes) == 0;
}

/**
 * @brief Returns true if the end of the specified string
 * matches the given target` string.
 *
 * @param dstr String to check
 * @param tgt String to match
 * @return true if `tgt` matches the end of `dstr`
 */
__device__ inline bool ends_with(cudf::string_view const dstr, cudf::string_view const& tgt)
{ return ends_with(dstr, tgt.data(), tgt.size_bytes()); }

}  // namespace udf
}  // namespace strings
}  // namespace cudf
