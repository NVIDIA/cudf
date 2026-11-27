/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/udf.hpp>

#include <cudf_test_fragments.hpp>

#include <string_view>

namespace cudf::test {

/// @brief Returns the shared embedded test module with static storage.
template <auto Fragment = cudf_test_fragments::udfs_lto_ir>
inline lto_fragments get_udf_fragments()
{
  static lto_fragment const fragment{lto_binary_type::FATBIN, cudf_test_fragments::get(Fragment)};
  return {&fragment, 1};
}

/// @brief Returns a named UDF from the shared embedded test module.
template <auto Fragment = cudf_test_fragments::udfs_lto_ir>
inline transform_udf get_udf(std::string_view symbol,
                             null_aware is_null_aware = null_aware::NO,
                             bool indexed             = false)
{
  return {symbol, get_udf_fragments<Fragment>(), is_null_aware, indexed};
}

/// @brief Returns a named UDF from the PTX version of the shared test module.
inline transform_udf get_ptx_udf(std::string_view symbol,
                                 null_aware is_null_aware = null_aware::NO,
                                 bool indexed             = false)
{
  return get_udf<cudf_test_fragments::udfs_ptx>(symbol, is_null_aware, indexed);
}

}  // namespace cudf::test
