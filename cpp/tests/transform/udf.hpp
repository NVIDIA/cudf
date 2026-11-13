/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/udf.hpp>

#include <cudf_test_fragments.hpp>

#include <string_view>

namespace cudf::test {

/**
 * @brief Returns a named UDF from the shared embedded transform-test module.
 *
 * The fragment has static storage. The caller must keep the symbol alive during evaluation.
 */
template <auto Fragment = cudf_test_fragments::udfs_lto_ir>
inline cudf::transform_udf get_transform_udf(std::string_view symbol,
                                             cudf::null_aware is_null_aware = cudf::null_aware::NO,
                                             bool indexed                   = false)
{
  static cudf::lto_fragment const fragment{cudf::lto_binary_type::FATBIN,
                                           cudf_test_fragments::get(Fragment)};
  return {symbol, {&fragment, 1}, is_null_aware, indexed};
}

/// @brief Returns a named UDF from the PTX version of the shared transform-test module.
inline cudf::transform_udf get_ptx_transform_udf(std::string_view symbol)
{
  return get_transform_udf<cudf_test_fragments::udfs_ptx>(symbol);
}

}  // namespace cudf::test
