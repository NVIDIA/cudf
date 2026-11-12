/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/udf.hpp>

#include <cudf_benchmark_fragments.hpp>

#include <string_view>

/**
 * @brief Returns a named UDF from the shared embedded benchmark module.
 *
 * The fragment has static storage. The caller must keep the symbol alive during evaluation.
 */
inline cudf::transform_udf get_benchmark_udf(std::string_view symbol)
{
  static auto const range =
    cudf_benchmark_fragments::file_ranges[cudf_benchmark_fragments::benchmark_udfs];
  static cudf::lto_fragment const fragment{
    cudf::lto_binary_type::FATBIN, cudf_benchmark_fragments::files.subspan(range[0], range[1])};
  return {symbol, {&fragment, 1}};
}
