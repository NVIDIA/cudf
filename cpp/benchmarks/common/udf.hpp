/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/udf.hpp>

#include <cudf_benchmark_fragments.hpp>

#include <string_view>

inline cudf::transform_udf get_benchmark_udf(std::string_view symbol)
{
  static cudf::lto_fragment const fragment{
    cudf::lto_binary_type::FATBIN,
    cudf_benchmark_fragments::get(cudf_benchmark_fragments::benchmark_udfs)};
  return {symbol, {&fragment, 1}};
}
