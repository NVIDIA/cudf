/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/hashing.hpp>
#include <cudf/types.hpp>

#include <cstdint>

extern "C" __device__ cudf::hash_value_type cudf_row_hash(int8_t const* columns,
                                                          cudf::size_type row_index,
                                                          cudf::hash_value_type seed);

__device__ inline cudf::hash_value_type cudf_row_hash(cudf::column_device_view const* columns,
                                                      cudf::size_type row_index,
                                                      cudf::hash_value_type seed)
{
  return cudf_row_hash(reinterpret_cast<int8_t const*>(columns), row_index, seed);
}
