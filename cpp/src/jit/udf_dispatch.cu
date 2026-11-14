/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/udf_abi.cuh"

namespace cudf_udf_abi {

/** @brief Selects all rows as true. */
__device__ inline cudf::errc cudf_all_true(bool* selected)
{
  *selected = true;
  return cudf::errc{};
}

/** @brief Indexed overload selecting every row. */
__device__ inline cudf::errc cudf_all_true(cudf::size_type, bool* selected)
{
  return cudf_all_true(selected);
}

}  // namespace cudf_udf_abi

#pragma nv_hdrstop  // Only callback symbols and physical signatures vary per dispatcher.

namespace cudf_udf_abi {
#include <cudf/detail/udf_dispatch.cuh>
}  // namespace cudf_udf_abi
