/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/errc.hpp>
#include <cudf/types.hpp>

namespace cudf {
extern "C" {

/** @brief Selects every row. */
__device__ errc cudf_all_true(bool* selected)
{
  *selected = true;
  return errc::SUCCESS;
}

/** @brief Selects every row with an indexed UDF signature. */
__device__ errc cudf_all_true_indexed(size_type, bool* selected) { return cudf_all_true(selected); }

/** @brief Selects every candidate pair with an indexed join UDF signature. */
__device__ errc cudf_all_true_join_indexed(size_type, size_type, bool* selected)
{
  return cudf_all_true(selected);
}

/** @brief Rejects every row. */
__device__ errc cudf_all_false(bool* selected)
{
  *selected = false;
  return errc::SUCCESS;
}

/** @brief Rejects every row with an indexed UDF signature. */
__device__ errc cudf_all_false_indexed(size_type, bool* selected)
{
  return cudf_all_false(selected);
}

/** @brief Rejects every candidate pair with an indexed join UDF signature. */
__device__ errc cudf_all_false_join_indexed(size_type, size_type, bool* selected)
{
  return cudf_all_false(selected);
}

}  // extern "C"
}  // namespace cudf
