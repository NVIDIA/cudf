/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "row_operator/jit/fragments/hash.cuh"

#include <cudf/hashing/detail/spark_murmurhash3.cuh>

namespace cudf_abi {

using ::cudf::hashing::detail::Spark_MurmurHash3_x86_32;

#define CUDF_DEFINE_ROW_FUNCTION(Layout) \
  CUDF_DEFINE_ROW_HASH(                  \
    spark_murmurhash3_x86_32, true, uint32_t, uint32_t, Spark_MurmurHash3_x86_32, Layout)

CUDF_FOR_EACH_COLUMN_LAYOUT(CUDF_DEFINE_ROW_FUNCTION)

#undef CUDF_DEFINE_ROW_FUNCTION
#undef CUDF_DEFINE_ROW_HASH

}  // namespace cudf_abi
