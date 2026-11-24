/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "row_operator/jit/fragments/hash.cuh"

#include <cudf/hashing/detail/xxhash_32.cuh>

namespace cudf_abi {

using ::cudf::hashing::detail::XXHash_32;

#define CUDF_DEFINE_ROW_FUNCTION(Layout) \
  CUDF_DEFINE_ROW_HASH(xxhash_32, false, uint32_t, uint32_t, XXHash_32, Layout)

CUDF_FOR_EACH_COLUMN_LAYOUT(CUDF_DEFINE_ROW_FUNCTION)

#undef CUDF_DEFINE_ROW_FUNCTION
#undef CUDF_DEFINE_ROW_HASH

}  // namespace cudf_abi
