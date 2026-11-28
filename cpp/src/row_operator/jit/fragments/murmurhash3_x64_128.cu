/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "row_operator/jit/fragments/hash.cuh"

#include <cudf/hashing/detail/murmurhash3_x64_128.cuh>

namespace cudf_abi {

using ::cudf::hashing::detail::MurmurHash3_x64_128;

// Pack the first result word into the low bits and the second into the high bits.
template <typename Element>
struct murmurhash3_x64_128_hasher {
  uint64_t seed;

  __device__ uint128_t operator()(Element const& value) const
  {
    auto const result = MurmurHash3_x64_128<Element>{seed}(value);
    return (static_cast<uint128_t>(result[1]) << 64) | result[0];
  }
};

#define CUDF_DEFINE_ROW_FUNCTION(Layout) \
  CUDF_DEFINE_ROW_HASH(                  \
    murmurhash3_x64_128, false, uint128_t, uint64_t, murmurhash3_x64_128_hasher, Layout)

CUDF_FOR_EACH_COLUMN_LAYOUT(CUDF_DEFINE_ROW_FUNCTION)

#undef CUDF_DEFINE_ROW_FUNCTION
#undef CUDF_DEFINE_ROW_HASH

}  // namespace cudf_abi
