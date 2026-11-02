/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/hashing.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace cudf {
namespace hashing::detail {

/**
 * @brief The default hash seed for algorithms exposing no seed parameter.
 *
 * This must be kept distinct from @p cudf::DEFAULT_HASH_SEED. The aim is to avoid situations
 * where hash-partitioning with the default hash seed correlates with the hash function
 * used internally. If that occurs, one can get catastrophic performance slowdowns because
 * only some small fraction of the available hashing buckets are used.
 *
 * Briefly, suppose that we have a hash key `h`, a partition count `P = 2^k (2 n + 1)` for
 * some `k` and `n`, and the hashing algorithm has a capacity `C`. There are two styles of
 * hashing in libcudf. The hashcsr algorithm ends up, if it also uses `h` as its hash key,
 * only filling at best `C / 2^k` of the available `C` buckets. The grouped aggregation
 * algorithms from cuco fill at best `C / gcd(P, C)` of the available `C` buckets. Both
 * scenarios lead to long linear probing chains.
 */
static constexpr uint32_t DEFAULT_ALGORITHM_HASH_SEED{0x68617368};  // 'hash'
static_assert(DEFAULT_ALGORITHM_HASH_SEED != cudf::DEFAULT_HASH_SEED,
              "Internal algorithm hash seed must be different from public default");

std::unique_ptr<column> murmurhash3_x86_32(table_view const& input,
                                           uint32_t seed,
                                           cuda::stream_ref,
                                           rmm::device_async_resource_ref mr);

std::unique_ptr<column> spark_murmurhash3_x86_32(table_view const& input,
                                                 uint32_t seed,
                                                 cuda::stream_ref,
                                                 rmm::device_async_resource_ref mr);

std::unique_ptr<table> murmurhash3_x64_128(table_view const& input,
                                           uint64_t seed,
                                           cuda::stream_ref,
                                           rmm::device_async_resource_ref mr);

std::unique_ptr<column> md5(table_view const& input,
                            cuda::stream_ref stream,
                            rmm::device_async_resource_ref mr);

std::unique_ptr<column> sha1(table_view const& input,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr);

std::unique_ptr<column> sha224(table_view const& input,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr);

std::unique_ptr<column> sha256(table_view const& input,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr);

std::unique_ptr<column> sha384(table_view const& input,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr);

std::unique_ptr<column> sha512(table_view const& input,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr);

std::unique_ptr<column> xxhash_32(table_view const& input,
                                  uint64_t seed,
                                  cuda::stream_ref,
                                  rmm::device_async_resource_ref mr);

std::unique_ptr<column> xxhash_64(table_view const& input,
                                  uint64_t seed,
                                  cuda::stream_ref,
                                  rmm::device_async_resource_ref mr);

/* SPDX-SnippetBegin
 * SPDX-SnippetCopyrightText: Copyright 2005-2014 Daniel James.
 * SPDX-License-Identifier: BSL-1.0
 *
 * Copyright 2005-2014 Daniel James.
 * Use, modification and distribution is subject to the Boost Software
 * License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 * https://www.boost.org/LICENSE_1_0.txt)
 */
/**
 * @brief Combines two hash values into a single hash value.
 *
 * Taken from the Boost hash_combine function.
 * https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
 *
 * @param lhs The first hash value
 * @param rhs The second hash value
 * @return Combined hash value
 */
CUDF_HOST_DEVICE constexpr uint32_t hash_combine(uint32_t lhs, uint32_t rhs)
{
  return lhs ^ (rhs + 0x9e37'79b9 + (lhs << 6) + (lhs >> 2));
}
// SPDX-SnippetEnd

/* SPDX-SnippetBegin
 * SPDX-SnippetCopyrightText: Copyright 2005-2014 Daniel James.
 * SPDX-License-Identifier: BSL-1.0
 *
 * Copyright 2005-2014 Daniel James.
 * Use, modification and distribution is subject to the Boost Software
 * License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 * https://www.boost.org/LICENSE_1_0.txt)
 */
/**
 * @brief Combines two 64-bit hash values into a single hash value.
 *
 * Adapted from Boost hash_combine function and modified for 64-bit.
 * https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
 *
 * @param lhs The first hash value
 * @param rhs The second hash value
 * @return Combined hash value
 */
CUDF_HOST_DEVICE constexpr uint64_t hash_combine(uint64_t lhs, uint64_t rhs)
{
  return lhs ^ (rhs + 0x9e37'79b9'7f4a'7c15 + (lhs << 6) + (lhs >> 2));
}
// SPDX-SnippetEnd

}  // namespace hashing::detail
}  // namespace cudf

// specialization of std::hash for cudf::data_type
namespace std {
template <>
struct hash<cudf::data_type> {
  std::size_t operator()(cudf::data_type const& type) const noexcept
  {
    return cudf::hashing::detail::hash_combine(
      std::hash<int32_t>{}(static_cast<int32_t>(type.id())), std::hash<int32_t>{}(type.scale()));
  }
};
}  // namespace std
