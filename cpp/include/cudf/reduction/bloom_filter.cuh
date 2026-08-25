/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/bloom_filter_policy.cuh>

#include <cstdint>

namespace cudf {

/**
 * @brief Policy describing the Apache Arrow Block-Split Bloom Filter layout.
 *
 * Uses cuco's `bloom_filter_policy` with the Apache Arrow layout: 256-bit blocks (8 x
 * `uint32_t`), 8 fingerprint bits per key, fully horizontal add (Theta=8), and fully vertical
 * contains (Phi=8). This layout is bit-compatible with Apache Arrow.
 *
 * @tparam Key The key type to generate a fingerprint for.
 * @tparam Hash The hash function used to generate a hash for each key.
 */
template <typename Key, typename Hash>
using arrow_bloom_filter_policy =
  cuco::bloom_filter_policy<Key, Hash, sizeof(std::uint32_t), 8, 8, 8, 1, 1, 8>;

/**
 * @brief Policy describing the Apache Arrow Block-Split Bloom Filter layout.
 *
 * @deprecated Deprecated in 26.10, to be removed in a future release. Use
 * `cudf::arrow_bloom_filter_policy` instead, which takes the key type explicitly rather than
 * deducing it from the hash functor.
 *
 * @tparam Hash The hash function used to generate a hash for each key.
 */
template <typename Hash>
using arrow_filter_policy [[deprecated("Use cudf::arrow_bloom_filter_policy<Key, Hash> instead")]] =
  cuco::parametric_filter_policy<Hash, std::uint32_t, 8, 8, 8, 1, 1, 8, false, false>;

}  // namespace cudf
