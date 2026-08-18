/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/bloom_filter_policies.cuh>

#include <cstdint>

namespace cudf {

/**
 * @brief Policy describing the Apache Arrow Block-Split Bloom Filter layout.
 *
 * Uses cuco's `parametric_filter_policy` with the Apache Arrow layout: 256-bit blocks (8 x
 * `uint32_t`), 8 fingerprint bits per key, fully horizontal add (Theta=8), and fully vertical
 * contains (Phi=8). This layout is bit-compatible with Apache Arrow.
 *
 * @tparam Hash The hash function used to generate a hash for each key.
 */
template <typename Hash>
using arrow_filter_policy =
  cuco::parametric_filter_policy<Hash, std::uint32_t, 8, 8, 8, 1, 1, 8, false, false>;

}  // namespace cudf
