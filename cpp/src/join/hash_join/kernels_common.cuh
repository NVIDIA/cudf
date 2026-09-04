/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Custom hash-join probe kernels that give cudf direct control over kernel launches.
// Uses the cuco ref type for hash-table access (storage, probing scheme, predicate).

#pragma once

#include "join/join_common_utils.hpp"

#include <cudf/hashing.hpp>
#include <cudf/types.hpp>

#include <cuco/pair.cuh>

namespace cudf::detail {

/// The probe key type stored in the hash table: {hash_value, row_index}.
using probe_key_type = cuco::pair<hash_value_type, size_type>;

}  // namespace cudf::detail
