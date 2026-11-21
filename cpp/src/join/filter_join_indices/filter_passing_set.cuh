/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/types.hpp>

#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/static_set.cuh>
#include <cuda/functional>
#include <cuda/std/functional>

#include <cstddef>

namespace cudf::detail {

using filter_passing_hash_type           = cuda::hash<size_type, cuda::hash_algorithm::xxhash_32>;
using filter_passing_probing_scheme_type = cuco::double_hashing<1, filter_passing_hash_type>;

/// Set of the left row indices with at least one pair passing the filter predicate.
using filter_passing_set_type = cuco::static_set<size_type,
                                                 cuco::extent<std::size_t>,
                                                 cuda::thread_scope_device,
                                                 cuda::std::equal_to<size_type>,
                                                 filter_passing_probing_scheme_type,
                                                 rmm::mr::polymorphic_allocator<char>>;

}  // namespace cudf::detail
