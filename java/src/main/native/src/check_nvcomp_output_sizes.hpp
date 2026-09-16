/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/exec_policy.hpp>

#include <cuda/stream>

namespace cudf {
namespace java {

/**
 * Check that the vector of expected uncompressed sizes matches the vector of actual compressed
 * sizes. Both vectors are assumed to be in device memory and contain num_chunks elements.
 */
bool check_nvcomp_output_sizes(std::size_t const* dev_uncompressed_sizes,
                               std::size_t const* dev_actual_uncompressed_sizes,
                               std::size_t num_chunks,
                               cuda::stream_ref stream);
}  // namespace java
}  // namespace cudf
