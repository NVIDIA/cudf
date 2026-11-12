/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/stream>

#include <cstdint>

namespace cudf_streaming::detail {

/**
 * @brief Set a single value in the device-accessible pointer @p data
 *
 * Performs `data[0] = value`.
 *
 * @param data Array to set value in.
 * @param value Value to set @p data to.
 * @param stream CUDA stream for kernel launches and memory operations.
 */
void set_value(std::uint64_t* data, std::uint64_t value, cuda::stream_ref stream);

/**
 * @brief Add the value in @p left into @p right.
 *
 * Performs `right[0] += left[0];`
 *
 * @param left Array to add from.
 * @param right Array to add into.
 * @param stream CUDA stream for kernel launches and memory operations.
 */
void add_values(std::uint64_t const* left, std::uint64_t* right, cuda::stream_ref stream);

}  // namespace cudf_streaming::detail
