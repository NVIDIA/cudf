/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/utilities/kernel_error.hpp"

#include <cuda/atomic>

namespace cudf::io::orc::detail {

enum class error_code : kernel_error::value_type {
  INVALID_DATA_STREAM  = 1 << 0,
  UNREPRESENTABLE_DATE = 1 << 1,
};

/**
 * @brief Record an encoding or decoding failure in a shared error bitmask.
 *
 * The caller checks for any error after synchronizing the stream. Columns and stripes
 * share the same status without counting individual failures.
 */
inline __device__ void set_error(kernel_error::pointer status,
                                 error_code kind = error_code::INVALID_DATA_STREAM)
{
  cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_device> error{*status};
  error.fetch_or(static_cast<kernel_error::value_type>(kind), cuda::std::memory_order_relaxed);
}

}  // namespace cudf::io::orc::detail
