/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/utilities/kernel_error.hpp"

#include <cuda/atomic>

namespace cudf::io::orc::detail {

enum class decode_error : kernel_error::value_type {
  INVALID_DATA_STREAM  = 1 << 0,
  UNREPRESENTABLE_DATE = 1 << 1,
};

/**
 * @brief Record a decoding failure without a shared error count wrapping back to zero.
 *
 * The reader checks for any error after synchronizing the decoding stream. Columns and
 * stripes share the same status, so their combined number of failures is not row-count bounded.
 */
inline __device__ void set_decode_error(kernel_error::pointer status,
                                        decode_error kind = decode_error::INVALID_DATA_STREAM)
{
  cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_device> error{*status};
  error.fetch_or(static_cast<kernel_error::value_type>(kind), cuda::std::memory_order_relaxed);
}

}  // namespace cudf::io::orc::detail
