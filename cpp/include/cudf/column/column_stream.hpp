/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

/**
 * @file
 * @brief Column device-buffer stream (deallocation) helpers
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup column_stream
 * @{
 */

/**
 * @brief Rebinds column's device buffers to use `stream` for deallocation.
 *
 * Recursively updates the deallocation stream on the column's data buffers, null masks,
 * and all child columns. Does not copy device memory or launch kernels.
 *
 * @note This function does not insert CUDA cross-stream ordering. The caller must ensure a
 * happens-before relationship from every stream that may still have in-flight work touching
 * this column's device memory to @p stream (for example `cudf::detail::join_streams` after
 * parallel work on a stream pool, or explicit events / synchronization).
 *
 * @note `column_view` instances (including nested child views) that already referenced @p col's
 * device data remain valid: no device memory is copied or relocated, so their pointers and
 * metadata still describe the same buffers now owned by the returned column.
 *
 * @param col Column to rebind; ownership is transferred from this rvalue
 * @param stream Stream used for future asynchronous deallocation of the buffers
 * @return Column with equivalent contents and rebinding applied
 */
[[nodiscard]] std::unique_ptr<column> rebind_stream(column&& col, rmm::cuda_stream_view stream);

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
