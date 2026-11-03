/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <optional>

namespace CUDF_EXPORT cudf {

/**
 * @brief Set the rmm resource to be used for pinned memory allocations.
 *
 * @param mr The rmm resource to be used for pinned allocations
 * @return The previous resource that was in use
 */
rmm::host_device_async_resource_ref set_pinned_memory_resource(
  rmm::host_device_async_resource_ref mr);

/**
 * @brief Get the rmm resource being used for pinned memory allocations.
 *
 * @return The rmm resource used for pinned allocations
 */
rmm::host_device_async_resource_ref get_pinned_memory_resource();

/**
 * @brief Options to configure the default pinned memory resource
 */
struct pinned_mr_options {
  std::optional<size_t> pool_size;  ///< The size of the pool to use for the default pinned memory
                                    ///< resource. If not set, the default pool size is used.
};

/**
 * @brief Configure the size of the default pinned memory resource.
 *
 * @param opts Options to configure the default pinned memory resource
 * @return True if this call successfully configured the pinned memory resource, false if a
 * a resource was already configured.
 */
bool config_default_pinned_memory_resource(pinned_mr_options const& opts);

/**
 * @brief Set the threshold size for using kernels for pinned memory copies.
 *
 * @param threshold The threshold size in bytes. If the size of the copy is less than this
 * threshold, the copy will be done using kernels. If the size is greater than or equal to this
 * threshold, the copy will be done by calling a CUDA runtime API (cudaMemcpyAsync or
 * cudaMemcpyBatchAsync).
 */
void set_kernel_pinned_copy_threshold(size_t threshold);

/**
 * @brief Get the threshold size for using kernels for pinned memory copies.
 *
 * @return The threshold size in bytes.
 */
size_t get_kernel_pinned_copy_threshold();

/**
 * @brief Set the threshold size for allocating host memory as pinned memory.
 *
 * @param threshold The threshold size in bytes. If the size of the allocation is less or equal to
 * this threshold, the memory will be allocated as pinned memory. If the size is greater than this
 * threshold, the memory will be allocated as pageable memory.
 */
void set_allocate_host_as_pinned_threshold(size_t threshold);

/**
 * @brief Get the threshold size for allocating host memory as pinned memory.
 *
 * @return The threshold size in bytes.
 */
size_t get_allocate_host_as_pinned_threshold();

}  // namespace CUDF_EXPORT cudf
