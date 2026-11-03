/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda/memory_resource>

#include <rapidsmpf/error.hpp>

#include <string>

/**
 * @brief Create a RMM memory resource.
 *
 * @param name The name of the resource:
 *  - `cuda`: use the default CUDA memory resource.
 *  - `async`: use a CUDA async memory resource.
 *  - `pool`: use a memory pool backed by a CUDA memory resource.
 *  - `managed`: use a CUDA managed memory resource.
 * @return An owning resource holding the created memory resource.
 */
inline cuda::mr::any_resource<cuda::mr::device_accessible> create_rmm_resource(
  std::string const& name)
{
  if (name == "cuda") {
    return rmm::mr::cuda_memory_resource{};
  } else if (name == "async") {
    return rmm::mr::cuda_async_memory_resource{};
  } else if (name == "managed") {
    return rmm::mr::managed_memory_resource{};
  } else if (name == "pool") {
    return rmm::mr::pool_memory_resource{rmm::mr::cuda_memory_resource{},
                                         rmm::percent_of_free_device_memory(80),
                                         rmm::percent_of_free_device_memory(80)};
  } else {
    RAPIDSMPF_FAIL("unknown RMM resource name: " + name);
  }
}
