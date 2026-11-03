/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_device.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <string>

/**
 * @brief Create memory resource for libcudf functions
 */
cuda::mr::any_resource<cuda::mr::device_accessible> create_memory_resource(std::string const& name)
{
  rmm::mr::cuda_memory_resource cuda_mr{};
  if (name == "pool") {
    return rmm::mr::pool_memory_resource{cuda_mr, rmm::percent_of_free_device_memory(50)};
  }
  if (name == "cuda") { return cuda_mr; }
  return rmm::mr::cuda_async_memory_resource{};
}
