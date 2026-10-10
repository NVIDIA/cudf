/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <string>

/**
 * @brief Create memory resource for libcudf functions
 */
cuda::mr::any_resource<cuda::mr::device_accessible> create_memory_resource()
{
  return rmm::mr::cuda_async_memory_resource{};
}
