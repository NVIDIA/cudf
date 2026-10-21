/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/memory_resource.hpp>

#include <cuda/devices>
#include <cuda/memory_resource>
#include <cuda/std/execution>
#include <cuda/stream>

namespace cudf {
namespace detail {
/**
 * @brief Make an execution environment to pass to CUB calls.
 * @tparam Envs... Variadic parameter pack of additional environment parameter types.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Memory resources used for temporaries and returned results.
 * @param envs Additional environments.
 *
 * @returns A new execution environment.
 */
template <typename... Envs>
inline auto make_env(cuda::stream_ref stream, cudf::memory_resources mr, Envs... envs)
{
  auto stream_property = cuda::std::execution::prop{cuda::get_stream_t{}, stream};
  auto resource_property =
    cuda::std::execution::prop{cuda::mr::get_memory_resource_t{}, mr.get_temporary_mr()};
  return cuda::std::execution::env{stream_property, resource_property, envs...};
}

}  // namespace detail
}  // namespace cudf
