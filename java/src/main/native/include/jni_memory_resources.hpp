/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "jni_cccl_any_resource.hpp"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <jni.h>

namespace cudf::jni {

/**
 * @brief Convert the handle returned by `Cuda.Stream.getStream()` to a stream.
 */
inline cuda::stream_ref get_stream(jlong stream_handle)
{
  return cuda::stream_ref{reinterpret_cast<cudaStream_t>(stream_handle)};
}

/**
 * @brief Build the memory resources described by a Java `MemoryResources`.
 *
 * Each handle is an `RmmDeviceMemoryResource` handle, or 0 for the current device resource.
 */
inline cudf::memory_resources get_memory_resources(jlong output_handle, jlong temporary_handle)
{
  auto const resource_or_current = [](jlong handle) -> rmm::device_async_resource_ref {
    if (handle == 0) { return cudf::get_current_device_resource_ref(); }
    return get_resource(handle);
  };
  return cudf::memory_resources{resource_or_current(output_handle),
                                resource_or_current(temporary_handle)};
}

}  // namespace cudf::jni
