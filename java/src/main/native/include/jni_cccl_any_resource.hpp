/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/memory_resource>

#include <jni.h>

namespace cudf::jni {

inline jlong make_jni_resource(cuda::mr::any_device_resource resource)
{
  return reinterpret_cast<jlong>(new cuda::mr::any_device_resource(std::move(resource)));
}

inline cuda::mr::any_device_resource& get_resource(jlong handle)
{
  return *reinterpret_cast<cuda::mr::any_device_resource*>(handle);
}

inline void delete_jni_resource(jlong handle)
{
  delete reinterpret_cast<cuda::mr::any_device_resource*>(handle);
}

}  // namespace cudf::jni
