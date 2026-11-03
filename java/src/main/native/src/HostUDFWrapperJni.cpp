/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"

#include <cudf/aggregation/host_udf.hpp>

extern "C" {

JNIEXPORT void JNICALL Java_ai_rapids_cudf_HostUDFWrapper_close(JNIEnv* env,
                                                                jclass class_object,
                                                                jlong ptr)
{
  JNI_TRY { cudf::jni::safe_delete<cudf::host_udf_base>(ptr); }
  JNI_CATCH(env, );
}

}  // extern "C"
