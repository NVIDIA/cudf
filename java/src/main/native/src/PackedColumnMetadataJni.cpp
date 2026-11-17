/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"

extern "C" {

JNIEXPORT jobject JNICALL Java_ai_rapids_cudf_PackedColumnMetadata_createMetadataDirectBuffer(
  JNIEnv* env, jclass, jlong j_metadata_ptr)
{
  JNI_NULL_CHECK(env, j_metadata_ptr, "metadata is null", nullptr);
  JNI_TRY
  {
    auto metadata = reinterpret_cast<std::vector<uint8_t>*>(j_metadata_ptr);
    return env->NewDirectByteBuffer(const_cast<uint8_t*>(metadata->data()), metadata->size());
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_PackedColumnMetadata_closeMetadata(JNIEnv* env,
                                                                              jclass,
                                                                              jlong j_metadata_ptr)
{
  JNI_NULL_CHECK(env, j_metadata_ptr, "metadata is null", );
  JNI_TRY { cudf::jni::safe_delete<std::vector<uint8_t>>(j_metadata_ptr); }
  JNI_CATCH(env, );
}

}  // extern "C"
