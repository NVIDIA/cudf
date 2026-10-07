/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"

extern "C" {
JNIEXPORT void JNICALL Java_ai_rapids_cudf_ChunkedPack_chunkedPackDelete(JNIEnv* env,
                                                                         jclass,
                                                                         jlong chunked_pack)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::safe_delete<cudf::chunked_pack>(chunked_pack);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ChunkedPack_chunkedPackGetTotalContiguousSize(
  JNIEnv* env, jclass, jlong chunked_pack)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto cs = reinterpret_cast<cudf::chunked_pack*>(chunked_pack);
    return cs->get_total_contiguous_size();
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jboolean JNICALL Java_ai_rapids_cudf_ChunkedPack_chunkedPackHasNext(JNIEnv* env,
                                                                              jclass,
                                                                              jlong chunked_pack)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto cs = reinterpret_cast<cudf::chunked_pack*>(chunked_pack);
    return cs->has_next();
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ChunkedPack_chunkedPackNext(
  JNIEnv* env, jclass, jlong chunked_pack, jlong user_ptr, jlong user_ptr_size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto cs               = reinterpret_cast<cudf::chunked_pack*>(chunked_pack);
    auto user_buffer_span = cudf::device_span<uint8_t>(reinterpret_cast<uint8_t*>(user_ptr),
                                                       static_cast<std::size_t>(user_ptr_size));
    return cs->next(user_buffer_span);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ChunkedPack_chunkedPackBuildMetadata(JNIEnv* env,
                                                                                 jclass,
                                                                                 jlong chunked_pack)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto cs = reinterpret_cast<cudf::chunked_pack*>(chunked_pack);
    std::unique_ptr<std::vector<uint8_t>> result = cs->build_metadata();
    return reinterpret_cast<jlong>(result.release());
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
