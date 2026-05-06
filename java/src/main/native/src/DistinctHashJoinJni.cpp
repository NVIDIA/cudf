/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"

#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/table/table_view.hpp>

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_DistinctHashJoin_create(
  JNIEnv* env, jclass, jlong j_build_keys, jboolean j_nulls_equal)
{
  JNI_NULL_CHECK(env, j_build_keys, "build keys table is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const build_keys = reinterpret_cast<cudf::table_view const*>(j_build_keys);
    auto const nulls_equal =
      j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;

    auto handle = std::make_unique<cudf::distinct_hash_join>(*build_keys, nulls_equal);
    return cudf::jni::release_as_jlong(handle);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_DistinctHashJoin_destroy(
  JNIEnv* env, jclass, jlong j_handle)
{
  JNI_NULL_CHECK(env, j_handle, "distinct hash join handle is null", );

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto handle = reinterpret_cast<cudf::distinct_hash_join*>(j_handle);
    delete handle;
  }
  JNI_CATCH(env, );
}

}  // extern "C"
