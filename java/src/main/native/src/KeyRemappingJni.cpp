/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"

#include <cudf/join/key_remapping.hpp>

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_KeyRemapping_create(
  JNIEnv* env, jclass, jlong j_table, jboolean j_compare_nulls, jboolean j_compute_metrics)
{
  JNI_NULL_CHECK(env, j_table, "table handle is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto tview     = reinterpret_cast<cudf::table_view const*>(j_table);
    auto nulleq    = j_compare_nulls ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
    auto metrics   = j_compute_metrics ? cudf::compute_metrics::YES : cudf::compute_metrics::NO;
    auto remap_ptr = new cudf::key_remapping(*tview, nulleq, metrics);
    return reinterpret_cast<jlong>(remap_ptr);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_KeyRemapping_destroy(JNIEnv* env, jclass, jlong j_handle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto remap_ptr = reinterpret_cast<cudf::key_remapping*>(j_handle);
    delete remap_ptr;
  }
  JNI_CATCH(env, );
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_KeyRemapping_getDistinctCount(JNIEnv* env,
                                                                         jclass,
                                                                         jlong j_handle)
{
  JNI_NULL_CHECK(env, j_handle, "handle is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto remap_ptr = reinterpret_cast<cudf::key_remapping*>(j_handle);
    return static_cast<jint>(remap_ptr->get_distinct_count());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_KeyRemapping_getMaxDuplicateCount(JNIEnv* env,
                                                                             jclass,
                                                                             jlong j_handle)
{
  JNI_NULL_CHECK(env, j_handle, "handle is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto remap_ptr = reinterpret_cast<cudf::key_remapping*>(j_handle);
    return static_cast<jint>(remap_ptr->get_max_duplicate_count());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_KeyRemapping_remapBuildKeys(JNIEnv* env,
                                                                        jclass,
                                                                        jlong j_handle)
{
  JNI_NULL_CHECK(env, j_handle, "handle is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto remap_ptr = reinterpret_cast<cudf::key_remapping*>(j_handle);
    auto result    = remap_ptr->remap_right_keys();
    return cudf::jni::release_as_jlong(result);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_KeyRemapping_remapProbeKeys(JNIEnv* env,
                                                                        jclass,
                                                                        jlong j_handle,
                                                                        jlong j_keys_table)
{
  JNI_NULL_CHECK(env, j_handle, "handle is null", 0);
  JNI_NULL_CHECK(env, j_keys_table, "keys table is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto remap_ptr = reinterpret_cast<cudf::key_remapping*>(j_handle);
    auto keys_view = reinterpret_cast<cudf::table_view const*>(j_keys_table);
    auto result    = remap_ptr->remap_left_keys(*keys_view);
    return cudf::jni::release_as_jlong(result);
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
