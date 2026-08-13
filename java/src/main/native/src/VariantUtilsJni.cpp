/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"

#include <cudf/io/experimental/variant.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_VariantUtils_getVariantFieldValue(
  JNIEnv* env, jclass, jlong variant_struct_handle, jstring j_path)
{
  JNI_NULL_CHECK(env, variant_struct_handle, "variant struct column is null", 0);
  JNI_NULL_CHECK(env, j_path, "path is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const& variant_struct = *reinterpret_cast<cudf::column_view const*>(variant_struct_handle);
    cudf::jni::native_jstring path(env, j_path);
    return cudf::jni::release_as_jlong(
      cudf::io::parquet::experimental::get_variant_field(variant_struct,
                                                         path.get(),
                                                         cudf::get_default_stream(),
                                                         cudf::get_current_device_resource_ref()));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_VariantUtils_castVariantValue(JNIEnv* env,
                                                                          jclass,
                                                                          jlong value_bytes_handle,
                                                                          jint cudf_type_id)
{
  JNI_NULL_CHECK(env, value_bytes_handle, "value bytes column is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const& value_bytes = *reinterpret_cast<cudf::column_view const*>(value_bytes_handle);
    return cudf::jni::release_as_jlong(cudf::io::parquet::experimental::cast_variant(
      value_bytes,
      cudf::data_type{static_cast<cudf::type_id>(cudf_type_id)},
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref()));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_VariantUtils_extractVariantField(
  JNIEnv* env, jclass, jlong variant_struct_handle, jstring j_path, jint cudf_type_id)
{
  JNI_NULL_CHECK(env, variant_struct_handle, "variant struct column is null", 0);
  JNI_NULL_CHECK(env, j_path, "path is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const& variant_struct = *reinterpret_cast<cudf::column_view const*>(variant_struct_handle);
    cudf::jni::native_jstring path(env, j_path);
    return cudf::jni::release_as_jlong(cudf::io::parquet::experimental::extract_variant_field(
      variant_struct,
      path.get(),
      cudf::data_type{static_cast<cudf::type_id>(cudf_type_id)},
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref()));
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
