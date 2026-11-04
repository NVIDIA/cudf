/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"
#include "dtype_utils.hpp"
#include "error.hpp"
#include "jni_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/experimental/variant.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <jni.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

namespace {

std::unique_ptr<cudf::column> make_status_column(cudf::size_type size,
                                                 bool initialize_to_success,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  auto status_column = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::UINT8}, size, cudf::mask_state::UNALLOCATED, stream, mr);
  if (initialize_to_success && size > 0) {
    auto status_view = status_column->mutable_view();
    CUDF_CUDA_TRY(
      cudaMemsetAsync(status_view.data<uint8_t>(), 0, size * sizeof(uint8_t), stream.get()));
  }
  return status_column;
}

jlongArray release_variant_result(JNIEnv* env,
                                  std::unique_ptr<cudf::column> value,
                                  std::unique_ptr<cudf::column> status)
{
  cudf::jni::native_jlongArray result(env, 2);
  // Complete all fallible JNI allocation/acquisition before transferring either column.
  auto handles = result.data();
  handles[0]   = cudf::jni::release_as_jlong(value);
  handles[1]   = cudf::jni::release_as_jlong(status);
  return result.get_jArray();
}

}  // namespace

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
                                                         path,
                                                         std::nullopt,
                                                         cudf::get_default_stream(),
                                                         cudf::get_current_device_resource_ref()));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_VariantUtils_getVariantTypeId(JNIEnv* env,
                                                                          jclass,
                                                                          jlong value_bytes_handle)
{
  JNI_NULL_CHECK(env, value_bytes_handle, "value bytes column is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const& value_bytes = *reinterpret_cast<cudf::column_view const*>(value_bytes_handle);
    return cudf::jni::release_as_jlong(cudf::io::parquet::experimental::get_variant_type_id(
      value_bytes, cudf::get_default_stream(), cudf::get_current_device_resource_ref()));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_VariantUtils_castVariantValue(
  JNIEnv* env, jclass, jlong value_bytes_handle, jint cudf_type_id, jint scale)
{
  JNI_NULL_CHECK(env, value_bytes_handle, "value bytes column is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const& value_bytes = *reinterpret_cast<cudf::column_view const*>(value_bytes_handle);
    return cudf::jni::release_as_jlong(
      cudf::io::parquet::experimental::cast_variant(value_bytes,
                                                    cudf::jni::make_data_type(cudf_type_id, scale),
                                                    std::nullopt,
                                                    cudf::get_default_stream(),
                                                    cudf::get_current_device_resource_ref()));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_VariantUtils_extractVariantField(
  JNIEnv* env, jclass, jlong variant_struct_handle, jstring j_path, jint cudf_type_id, jint scale)
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
      path,
      cudf::jni::make_data_type(cudf_type_id, scale),
      std::nullopt,
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref()));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_VariantUtils_getVariantFieldValueWithStatus(
  JNIEnv* env, jclass, jlong variant_struct_handle, jstring j_path)
{
  JNI_NULL_CHECK(env, variant_struct_handle, "variant struct column is null", nullptr);
  JNI_NULL_CHECK(env, j_path, "path is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const stream          = cudf::get_default_stream();
    auto const mr              = cudf::get_current_device_resource_ref();
    auto const& variant_struct = *reinterpret_cast<cudf::column_view const*>(variant_struct_handle);
    cudf::jni::native_jstring path(env, j_path);
    auto status      = make_status_column(variant_struct.size(), false, stream, mr);
    auto status_view = status->mutable_view();
    auto value       = cudf::io::parquet::experimental::get_variant_field(
      variant_struct, path, status_view, stream, mr);
    return release_variant_result(env, std::move(value), std::move(status));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_VariantUtils_castVariantValueWithStatus(JNIEnv* env,
                                                            jclass,
                                                            jlong value_bytes_handle,
                                                            jint cudf_type_id,
                                                            jint scale,
                                                            jlong prior_status_handle)
{
  JNI_NULL_CHECK(env, value_bytes_handle, "value bytes column is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const stream       = cudf::get_default_stream();
    auto const mr           = cudf::get_current_device_resource_ref();
    auto const& value_bytes = *reinterpret_cast<cudf::column_view const*>(value_bytes_handle);

    std::unique_ptr<cudf::column> status;
    if (prior_status_handle == 0) {
      status = make_status_column(value_bytes.size(), true, stream, mr);
    } else {
      auto const& prior_status = *reinterpret_cast<cudf::column_view const*>(prior_status_handle);
      status                   = std::make_unique<cudf::column>(prior_status, stream, mr);
    }

    auto status_view = status->mutable_view();
    auto value       = cudf::io::parquet::experimental::cast_variant(
      value_bytes, cudf::jni::make_data_type(cudf_type_id, scale), status_view, stream, mr);
    return release_variant_result(env, std::move(value), std::move(status));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_VariantUtils_extractVariantFieldWithStatus(
  JNIEnv* env, jclass, jlong variant_struct_handle, jstring j_path, jint cudf_type_id, jint scale)
{
  JNI_NULL_CHECK(env, variant_struct_handle, "variant struct column is null", nullptr);
  JNI_NULL_CHECK(env, j_path, "path is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const stream          = cudf::get_default_stream();
    auto const mr              = cudf::get_current_device_resource_ref();
    auto const& variant_struct = *reinterpret_cast<cudf::column_view const*>(variant_struct_handle);
    cudf::jni::native_jstring path(env, j_path);
    auto status      = make_status_column(variant_struct.size(), false, stream, mr);
    auto status_view = status->mutable_view();
    auto value       = cudf::io::parquet::experimental::extract_variant_field(
      variant_struct,
      path,
      cudf::jni::make_data_type(cudf_type_id, scale),
      status_view,
      stream,
      mr);
    return release_variant_result(env, std::move(value), std::move(status));
  }
  JNI_CATCH(env, nullptr);
}

}  // extern "C"
