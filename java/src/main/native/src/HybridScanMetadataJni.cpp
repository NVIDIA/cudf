/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"
#include "hybrid_scan_jni_internal.hpp"
#include "jni_utils.hpp"

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/utilities/span.hpp>

#include <cstdint>
#include <memory>

using namespace cudf::jni::hybrid_scan;

extern "C" {

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_HybridScanMetadata_createFromFooter(JNIEnv* env,
                                                        jclass,
                                                        jlong footer_address,
                                                        jlong footer_length,
                                                        jobjectArray j_column_names,
                                                        jint time_unit_type_id)
{
  JNI_NULL_CHECK(env, footer_address, "footer address is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const len         = checked_size_t(env, footer_length, "footerLength");
    auto const opts        = build_options(env, j_column_names, time_unit_type_id);
    auto const* footer_ptr = reinterpret_cast<uint8_t const*>(footer_address);
    auto metadata          = std::make_unique<exp_pq::hybrid_scan_metadata>(
      cudf::host_span<uint8_t const>{footer_ptr, len}, opts);
    return reinterpret_cast<jlong>(metadata.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_HybridScanMetadata_parquetMetadataAddress(JNIEnv* env,
                                                                                      jclass,
                                                                                      jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", 0);
  JNI_TRY
  {
    auto const* metadata = reinterpret_cast<exp_pq::hybrid_scan_metadata const*>(handle);
    return reinterpret_cast<jlong>(&metadata->parquet_metadata());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_HybridScanMetadata_destroy(JNIEnv* env,
                                                                      jclass,
                                                                      jlong handle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::safe_delete<exp_pq::hybrid_scan_metadata>(handle);
  }
  JNI_CATCH(env, );
}

}  // extern "C"
