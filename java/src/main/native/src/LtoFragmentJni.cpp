/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"

#include <cudf/udf.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>

extern "C" {

JNIEXPORT jbyteArray JNICALL
Java_ai_rapids_cudf_LtoFragment_compileCudaToLtoIrNative(JNIEnv* env,
                                                         jclass,
                                                         jstring j_source,
                                                         jobjectArray j_include_names,
                                                         jobjectArray j_includes,
                                                         jobjectArray j_include_paths,
                                                         jboolean j_use_pch,
                                                         jstring j_pch_key,
                                                         jstring j_std)
{
  JNI_NULL_CHECK(env, j_source, "CUDA source is null", nullptr);
  JNI_NULL_CHECK(env, j_include_names, "Header names are null", nullptr);
  JNI_NULL_CHECK(env, j_includes, "Header contents are null", nullptr);
  JNI_NULL_CHECK(env, j_include_paths, "Include paths are null", nullptr);
  JNI_NULL_CHECK(env, j_pch_key, "PCH key is null", nullptr);
  JNI_NULL_CHECK(env, j_std, "C++ standard is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstring source(env, j_source);
    cudf::jni::native_jstringArray include_names(env, j_include_names);
    cudf::jni::native_jstringArray includes(env, j_includes);
    cudf::jni::native_jstringArray include_paths(env, j_include_paths);
    cudf::jni::native_jstring pch_key(env, j_pch_key);
    cudf::jni::native_jstring standard(env, j_std);

    auto const compiled = cudf::compile_cuda_to_lto_ir(
      source.get(),
      {include_names.as_c_array(), static_cast<std::size_t>(include_names.size())},
      {includes.as_c_array(), static_cast<std::size_t>(includes.size())},
      {include_paths.as_c_array(), static_cast<std::size_t>(include_paths.size())},
      static_cast<bool>(j_use_pch),
      pch_key,
      standard);
    auto const fragment = compiled.view();
    CUDF_EXPECTS(
      fragment.data.size() <= static_cast<std::size_t>(std::numeric_limits<jsize>::max()),
      "Compiled fragment is too large for a Java byte array",
      std::overflow_error);
    cudf::jni::native_jbyteArray bytes{env,
                                       reinterpret_cast<jbyte const*>(fragment.data.data()),
                                       static_cast<jsize>(fragment.data.size())};
    return bytes.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

}  // extern "C"
