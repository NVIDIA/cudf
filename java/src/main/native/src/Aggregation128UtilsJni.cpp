/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "aggregation128_utils.hpp"
#include "cudf_jni_apis.hpp"
#include "dtype_utils.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation128Utils_extractInt32Chunk(
  JNIEnv* env, jclass, jlong j_column_view, jint j_out_dtype, jint j_chunk_idx)
{
  JNI_NULL_CHECK(env, j_column_view, "column is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto cview = reinterpret_cast<cudf::column_view const*>(j_column_view);
    auto dtype = cudf::jni::make_data_type(j_out_dtype, 0);
    return cudf::jni::release_as_jlong(cudf::jni::extract_chunk32(*cview, dtype, j_chunk_idx));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Aggregation128Utils_combineInt64SumChunks(
  JNIEnv* env, jclass, jlong j_table_view, jint j_dtype, jint j_scale)
{
  JNI_NULL_CHECK(env, j_table_view, "table is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto tview = reinterpret_cast<cudf::table_view const*>(j_table_view);
    std::unique_ptr<cudf::table> result =
      cudf::jni::assemble128_from_sum(*tview, cudf::jni::make_data_type(j_dtype, j_scale));
    return cudf::jni::convert_table_with_row_count_for_return(env, result);
  }
  JNI_CATCH(env, 0);
}
}
