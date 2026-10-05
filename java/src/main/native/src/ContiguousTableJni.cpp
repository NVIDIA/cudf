/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"

namespace {

#define CONTIGUOUS_TABLE_CLASS                  "ai/rapids/cudf/ContiguousTable"
#define CONTIGUOUS_TABLE_FACTORY_SIG(param_sig) "(" param_sig ")L" CONTIGUOUS_TABLE_CLASS ";"

jclass Contiguous_table_jclass;
jmethodID From_packed_table_method;

#define GROUP_BY_RESULT_CLASS "ai/rapids/cudf/ContigSplitGroupByResult"
jclass Contig_split_group_by_result_jclass;
jfieldID Contig_split_group_by_result_groups_field;
jfieldID Contig_split_group_by_result_uniq_key_columns_field;

}  // anonymous namespace

namespace cudf {
namespace jni {

bool cache_contiguous_table_jni(JNIEnv* env)
{
  jclass cls = env->FindClass(CONTIGUOUS_TABLE_CLASS);
  if (cls == nullptr) { return false; }

  From_packed_table_method =
    env->GetStaticMethodID(cls, "fromPackedTable", CONTIGUOUS_TABLE_FACTORY_SIG("JJJJJ"));
  if (From_packed_table_method == nullptr) { return false; }

  // Convert local reference to global so it cannot be garbage collected.
  Contiguous_table_jclass = static_cast<jclass>(env->NewGlobalRef(cls));
  if (Contiguous_table_jclass == nullptr) { return false; }
  return true;
}

void release_contiguous_table_jni(JNIEnv* env)
{ Contiguous_table_jclass = cudf::jni::del_global_ref(env, Contiguous_table_jclass); }

bool cache_contig_split_group_by_result_jni(JNIEnv* env)
{
  jclass cls = env->FindClass(GROUP_BY_RESULT_CLASS);
  if (cls == nullptr) { return false; }

  Contig_split_group_by_result_groups_field =
    env->GetFieldID(cls, "groups", "[Lai/rapids/cudf/ContiguousTable;");
  if (Contig_split_group_by_result_groups_field == nullptr) { return false; }
  Contig_split_group_by_result_uniq_key_columns_field =
    env->GetFieldID(cls, "uniqKeyColumns", "[J");
  if (Contig_split_group_by_result_uniq_key_columns_field == nullptr) { return false; }

  // Convert local reference to global so it cannot be garbage collected.
  Contig_split_group_by_result_jclass = static_cast<jclass>(env->NewGlobalRef(cls));
  if (Contig_split_group_by_result_jclass == nullptr) { return false; }
  return true;
}

void release_contig_split_group_by_result_jni(JNIEnv* env)
{ Contig_split_group_by_result_jclass = del_global_ref(env, Contig_split_group_by_result_jclass); }

jobject contig_split_group_by_result_from(JNIEnv* env, jobjectArray& groups)
{
  jobject gbr = env->AllocObject(Contig_split_group_by_result_jclass);
  env->SetObjectField(gbr, Contig_split_group_by_result_groups_field, groups);
  return gbr;
}

jobject contig_split_group_by_result_from(JNIEnv* env,
                                          jobjectArray& groups,
                                          jlongArray& uniq_key_columns)
{
  jobject gbr = env->AllocObject(Contig_split_group_by_result_jclass);
  env->SetObjectField(gbr, Contig_split_group_by_result_groups_field, groups);
  env->SetObjectField(gbr, Contig_split_group_by_result_uniq_key_columns_field, uniq_key_columns);
  return gbr;
}

jobject contiguous_table_from(JNIEnv* env, cudf::packed_columns& split, long row_count)
{
  jlong metadata_address   = reinterpret_cast<jlong>(split.metadata.get());
  jlong data_address       = reinterpret_cast<jlong>(split.gpu_data->data());
  jlong data_size          = static_cast<jlong>(split.gpu_data->size());
  jlong rmm_buffer_address = reinterpret_cast<jlong>(split.gpu_data.get());

  jobject contig_table_obj = env->CallStaticObjectMethod(Contiguous_table_jclass,
                                                         From_packed_table_method,
                                                         metadata_address,
                                                         data_address,
                                                         data_size,
                                                         rmm_buffer_address,
                                                         row_count);

  if (contig_table_obj != nullptr) {
    split.metadata.release();
    split.gpu_data.release();
  }

  return contig_table_obj;
}

native_jobjectArray<jobject> contiguous_table_array(JNIEnv* env, jsize length)
{
  return native_jobjectArray<jobject>(
    env, env->NewObjectArray(length, Contiguous_table_jclass, nullptr));
}

}  // namespace jni
}  // namespace cudf

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ContiguousTable_createPackedMetadata(
  JNIEnv* env, jclass, jlong j_table, jlong j_buffer_addr, jlong j_buffer_length)
{
  JNI_NULL_CHECK(env, j_table, "input table is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto table        = reinterpret_cast<cudf::table_view const*>(j_table);
    auto data_addr    = reinterpret_cast<uint8_t const*>(j_buffer_addr);
    auto data_size    = static_cast<size_t>(j_buffer_length);
    auto metadata_ptr = new std::vector<uint8_t>(cudf::pack_metadata(*table, data_addr, data_size));
    return reinterpret_cast<jlong>(metadata_ptr);
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
