/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "jni_utils.hpp"

#include <cudf/contiguous_split.hpp>

namespace cudf::jni {

/**
 * @brief Detach all columns from the specified table, and pointers to them as an array.
 *
 * This function takes a table (presumably returned by some operation), and turns it into an
 * array of column* (as jlongs).
 * The lifetime of the columns is decoupled from that of the table, and is managed by the caller.
 *
 * @param env The JNI environment
 * @param table_result the table to convert for return
 * @param extra_columns columns not in the table that will be appended to the result.
 */
jlongArray convert_table_for_return(
  JNIEnv* env,
  std::unique_ptr<cudf::table>& table_result,
  std::vector<std::unique_ptr<cudf::column>>&& extra_columns = {});

/**
 * @copydoc convert_table_for_return(JNIEnv*, std::unique_ptr<cudf::table>&,
 *                                   std::vector<std::unique_ptr<cudf::column>>&&)
 */
jlongArray convert_table_for_return(
  JNIEnv* env,
  std::unique_ptr<cudf::table>&& table_result,
  std::vector<std::unique_ptr<cudf::column>>&& extra_columns = {});

//
// ContiguousTable APIs
//

bool cache_contiguous_table_jni(JNIEnv* env);

void release_contiguous_table_jni(JNIEnv* env);

jobject contiguous_table_from(JNIEnv* env, cudf::packed_columns& split, long row_count);

native_jobjectArray<jobject> contiguous_table_array(JNIEnv* env, jsize length);

/**
 * @brief Cache the JNI jclass and JNI jfield of Java `ContigSplitGroupByResult`
 *
 * @param env the JNI Env pointer
 * @return if success
 */
bool cache_contig_split_group_by_result_jni(JNIEnv* env);

/**
 * @brief Release the JNI jclass and JNI jfield of Java `ContigSplitGroupByResult`
 *
 * @param env the JNI Env pointer
 */
void release_contig_split_group_by_result_jni(JNIEnv* env);

/**
 * @brief Construct a Java `ContigSplitGroupByResult` from contiguous tables.
 *
 * @param env the JNI Env pointer
 * @param groups the contiguous tables
 * @return a Java `ContigSplitGroupByResult`
 */
jobject contig_split_group_by_result_from(JNIEnv* env, jobjectArray& groups);

/**
 * @brief Construct a Java `ContigSplitGroupByResult` from contiguous tables.
 *
 * @param env the JNI Env pointer
 * @param groups the contiguous tables
 * @param groups the contiguous tables
 * @return a Java `ContigSplitGroupByResult`
 */
jobject contig_split_group_by_result_from(JNIEnv* env,
                                          jobjectArray& groups,
                                          jlongArray& uniq_key_columns);

//
// HostMemoryBuffer APIs
//

/**
 * Allocate a HostMemoryBuffer
 */
jobject allocate_host_buffer(JNIEnv* env,
                             jlong amount,
                             jboolean prefer_pinned,
                             jobject host_memory_allocator);

/**
 * Get the address of a HostMemoryBuffer
 */
jlong get_host_buffer_address(JNIEnv* env, jobject buffer);

/**
 * Get the length of a HostMemoryBuffer
 */
jlong get_host_buffer_length(JNIEnv* env, jobject buffer);

// Get the JNI environment, attaching the current thread to the JVM if necessary. If the thread
// needs to be attached, the thread will automatically detach when the thread terminates.
JNIEnv* get_jni_env(JavaVM* jvm);

/** Set the device to use for cudf */
void set_cudf_device(int device);

/**
 * If the current thread has not set the CUDA device via Cuda.setDevice then this could
 * set the device, throw an exception, or do nothing depending on how the application has
 * configured it via Cuda.setAutoSetDeviceMode.
 */
void auto_set_device(JNIEnv* env);

//
// DataSource APIs
//

bool cache_data_source_jni(JNIEnv* env);

void release_data_source_jni(JNIEnv* env);

}  // namespace cudf::jni
