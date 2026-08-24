/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"
#include "hybrid_scan_jni_internal.hpp"
#include "jni_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>
#include <utility>

using namespace cudf::jni::hybrid_scan;

namespace {

inline exp_pq::use_data_page_mask to_data_page_mask(bool use_data_page_mask)
{
  return use_data_page_mask ? exp_pq::use_data_page_mask::YES : exp_pq::use_data_page_mask::NO;
}

}  // namespace

extern "C" {

// ----------------------------------------------------------------------
// Two-step materialize (filter + payload)
// ----------------------------------------------------------------------

// Returns: [row_mask_col_handle, filter_table_col0_handle, ..., filter_table_colN_handle]
JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_HybridScanReader_materializeFilterColumns(JNIEnv* env,
                                                              jclass,
                                                              jlong handle,
                                                              jintArray j_row_groups,
                                                              jlongArray j_addrs,
                                                              jlongArray j_lens,
                                                              jboolean use_page_level_pruning)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder   = make_row_group_span(env, j_row_groups);
    auto spans    = make_device_spans(env, j_addrs, j_lens);
    auto stream   = cudf::get_default_stream();
    auto mr       = cudf::get_current_device_resource_ref();
    // Build the owned row mask. Seed with page index stats when page-level pruning is enabled,
    // otherwise build an all-true mask.
    auto row_mask_col = use_page_level_pruning
                          ? wrapper->reader->build_row_mask_with_page_index_stats(
                              holder.span(), wrapper->options, stream, mr)
                          : wrapper->reader->build_all_true_row_mask(holder.span(), stream, mr);
    auto mut_view     = row_mask_col->mutable_view();
    auto mode         = to_data_page_mask(use_page_level_pruning);
    auto result       = wrapper->reader->materialize_filter_columns(
      holder.span(), spans, mut_view, mode, wrapper->options, stream, mr);
    // Pack: [row_mask_handle, table_col0, ..., table_colN]
    // Hold row_mask_col owned until after convert_table_for_return + the table-cols
    // SetLongArrayRegion succeed; if either throws, the unique_ptr's destructor frees the
    // row mask normally. Release only at the final SetLongArrayRegion, which has known-valid
    // bounds and is the last possible throw point.
    jsize n_table_cols = static_cast<jsize>(result.tbl->num_columns());
    jlongArray out     = env->NewLongArray(1 + n_table_cols);
    if (out == nullptr) { return nullptr; }
    auto table_handles = cudf::jni::convert_table_for_return(env, result.tbl);
    cudf::jni::native_jlongArray table_arr(env, table_handles);
    env->SetLongArrayRegion(out, 1, n_table_cols, table_arr.data());
    jlong row_mask_handle = cudf::jni::release_as_jlong(std::move(row_mask_col));
    env->SetLongArrayRegion(out, 0, 1, &row_mask_handle);
    return out;
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_HybridScanReader_materializePayloadColumns(JNIEnv* env,
                                                               jclass,
                                                               jlong handle,
                                                               jintArray j_row_groups,
                                                               jlongArray j_addrs,
                                                               jlongArray j_lens,
                                                               jlong row_mask_view_handle,
                                                               jboolean use_page_level_pruning)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_NULL_CHECK(env, row_mask_view_handle, "row mask view handle is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper  = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder    = make_row_group_span(env, j_row_groups);
    auto spans     = make_device_spans(env, j_addrs, j_lens);
    auto* row_mask = reinterpret_cast<cudf::column_view const*>(row_mask_view_handle);
    auto mode      = to_data_page_mask(use_page_level_pruning);
    auto result =
      wrapper->reader->materialize_payload_columns(holder.span(),
                                                   spans,
                                                   *row_mask,
                                                   mode,
                                                   wrapper->options,
                                                   cudf::get_default_stream(),
                                                   cudf::get_current_device_resource_ref());
    return cudf::jni::convert_table_for_return(env, result.tbl);
  }
  JNI_CATCH(env, nullptr);
}

// ----------------------------------------------------------------------
// One-shot materialize (all columns)
// ----------------------------------------------------------------------

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_HybridScanReader_materializeAllColumns(
  JNIEnv* env, jclass, jlong handle, jintArray j_row_groups, jlongArray j_addrs, jlongArray j_lens)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder   = make_row_group_span(env, j_row_groups);
    auto spans    = make_device_spans(env, j_addrs, j_lens);
    auto result   = wrapper->reader->materialize_all_columns(holder.span(),
                                                           spans,
                                                           wrapper->options,
                                                           cudf::get_default_stream(),
                                                           cudf::get_current_device_resource_ref());
    return cudf::jni::convert_table_for_return(env, result.tbl);
  }
  JNI_CATCH(env, nullptr);
}

// ----------------------------------------------------------------------
// Chunked materialize
// ----------------------------------------------------------------------

// Builds the row mask, sets up chunking state in the underlying reader, and
// stores the owned row mask in the wrapper. Subsequent materializeFilterColumnsChunk
// calls mutate this column in place. takeFilterRowMask transfers ownership of the
// mutated column out to Java.
JNIEXPORT void JNICALL
Java_ai_rapids_cudf_HybridScanReader_setupChunkingForFilterColumns(JNIEnv* env,
                                                                   jclass,
                                                                   jlong handle,
                                                                   jlong chunk_read_limit,
                                                                   jlong pass_read_limit,
                                                                   jintArray j_row_groups,
                                                                   jboolean use_page_level_pruning,
                                                                   jlongArray j_addrs,
                                                                   jlongArray j_lens)
{
  JNI_NULL_CHECK(env, handle, "handle is null", );
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    // Validate read limits up-front so we don't allocate the row mask column on the GPU
    // before throwing on bad input.
    auto const chunk_limit = checked_size_t(env, chunk_read_limit, "chunk_read_limit");
    auto const pass_limit  = checked_size_t(env, pass_read_limit, "pass_read_limit");
    auto* wrapper          = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder            = make_row_group_span(env, j_row_groups);
    auto spans             = make_device_spans(env, j_addrs, j_lens);
    auto stream            = cudf::get_default_stream();
    auto mr                = cudf::get_current_device_resource_ref();
    // Discard any prior chunked-filter row mask: starting a new chunked filter pipeline
    // implicitly ends the previous one.
    wrapper->chunked_filter_row_mask.reset();
    auto row_mask_col = use_page_level_pruning
                          ? wrapper->reader->build_row_mask_with_page_index_stats(
                              holder.span(), wrapper->options, stream, mr)
                          : wrapper->reader->build_all_true_row_mask(holder.span(), stream, mr);
    auto mode         = to_data_page_mask(use_page_level_pruning);
    wrapper->reader->setup_chunking_for_filter_columns(chunk_limit,
                                                       pass_limit,
                                                       holder.span(),
                                                       row_mask_col->view(),
                                                       mode,
                                                       spans,
                                                       wrapper->options,
                                                       stream,
                                                       mr);
    // Store the owned column for the rest of the chunked filter pipeline.
    wrapper->chunked_filter_row_mask = std::move(row_mask_col);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_HybridScanReader_materializeFilterColumnsChunk(
  JNIEnv* env, jclass, jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    if (wrapper->chunked_filter_row_mask == nullptr) {
      JNI_THROW_NEW(env,
                    cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                    "no active chunked-filter pipeline; call setupChunkingForFilterColumns first",
                    nullptr);
    }
    auto mut_view = wrapper->chunked_filter_row_mask->mutable_view();
    auto result   = wrapper->reader->materialize_filter_columns_chunk(mut_view);
    return cudf::jni::convert_table_for_return(env, result.tbl);
  }
  JNI_CATCH(env, nullptr);
}

// Transfer ownership of the chunked-filter row mask out of the wrapper to Java.
// After this call the wrapper's slot is nullptr; subsequent chunk calls will fail until
// setupChunkingForFilterColumns is invoked again.
JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_HybridScanReader_takeFilterRowMask(JNIEnv* env,
                                                                               jclass,
                                                                               jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    if (wrapper->chunked_filter_row_mask == nullptr) {
      JNI_THROW_NEW(env,
                    cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                    "no chunked-filter row mask present; call setupChunkingForFilterColumns first "
                    "(or it has already been taken)",
                    0);
    }
    return cudf::jni::release_as_jlong(std::move(wrapper->chunked_filter_row_mask));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_HybridScanReader_setupChunkingForPayloadColumns(JNIEnv* env,
                                                                    jclass,
                                                                    jlong handle,
                                                                    jlong chunk_read_limit,
                                                                    jlong pass_read_limit,
                                                                    jintArray j_row_groups,
                                                                    jlong row_mask_view_handle,
                                                                    jboolean use_page_level_pruning,
                                                                    jlongArray j_addrs,
                                                                    jlongArray j_lens)
{
  JNI_NULL_CHECK(env, handle, "handle is null", );
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", );
  JNI_NULL_CHECK(env, row_mask_view_handle, "row mask view handle is null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const chunk_limit = checked_size_t(env, chunk_read_limit, "chunk_read_limit");
    auto const pass_limit  = checked_size_t(env, pass_read_limit, "pass_read_limit");
    auto* wrapper          = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder            = make_row_group_span(env, j_row_groups);
    auto spans             = make_device_spans(env, j_addrs, j_lens);
    auto* row_mask         = reinterpret_cast<cudf::column_view const*>(row_mask_view_handle);
    auto mode              = to_data_page_mask(use_page_level_pruning);
    wrapper->reader->setup_chunking_for_payload_columns(chunk_limit,
                                                        pass_limit,
                                                        holder.span(),
                                                        *row_mask,
                                                        mode,
                                                        spans,
                                                        wrapper->options,
                                                        cudf::get_default_stream(),
                                                        cudf::get_current_device_resource_ref());
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_HybridScanReader_materializePayloadColumnsChunk(
  JNIEnv* env, jclass, jlong handle, jlong row_mask_view_handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, row_mask_view_handle, "row mask view handle is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper  = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto* row_mask = reinterpret_cast<cudf::column_view const*>(row_mask_view_handle);
    auto result    = wrapper->reader->materialize_payload_columns_chunk(*row_mask);
    return cudf::jni::convert_table_for_return(env, result.tbl);
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_HybridScanReader_setupChunkingForAllColumns(JNIEnv* env,
                                                                jclass,
                                                                jlong handle,
                                                                jlong chunk_read_limit,
                                                                jlong pass_read_limit,
                                                                jintArray j_row_groups,
                                                                jlongArray j_addrs,
                                                                jlongArray j_lens)
{
  JNI_NULL_CHECK(env, handle, "handle is null", );
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const chunk_limit = checked_size_t(env, chunk_read_limit, "chunk_read_limit");
    auto const pass_limit  = checked_size_t(env, pass_read_limit, "pass_read_limit");
    auto* wrapper          = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder            = make_row_group_span(env, j_row_groups);
    auto spans             = make_device_spans(env, j_addrs, j_lens);
    wrapper->reader->setup_chunking_for_all_columns(chunk_limit,
                                                    pass_limit,
                                                    holder.span(),
                                                    spans,
                                                    wrapper->options,
                                                    cudf::get_default_stream(),
                                                    cudf::get_current_device_resource_ref());
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_HybridScanReader_materializeAllColumnsChunk(JNIEnv* env, jclass, jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto result   = wrapper->reader->materialize_all_columns_chunk();
    return cudf::jni::convert_table_for_return(env, result.tbl);
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jboolean JNICALL Java_ai_rapids_cudf_HybridScanReader_hasNextTableChunk(JNIEnv* env,
                                                                                  jclass,
                                                                                  jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", JNI_FALSE);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    return wrapper->reader->has_next_table_chunk() ? JNI_TRUE : JNI_FALSE;
  }
  JNI_CATCH(env, JNI_FALSE);
}

JNIEXPORT jobjectArray JNICALL Java_ai_rapids_cudf_HybridScanReader_constructRowGroupPasses(
  JNIEnv* env, jclass, jlong handle, jintArray j_row_groups, jlong pass_read_limit)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const pass_limit = checked_size_t(env, pass_read_limit, "pass_read_limit");
    auto* wrapper         = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder           = make_row_group_span(env, j_row_groups);
    auto passes           = wrapper->reader->construct_row_group_passes(holder.span(), pass_limit);
    jclass int_array_cls  = env->FindClass("[I");
    if (int_array_cls == nullptr) { return nullptr; }
    auto outer = env->NewObjectArray(passes.size(), int_array_cls, nullptr);
    if (outer == nullptr) { return nullptr; }
    for (size_t i = 0; i < passes.size(); ++i) {
      auto inner = sizes_to_jint_array(env, passes[i]);
      if (inner == nullptr) { return nullptr; }
      env->SetObjectArrayElement(outer, static_cast<jsize>(i), inner);
      env->DeleteLocalRef(inner);
    }
    return outer;
  }
  JNI_CATCH(env, nullptr);
}

}  // extern "C"
