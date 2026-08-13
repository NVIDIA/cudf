/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"
#include "hybrid_scan_jni_internal.hpp"
#include "jni_compiled_expr.hpp"
#include "jni_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <memory>
#include <utility>
#include <vector>

using namespace cudf::jni::hybrid_scan;

extern "C" {

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_HybridScanReader_createFromFooter(JNIEnv* env,
                                                      jclass,
                                                      jlong footer_address,
                                                      jlong footer_length,
                                                      jobjectArray j_column_names,
                                                      jbooleanArray j_binary_as_str,
                                                      jint time_unit_type_id)
{
  JNI_NULL_CHECK(env, footer_address, "footer address is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const len         = checked_size_t(env, footer_length, "footerLength");
    auto opts              = build_options(env, j_column_names, j_binary_as_str, time_unit_type_id);
    auto const* footer_ptr = reinterpret_cast<uint8_t const*>(footer_address);
    cudf::host_span<uint8_t const> footer_bytes{footer_ptr, len};
    auto wrapper = std::make_unique<hybrid_scan_reader_wrapper>(footer_bytes, std::move(opts));
    return reinterpret_cast<jlong>(wrapper.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_HybridScanReader_destroy(JNIEnv* env,
                                                                    jclass,
                                                                    jlong handle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
  }
  JNI_CATCH(env, );
}

// Install or clear the AST filter, then invalidate the cached column selection so the next
// filter/materialize/byte-range call re-selects columns against the new filter.
//
// parquet_reader_options::set_filter installs a filter but has no clear_filter. To support
// filter replacement / clearing without rebuilding the reader, the wrapper caches a
// filterless snapshot (`base_options`); we start from that snapshot on every call and then
// (optionally) install the new filter on top.
JNIEXPORT void JNICALL Java_ai_rapids_cudf_HybridScanReader_setFilter(JNIEnv* env,
                                                                      jclass,
                                                                      jlong handle,
                                                                      jlong filter_handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper    = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    wrapper->options = wrapper->base_options;
    if (filter_handle != 0) {
      auto const* expr = reinterpret_cast<cudf::jni::ast::compiled_expr const*>(filter_handle);
      wrapper->options.set_filter(expr->get_top_expression());
    }
    wrapper->reader->reset_column_selection();
  }
  JNI_CATCH(env, );
}

// ----------------------------------------------------------------------
// Metadata
// ----------------------------------------------------------------------

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_HybridScanReader_pageIndexByteRange(JNIEnv* env,
                                                                                     jclass,
                                                                                     jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto range    = wrapper->reader->page_index_byte_range();
    jlong vals[2] = {static_cast<jlong>(range.offset()), static_cast<jlong>(range.size())};
    auto result   = env->NewLongArray(2);
    if (result == nullptr) { return nullptr; }
    env->SetLongArrayRegion(result, 0, 2, vals);
    return result;
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_HybridScanReader_setupPageIndex(
  JNIEnv* env, jclass, jlong handle, jlong buffer_address, jlong buffer_length)
{
  JNI_NULL_CHECK(env, handle, "handle is null", );
  JNI_NULL_CHECK(env, buffer_address, "page index buffer is null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const len       = checked_size_t(env, buffer_length, "pageIndexBufferLength");
    auto* wrapper        = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto const* host_ptr = reinterpret_cast<uint8_t const*>(buffer_address);
    cudf::host_span<uint8_t const> bytes{host_ptr, len};
    wrapper->reader->setup_page_index(bytes);
  }
  JNI_CATCH(env, );
}

// ----------------------------------------------------------------------
// Row group enumeration
// ----------------------------------------------------------------------

JNIEXPORT jintArray JNICALL Java_ai_rapids_cudf_HybridScanReader_allRowGroups(JNIEnv* env,
                                                                              jclass,
                                                                              jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto rgs      = wrapper->reader->all_row_groups(wrapper->options);
    return sizes_to_jint_array(env, rgs);
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_HybridScanReader_totalRowsInRowGroups(
  JNIEnv* env, jclass, jlong handle, jintArray j_row_groups)
{
  JNI_NULL_CHECK(env, handle, "handle is null", 0);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder   = make_row_group_span(env, j_row_groups);
    auto total    = wrapper->reader->total_rows_in_row_groups(holder.span());
    return static_cast<jlong>(total);
  }
  JNI_CATCH(env, 0);
}

// ----------------------------------------------------------------------
// Filtering
// ----------------------------------------------------------------------

JNIEXPORT jintArray JNICALL Java_ai_rapids_cudf_HybridScanReader_filterRowGroupsWithStats(
  JNIEnv* env, jclass, jlong handle, jintArray j_row_groups)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder   = make_row_group_span(env, j_row_groups);
    auto filtered = wrapper->reader->filter_row_groups_with_stats(
      holder.span(), wrapper->options, cudf::get_default_stream());
    return sizes_to_jint_array(env, filtered);
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_HybridScanReader_secondaryFiltersByteRanges(
  JNIEnv* env, jclass, jlong handle, jintArray j_row_groups)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder   = make_row_group_span(env, j_row_groups);
    auto [bloom, dict] =
      wrapper->reader->secondary_filters_byte_ranges(holder.span(), wrapper->options);
    // Pack as [numBloom, bloom_o0, bloom_s0, ..., dict_o0, dict_s0, ...]
    auto const total_len = 1 + (bloom.size() + dict.size()) * 2;
    auto result          = env->NewLongArray(total_len);
    if (result == nullptr) { return nullptr; }
    std::vector<jlong> data;
    data.reserve(total_len);
    data.push_back(static_cast<jlong>(bloom.size()));
    for (auto const& r : bloom) {
      data.push_back(static_cast<jlong>(r.offset()));
      data.push_back(static_cast<jlong>(r.size()));
    }
    for (auto const& r : dict) {
      data.push_back(static_cast<jlong>(r.offset()));
      data.push_back(static_cast<jlong>(r.size()));
    }
    env->SetLongArrayRegion(result, 0, data.size(), data.data());
    return result;
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jintArray JNICALL Java_ai_rapids_cudf_HybridScanReader_filterRowGroupsWithDictionaryPages(
  JNIEnv* env, jclass, jlong handle, jlongArray j_addrs, jlongArray j_lens, jintArray j_row_groups)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder   = make_row_group_span(env, j_row_groups);
    auto spans    = make_device_spans(env, j_addrs, j_lens);
    auto filtered = wrapper->reader->filter_row_groups_with_dictionary_pages(
      spans, holder.span(), wrapper->options, cudf::get_default_stream());
    return sizes_to_jint_array(env, filtered);
  }
  JNI_CATCH(env, nullptr);
}

// ----------------------------------------------------------------------
// Byte ranges
// ----------------------------------------------------------------------

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_HybridScanReader_filterColumnChunksByteRanges(
  JNIEnv* env, jclass, jlong handle, jintArray j_row_groups)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder   = make_row_group_span(env, j_row_groups);
    auto ranges =
      wrapper->reader->filter_column_chunks_byte_ranges(holder.span(), wrapper->options);
    return ranges_to_jlong_array(env, ranges);
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_HybridScanReader_payloadColumnChunksByteRanges(
  JNIEnv* env, jclass, jlong handle, jintArray j_row_groups)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder   = make_row_group_span(env, j_row_groups);
    auto ranges =
      wrapper->reader->payload_column_chunks_byte_ranges(holder.span(), wrapper->options);
    return ranges_to_jlong_array(env, ranges);
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_HybridScanReader_allColumnChunksByteRanges(
  JNIEnv* env, jclass, jlong handle, jintArray j_row_groups)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto holder   = make_row_group_span(env, j_row_groups);
    auto ranges   = wrapper->reader->all_column_chunks_byte_ranges(holder.span(), wrapper->options);
    return ranges_to_jlong_array(env, ranges);
  }
  JNI_CATCH(env, nullptr);
}

}  // extern "C"
