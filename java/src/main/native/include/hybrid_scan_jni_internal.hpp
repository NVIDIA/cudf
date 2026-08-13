/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "jni_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <jni.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace cudf {
namespace jni {
namespace hybrid_scan {

namespace exp_pq = cudf::io::parquet::experimental;
using cudf::io::text::byte_range_info;

/**
 * @brief Wrapper that owns the C++ hybrid_scan_reader plus the parquet_reader_options that
 *        most reader methods need to be supplied alongside the call. Keeping them together
 *        avoids the need for the caller (Java) to round-trip the options across JNI on each
 *        call.
 *
 *        Also caches a filterless snapshot of the options (`base_options`) so that
 *        setFilter can clear or replace the filter without reconstructing the reader:
 *        parquet_reader_options exposes set_filter to install a filter but has no
 *        clear_filter, so replacing/clearing is done by copying base_options over
 *        `options` and optionally re-installing the new filter on top.
 */
struct hybrid_scan_reader_wrapper {
  cudf::io::parquet_reader_options options;
  cudf::io::parquet_reader_options base_options;
  std::unique_ptr<exp_pq::hybrid_scan_reader> reader;
  // Owns the row mask for the duration of a chunked filter-column pipeline.
  // Set by setup_chunking_for_filter_columns, mutated in place by each
  // materialize_filter_columns_chunk call, and transferred to Java by take_filter_row_mask.
  // nullptr outside of an active chunked-filter session.
  std::unique_ptr<cudf::column> chunked_filter_row_mask;

  hybrid_scan_reader_wrapper(cudf::host_span<uint8_t const> footer_bytes,
                             cudf::io::parquet_reader_options opts)
    : options(opts),
      base_options(std::move(opts)),
      reader(std::make_unique<exp_pq::hybrid_scan_reader>(footer_bytes, options))
  {
  }
};

/**
 * @brief Convert a Java int[] of row group indices into a host_span<size_type const>. The
 *        wrapper owns a vector that backs the span; capture it as a value to avoid dangling.
 */
struct row_group_span_holder {
  std::vector<cudf::size_type> storage;
  cudf::host_span<cudf::size_type const> span() const { return {storage.data(), storage.size()}; }
};

/**
 * @brief Build a parquet_reader_options from the supplied JNI args. The footer is provided
 *        separately because the hybrid_scan_reader does not consume it via the options.
 *        The returned options carry no filter; use HybridScanReader.setFilter (JNI:
 *        Java_ai_rapids_cudf_HybridScanReader_setFilter) to install one after construction.
 */
cudf::io::parquet_reader_options build_options(JNIEnv* env,
                                               jobjectArray j_column_names,
                                               jbooleanArray j_read_binary_as_string,
                                               jint time_unit_type_id);

row_group_span_holder make_row_group_span(JNIEnv* env, jintArray j_row_groups);

/**
 * @brief Convert a vector<byte_range_info> into a packed jlongArray laid out as
 *        [off0, len0, off1, len1, ...].
 */
jlongArray ranges_to_jlong_array(JNIEnv* env, std::vector<byte_range_info> const& ranges);

jintArray sizes_to_jint_array(JNIEnv* env, std::vector<cudf::size_type> const& vals);

std::vector<cudf::device_span<uint8_t const>> make_device_spans(JNIEnv* env,
                                                                jlongArray j_addrs,
                                                                jlongArray j_lens);

/** @brief Convert a jlong to size_t; throws Java IllegalArgumentException if @p value is negative
 * (@p name is embedded in the message). */
std::size_t checked_size_t(JNIEnv* env, jlong value, char const* name);

}  // namespace hybrid_scan
}  // namespace jni
}  // namespace cudf
