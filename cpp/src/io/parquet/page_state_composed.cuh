/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "page_decode.cuh"

namespace cudf::io::parquet::detail {

// Convenience macro to define error methods for page state structs that include a setup state.
// Cheaper and easier than trying to introduce inheritance or templates for this purpose.
#define CUDF_PARQUET_PAGE_STATE_ERROR_METHODS                                                  \
  inline __device__ void set_error_code(decode_error err)                                      \
  {                                                                                            \
    cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block> ref{setup.error};     \
    ref.fetch_or(static_cast<kernel_error::value_type>(err), cuda::std::memory_order_relaxed); \
  }                                                                                            \
  inline __device__ void reset_error_code()                                                    \
  {                                                                                            \
    cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block> ref{setup.error};     \
    ref.store(0, cuda::std::memory_order_release);                                             \
  }

// Composition A: full page decode (level decoding + nested output).
// Used by: decode_page_data, decode_split_page_data, decode_page_data_generic,
//          decode_delta_binary, decode_delta_byte_array, decode_delta_length_byte_array,
//          compute_string_page_bounds, compute_page_sizes.
// Includes setup (page metadata + error), stream (data source + dictionary), nesting,
// progress, and conversion scratch because these kernels both walk levels and materialize output.
struct full_page_decode_state {
  page_decode_setup_state setup;
  page_decode_stream_state stream;
  page_decode_nesting_state nesting;
  page_decode_progress_state progress;
  page_decode_output_state output_cvt;
  CUDF_PARQUET_PAGE_STATE_ERROR_METHODS
};

// Composition B: flat string size scan (no nesting, no progress).
// Used by: compute_page_string_sizes, compute_delta_page_string_sizes,
//          compute_delta_length_page_string_sizes.
// Includes setup (page metadata + error), stream (byte source + dictionary), and
// conversion scratch because these kernels size decoded strings without nested
// output assembly or per-value progress bookkeeping.
struct string_size_scan_state {
  page_decode_setup_state setup;
  page_decode_stream_state stream;
  page_decode_output_state output_cvt;
  CUDF_PARQUET_PAGE_STATE_ERROR_METHODS
};

// Composition C: level-only preprocess (RLE level decoding only, no output).
// Used by: preprocess_levels.
// Includes setup (page metadata + error) and stream (level byte ranges + RLE state)
// which are necessary to advance level streams.
struct level_scan_state {
  page_decode_setup_state setup;
  page_decode_stream_state stream;
  CUDF_PARQUET_PAGE_STATE_ERROR_METHODS
};

// Composition D: string offset preprocess (flat scan with progress tracking).
// Used by: preprocess_string_offsets.
// Includes setup (page metadata + error), stream (page bytes + dictionary), and
// progress (input counters) because this pass scans flat string payloads while tracking counts.
// FLBA (FIXED_LEN_BYTE_ARRAY) pages return before setup_local_page_info runs, and
// non-FLBA setup writes conversion scratch that this kernel never reads, so it stays out.
struct string_offset_scan_state {
  page_decode_setup_state setup;
  page_decode_stream_state stream;
  page_decode_progress_state progress;
  CUDF_PARQUET_PAGE_STATE_ERROR_METHODS
};
#undef CUDF_PARQUET_PAGE_STATE_ERROR_METHODS

}  // namespace cudf::io::parquet::detail
