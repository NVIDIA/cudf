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

// Shared memory state struct used by the preprocess_levels kernel.
// Includes setup (page metadata + error) and stream (level byte ranges + RLE state)
// which are necessary to advance level streams.
struct level_scan_state {
  page_decode_setup_state setup;
  page_decode_stream_state stream;
  CUDF_PARQUET_PAGE_STATE_ERROR_METHODS
};

// Shared memory state struct used by the compute_page_string_sizes,
// compute_delta_page_string_sizes, and compute_delta_length_page_string_sizes kernels. Includes
// setup (page metadata + error), stream (page bytes + dictionary), and output_cvt (dtype_len_in)
// because these scans read the conversion scratch written by setup_local_page_info for
// FIXED_LEN_BYTE_ARRAY size math but never walk nesting info or track progress counters.
struct string_size_scan_state {
  page_decode_setup_state setup;
  page_decode_stream_state stream;
  page_decode_output_state output_cvt;
  CUDF_PARQUET_PAGE_STATE_ERROR_METHODS
};

// Shared memory state struct used by the preprocess_string_offsets kernel.
// Includes setup (page metadata + error), stream (page bytes + dictionary), and
// progress (input counters) because this pass scans flat string payloads while tracking counts.
struct string_offset_scan_state {
  page_decode_setup_state setup;
  page_decode_stream_state stream;
  page_decode_progress_state progress;
  CUDF_PARQUET_PAGE_STATE_ERROR_METHODS
};
#undef CUDF_PARQUET_PAGE_STATE_ERROR_METHODS

}  // namespace cudf::io::parquet::detail
