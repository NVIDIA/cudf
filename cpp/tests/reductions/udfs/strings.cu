/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"

#include <cudf/strings/string_view.cuh>

using optional_string = cuda::std::optional<cudf::string_view>;

extern "C" {

__device__ errc string_update(size_type,
                              optional_int64* sum,
                              optional_int64* count,
                              cuda::std::optional<cudf::string_view> value)
{
  if (value) {
    *sum   = sum->value_or(0) + value->size_bytes();
    *count = count->value() + 1;
  }
  return errc::SUCCESS;
}

// SELECT MAX(value), OCTET_LENGTH(MAX(value)) FROM input;
__device__ errc string_max_init(optional_string* state)
{
  *state = cuda::std::nullopt;
  return errc::SUCCESS;
}

__device__ errc string_max_merge(optional_string* state, optional_string value)
{
  if (value && (!*state || **state < *value)) { *state = value; }
  return errc::SUCCESS;
}

__device__ errc string_max_update(size_type, optional_string* state, optional_string value)
{
  return string_max_merge(state, value);
}

__device__ errc string_max_finalize(optional_string* output,
                                    optional_int32* length,
                                    optional_string state)
{
  *output = state;
  *length = cuda::std::nullopt;
  if (state) { *length = state->size_bytes(); }
  return errc::SUCCESS;
}

__device__ errc string_max_segment_init(size_type, optional_string* state)
{
  return string_max_init(state);
}

__device__ errc string_max_segment_update(size_type,
                                          size_type row,
                                          optional_string* state,
                                          optional_string value)
{
  return string_max_update(row, state, value);
}

__device__ errc string_max_segment_merge(size_type, optional_string* state, optional_string value)
{
  return string_max_merge(state, value);
}

__device__ errc string_max_segment_finalize(size_type,
                                            optional_string* output,
                                            optional_int32* length,
                                            optional_string state)
{
  return string_max_finalize(output, length, state);
}

// Value-only callbacks implement COALESCE(MAX(value), '') and its byte length.
__device__ errc raw_string_max_init(cudf::string_view* state)
{
  *state = {};
  return errc::SUCCESS;
}

__device__ errc raw_string_max_merge(cudf::string_view* state, cudf::string_view value)
{
  if (*state < value) { *state = value; }
  return errc::SUCCESS;
}

__device__ errc raw_string_max_update(size_type, cudf::string_view* state, cudf::string_view value)
{
  return raw_string_max_merge(state, value);
}

__device__ errc raw_string_max_finalize(cudf::string_view* output,
                                        int32_t* length,
                                        cudf::string_view state)
{
  *output = state;
  *length = state.size_bytes();
  return errc::SUCCESS;
}

}  // extern "C"
