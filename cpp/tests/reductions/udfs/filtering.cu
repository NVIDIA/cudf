/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"

extern "C" {

__device__ errc checked_update(size_type row,
                               optional_int64* sum,
                               optional_int64* count,
                               optional_int32 value)
{
  if (value && *value < 0) { return errc::ARITHMETIC_OVERFLOW; }
  return sum_update(row, sum, count, value);
}

__device__ errc boolean_filter(size_type,
                               bool* selected,
                               optional_int32,
                               cuda::std::optional<bool> keep)
{
  *selected = keep.value_or(false);
  return errc::SUCCESS;
}

__device__ errc checked_boolean_update(size_type row,
                                       optional_int64* sum,
                                       optional_int64* count,
                                       optional_int32 value,
                                       cuda::std::optional<bool>)
{
  return checked_update(row, sum, count, value);
}

__device__ errc mask_filter(size_type row, bool* selected, optional_int32, void* user_data)
{
  auto const* mask = static_cast<cuda::std::uint32_t const*>(user_data);
  *selected        = (mask[row / 32] & (cuda::std::uint32_t{1} << (row % 32))) != 0;
  return errc::SUCCESS;
}

__device__ errc checked_mask_update(
  size_type row, optional_int64* sum, optional_int64* count, optional_int32 value, void*)
{
  return checked_update(row, sum, count, value);
}

__device__ errc segment_mask_filter(
  size_type segment, size_type row, bool* selected, optional_int32 value, void* user_data)
{
  auto const error = mask_filter(row, selected, value, user_data);
  *selected        = *selected && segment != 1;
  return error;
}

__device__ errc checked_segment_mask_update(
  size_type, size_type row, optional_int64* sum, optional_int64* count, optional_int32 value, void*)
{
  return checked_update(row, sum, count, value);
}

__device__ errc raw_mask_filter(size_type row, bool* selected, int32_t value, void* user_data)
{
  return mask_filter(row, selected, value, user_data);
}

__device__ errc raw_mask_update(size_type row, int64_t* sum, int32_t value, void*)
{
  return raw_update(row, sum, value);
}

}  // extern "C"
