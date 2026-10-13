/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file temp_storage_wrapper.cuh
 * @brief Temporary storage for cub calls and helper wrapper class
 */

#pragma once

#include "byte_array_view.cuh"
#include "statistics.cuh"

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cub/block/block_reduce.cuh>

namespace cudf {
namespace io {
namespace detail {

template <typename T, int block_size>
using cub_temp_storage = typename cub::BlockReduce<T, block_size>::TempStorage;
using statistics::byte_array_view;

#define MEMBER_NAME(TYPE) TYPE##_stats

#define DECLARE_MEMBER(TYPE) cub_temp_storage<TYPE, block_size> MEMBER_NAME(TYPE);

/**
 * @brief Templated union to hold temporary storage to be used by cub reduce
 * calls
 *
 * @tparam block_size Dimension of the block
 */
template <int block_size>
union block_reduce_storage {
  DECLARE_MEMBER(bool)
  DECLARE_MEMBER(int8_t)
  DECLARE_MEMBER(int16_t)
  DECLARE_MEMBER(int32_t)
  DECLARE_MEMBER(int64_t)
  DECLARE_MEMBER(__int128_t)
  DECLARE_MEMBER(uint8_t)
  DECLARE_MEMBER(uint16_t)
  DECLARE_MEMBER(uint32_t)
  DECLARE_MEMBER(uint64_t)
  DECLARE_MEMBER(float)
  DECLARE_MEMBER(double)
  DECLARE_MEMBER(string_view)
  DECLARE_MEMBER(byte_array_view)
};

#define STORAGE_WRAPPER_GET(TYPE)                                                                 \
  template <typename T>                                                                           \
  __device__ std::enable_if_t<std::is_same_v<T, TYPE>, cub_temp_storage<TYPE, block_size>&> get() \
  {                                                                                               \
    return storage.MEMBER_NAME(TYPE);                                                             \
  }

/**
 * @brief Templated wrapper for block_reduce_storage to return member reference based on requested
 * type
 *
 * @tparam block_size Dimension of the block
 */
template <int block_size>
struct storage_wrapper {
  block_reduce_storage<block_size>& storage;
  __device__ storage_wrapper(block_reduce_storage<block_size>& _temp_storage)
    : storage(_temp_storage)
  {
  }

  STORAGE_WRAPPER_GET(bool);
  STORAGE_WRAPPER_GET(int8_t);
  STORAGE_WRAPPER_GET(int16_t);
  STORAGE_WRAPPER_GET(int32_t);
  STORAGE_WRAPPER_GET(int64_t);
  STORAGE_WRAPPER_GET(__int128_t);
  STORAGE_WRAPPER_GET(uint8_t);
  STORAGE_WRAPPER_GET(uint16_t);
  STORAGE_WRAPPER_GET(uint32_t);
  STORAGE_WRAPPER_GET(uint64_t);
  STORAGE_WRAPPER_GET(float);
  STORAGE_WRAPPER_GET(double);
  STORAGE_WRAPPER_GET(string_view);
  STORAGE_WRAPPER_GET(byte_array_view);
};

#undef DECLARE_MEMBER
#undef MEMBER_NAME

}  // namespace detail
}  // namespace io
}  // namespace cudf
