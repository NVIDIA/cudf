/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/abi_traits.cuh"

#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column_device_view.cuh>

#include <cuda/std/type_traits>

template <typename Abi, typename Cudf>
void expect_mapping()
{
  static_assert((cuda::std::is_same_v<cudf::abi_to_cudf_type<Abi>, Cudf>),
                "ABI to Cudf type mapping failed");
  static_assert((cuda::std::is_same_v<cudf::cudf_to_abi_type<Cudf>, Abi>),
                "Cudf to ABI type mapping failed");
  static_assert(sizeof(Abi) == sizeof(Cudf), "Size mismatch between ABI and Cudf types");
  static_assert(alignof(Abi) == alignof(Cudf), "Alignment mismatch between ABI and Cudf types");
}

TEST(AbiMapTest, Types)
{
#define CHECK_ABI_TYPE(Abi, Cudf) expect_mapping<cudf_abi::Abi, Cudf>()
  CHECK_ABI_TYPE(int8_t, int8_t);
  CHECK_ABI_TYPE(int16_t, int16_t);
  CHECK_ABI_TYPE(int32_t, int32_t);
  CHECK_ABI_TYPE(int64_t, int64_t);
  CHECK_ABI_TYPE(uint8_t, uint8_t);
  CHECK_ABI_TYPE(uint16_t, uint16_t);
  CHECK_ABI_TYPE(uint32_t, uint32_t);
  CHECK_ABI_TYPE(uint64_t, uint64_t);
  CHECK_ABI_TYPE(float32_t, float);
  CHECK_ABI_TYPE(float64_t, double);
  CHECK_ABI_TYPE(bool8_t, bool);
  CHECK_ABI_TYPE(size_type, cudf::size_type);
  CHECK_ABI_TYPE(voidptr_t, cudf::voidptr_t);
  CHECK_ABI_TYPE(error_int, cudf::error_int);
  CHECK_ABI_TYPE(bitmask_type, cudf::bitmask_type);
  CHECK_ABI_TYPE(errc, cudf::errc);
  CHECK_ABI_TYPE(string_view, cudf::string_view);
  CHECK_ABI_TYPE(mutable_string_view, cudf::mutable_string_view);
  CHECK_ABI_TYPE(duration_D, cudf::duration_D);
  CHECK_ABI_TYPE(duration_h, cudf::duration_h);
  CHECK_ABI_TYPE(duration_m, cudf::duration_m);
  CHECK_ABI_TYPE(duration_s, cudf::duration_s);
  CHECK_ABI_TYPE(duration_ms, cudf::duration_ms);
  CHECK_ABI_TYPE(duration_us, cudf::duration_us);
  CHECK_ABI_TYPE(duration_ns, cudf::duration_ns);
  CHECK_ABI_TYPE(timestamp_D, cudf::timestamp_D);
  CHECK_ABI_TYPE(timestamp_h, cudf::timestamp_h);
  CHECK_ABI_TYPE(timestamp_m, cudf::timestamp_m);
  CHECK_ABI_TYPE(timestamp_s, cudf::timestamp_s);
  CHECK_ABI_TYPE(timestamp_ms, cudf::timestamp_ms);
  CHECK_ABI_TYPE(timestamp_us, cudf::timestamp_us);
  CHECK_ABI_TYPE(timestamp_ns, cudf::timestamp_ns);
  CHECK_ABI_TYPE(decimal32, numeric::decimal32);
  CHECK_ABI_TYPE(decimal64, numeric::decimal64);
  CHECK_ABI_TYPE(decimal128, numeric::decimal128);
#undef CHECK_ABI_TYPE
}

TEST(AbiMapTest, Wrappers)
{
  expect_mapping<cudf_abi::optional<cudf_abi::int64_t>, cuda::std::optional<int64_t>>();
  expect_mapping<cudf_abi::optional<cudf_abi::mutable_string_view>,
                 cuda::std::optional<cudf::mutable_string_view>>();
}

CUDF_TEST_PROGRAM_MAIN()
