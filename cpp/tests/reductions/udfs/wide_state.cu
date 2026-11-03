/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "common.cuh"

// Exercise optional state large enough to require scratch-backed block reduction.
#define FIELDS(F)                                                                                \
  F(0), F(1), F(2), F(3), F(4), F(5), F(6), F(7), F(8), F(9), F(10), F(11), F(12), F(13), F(14), \
    F(15), F(16), F(17), F(18), F(19), F(20), F(21), F(22), F(23), F(24), F(25), F(26), F(27),   \
    F(28), F(29), F(30), F(31)
#define POINTER(i) optional_int64* state##i
#define VALUE(i)   optional_int64 value##i

extern "C" {
__device__ errc wide_init(FIELDS(POINTER))
{
#define INIT(i) *state##i = 0
  FIELDS(INIT);
#undef INIT
  return errc::SUCCESS;
}

__device__ errc wide_update(size_type, FIELDS(POINTER), optional_int32 input)
{
  auto const value = input.value_or(0);
#define UPDATE(i) *state##i = state##i->value_or(0) + int64_t{i + 1} * value
  FIELDS(UPDATE);
#undef UPDATE
  return errc::SUCCESS;
}

__device__ errc wide_merge(FIELDS(POINTER), FIELDS(VALUE))
{
#define MERGE(i) *state##i = state##i->value_or(0) + value##i.value_or(0)
  FIELDS(MERGE);
#undef MERGE
  return errc::SUCCESS;
}

__device__ errc wide_finalize(optional_int64* output, FIELDS(VALUE))
{
  int64_t result = 0;
#define SUM(i) result += value##i.value_or(0)
  FIELDS(SUM);
#undef SUM
  *output = result;
  return errc::SUCCESS;
}
}
#undef VALUE
#undef POINTER
#undef FIELDS
