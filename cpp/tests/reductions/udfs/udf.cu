/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/errc.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>

#include <cuda/std/cstdint>
#include <cuda/std/optional>

extern "C" {
using cuda::std::int32_t;
using cuda::std::int64_t;
using cuda::std::nullopt;
using cuda::std::optional;
using cuda::std::uint32_t;
using cuda::std::uint64_t;
using cudf::errc;
using cudf::size_type;
using cudf::string_view;

__device__ errc sum_update(optional<int64_t>* sum,
                           optional<int64_t>* count,
                           optional<int32_t> value)
{
  if (value) {
    *sum   = sum->value_or(0) + *value;
    *count = count->value() + 1;
  }
  return errc::SUCCESS;
}

__device__ errc checked_update(optional<int64_t>* sum,
                               optional<int64_t>* count,
                               optional<int32_t> value)
{
  if (value && *value < 0) { return errc::ARITHMETIC_OVERFLOW; }
  return sum_update(sum, count, value);
}

__device__ errc boolean_filter(bool* selected, optional<bool> keep)
{
  *selected = keep.value_or(false);
  return errc::SUCCESS;
}

__device__ errc mask_filter(size_type row, bool* selected, void* user_data)
{
  auto const* mask = static_cast<uint32_t const*>(user_data);
  *selected        = (mask[row / 32] & (uint32_t{1} << (row % 32))) != 0;
  return errc::SUCCESS;
}

__device__ errc raw_mask_filter(size_type row, bool* selected, void* user_data)
{
  return mask_filter(row, selected, user_data);
}

__device__ errc error_filter(bool* selected)
{
  *selected = true;
  return errc::ARITHMETIC_OVERFLOW;
}

__device__ errc error_update(optional<int64_t>*, optional<int64_t>*, optional<int32_t>)
{
  return errc::DIVISION_BY_ZERO;
}

__device__ errc error_init(optional<int64_t>* sum, optional<int64_t>* count)
{
  *sum   = nullopt;
  *count = 0;
  return errc::ARITHMETIC_OVERFLOW;
}

__device__ errc error_merge(optional<int64_t>*,
                            optional<int64_t>*,
                            optional<int64_t>,
                            optional<int64_t>)
{
  return errc::DIVISION_BY_ZERO;
}

__device__ errc error_finalize(
  optional<int64_t>*, optional<int64_t>*, optional<double>*, optional<int64_t>, optional<int64_t>)
{
  return errc::DIVISION_BY_ZERO;
}

__device__ errc discount_update(optional<int64_t>* sum,
                                optional<int64_t>* count,
                                optional<int32_t> amount,
                                optional<int32_t> discount,
                                optional<int32_t> rate)
{
  if (amount && discount && rate) {
    *sum   = sum->value_or(0) + (int64_t{*amount} - *discount) * *rate;
    *count = count->value() + 1;
  }
  return errc::SUCCESS;
}

__device__ errc user_data_update(size_type row,
                                 optional<int64_t>* sum,
                                 optional<int64_t>* count,
                                 optional<int32_t> value,
                                 void* user_data)
{
  auto const* weights = static_cast<int64_t const*>(user_data);
  if (value) {
    *sum   = sum->value_or(0) + *value * weights[0] + weights[row + 1];
    *count = count->value() + 1;
  }
  return errc::SUCCESS;
}

__device__ errc tpch_q6_sales_finalize(
  optional<int64_t>*, optional<int64_t>*, optional<double>*, optional<int64_t>, optional<int64_t>);

__device__ errc fused_finalize(optional<int64_t>* out_sum,
                               optional<int64_t>* out_count,
                               optional<double>* out_average,
                               optional<bool>* out_qualifies,
                               optional<int64_t> sum,
                               optional<int64_t> count)
{
  *out_qualifies = nullopt;
  if (sum) { *out_qualifies = *sum >= 100; }
  return tpch_q6_sales_finalize(out_sum, out_count, out_average, sum, count);
}

__device__ errc string_update(optional<int64_t>* sum,
                              optional<int64_t>* count,
                              optional<string_view> value)
{
  if (value) {
    *sum   = sum->value_or(0) + value->size_bytes();
    *count = count->value() + 1;
  }
  return errc::SUCCESS;
}

// SELECT MAX(value), OCTET_LENGTH(MAX(value)) FROM input;
__device__ errc string_max_init(optional<string_view>* state)
{
  *state = nullopt;
  return errc::SUCCESS;
}

__device__ errc string_max_merge(optional<string_view>* state, optional<string_view> value)
{
  if (value && (!*state || **state < *value)) { *state = value; }
  return errc::SUCCESS;
}

__device__ errc string_max_update(optional<string_view>* state, optional<string_view> value)
{
  return string_max_merge(state, value);
}

__device__ errc string_max_finalize(optional<string_view>* output,
                                    optional<int32_t>* length,
                                    optional<string_view> state)
{
  *output = state;
  *length = nullopt;
  if (state) { *length = state->size_bytes(); }
  return errc::SUCCESS;
}

// Value-only callbacks implement COALESCE(MAX(value), '') and its byte length.
__device__ errc raw_string_max_init(string_view* state)
{
  *state = {};
  return errc::SUCCESS;
}

__device__ errc raw_string_max_merge(string_view* state, string_view value)
{
  if (*state < value) { *state = value; }
  return errc::SUCCESS;
}

__device__ errc raw_string_max_update(string_view* state, string_view value)
{
  return raw_string_max_merge(state, value);
}

__device__ errc raw_string_max_finalize(string_view* output, int32_t* length, string_view state)
{
  *output = state;
  *length = state.size_bytes();
  return errc::SUCCESS;
}

// Value-only callbacks implement COALESCE(SUM(value), 0), skipping null input rows.
__device__ errc raw_init(int64_t* sum)
{
  *sum = 0;
  return errc::SUCCESS;
}

__device__ errc raw_update(int64_t* sum, int32_t value)
{
  *sum += value;
  return errc::SUCCESS;
}

__device__ errc raw_merge(int64_t* sum, int64_t other)
{
  *sum += other;
  return errc::SUCCESS;
}

__device__ errc raw_finalize(int64_t* output, int64_t sum)
{
  *output = sum;
  return errc::SUCCESS;
}

// TPC-H Q6-inspired filter/project/reduce with synthetic predicates and extra COUNT/AVG outputs.
// SELECT SUM(price * quantity), COUNT(price * quantity), AVG(price * quantity)
// FROM sales WHERE active AND quantity > 0;

__device__ errc tpch_q6_sales_init(optional<int64_t>* sum, optional<int64_t>* count)
{
  *sum   = nullopt;
  *count = 0;
  return errc::SUCCESS;
}

__device__ errc tpch_q6_sales_update(optional<int64_t>* sum,
                                     optional<int64_t>* count,
                                     optional<int32_t> price,
                                     optional<int32_t> quantity,
                                     optional<bool> active)
{
  if (!active.value_or(false) || !quantity || *quantity <= 0 || !price) { return errc::SUCCESS; }
  *sum   = sum->value_or(0) + int64_t{*price} * *quantity;
  *count = count->value() + 1;
  return errc::SUCCESS;
}

__device__ errc tpch_q6_sales_merge(optional<int64_t>* sum,
                                    optional<int64_t>* count,
                                    optional<int64_t> other_sum,
                                    optional<int64_t> other_count)
{
  if (other_sum) { *sum = sum->value_or(0) + *other_sum; }
  *count = count->value() + other_count.value();
  return errc::SUCCESS;
}

__device__ errc tpch_q6_sales_finalize(optional<int64_t>* out_sum,
                                       optional<int64_t>* out_count,
                                       optional<double>* out_average,
                                       optional<int64_t> sum,
                                       optional<int64_t> count)
{
  *out_sum     = sum;
  *out_count   = count;
  *out_average = nullopt;
  if (count.value() != 0) { *out_average = static_cast<double>(sum.value()) / count.value(); }
  return errc::SUCCESS;
}

__device__ errc empty_init() { return errc::SUCCESS; }

__device__ errc empty_update(int32_t) { return errc::SUCCESS; }

__device__ errc empty_merge() { return errc::SUCCESS; }

__device__ errc empty_finalize() { return errc::SUCCESS; }

__device__ errc empty_finalize_error() { return errc::DIVISION_BY_ZERO; }

// Exercise optional state large enough to require scratch-backed block reduction.
#define FIELDS(F)                                                                                \
  F(0), F(1), F(2), F(3), F(4), F(5), F(6), F(7), F(8), F(9), F(10), F(11), F(12), F(13), F(14), \
    F(15), F(16), F(17), F(18), F(19), F(20), F(21), F(22), F(23), F(24), F(25), F(26), F(27),   \
    F(28), F(29), F(30), F(31)
#define POINTER(i) optional<int64_t>* state##i
#define VALUE(i)   optional<int64_t> value##i

__device__ errc wide_init(FIELDS(POINTER))
{
#define INIT(i) *state##i = 0
  FIELDS(INIT);
#undef INIT
  return errc::SUCCESS;
}

__device__ errc wide_update(FIELDS(POINTER), optional<int32_t> input)
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

__device__ errc wide_finalize(optional<int64_t>* output, FIELDS(VALUE))
{
  int64_t result = 0;
#define SUM(i) result += value##i.value_or(0)
  FIELDS(SUM);
#undef SUM
  *output = result;
  return errc::SUCCESS;
}
#undef VALUE
#undef POINTER
#undef FIELDS

}  // extern "C"
