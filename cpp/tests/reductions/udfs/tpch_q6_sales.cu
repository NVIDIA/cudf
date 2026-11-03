/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/errc.hpp>
#include <cudf/types.hpp>

#include <cuda/std/cstdint>
#include <cuda/std/optional>

using cuda::std::int32_t;
using cuda::std::int64_t;
using cudf::errc;
using cudf::size_type;
using optional_int64 = cuda::std::optional<int64_t>;

// TPC-H Q6-inspired filter/project/reduce with synthetic predicates and extra COUNT/AVG outputs.
// SELECT SUM(price * quantity), COUNT(price * quantity), AVG(price * quantity)
// FROM sales WHERE active AND quantity > 0;
extern "C" {

__device__ errc tpch_q6_sales_init(optional_int64* sum, optional_int64* count)
{
  *sum   = cuda::std::nullopt;
  *count = 0;
  return errc::SUCCESS;
}

__device__ errc tpch_q6_sales_update(size_type,
                                     optional_int64* sum,
                                     optional_int64* count,
                                     cuda::std::optional<int32_t> price,
                                     cuda::std::optional<int32_t> quantity,
                                     cuda::std::optional<bool> active)
{
  if (!active.value_or(false) || !quantity || *quantity <= 0 || !price) { return errc::SUCCESS; }
  *sum   = sum->value_or(0) + int64_t{*price} * *quantity;
  *count = count->value() + 1;
  return errc::SUCCESS;
}

__device__ errc tpch_q6_sales_merge(optional_int64* sum,
                                    optional_int64* count,
                                    optional_int64 other_sum,
                                    optional_int64 other_count)
{
  if (other_sum) { *sum = sum->value_or(0) + *other_sum; }
  *count = count->value() + other_count.value();
  return errc::SUCCESS;
}

__device__ errc tpch_q6_sales_finalize(optional_int64* out_sum,
                                       optional_int64* out_count,
                                       cuda::std::optional<double>* out_average,
                                       optional_int64 sum,
                                       optional_int64 count)
{
  *out_sum     = sum;
  *out_count   = count;
  *out_average = cuda::std::nullopt;
  if (count.value() != 0) { *out_average = static_cast<double>(sum.value()) / count.value(); }
  return errc::SUCCESS;
}

__device__ errc tpch_q6_sales_segment_init(size_type, optional_int64* sum, optional_int64* count)
{
  return tpch_q6_sales_init(sum, count);
}

__device__ errc tpch_q6_sales_segment_update(size_type,
                                             size_type row,
                                             optional_int64* sum,
                                             optional_int64* count,
                                             cuda::std::optional<int32_t> price,
                                             cuda::std::optional<int32_t> quantity,
                                             cuda::std::optional<bool> active)
{
  return tpch_q6_sales_update(row, sum, count, price, quantity, active);
}

__device__ errc tpch_q6_sales_segment_merge(size_type,
                                            optional_int64* sum,
                                            optional_int64* count,
                                            optional_int64 other_sum,
                                            optional_int64 other_count)
{
  return tpch_q6_sales_merge(sum, count, other_sum, other_count);
}

__device__ errc tpch_q6_sales_segment_finalize(size_type,
                                               optional_int64* out_sum,
                                               optional_int64* out_count,
                                               cuda::std::optional<double>* out_average,
                                               optional_int64 sum,
                                               optional_int64 count)
{
  return tpch_q6_sales_finalize(out_sum, out_count, out_average, sum, count);
}

}  // extern "C"
