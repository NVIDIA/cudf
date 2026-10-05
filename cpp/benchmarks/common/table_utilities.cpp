/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "table_utilities.hpp"

#include <cudf/reduction.hpp>
#include <cudf/transform.hpp>

#include <cmath>

int64_t estimate_size(cudf::column_view const& col)
{ return estimate_size(cudf::table_view({col})); }

int64_t estimate_size(cudf::table_view const& view)
{
  // Compute the size in bits for each row.
  auto const row_sizes = cudf::row_bit_count(view);
  // Accumulate the row sizes to compute a sum.
  auto const agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  cudf::data_type sum_dtype{cudf::type_id::INT64};
  auto const total_size_scalar = cudf::reduce(*row_sizes, *agg, sum_dtype);
  auto const total_size_in_bits =
    static_cast<cudf::numeric_scalar<int64_t>*>(total_size_scalar.get())->value();
  // Convert the size in bits to the size in bytes.
  return static_cast<int64_t>(std::ceil(static_cast<double>(total_size_in_bits) / 8));
}
