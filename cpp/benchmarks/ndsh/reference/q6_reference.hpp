/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "reference_io.hpp"

#include <cudf/utilities/error.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cstdint>

namespace ndsh {
struct q6_reference_result {
  cudf::size_type matched = 0;
  double revenue          = 0;
};

// Independent CPU reference for generated, non-null Q6 inputs; never part of timed execution.
inline q6_reference_result q6_cpu_reference(cudf::table_view projected, cuda::stream_ref stream)
{
  CUDF_EXPECTS(projected.num_columns() == 4, "Expected the four Q6 columns in projection order");
  q6_reference_result result;
  if (projected.num_rows() == 0) return result;
  using enum cudf::type_id;
  detail::reference_schema(projected, {FLOAT64, FLOAT64, TIMESTAMP_DAYS, INT8});
  detail::for_reference_batches(projected, stream, [&](detail::reference_batch const& batch) {
    auto const price    = batch.values<double>(0);
    auto const discount = batch.values<double>(1);
    auto const shipdate = batch.values<cudf::timestamp_D>(2);
    auto const quantity = batch.values<int8_t>(3);
    for (cudf::size_type i = 0; i < batch.count; ++i) {
      auto const date = shipdate[i].time_since_epoch().count();
      auto const d    = static_cast<float>(discount[i]);
      auto const q    = static_cast<float>(quantity[i]);
      // Epoch days: 1994-01-01 inclusive, 1995-01-01 exclusive.
      if (8766 <= date && date < 9131 && 0.05f <= d && d <= 0.07f && q < 24.0f) {
        ++result.matched;
        result.revenue += price[i] * discount[i];
      }
    }
  });
  return result;
}
}  // namespace ndsh
