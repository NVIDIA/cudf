/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "reference_io.hpp"

#include <cudf/utilities/error.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ndsh {
/** Q1 metrics: sum_qty, sum_base_price, sum_disc_price, sum_charge, avg_qty, avg_price, avg_disc,
 * count_order. */
struct q1_reference_result {
  std::map<std::pair<std::string, std::string>, std::array<double, 8>> groups;
  cudf::size_type matched = 0;
};

/** CPU-only oracle for the eight projected, non-null generated Q1 columns; outside timed work. */
inline q1_reference_result q1_cpu_reference(cudf::table_view projected, cuda::stream_ref stream)
{
  using enum cudf::type_id;
  detail::reference_schema(
    projected, {STRING, STRING, INT8, FLOAT64, FLOAT64, TIMESTAMP_DAYS, INT32, FLOAT64});
  q1_reference_result result;
  detail::for_reference_batches(projected, stream, [&](detail::reference_batch const& batch) {
    auto const flags    = batch.strings(0);
    auto const statuses = batch.strings(1);
    auto const quantity = batch.values<int8_t>(2);
    auto const price    = batch.values<double>(3);
    auto const discount = batch.values<double>(4);
    auto const shipdate = batch.values<cudf::timestamp_D>(5);
    auto const tax      = batch.values<double>(7);
    for (cudf::size_type i = 0; i < batch.count; ++i) {
      // 1998-09-02 is 10471 days after 1970-01-01; the boundary is inclusive.
      if (shipdate[i].time_since_epoch().count() > 10471) continue;
      auto& group             = result.groups[{flags[i], statuses[i]}];
      double const disc_price = price[i] * (1.0 - discount[i]);
      double const charge     = disc_price * (1.0 + tax[i]);
      // INT8 quantities over at most INT32 rows sum exactly in double.
      group[0] += quantity[i];
      group[1] += price[i];
      group[2] += disc_price;
      group[3] += charge;
      group[6] += discount[i];
      ++group[7];
      ++result.matched;
    }
  });
  for (auto& [key, group] : result.groups) {
    group[4] = group[0] / group[7];
    group[5] = group[1] / group[7];
    group[6] /= group[7];
  }
  return result;
}

/** Validate named, sorted Q1 output, including empty tables; use 1e-10 relative float tolerance. */
inline void check_q1_result(q1_reference_result const& expected,
                            table_with_names const& actual,
                            cuda::stream_ref stream)
{
  std::vector<std::string> const names{"l_returnflag",
                                       "l_linestatus",
                                       "sum_qty",
                                       "sum_base_price",
                                       "sum_disc_price",
                                       "sum_charge",
                                       "avg_qty",
                                       "avg_price",
                                       "avg_disc",
                                       "count_order"};
  using enum cudf::type_id;
  auto const table = detail::reference_output(
    actual,
    names,
    {STRING, STRING, INT64, FLOAT64, FLOAT64, FLOAT64, FLOAT64, FLOAT64, FLOAT64, INT32},
    expected.groups.size(),
    false);
  auto group      = expected.groups.begin();
  int64_t matched = 0;
  detail::for_reference_batches(table, stream, [&](detail::reference_batch const& batch) {
    auto const first    = group;
    auto const flags    = batch.strings(0);
    auto const statuses = batch.strings(1);
    for (cudf::size_type i = 0; i < batch.count; ++i, ++group) {
      CUDF_EXPECTS(group->first == std::make_pair(flags[i], statuses[i]),
                   "Q1 ordered keys mismatch");
    }
    for (int metric = 0; metric < 8; ++metric) {
      auto check = [&](auto type) {
        auto const values = batch.values<decltype(type)>(metric + 2);
        auto reference    = first;
        for (cudf::size_type i = 0; i < batch.count; ++i, ++reference) {
          double const value = values[i], want = reference->second[metric];
          bool const exact = metric == 0 || metric == 7;
          CUDF_EXPECTS(detail::reference_equal(value, want, exact),
                       "Q1 metric mismatch: " + names[metric + 2]);
          if (metric == 7) matched += static_cast<int64_t>(values[i]);
        }
      };
      if (metric == 0) {
        check(int64_t{});
      } else if (metric == 7) {
        check(int32_t{});
      } else {
        check(double{});
      }
    }
  });
  CUDF_EXPECTS(matched == expected.matched, "Q1 matched row count mismatch");
}
}  // namespace ndsh
