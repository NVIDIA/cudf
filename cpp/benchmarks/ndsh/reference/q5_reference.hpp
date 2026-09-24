/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file q5_reference.hpp
 * @brief Independent CPU joins, revenue aggregation, and output checks for Q5,
 * validating query results from generated, non-null inputs outside timed execution.
 */

#pragma once

#include "reference_io.hpp"

#include <cudf/utilities/error.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ndsh {
struct q5_reference_result {
  std::map<std::string, double> revenue;
  int64_t matched = 0;
};

/**
 * add_table borrows one complete projected table per call in dependency order, retaining
 * only CPU state. part and partsupp are ignored.
 * finish requires all six query tables, including empty ones.
 */
class q5_reference_builder {
 public:
  void add_table(std::string const& name, cudf::table_view projected, cuda::stream_ref stream)
  {
    if (name == "part" || name == "partsupp") return;
    using enum cudf::type_id;
    detail::reference_input_schema(projected,
                                   name,
                                   next_table_,
                                   {{"region", {INT8, STRING}},
                                    {"nation", {INT8, INT8, STRING}},
                                    {"supplier", {INT32, INT8}},
                                    {"customer", {INT32, INT8}},
                                    {"orders", {INT32, INT32, TIMESTAMP_DAYS}},
                                    {"lineitem", {INT32, INT32, FLOAT64, FLOAT64}}});
    // Check even discarded dimension rows, so filtering cannot conceal duplicate primary keys.
    std::unordered_set<int32_t> primary_keys;
    auto unique = [&](int32_t key) {
      CUDF_EXPECTS(primary_keys.insert(key).second, "Duplicate Q5 primary key in " + name);
    };
    detail::for_reference_batches(projected, stream, [&](detail::reference_batch const& batch) {
      if (name == "region") {
        auto const keys  = batch.values<int8_t>(0);
        auto const names = batch.strings(1);
        for (cudf::size_type i = 0; i < batch.count; ++i) {
          unique(keys[i]);
          if (names[i] == "ASIA") regions_.insert(keys[i]);
        }
      } else if (name == "nation") {
        auto const keys    = batch.values<int8_t>(0);
        auto const regions = batch.values<int8_t>(1);
        auto const names   = batch.strings(2);
        for (cudf::size_type i = 0; i < batch.count; ++i) {
          unique(keys[i]);
          if (regions_.contains(regions[i])) nations_.emplace(keys[i], names[i]);
        }
      } else if (name == "supplier" || name == "customer") {
        auto const keys    = batch.values<int32_t>(0);
        auto const nations = batch.values<int8_t>(1);
        auto& dimension    = name == "supplier" ? suppliers_ : customers_;
        for (cudf::size_type i = 0; i < batch.count; ++i) {
          unique(keys[i]);
          if (nations_.contains(nations[i])) dimension.emplace(keys[i], nations[i]);
        }
      } else if (name == "orders") {
        auto const customers = batch.values<int32_t>(0);
        auto const keys      = batch.values<int32_t>(1);
        auto const dates     = batch.values<cudf::timestamp_D>(2);
        for (cudf::size_type i = 0; i < batch.count; ++i) {
          unique(keys[i]);
          auto const date = dates[i].time_since_epoch().count();
          // 1994-01-01 inclusive through 1995-01-01 exclusive, in epoch days.
          if (date < 8766 || date >= 9131) continue;
          auto const customer = customers_.find(customers[i]);
          if (customer != customers_.end()) orders_.emplace(keys[i], customer->second);
        }
      } else {
        auto const orders    = batch.values<int32_t>(0);
        auto const suppliers = batch.values<int32_t>(1);
        auto const prices    = batch.values<double>(2);
        auto const discounts = batch.values<double>(3);
        for (cudf::size_type i = 0; i < batch.count; ++i) {
          auto const order = orders_.find(orders[i]);
          if (order == orders_.end()) continue;
          auto const supplier = suppliers_.find(suppliers[i]);
          // The supplier must match both its key and the customer's nation.
          if (supplier == suppliers_.end() || supplier->second != order->second) continue;
          ++result_.matched;
          result_.revenue[nations_.at(order->second)] += prices[i] * (1.0 - discounts[i]);
        }
      }
    });
    if (name == "orders") customers_.clear();
    ++next_table_;
  }

  q5_reference_result finish() const
  {
    CUDF_EXPECTS(next_table_ == 6, "Incomplete Q5 reference inputs");
    return result_;
  }

 private:
  std::size_t next_table_ = 0;
  std::unordered_set<int8_t> regions_;
  std::unordered_map<int8_t, std::string> nations_;
  std::unordered_map<int32_t, int8_t> suppliers_, customers_, orders_;
  q5_reference_result result_;
};

/** Check complete Q5 output in descending revenue order; ties may appear in any order. */
inline void check_q5_result(q5_reference_result const& expected,
                            table_with_names const& actual,
                            cuda::stream_ref stream)
{
  std::vector<std::string> const names{"n_name", "revenue"};
  auto const table = detail::reference_output(
    actual, names, {cudf::type_id::STRING, cudf::type_id::FLOAT64}, expected.revenue.size(), false);
  std::unordered_set<std::string> seen;
  double previous = std::numeric_limits<double>::infinity();
  detail::for_reference_batches(table, stream, [&](detail::reference_batch const& batch) {
    auto const countries = batch.strings(0);
    auto const revenues  = batch.values<double>(1);
    for (cudf::size_type i = 0; i < batch.count; ++i) {
      auto const group = expected.revenue.find(countries[i]);
      CUDF_EXPECTS(group != expected.revenue.end() && seen.insert(countries[i]).second,
                   "Unexpected or duplicate Q5 country: " + countries[i]);
      double const value = revenues[i], want = group->second;
      CUDF_EXPECTS(detail::reference_equal(value, want), "Q5 revenue mismatch: " + countries[i]);
      CUDF_EXPECTS(value <= previous, "Q5 revenues are not sorted descending");
      previous = value;
    }
  });
}
}  // namespace ndsh
