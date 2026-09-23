/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "reference_io.hpp"

#include <cudf/utilities/error.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ndsh {
using q9_group_key = std::pair<std::string, int16_t>;

struct q9_group_order {
  bool operator()(q9_group_key const& lhs, q9_group_key const& rhs) const
  {
    return lhs.first < rhs.first || (lhs.first == rhs.first && lhs.second > rhs.second);
  }
};

struct q9_reference_result {
  std::map<q9_group_key, double, q9_group_order> sum_profit;
  int64_t matched = 0;
};

namespace detail {
inline uint64_t q9_partsupp_key(int32_t supplier, int32_t part)
{
  return (uint64_t{static_cast<uint32_t>(supplier)} << 32) | static_cast<uint32_t>(part);
}
}  // namespace detail

/**
 * Setup-only CPU oracle: add_table borrows one complete, non-null projected table per call
 * in generator order, computing on bounded host copies without retaining device views.
 * finish requires all six tables, including empty ones. Duplicate partsupp pairs retain
 * join multiplicity.
 */
class q9_reference_builder {
 public:
  void add_table(std::string const& name, cudf::table_view projected, cuda::stream_ref stream)
  {
    using enum cudf::type_id;
    detail::reference_input_schema(projected,
                                   name,
                                   next_table_,
                                   {{"nation", {INT8, STRING}},
                                    {"supplier", {INT32, INT8}},
                                    {"partsupp", {INT32, INT32, FLOAT64}},
                                    {"orders", {INT32, TIMESTAMP_DAYS}},
                                    {"part", {INT32, STRING}},
                                    {"lineitem", {INT32, INT32, INT32, FLOAT64, FLOAT64, INT8}}});

    // Filtered dimensions must also reject duplicates among discarded rows.
    std::unordered_set<int32_t> primary_keys;

    detail::for_reference_batches(projected, stream, [&](detail::reference_batch const& batch) {
      if (name == "nation") {
        auto const keys  = batch.values<int8_t>(0);
        auto const names = batch.strings(1);
        for (cudf::size_type i = 0; i < batch.count; ++i) {
          CUDF_EXPECTS(nations_.try_emplace(keys[i], names[i]).second, "Duplicate Q9 nation key");
        }
      } else if (name == "supplier") {
        auto const keys    = batch.values<int32_t>(0);
        auto const nations = batch.values<int8_t>(1);
        for (cudf::size_type i = 0; i < batch.count; ++i) {
          CUDF_EXPECTS(primary_keys.insert(keys[i]).second, "Duplicate Q9 supplier key");
          if (nations_.contains(nations[i])) { suppliers_.emplace(keys[i], nations[i]); }
        }
      } else if (name == "partsupp") {
        auto const suppliers = batch.values<int32_t>(0);
        auto const parts     = batch.values<int32_t>(1);
        auto const costs     = batch.values<double>(2);
        for (cudf::size_type i = 0; i < batch.count; ++i) {
          auto const key = detail::q9_partsupp_key(suppliers[i], parts[i]);

          if (suppliers_.contains(suppliers[i])) { supply_costs_.emplace(key, costs[i]); }
        }
      } else if (name == "orders") {
        auto const keys  = batch.values<int32_t>(0);
        auto const dates = batch.values<cudf::timestamp_D>(1);
        for (cudf::size_type i = 0; i < batch.count; ++i) {
          CUDF_EXPECTS(!orders_.contains(keys[i]), "Duplicate Q9 order key");
          auto const date =
            std::chrono::sys_days{std::chrono::days{dates[i].time_since_epoch().count()}};
          auto const calendar = std::chrono::year_month_day{date};
          CUDF_EXPECTS(calendar.ok(), "Invalid Q9 order date");
          orders_.emplace(keys[i], static_cast<int16_t>(int{calendar.year()}));
        }
      } else if (name == "part") {
        auto const keys  = batch.values<int32_t>(0);
        auto const names = batch.strings(1);
        for (cudf::size_type i = 0; i < batch.count; ++i) {
          CUDF_EXPECTS(primary_keys.insert(keys[i]).second, "Duplicate Q9 part key");
          if (names[i].find("green") != std::string::npos) { green_parts_.insert(keys[i]); }
        }
      } else {
        auto const suppliers  = batch.values<int32_t>(0);
        auto const parts      = batch.values<int32_t>(1);
        auto const orders     = batch.values<int32_t>(2);
        auto const prices     = batch.values<double>(3);
        auto const discounts  = batch.values<double>(4);
        auto const quantities = batch.values<int8_t>(5);
        for (cudf::size_type i = 0; i < batch.count; ++i) {
          if (!green_parts_.contains(parts[i])) continue;
          auto const supplier = suppliers_.find(suppliers[i]);
          auto const order    = orders_.find(orders[i]);
          if (supplier == suppliers_.end() || order == orders_.end()) { continue; }
          auto const [cost_begin, cost_end] =
            supply_costs_.equal_range(detail::q9_partsupp_key(suppliers[i], parts[i]));
          for (auto cost = cost_begin; cost != cost_end; ++cost) {
            ++result_.matched;
            result_.sum_profit[{nations_.at(supplier->second), order->second}] +=
              prices[i] * (1.0 - discounts[i]) - cost->second * quantities[i];
          }
        }
      }
    });
    ++next_table_;
  }

  q9_reference_result finish() const
  {
    CUDF_EXPECTS(next_table_ == 6, "Incomplete Q9 reference inputs");
    return result_;
  }

 private:
  std::size_t next_table_ = 0;
  std::unordered_map<int8_t, std::string> nations_;
  std::unordered_map<int32_t, int8_t> suppliers_;
  std::unordered_multimap<uint64_t, double> supply_costs_;
  std::unordered_map<int32_t, int16_t> orders_;
  std::unordered_set<int32_t> green_parts_;
  q9_reference_result result_;
};

/** Check complete Q9 output ordered by nation ascending, then year descending. */
inline void check_q9_result(q9_reference_result const& expected,
                            table_with_names const& actual,
                            cuda::stream_ref stream)
{
  std::vector<std::string> const names{"nation", "o_year", "sum_profit"};
  using enum cudf::type_id;
  auto const table =
    detail::reference_output(actual, names, {STRING, INT16, FLOAT64}, expected.sum_profit.size());
  auto group = expected.sum_profit.begin();
  detail::for_reference_batches(table, stream, [&](detail::reference_batch const& batch) {
    auto const nations = batch.strings(0);
    auto const years   = batch.values<int16_t>(1);
    auto const profits = batch.values<double>(2);
    for (cudf::size_type i = 0; i < batch.count; ++i, ++group) {
      CUDF_EXPECTS(group->first == std::make_pair(nations[i], years[i]),
                   "Q9 ordered group keys mismatch");
      CUDF_EXPECTS(detail::reference_equal(profits[i], group->second), "Q9 sum_profit mismatch");
    }
  });
}
}  // namespace ndsh
