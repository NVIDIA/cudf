/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file q10_reference.hpp
 * @brief Independent CPU joins and output checks for Q10 customer revenue and attributes,
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
struct q10_customer_result {
  std::string name;
  double account_balance;
  std::string nation;
  std::string address;
  std::string phone;
  std::string comment;
  double revenue = 0;
};

struct q10_reference_result {
  std::map<int32_t, q10_customer_result> customers;
  int64_t matched = 0;
};

/**
 * add_table borrows one complete projected table per call in generator order, using bounded
 * host copies without retaining device views. finish requires all four tables, even if empty.
 */
class q10_reference_builder {
 public:
  void add_table(std::string const& name, cudf::table_view projected, cuda::stream_ref stream)
  {
    using enum cudf::type_id;
    detail::reference_input_schema(
      projected,
      name,
      next_table_,
      {{"nation", {STRING, INT8}},
       {"customer", {INT32, STRING, INT8, FLOAT64, STRING, STRING, STRING}},
       {"orders", {INT32, INT32, TIMESTAMP_DAYS}},
       {"lineitem", {FLOAT64, FLOAT64, INT32, STRING}}});

    // Filtered dimensions must also reject duplicates among discarded rows.
    std::unordered_set<int32_t> primary_keys;
    detail::for_reference_batches(projected, stream, [&](detail::reference_batch const& batch) {
      if (name == "nation") {
        auto const names = batch.strings(0);
        auto const keys  = batch.values<int8_t>(1);
        for (cudf::size_type i = 0; i < batch.count; ++i) {
          CUDF_EXPECTS(nations_.try_emplace(keys[i], names[i]).second, "Duplicate Q10 nation key");
        }
      } else if (name == "customer") {
        auto const keys      = batch.values<int32_t>(0);
        auto const names     = batch.strings(1);
        auto const nations   = batch.values<int8_t>(2);
        auto const balances  = batch.values<double>(3);
        auto const addresses = batch.strings(4);
        auto const phones    = batch.strings(5);
        auto const comments  = batch.strings(6);
        for (cudf::size_type i = 0; i < batch.count; ++i) {
          CUDF_EXPECTS(primary_keys.insert(keys[i]).second, "Duplicate Q10 customer key");
          auto const nation = nations_.find(nations[i]);
          if (nation == nations_.end()) continue;
          customers_.emplace(
            keys[i],
            q10_customer_result{
              names[i], balances[i], nation->second, addresses[i], phones[i], comments[i]});
        }
      } else if (name == "orders") {
        auto const customers = batch.values<int32_t>(0);
        auto const keys      = batch.values<int32_t>(1);
        auto const dates     = batch.values<cudf::timestamp_D>(2);
        for (cudf::size_type i = 0; i < batch.count; ++i) {
          CUDF_EXPECTS(primary_keys.insert(keys[i]).second, "Duplicate Q10 order key");
          auto const date = dates[i].time_since_epoch().count();
          // 1993-10-01 inclusive through 1994-01-01 exclusive, in epoch days.
          if (date < 8674 || date >= 8766 || !customers_.contains(customers[i])) continue;
          orders_.emplace(keys[i], customers[i]);
        }
      } else {
        auto const prices    = batch.values<double>(0);
        auto const discounts = batch.values<double>(1);
        auto const orders    = batch.values<int32_t>(2);
        auto const flags     = batch.strings(3);
        for (cudf::size_type i = 0; i < batch.count; ++i) {
          if (flags[i] != "R") continue;
          auto const order = orders_.find(orders[i]);
          if (order == orders_.end()) continue;
          ++result_.matched;
          auto const customer = customers_.find(order->second);
          CUDF_EXPECTS(customer != customers_.end(), "Missing Q10 customer for qualifying order");
          auto [result, inserted] = result_.customers.try_emplace(order->second, customer->second);
          result->second.revenue += prices[i] * (1.0 - discounts[i]);
        }
      }
    });
    ++next_table_;
  }

  q10_reference_result finish() const
  {
    CUDF_EXPECTS(next_table_ == 4, "Incomplete Q10 reference inputs");
    return result_;
  }

 private:
  std::size_t next_table_ = 0;
  std::unordered_map<int8_t, std::string> nations_;
  std::unordered_map<int32_t, q10_customer_result> customers_;
  std::unordered_map<int32_t, int32_t> orders_;
  q10_reference_result result_;
};

/**
 * Check complete Q10 output with exact customer attributes and descending revenue;
 * revenue uses the shared float tolerance, and ties may appear in any order.
 */
inline void check_q10_result(q10_reference_result const& expected,
                             table_with_names const& actual,
                             cuda::stream_ref stream)
{
  std::vector<std::string> const names{
    "c_custkey", "c_name", "c_acctbal", "c_phone", "n_name", "c_address", "c_comment", "revenue"};
  using enum cudf::type_id;
  auto const table =
    detail::reference_output(actual,
                             names,
                             {INT32, STRING, FLOAT64, STRING, STRING, STRING, STRING, FLOAT64},
                             expected.customers.size());
  std::unordered_set<int32_t> seen;
  double previous = std::numeric_limits<double>::infinity();
  detail::for_reference_batches(table, stream, [&](detail::reference_batch const& batch) {
    auto const keys           = batch.values<int32_t>(0);
    auto const customer_names = batch.strings(1);
    auto const balances       = batch.values<double>(2);
    auto const phones         = batch.strings(3);
    auto const nations        = batch.strings(4);
    auto const addresses      = batch.strings(5);
    auto const comments       = batch.strings(6);
    auto const revenues       = batch.values<double>(7);
    for (cudf::size_type i = 0; i < batch.count; ++i) {
      auto const customer = expected.customers.find(keys[i]);
      CUDF_EXPECTS(customer != expected.customers.end() && seen.insert(keys[i]).second,
                   "Unexpected or duplicate Q10 customer: " + std::to_string(keys[i]));
      auto const& want = customer->second;
      CUDF_EXPECTS(customer_names[i] == want.name && balances[i] == want.account_balance &&
                     nations[i] == want.nation && addresses[i] == want.address &&
                     phones[i] == want.phone && comments[i] == want.comment,
                   "Q10 customer attributes mismatch: " + std::to_string(keys[i]));
      double const value = revenues[i];
      CUDF_EXPECTS(detail::reference_equal(value, want.revenue),
                   "Q10 revenue mismatch: " + std::to_string(keys[i]));
      CUDF_EXPECTS(value <= previous, "Q10 revenues are not sorted descending");
      previous = value;
    }
  });
}
}  // namespace ndsh
