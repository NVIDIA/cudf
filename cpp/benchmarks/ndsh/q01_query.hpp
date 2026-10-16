/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file q01_query.hpp
 * @brief Shared Q1 execution for benchmarks and integration tests, aggregating lineitem
 * pricing by return flag and status with caller-supplied input and result handling.
 */

#pragma once

#include "utilities.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace ndsh::q1 {

inline std::vector<std::string> const q1_columns{"l_returnflag",
                                                 "l_linestatus",
                                                 "l_quantity",
                                                 "l_extendedprice",
                                                 "l_discount",
                                                 "l_shipdate",
                                                 "l_orderkey",
                                                 "l_tax"};

/**
 * @brief Calculate the discount price column
 *
 * @param discount The discount column
 * @param extendedprice The extended price column
 * @param stream The CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
[[nodiscard]] inline std::unique_ptr<cudf::column> calculate_disc_price(
  cudf::column_view const& discount,
  cudf::column_view const& extendedprice,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  auto const one = discount.type().id() == cudf::type_id::DECIMAL64
                     ? cudf::make_fixed_point_scalar<numeric::decimal64>(1L, numeric::scale_type{0})
                     : cudf::make_fixed_width_scalar<double>(1);
  auto const one_minus_discount =
    cudf::binary_operation(*one, discount, cudf::binary_operator::SUB, discount.type(), stream, mr);
  return cudf::binary_operation(extendedprice,
                                one_minus_discount->view(),
                                cudf::binary_operator::MUL,
                                discount.type(),
                                stream,
                                mr);
}

/**
 * @brief Calculate the charge column
 *
 * @param tax The tax column
 * @param disc_price The discount price column
 * @param stream The CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
[[nodiscard]] inline std::unique_ptr<cudf::column> calculate_charge(
  cudf::column_view const& tax,
  cudf::column_view const& disc_price,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  auto const one = tax.type().id() == cudf::type_id::DECIMAL64
                     ? cudf::make_fixed_point_scalar<numeric::decimal64>(1L, numeric::scale_type{0})
                     : cudf::make_fixed_width_scalar<double>(1);
  auto const one_plus_tax =
    cudf::binary_operation(*one, tax, cudf::binary_operator::ADD, tax.type(), stream, mr);
  return cudf::binary_operation(
    disc_price, one_plus_tax->view(), cudf::binary_operator::MUL, tax.type(), stream, mr);
}

/**
 * read returns an owning projected table, applying its predicate if filter_shipdate is false.
 * Otherwise filtering happens here. consume receives the result owner by reference and may
 * move it out; its return value is forwarded. This helper adds no final stream synchronization.
 */
template <typename Read, typename Consume>
auto execute_q1(Read&& read, bool filter_shipdate, Consume&& consume)
{
  auto const shipdate_ref = cudf::ast::column_reference(std::distance(
    q1_columns.begin(), std::find(q1_columns.begin(), q1_columns.end(), "l_shipdate")));
  auto shipdate_upper =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1998, 9, 2), true);
  auto const shipdate_upper_literal = cudf::ast::literal(shipdate_upper);
  auto const lineitem_pred          = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::LESS_EQUAL, shipdate_ref, shipdate_upper_literal);

  auto lineitem = read(q1_columns, lineitem_pred);
  if (filter_shipdate) { lineitem = apply_filter(lineitem, *lineitem_pred); }

  // Calculate the discount price and charge columns and append to lineitem table
  auto disc_price =
    calculate_disc_price(lineitem->column("l_discount"), lineitem->column("l_extendedprice"));
  auto charge = calculate_charge(lineitem->column("l_tax"), disc_price->view());
  (*lineitem).append(disc_price, "disc_price").append(charge, "charge");

  // Perform the group by operation
  auto const groupedby_table = apply_groupby(
    lineitem,
    groupby_context_t{
      {"l_returnflag", "l_linestatus"},
      {
        {"l_extendedprice",
         {{cudf::aggregation::Kind::SUM, "sum_base_price"},
          {cudf::aggregation::Kind::MEAN, "avg_price"}}},
        {"l_quantity",
         {{cudf::aggregation::Kind::SUM, "sum_qty"}, {cudf::aggregation::Kind::MEAN, "avg_qty"}}},
        {"l_discount",
         {
           {cudf::aggregation::Kind::MEAN, "avg_disc"},
         }},
        {"disc_price",
         {
           {cudf::aggregation::Kind::SUM, "sum_disc_price"},
         }},
        {"charge",
         {{cudf::aggregation::Kind::SUM, "sum_charge"},
          {cudf::aggregation::Kind::COUNT_ALL, "count_order"}}},
      }});

  // Perform the order by operation
  auto orderedby_table = apply_orderby(groupedby_table,
                                       {"l_returnflag", "l_linestatus"},
                                       {cudf::order::ASCENDING, cudf::order::ASCENDING});
  return consume(orderedby_table);
}

}  // namespace ndsh::q1
