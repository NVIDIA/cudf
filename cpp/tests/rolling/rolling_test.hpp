/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/utilities/traits.hpp>

// return true if the aggregation is valid for the specified ColumnType
// valid aggregations may still be further specialized (eg, is_string_specialized)
template <typename ColumnType, cudf::aggregation::Kind op>
static constexpr bool is_rolling_supported()
{
  if (cudf::is_numeric<ColumnType>() or cudf::is_duration<ColumnType>()) {
    return (op == cudf::aggregation::SUM) or (op == cudf::aggregation::MIN) or
           (op == cudf::aggregation::MAX) or (op == cudf::aggregation::COUNT_VALID) or
           (op == cudf::aggregation::COUNT_ALL) or (op == cudf::aggregation::MEAN) or
           (op == cudf::aggregation::ROW_NUMBER) or (op == cudf::aggregation::LEAD) or
           (op == cudf::aggregation::LAG) or (op == cudf::aggregation::COLLECT_LIST);
  } else if (cudf::is_timestamp<ColumnType>()) {
    return (op == cudf::aggregation::MIN) or (op == cudf::aggregation::MAX) or
           (op == cudf::aggregation::COUNT_VALID) or (op == cudf::aggregation::COUNT_ALL) or
           (op == cudf::aggregation::ROW_NUMBER) or (op == cudf::aggregation::LEAD) or
           (op == cudf::aggregation::LAG) or (op == cudf::aggregation::COLLECT_LIST);
  } else if (cudf::is_fixed_point<ColumnType>()) {
    return (op == cudf::aggregation::SUM) or (op == cudf::aggregation::MIN) or
           (op == cudf::aggregation::MAX) or (op == cudf::aggregation::COUNT_VALID) or
           (op == cudf::aggregation::COUNT_ALL) or (op == cudf::aggregation::ROW_NUMBER) or
           (op == cudf::aggregation::LEAD) or (op == cudf::aggregation::LAG) or
           (op == cudf::aggregation::COLLECT_LIST);
  } else if (std::is_same<ColumnType, cudf::string_view>()) {
    return (op == cudf::aggregation::MIN) or (op == cudf::aggregation::MAX) or
           (op == cudf::aggregation::COUNT_VALID) or (op == cudf::aggregation::COUNT_ALL) or
           (op == cudf::aggregation::ROW_NUMBER) or (op == cudf::aggregation::COLLECT_LIST);

  } else if (std::is_same<ColumnType, cudf::list_view>()) {
    return (op == cudf::aggregation::COUNT_VALID) or (op == cudf::aggregation::COUNT_ALL) or
           (op == cudf::aggregation::ROW_NUMBER) or (op == cudf::aggregation::COLLECT_LIST);
  } else if (std::is_same<ColumnType, cudf::struct_view>()) {
    // TODO: Add support for COUNT_VALID, COUNT_ALL, ROW_NUMBER.
    return op == cudf::aggregation::COLLECT_LIST;
  } else {
    return false;
  }
}
