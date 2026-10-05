/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rolling.hpp"
#include "rolling_operators.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf {
namespace detail {
namespace {
struct is_supported_rolling_aggregation_impl {
  template <typename T, aggregation::Kind kind>
  constexpr bool operator()() const noexcept
  {
    return (kind == aggregation::Kind::LEAD || kind == aggregation::Kind::LAG ||
            kind == aggregation::Kind::COLLECT_LIST || kind == aggregation::Kind::COLLECT_SET) ||
           corresponding_rolling_operator<T, kind>::type::is_supported();
  }
};
}  // namespace

bool is_valid_rolling_aggregation(data_type source, aggregation::Kind kind)
{ return dispatch_type_and_aggregation(source, kind, is_supported_rolling_aggregation_impl{}); }
}  // namespace detail
}  // namespace cudf
