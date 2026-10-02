/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/groupby.hpp>
#include <cudf/types.hpp>

#include <source_location>

enum class include_nth_aggregation : bool { NO, YES };
enum class test_streaming : bool { NO, YES };

void test_single_agg(cudf::column_view const& keys,
                     cudf::column_view const& values,
                     cudf::column_view const& expect_keys,
                     cudf::column_view const& expect_vals,
                     std::unique_ptr<cudf::groupby_aggregation>&& agg,
                     include_nth_aggregation include_nth          = include_nth_aggregation::NO,
                     cudf::null_policy include_null_keys          = cudf::null_policy::EXCLUDE,
                     cudf::sorted keys_are_sorted                 = cudf::sorted::NO,
                     std::vector<cudf::order> const& column_order = {},
                     std::vector<cudf::null_order> const& null_precedence = {},
                     cudf::sorted reference_keys_are_sorted               = cudf::sorted::NO,
                     test_streaming use_streaming                         = test_streaming::NO,
                     std::source_location const& location = std::source_location::current());
void test_sum_agg(cudf::column_view const& keys,
                  cudf::column_view const& values,
                  cudf::column_view const& expected_keys,
                  cudf::column_view const& expected_values,
                  std::source_location const& location = std::source_location::current());

void test_single_scan(cudf::column_view const& keys,
                      cudf::column_view const& values,
                      cudf::column_view const& expect_keys,
                      cudf::column_view const& expect_vals,
                      std::unique_ptr<cudf::groupby_scan_aggregation>&& agg,
                      cudf::null_policy include_null_keys          = cudf::null_policy::EXCLUDE,
                      cudf::sorted keys_are_sorted                 = cudf::sorted::NO,
                      std::vector<cudf::order> const& column_order = {},
                      std::vector<cudf::null_order> const& null_precedence = {},
                      std::source_location const& location = std::source_location::current());
