/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/memory_resource_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <functional>
#include <memory>
#include <utility>

namespace cudf::test::memory_resources_tests {

/**
 * @brief Run `api` with distinct output and poisoning temporary resources and compare its result
 * with a reference result computed using the current device resource.
 *
 * The API invocation runs with a current device resource that throws on allocation, so fallback to
 * the current resource fails the test. The returned object must own exactly the bytes allocated
 * from the output resource, and no temporary allocation may outlive the call.
 *
 * @param api Callable accepting `cudf::memory_resources` and returning an owning result
 * @param result_size Callable returning the number of output bytes owned by a result
 * @param expect_equal Callable comparing the reference result with the tested result
 * @param temporary Expected temporary-resource allocation activity
 */
template <typename Api, typename ResultSize, typename ExpectEqual>
void expect_api_routes_memory_resources(Api&& api,
                                        ResultSize&& result_size,
                                        ExpectEqual&& expect_equal,
                                        temporary_allocation_expectation temporary)
{
  auto const stream = cudf::test::get_default_stream();
  auto const reference =
    std::invoke(api, cudf::memory_resources{cudf::get_current_device_resource_ref()});
  auto harness = memory_resource_test_harness{cudf::get_current_device_resource_ref(),
                                              temporary_memory_checks::POISON};
  expect_api_uses_memory_resources(
    harness,
    std::forward<Api>(api),
    std::forward<ResultSize>(result_size),
    [&](auto const& result) { std::invoke(expect_equal, reference, result); },
    {output_allocation_expectation::EXACT, temporary},
    stream);
}

/**
 * @brief Column-returning overload of `expect_api_routes_memory_resources`.
 */
template <typename Api>
void expect_column_api_routes_memory_resources(Api&& api,
                                               temporary_allocation_expectation temporary)
{
  expect_api_routes_memory_resources(
    std::forward<Api>(api),
    [](std::unique_ptr<cudf::column> const& result) { return result->alloc_size(); },
    [](std::unique_ptr<cudf::column> const& expected, std::unique_ptr<cudf::column> const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view());
    },
    temporary);
}

/**
 * @brief Table-returning overload of `expect_api_routes_memory_resources`.
 */
template <typename Api>
void expect_table_api_routes_memory_resources(Api&& api, temporary_allocation_expectation temporary)
{
  expect_api_routes_memory_resources(
    std::forward<Api>(api),
    [](std::unique_ptr<cudf::table> const& result) { return result->alloc_size(); },
    [](std::unique_ptr<cudf::table> const& expected, std::unique_ptr<cudf::table> const& result) {
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());
    },
    temporary);
}

}  // namespace cudf::test::memory_resources_tests
