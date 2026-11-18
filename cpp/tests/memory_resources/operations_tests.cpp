/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "memory_resources_test_utilities.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/hashing.hpp>
#include <cudf/join/join.hpp>
#include <cudf/labeling/label_bins.hpp>
#include <cudf/reduction.hpp>
#include <cudf/reshape.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

using cudf::test::temporary_allocation_expectation;
using cudf::test::memory_resources_tests::expect_api_routes_memory_resources;
using cudf::test::memory_resources_tests::expect_column_api_routes_memory_resources;
using cudf::test::memory_resources_tests::expect_table_api_routes_memory_resources;

namespace {

auto make_numbers()
{
  return cudf::test::fixed_width_column_wrapper<int32_t>{{5, 4, 3, 2, 1, 0, -1, -2, 4, 3},
                                                         {1, 1, 0, 1, 1, 0, 1, 1, 1, 1}};
}

auto make_strings()
{
  return cudf::test::strings_column_wrapper{
    {"a", "bb", "", "dddd", "eeeee", "", "g", "hh", "bb", "a"}, {1, 1, 0, 1, 1, 1, 0, 1, 1, 1}};
}

}  // namespace

struct OperationsMemoryResourcesTest : public cudf::test::BaseFixture {};

TEST_F(OperationsMemoryResourcesTest, LabelBins)
{
  auto const input =
    cudf::test::fixed_width_column_wrapper<double>{{0.5, 1.5, 2.5, 3.5, 9.0}, {1, 1, 0, 1, 1}};
  auto left  = cudf::test::fixed_width_column_wrapper<double>{0, 1, 2, 3};
  auto right = cudf::test::fixed_width_column_wrapper<double>{1, 2, 3, 4};
  expect_column_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::label_bins(input,
                              left,
                              cudf::inclusive::YES,
                              right,
                              cudf::inclusive::NO,
                              cudf::test::get_default_stream(),
                              mr);
    },
    temporary_allocation_expectation::SOME);
}

TEST_F(OperationsMemoryResourcesTest, SortedOrder)
{
  auto numbers     = make_numbers();
  auto strings     = make_strings();
  auto const input = cudf::table_view{{strings, numbers}};
  expect_column_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::stable_sorted_order(input, {}, {}, cudf::test::get_default_stream(), mr);
    },
    temporary_allocation_expectation::SOME);
}

// stable_sort_by_key computes sorted_order with the current resource.
TEST_F(OperationsMemoryResourcesTest, DISABLED_Sort)
{
  auto numbers     = make_numbers();
  auto strings     = make_strings();
  auto const input = cudf::table_view{{strings, numbers}};
  expect_table_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::stable_sort(input, {}, {}, cudf::test::get_default_stream(), mr);
    },
    temporary_allocation_expectation::SOME);
}

// drop_nulls creates its keys table_device_view with the current resource.
TEST_F(OperationsMemoryResourcesTest, DISABLED_DropNulls)
{
  auto numbers     = make_numbers();
  auto strings     = make_strings();
  auto const input = cudf::table_view{{numbers, strings}};
  expect_table_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::drop_nulls(input, {0, 1}, cudf::test::get_default_stream(), mr);
    },
    temporary_allocation_expectation::SOME);
}

// distinct_indices checks for non-empty null rows with a column_device_view from the current
// resource.
TEST_F(OperationsMemoryResourcesTest, DISABLED_StableDistinct)
{
  auto numbers     = make_numbers();
  auto strings     = make_strings();
  auto const input = cudf::table_view{{numbers, strings}};
  expect_table_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::stable_distinct(input,
                                   {1},
                                   cudf::duplicate_keep_option::KEEP_FIRST,
                                   cudf::null_equality::EQUAL,
                                   cudf::nan_equality::ALL_EQUAL,
                                   cudf::test::get_default_stream(),
                                   mr);
    },
    temporary_allocation_expectation::SOME);
}

// binary_operation computes the output null count with the current resource.
TEST_F(OperationsMemoryResourcesTest, DISABLED_BinaryOperation)
{
  auto lhs = make_numbers();
  auto rhs = cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                                             {1, 0, 1, 1, 1, 1, 1, 1, 0, 1}};
  expect_column_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::binary_operation(lhs,
                                    rhs,
                                    cudf::binary_operator::ADD,
                                    cudf::data_type{cudf::type_id::INT64},
                                    cudf::test::get_default_stream(),
                                    mr);
    },
    temporary_allocation_expectation::SOME);
}

TEST_F(OperationsMemoryResourcesTest, Cast)
{
  auto input = make_numbers();
  expect_column_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::cast(
        input, cudf::data_type{cudf::type_id::FLOAT64}, cudf::test::get_default_stream(), mr);
    },
    temporary_allocation_expectation::NONE);
}

// murmurhash3_x86_32 checks for non-empty null rows with a column_device_view from the current
// resource.
TEST_F(OperationsMemoryResourcesTest, DISABLED_MurmurHash3)
{
  auto numbers     = make_numbers();
  auto strings     = make_strings();
  auto const input = cudf::table_view{{numbers, strings}};
  expect_column_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::hashing::murmurhash3_x86_32(
        input, cudf::DEFAULT_HASH_SEED, cudf::test::get_default_stream(), mr);
    },
    temporary_allocation_expectation::SOME);
}

// Strings interleave_columns allocates temporaries and thrust scratch from the current resource.
TEST_F(OperationsMemoryResourcesTest, DISABLED_InterleaveColumns)
{
  auto lhs         = make_strings();
  auto rhs         = make_strings();
  auto const input = cudf::table_view{{lhs, rhs}};
  expect_column_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::interleave_columns(input, cudf::test::get_default_stream(), mr);
    },
    temporary_allocation_expectation::SOME);
}

// dictionary::encode checks for non-empty null rows with a column_device_view from the current
// resource.
TEST_F(OperationsMemoryResourcesTest, DISABLED_DictionaryEncode)
{
  auto input = make_strings();
  expect_column_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::dictionary::encode(
        input, cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream(), mr);
    },
    temporary_allocation_expectation::SOME);
}

// simple_reduction allocates cub temporary storage from the current resource.
TEST_F(OperationsMemoryResourcesTest, DISABLED_ReduceSum)
{
  auto input             = make_numbers();
  auto const agg         = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  auto const output_type = cudf::data_type{cudf::type_id::INT64};
  expect_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::reduce(input, *agg, output_type, cudf::test::get_default_stream(), mr);
    },
    [&](auto const&) { return cudf::size_of(output_type) + sizeof(bool); },
    [](auto const& expected, auto const& result) {
      auto const stream = cudf::test::get_default_stream();
      using scalar_type = cudf::numeric_scalar<int64_t>;
      auto const& lhs   = static_cast<scalar_type const&>(*expected);
      auto const& rhs   = static_cast<scalar_type const&>(*result);
      EXPECT_EQ(lhs.is_valid(stream), rhs.is_valid(stream));
      EXPECT_EQ(lhs.value(stream), rhs.value(stream));
    },
    temporary_allocation_expectation::SOME);
}

// inner_join matches dictionaries and builds its hash_join with the current resource.
TEST_F(OperationsMemoryResourcesTest, DISABLED_InnerJoin)
{
  auto left_keys             = cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 2, 3, 4, 2, 7};
  auto right_keys            = cudf::test::fixed_width_column_wrapper<int32_t>{2, 4, 6, 0, 2};
  auto const left            = cudf::table_view{{left_keys}};
  auto const right           = cudf::table_view{{right_keys}};
  using index_vector         = std::unique_ptr<rmm::device_uvector<cudf::size_type>>;
  using join_result          = std::pair<index_vector, index_vector>;
  auto const to_sorted_pairs = [](join_result const& result) {
    auto const stream = cudf::test::get_default_stream();
    auto const lhs    = cudf::detail::make_std_vector(*result.first, stream);
    auto const rhs    = cudf::detail::make_std_vector(*result.second, stream);
    std::vector<std::pair<cudf::size_type, cudf::size_type>> pairs;
    std::transform(
      lhs.begin(), lhs.end(), rhs.begin(), std::back_inserter(pairs), [](auto l, auto r) {
        return std::pair{l, r};
      });
    std::sort(pairs.begin(), pairs.end());
    return pairs;
  };
  expect_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::inner_join(
        left, right, cudf::null_equality::EQUAL, cudf::test::get_default_stream(), mr);
    },
    [](join_result const& result) {
      return (result.first->size() + result.second->size()) * sizeof(cudf::size_type);
    },
    [&](join_result const& expected, join_result const& result) {
      EXPECT_EQ(to_sorted_pairs(expected), to_sorted_pairs(result));
    },
    temporary_allocation_expectation::SOME);
}
