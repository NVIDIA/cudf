/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/detail/groupby/direct_groupby.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/groupby.hpp>
#include <cudf/groupby/direct_groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {

using path_t = cudf::groupby::detail::direct_aggregate_path;
using keys_t = cudf::test::fixed_width_column_wrapper<std::uint32_t>;

using agg_factory = std::function<std::unique_ptr<cudf::groupby_aggregation>()>;

std::vector<agg_factory> all_supported_aggs()
{
  return {
    [] { return cudf::make_sum_aggregation<cudf::groupby_aggregation>(); },
    [] { return cudf::make_min_aggregation<cudf::groupby_aggregation>(); },
    [] { return cudf::make_max_aggregation<cudf::groupby_aggregation>(); },
    [] { return cudf::make_count_aggregation<cudf::groupby_aggregation>(); },
    [] {
      return cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE);
    },
    [] { return cudf::make_mean_aggregation<cudf::groupby_aggregation>(); },
  };
}

std::vector<cudf::groupby::aggregation_request> make_requests(
  std::vector<cudf::column_view> const& values, std::vector<agg_factory> const& aggs)
{
  std::vector<cudf::groupby::aggregation_request> requests;
  for (auto const& v : values) {
    auto& request  = requests.emplace_back();
    request.values = v;
    for (auto const& make_agg : aggs) {
      request.aggregations.push_back(make_agg());
    }
  }
  return requests;
}

// Direct groupby must produce exactly the hash groupby output, sorted by key
void expect_matches_groupby(cudf::column_view const& keys,
                            std::vector<cudf::column_view> const& values,
                            std::vector<agg_factory> const& aggs,
                            std::size_t capacity,
                            path_t path)
{
  auto const direct_requests = make_requests(values, aggs);
  auto const [direct_keys, direct_results] =
    cudf::groupby::detail::direct_aggregate(keys,
                                            direct_requests,
                                            capacity,
                                            path,
                                            cudf::get_default_stream(),
                                            cudf::get_current_device_resource_ref());

  auto const hash_requests = make_requests(values, aggs);
  auto const [hash_keys, hash_results] =
    cudf::groupby::groupby{cudf::table_view{{keys}}}.aggregate(hash_requests);

  std::vector<cudf::column_view> hash_result_views;
  for (auto const& result : hash_results) {
    for (auto const& col : result.results) {
      hash_result_views.push_back(col->view());
    }
  }
  auto const sorted_keys = cudf::sort(hash_keys->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_keys->get_column(0).view(), direct_keys->view());
  ASSERT_EQ(direct_results.size(), hash_results.size());
  if (hash_result_views.empty()) { return; }

  auto const sorted_results =
    cudf::sort_by_key(cudf::table_view{hash_result_views}, hash_keys->view());
  cudf::size_type idx = 0;
  for (auto const& result : direct_results) {
    for (auto const& col : result.results) {
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_results->get_column(idx++).view(), col->view());
    }
  }
}

struct random_input {
  std::vector<std::uint32_t> keys;
  std::vector<std::int32_t> ints;
  std::vector<double> doubles;
  std::vector<bool> valids;
};

// Keys only hit every `stride`-th slot so the output must be compacted
random_input make_input(cudf::size_type num_rows, std::uint32_t capacity, std::uint32_t stride)
{
  random_input input;
  std::uint64_t state   = 0x9e3779b97f4a7c15ULL;
  auto const next       = [&] {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  auto const num_distinct = std::max<std::uint32_t>(1, capacity / stride);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    input.keys.push_back(static_cast<std::uint32_t>(next() % num_distinct) * stride);
    auto const v = static_cast<std::int32_t>(next() % 2001) - 1000;
    input.ints.push_back(v);
    input.doubles.push_back(static_cast<double>(v) * 0.5);
    input.valids.push_back(next() % 7 != 0);
  }
  return input;
}

}  // namespace

struct DirectGroupbyPathTest : public cudf::test::BaseFixture,
                               public ::testing::WithParamInterface<path_t> {};

INSTANTIATE_TEST_SUITE_P(Paths,
                         DirectGroupbyPathTest,
                         ::testing::Values(path_t::AUTO, path_t::SHARED_MEMORY, path_t::GLOBAL_MEMORY));

TEST_P(DirectGroupbyPathTest, SmallCapacityWithNulls)
{
  for (std::uint32_t capacity : {1u, 4u, 37u, 1000u}) {
    auto const input = make_input(20000, capacity, 3);
    auto const keys  = keys_t(input.keys.begin(), input.keys.end());
    auto const ints  = cudf::test::fixed_width_column_wrapper<std::int32_t>(
      input.ints.begin(), input.ints.end(), input.valids.begin());
    auto const doubles = cudf::test::fixed_width_column_wrapper<double>(input.doubles.begin(),
                                                                        input.doubles.end());
    expect_matches_groupby(keys, {ints, doubles}, all_supported_aggs(), capacity, GetParam());
  }
}

TEST_P(DirectGroupbyPathTest, Decimal128Sum)
{
  auto const input = make_input(20000, 1000, 3);
  auto const keys  = keys_t(input.keys.begin(), input.keys.end());
  auto const reps  = std::vector<__int128_t>(input.ints.begin(), input.ints.end());
  auto const vals  = cudf::test::fixed_point_column_wrapper<__int128_t>(
    reps.begin(), reps.end(), input.valids.begin(), numeric::scale_type{-2});
  std::vector<agg_factory> const aggs{
    [] { return cudf::make_sum_aggregation<cudf::groupby_aggregation>(); },
    [] { return cudf::make_count_aggregation<cudf::groupby_aggregation>(); },
    [] { return cudf::make_mean_aggregation<cudf::groupby_aggregation>(); },
  };
  expect_matches_groupby(keys, {vals}, aggs, 1000, GetParam());
}

TEST_P(DirectGroupbyPathTest, AllNullGroupIsRetained)
{
  auto const keys = keys_t{0, 2, 2, 5, 0, 5};
  auto const vals = cudf::test::fixed_width_column_wrapper<std::int64_t>{{1, 9, 9, 4, 3, 6},
                                                                          {1, 0, 0, 1, 1, 1}};
  expect_matches_groupby(keys, {vals}, all_supported_aggs(), 8, GetParam());
}

TEST_P(DirectGroupbyPathTest, EveryKeyOccupied)
{
  auto const keys = keys_t{3, 1, 0, 2, 1, 3, 0, 2};
  auto const vals = cudf::test::fixed_width_column_wrapper<std::int32_t>{5, 6, 7, 8, 1, 2, 3, 4};
  expect_matches_groupby(keys, {vals}, all_supported_aggs(), 4, GetParam());
}

TEST_P(DirectGroupbyPathTest, NoRequests)
{
  auto const keys = keys_t{7, 1, 7, 3};
  expect_matches_groupby(keys, {}, {}, 10, GetParam());
}

struct DirectGroupbyTest : public cudf::test::BaseFixture {};

TEST_F(DirectGroupbyTest, LargeSparseCapacity)
{
  auto const capacity = std::uint32_t{1} << 20;
  auto const input    = make_input(100000, capacity, 97);
  auto const keys     = keys_t(input.keys.begin(), input.keys.end());
  auto const ints     = cudf::test::fixed_width_column_wrapper<std::int32_t>(
    input.ints.begin(), input.ints.end(), input.valids.begin());
  for (auto path : {path_t::AUTO, path_t::GLOBAL_MEMORY}) {
    expect_matches_groupby(keys, {ints}, all_supported_aggs(), capacity, path);
  }
}

TEST_F(DirectGroupbyTest, EmptyKeys)
{
  auto const keys     = keys_t{};
  auto const vals     = cudf::test::fixed_width_column_wrapper<std::int32_t>{};
  auto const requests = make_requests({vals}, all_supported_aggs());
  auto const [out_keys, results] = cudf::groupby::direct_aggregate(keys, requests, 16);
  EXPECT_EQ(out_keys->size(), 0);
  EXPECT_EQ(out_keys->type().id(), cudf::type_id::UINT32);
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].results.size(), all_supported_aggs().size());
}

TEST_F(DirectGroupbyTest, SharedMemoryPathThrowsWhenCapacityDoesNotFit)
{
  auto const keys     = keys_t{0, 1};
  auto const vals     = cudf::test::fixed_width_column_wrapper<std::int64_t>{1, 2};
  auto const requests = make_requests({vals}, all_supported_aggs());
  EXPECT_THROW(std::ignore = cudf::groupby::detail::direct_aggregate(keys,
                                                                     requests,
                                                                     std::size_t{1} << 24,
                                                                     path_t::SHARED_MEMORY,
                                                                     cudf::get_default_stream(),
                                                                     cudf::get_current_device_resource_ref()),
               std::invalid_argument);
}

TEST_F(DirectGroupbyTest, InvalidInputs)
{
  auto const vals     = cudf::test::fixed_width_column_wrapper<std::int32_t>{1, 2, 3};
  auto const requests = make_requests({vals}, {all_supported_aggs().front()});

  auto const int32_keys = cudf::test::fixed_width_column_wrapper<std::int32_t>{0, 1, 2};
  EXPECT_THROW(std::ignore = cudf::groupby::direct_aggregate(int32_keys, requests, 4),
               cudf::data_type_error);

  auto const null_keys = cudf::test::fixed_width_column_wrapper<std::uint32_t>{{0, 1, 2}, {1, 0, 1}};
  EXPECT_THROW(std::ignore = cudf::groupby::direct_aggregate(null_keys, requests, 4),
               std::invalid_argument);

  auto const short_keys = keys_t{0, 1};
  EXPECT_THROW(std::ignore = cudf::groupby::direct_aggregate(short_keys, requests, 4),
               std::invalid_argument);

  auto const keys = keys_t{0, 1, 2};
  auto const nunique_requests =
    make_requests({vals}, {[] { return cudf::make_nunique_aggregation<cudf::groupby_aggregation>(); }});
  EXPECT_THROW(std::ignore = cudf::groupby::direct_aggregate(keys, nunique_requests, 4),
               std::invalid_argument);

  EXPECT_THROW(std::ignore = cudf::groupby::direct_aggregate(
                 keys, requests, static_cast<std::size_t>(1) << 40),
               std::invalid_argument);
}
