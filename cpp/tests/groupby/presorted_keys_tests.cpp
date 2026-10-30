/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace {

using int32s  = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64s  = cudf::test::fixed_width_column_wrapper<int64_t>;
using doubles = cudf::test::fixed_width_column_wrapper<double>;
using strings = cudf::test::strings_column_wrapper;
using structs = cudf::test::structs_column_wrapper;
using lists   = cudf::test::lists_column_wrapper<int32_t>;

using agg_factory  = std::function<std::unique_ptr<cudf::groupby_aggregation>()>;
using scan_factory = std::function<std::unique_ptr<cudf::groupby_scan_aggregation>()>;

std::vector<std::unique_ptr<cudf::column>> aggregate(cudf::column_view const& keys,
                                                     cudf::column_view const& values,
                                                     std::vector<agg_factory> const& aggs,
                                                     cudf::null_policy null_keys,
                                                     cudf::sorted keys_are_sorted,
                                                     std::unique_ptr<cudf::table>& out_keys)
{
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = values;
  for (auto const& make_agg : aggs) {
    requests[0].aggregations.push_back(make_agg());
  }
  // NTH_ELEMENT has no hash implementation, so the unsorted reference also uses sort groupby and
  // returns groups in key order.
  requests[0].aggregations.push_back(
    cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0));
  cudf::groupby::groupby gb(cudf::table_view{{keys}}, null_keys, keys_are_sorted);
  auto [result_keys, results] = gb.aggregate(requests);
  out_keys                    = std::move(result_keys);
  results[0].results.pop_back();
  return std::move(results[0].results);
}

std::unique_ptr<cudf::column> sort_list_elements(std::unique_ptr<cudf::column> col)
{
  return cudf::lists::sort_lists(
    cudf::lists_column_view{col->view()}, cudf::order::ASCENDING, cudf::null_order::AFTER);
}

void expect_well_formed_lists(cudf::column_view const& col)
{
  if (col.type().id() != cudf::type_id::LIST) { return; }
  auto const lcv     = cudf::lists_column_view{col};
  auto const offsets = cudf::test::to_host<int32_t>(lcv.offsets()).first;
  EXPECT_EQ(offsets.front(), 0);
  EXPECT_EQ(offsets.back(), lcv.child().size());
}

/**
 * @brief Checks that aggregating presorted keys with `sorted::YES` matches `sorted::NO`.
 *
 * The reference is computed with `sorted::NO` on deep copies of `keys` and `values`, so it never
 * sees a sliced input. `keys` and `values` are then aggregated as given with both `sorted::YES`
 * and `sorted::NO`, and the copies with `sorted::YES`.
 */
void expect_presorted_matches_unsorted(cudf::column_view const& keys,
                                       cudf::column_view const& values,
                                       std::vector<agg_factory> const& aggs,
                                       cudf::null_policy null_keys,
                                       bool unordered_list_elements = false)
{
  auto const keys_copy   = cudf::column(keys);
  auto const values_copy = cudf::column(values);

  std::unique_ptr<cudf::table> expected_keys;
  auto expected =
    aggregate(keys_copy, values_copy, aggs, null_keys, cudf::sorted::NO, expected_keys);
  if (unordered_list_elements) {
    for (auto& col : expected) {
      col = sort_list_elements(std::move(col));
    }
  }

  struct variant {
    char const* name;
    cudf::column_view keys;
    cudf::column_view values;
    cudf::sorted keys_are_sorted;
  };
  for (auto const& v : {variant{"input, sorted::YES", keys, values, cudf::sorted::YES},
                        variant{"input, sorted::NO", keys, values, cudf::sorted::NO},
                        variant{"copy, sorted::YES", keys_copy, values_copy, cudf::sorted::YES}}) {
    SCOPED_TRACE(v.name);
    std::unique_ptr<cudf::table> got_keys;
    auto got = aggregate(v.keys, v.values, aggs, null_keys, v.keys_are_sorted, got_keys);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_keys->view(), got_keys->view());
    for (std::size_t i = 0; i < aggs.size(); ++i) {
      SCOPED_TRACE("aggregation " + std::to_string(i));
      expect_well_formed_lists(got[i]->view());
      if (unordered_list_elements) { got[i] = sort_list_elements(std::move(got[i])); }
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected[i]->view(), got[i]->view());
    }
  }
}

// 20 ascending keys; the last two are null, which sorts them last.
int32s keys_with_nulls()
{
  return int32s{{0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 0, 0},
                {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0}};
}

int32s keys_without_nulls()
{
  return int32s{0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6};
}

int32s int_values()
{
  return int32s{{9, 3, 7, 1, 8, 2, 6, 5, 4, 0, 11, 15, 13, 12, 14, 10, 19, 17, 16, 18},
                {1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0}};
}

std::vector<agg_factory> fixed_width_aggs()
{
  using namespace cudf;
  return {
    [] { return make_sum_aggregation<groupby_aggregation>(); },
    [] { return make_product_aggregation<groupby_aggregation>(); },
    [] { return make_min_aggregation<groupby_aggregation>(); },
    [] { return make_max_aggregation<groupby_aggregation>(); },
    [] { return make_argmin_aggregation<groupby_aggregation>(); },
    [] { return make_argmax_aggregation<groupby_aggregation>(); },
    [] { return make_count_aggregation<groupby_aggregation>(); },
    [] { return make_count_aggregation<groupby_aggregation>(null_policy::INCLUDE); },
    [] { return make_mean_aggregation<groupby_aggregation>(); },
    [] { return make_m2_aggregation<groupby_aggregation>(); },
    [] { return make_variance_aggregation<groupby_aggregation>(); },
    [] { return make_std_aggregation<groupby_aggregation>(); },
    [] { return make_median_aggregation<groupby_aggregation>(); },
    [] { return make_quantile_aggregation<groupby_aggregation>({0.25, 0.75}); },
    [] { return make_nunique_aggregation<groupby_aggregation>(); },
    [] { return make_nth_element_aggregation<groupby_aggregation>(1); },
    [] { return make_nth_element_aggregation<groupby_aggregation>(-1, null_policy::EXCLUDE); },
    [] { return make_collect_list_aggregation<groupby_aggregation>(); },
    [] { return make_collect_list_aggregation<groupby_aggregation>(null_policy::EXCLUDE); },
    [] { return make_histogram_aggregation<groupby_aggregation>(); },
    [] { return make_bitwise_aggregation<groupby_aggregation>(bitwise_op::OR); },
    [] { return make_top_k_aggregation<groupby_aggregation>(2); },
    [] { return make_tdigest_aggregation<groupby_aggregation>(100); },
  };
}

void expect_each_and_all_fixed_width_aggs(cudf::column_view const& keys,
                                          cudf::column_view const& values,
                                          cudf::null_policy null_keys)
{
  auto const aggs = fixed_width_aggs();
  for (std::size_t i = 0; i < aggs.size(); ++i) {
    SCOPED_TRACE("fixed-width aggregation " + std::to_string(i));
    expect_presorted_matches_unsorted(keys, values, {aggs[i]}, null_keys);
  }
  // A single request exercises reuse of cached grouped and sorted values.
  SCOPED_TRACE("all fixed-width aggregations");
  expect_presorted_matches_unsorted(keys, values, aggs, null_keys);
}

}  // namespace

struct groupby_presorted_keys_test : public cudf::test::BaseFixture {};

TEST_F(groupby_presorted_keys_test, FixedWidthIncludeNullKeys)
{
  auto const keys   = keys_with_nulls();
  auto const values = int_values();
  expect_each_and_all_fixed_width_aggs(keys, values, cudf::null_policy::INCLUDE);
  expect_each_and_all_fixed_width_aggs(
    cudf::slice(keys, {2, 20})[0], cudf::slice(values, {2, 20})[0], cudf::null_policy::INCLUDE);
}

TEST_F(groupby_presorted_keys_test, FixedWidthExcludeNullKeys)
{
  auto const keys   = keys_with_nulls();
  auto const values = int_values();
  expect_each_and_all_fixed_width_aggs(keys, values, cudf::null_policy::EXCLUDE);
  expect_each_and_all_fixed_width_aggs(
    cudf::slice(keys, {3, 20})[0], cudf::slice(values, {3, 20})[0], cudf::null_policy::EXCLUDE);
}

TEST_F(groupby_presorted_keys_test, FixedWidthNoNullKeys)
{
  auto const keys   = keys_without_nulls();
  auto const values = int_values();
  expect_each_and_all_fixed_width_aggs(
    cudf::slice(keys, {1, 19})[0], cudf::slice(values, {1, 19})[0], cudf::null_policy::EXCLUDE);
}

TEST_F(groupby_presorted_keys_test, CollectSet)
{
  auto const keys   = keys_with_nulls();
  auto const values = int_values();
  expect_presorted_matches_unsorted(
    cudf::slice(keys, {2, 20})[0],
    cudf::slice(values, {2, 20})[0],
    {[] { return cudf::make_collect_set_aggregation<cudf::groupby_aggregation>(); }},
    cudf::null_policy::INCLUDE,
    true);
}

TEST_F(groupby_presorted_keys_test, Strings)
{
  auto const keys   = keys_with_nulls();
  auto const values = strings{{"j", "c", "h", "a", "i", "b", "g", "f", "e", "k",
                               "l", "p", "n", "m", "o", "q", "t", "r", "s", "u"},
                              {1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0}};
  using namespace cudf;
  expect_presorted_matches_unsorted(
    slice(keys, {2, 20})[0],
    slice(values, {2, 20})[0],
    {[] { return make_min_aggregation<groupby_aggregation>(); },
     [] { return make_max_aggregation<groupby_aggregation>(); },
     [] { return make_argmin_aggregation<groupby_aggregation>(); },
     [] { return make_nunique_aggregation<groupby_aggregation>(); },
     [] { return make_collect_list_aggregation<groupby_aggregation>(); }},
    null_policy::INCLUDE);
}

TEST_F(groupby_presorted_keys_test, Dictionary)
{
  auto const keys   = keys_with_nulls();
  auto const values = cudf::test::dictionary_column_wrapper<int32_t>(
    {9, 3, 7, 1, 8, 2, 6, 5, 4, 0, 11, 15, 13, 12, 14, 10, 19, 17, 16, 18},
    {1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0});
  using namespace cudf;
  expect_presorted_matches_unsorted(
    slice(keys, {2, 20})[0],
    slice(values, {2, 20})[0],
    {[] { return make_min_aggregation<groupby_aggregation>(); },
     [] { return make_max_aggregation<groupby_aggregation>(); },
     [] { return make_argmax_aggregation<groupby_aggregation>(); },
     [] { return make_count_aggregation<groupby_aggregation>(); },
     [] { return make_nth_element_aggregation<groupby_aggregation>(1); }},
    null_policy::INCLUDE);
}

TEST_F(groupby_presorted_keys_test, NestedValues)
{
  auto const keys = keys_with_nulls();
  auto c0         = int32s{{9, 3, 7, 1, 8, 2, 6, 5, 4, 0, 11, 15, 13, 12, 14, 10, 19, 17, 16, 18},
                           {1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0}};
  auto c1         = strings{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
                    "k", "l", "m", "n", "o", "p", "q", "r", "s", "t"};
  auto const struct_values =
    structs{{c0, c1}, {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
  auto const list_values =
    lists{{1, 2}, {3},      {},   {4, 5, 6}, {7},      {8, 8}, {9},  {10, 11}, {12}, {},
          {13},   {14, 15}, {16}, {17},      {18, 19}, {20},   {21}, {22, 23}, {24}, {25}};
  using namespace cudf;
  for (auto const& values : {column_view{struct_values}, column_view{list_values}}) {
    SCOPED_TRACE(values.type().id() == type_id::STRUCT ? "struct values" : "list values");
    expect_presorted_matches_unsorted(
      slice(keys, {2, 20})[0],
      slice(values, {2, 20})[0],
      {[] { return make_nth_element_aggregation<groupby_aggregation>(1); },
       [] { return make_collect_list_aggregation<groupby_aggregation>(); },
       [] { return make_count_aggregation<groupby_aggregation>(); },
       [] { return make_argmin_aggregation<groupby_aggregation>(); },
       [] { return make_min_aggregation<groupby_aggregation>(); },
       [] { return make_max_aggregation<groupby_aggregation>(); }},
      null_policy::INCLUDE);
  }
}

TEST_F(groupby_presorted_keys_test, MergeM2)
{
  auto const keys   = keys_without_nulls();
  auto counts       = int64s{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
  auto means        = doubles{1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10.,
                       11., 12., 13., 14., 15., 16., 17., 18., 19., 20.};
  auto m2s          = doubles{.5,  1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.,
                     5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10.};
  auto const values = structs{{counts, means, m2s}};
  expect_presorted_matches_unsorted(
    cudf::slice(keys, {4, 18})[0],
    cudf::slice(values, {4, 18})[0],
    {[] { return cudf::make_merge_m2_aggregation<cudf::groupby_aggregation>(); }},
    cudf::null_policy::EXCLUDE);
}

TEST_F(groupby_presorted_keys_test, MergeListsAndSets)
{
  auto const keys = keys_without_nulls();
  auto const values =
    lists{{1, 2}, {3},      {},   {4, 5, 6}, {7},      {8, 8}, {9},  {10, 11}, {12}, {},
          {13},   {14, 15}, {16}, {17},      {18, 19}, {20},   {21}, {22, 23}, {24}, {25}};
  auto const sliced_keys   = cudf::slice(keys, {5, 18})[0];
  auto const sliced_values = cudf::slice(values, {5, 18})[0];
  expect_presorted_matches_unsorted(
    sliced_keys,
    sliced_values,
    {[] { return cudf::make_merge_lists_aggregation<cudf::groupby_aggregation>(); }},
    cudf::null_policy::EXCLUDE);
  expect_presorted_matches_unsorted(
    sliced_keys,
    sliced_values,
    {[] { return cudf::make_merge_sets_aggregation<cudf::groupby_aggregation>(); }},
    cudf::null_policy::EXCLUDE,
    true);
}

TEST_F(groupby_presorted_keys_test, MergeHistogram)
{
  std::vector<int32_t> histogram_values;
  std::vector<int64_t> histogram_counts;
  std::vector<int32_t> offsets{0};
  for (int i = 0; i < 20; ++i) {
    histogram_values.push_back(i % 3);
    histogram_counts.push_back(i + 1);
    histogram_values.push_back(10 + i % 2);
    histogram_counts.push_back(2);
    offsets.push_back(2 * (i + 1));
  }
  auto entries = int32s(histogram_values.begin(), histogram_values.end());
  auto counts  = int64s(histogram_counts.begin(), histogram_counts.end());
  auto child   = structs{{entries, counts}};
  auto const values =
    cudf::make_lists_column(20,
                            int32s(offsets.begin(), offsets.end()).release(),
                            child.release(),
                            0,
                            cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
  auto const keys = keys_without_nulls();
  expect_presorted_matches_unsorted(
    cudf::slice(keys, {5, 18})[0],
    cudf::slice(values->view(), {5, 18})[0],
    {[] { return cudf::make_merge_histogram_aggregation<cudf::groupby_aggregation>(); }},
    cudf::null_policy::EXCLUDE,
    true);
}

TEST_F(groupby_presorted_keys_test, MergeTdigest)
{
  // One tdigest per key 0..19.
  std::vector<int32_t> digest_keys;
  std::vector<double> digest_inputs;
  for (int i = 0; i < 40; ++i) {
    digest_keys.push_back(i / 2);
    digest_inputs.push_back((i * 7) % 13);
  }
  auto const dk = int32s(digest_keys.begin(), digest_keys.end());
  auto const dv = doubles(digest_inputs.begin(), digest_inputs.end());
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = dv;
  requests[0].aggregations.push_back(
    cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(100));
  cudf::groupby::groupby gb(cudf::table_view{{dk}}, cudf::null_policy::EXCLUDE, cudf::sorted::YES);
  auto digests = std::move(gb.aggregate(requests).second[0].results[0]);
  ASSERT_EQ(digests->size(), 20);

  auto const keys = keys_without_nulls();
  expect_presorted_matches_unsorted(
    cudf::slice(keys, {5, 18})[0],
    cudf::slice(digests->view(), {5, 18})[0],
    {[] { return cudf::make_merge_tdigest_aggregation<cudf::groupby_aggregation>(100); }},
    cudf::null_policy::EXCLUDE);
}

TEST_F(groupby_presorted_keys_test, Covariance)
{
  auto const keys = keys_without_nulls();
  auto make_x     = [] {
    return doubles{{1., 5., 2., 8., 3., 9., 4., 7., 6., 0., 2., 4., 8., 6., 1., 3., 9., 5., 7., 2.},
                       {1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
  };
  auto make_y = [] {
    return doubles{3., 1., 4., 1., 5., 9., 2., 6., 5., 3., 5., 8., 9., 7., 9., 3., 2., 3., 8., 4.};
  };
  auto x0           = make_x();
  auto y0           = make_y();
  auto x1           = make_x();
  auto y1           = make_y();
  auto const values = structs{{x0, y0}};
  auto const nullable_values =
    structs{{x1, y1}, {1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1}};
  using namespace cudf;
  expect_presorted_matches_unsorted(
    keys,
    values,
    {[] { return make_covariance_aggregation<groupby_aggregation>(); },
     [] { return make_correlation_aggregation<groupby_aggregation>(correlation_type::PEARSON); }},
    null_policy::EXCLUDE);
  expect_presorted_matches_unsorted(
    keys,
    nullable_values,
    {[] { return make_covariance_aggregation<groupby_aggregation>(); }},
    null_policy::EXCLUDE);
}

namespace {

std::vector<std::unique_ptr<cudf::column>> scan(cudf::column_view const& keys,
                                                cudf::column_view const& values,
                                                std::vector<scan_factory> const& aggs,
                                                cudf::null_policy null_keys,
                                                cudf::sorted keys_are_sorted,
                                                std::unique_ptr<cudf::table>& out_keys)
{
  std::vector<cudf::groupby::scan_request> requests(1);
  requests[0].values = values;
  for (auto const& make_agg : aggs) {
    requests[0].aggregations.push_back(make_agg());
  }
  cudf::groupby::groupby gb(cudf::table_view{{keys}}, null_keys, keys_are_sorted);
  auto [result_keys, results] = gb.scan(requests);
  out_keys                    = std::move(result_keys);
  return std::move(results[0].results);
}

}  // namespace

TEST_F(groupby_presorted_keys_test, ScanExcludeNullKeys)
{
  auto const keys   = keys_with_nulls();
  auto const values = int_values();
  using namespace cudf;
  std::vector<scan_factory> const aggs{
    [] { return make_sum_aggregation<groupby_scan_aggregation>(); },
    [] { return make_min_aggregation<groupby_scan_aggregation>(); },
    [] { return make_max_aggregation<groupby_scan_aggregation>(); },
    [] { return make_count_aggregation<groupby_scan_aggregation>(); },
    [] { return make_rank_aggregation<groupby_scan_aggregation>(rank_method::MIN); }};

  for (auto const& [k, v] : {std::pair{column_view{keys}, column_view{values}},
                             std::pair{slice(keys, {3, 20})[0], slice(values, {3, 20})[0]}}) {
    std::unique_ptr<table> expected_keys;
    auto const expected =
      scan(column(k), column(v), aggs, null_policy::EXCLUDE, sorted::NO, expected_keys);
    std::unique_ptr<table> got_keys;
    auto const got = scan(k, v, aggs, null_policy::EXCLUDE, sorted::YES, got_keys);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_keys->view(), got_keys->view());
    for (std::size_t i = 0; i < aggs.size(); ++i) {
      SCOPED_TRACE("scan " + std::to_string(i));
      EXPECT_EQ(got[i]->size(), got_keys->num_rows());
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected[i]->view(), got[i]->view());
    }
  }
}
