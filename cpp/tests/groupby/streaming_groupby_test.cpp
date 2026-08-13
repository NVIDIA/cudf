/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <vector>

static std::vector<cudf::size_type> const KEY_COL{0};
static cudf::size_type constexpr DEFAULT_MAX_DISTINCT_KEYS = 1024;

namespace {

void sort_and_compare(std::unique_ptr<cudf::table>& lhs_keys,
                      std::vector<cudf::groupby::aggregation_result>& lhs_results,
                      std::unique_ptr<cudf::table>& rhs_keys,
                      std::vector<cudf::groupby::aggregation_result>& rhs_results,
                      std::vector<cudf::null_order> const& null_prec = {})
{
  auto const lhs_order = cudf::sorted_order(lhs_keys->view(), {}, null_prec);
  auto const rhs_order = cudf::sorted_order(rhs_keys->view(), {}, null_prec);

  EXPECT_EQ(lhs_keys->num_rows(), rhs_keys->num_rows());

  ASSERT_EQ(lhs_results.size(), rhs_results.size());
  for (size_t r = 0; r < lhs_results.size(); ++r) {
    ASSERT_EQ(lhs_results[r].results.size(), rhs_results[r].results.size());
    for (size_t c = 0; c < lhs_results[r].results.size(); ++c) {
      auto const lhs_sorted =
        cudf::gather(cudf::table_view{{lhs_results[r].results[c]->view()}}, *lhs_order);
      auto const rhs_sorted =
        cudf::gather(cudf::table_view{{rhs_results[r].results[c]->view()}}, *rhs_order);

      auto const& lhs_col = lhs_sorted->get_column(0);
      auto const& rhs_col = rhs_sorted->get_column(0);

      if (lhs_col.type() == rhs_col.type()) {
        CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(lhs_col, rhs_col);
      } else if (cudf::is_fixed_width(lhs_col.type()) && cudf::is_fixed_width(rhs_col.type())) {
        auto const lhs_cast = cudf::cast(lhs_col, rhs_col.type());
        CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*lhs_cast, rhs_col);
      } else {
        CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(lhs_col, rhs_col);
      }
    }
  }
}

void verify_against_groupby(
  std::unique_ptr<cudf::table>& streaming_keys,
  std::vector<cudf::groupby::aggregation_result>& streaming_results,
  std::vector<cudf::table_view> const& batches,
  std::vector<cudf::size_type> const& key_indices,
  std::vector<cudf::groupby::streaming_aggregation_request> const& requests,
  cudf::null_policy null_handling = cudf::null_policy::EXCLUDE)
{
  auto const all_data = cudf::concatenate(batches);

  std::vector<cudf::column_view> key_cols;
  for (auto idx : key_indices) {
    key_cols.push_back(all_data->view().column(idx));
  }
  cudf::groupby::groupby reference_groupby{cudf::table_view{key_cols}, null_handling};

  std::vector<cudf::groupby::aggregation_request> ref_requests;
  for (auto const& req : requests) {
    cudf::groupby::aggregation_request ref_req;
    ref_req.values = all_data->view().column(req.column_index);
    ref_req.aggregations.push_back(std::unique_ptr<cudf::groupby_aggregation>{
      dynamic_cast<cudf::groupby_aggregation*>(req.aggregation->clone().release())});
    ref_requests.push_back(std::move(ref_req));
  }

  auto [ref_keys, ref_results] = reference_groupby.aggregate(ref_requests);

  sort_and_compare(streaming_keys, streaming_results, ref_keys, ref_results);
}

void check(std::unique_ptr<cudf::table>& keys,
           std::vector<cudf::groupby::aggregation_result>& results,
           cudf::table_view expect_keys,
           std::vector<cudf::column_view> const& expect_vals)
{
  auto const order       = cudf::sorted_order(keys->view());
  auto const sorted_keys = cudf::gather(keys->view(), *order);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expect_keys, sorted_keys->view());

  size_t val_idx = 0;
  for (auto& agg_res : results) {
    for (auto& col : agg_res.results) {
      auto const sorted = cudf::gather(cudf::table_view{{col->view()}}, *order);
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect_vals[val_idx], sorted->get_column(0));
      ++val_idx;
    }
  }
}

cudf::groupby::streaming_aggregation_request make_req(
  cudf::size_type col_idx, std::unique_ptr<cudf::groupby_aggregation>&& agg)
{
  cudf::groupby::streaming_aggregation_request req;
  req.column_index = col_idx;
  req.aggregation  = std::move(agg);
  return req;
}

std::vector<cudf::groupby::streaming_aggregation_request> single_agg_req(
  cudf::size_type col_idx, std::unique_ptr<cudf::groupby_aggregation>&& agg)
{
  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(col_idx, std::move(agg)));
  return reqs;
}

}  // namespace

struct StreamingGroupbyTest : public cudf::test::BaseFixture {};

TEST_F(StreamingGroupbyTest, MemoryResource)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{1, 2, 3, 1};
  cudf::test::fixed_width_column_wrapper<int32_t> values{10, 20, 30, 40};
  cudf::table_view batch{{keys, values}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  auto mr   = rmm::mr::statistics_resource_adaptor(cudf::get_current_device_resource_ref());

  cudf::groupby::streaming_groupby streaming_agg(
    KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS, cudf::null_policy::EXCLUDE, mr);
  streaming_agg.aggregate(batch);

  EXPECT_GT(mr.get_bytes_counter().peak, 0);
}

TEST_F(StreamingGroupbyTest, SumTwoBatches)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 3, 1};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20, 30, 40};

  cudf::test::fixed_width_column_wrapper<K> keys2{2, 3, 1, 4};
  cudf::test::fixed_width_column_wrapper<V> vals2{5, 15, 25, 35};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, MinMaxTwoBatches)
{
  using K = int32_t;
  using V = double;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals1{5.0, 2.0, 8.0};

  cudf::test::fixed_width_column_wrapper<K> keys2{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<V> vals2{3.0, 9.0, 1.0};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(1, cudf::make_max_aggregation<cudf::groupby_aggregation>()));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, CountValidTwoBatches)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{{10, 20, 30, 40}, {true, false, true, true}};

  cudf::test::fixed_width_column_wrapper<K> keys2{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals2{{50, 60}, {false, true}};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, MeanTwoBatches)
{
  using K = int32_t;
  using V = double;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals1{10.0, 20.0, 30.0};

  cudf::test::fixed_width_column_wrapper<K> keys2{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals2{50.0, 40.0};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_mean_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, ProductTwoBatches)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{3, 5};

  cudf::test::fixed_width_column_wrapper<K> keys2{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals2{4, 2};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_product_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, MaxMinOnIntegers)
{
  using K = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> vals1{0, 1};

  cudf::test::fixed_width_column_wrapper<K> keys2{1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> vals2{1, 1};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(1, cudf::make_max_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>()));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, MergeTwoObjects)
{
  using K = int32_t;
  using V = int32_t;
  using R = int64_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20, 30};

  cudf::test::fixed_width_column_wrapper<K> keys2{2, 3};
  cudf::test::fixed_width_column_wrapper<V> vals2{40, 50};

  auto reqs1 = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby worker1(KEY_COL, reqs1, DEFAULT_MAX_DISTINCT_KEYS);
  worker1.aggregate(cudf::table_view{{keys1, vals1}});

  auto reqs2 = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby worker2(KEY_COL, reqs2, DEFAULT_MAX_DISTINCT_KEYS);
  worker2.aggregate(cudf::table_view{{keys2, vals2}});

  worker1.merge(worker2);
  auto [keys, results] = worker1.finalize();

  cudf::test::fixed_width_column_wrapper<K> ek{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R> ev{40, 60, 50};
  check(keys, results, cudf::table_view{{ek}}, {ev});
}

TEST_F(StreamingGroupbyTest, EmptyBatch)
{
  using K = int32_t;
  using V = int32_t;
  using R = int64_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20};

  cudf::test::fixed_width_column_wrapper<K> keys_empty{};
  cudf::test::fixed_width_column_wrapper<V> vals_empty{};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(cudf::table_view{{keys_empty, vals_empty}});
  streaming_agg.aggregate(cudf::table_view{{keys1, vals1}});
  auto [keys, results] = streaming_agg.finalize();

  cudf::test::fixed_width_column_wrapper<K> ek{1, 2};
  cudf::test::fixed_width_column_wrapper<R> ev{10, 20};
  check(keys, results, cudf::table_view{{ek}}, {ev});
}

TEST_F(StreamingGroupbyTest, SingleBatch)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20, 30, 40, 50};

  cudf::table_view batch1{{keys1, vals1}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, NewKeysInLaterBatches)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20};

  cudf::test::fixed_width_column_wrapper<K> keys2{3, 4};
  cudf::test::fixed_width_column_wrapper<V> vals2{30, 40};

  cudf::test::fixed_width_column_wrapper<K> keys3{1, 4};
  cudf::test::fixed_width_column_wrapper<V> vals3{50, 60};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};
  cudf::table_view batch3{{keys3, vals3}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  streaming_agg.aggregate(batch3);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2, batch3}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, MultipleRequestsOnDifferentColumns)
{
  using K = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1};
  cudf::test::fixed_width_column_wrapper<int32_t> col_a1{10, 20, 30};
  cudf::test::fixed_width_column_wrapper<double> col_b1{1.0, 2.0, 3.0};

  cudf::test::fixed_width_column_wrapper<K> keys2{2, 1};
  cudf::test::fixed_width_column_wrapper<int32_t> col_a2{40, 50};
  cudf::test::fixed_width_column_wrapper<double> col_b2{4.0, 5.0};

  cudf::table_view batch1{{keys1, col_a1, col_b1}};
  cudf::table_view batch2{{keys2, col_a2, col_b2}};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(2, cudf::make_min_aggregation<cudf::groupby_aggregation>()));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, FinalizeDoesNotModifyState)
{
  using K = int32_t;
  using V = int32_t;
  using R = int64_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20};

  cudf::test::fixed_width_column_wrapper<K> keys2{1};
  cudf::test::fixed_width_column_wrapper<V> vals2{30};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(cudf::table_view{{keys1, vals1}});

  {
    auto [k1, r1] = streaming_agg.finalize();
  }

  streaming_agg.aggregate(cudf::table_view{{keys2, vals2}});
  auto [keys, results] = streaming_agg.finalize();

  cudf::test::fixed_width_column_wrapper<K> ek{1, 2};
  cudf::test::fixed_width_column_wrapper<R> ev{40, 20};
  check(keys, results, cudf::table_view{{ek}}, {ev});
}

TEST_F(StreamingGroupbyTest, NullKeysExcluded)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{{1, 2, 3}, {true, false, true}};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20, 30};

  cudf::test::fixed_width_column_wrapper<K> keys2{{1, 2}, {true, false}};
  cudf::test::fixed_width_column_wrapper<V> vals2{40, 50};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(
    KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS, cudf::null_policy::EXCLUDE);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(
    keys, results, {batch1, batch2}, KEY_COL, reqs, cudf::null_policy::EXCLUDE);
}

TEST_F(StreamingGroupbyTest, NullKeysIncluded)
{
  using K = int32_t;
  using V = int32_t;
  using R = int64_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{{1, 2, 3}, {true, false, true}};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20, 30};

  cudf::test::fixed_width_column_wrapper<K> keys2{{1, 2}, {true, false}};
  cudf::test::fixed_width_column_wrapper<V> vals2{40, 50};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(
    KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS, cudf::null_policy::INCLUDE);
  streaming_agg.aggregate(cudf::table_view{{keys1, vals1}});
  streaming_agg.aggregate(cudf::table_view{{keys2, vals2}});
  auto [keys, results] = streaming_agg.finalize();

  EXPECT_EQ(keys->num_rows(), 3);

  auto const order       = cudf::sorted_order(keys->view(), {}, {cudf::null_order::AFTER});
  auto const sorted_keys = cudf::gather(keys->view(), *order);
  auto const sorted_vals = cudf::gather(cudf::table_view{{results[0].results[0]->view()}}, *order);

  cudf::test::fixed_width_column_wrapper<K> ek{{1, 3, 2}, {true, true, false}};
  cudf::test::fixed_width_column_wrapper<R> ev{50, 30, 70};
  CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view{{ek}}, sorted_keys->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(ev, sorted_vals->get_column(0));
}

TEST_F(StreamingGroupbyTest, AllNullKeysExcluded)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{{1, 2}, {false, false}};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(
    KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS, cudf::null_policy::EXCLUDE);
  streaming_agg.aggregate(cudf::table_view{{keys1, vals1}});
  auto [keys, results] = streaming_agg.finalize();

  EXPECT_EQ(keys->num_rows(), 0);
  EXPECT_EQ(results[0].results[0]->size(), 0);
}

template <typename V>
struct StreamingGroupbySumTypedTest : public cudf::test::BaseFixture {};

using SumSupportedTypes =
  cudf::test::Concat<cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>,
                     cudf::test::DurationTypes>;

TYPED_TEST_SUITE(StreamingGroupbySumTypedTest, SumSupportedTypes);

TYPED_TEST(StreamingGroupbySumTypedTest, TwoBatches)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  cudf::test::fixed_width_column_wrapper<K> keys2{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<V> vals2{10, 20, 30};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

template <typename V>
struct StreamingGroupbyMinTypedTest : public cudf::test::BaseFixture {};

using MinSupportedTypes =
  cudf::test::Concat<cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>,
                     cudf::test::DurationTypes>;

TYPED_TEST_SUITE(StreamingGroupbyMinTypedTest, MinSupportedTypes);

TYPED_TEST(StreamingGroupbyMinTypedTest, TwoBatches)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals1{5, 2, 8};

  cudf::test::fixed_width_column_wrapper<K> keys2{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals2{3, 9};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, VarianceBasic)
{
  using K = int32_t;
  using V = double;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 3, 1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{0, 1, 2, 3, 4};

  cudf::test::fixed_width_column_wrapper<K> keys2{2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals2{5, 6, 7, 8, 9};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_variance_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, StdBasic)
{
  using K = int32_t;
  using V = double;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 3, 1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{0, 1, 2, 3, 4};

  cudf::test::fixed_width_column_wrapper<K> keys2{2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals2{5, 6, 7, 8, 9};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_std_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, UnsupportedAggThrows)
{
  auto reqs = single_agg_req(1, cudf::make_collect_list_aggregation<cudf::groupby_aggregation>());
  EXPECT_THROW(cudf::groupby::streaming_groupby(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS),
               std::invalid_argument);
}

TEST_F(StreamingGroupbyTest, BatchExceedsMaxDistinctKeysThrows)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 4, 5};
  cudf::test::fixed_width_column_wrapper<V> vals{10, 20, 30, 40, 50};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, 3);
  EXPECT_THROW(streaming_agg.aggregate(cudf::table_view{{keys, vals}}), std::invalid_argument);
}

TEST_F(StreamingGroupbyTest, DisjointKeysAcrossBatches)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20};

  cudf::test::fixed_width_column_wrapper<K> keys2{3, 4};
  cudf::test::fixed_width_column_wrapper<V> vals2{30, 40};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, AllDuplicateKeysAcrossBatches)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20};

  cudf::test::fixed_width_column_wrapper<K> keys2{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals2{30, 40};

  cudf::test::fixed_width_column_wrapper<K> keys3{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals3{50, 60};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};
  cudf::table_view batch3{{keys3, vals3}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  streaming_agg.aggregate(batch3);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2, batch3}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, SingleRowBatches)
{
  using K = int32_t;
  using V = int32_t;

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);

  std::vector<cudf::table_view> batches;
  std::vector<std::unique_ptr<cudf::column>> key_owners;
  std::vector<std::unique_ptr<cudf::column>> val_owners;

  for (int32_t i = 0; i < 10; ++i) {
    auto k = std::make_unique<cudf::column>(cudf::test::fixed_width_column_wrapper<K>{i % 3});
    auto v = std::make_unique<cudf::column>(cudf::test::fixed_width_column_wrapper<V>{i * 10});
    cudf::table_view batch{{k->view(), v->view()}};
    streaming_agg.aggregate(batch);
    key_owners.push_back(std::move(k));
    val_owners.push_back(std::move(v));
  }

  auto [keys, results] = streaming_agg.finalize();
  EXPECT_EQ(keys->num_rows(), 3);
}

// Regression test for staging-offset corruption: when a batch has internal duplicate keys,
// the last distinct key's canonical position in the staging buffer equals the distinct-key
// count, not the end of the written range.  Without the fix, the next batch writes at the
// wrong offset and corrupts the canonical entry for that key.
TEST_F(StreamingGroupbyTest, InternalDuplicatesDoNotCorruptStaging)
{
  using K = int32_t;
  using V = int32_t;

  // Batch 1: key 1 is a duplicate at position 1, so key 3 ends up at position 3
  // (_num_unique_keys == 3).  Without the fix, batch 2 writes at offset 3, overwriting
  // the canonical slot for key 3 with key 4, making them appear identical.
  cudf::test::fixed_width_column_wrapper<K> keys1{1, 1, 2, 3};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 10, 20, 30};

  cudf::test::fixed_width_column_wrapper<K> keys2{4, 5};
  cudf::test::fixed_width_column_wrapper<V> vals2{40, 50};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, SumAndMeanOnSameColumn)
{
  using K = int32_t;
  using V = double;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals1{10.0, 20.0, 30.0};

  cudf::test::fixed_width_column_wrapper<K> keys2{2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals2{40.0, 50.0};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(1, cudf::make_mean_aggregation<cudf::groupby_aggregation>()));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

// Test that many small batches accumulate correctly within the fixed-capacity key table.
TEST_F(StreamingGroupbyTest, ManySmallBatches)
{
  using K = int32_t;
  using V = int32_t;

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  // 10 batches of 4 rows = 40 total rows; set max_distinct_keys=40 to fit all rows.
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, 40);

  std::vector<cudf::table_view> batches;
  std::vector<std::unique_ptr<cudf::column>> key_owners;
  std::vector<std::unique_ptr<cudf::column>> val_owners;

  for (int32_t b = 0; b < 10; ++b) {
    auto k = std::make_unique<cudf::column>(
      cudf::test::fixed_width_column_wrapper<K>{b % 8, (b + 1) % 8, (b + 2) % 8, (b + 3) % 8});
    auto v = std::make_unique<cudf::column>(
      cudf::test::fixed_width_column_wrapper<V>{b * 10, b * 10 + 1, b * 10 + 2, b * 10 + 3});
    cudf::table_view batch{{k->view(), v->view()}};
    streaming_agg.aggregate(batch);
    batches.push_back(batch);
    key_owners.push_back(std::move(k));
    val_owners.push_back(std::move(v));
  }

  auto [keys, results] = streaming_agg.finalize();
  verify_against_groupby(keys, results, batches, KEY_COL, reqs);
}

// Test that exceeding distinct-key capacity throws.
TEST_F(StreamingGroupbyTest, ExceedsDistinctKeyCapacityThrows)
{
  using K = int32_t;
  using V = int32_t;

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  // max_distinct_keys=4: can hold at most 4 distinct keys.
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, 4);

  // Batch with 4 distinct keys fills distinct-key capacity.
  cudf::test::fixed_width_column_wrapper<K> k1{0, 1, 2, 3};
  cudf::test::fixed_width_column_wrapper<V> v1{10, 20, 30, 40};
  streaming_agg.aggregate(cudf::table_view{{k1, v1}});

  // Any further batch with a new distinct key exceeds distinct-key capacity (4 + 1 > 4).
  cudf::test::fixed_width_column_wrapper<K> k2{4};
  cudf::test::fixed_width_column_wrapper<V> v2{50};
  EXPECT_THROW(streaming_agg.aggregate(cudf::table_view{{k2, v2}}), cudf::logic_error);
}

// Test that sliced input columns with non-zero offsets work correctly.
TEST_F(StreamingGroupbyTest, SlicedInputColumns)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> full_keys{0, 1, 2, 3, 1, 2};
  cudf::test::fixed_width_column_wrapper<V> full_vals{10, 20, 30, 40, 50, 60};

  // Slice to get a view with offset=2: keys={2,3,1,2}, vals={30,40,50,60}
  auto sliced = cudf::slice(cudf::table_view{{full_keys, full_vals}}, {2, 6});
  ASSERT_EQ(sliced[0].num_rows(), 4);

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(sliced[0]);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {sliced[0]}, KEY_COL, reqs);
}

// Test that finalize() before any aggregate() throws.
TEST_F(StreamingGroupbyTest, FinalizeBeforeAggregateThrows)
{
  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  EXPECT_THROW(static_cast<void>(streaming_agg.finalize()), cudf::logic_error);
}

// Test merge with MEAN aggregation (compound: SUM + COUNT intermediates).
TEST_F(StreamingGroupbyTest, MergeMeanTwoBatches)
{
  using K = int32_t;
  using V = double;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals1{10.0, 20.0, 30.0};

  cudf::test::fixed_width_column_wrapper<K> keys2{2, 1, 3};
  cudf::test::fixed_width_column_wrapper<V> vals2{40.0, 50.0, 60.0};

  auto reqs1 = single_agg_req(1, cudf::make_mean_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby worker1(KEY_COL, reqs1, DEFAULT_MAX_DISTINCT_KEYS);
  worker1.aggregate(cudf::table_view{{keys1, vals1}});

  auto reqs2 = single_agg_req(1, cudf::make_mean_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby worker2(KEY_COL, reqs2, DEFAULT_MAX_DISTINCT_KEYS);
  worker2.aggregate(cudf::table_view{{keys2, vals2}});

  worker1.merge(worker2);
  auto [keys, results] = worker1.finalize();

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};
  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs1);
}

// Test merge with COUNT_VALID aggregation (counts must be summed, not incremented).
TEST_F(StreamingGroupbyTest, MergeCountTwoBatches)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20, 30, 40};

  cudf::test::fixed_width_column_wrapper<K> keys2{2, 1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals2{50, 60, 70};

  auto reqs1 = single_agg_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));
  cudf::groupby::streaming_groupby worker1(KEY_COL, reqs1, DEFAULT_MAX_DISTINCT_KEYS);
  worker1.aggregate(cudf::table_view{{keys1, vals1}});

  auto reqs2 = single_agg_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));
  cudf::groupby::streaming_groupby worker2(KEY_COL, reqs2, DEFAULT_MAX_DISTINCT_KEYS);
  worker2.aggregate(cudf::table_view{{keys2, vals2}});

  worker1.merge(worker2);
  auto [keys, results] = worker1.finalize();

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};
  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs1);
}

// Test merge with VARIANCE aggregation (compound: SUM_OF_SQUARES + SUM + COUNT intermediates).
TEST_F(StreamingGroupbyTest, MergeVarianceTwoBatches)
{
  using K = int32_t;
  using V = double;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 3, 1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{0, 1, 2, 3, 4};

  cudf::test::fixed_width_column_wrapper<K> keys2{2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals2{5, 6, 7, 8, 9};

  auto reqs1 = single_agg_req(1, cudf::make_variance_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby worker1(KEY_COL, reqs1, DEFAULT_MAX_DISTINCT_KEYS);
  worker1.aggregate(cudf::table_view{{keys1, vals1}});

  auto reqs2 = single_agg_req(1, cudf::make_variance_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby worker2(KEY_COL, reqs2, DEFAULT_MAX_DISTINCT_KEYS);
  worker2.aggregate(cudf::table_view{{keys2, vals2}});

  worker1.merge(worker2);
  auto [keys, results] = worker1.finalize();

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};
  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs1);
}

TEST_F(StreamingGroupbyTest, SumOfSquaresBasic)
{
  using K = int32_t;
  using V = double;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals1{3.0, 4.0, 5.0};

  cudf::test::fixed_width_column_wrapper<K> keys2{2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals2{6.0, 7.0};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, M2Basic)
{
  using K = int32_t;
  using V = double;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{1.0, 2.0, 3.0, 4.0};

  cudf::test::fixed_width_column_wrapper<K> keys2{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals2{5.0, 6.0};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_m2_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, StdWithNullValues)
{
  using K = int32_t;
  using V = double;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{{1.0, 2.0, 3.0, 4.0}, {true, true, false, true}};

  cudf::test::fixed_width_column_wrapper<K> keys2{1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals2{{5.0, 6.0, 7.0}, {true, false, true}};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_std_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

// ===== String key tests =====

TEST_F(StreamingGroupbyTest, StringKeySumTwoBatches)
{
  using V = int32_t;

  cudf::test::strings_column_wrapper keys1{"a", "b", "c", "a"};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20, 30, 40};

  cudf::test::strings_column_wrapper keys2{"b", "c", "a", "d"};
  cudf::test::fixed_width_column_wrapper<V> vals2{5, 15, 25, 35};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, StringKeyNonAsciiUtf8)
{
  using V = int32_t;

  cudf::test::strings_column_wrapper keys1{"αλφα", "βητα", "γαμμα", "αλφα"};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20, 30, 40};

  cudf::test::strings_column_wrapper keys2{"βητα", "δελτα", "αλφα", "🙂"};
  cudf::test::fixed_width_column_wrapper<V> vals2{5, 15, 25, 35};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, StringKeyMinMaxTwoBatches)
{
  cudf::test::strings_column_wrapper keys1{"cat", "dog", "cat"};
  cudf::test::fixed_width_column_wrapper<double> vals1{5.0, 2.0, 8.0};

  cudf::test::strings_column_wrapper keys2{"cat", "dog", "bird"};
  cudf::test::fixed_width_column_wrapper<double> vals2{3.0, 9.0, 1.0};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(1, cudf::make_max_aggregation<cudf::groupby_aggregation>()));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, StringKeyManySmallBatches)
{
  using V = int32_t;

  std::vector<std::string> key_universe{"alpha", "beta", "gamma", "delta"};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);

  std::vector<cudf::table_view> batches;
  std::vector<std::unique_ptr<cudf::column>> key_owners;
  std::vector<std::unique_ptr<cudf::column>> val_owners;

  for (int32_t b = 0; b < 8; ++b) {
    auto k = std::make_unique<cudf::column>(
      cudf::test::strings_column_wrapper{key_universe[b % 4], key_universe[(b + 1) % 4]});
    auto v =
      std::make_unique<cudf::column>(cudf::test::fixed_width_column_wrapper<V>{b * 10, b * 10 + 1});
    cudf::table_view batch{{k->view(), v->view()}};
    streaming_agg.aggregate(batch);
    batches.push_back(batch);
    key_owners.push_back(std::move(k));
    val_owners.push_back(std::move(v));
  }

  auto [keys, results] = streaming_agg.finalize();
  verify_against_groupby(keys, results, batches, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, StringKeyDisjointBatches)
{
  using V = int32_t;

  cudf::test::strings_column_wrapper keys1{"x", "y"};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20};

  cudf::test::strings_column_wrapper keys2{"z", "w"};
  cudf::test::fixed_width_column_wrapper<V> vals2{30, 40};

  cudf::test::strings_column_wrapper keys3{"x", "w"};
  cudf::test::fixed_width_column_wrapper<V> vals3{50, 60};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};
  cudf::table_view batch3{{keys3, vals3}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  streaming_agg.aggregate(batch3);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2, batch3}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, StringKeyNullKeysExcluded)
{
  using V = int32_t;

  cudf::test::strings_column_wrapper keys1(std::initializer_list<std::string>{"a", "b", "c"},
                                           std::initializer_list<bool>{true, false, true});
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20, 30};

  cudf::test::strings_column_wrapper keys2(std::initializer_list<std::string>{"a", "b"},
                                           std::initializer_list<bool>{true, false});
  cudf::test::fixed_width_column_wrapper<V> vals2{40, 50};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(
    KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS, cudf::null_policy::EXCLUDE);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(
    keys, results, {batch1, batch2}, KEY_COL, reqs, cudf::null_policy::EXCLUDE);
}

TEST_F(StreamingGroupbyTest, StringKeyMerge)
{
  using V = int32_t;

  cudf::test::strings_column_wrapper keys1{"a", "b"};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20};

  cudf::test::strings_column_wrapper keys2{"b", "c"};
  cudf::test::fixed_width_column_wrapper<V> vals2{30, 40};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby obj1(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  obj1.aggregate(cudf::table_view{{keys1, vals1}});

  cudf::groupby::streaming_groupby obj2(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  obj2.aggregate(cudf::table_view{{keys2, vals2}});

  obj1.merge(obj2);
  auto [keys, results] = obj1.finalize();

  verify_against_groupby(keys,
                         results,
                         {cudf::table_view{{keys1, vals1}}, cudf::table_view{{keys2, vals2}}},
                         KEY_COL,
                         reqs);
}

TEST_F(StreamingGroupbyTest, CountAllTwoBatches)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals1{{10, 20, 30}, {true, false, true}};

  cudf::test::fixed_width_column_wrapper<K> keys2{2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals2{{40, 50}, {false, true}};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_count_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, MultiColumnKeys)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> k1a{1, 1, 2};
  cudf::test::fixed_width_column_wrapper<K> k1b{10, 20, 10};
  cudf::test::fixed_width_column_wrapper<V> v1{100, 200, 300};

  cudf::test::fixed_width_column_wrapper<K> k2a{1, 2};
  cudf::test::fixed_width_column_wrapper<K> k2b{10, 10};
  cudf::test::fixed_width_column_wrapper<V> v2{400, 500};

  cudf::table_view batch1{{k1a, k1b, v1}};
  cudf::table_view batch2{{k2a, k2b, v2}};

  std::vector<cudf::size_type> key_cols{0, 1};
  auto reqs = single_agg_req(2, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(key_cols, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, key_cols, reqs);
}

// A single batch larger than `max_distinct_keys` cannot be encoded because transient
// batch values (`max_distinct_keys + row_idx`) would collide with stored dense IDs.
TEST_F(StreamingGroupbyTest, BatchExceedingMaxDistinctKeysThrows)
{
  using K = int32_t;
  using V = int32_t;

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, 3);

  cudf::test::fixed_width_column_wrapper<K> k{0, 1, 2, 3};
  cudf::test::fixed_width_column_wrapper<V> v{10, 20, 30, 40};
  EXPECT_THROW(streaming_agg.aggregate(cudf::table_view{{k, v}}), std::invalid_argument);
}

// Cumulative input rows are not bounded by `max_distinct_keys` — only cumulative distinct
// keys are.  Re-feeding the same batch many times keeps distinct_keys constant and
// must never throw, regardless of how many cumulative rows have been processed.
TEST_F(StreamingGroupbyTest, CumulativeRowsCanExceedMaxDistinctKeys)
{
  using K = int32_t;
  using V = int32_t;

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, 3);

  cudf::test::fixed_width_column_wrapper<K> k{0, 1, 2};
  cudf::test::fixed_width_column_wrapper<V> v{10, 20, 30};
  cudf::table_view batch{{k, v}};

  // Five repeats: 15 cumulative rows >> max_distinct_keys=3, distinct_keys stays at 3.
  streaming_agg.aggregate(batch);
  streaming_agg.aggregate(batch);
  streaming_agg.aggregate(batch);
  streaming_agg.aggregate(batch);
  streaming_agg.aggregate(batch);

  EXPECT_EQ(streaming_agg.distinct_keys(), 3);

  auto [keys, results] = streaming_agg.finalize();
  verify_against_groupby(keys, results, {batch, batch, batch, batch, batch}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, StructKeySumTwoBatches)
{
  using V = int32_t;

  // Struct key: {int, int}
  cudf::test::fixed_width_column_wrapper<int32_t> s1a{1, 1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> s1b{10, 20, 10};
  auto keys1 = cudf::test::structs_column_wrapper{{s1a, s1b}};
  cudf::test::fixed_width_column_wrapper<V> vals1{100, 200, 300};

  cudf::test::fixed_width_column_wrapper<int32_t> s2a{1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> s2b{10, 10};
  auto keys2 = cudf::test::structs_column_wrapper{{s2a, s2b}};
  cudf::test::fixed_width_column_wrapper<V> vals2{400, 500};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}
