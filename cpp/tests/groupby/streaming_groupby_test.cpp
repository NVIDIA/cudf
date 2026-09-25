/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
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
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <atomic>
#include <thread>
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

TEST_F(StreamingGroupbyTest, ConcurrentAggregate)
{
  using K = int32_t;
  using V = int32_t;

  constexpr int num_batches = 8;

  // Every batch re-hits keys 0 and 1 and introduces one key of its own, so concurrent calls
  // both collide on existing groups and discover new keys at the same time.
  std::vector<cudf::test::fixed_width_column_wrapper<K>> keys;
  std::vector<cudf::test::fixed_width_column_wrapper<V>> vals;
  keys.reserve(num_batches);
  vals.reserve(num_batches);
  for (int i = 0; i < num_batches; ++i) {
    keys.emplace_back(std::initializer_list<K>{0, 1, static_cast<K>(i + 2)});
    vals.emplace_back(std::initializer_list<V>{1, 10, 100});
  }

  std::vector<cudf::table_view> batches;
  batches.reserve(num_batches);
  for (int i = 0; i < num_batches; ++i) {
    batches.push_back(cudf::table_view{{keys[i], vals[i]}});
  }

  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  auto const stream_device = cuda::device_ref{device};
  std::vector<std::unique_ptr<cuda::stream>> streams;
  streams.reserve(num_batches);
  for (int i = 0; i < num_batches; ++i) {
    streams.push_back(std::make_unique<cuda::stream>(stream_device));
  }

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);

  std::vector<std::thread> threads;
  std::vector<std::exception_ptr> errors(num_batches);
  // `ready` lets the main thread wait until every worker is spinning, and `start` then releases
  // them together, so the aggregate() calls actually overlap.
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  threads.reserve(num_batches);
  for (int i = 0; i < num_batches; ++i) {
    threads.emplace_back([&, i] {
      CUDF_CUDA_TRY(cudaSetDevice(device));
      ready.fetch_add(1, std::memory_order_relaxed);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      try {
        streaming_agg.aggregate(batches[i], *streams[i]);
      } catch (...) {
        errors[i] = std::current_exception();
      }
    });
  }
  while (ready.load(std::memory_order_relaxed) != num_batches) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }
  for (auto const& error : errors) {
    EXPECT_FALSE(error);
  }
  for (auto const& stream : streams) {
    stream->sync();
  }

  auto [out_keys, results] = streaming_agg.finalize();
  verify_against_groupby(out_keys, results, batches, KEY_COL, reqs);
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

  // The object is now invalidated: even an empty batch is rejected, while finalize() still
  // recovers the groups.  The packed-key path these keys take finds the bound crossed only
  // after the batch has gone in, so the rejected batch's group is there too.
  EXPECT_EQ(streaming_agg.distinct_keys(), 5);
  cudf::test::fixed_width_column_wrapper<K> k_empty{};
  cudf::test::fixed_width_column_wrapper<V> v_empty{};
  EXPECT_THROW(streaming_agg.aggregate(cudf::table_view{{k_empty, v_empty}}), cudf::logic_error);
  auto [keys, results] = streaming_agg.finalize();
  EXPECT_EQ(keys->num_rows(), 5);
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

// ===== Packed-key table tests =====
// Integer keys whose widths fit in eight bytes between them, aggregated by SUM, MIN, MAX, COUNT
// or SUM_OF_SQUARES, are kept in the packed-key table rather than the generic hash set.  These
// tests cover what is particular to it: packing keys of several widths into one word, the key
// that equals the set's empty-slot sentinel, growth, chunking, nulls in keys and values, and
// merging.

TEST_F(StreamingGroupbyTest, PackedMixedWidthKeys)
{
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<int8_t> k1a{1, -1, 1, 2, -1};
  cudf::test::fixed_width_column_wrapper<int16_t> k1b{-300, 300, -300, 5, 300};
  cudf::test::fixed_width_column_wrapper<int32_t> k1c{7, 7, 7, 100000, 8};
  cudf::test::fixed_width_column_wrapper<V> v1{1, 2, 3, 4, 5};

  cudf::test::fixed_width_column_wrapper<int8_t> k2a{1, -1, 3};
  cudf::test::fixed_width_column_wrapper<int16_t> k2b{-300, 300, 0};
  cudf::test::fixed_width_column_wrapper<int32_t> k2c{7, 8, 0};
  cudf::test::fixed_width_column_wrapper<V> v2{10, 20, 30};

  cudf::table_view batch1{{k1a, k1b, k1c, v1}};
  cudf::table_view batch2{{k2a, k2b, k2c, v2}};

  std::vector<cudf::size_type> key_cols{0, 1, 2};
  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(3, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(3, cudf::make_min_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(3, cudf::make_max_aggregation<cudf::groupby_aggregation>()));

  cudf::groupby::streaming_groupby streaming_agg(key_cols, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  EXPECT_EQ(streaming_agg.distinct_keys(), 5);

  auto [keys, results] = streaming_agg.finalize();
  verify_against_groupby(keys, results, {batch1, batch2}, key_cols, reqs);
}

// A key whose packed bits are all ones equals the set's empty-slot sentinel; its rows go to the
// spare slot and it comes out as a group like any other.
TEST_F(StreamingGroupbyTest, PackedSentinelKey)
{
  using K = int64_t;
  using V = int32_t;

  constexpr K k_min = std::numeric_limits<K>::min();
  constexpr K k_max = std::numeric_limits<K>::max();

  cudf::test::fixed_width_column_wrapper<K> keys1{K{-1}, K{5}, K{-1}, k_max};
  cudf::test::fixed_width_column_wrapper<V> vals1{1, 2, 3, 4};
  cudf::test::fixed_width_column_wrapper<K> keys2{K{-1}, K{5}, k_min};
  cudf::test::fixed_width_column_wrapper<V> vals2{10, 20, 30};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE)));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  EXPECT_EQ(streaming_agg.distinct_keys(), 3);
  streaming_agg.aggregate(batch2);
  EXPECT_EQ(streaming_agg.distinct_keys(), 4);

  auto [keys, results] = streaming_agg.finalize();
  cudf::test::fixed_width_column_wrapper<K> ek{k_min, K{-1}, K{5}, k_max};
  cudf::test::fixed_width_column_wrapper<int64_t> esum{30, 14, 22, 4};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> ecount{1, 3, 2, 1};
  check(keys, results, cudf::table_view{{ek}}, {esum, ecount});
}

TEST_F(StreamingGroupbyTest, PackedSentinelKeyUnsigned)
{
  using K = uint64_t;
  using V = double;

  constexpr K k_max = std::numeric_limits<K>::max();

  cudf::test::fixed_width_column_wrapper<K> keys1{k_max, K{0}, k_max};
  cudf::test::fixed_width_column_wrapper<V> vals1{1.5, 2.5, 3.5};
  cudf::test::fixed_width_column_wrapper<K> keys2{K{0}, k_max, K{7}};
  cudf::test::fixed_width_column_wrapper<V> vals2{-1.0, -2.0, 0.0};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(1, cudf::make_max_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(1, cudf::make_mean_aggregation<cudf::groupby_aggregation>()));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  EXPECT_EQ(streaming_agg.distinct_keys(), 3);

  auto [keys, results] = streaming_agg.finalize();
  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

// The sentinel key's group merges like any other.
TEST_F(StreamingGroupbyTest, PackedSentinelKeyMerges)
{
  using K = int64_t;
  using V = int32_t;
  using R = int64_t;

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby worker1(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  cudf::groupby::streaming_groupby worker2(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);

  cudf::test::fixed_width_column_wrapper<K> k1{-1, 1};
  cudf::test::fixed_width_column_wrapper<V> v1{10, 20};
  worker1.aggregate(cudf::table_view{{k1, v1}});
  cudf::test::fixed_width_column_wrapper<K> k2{2, 3, 4, -1};
  cudf::test::fixed_width_column_wrapper<V> v2{30, 40, 50, 60};
  worker1.aggregate(cudf::table_view{{k2, v2}});
  EXPECT_EQ(worker1.distinct_keys(), 5);

  cudf::test::fixed_width_column_wrapper<K> k3{-1, 5, 1};
  cudf::test::fixed_width_column_wrapper<V> v3{100, 200, 300};
  worker2.aggregate(cudf::table_view{{k3, v3}});

  worker1.merge(worker2);
  EXPECT_EQ(worker1.distinct_keys(), 6);

  auto [keys, results] = worker1.finalize();
  cudf::test::fixed_width_column_wrapper<K> ek{-1, 1, 2, 3, 4, 5};
  cudf::test::fixed_width_column_wrapper<R> ev{170, 320, 30, 40, 50, 200};
  check(keys, results, cudf::table_view{{ek}}, {ev});
}

// Many distinct two-column keys, exactly as many as `max_distinct_keys`, each group with several
// accumulators.
TEST_F(StreamingGroupbyTest, PackedManyKeysUpToMaxDistinctKeys)
{
  using V = int32_t;

  constexpr int num_batches = 10;
  constexpr int batch_rows  = 500;

  std::vector<cudf::size_type> key_cols{0, 1};
  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(2, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(2, cudf::make_min_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(2, cudf::make_max_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(
    2, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE)));
  constexpr cudf::size_type num_distinct = num_batches * (batch_rows / 2) + batch_rows / 2;
  cudf::groupby::streaming_groupby streaming_agg(key_cols, reqs, num_distinct);

  std::vector<std::unique_ptr<cudf::column>> owners;
  std::vector<cudf::table_view> batches;
  for (int b = 0; b < num_batches; ++b) {
    // The first half of every batch is keys of its own, the second half keys every batch has.
    std::vector<int32_t> ka(batch_rows);
    std::vector<int16_t> kb(batch_rows);
    std::vector<V> vals(batch_rows);
    for (int i = 0; i < batch_rows; ++i) {
      ka[i]   = i < batch_rows / 2 ? b * batch_rows + i : i;
      kb[i]   = static_cast<int16_t>(i % 3 - 1);
      vals[i] = (b * 1000 + i) % 97 - 48;
    }
    auto a = std::make_unique<cudf::column>(
      cudf::test::fixed_width_column_wrapper<int32_t>(ka.begin(), ka.end()));
    auto c = std::make_unique<cudf::column>(
      cudf::test::fixed_width_column_wrapper<int16_t>(kb.begin(), kb.end()));
    auto v = std::make_unique<cudf::column>(
      cudf::test::fixed_width_column_wrapper<V>(vals.begin(), vals.end()));
    batches.push_back(cudf::table_view{{a->view(), c->view(), v->view()}});
    owners.push_back(std::move(a));
    owners.push_back(std::move(c));
    owners.push_back(std::move(v));
    streaming_agg.aggregate(batches.back());
  }
  EXPECT_EQ(streaming_agg.distinct_keys(), num_distinct);

  auto [keys, results] = streaming_agg.finalize();
  verify_against_groupby(keys, results, batches, key_cols, reqs);
}

// A batch of more rows than the minimum chunk, over a few keys.
TEST_F(StreamingGroupbyTest, PackedLargeBatchFewKeys)
{
  using K = int32_t;
  using V = int64_t;

  constexpr cudf::size_type num_rows = (1 << 18) + 12'345;
  constexpr K num_groups             = 9;

  std::vector<K> key_values(num_rows);
  std::vector<V> val_values(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    key_values[i] = (i * 7) % num_groups;
    val_values[i] = i % 11 - 5;
  }
  cudf::test::fixed_width_column_wrapper<K> keys(key_values.begin(), key_values.end());
  cudf::test::fixed_width_column_wrapper<V> vals(val_values.begin(), val_values.end());
  cudf::table_view batch{{keys, vals}};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE)));
  reqs.push_back(make_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>()));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, num_rows);
  streaming_agg.aggregate(batch);
  streaming_agg.aggregate(batch);
  EXPECT_EQ(streaming_agg.distinct_keys(), num_groups);

  auto [out_keys, results] = streaming_agg.finalize();
  verify_against_groupby(out_keys, results, {batch, batch}, KEY_COL, reqs);
}

// Null values arriving after groups exist: the groups made before stay valid, a group that only
// ever saw nulls is null, and a count of all rows still counts them.
TEST_F(StreamingGroupbyTest, PackedNullValuesInLaterBatch)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20, 30};
  cudf::test::fixed_width_column_wrapper<K> keys2{1, 4, 4, 2, 3};
  cudf::test::fixed_width_column_wrapper<V> vals2{{0, 0, 0, 5, 0},
                                                  {false, false, false, true, false}};
  cudf::test::fixed_width_column_wrapper<K> keys3{4, 5};
  cudf::test::fixed_width_column_wrapper<V> vals3{{0, 50}, {false, true}};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};
  cudf::table_view batch3{{keys3, vals3}};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(1, cudf::make_max_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE)));
  reqs.push_back(make_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE)));
  reqs.push_back(make_req(1, cudf::make_mean_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(1, cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>()));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  streaming_agg.aggregate(batch3);
  EXPECT_EQ(streaming_agg.distinct_keys(), 5);

  auto [keys, results] = streaming_agg.finalize();
  verify_against_groupby(keys, results, {batch1, batch2, batch3}, KEY_COL, reqs);

  // Group 4 only ever saw nulls: its MIN is null, its counts are 0 and 3.
  auto const order   = cudf::sorted_order(keys->view());
  auto const min_col = cudf::gather(cudf::table_view{{results[0].results[0]->view()}}, *order);
  cudf::test::fixed_width_column_wrapper<V> emin{{10, 5, 30, 0, 50},
                                                 {true, true, true, false, true}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(emin, min_col->get_column(0));
}

// With `null_policy::INCLUDE`, a validity bit per key column is part of the packed key, so a
// null in either of two key columns makes a group of its own.
TEST_F(StreamingGroupbyTest, PackedNullKeysIncludedMultiColumn)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> k1a{{1, 1, 2, 2}, {true, false, true, true}};
  cudf::test::fixed_width_column_wrapper<K> k1b{{5, 5, 5, 0}, {true, true, false, true}};
  cudf::test::fixed_width_column_wrapper<V> v1{1, 2, 3, 4};
  cudf::test::fixed_width_column_wrapper<K> k2a{{1, 2, 1}, {false, true, true}};
  cudf::test::fixed_width_column_wrapper<K> k2b{{5, 5, 5}, {true, false, true}};
  cudf::test::fixed_width_column_wrapper<V> v2{10, 20, 30};

  cudf::table_view batch1{{k1a, k1b, v1}};
  cudf::table_view batch2{{k2a, k2b, v2}};

  std::vector<cudf::size_type> key_cols{0, 1};
  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(2, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(2, cudf::make_max_aggregation<cudf::groupby_aggregation>()));

  cudf::groupby::streaming_groupby streaming_agg(
    key_cols, reqs, DEFAULT_MAX_DISTINCT_KEYS, cudf::null_policy::INCLUDE);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  EXPECT_EQ(streaming_agg.distinct_keys(), 4);

  auto [keys, results] = streaming_agg.finalize();
  verify_against_groupby(
    keys, results, {batch1, batch2}, key_cols, reqs, cudf::null_policy::INCLUDE);

  // The output key columns carry the null groups.
  auto const order =
    cudf::sorted_order(keys->view(), {}, {cudf::null_order::AFTER, cudf::null_order::AFTER});
  auto const sorted_keys = cudf::gather(keys->view(), *order);
  cudf::test::fixed_width_column_wrapper<K> eka{{1, 2, 2, 1}, {true, true, true, false}};
  cudf::test::fixed_width_column_wrapper<K> ekb{{5, 0, 5, 5}, {true, true, false, true}};
  CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view{{eka, ekb}}, sorted_keys->view());
}

// A key column that is nullable only from the second batch on: its null rows are skipped, and
// the output key column is nullable as the input was.
TEST_F(StreamingGroupbyTest, PackedNullKeysExcludedFromLaterBatch)
{
  using K = int32_t;
  using V = int32_t;
  using R = int64_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20, 30};
  cudf::test::fixed_width_column_wrapper<K> keys2{{1, 2, 4}, {true, false, true}};
  cudf::test::fixed_width_column_wrapper<V> vals2{1, 2, 4};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(cudf::table_view{{keys1, vals1}});
  streaming_agg.aggregate(cudf::table_view{{keys2, vals2}});
  EXPECT_EQ(streaming_agg.distinct_keys(), 4);

  auto [keys, results] = streaming_agg.finalize();
  EXPECT_TRUE(keys->get_column(0).nullable());
  EXPECT_EQ(keys->get_column(0).null_count(), 0);
  cudf::test::fixed_width_column_wrapper<K> ek{{1, 2, 3, 4}, cudf::test::iterators::no_nulls()};
  cudf::test::fixed_width_column_wrapper<R> ev{11, 20, 30, 4};
  check(keys, results, cudf::table_view{{ek}}, {ev});
}

// Merging when only one side has seen null values, in either direction: the side without nulls
// has every aggregate valid, the side with them has a group that is null.
TEST_F(StreamingGroupbyTest, PackedMergeNullValuesOnOneSide)
{
  using K = int32_t;
  using V = int32_t;

  auto const run = [](bool nulls_in_target) {
    auto reqs = single_agg_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>());
    cudf::groupby::streaming_groupby plain(KEY_COL, reqs, 4);
    cudf::groupby::streaming_groupby with_nulls(KEY_COL, reqs, 4);

    cudf::test::fixed_width_column_wrapper<K> k1{1, 2};
    cudf::test::fixed_width_column_wrapper<V> v1{10, 20};
    plain.aggregate(cudf::table_view{{k1, v1}});

    cudf::test::fixed_width_column_wrapper<K> k2{2, 3, 3, 4};
    cudf::test::fixed_width_column_wrapper<V> v2{{15, 0, 0, 40}, {true, false, false, true}};
    with_nulls.aggregate(cudf::table_view{{k2, v2}});

    auto& target = nulls_in_target ? with_nulls : plain;
    auto& source = nulls_in_target ? plain : with_nulls;
    target.merge(source);
    EXPECT_EQ(target.distinct_keys(), 4);

    auto [keys, results] = target.finalize();
    cudf::test::fixed_width_column_wrapper<K> ek{1, 2, 3, 4};
    cudf::test::fixed_width_column_wrapper<V> ev{{10, 15, 0, 40}, {true, true, false, true}};
    check(keys, results, cudf::table_view{{ek}}, {ev});
  };
  run(false);
  run(true);
}

TEST_F(StreamingGroupbyTest, PackedBoolAndTimestampKeys)
{
  using ts_ms = cudf::timestamp_ms;
  using dur_s = cudf::duration_s;
  using ts_ns = cudf::timestamp_ns;

  cudf::test::fixed_width_column_wrapper<bool> k1a{true, false, true, true};
  cudf::test::fixed_width_column_wrapper<ts_ms, int64_t> k1b{1000, 2000, 1000, -5};
  cudf::test::fixed_width_column_wrapper<dur_s, int64_t> v1a{1, 2, 3, 4};
  cudf::test::fixed_width_column_wrapper<ts_ns, int64_t> v1b{100, 200, 50, -7};

  cudf::test::fixed_width_column_wrapper<bool> k2a{false, true, false};
  cudf::test::fixed_width_column_wrapper<ts_ms, int64_t> k2b{2000, 1000, 1000};
  cudf::test::fixed_width_column_wrapper<dur_s, int64_t> v2a{10, 20, 30};
  cudf::test::fixed_width_column_wrapper<ts_ns, int64_t> v2b{-1, 300, 0};

  cudf::table_view batch1{{k1a, k1b, v1a, v1b}};
  cudf::table_view batch2{{k2a, k2b, v2a, v2b}};

  std::vector<cudf::size_type> key_cols{0, 1};
  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(2, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(2, cudf::make_min_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(3, cudf::make_max_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(3, cudf::make_min_aggregation<cudf::groupby_aggregation>()));

  cudf::groupby::streaming_groupby streaming_agg(key_cols, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  EXPECT_EQ(streaming_agg.distinct_keys(), 4);

  auto [keys, results] = streaming_agg.finalize();
  verify_against_groupby(keys, results, {batch1, batch2}, key_cols, reqs);
}

// Float values: SUM accumulates in double and narrows to float; MIN and MAX order by the bits.
TEST_F(StreamingGroupbyTest, PackedFloat32Values)
{
  using K = int16_t;
  using V = float;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{1.5f, -2.25f, 0.f, -0.5f, 8.f};
  cudf::test::fixed_width_column_wrapper<K> keys2{3, 1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals2{-1e6f, 1e-3f, -7.f};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(1, cudf::make_max_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(1, cudf::make_mean_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(1, cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>()));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);

  auto [keys, results] = streaming_agg.finalize();
  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

// Unsigned values above the signed range compare as unsigned, and small unsigned sums widen.
TEST_F(StreamingGroupbyTest, PackedUnsignedValues)
{
  using K = uint8_t;

  constexpr uint64_t big = std::numeric_limits<uint64_t>::max();
  constexpr uint64_t mid = uint64_t{1} << 63;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1, 255};
  cudf::test::fixed_width_column_wrapper<uint64_t> v1a{big, uint64_t{1}, mid, uint64_t{0}};
  cudf::test::fixed_width_column_wrapper<uint8_t> v1b{200, 100, 200, 255};
  cudf::test::fixed_width_column_wrapper<K> keys2{2, 255};
  cudf::test::fixed_width_column_wrapper<uint64_t> v2a{mid, big};
  cudf::test::fixed_width_column_wrapper<uint8_t> v2b{255, 255};

  cudf::table_view batch1{{keys1, v1a, v1b}};
  cudf::table_view batch2{{keys2, v2a, v2b}};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(1, cudf::make_max_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(2, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(2, cudf::make_max_aggregation<cudf::groupby_aggregation>()));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);

  auto [keys, results] = streaming_agg.finalize();
  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

// A COUNT reads only the value column's null mask, so it takes the packed table whatever the
// column's type.
TEST_F(StreamingGroupbyTest, PackedCountOnStringValues)
{
  using K = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1, 2};
  cudf::test::strings_column_wrapper vals1({"a", "", "c", "d"}, {true, false, true, true});
  cudf::test::fixed_width_column_wrapper<K> keys2{3, 1};
  cudf::test::strings_column_wrapper vals2({"", "f"}, {false, true});

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE)));
  reqs.push_back(make_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE)));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);

  auto [keys, results] = streaming_agg.finalize();
  cudf::test::fixed_width_column_wrapper<K> ek{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> evalid{3, 1, 0};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> eall{3, 2, 1};
  check(keys, results, cudf::table_view{{ek}}, {evalid, eall});
}

// Keys the packed table cannot hold take the generic path and give the same results.
TEST_F(StreamingGroupbyTest, FloatKeysTakeGenericPath)
{
  using K = double;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1.5, 2.5, 1.5};
  cudf::test::fixed_width_column_wrapper<V> vals1{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<K> keys2{2.5, -0.0};
  cudf::test::fixed_width_column_wrapper<V> vals2{4, 5};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);

  auto [keys, results] = streaming_agg.finalize();
  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

// Eight bytes of keys leave no room for the validity bits of `null_policy::INCLUDE`, so such
// keys take the generic path, which handles null keys the same way.
TEST_F(StreamingGroupbyTest, WideKeysWithNullsIncludedTakeGenericPath)
{
  using K = uint64_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{{1, 2, 1}, {true, false, true}};
  cudf::test::fixed_width_column_wrapper<V> vals1{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<K> keys2{{2, 3}, {false, true}};
  cudf::test::fixed_width_column_wrapper<V> vals2{4, 5};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby streaming_agg(
    KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS, cudf::null_policy::INCLUDE);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  EXPECT_EQ(streaming_agg.distinct_keys(), 3);

  auto [keys, results] = streaming_agg.finalize();
  verify_against_groupby(
    keys, results, {batch1, batch2}, KEY_COL, reqs, cudf::null_policy::INCLUDE);
}

// Two objects with the same aggregations but keys of different types cannot be merged.
TEST_F(StreamingGroupbyTest, MergeDifferentKeyTypesThrows)
{
  using V = int32_t;

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby int_keys(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  cudf::groupby::streaming_groupby wide_keys(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  cudf::groupby::streaming_groupby string_keys(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);

  cudf::test::fixed_width_column_wrapper<int32_t> k1{1, 2};
  cudf::test::fixed_width_column_wrapper<V> v1{1, 2};
  int_keys.aggregate(cudf::table_view{{k1, v1}});
  cudf::test::fixed_width_column_wrapper<int64_t> k2{1, 2};
  cudf::test::fixed_width_column_wrapper<V> v2{1, 2};
  wide_keys.aggregate(cudf::table_view{{k2, v2}});
  cudf::test::strings_column_wrapper k3{"1", "2"};
  cudf::test::fixed_width_column_wrapper<V> v3{1, 2};
  string_keys.aggregate(cudf::table_view{{k3, v3}});

  EXPECT_THROW(int_keys.merge(wide_keys), std::invalid_argument);
  EXPECT_THROW(int_keys.merge(string_keys), std::invalid_argument);
  EXPECT_THROW(string_keys.merge(int_keys), std::invalid_argument);
}

// The same seven requests on one column through the generic path: MEAN decomposes to SUM and
// COUNT_VALID, both of which are also requested on their own, and finalize must not count the
// shared intermediates twice.
TEST_F(StreamingGroupbyTest, StringKeyManyAggregationsOnOneColumn)
{
  using V = int32_t;

  cudf::test::strings_column_wrapper keys1{"a", "b", "c"};
  // A null in the first batch, so the results are nullable from the start (they are sized and
  // typed by the first batch) and the group of only nulls comes out null.
  cudf::test::fixed_width_column_wrapper<V> vals1{{10, 0, 30}, {true, false, true}};
  cudf::test::strings_column_wrapper keys2{"a", "d", "d", "b", "c"};
  cudf::test::fixed_width_column_wrapper<V> vals2{{0, 0, 0, 5, 0},
                                                  {false, false, false, true, false}};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(1, cudf::make_max_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE)));
  reqs.push_back(make_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE)));
  reqs.push_back(make_req(1, cudf::make_mean_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(1, cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>()));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  EXPECT_EQ(streaming_agg.distinct_keys(), 4);

  auto [keys, results] = streaming_agg.finalize();
  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}
