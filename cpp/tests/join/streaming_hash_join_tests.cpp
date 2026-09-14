/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Single batch behavior is covered by parametrizing the shared join tests: the
// `algorithm::STREAMING_HASH` arm in tests/join/join_tests.cpp runs streaming_hash_join through
// the same cases as the other join implementations, so result correctness, null handling,
// dictionary and nested keys, and empty inputs are not duplicated here.
//
// This file covers what a single call through that harness cannot reach: inserting more than one
// batch, concurrent inserts from multiple threads, batch ID capacity limits, schema validation
// across partitions, and constructor argument validation.

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/streaming_hash_join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda/stream>

#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

namespace {

using cudf::size_type;
template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

using join_match = std::tuple<size_type, size_type, size_type>;

std::vector<join_match> to_sorted_host_matches(cudf::device_span<size_type const> left_indices,
                                               cudf::device_span<size_type const> batch_indices,
                                               cudf::device_span<size_type const> row_indices,
                                               cuda::stream_ref stream)
{
  auto const h_left  = cudf::detail::make_host_vector(left_indices, stream);
  auto const h_batch = cudf::detail::make_host_vector(batch_indices, stream);
  auto const h_row   = cudf::detail::make_host_vector(row_indices, stream);

  std::vector<join_match> matches;
  matches.reserve(h_left.size());
  for (std::size_t i = 0; i < h_left.size(); ++i) {
    matches.emplace_back(h_left[i], h_batch[i], h_row[i]);
  }
  std::sort(matches.begin(), matches.end());
  return matches;
}

}  // namespace

struct StreamingHashJoinTest : public cudf::test::BaseFixture {};

TEST_F(StreamingHashJoinTest, MultiplePartitionsReturnBatchLocalIndices)
{
  auto const stream = cudf::test::get_default_stream();
  column_wrapper<int32_t> right0{1, 2, 5};
  column_wrapper<int32_t> right1{2, 3, 5};
  column_wrapper<int32_t> right2{2, 5, 9};
  cudf::table_view right0_view{{right0}};
  cudf::table_view right1_view{{right1}};
  cudf::table_view right2_view{{right2}};
  column_wrapper<int32_t> left{2, 5, 3, 7};
  cudf::table_view left_view{{left}};

  std::vector<size_type> const keys{0};
  cudf::streaming_hash_join joiner{
    right0_view,
    keys,
    right0_view.num_rows() + right1_view.num_rows() + right2_view.num_rows(),
    // A non-power-of-two maximum requires two batch-ID bits.
    /*max_num_batches=*/3,
    cudf::nullable_join::NO,
    cudf::null_equality::EQUAL};
  joiner.insert(right0_view, stream);
  joiner.insert(right1_view, stream);
  joiner.insert(right2_view, stream);

  auto [left_indices, right]         = joiner.inner_join(left_view, {}, stream);
  auto& [batch_indices, row_indices] = right;
  auto const actual = to_sorted_host_matches(*left_indices, *batch_indices, *row_indices, stream);
  std::vector<join_match> const expected{
    {0, 0, 1}, {0, 1, 0}, {0, 2, 0}, {1, 0, 2}, {1, 1, 2}, {1, 2, 1}, {2, 1, 1}};
  EXPECT_EQ(actual, expected);
}

TEST_F(StreamingHashJoinTest, ConcurrentInsert)
{
  auto const stream               = cudf::test::get_default_stream();
  constexpr size_type num_batches = 16;
  std::vector<int32_t> values(num_batches);
  std::iota(values.begin(), values.end(), 0);
  column_wrapper<int32_t> right(values.begin(), values.end());
  cudf::table_view const right_view{{right}};
  std::vector<size_type> slice_indices;
  slice_indices.reserve(2 * num_batches);
  for (size_type i = 0; i < num_batches; ++i) {
    slice_indices.push_back(i);
    slice_indices.push_back(i + 1);
  }
  auto const right_partitions = cudf::slice(right_view, slice_indices);
  column_wrapper<int32_t> left(values.begin(), values.end());

  std::vector<std::unique_ptr<rmm::cuda_stream>> streams;
  streams.reserve(num_batches);
  for (size_type i = 0; i < num_batches; ++i) {
    streams.push_back(std::make_unique<rmm::cuda_stream>());
  }

  std::vector<size_type> const keys{0};
  // Construct on a stream of its own so the inserts below run on different streams than the
  // hash table was built on, then synchronize it as the `insert()` docs require.
  rmm::cuda_stream const build_stream;
  cudf::streaming_hash_join joiner{right_view,
                                   keys,
                                   /*total_right_rows=*/num_batches,
                                   /*max_num_batches=*/num_batches,
                                   cudf::nullable_join::NO,
                                   cudf::null_equality::EQUAL,
                                   /*load_factor=*/0.5,
                                   build_stream.view()};
  build_stream.synchronize();

  auto const device = rmm::get_current_cuda_device();
  std::vector<std::thread> threads;
  std::vector<std::exception_ptr> errors(num_batches);
  // `ready` lets the main thread wait until every worker is spawned and spinning, and `start`
  // then releases them together, so the inserts actually overlap instead of trickling in as
  // each thread is created.
  std::atomic<size_type> ready{0};
  std::atomic<bool> start{false};
  threads.reserve(num_batches);
  for (size_type i = 0; i < num_batches; ++i) {
    threads.emplace_back([&, i] {
      rmm::cuda_set_device_raii const device_guard{device};
      ready.fetch_add(1, std::memory_order_relaxed);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      try {
        joiner.insert(right_partitions[i], streams[i]->view());
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
  for (auto const& insert_stream : streams) {
    insert_stream->synchronize();
  }

  auto [left_indices, right_indices] = joiner.inner_join(cudf::table_view{{left}}, {}, stream);
  auto& [batch_indices, row_indices] = right_indices;
  auto const actual = to_sorted_host_matches(*left_indices, *batch_indices, *row_indices, stream);
  ASSERT_EQ(actual.size(), num_batches);
  std::vector<size_type> actual_left_indices;
  std::vector<size_type> actual_batch_indices;
  actual_left_indices.reserve(num_batches);
  actual_batch_indices.reserve(num_batches);
  for (auto const [left_index, batch_index, row_index] : actual) {
    actual_left_indices.push_back(left_index);
    actual_batch_indices.push_back(batch_index);
    EXPECT_EQ(row_index, 0);
  }
  std::sort(actual_left_indices.begin(), actual_left_indices.end());
  std::sort(actual_batch_indices.begin(), actual_batch_indices.end());
  std::vector<size_type> expected_indices(num_batches);
  std::iota(expected_indices.begin(), expected_indices.end(), 0);
  EXPECT_EQ(actual_left_indices, expected_indices);
  EXPECT_EQ(actual_batch_indices, expected_indices);
}

TEST_F(StreamingHashJoinTest, MaxNumBatchesCountsEmptyBatches)
{
  auto const stream = cudf::test::get_default_stream();

  column_wrapper<int32_t> schema_exemplar{};
  cudf::table_view const schema{{schema_exemplar}};
  std::vector<size_type> const keys{0};
  cudf::streaming_hash_join joiner{schema,
                                   keys,
                                   /*total_right_rows=*/1,
                                   /*max_num_batches=*/1,
                                   cudf::nullable_join::NO,
                                   cudf::null_equality::EQUAL};

  column_wrapper<int32_t> empty{};
  joiner.insert(cudf::table_view{{empty}}, stream);

  column_wrapper<int32_t> nonempty{1};
  EXPECT_THROW(joiner.insert(cudf::table_view{{nonempty}}, stream), std::invalid_argument);
}

TEST_F(StreamingHashJoinTest, InnerJoinBeforeInsertThrows)
{
  auto const stream = cudf::test::get_default_stream();

  column_wrapper<int32_t> right_schema_exemplar{};
  cudf::table_view const right_schema{{right_schema_exemplar}};
  std::vector<size_type> const right_key_indices{0};

  cudf::streaming_hash_join streaming_joiner{right_schema,
                                             right_key_indices,
                                             /*total_right_rows=*/4,
                                             /*max_num_batches=*/1,
                                             cudf::nullable_join::NO,
                                             cudf::null_equality::EQUAL};
  column_wrapper<int32_t> left_keys{1, 2, 3};
  cudf::table_view left_view{{left_keys}};
  EXPECT_THROW(static_cast<void>(streaming_joiner.inner_join(left_view, {}, stream)),
               std::logic_error);
}

TEST_F(StreamingHashJoinTest, SchemaMismatchThrows)
{
  auto const stream = cudf::test::get_default_stream();

  column_wrapper<int32_t> right_schema_exemplar{};
  cudf::table_view const right_schema{{right_schema_exemplar}};
  std::vector<size_type> const right_key_indices{0};
  cudf::streaming_hash_join streaming_joiner{right_schema,
                                             right_key_indices,
                                             /*total_right_rows=*/4,
                                             /*max_num_batches=*/2,
                                             cudf::nullable_join::NO,
                                             cudf::null_equality::EQUAL};

  column_wrapper<int64_t> wrong_type_keys{1L, 2L, 3L};
  cudf::table_view wrong_type_view{{wrong_type_keys}};
  EXPECT_THROW(streaming_joiner.insert(wrong_type_view, stream), std::invalid_argument);

  column_wrapper<int32_t> col_a{1, 2, 3};
  column_wrapper<int32_t> col_b{4, 5, 6};
  cudf::table_view wrong_count_view{{col_a, col_b}};
  EXPECT_THROW(streaming_joiner.insert(wrong_count_view, stream), std::invalid_argument);

  streaming_joiner.insert(cudf::table_view{{col_a}}, stream);
  EXPECT_THROW(streaming_joiner.insert(cudf::table_view{{col_a}}, stream), std::invalid_argument);
}

TEST_F(StreamingHashJoinTest, UsesSelectedKeysAndPreservesEmptyBatchIds)
{
  auto const stream = cudf::test::get_default_stream();
  column_wrapper<int32_t> empty_payload{};
  column_wrapper<int64_t> empty_keys{};
  cudf::table_view empty{{empty_payload, empty_keys}};
  column_wrapper<int32_t> payload{100, 200, 300};
  column_wrapper<int64_t> keys{7, 8, 7};
  cudf::table_view right{{payload, keys}};
  column_wrapper<int64_t> left_keys{7, 9};
  cudf::table_view left{{left_keys}};

  std::vector<size_type> const key_indices{1};
  cudf::streaming_hash_join joiner{right,
                                   key_indices,
                                   right.num_rows(),
                                   /*max_num_batches=*/2,
                                   cudf::nullable_join::NO,
                                   cudf::null_equality::EQUAL};
  joiner.insert(empty, stream);
  joiner.insert(right, stream);

  auto [left_indices, right_indices] = joiner.inner_join(left, /*output_size=*/2, stream);
  auto& [batch_indices, row_indices] = right_indices;
  auto const actual = to_sorted_host_matches(*left_indices, *batch_indices, *row_indices, stream);
  std::vector<join_match> const expected{{0, 1, 0}, {0, 1, 2}};
  EXPECT_EQ(actual, expected);
}

TEST_F(StreamingHashJoinTest, NullEqualityAcrossPartitions)
{
  auto const stream = cudf::test::get_default_stream();
  column_wrapper<int32_t> right0{{1, 0}, {1, 0}};
  column_wrapper<int32_t> right1{{0, 2}, {0, 1}};
  cudf::table_view right0_view{{right0}};
  cudf::table_view right1_view{{right1}};
  column_wrapper<int32_t> left_keys{{0, 1, 2}, {0, 1, 1}};
  cudf::table_view left{{left_keys}};
  std::vector<size_type> const key_indices{0};

  auto run = [&](cudf::null_equality nulls_equal) {
    cudf::streaming_hash_join joiner{right0_view,
                                     key_indices,
                                     right0_view.num_rows() + right1_view.num_rows(),
                                     /*max_num_batches=*/2,
                                     cudf::nullable_join::YES,
                                     nulls_equal};
    joiner.insert(right0_view, stream);
    joiner.insert(right1_view, stream);
    auto [left_indices, right_indices] = joiner.inner_join(left, {}, stream);
    auto& [batch_indices, row_indices] = right_indices;
    return to_sorted_host_matches(*left_indices, *batch_indices, *row_indices, stream);
  };

  std::vector<join_match> const equal_expected{{0, 0, 1}, {0, 1, 0}, {1, 0, 0}, {2, 1, 1}};
  std::vector<join_match> const unequal_expected{{1, 0, 0}, {2, 1, 1}};
  EXPECT_EQ(run(cudf::null_equality::EQUAL), equal_expected);
  EXPECT_EQ(run(cudf::null_equality::UNEQUAL), unequal_expected);
}

TEST_F(StreamingHashJoinTest, NestedKeysAcrossPartitions)
{
  auto const stream = cudf::test::get_default_stream();
  column_wrapper<int32_t> right0_child{1, 2};
  column_wrapper<int32_t> right1_child{2, 3};
  column_wrapper<int32_t> left_child{2, 3};
  auto right0_struct = cudf::test::structs_column_wrapper{{right0_child}};
  auto right1_struct = cudf::test::structs_column_wrapper{{right1_child}};
  auto left_struct   = cudf::test::structs_column_wrapper{{left_child}};
  cudf::table_view right0{{right0_struct}};
  cudf::table_view right1{{right1_struct}};
  cudf::table_view left{{left_struct}};

  std::vector<size_type> const key_indices{0};
  cudf::streaming_hash_join joiner{right0,
                                   key_indices,
                                   right0.num_rows() + right1.num_rows(),
                                   /*max_num_batches=*/2,
                                   cudf::nullable_join::NO,
                                   cudf::null_equality::EQUAL};
  joiner.insert(right0, stream);
  joiner.insert(right1, stream);

  auto [left_indices, right_indices] = joiner.inner_join(left, {}, stream);
  auto& [batch_indices, row_indices] = right_indices;
  auto const actual = to_sorted_host_matches(*left_indices, *batch_indices, *row_indices, stream);
  std::vector<join_match> const expected{{0, 0, 1}, {0, 1, 0}, {1, 1, 1}};
  EXPECT_EQ(actual, expected);
}

TEST_F(StreamingHashJoinTest, SlicedPartitionReturnsSliceLocalRows)
{
  auto const stream = cudf::test::get_default_stream();
  column_wrapper<int32_t> right0_owner{99, 4, 5, 88};
  cudf::table_view right0_owner_view{{right0_owner}};
  auto const right0 = cudf::slice(right0_owner_view, {1, 3}).front();
  column_wrapper<int32_t> right1_keys{5, 6};
  cudf::table_view right1{{right1_keys}};
  column_wrapper<int32_t> left_keys{5};
  cudf::table_view left{{left_keys}};

  std::vector<size_type> const key_indices{0};
  cudf::streaming_hash_join joiner{right0_owner_view,
                                   key_indices,
                                   right0.num_rows() + right1.num_rows(),
                                   /*max_num_batches=*/2,
                                   cudf::nullable_join::NO,
                                   cudf::null_equality::EQUAL};
  joiner.insert(right0, stream);
  joiner.insert(right1, stream);

  auto [left_indices, right_indices] = joiner.inner_join(left, {}, stream);
  auto& [batch_indices, row_indices] = right_indices;
  auto const actual = to_sorted_host_matches(*left_indices, *batch_indices, *row_indices, stream);
  std::vector<join_match> const expected{{0, 0, 1}, {0, 1, 0}};
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(left_indices->size(), 2);
}

TEST_F(StreamingHashJoinTest, GetPartitionResolvesBatchIds)
{
  auto const stream               = cudf::test::get_default_stream();
  constexpr size_type num_batches = 4;
  std::vector<int32_t> values(num_batches);
  std::iota(values.begin(), values.end(), 0);
  column_wrapper<int32_t> right(values.begin(), values.end());
  cudf::table_view const right_view{{right}};
  std::vector<size_type> slice_indices;
  slice_indices.reserve(2 * num_batches);
  for (size_type i = 0; i < num_batches; ++i) {
    slice_indices.push_back(i);
    slice_indices.push_back(i + 1);
  }
  auto const right_partitions = cudf::slice(right_view, slice_indices);

  std::vector<size_type> const keys{0};
  cudf::streaming_hash_join joiner{right_view,
                                   keys,
                                   /*total_right_rows=*/num_batches,
                                   /*max_num_batches=*/num_batches,
                                   cudf::nullable_join::NO,
                                   cudf::null_equality::EQUAL,
                                   /*load_factor=*/0.5,
                                   stream};
  for (auto const& partition : right_partitions) {
    joiner.insert(partition, stream);
  }

  // Every batch ID resolves to a partition holding the value that batch was built from, without
  // the caller tracking which `insert()` call was assigned which ID.
  for (size_type batch_id = 0; batch_id < num_batches; ++batch_id) {
    auto const resolved = joiner.get_partition(batch_id);
    ASSERT_EQ(resolved.num_columns(), 1);
    ASSERT_EQ(resolved.num_rows(), 1);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(resolved.column(0), right_partitions[batch_id].column(0));
  }

  EXPECT_THROW(static_cast<void>(joiner.get_partition(-1)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(joiner.get_partition(num_batches)), std::out_of_range);
}

TEST_F(StreamingHashJoinTest, MemoryResources)
{
  auto const stream = cudf::test::get_default_stream();

  column_wrapper<int32_t> right_keys{1, 2};
  cudf::table_view right{{right_keys}};
  column_wrapper<int32_t> left_keys{2};
  cudf::table_view left{{left_keys}};
  std::vector<size_type> const key_indices{0};

  auto persistent_mr =
    rmm::mr::statistics_resource_adaptor(cudf::get_current_device_resource_ref());
  cudf::streaming_hash_join joiner{right,
                                   key_indices,
                                   right.num_rows(),
                                   /*max_num_batches=*/1,
                                   cudf::nullable_join::NO,
                                   cudf::null_equality::EQUAL,
                                   /*load_factor=*/0.5,
                                   stream,
                                   persistent_mr};
  EXPECT_GT(persistent_mr.get_bytes_counter().peak, 0);
  joiner.insert(right, stream);

  auto output_mr  = rmm::mr::statistics_resource_adaptor(cudf::get_current_device_resource_ref());
  auto scratch_mr = rmm::mr::statistics_resource_adaptor(cudf::get_current_device_resource_ref());
  // Snapshot the persistent resource so we can tell the output did not come out of it. Probing does
  // transiently allocate a counter from it, since that comes from the container's own allocator, so
  // compare outstanding bytes rather than the peak.
  auto const persistent_bytes_before = persistent_mr.get_bytes_counter().value;
  auto [left_indices, right_indices] =
    joiner.inner_join(left, {}, stream, cudf::memory_resources{output_mr, scratch_mr});
  EXPECT_GT(output_mr.get_bytes_counter().peak, 0);
  // Probe scratch goes to the temporary resource and is released before the call returns.
  EXPECT_GT(scratch_mr.get_bytes_counter().peak, 0);
  EXPECT_EQ(scratch_mr.get_bytes_counter().value, 0);
  EXPECT_EQ(persistent_mr.get_bytes_counter().value, persistent_bytes_before);
  EXPECT_EQ(left_indices->size(), 1);
  EXPECT_EQ(right_indices.first->size(), 1);
  EXPECT_EQ(right_indices.second->size(), 1);
}

TEST_F(StreamingHashJoinTest, ConstructorValidatesArguments)
{
  column_wrapper<int32_t> single_schema_exemplar{};
  cudf::table_view const single_schema{{single_schema_exemplar}};
  cudf::table_view const empty_schema{};
  std::vector<size_type> const single_key{0};

  EXPECT_THROW(cudf::streaming_hash_join(single_schema,
                                         single_key,
                                         4,
                                         /*max_num_batches=*/0,
                                         cudf::nullable_join::NO,
                                         cudf::null_equality::EQUAL),
               std::invalid_argument);

  EXPECT_THROW(cudf::streaming_hash_join(empty_schema,
                                         single_key,
                                         4,
                                         /*max_num_batches=*/1,
                                         cudf::nullable_join::NO,
                                         cudf::null_equality::EQUAL),
               std::invalid_argument);

  EXPECT_THROW(cudf::streaming_hash_join(single_schema,
                                         std::vector<size_type>{},
                                         4,
                                         /*max_num_batches=*/1,
                                         cudf::nullable_join::NO,
                                         cudf::null_equality::EQUAL),
               std::invalid_argument);

  EXPECT_THROW(cudf::streaming_hash_join(single_schema,
                                         std::vector<size_type>{5},
                                         4,
                                         /*max_num_batches=*/1,
                                         cudf::nullable_join::NO,
                                         cudf::null_equality::EQUAL),
               std::invalid_argument);

  EXPECT_THROW(
    cudf::streaming_hash_join(
      single_schema, single_key, -1, 1, cudf::nullable_join::NO, cudf::null_equality::EQUAL),
    std::invalid_argument);

  EXPECT_THROW(cudf::streaming_hash_join(single_schema,
                                         single_key,
                                         4,
                                         /*max_num_batches=*/1,
                                         cudf::nullable_join::NO,
                                         cudf::null_equality::EQUAL,
                                         /*load_factor=*/0.0),
               std::invalid_argument);
  EXPECT_THROW(cudf::streaming_hash_join(single_schema,
                                         single_key,
                                         4,
                                         /*max_num_batches=*/1,
                                         cudf::nullable_join::NO,
                                         cudf::null_equality::EQUAL,
                                         /*load_factor=*/1.5),
               std::invalid_argument);
}
