/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/contiguous_split.hpp>
#include <cudf/utilities/traits.hpp>

#include <cudf_streaming/partition_utils.hpp>
#include <cudf_streaming/utils.hpp>

#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/memory/spill.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/utils/misc.hpp>

#include <memory>
#include <tuple>

using namespace cudf_streaming;

class NumOfPartitions : public cudf::test::BaseFixtureWithParam<std::tuple<int, int>> {};

// test different `num_partitions` and `num_rows`.
INSTANTIATE_TEST_SUITE_P(Partitions,
                         NumOfPartitions,
                         testing::Combine(testing::Range(1, 10),     // num_partitions
                                          testing::Range(1, 100, 9)  // num_rows
                                          ));

TEST_P(NumOfPartitions, partition_and_pack)
{
  int const num_partitions    = std::get<0>(GetParam());
  int const num_rows          = std::get<1>(GetParam());
  std::int64_t const seed     = 42;
  cudf::hash_id const hash_fn = cudf::hash_id::HASH_MURMUR3;
  auto stream                 = cudf::get_default_stream();
  auto br                     = rapidsmpf::BufferResource::create(mr());

  cudf::table expect = random_table_with_index(seed, static_cast<std::size_t>(num_rows), 0, 10);

  auto chunks = partition_and_pack(expect, {1}, num_partitions, hash_fn, seed, stream, br.get());

  // Convert to a vector
  std::vector<rapidsmpf::PackedData> chunks_vector;
  for (auto& [_, chunk] : chunks) {
    chunks_vector.push_back(std::move(chunk));
  }
  EXPECT_EQ(chunks_vector.size(), num_partitions);

  auto result = unpack_and_concat(std::move(chunks_vector), stream, br.get());

  // Compare the input table with the result. We ignore the row order by
  // sorting by their index (first column).
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sort_table(expect), sort_table(result));
}

TEST_P(NumOfPartitions, round_trip_with_reservations)
{
  int const num_partitions    = std::get<0>(GetParam());
  int const num_rows          = std::get<1>(GetParam());
  std::int64_t const seed     = 42;
  cudf::hash_id const hash_fn = cudf::hash_id::HASH_MURMUR3;
  auto stream                 = cudf::get_default_stream();
  auto br                     = rapidsmpf::BufferResource::create(mr());

  cudf::table expect = random_table_with_index(seed, static_cast<std::size_t>(num_rows), 0, 10);

  // Each reservation is sized by the matching cost function and must come back empty,
  // having been consumed as the allocations landed.
  auto pack_res = br->reserve_or_fail(partition_and_pack_cost(expect, stream, br->device_mr()),
                                      rapidsmpf::MemoryType::DEVICE);
  auto packed   = partition_and_pack(expect, {1}, num_partitions, hash_fn, seed, stream, pack_res);
  EXPECT_EQ(packed.size(), num_partitions);
  EXPECT_EQ(pack_res.size(), 0);

  std::vector<cudf::size_type> splits;
  for (int i = 1; i < num_partitions; ++i) {
    splits.emplace_back(i * num_rows / num_partitions);
  }
  auto split_res = br->reserve_or_fail(split_and_pack_cost(expect, stream, br->device_mr()),
                                       rapidsmpf::MemoryType::DEVICE);
  auto split     = split_and_pack(expect, splits, stream, split_res);
  EXPECT_EQ(split.size(), num_partitions);
  EXPECT_EQ(split_res.size(), 0);

  for (auto* chunks : {&packed, &split}) {
    auto chunks_vector = rapidsmpf::to_vector(std::move(*chunks));
    auto concat_res =
      br->reserve_or_fail(unpack_and_concat_cost(chunks_vector), rapidsmpf::MemoryType::DEVICE);
    auto result = unpack_and_concat(std::move(chunks_vector), stream, concat_res);
    EXPECT_EQ(concat_res.size(), 0);
    // `to_vector()` does not preserve partition order, so compare sorted by the index.
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sort_table(expect), sort_table(result));
  }
}

class PartitionReservation : public cudf::test::BaseFixture {};

TEST_F(PartitionReservation, rejects_invalid_reservations)
{
  auto stream     = cudf::get_default_stream();
  auto br         = rapidsmpf::BufferResource::create(mr());
  auto table      = random_table_with_index(42, 100, 0, 10);
  auto const cost = partition_and_pack_cost(table, stream, br->device_mr());
  ASSERT_GT(cost, 1);

  auto const pack = [&](rapidsmpf::MemoryReservation& res) {
    return partition_and_pack(table, {1}, 4, cudf::hash_id::HASH_MURMUR3, 42, stream, res);
  };

  auto host = br->reserve_or_fail(cost, rapidsmpf::MemoryType::HOST);
  EXPECT_THROW(std::ignore = pack(host), std::invalid_argument);

  auto too_small = br->reserve_or_fail(cost - 1, rapidsmpf::MemoryType::DEVICE);
  EXPECT_THROW(std::ignore = pack(too_small), rapidsmpf::reservation_error);
  // The reorder and the pack are split off separately, so the total is checked before
  // either. Nothing was taken, leaving the caller free to reserve more and retry.
  EXPECT_EQ(too_small.size(), cost - 1);
}

TEST_F(PartitionReservation, empty_table)
{
  auto stream = cudf::get_default_stream();
  auto br     = rapidsmpf::BufferResource::create(mr());
  auto table  = random_table_with_index(42, 0, 0, 10);

  // An empty table skips the reorder, so the cost is the packed size alone.
  auto reservation = br->reserve_or_fail(partition_and_pack_cost(table, stream, br->device_mr()),
                                         rapidsmpf::MemoryType::DEVICE);
  auto packed =
    partition_and_pack(table, {1}, 4, cudf::hash_id::HASH_MURMUR3, 42, stream, reservation);
  EXPECT_EQ(packed.size(), 4);
  EXPECT_EQ(reservation.size(), 0);
}

TEST_F(PartitionReservation, cost_rejects_unusable_partitions)
{
  auto stream = cudf::get_default_stream();
  auto br     = rapidsmpf::BufferResource::create(mr());
  auto table  = random_table_with_index(42, 100, 0, 10);
  auto chunks = rapidsmpf::to_vector(
    partition_and_pack(table, {1}, 4, cudf::hash_id::HASH_MURMUR3, 42, stream, br.get()));

  std::vector<rapidsmpf::PackedData const*> null_partition{nullptr};
  EXPECT_THROW(std::ignore = unpack_and_concat_cost(null_partition), std::invalid_argument);

  // Moving a partition away leaves the source with null members.
  auto const moved_away = std::move(chunks.at(0));
  std::vector<rapidsmpf::PackedData const*> consumed{&chunks.at(0)};
  EXPECT_THROW(std::ignore = unpack_and_concat_cost(consumed), std::invalid_argument);
}

TEST_F(PartitionReservation, unspill_reservation)
{
  auto stream = cudf::get_default_stream();
  auto br     = rapidsmpf::BufferResource::create(mr());
  auto table  = random_table_with_index(42, 100, 0, 10);

  auto chunks = rapidsmpf::to_vector(
    partition_and_pack(table, {1}, 4, cudf::hash_id::HASH_MURMUR3, 42, stream, br.get()));
  auto const device_cost = unpack_and_concat_cost(chunks);

  // Spill the partitions so the unspill actually allocates. The unspill is the one part
  // of `unpack_and_concat()` whose reservation is consumed by the buffer resource rather
  // than merely accounted for, so its share must be exact.
  chunks          = rapidsmpf::spill_partitions(std::move(chunks), br.get());
  auto const cost = unpack_and_concat_cost(chunks);
  ASSERT_EQ(cost, 2 * device_cost);  // Every partition now needs unspilling.
  ASSERT_GT(cost, 1);

  // The unspill and the concatenation are split off separately, so the total is checked
  // before either. Without that the first split would have consumed the reservation.
  auto too_small = br->reserve_or_fail(cost - 1, rapidsmpf::MemoryType::DEVICE);
  EXPECT_THROW(std::ignore = unpack_and_concat(std::move(chunks), stream, too_small),
               rapidsmpf::reservation_error);
  EXPECT_EQ(too_small.size(), cost - 1);

  // Nothing was consumed, so reserving more and retrying works.
  auto reservation = br->reserve_or_fail(cost, rapidsmpf::MemoryType::DEVICE);
  auto result      = unpack_and_concat(std::move(chunks), stream, reservation);
  EXPECT_EQ(reservation.size(), 0);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sort_table(table), sort_table(result));
}

TEST_P(NumOfPartitions, split_and_pack)
{
  int const num_partitions = std::get<0>(GetParam());
  int const num_rows       = std::get<1>(GetParam());
  std::int64_t const seed  = 42;
  auto stream              = cudf::get_default_stream();
  auto br = rapidsmpf::BufferResource::create(cudf::get_current_device_resource_ref());

  cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);

  std::vector<cudf::size_type> splits;
  for (int i = 1; i < num_partitions; ++i) {
    splits.emplace_back(i * num_rows / num_partitions);
  }

  auto chunks = split_and_pack(expect, splits, stream, br.get());

  // Convert to a vector (restoring the original order).
  std::vector<rapidsmpf::PackedData> chunks_vector;
  for (int i = 0; i < num_partitions; ++i) {
    chunks_vector.emplace_back(std::move(chunks.at(i)));
  }
  EXPECT_EQ(chunks_vector.size(), num_partitions);

  auto result = unpack_and_concat(std::move(chunks_vector), stream, br.get());

  // Compare the input table with the result.
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expect, *result);
}
