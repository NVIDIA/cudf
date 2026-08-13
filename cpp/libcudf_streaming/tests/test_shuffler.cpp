/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "environment.hpp"
#include "utils.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf_streaming/partition_utils.hpp>

#include <gtest/gtest.h>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/memory/spill.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/utils/misc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <thread>

extern Environment* GlobalEnvironment;
using namespace cudf_streaming;

using MemoryLimitsMap = std::unordered_map<rapidsmpf::MemoryType, std::int64_t>;

// Help function to get the `memory_limits` argument for a `BufferResource`
// that prioritizes the specified memory type.
MemoryLimitsMap get_memory_limits_map(rapidsmpf::MemoryType priorities)
{
  using namespace rapidsmpf;

  // We set all memory types to be unlimited.
  MemoryLimitsMap ret = {{MemoryType::DEVICE, std::numeric_limits<std::int64_t>::max()},
                         {MemoryType::HOST, std::numeric_limits<std::int64_t>::max()}};

  // And then set device memory to zero if it isn't prioritized.
  if (priorities != MemoryType::DEVICE) { ret.at(MemoryType::DEVICE) = 0; }
  // Note, we never set host memory to zero because it is used to allocate
  // stuff like metadata and control messages.
  return ret;
}

void test_shuffler(std::shared_ptr<rapidsmpf::Communicator> const& comm,
                   rapidsmpf::shuffler::Shuffler& shuffler,
                   rapidsmpf::shuffler::PartID total_num_partitions,
                   std::size_t total_num_rows,
                   std::int64_t seed,
                   cudf::hash_id hash_fn,
                   rmm::cuda_stream_view stream,
                   rapidsmpf::BufferResource* br)
{
  // To expose unexpected deadlocks, we use a 30s timeout. In a normal run, the
  // shuffle shouldn't get near 30s.
  std::chrono::seconds const wait_timeout(30);

  // Every rank creates the full input table and all the expected partitions (also
  // partitions this rank might not get after the shuffle).
  cudf::table full_input_table = random_table_with_index(seed, total_num_rows, 0, 10);
  auto [expect_partitions, owner] =
    cudf_streaming::partition_and_split(full_input_table,
                                        {1},
                                        static_cast<std::int32_t>(total_num_partitions),
                                        hash_fn,
                                        seed,
                                        stream,
                                        br,
                                        rapidsmpf::AllowOverbooking::YES);

  cudf::size_type row_offset = 0;
  cudf::size_type partiton_size =
    full_input_table.num_rows() / static_cast<cudf::size_type>(total_num_partitions);
  for (rapidsmpf::shuffler::PartID i = 0; i < total_num_partitions; ++i) {
    // To simulate that `full_input_table` is distributed between multiple ranks,
    // we divided them into `total_num_partitions` number of partitions and pick
    // the partitions this rank should use as input. We pick using round robin but
    // any distribution would work (as long as no rows are picked by multiple
    // ranks).
    // TODO: we should test different distributions of the input partitions.
    if (rapidsmpf::shuffler::Shuffler::round_robin(comm, i, total_num_partitions) == comm->rank()) {
      cudf::size_type row_end = row_offset + partiton_size;
      if (i == total_num_partitions - 1) {
        // Include the reminder of rows in the very last partition.
        row_end = full_input_table.num_rows();
      }
      // Select the partition from the full input table.
      auto slice = cudf::slice(full_input_table, {row_offset, row_end}).at(0);
      // Hash the `slice` into chunks and pack (serialize) them.
      auto packed_chunks =
        cudf_streaming::partition_and_pack(slice,
                                           {1},
                                           static_cast<std::int32_t>(total_num_partitions),
                                           hash_fn,
                                           seed,
                                           stream,
                                           br,
                                           rapidsmpf::AllowOverbooking::YES);
      // Add the chunks to the shuffle
      shuffler.insert(std::move(packed_chunks));
    }
    row_offset += partiton_size;
  }
  // Tell the shuffler that we have no more input partitions.
  shuffler.insert_finished();

  EXPECT_NO_THROW(shuffler.wait(wait_timeout));
  for (auto finished_partition : shuffler.local_partitions()) {
    auto packed_chunks = shuffler.extract(finished_partition);
    auto result        = cudf_streaming::unpack_and_concat(
      rapidsmpf::unspill_partitions(std::move(packed_chunks), br, rapidsmpf::AllowOverbooking::YES),
      stream,
      br,
      rapidsmpf::AllowOverbooking::YES);

    // We should only receive the partitions assigned to this rank.
    EXPECT_EQ(shuffler.partition_owner(comm, finished_partition, total_num_partitions),
              comm->rank());

    // Check the result while ignoring the row order.
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sort_table(result),
                                       sort_table(expect_partitions[finished_partition]));
  }
}

class MemoryLimits_NumPartition
  : public cudf::test::BaseFixtureWithParam<
      std::tuple<MemoryLimitsMap, rapidsmpf::shuffler::PartID, std::size_t>> {
 public:
  void SetUp() override
  {
    stream               = cudf::get_default_stream();
    memory_limits        = std::get<0>(GetParam());
    total_num_partitions = std::get<1>(GetParam());
    total_num_rows       = std::get<2>(GetParam());
    br = rapidsmpf::BufferResource::create(mr(), rapidsmpf::PinnedMemoryDisabled, memory_limits);

    shuffler = std::make_unique<rapidsmpf::shuffler::Shuffler>(GlobalEnvironment->comm_,
                                                               0,  // op_id
                                                               total_num_partitions,
                                                               br.get());
  }

  void TearDown() override { shuffler.reset(); }

 protected:
  MemoryLimitsMap memory_limits;
  rapidsmpf::shuffler::PartID total_num_partitions;
  std::size_t total_num_rows;
  std::int64_t seed     = 42;
  cudf::hash_id hash_fn = cudf::hash_id::HASH_MURMUR3;
  rmm::cuda_stream_view stream;
  std::shared_ptr<rapidsmpf::BufferResource> br;
  std::unique_ptr<rapidsmpf::shuffler::Shuffler> shuffler;
};

// test different `memory_available` and `total_num_partitions`.
INSTANTIATE_TEST_SUITE_P(
  Shuffler,
  MemoryLimits_NumPartition,
  testing::Combine(testing::ValuesIn({get_memory_limits_map(rapidsmpf::MemoryType::HOST),
                                      get_memory_limits_map(rapidsmpf::MemoryType::DEVICE)}),
                   testing::Values(1, 2, 5, 10),        // total_num_partitions
                   testing::Values(1, 9, 100, 100'000)  // total_num_rows
                   ),
  [](testing::TestParamInfo<MemoryLimits_NumPartition::ParamType> const& info) {
    return std::to_string(info.index) + "__nparts_" + std::to_string(std::get<1>(info.param)) +
           "__nrows_" + std::to_string(std::get<2>(info.param));
  });

TEST_P(MemoryLimits_NumPartition, round_trip)
{
  EXPECT_NO_FATAL_FAILURE(test_shuffler(GlobalEnvironment->comm_,
                                        *shuffler,
                                        total_num_partitions,
                                        total_num_rows,
                                        seed,
                                        hash_fn,
                                        stream,
                                        br.get()));
}

// Test that the same communicator can be used concurrently by multiple shufflers in
// separate threads
class ConcurrentShuffleTest : public cudf::test::BaseFixtureWithParam<std::tuple<int, int>> {
 public:
  void SetUp() override
  {
    num_shufflers        = std::get<0>(GetParam());
    total_num_partitions = static_cast<rapidsmpf::shuffler::PartID>(std::get<1>(GetParam()));

    // these resources will be used by multiple threads to instantiate shufflers
    br     = rapidsmpf::BufferResource::create(mr());
    stream = cudf::get_default_stream();
  }

  void TearDown() override {}

  int num_shufflers;
  rapidsmpf::shuffler::PartID total_num_partitions;

  rmm::cuda_stream_view stream;
  std::shared_ptr<rapidsmpf::BufferResource> br;
};

TEST_P(ConcurrentShuffleTest, round_trip)
{
  std::vector<std::future<void>> futures;
  futures.reserve(static_cast<std::size_t>(num_shufflers));

  for (int t_id = 0; t_id < num_shufflers; t_id++) {
    futures.push_back(std::async(std::launch::async, [this, t_id] {
      rapidsmpf::shuffler::Shuffler shuffler(GlobalEnvironment->comm_,
                                             t_id,  // op_id, use t_id as a proxy
                                             total_num_partitions,
                                             br.get());
      EXPECT_NO_FATAL_FAILURE(test_shuffler(GlobalEnvironment->comm_,
                                            shuffler,
                                            total_num_partitions,
                                            100'000,  // total_num_rows
                                            t_id,     // seed
                                            cudf::hash_id::HASH_MURMUR3,
                                            stream,
                                            br.get()));
    }));
  }

  for (auto& f : futures) {
    ASSERT_NO_THROW(f.get());
  }
}

// test different `num_shufflers` and `total_num_partitions`.
INSTANTIATE_TEST_SUITE_P(ConcurrentShuffle,
                         ConcurrentShuffleTest,
                         testing::Combine(testing::ValuesIn({1, 2, 4}),    // num_shufflers
                                          testing::ValuesIn({1, 10, 100})  // total_num_partitions
                                          ),
                         [](testing::TestParamInfo<ConcurrentShuffleTest::ParamType> const& info) {
                           return "num_shufflers_" + std::to_string(std::get<0>(info.param)) +
                                  "__total_num_partitions_" +
                                  std::to_string(std::get<1>(info.param));
                         });

TEST(Shuffler, SpillOnInsertAndExtraction)
{
  rapidsmpf::shuffler::PartID const total_num_partitions = 2;
  std::int64_t const seed                                = 42;
  cudf::hash_id const hash_fn                            = cudf::hash_id::HASH_MURMUR3;
  auto stream                                            = cudf::get_default_stream();

  // Control spilling by adjusting the DEVICE memory limit at runtime.
  // `memory_available(DEVICE)` is computed as `limit - current_allocated()`, so a
  // sufficiently large positive limit reliably keeps available memory > 0 (no spill),
  // while a sufficiently large negative limit reliably keeps available memory < 0
  // (force spill), regardless of how many bytes are currently allocated.
  constexpr std::int64_t k_no_spill_limit    = (1LL << 40);
  constexpr std::int64_t k_force_spill_limit = -(1LL << 40);
  // `BufferResource` wraps the supplied resource in its own tracking adaptor,
  // exposed via `device_mr_adaptor()`, so the test can observe per-rank
  // allocation counts via `get_main_record().num_current_allocs()`.
  auto br        = rapidsmpf::BufferResource::create(cudf::get_current_device_resource_ref(),
                                              rapidsmpf::PinnedMemoryDisabled,
                                                     {{rapidsmpf::MemoryType::DEVICE, k_no_spill_limit}},
                                              std::nullopt  // disable periodic spill check
  );
  auto const& mr = br->device_mr_adaptor();

  // Create a communicator of size 1, such that each shuffler will run locally.
  auto comm = GlobalEnvironment->split_comm();
  EXPECT_EQ(comm->nranks(), 1);

  // Create a shuffler and input chunks.
  rapidsmpf::shuffler::Shuffler shuffler(comm,
                                         0,  // op_id
                                         total_num_partitions,
                                         br.get());
  cudf::table input_table = random_table_with_index(seed, 1000, 0, 10);
  auto input_chunks =
    cudf_streaming::partition_and_pack(input_table,
                                       {1},
                                       total_num_partitions,
                                       hash_fn,
                                       seed,
                                       stream,
                                       br.get(),
                                       rapidsmpf::AllowOverbooking::YES);  // with overbooking

  // Insert spills does nothing when device memory is available, we start
  // with 2 device allocations.
  EXPECT_EQ(mr.get_main_record().num_current_allocs(), 2);
  shuffler.insert(std::move(input_chunks));
  // And we end with two 2 device allocations.
  EXPECT_EQ(mr.get_main_record().num_current_allocs(), 2);

  // Let's force spilling.
  br->set_memory_limit(rapidsmpf::MemoryType::DEVICE, k_force_spill_limit);

  {
    // Now extract triggers spilling of the partition not being extracted.
    std::vector<rapidsmpf::PackedData> output_chunks = rapidsmpf::unspill_partitions(
      shuffler.extract(0), br.get(), rapidsmpf::AllowOverbooking::YES);
    EXPECT_EQ(mr.get_main_record().num_current_allocs(), 1);

    // And insert also triggers spilling. We end up with zero device allocations.
    std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunk;
    chunk.emplace(0, std::move(output_chunks.at(0)));
    shuffler.insert(std::move(chunk));
    EXPECT_EQ(mr.get_main_record().num_current_allocs(), 0);
  }

  // Extract and unspill both partitions.
  std::vector<rapidsmpf::PackedData> out0 =
    rapidsmpf::unspill_partitions(shuffler.extract(0), br.get(), rapidsmpf::AllowOverbooking::YES);
  EXPECT_EQ(mr.get_main_record().num_current_allocs(), 1);
  std::vector<rapidsmpf::PackedData> out1 =
    rapidsmpf::unspill_partitions(shuffler.extract(1), br.get(), rapidsmpf::AllowOverbooking::YES);
  EXPECT_EQ(mr.get_main_record().num_current_allocs(), 2);

  // Disable spilling and insert the first partition.
  br->set_memory_limit(rapidsmpf::MemoryType::DEVICE, k_no_spill_limit);
  {
    std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunk;
    chunk.emplace(0, std::move(out0.at(0)));
    shuffler.insert(std::move(chunk));
  }
  EXPECT_EQ(mr.get_main_record().num_current_allocs(), 2);

  // Enable spilling and insert the second partition, which should trigger spilling
  // of both the first partition already in the shuffler and the second partition
  // that are being inserted.
  br->set_memory_limit(rapidsmpf::MemoryType::DEVICE, k_force_spill_limit);
  {
    std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunk;
    chunk.emplace(1, std::move(out1.at(0)));
    shuffler.insert(std::move(chunk));
  }
  EXPECT_EQ(mr.get_main_record().num_current_allocs(), 0);
  shuffler.insert_finished();
}

// check cudf pack conditions for empty table
TEST(EmptyPartitions, cudf_pack)
{
  auto stream     = cudf::get_default_stream();
  cudf::table tbl = random_table_with_index(0, 0, 0, 0);
  EXPECT_EQ(0, tbl.num_rows());

  // following conditions should be met for an empty cudf table
  auto packed = cudf::pack(tbl, stream);
  EXPECT_TRUE(packed.metadata);
  EXPECT_TRUE(packed.gpu_data);
  EXPECT_EQ(0, packed.gpu_data->size());
}

// Test that multiple threads can call wait() concurrently.
TEST(Shuffler, concurrent_wait)
{
  auto const& comm = GlobalEnvironment->comm_;
  auto stream      = cudf::get_default_stream();
  auto br          = rapidsmpf::BufferResource::create(cudf::get_current_device_resource_ref());

  // Use more partitions than ranks so each rank owns multiple partitions, ensuring
  // multiple threads call wait() concurrently on the same shuffler.
  auto const total_num_partitions =
    rapidsmpf::safe_cast<rapidsmpf::shuffler::PartID>(comm->nranks()) * 8;
  constexpr std::size_t total_num_rows = 1000;
  constexpr cudf::hash_id hash_fn      = cudf::hash_id::HASH_MURMUR3;
  constexpr std::int64_t seed          = 42;
  constexpr auto wait_timeout          = std::chrono::seconds{30};

  rapidsmpf::shuffler::Shuffler shuffler(comm, 0, total_num_partitions, br.get());

  cudf::table full_input = random_table_with_index(seed, total_num_rows, 0, 10);
  auto [expected, owner] =
    cudf_streaming::partition_and_split(full_input,
                                        {1},
                                        static_cast<std::int32_t>(total_num_partitions),
                                        hash_fn,
                                        seed,
                                        stream,
                                        br.get(),
                                        rapidsmpf::AllowOverbooking::YES);

  {
    std::vector<std::future<void>> insert_futures;
    cudf::size_type row_offset = 0;
    cudf::size_type part_size =
      full_input.num_rows() / static_cast<cudf::size_type>(total_num_partitions);
    for (rapidsmpf::shuffler::PartID i = 0; i < total_num_partitions; ++i) {
      if (rapidsmpf::shuffler::Shuffler::round_robin(comm, i, total_num_partitions) ==
          comm->rank()) {
        cudf::size_type row_end = row_offset + part_size;
        if (i == total_num_partitions - 1) { row_end = full_input.num_rows(); }
        auto slice = cudf::slice(full_input, {row_offset, row_end}).at(0);
        insert_futures.push_back(std::async(std::launch::async, [&, slice] {
          shuffler.insert(
            cudf_streaming::partition_and_pack(slice,
                                               {1},
                                               static_cast<std::int32_t>(total_num_partitions),
                                               hash_fn,
                                               seed,
                                               br->stream_pool()->get_stream(),
                                               br.get(),
                                               rapidsmpf::AllowOverbooking::YES));
        }));
      }
      row_offset += part_size;
    }
    std::ranges::for_each(insert_futures, [](auto& f) { f.get(); });
    shuffler.insert_finished();
  }

  auto local_pids = shuffler.local_partitions();
  std::vector<std::future<void>> futures;
  for (auto pid : local_pids) {
    futures.push_back(std::async(std::launch::async, [&, pid] {
      EXPECT_NO_THROW(shuffler.wait(wait_timeout));
      auto chunks = shuffler.extract(pid);
      auto result = cudf_streaming::unpack_and_concat(
        rapidsmpf::unspill_partitions(
          std::move(chunks), br.get(), rapidsmpf::AllowOverbooking::YES),
        stream,
        br.get(),
        rapidsmpf::AllowOverbooking::YES);
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sort_table(result), sort_table(expected[pid]));
    }));
  }
  std::ranges::for_each(futures, [](auto& f) { f.get(); });
}

// Test that reusing an OpID after a completed shuffle doesn't cause cross-matching of
// messages between the old and new shuffle.
//
// On rank 0 we inject a stream-ordered delay into device allocations so that received
// chunks stay "not ready" in the event loop. With small messages, other ranks can finish
// and move on to the next shuffle. Its messages will then be matched on rank 0 by the
// blocked previous shuffle, unless recv gating correctly prevents cross-talk.
TEST(Shuffler, opid_reuse)
{
  auto const& comm = GlobalEnvironment->comm_;
  if (comm->nranks() == 1) { GTEST_SKIP() << "OpID reuse test requires multiple ranks"; }

  auto stream = cudf::get_default_stream();
  auto const total_num_partitions =
    rapidsmpf::safe_cast<rapidsmpf::shuffler::PartID>(comm->nranks());
  constexpr std::size_t total_num_rows = 1000;
  constexpr cudf::hash_id hash_fn      = cudf::hash_id::HASH_MURMUR3;
  constexpr rapidsmpf::OpID op_id      = 0;
  constexpr auto wait_timeout          = std::chrono::seconds{30};

  rmm::mr::cuda_memory_resource mr;
  auto br = rapidsmpf::BufferResource::create(mr);

  // On rank 0, wrap the device MR with a delayed version for the shuffler.
  std::unique_ptr<DelayedMemoryResource> delayed_mr;
  std::shared_ptr<rapidsmpf::BufferResource> delayed_br;
  rapidsmpf::BufferResource* shuffler_br = br.get();
  if (comm->rank() == 0) {
    delayed_mr  = std::make_unique<DelayedMemoryResource>(mr, std::chrono::milliseconds(500));
    delayed_br  = rapidsmpf::BufferResource::create(*delayed_mr);
    shuffler_br = delayed_br.get();
  }

  auto insert_data = [&](rapidsmpf::shuffler::Shuffler& shuffler, std::int64_t seed) {
    cudf::table full_input     = random_table_with_index(seed, total_num_rows, 0, 10);
    cudf::size_type row_offset = 0;
    cudf::size_type part_size =
      full_input.num_rows() / static_cast<cudf::size_type>(total_num_partitions);
    for (rapidsmpf::shuffler::PartID i = 0; i < total_num_partitions; ++i) {
      if (rapidsmpf::shuffler::Shuffler::round_robin(comm, i, total_num_partitions) ==
          comm->rank()) {
        cudf::size_type row_end = row_offset + part_size;
        if (i == total_num_partitions - 1) { row_end = full_input.num_rows(); }
        auto slice = cudf::slice(full_input, {row_offset, row_end}).at(0);
        auto packed =
          cudf_streaming::partition_and_pack(slice,
                                             {1},
                                             static_cast<std::int32_t>(total_num_partitions),
                                             hash_fn,
                                             seed,
                                             stream,
                                             br.get(),
                                             rapidsmpf::AllowOverbooking::YES);
        shuffler.insert(std::move(packed));
      }
      row_offset += part_size;
    }
  };

  auto validate_results = [&](rapidsmpf::shuffler::Shuffler& shuffler, std::int64_t seed) {
    cudf::table full_input = random_table_with_index(seed, total_num_rows, 0, 10);
    auto [expected, owner] =
      cudf_streaming::partition_and_split(full_input,
                                          {1},
                                          static_cast<std::int32_t>(total_num_partitions),
                                          hash_fn,
                                          seed,
                                          stream,
                                          br.get(),
                                          rapidsmpf::AllowOverbooking::YES);
    for (auto pid : shuffler.local_partitions()) {
      auto chunks = shuffler.extract(pid);
      auto result = cudf_streaming::unpack_and_concat(
        rapidsmpf::unspill_partitions(
          std::move(chunks), br.get(), rapidsmpf::AllowOverbooking::YES),
        stream,
        br.get(),
        rapidsmpf::AllowOverbooking::YES);
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sort_table(result), sort_table(expected[pid]));
    }
  };

  rapidsmpf::shuffler::Shuffler shuffle1(comm, op_id, total_num_partitions, shuffler_br);
  insert_data(shuffle1, 42);
  shuffle1.insert_finished();
  EXPECT_NO_THROW(shuffle1.wait(wait_timeout));

  rapidsmpf::shuffler::Shuffler shuffle2(comm, op_id, total_num_partitions, shuffler_br);
  insert_data(shuffle2, 123);
  shuffle2.insert_finished();
  EXPECT_NO_THROW(shuffle2.wait(wait_timeout));

  validate_results(shuffle1, 42);
  validate_results(shuffle2, 123);
}

// Same as opid_reuse but with total_num_partitions=1, so only rank 0 owns a partition.
// All other ranks have empty local_partitions and empty recv loops. This exercises the
// edge case where non-partition-owning ranks must still correctly handle op_id reuse.
TEST(Shuffler, opid_reuse_with_empty_partitions)
{
  auto const& comm = GlobalEnvironment->comm_;
  if (comm->nranks() == 1) { GTEST_SKIP() << "OpID reuse test requires multiple ranks"; }

  auto stream                                                = cudf::get_default_stream();
  constexpr rapidsmpf::shuffler::PartID total_num_partitions = 1;
  constexpr std::size_t total_num_rows                       = 1000;
  constexpr cudf::hash_id hash_fn                            = cudf::hash_id::HASH_MURMUR3;
  constexpr rapidsmpf::OpID op_id                            = 0;
  constexpr auto wait_timeout                                = std::chrono::seconds{30};

  rmm::mr::cuda_memory_resource mr;
  auto br = rapidsmpf::BufferResource::create(mr);

  // On rank 0, wrap the device MR with a delayed version for the shuffler.
  std::unique_ptr<DelayedMemoryResource> delayed_mr;
  std::shared_ptr<rapidsmpf::BufferResource> delayed_br;
  rapidsmpf::BufferResource* shuffler_br = br.get();
  if (comm->rank() == 0) {
    delayed_mr  = std::make_unique<DelayedMemoryResource>(mr, std::chrono::milliseconds(500));
    delayed_br  = rapidsmpf::BufferResource::create(*delayed_mr);
    shuffler_br = delayed_br.get();
  }

  auto insert_data = [&](rapidsmpf::shuffler::Shuffler& shuffler, std::int64_t seed) {
    cudf::table full_input = random_table_with_index(seed, total_num_rows, 0, 10);
    // With total_num_partitions=1, only rank 0 owns the single partition.
    if (rapidsmpf::shuffler::Shuffler::round_robin(comm, 0, total_num_partitions) == comm->rank()) {
      auto packed =
        cudf_streaming::partition_and_pack(full_input,
                                           {1},
                                           static_cast<std::int32_t>(total_num_partitions),
                                           hash_fn,
                                           seed,
                                           stream,
                                           br.get(),
                                           rapidsmpf::AllowOverbooking::YES);
      shuffler.insert(std::move(packed));
    }
  };

  auto validate_results = [&](rapidsmpf::shuffler::Shuffler& shuffler, std::int64_t seed) {
    cudf::table full_input = random_table_with_index(seed, total_num_rows, 0, 10);
    auto [expected, owner] =
      cudf_streaming::partition_and_split(full_input,
                                          {1},
                                          static_cast<std::int32_t>(total_num_partitions),
                                          hash_fn,
                                          seed,
                                          stream,
                                          br.get(),
                                          rapidsmpf::AllowOverbooking::YES);
    for (auto pid : shuffler.local_partitions()) {
      auto chunks = shuffler.extract(pid);
      auto result = cudf_streaming::unpack_and_concat(
        rapidsmpf::unspill_partitions(
          std::move(chunks), br.get(), rapidsmpf::AllowOverbooking::YES),
        stream,
        br.get(),
        rapidsmpf::AllowOverbooking::YES);
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sort_table(result), sort_table(expected[pid]));
    }
  };

  rapidsmpf::shuffler::Shuffler shuffle1(comm, op_id, total_num_partitions, shuffler_br);
  insert_data(shuffle1, 42);
  shuffle1.insert_finished();
  EXPECT_NO_THROW(shuffle1.wait(wait_timeout));

  rapidsmpf::shuffler::Shuffler shuffle2(comm, op_id, total_num_partitions, shuffler_br);
  insert_data(shuffle2, 123);
  shuffle2.insert_finished();
  EXPECT_NO_THROW(shuffle2.wait(wait_timeout));

  validate_results(shuffle1, 42);
  validate_results(shuffle2, 123);
}
