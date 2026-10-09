/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../utils.hpp"
#include "base_streaming_fixture.hpp"

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf_streaming/table_chunk.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cuda/stream>

#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/owning_wrapper.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/utils/string.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace cudf_streaming;

using StreamingTableChunkParam = std::tuple<rapidsmpf::MemoryType, int, int>;

class StreamingTableChunk : public BaseStreamingFixture,
                            public ::testing::WithParamInterface<StreamingTableChunkParam> {
 protected:
  void SetUp() override
  {
    rapidsmpf::config::Options options(rapidsmpf::config::get_environment_variables());

    std::unordered_map<rapidsmpf::MemoryType, std::int64_t> memory_limits{};
    auto stream_pool = std::make_shared<rapidsmpf::StreamPool>(16);
    stream           = cudf::get_default_stream();
    // Enable pinned host memory only when supported; otherwise the non-pinned
    // params still run and the PINNED_HOST cases skip in the test bodies.
    auto pinned_pool_properties = rapidsmpf::is_pinned_memory_resources_supported()
                                    ? rapidsmpf::PinnedPoolProperties{}
                                    : rapidsmpf::PinnedMemoryDisabled;
    br                          = rapidsmpf::BufferResource::create(
      mr_cuda,                            // device_mr
      std::move(pinned_pool_properties),  // pinned_pool_properties
      memory_limits,                      // memory_limits
      std::chrono::milliseconds{1},       // periodic_spill_check
      stream_pool,                        // stream_pool
      rapidsmpf::Statistics::disabled(),  // statistics
      spill_dir.path());
    ctx = std::make_shared<rapidsmpf::streaming::Context>(
      options, GlobalEnvironment->comm_->logger(), br);
  }

  /// @brief A resource recording into @p stats, since the fixture's has statistics disabled.
  std::shared_ptr<rapidsmpf::BufferResource> make_tracked_br(
    std::shared_ptr<rapidsmpf::Statistics> stats)
  {
    return rapidsmpf::BufferResource::create(
      mr_cuda,
      rapidsmpf::is_pinned_memory_resources_supported()
        ? std::optional<rapidsmpf::PinnedPoolProperties>{rapidsmpf::PinnedPoolProperties{}}
        : rapidsmpf::PinnedMemoryDisabled,
      std::unordered_map<rapidsmpf::MemoryType, std::int64_t>{},
      std::nullopt,
      std::make_shared<rapidsmpf::StreamPool>(16),
      std::move(stats),
      spill_dir.path());
  }

  /// @brief The number of spills recorded in @p stats.
  static std::size_t spill_samples(rapidsmpf::Statistics const& stats)
  {
    return stats.has_stat("buffer-spilled-time") ? stats.get_stat("buffer-spilled-time").count()
                                                 : 0UL;
  }

  TempDir spill_dir;
  cuda::stream_ref stream{cudaStream_t{cudaStreamDefault}};
  rmm::mr::cuda_memory_resource mr_cuda;
  std::shared_ptr<rapidsmpf::BufferResource> br;
  std::shared_ptr<rapidsmpf::streaming::Context> ctx;
};

TEST_F(StreamingTableChunk, FromTable)
{
  constexpr unsigned int num_rows = 100;
  constexpr std::int64_t seed     = 1337;

  cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);

  table_chunk chunk{std::make_unique<cudf::table>(expect), stream};
  EXPECT_EQ(chunk.stream().get(), stream.get());
  EXPECT_TRUE(chunk.is_available());
  EXPECT_TRUE(chunk.is_spillable());
  EXPECT_EQ(chunk.make_available_cost(), 0);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(chunk.table_view(), expect);

  auto chunk2 = chunk.make_available(
    br->reserve_or_fail(chunk.make_available_cost(), rapidsmpf::MemoryType::DEVICE));
  EXPECT_FALSE(chunk.is_available());
  EXPECT_TRUE(chunk2.is_available());
  EXPECT_TRUE(chunk2.is_spillable());
  EXPECT_EQ(chunk2.make_available_cost(), 0);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(chunk2.table_view(), expect);
}

TEST_F(StreamingTableChunk, TableChunkOwner)
{
  constexpr unsigned int num_rows = 100;
  constexpr std::int64_t seed     = 1337;
  constexpr std::uint64_t seq     = 42;

  cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);
  // Static because the deleter function is a void(*)(void*) which precludes the use of
  // a lambda with captures.
  static std::size_t num_deletions{0};
  auto deleter = [](void* p) {
    num_deletions++;
    delete static_cast<int*>(p);
  };
  auto make_chunk = [&](table_chunk::exclusive_view exclusive_view) {
    return table_chunk{expect, stream, rapidsmpf::OwningWrapper(new int, deleter), exclusive_view};
  };
  auto check_chunk = [&](table_chunk const& chunk, bool is_spillable) {
    EXPECT_EQ(chunk.stream().get(), stream.get());
    EXPECT_TRUE(chunk.is_available());
    EXPECT_EQ(chunk.is_spillable(), is_spillable);
    EXPECT_EQ(chunk.make_available_cost(), 0);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(chunk.table_view(), expect);
  };
  {
    auto chunk = make_chunk(table_chunk::exclusive_view::NO);
    check_chunk(chunk, false);
    EXPECT_EQ(num_deletions, 0);
  }
  EXPECT_EQ(num_deletions, 1);
  {
    auto msg =
      to_message(seq, std::make_unique<table_chunk>(make_chunk(table_chunk::exclusive_view::NO)));
    EXPECT_EQ(num_deletions, 1);
  }
  EXPECT_EQ(num_deletions, 2);
  {
    auto msg =
      to_message(seq, std::make_unique<table_chunk>(make_chunk(table_chunk::exclusive_view::YES)));
    auto chunk = msg.release<table_chunk>();
    check_chunk(chunk, true);
    EXPECT_EQ(num_deletions, 2);
  }
  EXPECT_EQ(num_deletions, 3);
  {
    auto chunk = make_chunk(table_chunk::exclusive_view::YES);
    check_chunk(chunk, true);
    auto res = br->reserve_or_fail(chunk.data_alloc_size(rapidsmpf::MemoryType::DEVICE),
                                   rapidsmpf::MemoryType::DEVICE);
    // This is like spilling since the original `chunk` is ExclusiveView::YES and
    // overwritten.
    chunk = chunk.copy(res);
    EXPECT_EQ(num_deletions, 4);
  }
}

TEST_F(StreamingTableChunk, FromPackedDataOnDevice)
{
  constexpr unsigned int num_rows = 100;
  constexpr std::int64_t seed     = 1337;

  cudf::table expect  = random_table_with_index(seed, num_rows, 0, 10);
  auto packed_columns = cudf::pack(expect, stream);

  auto packed_data = std::make_unique<rapidsmpf::PackedData>(
    std::move(packed_columns.metadata), br->move(std::move(packed_columns.gpu_data), stream));
  table_chunk chunk{std::move(packed_data)};

  EXPECT_EQ(chunk.stream().get(), stream.get());
  // chunk was created from packed data on device, so it is available and make available
  // cost is 0.
  EXPECT_TRUE(chunk.is_available());
  EXPECT_TRUE(chunk.is_spillable());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expect, chunk.table_view());
  EXPECT_EQ(chunk.make_available_cost(), 0);

  auto chunk2 = chunk.make_available(
    br->reserve_or_fail(chunk.make_available_cost(), rapidsmpf::MemoryType::DEVICE));
  EXPECT_FALSE(chunk.is_available());
  EXPECT_TRUE(chunk2.is_available());
  EXPECT_TRUE(chunk2.is_spillable());
  EXPECT_EQ(chunk2.make_available_cost(), 0);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(chunk2.table_view(), expect);
}

INSTANTIATE_TEST_SUITE_P(StreamingTableChunkWithSpillTargets,
                         StreamingTableChunk,
                         ::testing::Combine(::testing::Values(rapidsmpf::MemoryType::PINNED_HOST,
                                                              rapidsmpf::MemoryType::HOST,
                                                              rapidsmpf::MemoryType::DISK),
                                            ::testing::Values(2, 0),
                                            ::testing::Values(100, 0)),
                         ([](testing::TestParamInfo<StreamingTableChunkParam> const& info) {
                           auto const [memory_type, ncols, nrows] = info.param;
                           std::stringstream ss;
                           ss << memory_type << "_" << ncols << "cols_" << nrows << "rows";
                           return ss.str();
                         }));

TEST_P(StreamingTableChunk, FromPackedDataOn)
{
  auto const [spill_mem_type, ncols, nrows] = GetParam();
  if (spill_mem_type == rapidsmpf::MemoryType::PINNED_HOST &&
      !rapidsmpf::is_pinned_memory_resources_supported()) {
    GTEST_SKIP() << "MemoryType::PINNED_HOST isn't supported on the system.";
  }

  constexpr std::int64_t seed = 1337;

  cudf::table expect     = random_table(seed, nrows, ncols, 0, 10);
  auto packed_columns    = cudf::pack(expect, stream);
  std::size_t const size = packed_columns.gpu_data->size();

  // Move the gpu_data to a Buffer (still device memory).
  auto gpu_data_on_device = br->move(std::move(packed_columns.gpu_data), stream);

  // Copy the GPU data to the current spill target memory type.
  auto [res, _] = br->reserve(spill_mem_type, size, rapidsmpf::AllowOverbooking::YES);
  auto gpu_data_in_spill_memory = br->move(std::move(gpu_data_on_device), res);

  auto packed_data = std::make_unique<rapidsmpf::PackedData>(std::move(packed_columns.metadata),
                                                             std::move(gpu_data_in_spill_memory));
  table_chunk chunk{std::move(packed_data)};
  auto const expected_shape = std::pair<cudf::size_type, cudf::size_type>{nrows, ncols};
  EXPECT_EQ(chunk.shape(), expected_shape);

  EXPECT_EQ(chunk.stream().get(), stream.get());
  EXPECT_FALSE(chunk.is_available());
  EXPECT_TRUE(chunk.is_spillable());
  EXPECT_THROW(std::ignore = chunk.table_view(), std::invalid_argument);
  EXPECT_EQ(chunk.make_available_cost(), size);

  auto chunk2 = chunk.make_available(
    br->reserve_or_fail(chunk.make_available_cost(), rapidsmpf::MemoryType::DEVICE));
  EXPECT_EQ(chunk2.shape(), expected_shape);
  EXPECT_FALSE(chunk.is_available());
  EXPECT_TRUE(chunk2.is_available());
  EXPECT_TRUE(chunk2.is_spillable());
  EXPECT_EQ(chunk2.make_available_cost(), 0);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(chunk2.table_view(), expect);
}

TEST_F(StreamingTableChunk, DeviceToDeviceCopy)
{
  constexpr unsigned int num_rows = 100;
  constexpr std::int64_t seed     = 1337;

  auto expect = random_table_with_index(seed, num_rows, 0, 10);

  cudf_streaming::table_chunk chunk{std::make_unique<cudf::table>(expect), stream};
  EXPECT_TRUE(chunk.is_available());

  auto res    = br->reserve_or_fail(chunk.data_alloc_size(rapidsmpf::MemoryType::DEVICE),
                                 rapidsmpf::MemoryType::DEVICE);
  auto chunk2 = chunk.copy(res);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(chunk2.table_view(), expect);
}

TEST_F(StreamingTableChunk, ShapeOnAvailableAndSpilledChunk)
{
  constexpr unsigned int num_rows = 64;
  constexpr std::int64_t seed     = 2025;

  cudf::table expect = random_table_with_index(seed, num_rows, 0, 5);
  auto const expected_shape =
    std::pair<cudf::size_type, cudf::size_type>{expect.num_rows(), expect.num_columns()};

  table_chunk device_chunk{std::make_unique<cudf::table>(expect), stream};
  EXPECT_TRUE(device_chunk.is_available());
  EXPECT_EQ(device_chunk.shape(), expected_shape);

  auto [res, _]   = br->reserve(rapidsmpf::MemoryType::HOST,
                              device_chunk.data_alloc_size(rapidsmpf::MemoryType::DEVICE),
                              rapidsmpf::AllowOverbooking::YES);
  auto host_chunk = device_chunk.copy(res);

  EXPECT_FALSE(host_chunk.is_available());
  EXPECT_EQ(host_chunk.shape(), expected_shape);

  device_chunk = host_chunk.make_available(
    br->reserve_or_fail(host_chunk.make_available_cost(), rapidsmpf::MemoryType::DEVICE));
  EXPECT_TRUE(device_chunk.is_available());
  EXPECT_EQ(device_chunk.shape(), expected_shape);
}

TEST_P(StreamingTableChunk, RoundTripCopy)
{
  auto const [spill_mem_type, ncols, nrows] = GetParam();
  if (spill_mem_type == rapidsmpf::MemoryType::PINNED_HOST &&
      !rapidsmpf::is_pinned_memory_resources_supported()) {
    GTEST_SKIP() << "MemoryType::PINNED_HOST isn't supported on the system.";
  }

  constexpr std::int64_t seed = 2025;

  auto expect               = random_table(seed, nrows, ncols, 0, 5);
  auto const expected_shape = std::pair<cudf::size_type, cudf::size_type>{nrows, ncols};

  table_chunk dev_chunk{std::make_unique<cudf::table>(expect), stream};
  EXPECT_TRUE(dev_chunk.is_available());
  EXPECT_EQ(dev_chunk.shape(), expected_shape);
  EXPECT_TRUE(dev_chunk.is_spillable());
  EXPECT_EQ(dev_chunk.stream().get(), stream.get());
  EXPECT_EQ(dev_chunk.make_available_cost(), 0);
  {
    auto cd = get_content_description(dev_chunk);
    EXPECT_EQ(cd.spillable(), dev_chunk.is_spillable());
    for (auto mem_type : rapidsmpf::MEMORY_TYPES) {
      EXPECT_EQ(cd.content_size(mem_type), dev_chunk.data_alloc_size(mem_type));
    }
  }

  // Copy to host memory -> new chunk should be unavailable.
  auto host_res =
    br->reserve_or_fail(dev_chunk.data_alloc_size(rapidsmpf::MemoryType::DEVICE), spill_mem_type);
  auto host_copy = dev_chunk.copy(host_res);
  EXPECT_FALSE(host_copy.is_available());
  EXPECT_EQ(host_copy.shape(), expected_shape);
  EXPECT_TRUE(host_copy.is_spillable());
  EXPECT_EQ(host_copy.stream().get(), stream.get());
  EXPECT_EQ(host_copy.make_available_cost(),
            dev_chunk.data_alloc_size(rapidsmpf::MemoryType::DEVICE));
  {
    auto cd = get_content_description(host_copy);
    EXPECT_EQ(cd.spillable(), host_copy.is_spillable());
    for (auto mem_type : rapidsmpf::MEMORY_TYPES) {
      EXPECT_EQ(cd.content_size(mem_type), host_copy.data_alloc_size(mem_type));
    }
  }

  // Disk-to-disk copies are unsupported (see DiskToDiskCopyUnsupported); keep the disk
  // chunk for the round trip.
  auto const host_cost = host_copy.make_available_cost();
  auto host_copy2      = [&] {
    if (spill_mem_type == rapidsmpf::MemoryType::DISK) { return std::move(host_copy); }
    auto host_res2 = br->reserve_or_fail(host_copy.data_alloc_size(spill_mem_type), spill_mem_type);
    return host_copy.copy(host_res2);
  }();
  EXPECT_FALSE(host_copy2.is_available());
  EXPECT_EQ(host_copy2.shape(), expected_shape);
  EXPECT_TRUE(host_copy2.is_spillable());
  EXPECT_EQ(host_copy2.stream().get(), stream.get());
  EXPECT_EQ(host_copy2.make_available_cost(), host_cost);
  {
    auto cd = get_content_description(host_copy2);
    EXPECT_EQ(cd.spillable(), host_copy2.is_spillable());
    for (auto mem_type : rapidsmpf::MEMORY_TYPES) {
      EXPECT_EQ(cd.content_size(mem_type), host_copy2.data_alloc_size(mem_type));
    }
  }

  // Bring the new host copy back to device and verify equality.
  auto dev_res =
    br->reserve_or_fail(host_copy2.data_alloc_size(spill_mem_type), rapidsmpf::MemoryType::DEVICE);
  auto dev_back = host_copy2.make_available(dev_res);
  EXPECT_TRUE(dev_back.is_available());
  EXPECT_EQ(dev_back.shape(), expected_shape);
  EXPECT_TRUE(dev_back.is_spillable());
  EXPECT_EQ(dev_back.stream().get(), stream.get());
  EXPECT_EQ(dev_back.make_available_cost(), 0);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(dev_back.table_view(), expect);
  {
    auto cd = get_content_description(dev_back);
    EXPECT_EQ(cd.spillable(), dev_back.is_spillable());
    for (auto mem_type : rapidsmpf::MEMORY_TYPES) {
      EXPECT_EQ(cd.content_size(mem_type), dev_back.data_alloc_size(mem_type));
    }
  }

  // Sanity check: a second device copy should also remain equivalent.
  auto dev_res2  = br->reserve_or_fail(dev_back.data_alloc_size(rapidsmpf::MemoryType::DEVICE),
                                      rapidsmpf::MemoryType::DEVICE);
  auto dev_copy2 = dev_back.copy(dev_res2);
  EXPECT_TRUE(dev_copy2.is_available());
  EXPECT_EQ(dev_copy2.shape(), expected_shape);
  EXPECT_EQ(dev_copy2.make_available_cost(), 0);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(dev_copy2.table_view(), expect);
  {
    auto cd = get_content_description(dev_copy2);
    EXPECT_EQ(cd.spillable(), dev_copy2.is_spillable());
    for (auto mem_type : rapidsmpf::MEMORY_TYPES) {
      EXPECT_EQ(cd.content_size(mem_type), dev_copy2.data_alloc_size(mem_type));
    }
  }
}

TEST_F(StreamingTableChunk, DiskToDiskCopyUnsupported)
{
  auto expect = random_table(2025, 100, 2, 0, 5);
  table_chunk dev_chunk{std::make_unique<cudf::table>(expect), stream};

  auto disk_res   = br->reserve_or_fail(dev_chunk.data_alloc_size(rapidsmpf::MemoryType::DEVICE),
                                      rapidsmpf::MemoryType::DISK);
  auto disk_chunk = dev_chunk.copy(disk_res);
  ASSERT_GT(disk_chunk.data_alloc_size(rapidsmpf::MemoryType::DISK), 0);

  // A disk chunk can't be copied into another disk reservation, a documented restriction.
  auto disk_res2 = br->reserve_or_fail(disk_chunk.data_alloc_size(rapidsmpf::MemoryType::DISK),
                                       rapidsmpf::MemoryType::DISK);
  EXPECT_THROW(std::ignore = disk_chunk.copy(disk_res2), std::invalid_argument);
}

TEST_F(StreamingTableChunk, DiskMoveToHost)
{
  auto expect = random_table(2025, 100, 2, 0, 5);
  table_chunk dev_chunk{std::make_unique<cudf::table>(expect), stream};
  auto const size = dev_chunk.data_alloc_size(rapidsmpf::MemoryType::DEVICE);

  auto disk_res   = br->reserve_or_fail(size, rapidsmpf::MemoryType::DISK);
  auto disk_chunk = dev_chunk.copy(disk_res);
  ASSERT_EQ(disk_chunk.data_alloc_size(rapidsmpf::MemoryType::DISK), size);

  // DISK -> HOST
  auto host_res   = br->reserve_or_fail(size, rapidsmpf::MemoryType::HOST);
  auto host_chunk = disk_chunk.move(host_res);
  EXPECT_FALSE(host_chunk.is_available());
  EXPECT_EQ(host_chunk.data_alloc_size(rapidsmpf::MemoryType::HOST), size);
  EXPECT_EQ(host_chunk.data_alloc_size(rapidsmpf::MemoryType::DISK), 0);
  EXPECT_EQ(host_chunk.data_alloc_size(rapidsmpf::MemoryType::DEVICE), 0);

  auto dev_back = host_chunk.make_available(
    br->reserve_or_fail(host_chunk.make_available_cost(), rapidsmpf::MemoryType::DEVICE));
  ASSERT_TRUE(dev_back.is_available());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(dev_back.table_view(), expect);
}

TEST_F(StreamingTableChunk, DiskRoundTripStrings)
{
  // Variable-width strings of odd lengths with nulls, so the packed buffer has offsets,
  // chars, a validity mask and non-trivial alignment padding.
  constexpr int num_rows = 1001;
  std::vector<std::string> strs;
  std::vector<bool> valid;
  std::vector<std::int32_t> ints;
  for (int i = 0; i < num_rows; ++i) {
    strs.emplace_back(static_cast<std::size_t>(i % 13), static_cast<char>('a' + i % 26));
    valid.push_back(i % 7 != 0);
    ints.push_back(i);
  }
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(
    cudf::test::strings_column_wrapper(strs.begin(), strs.end(), valid.begin()).release());
  cols.push_back(cudf::test::fixed_width_column_wrapper<std::int32_t>(ints.begin(), ints.end())
                   .release());
  cudf::table expect{std::move(cols)};

  table_chunk dev_chunk{std::make_unique<cudf::table>(expect), stream};
  auto const size = dev_chunk.data_alloc_size(rapidsmpf::MemoryType::DEVICE);

  auto disk_res   = br->reserve_or_fail(size, rapidsmpf::MemoryType::DISK);
  auto disk_chunk = dev_chunk.copy(disk_res);
  EXPECT_FALSE(disk_chunk.is_available());
  EXPECT_EQ(disk_chunk.data_alloc_size(rapidsmpf::MemoryType::DISK), size);

  // DISK -> HOST -> DEVICE.
  auto host_res   = br->reserve_or_fail(size, rapidsmpf::MemoryType::HOST);
  auto host_chunk = disk_chunk.move(host_res);
  EXPECT_EQ(host_chunk.data_alloc_size(rapidsmpf::MemoryType::HOST), size);
  auto from_host = host_chunk.make_available(
    br->reserve_or_fail(host_chunk.make_available_cost(), rapidsmpf::MemoryType::DEVICE));
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(from_host.table_view(), expect);

  // DISK -> DEVICE directly (the move above consumed the first disk chunk).
  auto disk_res2   = br->reserve_or_fail(size, rapidsmpf::MemoryType::DISK);
  auto disk_chunk2 = dev_chunk.copy(disk_res2);
  auto from_disk   = disk_chunk2.make_available(
    br->reserve_or_fail(disk_chunk2.make_available_cost(), rapidsmpf::MemoryType::DEVICE));
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(from_disk.table_view(), expect);
}

TEST_P(StreamingTableChunk, MoveThroughMessage)
{
  auto const [spill_mem_type, ncols, nrows] = GetParam();
  if (spill_mem_type == rapidsmpf::MemoryType::PINNED_HOST &&
      !rapidsmpf::is_pinned_memory_resources_supported()) {
    GTEST_SKIP() << "MemoryType::PINNED_HOST isn't supported on the system.";
  }
  // A move and a copy differ only in the statistic, so record it to tell them apart.
  auto stats                = rapidsmpf::Statistics::create();
  auto tracked_br           = make_tracked_br(stats);
  auto expect               = random_table(2025, nrows, ncols, 0, 5);
  auto const expected_shape = std::pair<cudf::size_type, cudf::size_type>{nrows, ncols};
  auto msg =
    to_message(0, std::make_unique<table_chunk>(std::make_unique<cudf::table>(expect), stream));

  auto const packed_size = msg.copy_cost();
  auto host_res          = tracked_br->reserve_or_fail(packed_size, spill_mem_type);
  auto spilled           = msg.move(host_res);
  EXPECT_TRUE(msg.empty());
  EXPECT_EQ(spilled.get<table_chunk>().shape(), expected_shape);
  {
    auto const& cd = spilled.content_description();
    EXPECT_TRUE(cd.spillable());
    EXPECT_EQ(cd.content_size(rapidsmpf::MemoryType::DEVICE), 0);
    EXPECT_EQ(cd.content_size(spill_mem_type), packed_size);
  }

  // Moving back closes the spill. One sample means both hops ran the move callback, so it
  // was passed on to the spilled message rather than dropped.
  auto dev_res = tracked_br->reserve_or_fail(spilled.copy_cost(), rapidsmpf::MemoryType::DEVICE);
  auto back    = spilled.move(dev_res);
  {
    auto const& cd = back.content_description();
    EXPECT_TRUE(cd.spillable());
    EXPECT_EQ(cd.content_size(rapidsmpf::MemoryType::DEVICE), packed_size);
    EXPECT_EQ(cd.content_size(spill_mem_type), 0);
  }
  EXPECT_EQ(spill_samples(*stats), (nrows > 0 && ncols > 0) ? 1UL : 0UL);

  // Packed data on device unpacks straight away, so the round trip can be compared.
  auto const& chunk = back.get<table_chunk>();
  ASSERT_TRUE(chunk.is_available());
  EXPECT_EQ(chunk.shape(), expected_shape);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(chunk.table_view(), expect);
}

TEST_F(StreamingTableChunk, MoveRequiresOwnership)
{
  cudf::table table = random_table_with_index(2025, 64, 0, 5);
  table_chunk chunk{table,
                    stream,
                    rapidsmpf::OwningWrapper(new int, [](void* p) { delete static_cast<int*>(p); }),
                    table_chunk::exclusive_view::NO};
  auto res = br->reserve_or_fail(chunk.data_alloc_size(rapidsmpf::MemoryType::DEVICE),
                                 rapidsmpf::MemoryType::HOST);
  EXPECT_THROW(std::ignore = chunk.move(res), std::invalid_argument);
  EXPECT_TRUE(chunk.is_available());  // Left untouched.
}

TEST_P(StreamingTableChunk, SpillTrackingOnHostMove)
{
  auto const [spill_mem_type, ncols, nrows] = GetParam();
  if (spill_mem_type == rapidsmpf::MemoryType::PINNED_HOST &&
      !rapidsmpf::is_pinned_memory_resources_supported()) {
    GTEST_SKIP() << "MemoryType::PINNED_HOST isn't supported on the system.";
  }

  // `buffer-spilled-time` is the only way to observe that a spill token was handed to the
  // host buffer.
  auto stats      = rapidsmpf::Statistics::create();
  auto tracked_br = make_tracked_br(stats);

  auto round_trip = [&](cudf::table table, bool move) {
    table_chunk dev{std::make_unique<cudf::table>(std::move(table)), stream};
    auto host_res = tracked_br->reserve_or_fail(dev.data_alloc_size(rapidsmpf::MemoryType::DEVICE),
                                                spill_mem_type);
    auto host     = move ? dev.move(host_res) : dev.copy(host_res);
    auto dev_res  = tracked_br->reserve_or_fail(host.data_alloc_size(spill_mem_type),
                                               rapidsmpf::MemoryType::DEVICE);
    return host.make_available(dev_res);
  };

  auto const host_name = rapidsmpf::to_lower(rapidsmpf::to_string(spill_mem_type));
  // A copy leaves the table on device, so it is not a spill.
  std::ignore = round_trip(random_table(2025, nrows, ncols, 0, 5), /* move = */ false);
  EXPECT_EQ(spill_samples(*stats), 0UL);

  // A move releases the table, so the round trip is recorded once.
  std::ignore = round_trip(random_table(2025, nrows, ncols, 0, 5), /* move = */ true);
  if (nrows > 0 && ncols > 0) {
    EXPECT_EQ(spill_samples(*stats), 1UL);
    EXPECT_GT(stats->get_stat("copy-device-to-" + host_name + "-bytes").value(), 0);
  } else {
    // An empty table still needs packing metadata, but transfers no data.
    EXPECT_EQ(spill_samples(*stats), 0UL);
    EXPECT_FALSE(stats->has_stat("copy-device-to-" + host_name + "-bytes"));
    EXPECT_FALSE(stats->has_stat("copy-" + host_name + "-to-device-bytes"));
  }
}

TEST_F(StreamingTableChunk, ToMessageRoundTrip)
{
  constexpr unsigned int num_rows = 64;
  constexpr std::int64_t seed     = 2025;
  constexpr std::uint64_t seq     = 7;

  auto expect = random_table_with_index(seed, num_rows, 0, 5);
  auto chunk  = std::make_unique<table_chunk>(std::make_unique<cudf::table>(expect), stream);

  rapidsmpf::streaming::Message m = to_message(seq, std::move(chunk));
  EXPECT_FALSE(m.empty());
  EXPECT_TRUE(m.holds<table_chunk>());
  EXPECT_TRUE(m.content_description().spillable());
  EXPECT_EQ(m.content_description().content_size(rapidsmpf::MemoryType::HOST), 0);
  EXPECT_EQ(m.content_description().content_size(rapidsmpf::MemoryType::DEVICE), 1024);
  EXPECT_EQ(m.sequence_number(), seq);

  // Deep-copy: device to host.
  auto reservation = br->reserve_or_fail(m.copy_cost(), rapidsmpf::MemoryType::HOST);
  rapidsmpf::streaming::Message m2 = m.copy(reservation);
  EXPECT_EQ(reservation.size(), 0);
  EXPECT_FALSE(m2.empty());
  EXPECT_TRUE(m2.holds<table_chunk>());
  EXPECT_TRUE(m2.content_description().spillable());
  EXPECT_EQ(m2.content_description().content_size(rapidsmpf::MemoryType::HOST), 1024);
  EXPECT_EQ(m2.content_description().content_size(rapidsmpf::MemoryType::DEVICE), 0);
  EXPECT_EQ(m2.sequence_number(), seq);

  // Deep-copy: host to host.
  reservation = br->reserve_or_fail(m2.copy_cost(), rapidsmpf::MemoryType::HOST);
  rapidsmpf::streaming::Message m3 = m.copy(reservation);
  EXPECT_EQ(reservation.size(), 0);
  EXPECT_FALSE(m3.empty());
  EXPECT_TRUE(m3.holds<table_chunk>());
  EXPECT_TRUE(m3.content_description().spillable());
  EXPECT_EQ(m3.content_description().content_size(rapidsmpf::MemoryType::HOST), 1024);
  EXPECT_EQ(m3.content_description().content_size(rapidsmpf::MemoryType::DEVICE), 0);
  EXPECT_EQ(m3.sequence_number(), seq);

  // Copy the chunk back to device and verify.
  {
    auto chunk = m3.release<table_chunk>();
    auto res   = br->reserve_or_fail(chunk.make_available_cost(), rapidsmpf::MemoryType::DEVICE);
    chunk      = chunk.make_available(res);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(chunk.table_view(), expect);
  }

  // Deep-copy: host to device.
  reservation = br->reserve_or_fail(m2.copy_cost(), rapidsmpf::MemoryType::DEVICE);
  rapidsmpf::streaming::Message m4 = m.copy(reservation);
  EXPECT_EQ(reservation.size(), 0);
  EXPECT_FALSE(m4.empty());
  EXPECT_TRUE(m4.holds<table_chunk>());
  EXPECT_TRUE(m4.content_description().spillable());
  EXPECT_EQ(m4.content_description().content_size(rapidsmpf::MemoryType::HOST), 0);
  EXPECT_EQ(m4.content_description().content_size(rapidsmpf::MemoryType::DEVICE), 1024);
  EXPECT_EQ(m4.sequence_number(), seq);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(m4.get<table_chunk>().table_view(), expect);

  // Deep-copy: device to device.
  reservation = br->reserve_or_fail(m4.copy_cost(), rapidsmpf::MemoryType::DEVICE);
  rapidsmpf::streaming::Message m5 = m.copy(reservation);
  EXPECT_EQ(reservation.size(), 0);
  EXPECT_FALSE(m5.empty());
  EXPECT_TRUE(m5.holds<table_chunk>());
  EXPECT_TRUE(m5.content_description().spillable());
  EXPECT_EQ(m5.content_description().content_size(rapidsmpf::MemoryType::HOST), 0);
  EXPECT_EQ(m5.content_description().content_size(rapidsmpf::MemoryType::DEVICE), 1024);
  EXPECT_EQ(m5.sequence_number(), seq);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(m5.get<table_chunk>().table_view(), expect);
}

TEST_F(StreamingTableChunk, ToMessageNotSpillable)
{
  constexpr unsigned int num_rows = 100;
  constexpr std::int64_t seed     = 1337;
  constexpr std::uint64_t seq     = 42;

  cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);

  auto deleter = [](void* p) { delete static_cast<int*>(p); };
  auto chunk   = std::make_unique<table_chunk>(
    expect, stream, rapidsmpf::OwningWrapper(new int, deleter), table_chunk::exclusive_view::NO);

  rapidsmpf::streaming::Message m = to_message(seq, std::move(chunk));
  EXPECT_FALSE(m.empty());
  EXPECT_TRUE(m.holds<table_chunk>());
  EXPECT_FALSE(m.content_description().spillable());
  EXPECT_EQ(m.content_description().content_size(rapidsmpf::MemoryType::HOST), 0);
  EXPECT_EQ(m.content_description().content_size(rapidsmpf::MemoryType::DEVICE),
            cudf::packed_size(expect.view(), stream, rmm::mr::get_current_device_resource_ref()));
  // packed size is greater than or equal to the alloc size due to buffer alignments.
  EXPECT_GE(m.content_description().content_size(rapidsmpf::MemoryType::DEVICE),
            expect.alloc_size());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(m.get<table_chunk>().table_view(), expect);
}

TEST_F(StreamingTableChunk, ToPackedDataFromPackedChunk)
{
  constexpr unsigned int num_rows = 100;
  constexpr std::int64_t seed     = 1337;

  cudf::table expect  = random_table_with_index(seed, num_rows, 0, 10);
  auto packed_columns = cudf::pack(expect, stream);
  table_chunk chunk{std::make_unique<rapidsmpf::PackedData>(
    std::move(packed_columns.metadata), br->move(std::move(packed_columns.gpu_data), stream))};
  EXPECT_TRUE(chunk.is_available());

  auto [reservation, overbooking] = br->reserve(
    rapidsmpf::MemoryType::DEVICE, chunk.into_packed_data_cost(), rapidsmpf::AllowOverbooking::NO);
  auto packed = std::move(chunk).into_packed_data(reservation);
  EXPECT_FALSE(chunk.is_available());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expect, table_chunk{std::move(packed)}.table_view());
}

TEST_F(StreamingTableChunk, ToPackedDataFromTable)
{
  constexpr unsigned int num_rows = 100;
  constexpr std::int64_t seed     = 1337;

  cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);
  table_chunk chunk{std::make_unique<cudf::table>(expect), stream};
  EXPECT_TRUE(chunk.is_available());

  auto [reservation, overbooking] = br->reserve(
    rapidsmpf::MemoryType::DEVICE, chunk.into_packed_data_cost(), rapidsmpf::AllowOverbooking::NO);
  auto packed = std::move(chunk).into_packed_data(reservation);
  EXPECT_FALSE(chunk.is_available());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expect, table_chunk{std::move(packed)}.table_view());
}

TEST_F(StreamingTableChunk, ToPackedDataCost)
{
  constexpr unsigned int num_rows = 100;
  constexpr std::int64_t seed     = 1337;

  cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);

  // Packing an unpacked table allocates, so the cost is the packed size.
  table_chunk from_table{std::make_unique<cudf::table>(expect), stream};
  EXPECT_EQ(from_table.into_packed_data_cost(),
            from_table.data_alloc_size(rapidsmpf::MemoryType::DEVICE));
  EXPECT_GT(from_table.into_packed_data_cost(), 0);

  // Already-packed data is moved out, so it costs nothing.
  auto packed_columns = cudf::pack(expect, stream);
  table_chunk from_packed{std::make_unique<rapidsmpf::PackedData>(
    std::move(packed_columns.metadata), br->move(std::move(packed_columns.gpu_data), stream))};
  EXPECT_EQ(from_packed.into_packed_data_cost(), 0);
}

TEST_F(StreamingTableChunk, ToPackedDataWithReservation)
{
  constexpr unsigned int num_rows = 100;
  constexpr std::int64_t seed     = 1337;

  cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);
  table_chunk chunk{std::make_unique<cudf::table>(expect), stream};
  auto const cost = chunk.into_packed_data_cost();

  auto [reservation, overbooking] =
    br->reserve(rapidsmpf::MemoryType::DEVICE, cost, rapidsmpf::AllowOverbooking::NO);
  EXPECT_EQ(overbooking, 0);

  auto packed = std::move(chunk).into_packed_data(reservation);
  // The pack consumed exactly the cost.
  EXPECT_EQ(reservation.size(), 0);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expect, table_chunk{std::move(packed)}.table_view());
}

TEST_F(StreamingTableChunk, ToPackedDataRejectsSmallReservation)
{
  constexpr unsigned int num_rows = 100;
  constexpr std::int64_t seed     = 1337;

  cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);
  table_chunk chunk{std::make_unique<cudf::table>(expect), stream};

  auto [reservation, overbooking] = br->reserve(rapidsmpf::MemoryType::DEVICE,
                                                chunk.into_packed_data_cost() - 1,
                                                rapidsmpf::AllowOverbooking::NO);
  EXPECT_EQ(overbooking, 0);
  EXPECT_THROW(
    { [[maybe_unused]] auto packed = std::move(chunk).into_packed_data(reservation); },
    rapidsmpf::reservation_error);
}

TEST_P(StreamingTableChunk, ToMessageCopy)
{
  auto const [spill_mem_type, ncols, nrows] = GetParam();
  if (spill_mem_type == rapidsmpf::MemoryType::PINNED_HOST &&
      !rapidsmpf::is_pinned_memory_resources_supported()) {
    GTEST_SKIP() << "MemoryType::PINNED_HOST isn't supported on the system.";
  }

  constexpr std::int64_t seed = 2025;
  constexpr std::uint64_t seq = 7;

  auto expect = random_table(seed, nrows, ncols, 0, 5);
  auto const expected_packed_size =
    cudf::packed_size(expect.view(), stream, rmm::mr::get_current_device_resource_ref());
  EXPECT_GE(expected_packed_size, expect.alloc_size());
  auto chunk = std::make_unique<table_chunk>(std::make_unique<cudf::table>(expect), stream);

  rapidsmpf::streaming::Message m = to_message(seq, std::move(chunk));
  EXPECT_EQ(m.sequence_number(), seq);
  EXPECT_FALSE(m.empty());
  EXPECT_TRUE(m.holds<table_chunk>());
  EXPECT_TRUE(m.content_description().spillable());
  EXPECT_EQ(m.content_description().content_size(rapidsmpf::MemoryType::HOST), 0);
  EXPECT_EQ(m.content_description().content_size(rapidsmpf::MemoryType::DEVICE),
            expected_packed_size);
  EXPECT_EQ(m.copy_cost(), expected_packed_size);

  // Deep copy: device → mem_type.
  // The copy cost includes cudf's packed-buffer alignment and is therefore sufficient
  // before pack() allocates its output.
  auto reservation                 = br->reserve_or_fail(m.copy_cost(), spill_mem_type);
  rapidsmpf::streaming::Message m2 = m.copy(reservation);
  EXPECT_EQ(reservation.size(), 0);
  EXPECT_FALSE(m2.empty());
  EXPECT_TRUE(m2.holds<table_chunk>());
  EXPECT_TRUE(m2.content_description().spillable());
  EXPECT_EQ(m2.copy_cost(), expected_packed_size);
  EXPECT_EQ(m2.content_description().content_size(spill_mem_type), expected_packed_size);
  EXPECT_EQ(m2.content_description().content_size(rapidsmpf::MemoryType::DEVICE), 0);
  EXPECT_EQ(m2.sequence_number(), seq);
}
