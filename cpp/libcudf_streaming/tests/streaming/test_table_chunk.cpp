/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../utils.hpp"
#include "base_streaming_fixture.hpp"

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf_streaming/table_chunk.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <rapidsmpf/owning_wrapper.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>

#include <cstdint>
#include <memory>

using namespace cudf_streaming;

class StreamingTableChunk : public BaseStreamingFixture,
                            public ::testing::WithParamInterface<rapidsmpf::MemoryType> {
 protected:
  void SetUp() override
  {
    rapidsmpf::config::Options options(rapidsmpf::config::get_environment_variables());

    std::unordered_map<rapidsmpf::MemoryType, std::int64_t> memory_limits{};
    auto stream_pool =
      std::make_shared<rmm::cuda_stream_pool>(16, rmm::cuda_stream::flags::non_blocking);
    stream = cudf::get_default_stream();
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
      rapidsmpf::Statistics::disabled()   // statistics
    );
    ctx = std::make_shared<rapidsmpf::streaming::Context>(
      options, GlobalEnvironment->comm_->logger(), br);
  }

  rmm::cuda_stream_view stream;
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
  EXPECT_EQ(chunk.stream().value(), stream.value());
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
    EXPECT_EQ(chunk.stream().value(), stream.value());
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

  EXPECT_EQ(chunk.stream().value(), stream.value());
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
                         ::testing::ValuesIn(rapidsmpf::SPILL_TARGET_MEMORY_TYPES),
                         [](testing::TestParamInfo<rapidsmpf::MemoryType> const& info) {
                           return std::string{rapidsmpf::to_string(info.param)};
                         });

TEST_P(StreamingTableChunk, FromPackedDataOn)
{
  auto const spill_mem_type = GetParam();
  if (spill_mem_type == rapidsmpf::MemoryType::PINNED_HOST &&
      !rapidsmpf::is_pinned_memory_resources_supported()) {
    GTEST_SKIP() << "MemoryType::PINNED_HOST isn't supported on the system.";
  }

  constexpr unsigned int num_rows = 100;
  constexpr std::int64_t seed     = 1337;

  cudf::table expect     = random_table_with_index(seed, num_rows, 0, 10);
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

  EXPECT_EQ(chunk.stream().value(), stream.value());
  EXPECT_FALSE(chunk.is_available());
  EXPECT_TRUE(chunk.is_spillable());
  EXPECT_THROW(std::ignore = chunk.table_view(), std::invalid_argument);
  EXPECT_EQ(chunk.make_available_cost(), size);

  auto chunk2 = chunk.make_available(
    br->reserve_or_fail(chunk.make_available_cost(), rapidsmpf::MemoryType::DEVICE));
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

TEST_P(StreamingTableChunk, DeviceToHostRoundTripCopy)
{
  auto const spill_mem_type = GetParam();
  if (spill_mem_type == rapidsmpf::MemoryType::PINNED_HOST &&
      !rapidsmpf::is_pinned_memory_resources_supported()) {
    GTEST_SKIP() << "MemoryType::PINNED_HOST isn't supported on the system.";
  }

  constexpr unsigned int num_rows = 64;
  constexpr std::int64_t seed     = 2025;

  auto expect = random_table_with_index(seed, num_rows, 0, 5);

  table_chunk dev_chunk{std::make_unique<cudf::table>(expect), stream};
  EXPECT_TRUE(dev_chunk.is_available());
  EXPECT_TRUE(dev_chunk.is_spillable());
  EXPECT_EQ(dev_chunk.stream().value(), stream.value());
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
  EXPECT_TRUE(host_copy.is_spillable());
  EXPECT_EQ(host_copy.stream().value(), stream.value());
  EXPECT_GT(host_copy.make_available_cost(), 0);
  {
    auto cd = get_content_description(host_copy);
    EXPECT_EQ(cd.spillable(), host_copy.is_spillable());
    for (auto mem_type : rapidsmpf::MEMORY_TYPES) {
      EXPECT_EQ(cd.content_size(mem_type), host_copy.data_alloc_size(mem_type));
    }
  }

  // Host to host copy.
  auto host_res2  = br->reserve_or_fail(host_copy.data_alloc_size(spill_mem_type), spill_mem_type);
  auto host_copy2 = host_copy.copy(host_res2);
  EXPECT_FALSE(host_copy2.is_available());
  EXPECT_TRUE(host_copy2.is_spillable());
  EXPECT_EQ(host_copy2.stream().value(), stream.value());
  EXPECT_EQ(host_copy2.make_available_cost(), host_copy.make_available_cost());
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
  EXPECT_TRUE(dev_back.is_spillable());
  EXPECT_EQ(dev_back.stream().value(), stream.value());
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

  auto packed = std::move(chunk).into_packed_data(br.get());
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

  auto packed = std::move(chunk).into_packed_data(br.get());
  EXPECT_FALSE(chunk.is_available());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expect, table_chunk{std::move(packed)}.table_view());
}

TEST_P(StreamingTableChunk, ToMessageUnalignedSize)
{
  auto const spill_mem_type = GetParam();
  if (spill_mem_type == rapidsmpf::MemoryType::PINNED_HOST &&
      !rapidsmpf::is_pinned_memory_resources_supported()) {
    GTEST_SKIP() << "MemoryType::PINNED_HOST isn't supported on the system.";
  }

  constexpr unsigned int num_rows = 5;
  constexpr std::int64_t seed     = 2025;
  constexpr std::uint64_t seq     = 7;

  auto expect = random_table_with_index(seed, num_rows, 0, 5);
  auto const expected_packed_size =
    cudf::packed_size(expect.view(), stream, rmm::mr::get_current_device_resource_ref());
  EXPECT_EQ(expect.alloc_size(), 80);
  EXPECT_EQ(expected_packed_size, 128);
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

  // Deep copy: device → host.
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
