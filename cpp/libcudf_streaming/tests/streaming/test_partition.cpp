/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../utils.hpp"
#include "base_streaming_fixture.hpp"

#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf_streaming/partition.hpp>
#include <cudf_streaming/table_chunk.hpp>

#include <gmock/gmock.h>
#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/streaming/chunks/partition.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/leaf_actor.hpp>

using namespace cudf_streaming;

using StreamingPartition = BaseStreamingFixture;

TEST_F(StreamingPartition, PackUnpackRoundTrip)
{
  int const num_partitions              = 5;
  int const num_rows                    = 100;
  int const num_chunks                  = 10;
  std::int64_t const seed               = 42;
  constexpr cudf::hash_id hash_function = cudf::hash_id::HASH_MURMUR3;

  std::vector<cudf::table> expects;
  for (int i = 0; i < num_chunks; ++i) {
    expects.push_back(random_table_with_index(seed + i, num_rows, 0, 10));
  }

  std::vector<rapidsmpf::streaming::Message> inputs;
  for (int i = 0; i < num_chunks; ++i) {
    inputs.emplace_back(to_message(
      i,
      std::make_unique<table_chunk>(
        std::make_unique<cudf::table>(expects[i], stream, ctx->br()->device_mr()), stream)));
  }

  // Create and run the streaming pipeline.
  std::vector<rapidsmpf::streaming::Message> outputs;
  {
    std::vector<rapidsmpf::streaming::Actor> actors;
    auto ch1 = ctx->create_channel();
    actors.push_back(rapidsmpf::streaming::actor::push_to_channel(ctx, ch1, std::move(inputs)));

    auto ch2 = ctx->create_channel();
    actors.push_back(cudf_streaming::actor::partition_and_pack(
      ctx, ch1, ch2, {1}, num_partitions, hash_function, seed));

    auto ch3 = ctx->create_channel();
    actors.push_back(cudf_streaming::actor::unpack_and_concat(ctx, ch2, ch3));

    actors.push_back(rapidsmpf::streaming::actor::pull_from_channel(ctx, ch3, outputs));

    rapidsmpf::streaming::run_actor_network(std::move(actors));
  }

  EXPECT_EQ(expects.size(), outputs.size());
  for (std::size_t i = 0; i < expects.size(); ++i) {
    EXPECT_EQ(outputs[i].sequence_number(), i);
    auto output = outputs[i].release<table_chunk>();
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sort_table(output.table_view()),
                                       sort_table(expects[i].view()));
  }
}
