/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../utils.hpp"
#include "base_streaming_fixture.hpp"

#include <cudf_test/table_utilities.hpp>

#include <cudf_streaming/table_chunk.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/content_description.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/leaf_actor.hpp>
#include <rapidsmpf/streaming/core/queue.hpp>

#include <atomic>
#include <memory>
#include <vector>

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;
namespace actor = rapidsmpf::streaming::actor;
using namespace cudf_streaming;

using StreamingLeafTasks = BaseStreamingFixture;

TEST_F(StreamingLeafTasks, PushAndPullChunks)
{
  constexpr int num_rows   = 100;
  constexpr int num_chunks = 10;

  std::vector<cudf::table> expects;
  for (int i = 0; i < num_chunks; ++i) {
    expects.emplace_back(random_table_with_index(i, num_rows, 0, 10));
  }

  std::vector<Actor> actors;
  auto ch1 = ctx->create_channel();

  // Note, we use a scope to check that coroutines keeps the input alive.
  {
    std::vector<Message> inputs;
    for (int i = 0; i < num_chunks; ++i) {
      inputs.emplace_back(to_message(
        i,
        std::make_unique<table_chunk>(
          std::make_unique<cudf::table>(expects[i], stream, ctx->br()->device_mr()), stream)));
    }

    actors.push_back(actor::push_to_channel(ctx, ch1, std::move(inputs)));
  }

  std::vector<Message> outputs;
  actors.push_back(actor::pull_from_channel(ctx, ch1, outputs));

  run_actor_network(std::move(actors));

  EXPECT_EQ(expects.size(), outputs.size());
  for (std::size_t i = 0; i < expects.size(); ++i) {
    EXPECT_EQ(outputs[i].sequence_number(), i);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(outputs[i].get<table_chunk>().table_view(),
                                       expects[i].view());
  }
}
