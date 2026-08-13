/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "../environment.hpp"

#include <cudf_test/cudf_gtest.hpp>

#include <cudf/utilities/default_stream.hpp>

#include <rmm/mr/cuda_memory_resource.hpp>

#include <gmock/gmock.h>
#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

extern Environment* GlobalEnvironment;

class BaseStreamingFixture : public ::testing::Test {
 protected:
  void SetUp() override
  {
    SetUpWithThreads(1);  // default number of streaming threads
  }

  void TearDown() override
  {
    ctx.reset();
    br.reset();
  }

  void SetUpWithThreads(int num_streaming_threads,
                        std::unordered_map<rapidsmpf::MemoryType, std::int64_t> memory_limits = {})
  {
    // create a new options object, since we can not modify values in the global
    // options object
    auto env_vars                     = rapidsmpf::config::get_environment_variables();
    env_vars["num_streaming_threads"] = std::to_string(num_streaming_threads);
    rapidsmpf::config::Options options(std::move(env_vars));

    stream = cudf::get_default_stream();
    br     = rapidsmpf::BufferResource::create(
      mr_cuda, rapidsmpf::PinnedMemoryDisabled, std::move(memory_limits));
    ctx = std::make_shared<rapidsmpf::streaming::Context>(
      std::move(options), GlobalEnvironment->comm_->logger(), br);
  }

  rmm::cuda_stream_view stream;
  rmm::mr::cuda_memory_resource mr_cuda;
  std::shared_ptr<rapidsmpf::BufferResource> br;
  std::shared_ptr<rapidsmpf::streaming::Context> ctx;
};
