/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../utils.hpp"
#include "base_streaming_fixture.hpp"

#include <cudf_test/column_wrapper.hpp>

#include <cudf/table/table.hpp>

#include <cudf_streaming/approx_distinct_count.hpp>
#include <cudf_streaming/table_chunk.hpp>

#include <gtest/gtest.h>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/leaf_actor.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace {

using CardinalityEstimatorTest = BaseStreamingFixture;

TEST_F(CardinalityEstimatorTest, EstimatesGlobalUnionAndCountsRows)
{
  constexpr std::size_t distinct_values = 10'000;
  constexpr std::size_t repetitions     = 2;
  auto values                           = iota_vector<std::int64_t>(distinct_values);
  auto const dups                       = values;
  values.insert(values.end(), dups.begin(), dups.end());

  cudf::test::fixed_width_column_wrapper<std::int64_t> column(values.begin(), values.end());
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column.release());
  auto table = std::make_unique<cudf::table>(std::move(columns));

  std::vector<rapidsmpf::streaming::Message> inputs;
  inputs.push_back(cudf_streaming::to_message(
    0, std::make_unique<cudf_streaming::table_chunk>(std::move(table), stream)));

  auto ch_in  = ctx->create_channel();
  auto ch_out = ctx->create_channel();
  cudf_streaming::cardinality_estimator estimator{
    ctx, GlobalEnvironment->comm_, rapidsmpf::OpID{0}, 14};

  std::vector<rapidsmpf::streaming::Message> outputs;
  std::vector<rapidsmpf::streaming::Actor> actors;
  actors.push_back(rapidsmpf::streaming::actor::push_to_channel(ctx, ch_in, std::move(inputs)));
  actors.push_back(estimator.estimate(ch_in, ch_out));
  actors.push_back(rapidsmpf::streaming::actor::pull_from_channel(ctx, ch_out, outputs));

  rapidsmpf::streaming::run_actor_network(std::move(actors));

  ASSERT_EQ(outputs.size(), 1);
  auto const estimate = outputs.front().release<cudf_streaming::cardinality_estimate>();
  EXPECT_EQ(estimate.row_count,
            distinct_values * repetitions *
              rapidsmpf::safe_cast<std::size_t>(GlobalEnvironment->comm_->nranks()));
  EXPECT_NEAR(
    estimate.distinct_count, distinct_values, static_cast<double>(distinct_values) * 0.05);
}

TEST_F(CardinalityEstimatorTest, EmptyInput)
{
  auto ch_in  = ctx->create_channel();
  auto ch_out = ctx->create_channel();
  cudf_streaming::cardinality_estimator estimator{
    ctx, GlobalEnvironment->comm_, rapidsmpf::OpID{0}};

  std::vector<rapidsmpf::streaming::Message> outputs;
  std::vector<rapidsmpf::streaming::Actor> actors;
  actors.push_back(rapidsmpf::streaming::actor::push_to_channel(ctx, ch_in, {}));
  actors.push_back(estimator.estimate(ch_in, ch_out));
  actors.push_back(rapidsmpf::streaming::actor::pull_from_channel(ctx, ch_out, outputs));

  rapidsmpf::streaming::run_actor_network(std::move(actors));

  ASSERT_EQ(outputs.size(), 1);
  auto const estimate = outputs.front().release<cudf_streaming::cardinality_estimate>();
  EXPECT_EQ(estimate.row_count, 0);
  EXPECT_EQ(estimate.distinct_count, 0);
}

TEST_F(CardinalityEstimatorTest, SamplesSelectedColumnsAndForwardsInput)
{
  cudf::test::fixed_width_column_wrapper<std::int64_t> keys{0, 0, 1, 1};
  cudf::test::fixed_width_column_wrapper<std::int64_t> values{0, 1, 2, 3};
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(keys.release());
  columns.push_back(values.release());
  auto table = std::make_unique<cudf::table>(std::move(columns));

  std::vector<rapidsmpf::streaming::Message> inputs;
  inputs.push_back(cudf_streaming::to_message(
    7, std::make_unique<cudf_streaming::table_chunk>(std::move(table), stream)));

  auto ch_in      = ctx->create_channel();
  auto ch_sampled = ctx->create_channel();
  auto ch_out     = ctx->create_channel();
  cudf_streaming::cardinality_estimator estimator{
    ctx, GlobalEnvironment->comm_, rapidsmpf::OpID{0}, 14};

  std::vector<rapidsmpf::streaming::Message> sampled;
  std::vector<rapidsmpf::streaming::Message> estimates;
  std::vector<rapidsmpf::streaming::Actor> actors;
  actors.push_back(rapidsmpf::streaming::actor::push_to_channel(ctx, ch_in, std::move(inputs)));
  actors.push_back(estimator.estimate(ch_in, ch_out, ch_sampled, {0}));
  actors.push_back(rapidsmpf::streaming::actor::pull_from_channel(ctx, ch_sampled, sampled));
  actors.push_back(rapidsmpf::streaming::actor::pull_from_channel(ctx, ch_out, estimates));

  rapidsmpf::streaming::run_actor_network(std::move(actors));

  ASSERT_EQ(sampled.size(), 1);
  EXPECT_EQ(sampled.front().sequence_number(), 7);
  ASSERT_EQ(estimates.size(), 1);
  auto const estimate = estimates.front().release<cudf_streaming::cardinality_estimate>();
  EXPECT_EQ(estimate.row_count,
            4 * rapidsmpf::safe_cast<std::size_t>(GlobalEnvironment->comm_->nranks()));
  EXPECT_NEAR(estimate.distinct_count, 2, 1);
}

TEST_F(CardinalityEstimatorTest, RejectsInvalidPrecision)
{
  EXPECT_THROW(
    cudf_streaming::cardinality_estimator(ctx, GlobalEnvironment->comm_, rapidsmpf::OpID{0}, 3),
    std::invalid_argument);
  EXPECT_THROW(
    cudf_streaming::cardinality_estimator(ctx, GlobalEnvironment->comm_, rapidsmpf::OpID{0}, 19),
    std::invalid_argument);
}

}  // namespace
