/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/message.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace cudf_streaming {

/**
 * @brief Global row-count and approximate distinct-row statistics.
 */
struct cardinality_estimate {
  std::size_t row_count{};       ///< Exact number of input rows across all ranks.
  std::size_t distinct_count{};  ///< Approximate number of distinct rows across all ranks.
};

/**
 * @brief Convert a cardinality estimate into a streaming message.
 *
 * @param sequence_number Ordering identifier for the message.
 * @param estimate Estimate payload.
 * @return Message containing the estimate.
 */
[[nodiscard]] rapidsmpf::streaming::Message to_message(
  std::uint64_t sequence_number, std::unique_ptr<cardinality_estimate> estimate);

/**
 * @brief Distributed approximate distinct-row estimator.
 *
 * Each rank builds a local HyperLogLog sketch. The sketches are merged
 * register-wise and row counts are summed in a single all-reduce, producing the
 * global union cardinality and exact global row count on every rank.
 * Nulls and NaNs are included in the distinct-row sketch.
 */
class cardinality_estimator {
 public:
  /**
   * @brief Construct a distributed cardinality estimator.
   *
   * @param ctx Streaming context.
   * @param comm Communicator for the collective operation.
   * @param tag Collective operation identifier.
   * @param precision HyperLogLog precision in the range [4, 18].
   */
  explicit cardinality_estimator(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                 std::shared_ptr<rapidsmpf::Communicator> comm,
                                 rapidsmpf::OpID tag,
                                 std::int32_t precision = 14);

  /**
   * @brief Get the communicator associated with this estimator.
   *
   * @return Shared pointer to the communicator.
   */
  [[nodiscard]] std::shared_ptr<rapidsmpf::Communicator> const& comm() const noexcept
  {
    return comm_;
  }

  /**
   * @brief Get the HyperLogLog precision.
   *
   * @return Precision used for the sketch.
   */
  [[nodiscard]] std::int32_t precision() const noexcept { return precision_; }

  /**
   * @brief Get the collective operation identifier.
   *
   * @return Collective operation identifier.
   */
  [[nodiscard]] rapidsmpf::OpID tag() const noexcept { return tag_; }

  /**
   * @brief Estimate global input cardinality.
   *
   * The input channel must contain `table_chunk` payloads. The output channel
   * receives exactly one `cardinality_estimate`. When @p ch_sampled is provided, each input chunk
   * is forwarded unchanged after it has been added to the sketch. The sampled channel must be
   * consumed concurrently: waiting for the cardinality estimate before consuming sampled chunks
   * can deadlock when channel backpressure blocks forwarding.
   *
   * @param ch_in Input channel of table chunks.
   * @param ch_out Output channel receiving one estimate.
   * @param ch_sampled Optional output channel receiving the input chunks.
   * @param column_indices Columns whose row tuples are counted. An empty vector selects all
   * columns.
   * @return Coroutine representing the estimation.
   */
  [[nodiscard]] rapidsmpf::streaming::Actor estimate(
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_sampled = nullptr,
    std::vector<cudf::size_type> column_indices               = {});

 private:
  std::shared_ptr<rapidsmpf::streaming::Context> ctx_{};
  std::shared_ptr<rapidsmpf::Communicator> comm_{};
  rapidsmpf::OpID tag_{};
  std::int32_t precision_{};
};

}  // namespace cudf_streaming
