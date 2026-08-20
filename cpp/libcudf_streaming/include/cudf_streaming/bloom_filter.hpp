/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/types.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace cudf_streaming {

/**
 * @brief Utility managing construction and use of a bloom filter.
 *
 * This class provides methods to build a bloom filter from a stream of `table_chunk`s and
 * then apply that filter to a different stream.
 *
 * A bloom filter is a fixed size probabilistic data structure that provides approximate
 * set membership queries with no false negatives. That is, let `A` be some set and `f(A)`
 * be the bloom filter representation of that set. Then, for all `a ∈ A` it holds that `a
 * ∈ f(A)`. Conversely, there is a false positive rate that increases with the number of
 * distinct values inserted into the bloom filter, and decreases with the number of filter
 * blocks. That is, for any given bloom filter, there exists `a ∉ A` such that `a ∈ f(A)`.
 *
 * See https://arxiv.org/pdf/2512.15595 for details on the GPU implementation used.
 *
 * We use bloom filters to provide runtime pre-filtering of tables during shuffle-based
 * joins. We gather the keys that will match from the build side and use those to
 * pre-filter the probe side before shuffling.
 */
struct bloom_filter {
  /**
   * @brief Construct storage for a bloom filter.
   *
   * @param ctx Streaming context.
   * @param comm Communicator for the collective operation.
   * @param seed Hash seed used when hashing values into the filter.
   * @param filter_size Filter storage size in bytes. Must be positive and satisfy
   * `aligned_size(filter_size) == filter_size`, and must not exceed the maximum size supported by
   * the filter policy.
   *
   * @throws std::logic_error If `filter_size` is zero, incorrectly aligned, or exceeds the policy
   * maximum.
   */
  explicit bloom_filter(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                        std::shared_ptr<rapidsmpf::Communicator> comm,
                        std::uint64_t seed,
                        std::size_t filter_size);

  /**
   * @brief Find the largest valid filter size no greater than a byte count.
   *
   * @param size Byte count to align.
   * @return Largest valid filter size less than or equal to `size`.
   */
  [[nodiscard]] static std::size_t aligned_size(std::size_t size) noexcept;

  /**
   * @brief Gets the communicator associated with this bloom_filter.
   *
   * @return Shared pointer to communicator.
   */
  [[nodiscard]] std::shared_ptr<rapidsmpf::Communicator> const& comm() const noexcept
  {
    return comm_;
  }

  /**
   * @brief Build a bloom filter from the input channel.
   *
   * @param ch_in Input channel of `table_chunk`s to build bloom filter for.
   * @param ch_out Output channel receiving a single message containing the bloom
   * filter.
   * @param tag Disambiguating tag to combine filters across ranks.
   * @return Coroutine representing the construction of the bloom filter.
   */
  [[nodiscard]] rapidsmpf::streaming::Actor build(
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    rapidsmpf::OpID tag);

  /**
   * @brief Apply a bloom filter to an input channel.
   *
   * @param bloom_filter Channel containing the bloom filter (a single message).
   * @param ch_in Input channel of `table_chunk`s to apply bloom filter to.
   * @param ch_out Output channel receiving filtered `table_chunk`s.
   * @param keys Indices selecting the key columns for the hash fingerprint
   *
   * @note The application of the bloom filter expects _exactly one_ message to come
   * through the `bloom_filter` channel, which must be drained after that message is
   * sent.
   *
   * @return Coroutine representing the application of the bloom filter.
   */
  [[nodiscard]] rapidsmpf::streaming::Actor apply(
    std::shared_ptr<rapidsmpf::streaming::Channel> bloom_filter,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::vector<cudf::size_type> keys);

 private:
  std::shared_ptr<rapidsmpf::streaming::Context> ctx_{};
  std::shared_ptr<rapidsmpf::Communicator> comm_{};
  std::uint64_t seed_{};
  std::size_t filter_size_{};
};
}  // namespace cudf_streaming
