/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/types.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/streaming/coll/allgather.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace rapidsmpf::ndsh {
///< @brief Treatment of keys in the result of a join
enum class KeepKeys : bool {
  NO,   ///< Key columns do not appear in the output
  YES,  ///< Key columns do appear in the output
};

/**
 * @brief Broadcast the concatenation of all input messages to all ranks.
 *
 * @note Receives all input chunks, gathers from all ranks, and then provides concatenated
 * output.
 *
 * @param ctx Streaming context
 * @param comm Communicator for the collective operation.
 * @param ch_in Input channel of `table_chunk`s
 * @param tag Disambiguating tag for allgather
 * @param ordered Should the concatenated output be ordered
 *
 * @return Message containing the concatenation of all the input table chunks.
 */
[[nodiscard]] coro::task<streaming::Message> broadcast(
  std::shared_ptr<streaming::Context> ctx,
  std::shared_ptr<Communicator> comm,
  std::shared_ptr<streaming::Channel> ch_in,
  OpID tag,
  streaming::AllGather::Ordered ordered = streaming::AllGather::Ordered::YES);

/**
 * @brief Broadcast the concatenation of all input messages to all ranks.
 *
 * @note Receives all input chunks, gathers from all ranks, and then provides concatenated
 * output.
 *
 * @param ctx Streaming context
 * @param comm Communicator for the collective operation.
 * @param ch_in Input channel of `table_chunk`s
 * @param ch_out Input channel of a single `table_chunk`
 * @param tag Disambiguating tag for allgather
 * @param ordered Should the concatenated output be ordered
 *
 * @return Coroutine representing the broadcast
 */
[[nodiscard]] streaming::Actor broadcast(
  std::shared_ptr<streaming::Context> ctx,
  std::shared_ptr<Communicator> comm,
  std::shared_ptr<streaming::Channel> ch_in,
  std::shared_ptr<streaming::Channel> ch_out,
  OpID tag,
  streaming::AllGather::Ordered ordered = streaming::AllGather::Ordered::YES);

/**
 * @brief Perform a streaming inner join between two tables.
 *
 * @note This performs a broadcast join, broadcasting the table represented by the `left`
 * channel to all ranks, and then streaming through the chunks of the `right` channel.
 *
 * @param ctx Streaming context.
 * @param comm Communicator for the collective operation.
 * @param left Channel of `table_chunk`s used as the broadcasted build side.
 * @param right Channel of `table_chunk`s joined in turn against the build side.
 * @param ch_out Output channel of `table_chunk`s.
 * @param left_on Column indices of the keys in the left table.
 * @param right_on Column indices of the keys in the right table.
 * @param tag Disambiguating tag for the broadcast of the left table.
 * @param keep_keys Does the result contain the key columns, or only "carrier" value
 * columns
 *
 * @return Coroutine representing the completion of the join.
 */
[[nodiscard]] streaming::Actor inner_join_broadcast(
  std::shared_ptr<streaming::Context> ctx,
  std::shared_ptr<Communicator> comm,
  // We will always choose left as build table and do "broadcast" joins
  std::shared_ptr<streaming::Channel> left,
  std::shared_ptr<streaming::Channel> right,
  std::shared_ptr<streaming::Channel> ch_out,
  std::vector<cudf::size_type> left_on,
  std::vector<cudf::size_type> right_on,
  OpID tag,
  KeepKeys keep_keys = KeepKeys::YES);
/**
 * @brief Perform a streaming inner join between two tables.
 *
 * @note This performs a shuffle join, the left and right channels are required to provide
 * hash-partitioned data in-order.
 *
 * @param ctx Streaming context.
 * @param comm Communicator for the collective operation.
 * @param left Channel of `table_chunk`s in hash-partitioned order.
 * @param right Channel of `table_chunk`s in matching hash-partitioned order.
 * @param ch_out Output channel of `table_chunk`s.
 * @param left_on Column indices of the keys in the left table.
 * @param right_on Column indices of the keys in the right table.
 * @param keep_keys Does the result contain the key columns, or only "carrier" value
 * columns
 *
 * @return Coroutine representing the completion of the join.
 */
[[nodiscard]] streaming::Actor inner_join_shuffle(std::shared_ptr<streaming::Context> ctx,
                                                  std::shared_ptr<Communicator> comm,
                                                  std::shared_ptr<streaming::Channel> left,
                                                  std::shared_ptr<streaming::Channel> right,
                                                  std::shared_ptr<streaming::Channel> ch_out,
                                                  std::vector<cudf::size_type> left_on,
                                                  std::vector<cudf::size_type> right_on,
                                                  KeepKeys keep_keys = KeepKeys::YES);

/**
 * @brief Perform a streaming left semi join between two tables.
 *
 * @note This performs a broadcast join, broadcasting the table represented by the `left`
 * channel to all ranks, and then streaming through the chunks of the `right` channel.
 * The `right` channel is required to provide hash-partitioned data in-order.
 * All of the chunks from the `left` channel must fit in memory at once.
 *
 * @param ctx Streaming context.
 * @param comm Communicator for the collective operation.
 * @param left Channel of `table_chunk`s.
 * @param right Channel of `table_chunk`s in hash-partitioned order (shuffled).
 * @param ch_out Output channel of `table_chunk`s.
 * @param left_on Column indices of the keys in the left table.
 * @param right_on Column indices of the keys in the right table.
 * @param tag Disambiguating tag for the broadcast of the left table.
 * @param keep_keys Does the result contain the key columns, or only "carrier" value
 * columns
 *
 * @return Coroutine representing the completion of the join.
 */
streaming::Actor left_semi_join_broadcast_left(
  std::shared_ptr<streaming::Context> ctx,
  std::shared_ptr<Communicator> comm,
  // We will always choose left as build table and do "broadcast" joins
  std::shared_ptr<streaming::Channel> left,
  std::shared_ptr<streaming::Channel> right,
  std::shared_ptr<streaming::Channel> ch_out,
  std::vector<cudf::size_type> left_on,
  std::vector<cudf::size_type> right_on,
  OpID tag,
  KeepKeys keep_keys);

/**
 * @brief Perform a streaming left semi join between two tables.
 *
 * @note This performs a shuffle join, the left and right channels are required to provide
 * hash-partitioned data in-order.
 *
 * @param ctx Streaming context.
 * @param comm Communicator for the collective operation.
 * @param left Channel of `table_chunk`s in hash-partitioned order.
 * @param right Channel of `table_chunk`s in matching hash-partitioned order.
 * @param ch_out Output channel of `table_chunk`s.
 * @param left_on Column indices of the keys in the left table.
 * @param right_on Column indices of the keys in the right table.
 * @param tag Disambiguating tag for the broadcast of the left table.
 * @param keep_keys Does the result contain the key columns, or only "carrier" value
 * columns
 *
 * @return Coroutine representing the completion of the join.
 */

streaming::Actor left_semi_join_shuffle(std::shared_ptr<streaming::Context> ctx,
                                        std::shared_ptr<Communicator> comm,
                                        std::shared_ptr<streaming::Channel> left,
                                        std::shared_ptr<streaming::Channel> right,
                                        std::shared_ptr<streaming::Channel> ch_out,
                                        std::vector<cudf::size_type> left_on,
                                        std::vector<cudf::size_type> right_on,
                                        KeepKeys keep_keys = KeepKeys::YES);

/**
 * @brief Shuffle the input channel by hash-partitioning on given key columns.
 *
 * @param ctx Streaming context.
 * @param comm Communicator for the collective operation.
 * @param ch_in Channel of `table_chunk`s to shuffle.
 * @param ch_out Channel of shuffled `table_chunk`s.
 * @param keys Indices of key columns to shuffle on.
 * @param num_partitions Number of output partitions of the shuffle.
 * @param tag Disambiguating tag for the shuffle.
 *
 * @return Coroutine representing the completion of the shuffle.
 */
[[nodiscard]] streaming::Actor shuffle(std::shared_ptr<streaming::Context> ctx,
                                       std::shared_ptr<Communicator> comm,
                                       std::shared_ptr<streaming::Channel> ch_in,
                                       std::shared_ptr<streaming::Channel> ch_out,
                                       std::vector<cudf::size_type> keys,
                                       std::uint32_t num_partitions,
                                       OpID tag);

}  // namespace rapidsmpf::ndsh
