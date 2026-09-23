/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/partitioning.hpp>

#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace cudf_streaming::actor {

/**
 * @brief Asynchronously partitions input tables into multiple packed (serialized) tables.
 *
 * This is a streaming version of `rapidsmpf::partition_and_split` that operates on table
 * chunks using channels.
 *
 * It receives tables from an input channel, partitions each row into one of
 * `num_partitions` based on a hash of the selected columns, packs the resulting
 * partitions, and sends them to an output channel.
 *
 * @param ctx The actor context to use.
 * @param ch_in Input channel providing `table_chunk`s to partition.
 * @param ch_out Output channel to which `PartitionMapChunk`s are sent.
 * @param columns_to_hash Indices of input columns to hash.
 * @param num_partitions The number of partitions to use.
 * @param hash_function Hash function to use for partitioning.
 * @param seed Seed value for the hash function.
 *
 * @return Streaming actor representing the asynchronous partitioning and packing
 * operation.
 *
 * @throws std::out_of_range if any index in `columns_to_hash` is invalid.
 *
 * @see rapidsmpf::partition_and_split
 */
rapidsmpf::streaming::Actor partition_and_pack(
  std::shared_ptr<rapidsmpf::streaming::Context> ctx,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
  std::vector<cudf::size_type> columns_to_hash,
  int num_partitions,
  cudf::hash_id hash_function,
  std::uint32_t seed);

/**
 * @brief Asynchronously unpacks and concatenates packed partitions.
 *
 * This is a streaming version of `rapidsmpf::unpack_and_concat` that operates on
 * packed partition chunks using channels.
 *
 * It receives packed partitions from the input channel, deserializes and concatenates
 * them, and sends the resulting tables to the output channel. Empty partitions are
 * ignored.
 *
 * @param ctx The actor context to use.
 * @param ch_in Input channel providing packed partitions as PartitionMapChunk or
 * PartitionVectorChunk.
 * @param ch_out Output channel to which unpacked and concatenated tables are sent.
 *
 * @return Streaming actor representing the asynchronous unpacking and concatenation
 * operation.
 *
 * @see rapidsmpf::unpack_and_concat
 */
rapidsmpf::streaming::Actor unpack_and_concat(
  std::shared_ptr<rapidsmpf::streaming::Context> ctx,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_out);

}  // namespace cudf_streaming::actor
