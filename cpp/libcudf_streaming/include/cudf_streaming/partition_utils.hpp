/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/contiguous_split.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>

#include <memory>
#include <unordered_map>
#include <vector>

namespace cudf_streaming {

/**
 * @brief Partitions rows from the input table into multiple output tables.
 *
 * @param table The table to partition.
 * @param columns_to_hash Indices of input columns to hash.
 * @param num_partitions The number of partitions.
 * @param hash_function Hash function to use.
 * @param seed Seed value to the hash function.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param br Buffer resource for memory allocations.
 * @param allow_overbooking If true, allow overbooking (true by default)
 *
 * @return A vector of each partition and a table that owns the device memory.
 *
 * @throws std::out_of_range if index is `columns_to_hash` is invalid
 *
 * @see cudf::hash_partition
 * @see cudf::split
 */
[[nodiscard]] std::pair<std::vector<cudf::table_view>, std::unique_ptr<cudf::table>>
partition_and_split(
  cudf::table_view const& table,
  std::vector<cudf::size_type> const& columns_to_hash,
  int num_partitions,
  cudf::hash_id hash_function,
  std::uint32_t seed,
  rmm::cuda_stream_view stream,
  rapidsmpf::BufferResource* br,
  rapidsmpf::AllowOverbooking allow_overbooking = rapidsmpf::AllowOverbooking::YES);

/**
 * @brief Partitions rows from the input table into multiple packed (serialized) tables.
 *
 * @param table The table to partition.
 * @param columns_to_hash Indices of input columns to hash.
 * @param num_partitions The number of partitions to use.
 * @param hash_function Hash function to use.
 * @param seed Seed value to the hash function.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param br Buffer resource for memory allocations.
 * @param allow_overbooking If true, allow overbooking (true by default)
 * // TODO: disable this by default https://github.com/rapidsmpf/rapidsmpf/issues/449
 *
 * @return A map of partition IDs and their packed tables.
 *
 * @throws std::out_of_range if index is `columns_to_hash` is invalid
 *
 * @see unpack_and_concat
 * @see cudf::hash_partition
 * @see cudf::pack
 */
[[nodiscard]] std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData>
partition_and_pack(
  cudf::table_view const& table,
  std::vector<cudf::size_type> const& columns_to_hash,
  int num_partitions,
  cudf::hash_id hash_function,
  std::uint32_t seed,
  rmm::cuda_stream_view stream,
  rapidsmpf::BufferResource* br,
  rapidsmpf::AllowOverbooking allow_overbooking = rapidsmpf::AllowOverbooking::YES);

/**
 * @brief Splits rows from the input table into multiple packed (serialized) tables.
 *
 * @param table The table to split and pack into partitions.
 * @param splits The split points, equivalent to cudf::split(), i.e. one less than
 * the number of result partitions.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param br Buffer resource for memory allocations.
 * @param allow_overbooking If true, allow overbooking (true by default)
 * // TODO: disable this by default https://github.com/rapidsmpf/rapidsmpf/issues/449
 *
 * @return A map of partition IDs and their packed tables.
 *
 * @throws std::out_of_range if the splits are invalid.
 *
 * @see unpack_and_concat
 * @see cudf::split
 * @see partition_and_pack
 */
[[nodiscard]] std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> split_and_pack(
  cudf::table_view const& table,
  std::vector<cudf::size_type> const& splits,
  rmm::cuda_stream_view stream,
  rapidsmpf::BufferResource* br,
  rapidsmpf::AllowOverbooking allow_overbooking = rapidsmpf::AllowOverbooking::YES);

/**
 * @brief Unpack (deserialize) input partitions and concatenate them into a single table.
 *
 * Empty partitions are ignored.
 *
 * The unpacking of each partition is stream-ordered on that partition's own CUDA stream.
 * The returned table is stream-ordered on the provided @p stream and synchronized with
 * the unpacking.
 *
 * @param partitions Packed input tables (partitions).
 * @param stream CUDA stream on which concatenation occurs and on which the resulting
 * table is ordered.
 * @param br Buffer resource used for memory allocations.
 * @param allow_overbooking If true, allow overbooking (true by default).
 * @return The concatenated table resulting from unpacking the input partitions.
 *
 * @throws rapidsmpf::reservation_error If the buffer resource cannot reserve enough
 * memory to concatenate all partitions.
 * @throws std::logic_error If the partitions are not in device memory.
 *
 * @see partition_and_pack
 * @see cudf::unpack
 * @see cudf::concatenate
 */
[[nodiscard]] std::unique_ptr<cudf::table> unpack_and_concat(
  std::vector<rapidsmpf::PackedData>&& partitions,
  rmm::cuda_stream_view stream,
  rapidsmpf::BufferResource* br,
  rapidsmpf::AllowOverbooking allow_overbooking = rapidsmpf::AllowOverbooking::YES);

}  // namespace cudf_streaming
