/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/contiguous_split.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>

#include <cuda/stream>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>

#include <memory>
#include <optional>
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
  cuda::stream_ref stream,
  rapidsmpf::BufferResource* br,
  rapidsmpf::AllowOverbooking allow_overbooking = rapidsmpf::AllowOverbooking::YES);

/**
 * @brief Peak device memory (in bytes) required by `partition_and_pack()`.
 *
 * Covers both the table that `cudf::hash_partition()` produces and the packed
 * partitions that `cudf::contiguous_split()` produces from it, which are alive at
 * the same time.
 *
 * @note This is an estimate. Both allocations are made by libcudf against
 * `BufferResource::device_mr()`, so a reservation of this size is accounted for
 * rather than consumed, see `split_and_pack_cost()`.
 *
 * @param table The table to partition.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param temp_mr Memory resource used for temporary allocations.
 * @return The peak size in bytes.
 *
 * @see partition_and_pack
 */
[[nodiscard]] std::size_t partition_and_pack_cost(cudf::table_view const& table,
                                                  cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref temp_mr);

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
 * @param allow_overbooking If true, allow overbooking (true by default). TODO: disable this by
 * default https://github.com/rapidsmpf/rapidsmpf/issues/449
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
  cuda::stream_ref stream,
  rapidsmpf::BufferResource* br,
  rapidsmpf::AllowOverbooking allow_overbooking = rapidsmpf::AllowOverbooking::YES);

/**
 * @brief Partitions rows from the input table into multiple packed (serialized) tables,
 * using a caller-provided memory reservation.
 *
 * Behaves like the `allow_overbooking` overload, except that @p reservation stands in
 * for the device
 * memory reservations it would otherwise make. `partition_and_pack_cost()` bytes are
 * consumed from it as the allocations land, leaving it empty on return.
 *
 * @param table The table to partition.
 * @param columns_to_hash Indices of input columns to hash.
 * @param num_partitions The number of partitions to use.
 * @param hash_function Hash function to use.
 * @param seed Seed value to the hash function.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param reservation Device memory reservation covering `partition_and_pack_cost()`.
 * @param packed_bytes `cudf::packed_size()` of @p table, when the caller already has
 * it. Computed internally when not given, which syncs the stream.
 * @return A map of partition IDs and their packed tables.
 *
 * @throws std::out_of_range if index is `columns_to_hash` is invalid
 * @throws std::invalid_argument if @p reservation is not a device memory reservation.
 * @throws rapidsmpf::reservation_error if @p reservation is smaller than
 * `partition_and_pack_cost()`.
 *
 * @see partition_and_pack_cost
 * @see unpack_and_concat
 * @see cudf::hash_partition
 * @see cudf::pack
 */
[[nodiscard]] std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData>
partition_and_pack(cudf::table_view const& table,
                   std::vector<cudf::size_type> const& columns_to_hash,
                   int num_partitions,
                   cudf::hash_id hash_function,
                   std::uint32_t seed,
                   cuda::stream_ref stream,
                   rapidsmpf::MemoryReservation& reservation,
                   std::optional<std::size_t> packed_bytes = std::nullopt);

/**
 * @brief Peak device memory (in bytes) required by `split_and_pack()`.
 *
 * Covers the packed partitions that `cudf::contiguous_split()` produces.
 *
 * @note This is an estimate. `cudf::contiguous_split()` aligns every column buffer
 * of every partition, so it allocates somewhat more than this for more than one
 * partition. The allocation is made by libcudf against
 * `BufferResource::device_mr()`, so a reservation of this size is accounted for
 * rather than consumed and the shortfall does not fail the call.
 *
 * @param table The table to split and pack into partitions.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param temp_mr Memory resource used for temporary allocations.
 * @return The peak size in bytes.
 *
 * @see split_and_pack
 */
[[nodiscard]] std::size_t split_and_pack_cost(cudf::table_view const& table,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref temp_mr);

/**
 * @brief Splits rows from the input table into multiple packed (serialized) tables.
 *
 * @param table The table to split and pack into partitions.
 * @param splits The split points, equivalent to cudf::split(), i.e. one less than
 * the number of result partitions.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param br Buffer resource for memory allocations.
 * @param allow_overbooking If true, allow overbooking (true by default). TODO: disable this by
 * default https://github.com/rapidsmpf/rapidsmpf/issues/449
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
  cuda::stream_ref stream,
  rapidsmpf::BufferResource* br,
  rapidsmpf::AllowOverbooking allow_overbooking = rapidsmpf::AllowOverbooking::YES);

/**
 * @brief Splits rows from the input table into multiple packed (serialized) tables,
 * using a caller-provided memory reservation.
 *
 * Behaves like the `allow_overbooking` overload, except that @p reservation stands in
 * for the device
 * memory reservation it would otherwise make. `split_and_pack_cost()` bytes are consumed
 * from it as the allocation lands.
 *
 * @param table The table to split and pack into partitions.
 * @param splits The split points, equivalent to cudf::split(), i.e. one less than
 * the number of result partitions.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param reservation Device memory reservation covering `split_and_pack_cost()`.
 * @param packed_bytes `cudf::packed_size()` of @p table, when the caller already has
 * it. Computed internally when not given, which syncs the stream.
 * @return A map of partition IDs and their packed tables.
 *
 * @throws std::out_of_range if the splits are invalid.
 * @throws std::invalid_argument if @p reservation is not a device memory reservation.
 * @throws rapidsmpf::reservation_error if @p reservation is smaller than
 * `split_and_pack_cost()`.
 *
 * @see split_and_pack_cost
 * @see unpack_and_concat
 * @see cudf::split
 */
[[nodiscard]] std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> split_and_pack(
  cudf::table_view const& table,
  std::vector<cudf::size_type> const& splits,
  cuda::stream_ref stream,
  rapidsmpf::MemoryReservation& reservation,
  std::optional<std::size_t> packed_bytes = std::nullopt);

/**
 * @brief Peak device memory (in bytes) required by `unpack_and_concat()`.
 *
 * Covers moving the host-resident partitions back to device memory and the
 * concatenated output.
 *
 * @note The unspill share is exact, since the buffer resource consumes it while
 * moving each partition. The concatenation share is an estimate, as in
 * `split_and_pack_cost()`.
 *
 * @param partitions Packed input tables (partitions).
 * @return The peak size in bytes.
 *
 * @see unpack_and_concat
 */
[[nodiscard]] std::size_t unpack_and_concat_cost(
  std::vector<rapidsmpf::PackedData> const& partitions);

/**
 * @brief Peak device memory (in bytes) required by `unpack_and_concat()`.
 *
 * Overload for callers that hold the partitions individually rather than in a
 * contiguous container.
 *
 * @param partitions Pointers to the packed input tables (partitions). Must not be null,
 * and must not have been moved from.
 * @return The peak size in bytes.
 *
 * @throws std::invalid_argument if a partition is null or has been moved from.
 *
 * @see unpack_and_concat
 */
[[nodiscard]] std::size_t unpack_and_concat_cost(
  std::vector<rapidsmpf::PackedData const*> const& partitions);

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
  cuda::stream_ref stream,
  rapidsmpf::BufferResource* br,
  rapidsmpf::AllowOverbooking allow_overbooking = rapidsmpf::AllowOverbooking::YES);

/**
 * @brief Unpack (deserialize) input partitions and concatenate them into a single table,
 * using a caller-provided memory reservation.
 *
 * Behaves like the `allow_overbooking` overload, except that @p reservation stands in
 * for the device
 * memory reservations it would otherwise make. `unpack_and_concat_cost()` bytes are
 * consumed from it as the allocations land, leaving it empty on return.
 *
 * @param partitions Packed input tables (partitions).
 * @param stream CUDA stream on which concatenation occurs and on which the resulting
 * table is ordered.
 * @param reservation Device memory reservation covering `unpack_and_concat_cost()`.
 * @return The concatenated table resulting from unpacking the input partitions.
 *
 * @throws std::invalid_argument if @p reservation is not a device memory reservation.
 * @throws rapidsmpf::reservation_error if @p reservation is smaller than
 * `unpack_and_concat_cost()`.
 * @throws std::logic_error If the partitions are not in device memory.
 *
 * @see unpack_and_concat_cost
 * @see partition_and_pack
 */
[[nodiscard]] std::unique_ptr<cudf::table> unpack_and_concat(
  std::vector<rapidsmpf::PackedData>&& partitions,
  cuda::stream_ref stream,
  rapidsmpf::MemoryReservation& reservation);

}  // namespace cudf_streaming
