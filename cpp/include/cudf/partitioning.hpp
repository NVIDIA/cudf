/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/hashing.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <vector>

/**
 * @file
 * @brief Column partitioning APIs
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup reorder_partition
 * @{
 */

/**
 * @brief Identifies the hash function to be used in hash partitioning
 */
enum class hash_id : int32_t {
  HASH_IDENTITY = 0,  ///< Identity hash function that simply returns the key to be hashed
  HASH_MURMUR3        ///< Murmur3 hash function
};

/**
 * @brief Partitions rows of `t` according to the mapping specified by
 * `partition_map`.
 *
 * For each row at `i` in `t`, `partition_map[i]` indicates which partition row
 * `i` belongs to. `partition` creates a new table by rearranging the rows of
 * `t` such that rows in the same partition are contiguous. The returned table
 * is in ascending partition order from `[0, num_partitions)`. The order within
 * each partition is undefined.
 *
 * Returns a `vector<size_type>` of `num_partitions + 1` values that indicate
 * the starting position of each partition within the returned table, i.e.,
 * partition `i` starts at `offsets[i]` (inclusive) and ends at `offset[i+1]`
 * (exclusive). As a result, if value `j` in `[0, num_partitions)` does not
 * appear in `partition_map`, partition `j` will be empty, i.e.,
 * `offsets[j+1] - offsets[j] == 0`.
 *
 * Values in `partition_map` must be in the range `[0, num_partitions)`,
 * otherwise behavior is undefined.
 *
 * @throw cudf::logic_error when `partition_map` is a non-integer type
 * @throw cudf::logic_error when `partition_map.has_nulls() == true`
 * @throw cudf::logic_error when `partition_map.size() != t.num_rows()`
 *
 * @param t The table to partition
 * @param partition_map Non-nullable column of integer values that map each row
 * in `t` to its partition.
 * @param num_partitions The total number of partitions
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return Pair containing the reordered table and vector of `num_partitions +
 * 1` offsets to each partition such that the size of partition `i` is
 * determined by `offset[i+1] - offset[i]`.
 */
std::pair<std::unique_ptr<table>, std::vector<size_type>> partition(
  table_view const& t,
  column_view const& partition_map,
  size_type num_partitions,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Partitions rows from the input table into multiple output tables.
 *
 * Partitions rows of `input` into `num_partitions` bins based on the hash
 * value of the columns specified by `columns_to_hash`. Rows partitioned into
 * the same bin are grouped consecutively in the output table. Returns a vector
 * of `num_partitions + 1` offsets, where partition `i` contains rows in the range
 * `[offsets[i], offsets[i+1])`. The last offset is always equal to the total
 * number of rows in the output table.
 *
 * An empty `columns_to_hash` is treated as empty, producing an empty result even when `input` has a
 * non-zero row count.
 *
 * @throw std::out_of_range if index is `columns_to_hash` is invalid
 *
 * @param input The table to partition
 * @param columns_to_hash Indices of input columns to hash
 * @param num_partitions The number of partitions to use
 * @param hash_function Optional hash id that chooses the hash function to use
 * @param seed Optional seed value to the hash function
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 *
 * @returns An output table and a vector of `num_partitions + 1` row offsets to each partition
 */
std::pair<std::unique_ptr<table>, std::vector<size_type>> hash_partition(
  table_view const& input,
  std::vector<size_type> const& columns_to_hash,
  int num_partitions,
  hash_id hash_function             = hash_id::HASH_MURMUR3,
  uint32_t seed                     = DEFAULT_HASH_SEED,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Partitions rows from the input table into multiple output tables.
 *
 * Partitions rows of `input` into `num_partitions` bins based on the hash
 * value of the rows specified by the `keys` table. Rows partitioned into
 * the same bin are grouped consecutively in the output table. Returns a vector
 * of `num_partitions + 1` offsets, where partition `i` contains rows in the range
 * `[offsets[i], offsets[i+1])`. The last offset is always equal to the total
 * number of rows in the output table.
 *
 * A zero-column `keys` table is treated as empty, producing an empty result even when `input` has a
 * non-zero row count.
 *
 * @throw std::invalid_argument if `keys` is not empty and does not have the same number of rows as
 * `input`.
 *
 * @param input The table to partition
 * @param keys The table of keys to hash defining the partitioning
 * @param num_partitions The number of partitions to use
 * @param hash_function Optional hash id that chooses the hash function to use
 * @param seed Optional seed value to the hash function
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 *
 * @returns An output table and a vector of `num_partitions + 1` row offsets to each partition
 */
std::pair<std::unique_ptr<table>, std::vector<size_type>> hash_partition(
  table_view const& input,
  table_view const& keys,
  int num_partitions,
  hash_id hash_function             = hash_id::HASH_MURMUR3,
  uint32_t seed                     = DEFAULT_HASH_SEED,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Round-robin partition.
 *
 * Returns a new table with rows re-arranged into partition groups and
 * a vector of `num_partitions + 1` offsets, where partition `i` contains rows
 * in the range `[offsets[i], offsets[i+1])`. The last offset is always equal
 * to the total number of rows in the output table.
 * Rows are assigned partitions based on their row index in the table,
 * in a round robin fashion.
 *
 * @throws cudf::logic_error if `num_partitions <= 1`
 * @throws cudf::logic_error if `start_partition >= num_partitions`
 *
 * A good analogy for the algorithm is dealing out cards:
 *
 *  1. The deck of cards is represented as the rows in the table.
 *  2. The number of partitions is the number of players being dealt cards.
 *  3. the start_partition indicates which player starts getting cards first.
 *
 * The algorithm has two outcomes:
 *
 *  1. Another deck of cards formed by stacking each
 *      player's cards back into a deck again,
 *      preserving the order of cards dealt to each player,
 *      starting with player 0.
 *  2. A vector into the output deck indicating where a player's cards start.
 *
 * A player's deck (partition) is the range of cards starting
 * at the corresponding offset and ending at the next player's
 * starting offset or the last card in the deck if it's the last player.
 *
 * When num_partitions > nrows, we have more players than cards.
 * We start dealing to the first indicated player and continuing
 * around the players until we run out of cards before we run out of players.
 * Players that did not get any cards are represented by
 * `offset[i] == offset[i+1]`
 * meaning there are no cards (rows) in their deck (partition).
 *
 * ```
 * Example 1:
 * input:
 * table => col 1 {0, ..., 12}
 * num_partitions = 3
 * start_partition = 0
 *
 * output: pair<table, partition_offsets>
 * table => col 1 {0,3,6,9,12,1,4,7,10,2,5,8,11}
 * partition_offsets => {0,5,9,13}
 *
 * Example 2:
 * input:
 * table => col 1 {0, ..., 12}
 * num_partitions = 3
 * start_partition = 1
 *
 * output: pair<table, partition_offsets>
 * table => col 1 {2,5,8,11,0,3,6,9,12,1,4,7,10}
 * partition_offsets => {0,4,9,13}
 *
 * Example 3:
 * input:
 * table => col 1 {0, ..., 10}
 * num_partitions = 3
 * start_partition = 0
 *
 * output: pair<table, partition_offsets>
 * table => col 1 {0,3,6,9,1,4,7,10,2,5,8}
 * partition_offsets => {0,4,8,11}
 *
 * Example 4:
 * input:
 * table => col 1 {0, ..., 10}
 * num_partitions = 3
 * start_partition = 1
 *
 * output: pair<table, partition_offsets>
 * table => col 1 {2,5,8,0,3,6,9,1,4,7,10}
 * partition_offsets => {0,3,7,11}
 *
 * Example 5:
 * input:
 * table => col 1 {0, ..., 10}
 * num_partitions = 3
 * start_partition = 2
 *
 * output: pair<table, partition_offsets>
 * table => col 1 {1,4,7,10,2,5,8,0,3,6,9}
 * partition_offsets => {0,4,7,11}
 *
 * Example 6:
 * input:
 * table => col 1 {0, ..., 10}
 * num_partitions = 15 > num_rows = 11
 * start_partition = 2
 *
 * output: pair<table, partition_offsets>
 * table => col 1 {0,1,2,3,4,5,6,7,8,9,10}
 * partition_offsets => {0,0,0,1,2,3,4,5,6,7,8,9,10,11,11,11}
 *
 * Example 7:
 * input:
 * table => col 1 {0, ..., 10}
 * num_partitions = 15 > num_rows = 11
 * start_partition = 10
 *
 * output: pair<table, partition_offsets>
 * table => col 1 {5,6,7,8,9,10,0,1,2,3,4}
 * partition_offsets => {0,1,2,3,4,5,6,6,6,6,6,7,8,9,10,11}
 *
 * Example 8:
 * input:
 * table => col 1 {0, ..., 10}
 * num_partitions = 15 > num_rows = 11
 * start_partition = 14
 *
 * output: pair<table, partition_offsets>
 * table => col 1 {1,2,3,4,5,6,7,8,9,10,0}
 * partition_offsets => {0,1,2,3,4,5,6,7,8,9,10,10,10,10,10,11}
 *
 * Example 9:
 * input:
 * table => col 1 {0, ..., 10}
 * num_partitions = 11 == num_rows = 11
 * start_partition = 2
 *
 * output: pair<table, partition_offsets>
 * table => col 1 {9,10,0,1,2,3,4,5,6,7,8}
 * partition_offsets => {0,1,2,3,4,5,6,7,8,9,10,11}
 * ```
 *
 * @param[in] input The input table to be round-robin partitioned
 * @param[in] num_partitions Number of partitions for the table
 * @param[in] start_partition Index of the 1st partition
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource used to allocate the returned table's device memory
 *
 * @return A std::pair consisting of a unique_ptr to the partitioned table
 * and a vector of `num_partitions + 1` partition offsets for each partition within the table.
 */
std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> round_robin_partition(
  table_view const& input,
  cudf::size_type num_partitions,
  cudf::size_type start_partition   = 0,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
