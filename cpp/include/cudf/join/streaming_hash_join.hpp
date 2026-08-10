/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>
#include <optional>
#include <span>
#include <utility>

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 * @file
 */

namespace detail {
class streaming_hash_join;
}  // namespace detail

/**
 * @brief Streaming hash join that accepts the right (build) side incrementally via `insert()`.
 *
 * The persistent hash table is sized at construction time to accommodate `total_right_rows`
 * cumulative right-side rows. Right-side partitions are fed in via `insert()` and are not
 * deep-copied; the caller must keep the source columns of every inserted partition alive until
 * this object is destroyed.
 *
 * This shape mirrors `cudf::groupby::streaming_groupby`. It is intended for query engines that
 * receive partitioned right-side data (e.g. inter-GPU exchange), avoiding the ~2x peak memory of
 * concatenating partitions before building a join table.
 *
 * @note All NaNs are considered equal.
 */
class streaming_hash_join {
 public:
  streaming_hash_join() = delete;
  ~streaming_hash_join();
  streaming_hash_join(streaming_hash_join const&)            = delete;
  streaming_hash_join& operator=(streaming_hash_join const&) = delete;

  /** @brief Move constructor. */
  streaming_hash_join(streaming_hash_join&&) noexcept;

  /**
   * @brief Move assignment operator.
   * @return Reference to this object.
   */
  streaming_hash_join& operator=(streaming_hash_join&&) noexcept;

  /**
   * @brief Construct a streaming hash join with a persistent hash table sized to accommodate
   *        `total_right_rows` cumulative right-side rows.
   *
   * @throws std::invalid_argument if `right_schema` is empty
   * @throws std::invalid_argument if `right_key_indices` is empty or out of range
   * @throws std::invalid_argument if `total_right_rows` is negative
   * @throws std::invalid_argument if `max_num_batches` is not positive
   * @throws std::invalid_argument if `load_factor` is not in (0, 1]
   *
   * @param right_schema Column types of every right-side partition. All partitions inserted later
   *                     must have the same schema.
   * @param right_key_indices Indices into `right_schema` identifying the join-key columns.
   * @param total_right_rows Upper bound on the cumulative number of right-side rows that will be
   *                         inserted; the persistent hash table is sized accordingly.
   * @param max_num_batches Maximum number of batches. The batch ID uses
   *                        `ceil(log2(max_num_batches))` high row-hash bits.
   * @param has_nulls Whether the right table (or any later left table) may contain nulls in the
   *                  key columns.
   * @param compare_nulls Controls whether null join-key values should match or not.
   * @param load_factor Target hash-table occupancy ratio in (0, 1]. Defaults to 0.5.
   * @param stream CUDA stream used to allocate and initialize the persistent hash table.
   * @param mr Device memory resource used for persistent allocations. The resource is owned by the
   *           join object and must be copyable.
   */
  streaming_hash_join(std::span<data_type const> right_schema,
                      std::span<size_type const> right_key_indices,
                      size_type total_right_rows,
                      size_type max_num_batches,
                      nullable_join has_nulls,
                      null_equality compare_nulls,
                      double load_factor           = 0.5,
                      rmm::cuda_stream_view stream = cudf::get_default_stream(),
                      cuda::mr::any_resource<cuda::mr::device_accessible> mr =
                        cudf::get_current_device_resource_ref());

  /**
   * @brief Insert a right-side partition into the persistent hash table.
   *
   * The partition is not deep-copied; the caller must keep `right_partition` and the columns it
   * views alive until this object is destroyed. The row index stored for each hash-table entry is
   * local to this partition.
   *
   * This function may be called concurrently from multiple host threads. Batch IDs are assigned
   * in an unspecified order when calls overlap. All `insert()` calls must return, and the caller
   * must establish the necessary CUDA stream dependencies, before calling `inner_join()`.
   *
   * @throws std::invalid_argument if `right_partition`'s schema does not match the schema passed
   *                               to the constructor
   * @throws std::invalid_argument if inserting this partition would push the cumulative row count
   *                               above `total_right_rows`
   * @throws std::invalid_argument if inserting this partition would exceed `max_num_batches`
   *
   * @param right_partition The right-side partition to insert.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   */
  void insert(cudf::table_view const& right_partition,
              rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Returns the row indices that can be used to construct the result of an inner join
   *        between the accumulated right-side partitions and the given `left` table.
   *
   * The returned right-side indices identify the source partition and the row within that
   * partition.
   *
   * @param left The left table, from which the tuples are probed.
   * @param output_size Optional exact output size hint to avoid an extra count pass.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned device memory.
   * @return Pair `[left_indices, [right_batch_indices, right_row_indices]]`. For each match the
   *         right side is identified by `(batch_idx, row_idx)`, where `batch_idx` is the
   *         insertion order of the partition this row came from and `row_idx` is the local
   *         row index within that partition.
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                                    std::unique_ptr<rmm::device_uvector<size_type>>>>
  inner_join(cudf::table_view const& left,
             std::optional<std::size_t> output_size = {},
             rmm::cuda_stream_view stream           = cudf::get_default_stream(),
             rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

 private:
  std::unique_ptr<cudf::detail::streaming_hash_join> _impl;
};

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
