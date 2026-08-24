/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace cudf_streaming::detail {

/**
 * @brief A bloom filter, used for approximate set membership queries.
 *
 * @note All methods of this class launch work on the streams provided. It is the caller's
 * responsibility to ensure that data are valid to read/write on that stream as
 * appropriate.
 */
struct device_bloom_filter {
  /**
   * @brief Create a filter.
   *
   * @param filter_size Filter storage size in bytes. Must be a positive multiple of the filter
   * block size and no greater than the maximum supported by the filter policy.
   * @param seed Seed used for hashing each value.
   * @param storage Storage to view as a bloom filter, must be appropriately
   * initialized.
   */
  device_bloom_filter(std::size_t filter_size, std::uint64_t seed, void* storage);

  /**
   * @brief Create a read-only filter.
   *
   * @param filter_size Filter storage size in bytes. Must be a positive multiple of the filter
   * block size and no greater than the maximum supported by the filter policy.
   * @param seed Seed used for hashing each value.
   * @param storage View of storage, must be appropriately initialized.
   *
   * @return A const-qualified bloom filter viewing the underlying storage.
   */
  static device_bloom_filter const view(std::size_t filter_size,
                                        std::uint64_t seed,
                                        void const* storage);

  /**
   * @brief Create uninitialized storage for a filter.
   *
   * @param filter_size Filter storage size in bytes. Must be a positive multiple of the filter
   * block size and no greater than the maximum supported by the filter policy.
   * @param stream CUDA stream for device operations.
   * @param mr Memory resource for allocations.
   *
   * @return Unique pointer to a device buffer containing storage for the requested
   * filter size.
   */
  static std::unique_ptr<rmm::device_buffer> storage(std::size_t filter_size,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr);

  /**
   * @brief Find the largest valid filter size no greater than a byte count.
   *
   * @param size Byte count to align.
   * @return Largest valid filter size less than or equal to `size`.
   */
  [[nodiscard]] static std::size_t aligned_size(std::size_t size) noexcept;

  /**
   *  Return the largest storage size supported by the filter policy.
   *
   *  Maximum valid filter size in bytes.
   */
  [[nodiscard]] static std::size_t max_size() noexcept;

  /**
   * @brief Add values to the filter.
   *
   * @param values_to_hash table of values to hash (with cudf::hashing::xxhash_64())
   * @param stream CUDA stream for allocations and device operations.
   * @param mr Memory resource for allocations.
   */
  void add(cudf::table_view const& values_to_hash,
           rmm::cuda_stream_view stream,
           rmm::device_async_resource_ref mr);

  /**
   * @brief Merge two filters, computing their union.
   *
   * @param other Other filter to merge into this one.
   * @param stream CUDA stream for device operations.
   *
   * @throws std::logic_error If `other` is not compatible with this filter.
   */
  void merge(device_bloom_filter const& other, rmm::cuda_stream_view stream);

  /**
   * @brief Return a mask of which rows are contained in the filter.
   *
   * @param values Value to check for set membership
   * @param stream CUDA stream for allocations and device operations.
   * @param mr Memory resource for allocations.
   *
   * @return Mask vector to be used for filtering the table.
   */
  [[nodiscard]] rmm::device_uvector<bool> contains(cudf::table_view const& values,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr) const;

  /**
   * @brief @return Pointer to the underlying storage.
   */
  [[nodiscard]] void* data() noexcept;

  /**
   * @brief @return Const Pointer to the underlying storage.
   */
  [[nodiscard]] void const* data() const noexcept;

  /**
   * @brief @return Size in bytes of the underlying storage.
   */
  [[nodiscard]] std::size_t size() const noexcept;

 private:
  std::size_t num_blocks_;  ///< Number of blocks used in the filter.
  std::uint64_t seed_;      ///< Seed used when hashing values.
  void* storage_;           ///< Backing storage.
};

}  // namespace cudf_streaming::detail
