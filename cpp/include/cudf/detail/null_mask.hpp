/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream_ref>

#include <span>
#include <vector>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::create_null_mask(size_type, mask_state, rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
rmm::device_buffer create_null_mask(size_type size,
                                    mask_state state,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::set_null_mask(bitmask_type*, size_type, size_type, bool)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
void set_null_mask(bitmask_type* bitmask,
                   size_type begin_bit,
                   size_type end_bit,
                   bool valid,
                   cuda::stream_ref stream);

/**
 * @brief Given a bitmask, counts the number of set (1) bits in the range
 * `[start, stop)`.
 *
 * @throws std::invalid_argument if `bitmask == nullptr`
 * @throws std::invalid_argument if `start > stop`
 * @throws std::invalid_argument if `start < 0`
 *
 * @param bitmask Bitmask residing in device memory whose bits will be counted.
 * @param start Index of the first bit to count (inclusive).
 * @param stop Index of the last bit to count (exclusive).
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return The number of non-zero bits in the specified range.
 */
cudf::size_type count_set_bits(bitmask_type const* bitmask,
                               size_type start,
                               size_type stop,
                               cuda::stream_ref stream);

/**
 * @brief Given a bitmask, counts the number of unset (0) bits in the range
 * `[start, stop)`.
 *
 * @throws std::invalid_argument if `bitmask == nullptr`
 * @throws std::invalid_argument if `start > stop`
 * @throws std::invalid_argument if `start < 0`
 *
 * @param bitmask Bitmask residing in device memory whose bits will be counted.
 * @param start Index of the first bit to count (inclusive).
 * @param stop Index of the last bit to count (exclusive).
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return The number of zero bits in the specified range.
 */
cudf::size_type count_unset_bits(bitmask_type const* bitmask,
                                 size_type start,
                                 size_type stop,
                                 cuda::stream_ref stream);

/**
 * @brief Given a bitmask, counts the number of set (1) bits in every range
 * `[indices[2*i], indices[(2*i)+1])` (where 0 <= i < indices.size() / 2).
 *
 * @throws cudf::logic_error if `bitmask == nullptr`
 * @throws cudf::logic_error if `indices.size() % 2 != 0`
 * @throws cudf::logic_error if `indices[2*i] < 0 or indices[2*i] > indices[(2*i)+1]`
 *
 * @param[in] bitmask Bitmask residing in device memory whose bits will be counted.
 * @param[in] indices A host_span of indices specifying ranges to count the number of set bits.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @return A vector storing the number of non-zero bits in the specified ranges.
 */
std::vector<size_type> segmented_count_set_bits(bitmask_type const* bitmask,
                                                host_span<size_type const> indices,
                                                cuda::stream_ref stream);

/**
 * @brief Given a bitmask, counts the number of unset (0) bits in every range
 * `[indices[2*i], indices[(2*i)+1])` (where 0 <= i < indices.size() / 2).
 *
 * @throws cudf::logic_error if `bitmask == nullptr`
 * @throws cudf::logic_error if `indices.size() % 2 != 0`
 * @throws cudf::logic_error if `indices[2*i] < 0 or indices[2*i] > indices[(2*i)+1]`
 *
 * @param[in] bitmask Bitmask residing in device memory whose bits will be counted.
 * @param[in] indices A host_span of indices specifying ranges to count the number of unset bits.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @return A vector storing the number of zero bits in the specified ranges.
 */
std::vector<size_type> segmented_count_unset_bits(bitmask_type const* bitmask,
                                                  host_span<size_type const> indices,
                                                  cuda::stream_ref stream);

/**
 * @brief Given a validity bitmask, counts the number of valid elements (set bits)
 * in the range `[start, stop)`.
 *
 * If `bitmask == nullptr`, all elements are assumed to be valid and the
 * function returns `stop-start`.
 *
 * @throws std::invalid_argument if `start > stop`
 * @throws std::invalid_argument if `start < 0`
 *
 * @param[in] bitmask Validity bitmask residing in device memory.
 * @param[in] start Index of the first bit to count (inclusive).
 * @param[in] stop Index of the last bit to count (exclusive).
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @return The number of valid elements in the specified range.
 */
cudf::size_type valid_count(bitmask_type const* bitmask,
                            size_type start,
                            size_type stop,
                            cuda::stream_ref stream);

/**
 * @copydoc null_count(bitmask_type const* bitmask, size_type start, size_type stop)
 *
 * @param stream Stream view on which to allocate resources and queue execution.
 */
cudf::size_type null_count(bitmask_type const* bitmask,
                           size_type start,
                           size_type stop,
                           cuda::stream_ref stream);

/**
 * @copydoc cudf::segmented_valid_count
 */
std::vector<size_type> segmented_valid_count(bitmask_type const* bitmask,
                                             std::span<size_type const> indices,
                                             cuda::stream_ref stream);

/**
 * @copydoc cudf::segmented_null_count
 */
std::vector<size_type> segmented_null_count(bitmask_type const* bitmask,
                                            std::span<size_type const> indices,
                                            cuda::stream_ref stream);

/**
 * @copydoc cudf::copy_bitmask(bitmask_type const*, size_type, size_type,
 *rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
rmm::device_buffer copy_bitmask(bitmask_type const* mask,
                                size_type begin_bit,
                                size_type end_bit,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::copy_bitmask(column_view const& view, rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
rmm::device_buffer copy_bitmask(column_view const& view,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr);

/**
 * @copydoc bitmask_and(host_span<bitmask_type const* const>, host_span<size_type> const,
 * size_type, rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
std::pair<rmm::device_buffer, size_type> bitmask_and(host_span<bitmask_type const* const> masks,
                                                     host_span<size_type const> masks_begin_bits,
                                                     size_type mask_size_bits,
                                                     cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::bitmask_and
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::pair<rmm::device_buffer, size_type> bitmask_and(table_view const& view,
                                                     cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::segmented_bitmask_and
 */
std::pair<std::vector<std::unique_ptr<rmm::device_buffer>>, std::vector<size_type>>
segmented_bitmask_and(host_span<column_view const> colviews,
                      host_span<size_type const> segment_offsets,
                      cuda::stream_ref stream,
                      rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::segmented_bitmask_and
 */
std::pair<std::vector<std::unique_ptr<rmm::device_buffer>>, std::vector<size_type>>
segmented_bitmask_and(host_span<bitmask_type const* const> masks,
                      host_span<size_type const> segment_offsets,
                      size_type mask_size_bits,
                      cuda::stream_ref stream,
                      rmm::device_async_resource_ref mr);
/**
 * @copydoc cudf::bitmask_or
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::pair<rmm::device_buffer, size_type> bitmask_or(table_view const& view,
                                                    cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr);

/**
 * @brief Performs a bitwise AND of the specified bitmasks,
 *        and writes in place to destination
 *
 * @param dest_mask Destination to which the AND result is written
 * @param masks The list of data pointers of the bitmasks to be ANDed
 * @param masks_begin_bits The bit offsets from which each mask is to be ANDed
 * @param mask_size_bits The number of bits to be ANDed in each mask
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Count of set bits
 */
cudf::size_type inplace_bitmask_and(device_span<bitmask_type> dest_mask,
                                    host_span<bitmask_type const* const> masks,
                                    host_span<size_type const> masks_begin_bits,
                                    size_type mask_size_bits,
                                    cuda::stream_ref stream);

/**
 * @brief Recursively set valid null masks for all children.
 *
 * This function applies all valid null masks to the output column if input column satisfies
 * `nullable() == true` condition
 *
 * @param input input column to check for nullability
 * @param output output column to mirror nullability of input
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
void set_all_valid_null_masks(column_view const& input,
                              column& output,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr);

}  // namespace detail

}  // namespace cudf
