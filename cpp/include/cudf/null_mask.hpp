/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/stream>

#include <span>
#include <vector>

/**
 * @file
 * @brief APIs for managing validity bitmasks
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_nullmask
 * @{
 */

/**
 * @brief Returns the null count for a null mask of the specified `state`
 * representing `size` elements
 *
 * @throw std::invalid_argument if state is UNINITIALIZED
 *
 * @param state The state of the null mask
 * @param size The number of elements represented by the mask
 * @return The count of null elements
 */
size_type state_null_count(mask_state state, size_type size);

/**
 * @brief Computes the required bytes necessary to represent the specified
 * number of bits with a given padding boundary
 *
 * @note The Arrow specification for the null bitmask requires a 64B padding
 * boundary.
 *
 * @param number_of_bits The number of bits that need to be represented
 * @param padding_boundary The value returned will be rounded up to a multiple
 * of this value
 * @return The necessary number of bytes
 */
std::size_t bitmask_allocation_size_bytes(size_type number_of_bits,
                                          std::size_t padding_boundary = 64);

/**
 * @brief Returns the number of `bitmask_type` words required to represent the
 * specified number of bits
 *
 * Unlike `bitmask_allocation_size_bytes`, which returns the number of *bytes*
 * needed for a bitmask allocation (including padding), this function returns
 * the *actual* number `bitmask_type` elements necessary to represent
 * `number_of_bits`. This is useful when one wishes to process all of the bits
 * in a bitmask and ignore the padding/slack bits.
 *
 * @param number_of_bits The number of bits that need to be represented
 * @return The necessary number of `bitmask_type` elements
 */
size_type num_bitmask_words(size_type number_of_bits);

/**
 * @brief Creates a `device_buffer` for use as a null value indicator bitmask of
 * a `column`
 *
 * @param size The number of elements to be represented by the mask
 * @param state The desired state of the mask
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device_buffer
 * @return A `device_buffer` for use as a null bitmask
 * satisfying the desired size and state
 */
rmm::device_buffer create_null_mask(
  size_type size,
  mask_state state,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Sets a pre-allocated bitmask buffer to a given state in the range
 * `[begin_bit, end_bit)`
 *
 * Sets `[begin_bit, end_bit)` bits of bitmask to valid if `valid==true`
 * or null otherwise.
 *
 * @param bitmask Pointer to bitmask (e.g. returned by `column_view::null_mask()`)
 * @param begin_bit Index of the first bit to set (inclusive)
 * @param end_bit Index of the last bit to set (exclusive)
 * @param valid If true set all entries to valid; otherwise, set all to null
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void set_null_mask(bitmask_type* bitmask,
                   size_type begin_bit,
                   size_type end_bit,
                   bool valid,
                   cuda::stream_ref stream = cudf::get_default_stream());

/**
 * @brief Sets a vector of non-overlapping pre-allocated bitmask buffers to given states in the
 * corresponding ranges in bulk
 *
 * Sets bit ranges `[begin_bit, end_bit)` of given bitmasks to specified valid states. The bitmask
 * bit ranges must be non-overlapping. i.e., attempting to set a physical bit concurrently across
 * bitmasks will result in undefined behavior. This utility is optimized for bulk operation on 16
 * or more bitmasks sized 2^24 bits or less.
 *
 * @param bitmasks Pointers to bitmasks (e.g. returned by `column_view::null_mask()`)
 * @param begin_bits Indices of the first bits to set (inclusive)
 * @param end_bits Indices of the last bits to set (exclusive)
 * @param valids Booleans indicating if the corresponding bitmasks should be set to valid or null
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void set_null_masks_safe(cudf::host_span<bitmask_type*> bitmasks,
                         cudf::host_span<size_type const> begin_bits,
                         cudf::host_span<size_type const> end_bits,
                         cudf::host_span<bool const> valids,
                         cuda::stream_ref stream = cudf::get_default_stream());

/**
 * @brief Sets a vector of non-overlapping pre-allocated bitmask buffers to given states in the
 * corresponding non-aliasing ranges in bulk
 *
 * Sets bit ranges `[begin_bit, end_bit)` of given bitmasks to specified valid states. The bitmask
 * bit ranges must be non-overlapping and non-aliasing. i.e., attempting to concurrently set bits
 * within the same physical word across bitmasks will result in undefined behavior. This utility is
 * optimized for bulk operation on 16 or more bitmasks sized 2^24 bits or less.
 *
 * @param bitmasks Pointers to bitmasks (e.g. returned by `column_view::null_mask()`)
 * @param begin_bits Indices of the first bits to set (inclusive)
 * @param end_bits Indices of the last bits to set (exclusive)
 * @param valids Booleans indicating if the corresponding bitmasks should be set to valid or null
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void set_null_masks_unsafe(cudf::host_span<bitmask_type*> bitmasks,
                           cudf::host_span<size_type const> begin_bits,
                           cudf::host_span<size_type const> end_bits,
                           cudf::host_span<bool const> valids,
                           cuda::stream_ref stream = cudf::get_default_stream());

/**
 * @brief Creates a `device_buffer` from a slice of bitmask defined by a range
 * of indices `[begin_bit, end_bit)`
 *
 * Returns empty `device_buffer` if `bitmask == nullptr`.
 *
 * @throws cudf::logic_error if `begin_bit > end_bit`
 * @throws cudf::logic_error if `begin_bit < 0`
 *
 * @param mask Bitmask residing in device memory whose bits will be copied
 * @param begin_bit Index of the first bit to be copied (inclusive)
 * @param end_bit Index of the last bit to be copied (exclusive)
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device_buffer
 * @return A `device_buffer` containing the bits
 * `[begin_bit, end_bit)` from `mask`.
 */
rmm::device_buffer copy_bitmask(
  bitmask_type const* mask,
  size_type begin_bit,
  size_type end_bit,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Copies `view`'s bitmask from the bits
 * `[view.offset(), view.offset() + view.size())` into a `device_buffer`
 *
 * Returns empty `device_buffer` if the column is not nullable
 *
 * @param view Column view whose bitmask needs to be copied
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device_buffer
 * @return A `device_buffer` containing the bits
 * `[view.offset(), view.offset() + view.size())` from `view`'s bitmask.
 */
rmm::device_buffer copy_bitmask(
  column_view const& view,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Performs bitwise AND of the bitmasks of columns of a table. Returns
 * a pair of resulting mask and count of unset bits
 *
 * If any of the columns isn't nullable, it is considered all valid.
 * If no column in the table is nullable, an empty bitmask is returned.
 *
 * @param view The table of columns
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device_buffer
 * @return A pair of resulting bitmask and count of unset bits
 */
std::pair<rmm::device_buffer, size_type> bitmask_and(
  table_view const& view,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Performs bitwise AND of the bitmasks provided
 *
 * Each row bit is true only if the corresponding bits in each bitmask is true.
 *
 * @param masks Each mask to perform the bitwise compare
 * @param begin_bits Offsets to the first bit of each item in masks
 * @param mask_size The number of bits to process in each mask
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device_buffer
 * @return A pair of resulting bitmask of size mask_size and the count of unset bits
 */
std::pair<rmm::device_buffer, size_type> bitmask_and(
  host_span<bitmask_type const* const> masks,
  host_span<size_type const> begin_bits,
  size_type mask_size,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Performs segmented bitwise AND operations on the null masks of the input columns based
 * on defined segments
 *
 * For each segment, it computes the bitwise AND of the bitmasks of all
 * columns within that segment. Returns a pair containing (i) a vector of unique pointers to
 * device buffers, with each buffer containing the resulting bitmask for a segment, and (ii) a
 * vector of integers representing the count of null (unset) bits for each segment
 *
 * The function assumes all the input columns passed are nullable.
 *
 * @param colviews A span containing column views whose bitmasks will be ANDed within their
 * respective segments
 * @param segment_offsets A span containing the starting positions of each segment
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device_buffer
 * @return A pair of vectors containing resulting bitmask and count of unset bits for each segment
 */
std::pair<std::vector<std::unique_ptr<rmm::device_buffer>>, std::vector<size_type>>
segmented_bitmask_and(host_span<column_view const> colviews,
                      host_span<size_type const> segment_offsets,
                      cuda::stream_ref stream           = cudf::get_default_stream(),
                      rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Performs segmented bitwise AND operations on the null masks on defined segments
 *
 * For each segment, it computes the bitwise AND of the bitmasks of all
 * masks with that segment.
 *
 * Returns a pair containing (i) a vector of unique pointers to
 * device buffers, with each buffer containing the resulting bitmask for a segment, and (ii) a
 * vector of integers representing the count of null (unset) bits for each segment
 *
 * @param masks A span containing bitmasks that will be ANDed within their
 * respective segments
 * @param segment_offsets A span containing the starting positions of each segment
 * @param mask_size_bits Number of bits in each mask
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device_buffer
 * @return A pair of vectors containing resulting bitmask and count of unset bits for each segment
 */
std::pair<std::vector<std::unique_ptr<rmm::device_buffer>>, std::vector<size_type>>
segmented_bitmask_and(host_span<bitmask_type const* const> masks,
                      host_span<size_type const> segment_offsets,
                      size_type mask_size_bits,
                      cuda::stream_ref stream           = cudf::get_default_stream(),
                      rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Performs bitwise OR of the bitmasks of columns of a table. Returns
 * a pair of resulting mask and count of unset bits
 *
 * If any of the columns isn't nullable, it is considered all valid.
 * If no column in the table is nullable, an empty bitmask is returned.
 *
 * @param view The table of columns
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device_buffer
 * @return A pair of resulting bitmask and count of unset bits
 */
std::pair<rmm::device_buffer, size_type> bitmask_or(
  table_view const& view,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Given a validity bitmask, counts the number of null elements (unset bits)
 * in the range `[start, stop)`.
 *
 * If `bitmask == nullptr`, all elements are assumed to be valid and the
 * function returns `0`.
 *
 * @throws std::invalid_argument if `start > stop`
 * @throws std::invalid_argument if `start < 0`
 *
 * @param bitmask Validity bitmask residing in device memory
 * @param start Index of the first bit to count (inclusive)
 * @param stop Index of the last bit to count (exclusive)
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return The number of null elements in the specified range
 */
size_type null_count(bitmask_type const* bitmask,
                     size_type start,
                     size_type stop,
                     cuda::stream_ref stream = cudf::get_default_stream());

/**
 * @brief Given a list of validity bitmasks, counts the number of null elements (unset bits) in the
 * range `[start, stop)` for each bitmask.
 *
 * The same bit range `[start, stop)` is used for all bitmasks.
 * If a bitmask pointer is `nullptr`, all elements corresponding to that bitmask are assumed to be
 * valid and the null count is `0`.
 *
 * @throws std::invalid_argument if `start > stop`
 * @throws std::invalid_argument if `start < 0`
 *
 * @param bitmasks Validity bitmasks residing in device memory
 * @param start Index of the first bit to count (inclusive)
 * @param stop Index of the last bit to count (exclusive)
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A vector of null counts for each bitmask
 */
std::vector<size_type> batch_null_count(host_span<bitmask_type const* const> bitmasks,
                                        size_type start,
                                        size_type stop,
                                        cuda::stream_ref stream = cudf::get_default_stream());

/**
 * @brief Given a validity bitmask, counts the number of valid elements (set
 * bits) in every range `[indices[2*i], indices[(2*i)+1])` (where 0 <= i <
 * indices.size() / 2).
 *
 * If `bitmask == nullptr`, all elements are assumed to be valid and a vector of
 * length `indices.size()` containing segment lengths is returned.
 *
 * @throws cudf::logic_error if `indices.size() % 2 != 0`.
 * @throws cudf::logic_error if `indices[2*i] < 0 or indices[2*i] > indices[(2*i)+1]`.
 *
 * @param[in] bitmask Validity bitmask residing in device memory.
 * @param[in] indices A span of indices specifying ranges to count the number of valid
 * elements.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @return A vector storing the number of valid elements in each specified range.
 */
std::vector<size_type> segmented_valid_count(bitmask_type const* bitmask,
                                             std::span<size_type const> indices,
                                             cuda::stream_ref stream = cudf::get_default_stream());

/**
 * @brief Given a validity bitmask, counts the number of null elements (unset
 * bits) in every range `[indices[2*i], indices[(2*i)+1])` (where 0 <= i <
 * indices.size() / 2).
 *
 * If `bitmask == nullptr`, all elements are assumed to be valid and a vector of
 * length `indices.size()` containing all zeros is returned.
 *
 * @throws cudf::logic_error if `indices.size() % 2 != 0`
 * @throws cudf::logic_error if `indices[2*i] < 0 or indices[2*i] > indices[(2*i)+1]`
 *
 * @param[in] bitmask Validity bitmask residing in device memory.
 * @param[in] indices A span of indices specifying ranges to count the number of null elements.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @return A vector storing the number of null elements in each specified range.
 */
std::vector<size_type> segmented_null_count(bitmask_type const* bitmask,
                                            std::span<size_type const> indices,
                                            cuda::stream_ref stream = cudf::get_default_stream());

/**
 * @brief Given a validity bitmask, returns the index of the first set bit
 * in the range `[start, stop)` relative to start
 *
 * @param bitmask Validity bitmask residing in device memory
 * @param start Index of the first bit to check (inclusive)
 * @param stop Index of the last bit to check (exclusive)
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return The index of the first set bit in the specified range relative to start,
 *         or `stop-start` if no set bit is found (all nulls)
 */
size_type index_of_first_set_bit(bitmask_type const* bitmask,
                                 size_type start,
                                 size_type stop,
                                 cuda::stream_ref stream = cudf::get_default_stream());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
