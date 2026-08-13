/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <initializer_list>
#include <span>

namespace CUDF_EXPORT cudf {
namespace detail {
/**
 * @brief Constructs a zero-copy `column_view`/`mutable_column_view` of the
 * elements in the range `[begin,end)` in `input`.
 *
 * @note It is the caller's responsibility to ensure that the returned view
 * does not outlive the viewed device memory.
 *
 * @throws cudf::logic_error if `begin < 0`, `end < begin` or
 * `end > input.size()`.
 *
 * @tparam ColumnView Must be either cudf::column_view or cudf::mutable_column_view
 * @param input View of input column to slice
 * @param begin Index of the first desired element in the slice (inclusive).
 * @param end Index of the last desired element in the slice (exclusive).
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return ColumnView View of the elements `[begin,end)` from `input`.
 */
template <typename ColumnView>
ColumnView slice(ColumnView const& input,
                 size_type begin,
                 size_type end,
                 rmm::cuda_stream_view stream);

/**
 * @copydoc cudf::slice(column_view const&, std::span<size_type const>)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::vector<column_view> slice(column_view const& input,
                               std::span<size_type const> indices,
                               rmm::cuda_stream_view stream);
/**
 * @copydoc cudf::slice(column_view const&, std::initializer_list<size_type>)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::vector<column_view> slice(column_view const& input,
                               std::initializer_list<size_type> indices,
                               rmm::cuda_stream_view stream);

/**
 * @copydoc cudf::slice(table_view const&, std::span<size_type const>)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::vector<table_view> slice(table_view const& input,
                              std::span<size_type const> indices,
                              rmm::cuda_stream_view stream);
/**
 * @copydoc cudf::slice(table_view const&, std::initializer_list<size_type>)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::vector<table_view> slice(table_view const& input,
                              std::initializer_list<size_type> indices,
                              rmm::cuda_stream_view stream);

/**
 * @copydoc cudf::split(column_view const&, std::span<size_type const>)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::vector<column_view> split(column_view const& input,
                               std::span<size_type const> splits,
                               rmm::cuda_stream_view stream);
/**
 * @copydoc cudf::split(column_view const&, std::initializer_list<size_type>)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::vector<column_view> split(column_view const& input,
                               std::initializer_list<size_type> splits,
                               rmm::cuda_stream_view stream);

/**
 * @copydoc cudf::split(table_view const&, std::span<size_type const>)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::vector<table_view> split(table_view const& input,
                              std::span<size_type const> splits,
                              rmm::cuda_stream_view stream);
/**
 * @copydoc cudf::split(table_view const&, std::initializer_list<size_type>)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::vector<table_view> split(table_view const& input,
                              std::initializer_list<size_type> splits,
                              rmm::cuda_stream_view stream);

/**
 * @copydoc cudf::shift(column_view const&,size_type,scalar const&,
 * rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> shift(column_view const& input,
                              size_type offset,
                              scalar const& fill_value,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr);

/**
 * @brief Performs segmented shifts for specified values.
 *
 * For each segment, `i`th element is determined by the `i - offset`th element
 * of the segment. If `i - offset < 0 or >= segment_size`, the value is determined by
 * @p fill_value.
 *
 * Example:
 * @code{.pseudo}
 * segmented_values: { 3 1 2 | 3 5 3 | 2 6 }
 * segment_offsets: {0 3 6 8}
 * offset: 2
 * fill_value: @
 * result: { @ @ 3 | @ @ 3 | @ @ }
 * -------------------------------------------------
 * segmented_values: { 3 1 2 | 3 5 3 | 2 6 }
 * segment_offsets: {0 3 6 8}
 * offset: -1
 * fill_value: -1
 * result: { 1 2 -1 | 5 3 -1 | 6 -1 }
 * @endcode
 *
 * @param segmented_values Segmented column, specified by @p segment_offsets
 * @param segment_offsets Each segment's offset of @p segmented_values. A list of offsets
 * with size `num_segments + 1`. The size of each segment is `segment_offsets[i+1] -
 * segment_offsets[i]`.
 * @param offset The offset by which to shift the input
 * @param fill_value Fill value for indeterminable outputs
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @note If `offset == 0`, a copy of @p segmented_values is returned.
 */
std::unique_ptr<column> segmented_shift(column_view const& segmented_values,
                                        device_span<size_type const> segment_offsets,
                                        size_type offset,
                                        scalar const& fill_value,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::allocate_like(column_view const&, size_type, mask_allocation_policy,
 * rmm::device_async_resource_ref)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> allocate_like(column_view const& input,
                                      size_type size,
                                      mask_allocation_policy mask_alloc,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::copy_if_else( column_view const&, column_view const&,
 * column_view const&, rmm::device_async_resource_ref)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> copy_if_else(column_view const& lhs,
                                     column_view const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::copy_if_else( scalar const&, column_view const&,
 * column_view const&, rmm::device_async_resource_ref)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> copy_if_else(scalar const& lhs,
                                     column_view const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::copy_if_else( column_view const&, scalar const&,
 * column_view const&, rmm::device_async_resource_ref)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> copy_if_else(column_view const& lhs,
                                     scalar const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::copy_if_else( scalar const&, scalar const&,
 * column_view const&, rmm::device_async_resource_ref)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> copy_if_else(scalar const& lhs,
                                     scalar const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::sample
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> sample(table_view const& input,
                              size_type const n,
                              sample_with_replacement replacement,
                              int64_t const seed,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::get_element
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<scalar> get_element(column_view const& input,
                                    size_type index,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::has_nonempty_nulls
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
bool has_nonempty_nulls(column_view const& input, rmm::cuda_stream_view stream);

/**
 * @copydoc cudf::may_have_nonempty_nulls
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
bool may_have_nonempty_nulls(column_view const& input, rmm::cuda_stream_view stream);

/**
 * @copydoc cudf::purge_nonempty_nulls
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> purge_nonempty_nulls(column_view const& input,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
