// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <memory>

/**
 * @brief Output construction for identity and all-null partition paths
 */
namespace cudf::detail::partition_identity {
/**
 * @brief Determine whether a column is fixed-width or STRING and contains only nulls
 *
 * Recognizes all-null fixed-width or STRING columns using host metadata only. Columns containing
 * valid values, including empty strings, are not eligible.
 *
 * @param input Column whose type, size and null count are inspected
 * @return Whether the column can be reconstructed by `create_all_null_column` without reading its
 * values
 */
inline bool is_flat_and_all_null(column_view const& input)
{
  bool const flat = is_fixed_width(input.type()) || input.type().id() == type_id::STRING;
  return flat && input.null_count() == input.size();
}

/**
 * @brief Construct an all-null partition output without copying input values
 *
 * Creates an all-null fixed-width or STRING column. STRING offsets are initialized to zero and
 * the character buffer is empty. Row permutation leaves these logical values unchanged.
 *
 * @pre `is_flat_and_all_null(input)` is true
 *
 * @param input Supported all-null column whose logical type, size and null count are preserved
 * @param stream CUDA stream used for device operations and memory allocation
 * @param mr Device memory resource used to allocate the output buffers
 * @return A newly allocated column with the same logical values as the input
 */
inline std::unique_ptr<column> create_all_null_column(column_view const& input,
                                                      cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr)
{
  auto const rows = input.size();
  if (input.type().id() != type_id::STRING)
    return make_fixed_width_column(input.type(), rows, mask_state::ALL_NULL, stream, mr);
  auto offsets = cudf::strings::detail::create_offsets_child_column(0, rows + 1, stream, mr);
  CUDF_CUDA_TRY(
    cudaMemsetAsync(offsets->mutable_view().head(),
                    0,
                    static_cast<std::size_t>(offsets->size()) * size_of(offsets->type()),
                    stream.get()));
  return make_strings_column(
    rows,
    std::move(offsets),
    rmm::device_buffer(0, stream, mr),
    input.null_count(),
    cudf::detail::create_null_mask(
      rows, input.null_count() ? mask_state::ALL_NULL : mask_state::UNALLOCATED, stream, mr));
}

/**
 * @brief Copy a table for a partition path that preserves the input row order
 *
 * Uses the standard table-copy constructor to produce independent output storage, including
 * nested columns, while preserving the input schema and logical row contents.
 *
 * @param input Table to copy
 * @param stream CUDA stream used for device operations and memory allocation
 * @param mr Device memory resource used to allocate the output buffers
 * @return An owning deep copy of the input table
 */
inline std::unique_ptr<table> copy(table_view const& input,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  return std::make_unique<table>(input, stream, mr);
}
}  // namespace cudf::detail::partition_identity
