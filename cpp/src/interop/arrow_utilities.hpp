/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <nanoarrow/nanoarrow.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cudf {
namespace detail {

/**
 * @brief constants for buffer indexes of Arrow arrays
 *
 */
static constexpr int validity_buffer_idx         = 0;
static constexpr int fixed_width_data_buffer_idx = 1;

/**
 * @brief Map ArrowType id to cudf column type id
 *
 * @param arrow_view SchemaView to pull the logical and storage types from
 * @return Column type id
 */
data_type arrow_to_cudf_type(ArrowSchemaView const* arrow_view);

/**
 * @brief Check whether the given schema view describes an Arrow fixed-size-list
 *
 * @param arrow_view SchemaView to check
 * @return True if the schema describes a fixed-size-list
 */
bool is_fixed_size_list(ArrowSchemaView const* arrow_view);

/**
 * @brief Validated physical bounds for an Arrow fixed-size-list array
 */
struct fixed_size_list_layout {
  int32_t width;         ///< Child elements per row and LIST offset increment
  size_type num_rows;    ///< Number of output rows
  int64_t row_offset;    ///< First logical row in the Arrow array
  int64_t row_end;       ///< One-past-last logical row
  int64_t child_offset;  ///< First referenced child element
  int64_t child_length;  ///< Number of referenced child elements
  int64_t child_end;     ///< One-past-last referenced child element
};

/**
 * @brief Return the number of child elements per row of a fixed-size-list schema
 *
 * @throw cudf::data_type_error if `arrow_view` is not a fixed-size-list
 * @throw std::invalid_argument if the declared width is negative
 * @throw std::overflow_error if the declared width exceeds the INT32 LIST offset range
 *
 * @param arrow_view SchemaView to pull the fixed size from
 * @return Number of child elements per row
 */
int32_t fixed_size_list_width(ArrowSchemaView const* arrow_view);

/**
 * @brief Validate and compute fixed-size-list row and child bounds
 *
 * @throw std::invalid_argument if row metadata is negative
 * @throw std::overflow_error if Arrow bounds overflow `int64_t`, output row counts exceed
 * `size_type`, or synthesized LIST offsets exceed INT32
 *
 * @param arrow_view Fixed-size-list schema view
 * @param input Arrow array carrying row offset and length
 * @return Validated source bounds and output sizes
 */
fixed_size_list_layout get_fixed_size_list_layout(ArrowSchemaView const* arrow_view,
                                                  ArrowArray const* input);

/**
 * @brief Map cudf column type id to ArrowType id
 *
 * @param id Column type id
 * @return ArrowType id
 */
ArrowType id_to_arrow_type(cudf::type_id id);

/**
 * @brief Map cudf column type id to the storage type for Arrow
 *
 * Specifically this is for handling the underlying storage type of
 * timestamps and durations.
 *
 * @param id column type id
 * @return ArrowType storage type
 */
ArrowType id_to_arrow_storage_type(cudf::type_id id);

/**
 * @brief cudf types used to import the indices of an Arrow dictionary array
 */
struct dictionary_indices_types {
  data_type storage;  ///< Type used to read the Arrow indices buffer
  data_type output;   ///< Signed type of the resulting cudf dictionary indices
};

/**
 * @brief Return the cudf types used to import Arrow dictionary indices
 *
 * Signed index types are returned unchanged. Unsigned index types are mapped to the
 * narrowest signed type that can represent every valid index `[0, num_keys - 1]`:
 * the same-width signed type when it fits, otherwise the next wider signed type.
 *
 * @throw cudf::data_type_error if `storage_type` is not an 8, 16, or 32-bit unsigned or
 * an 8, 16, 32, or 64-bit signed integer type
 *
 * @param storage_type Arrow storage type of the dictionary indices
 * @param num_keys Number of keys in the dictionary, at most `size_type` max
 * @return Types to read the Arrow buffer with and to output
 */
dictionary_indices_types get_dictionary_indices_types(ArrowType storage_type, int64_t num_keys);

/**
 * @brief Helper to initialize ArrowArray struct
 *
 * @param arr Pointer to ArrowArray to initialize
 * @param storage_type The type to initialize with
 * @param column view for column to get the length and null count from
 * @return nanoarrow status code, should be NANOARROW_OK if there are no errors
 */
int initialize_array(ArrowArray* arr, ArrowType storage_type, cudf::column_view column);

/**
 * @brief Return the number of rows of an Arrow array as a cudf row count
 *
 * @throw std::overflow_error if the column, or its offsets child when `has_offsets`, would have
 * more rows than `size_type` can hold
 *
 * @param num_rows Number of rows of the Arrow array
 * @param has_offsets Whether the cudf column has an offsets child of `num_rows + 1` rows
 * @return `num_rows` as a `size_type`
 */
size_type checked_row_count(int64_t num_rows, bool has_offsets = false);

/**
 * @brief Copy the offsets of a string or list Arrow array, rebased to start at zero
 *
 * Writes `num_rows + 1` offsets, from row `first_row` of `offsets`, to `out`.
 *
 * @throw std::overflow_error if a rebased offset does not fit in `Out`
 *
 * @param offsets Arrow offsets buffer
 * @param first_row First row to copy, including the array's own offset
 * @param num_rows Number of rows
 * @param out Output for `num_rows + 1` offsets
 */
template <typename In, typename Out>
void rebase_offsets(In const* offsets, int64_t first_row, int64_t num_rows, Out* out)
{
  auto const first = offsets[first_row];
  CUDF_EXPECTS(static_cast<int64_t>(offsets[first_row + num_rows] - first) <=
                 static_cast<int64_t>(std::numeric_limits<Out>::max()),
               "offsets exceed the bounds of the cudf offsets type",
               std::overflow_error);
  for (int64_t i = 0; i <= num_rows; ++i) {
    out[i] = static_cast<Out>(offsets[first_row + i] - first);
  }
}

}  // namespace detail
}  // namespace cudf
