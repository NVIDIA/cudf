/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <nanoarrow/nanoarrow.h>

#include <cstdint>

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
 * @brief Helper to initialize ArrowArray struct
 *
 * @param arr Pointer to ArrowArray to initialize
 * @param storage_type The type to initialize with
 * @param column view for column to get the length and null count from
 * @return nanoarrow status code, should be NANOARROW_OK if there are no errors
 */
int initialize_array(ArrowArray* arr, ArrowType storage_type, cudf::column_view column);

}  // namespace detail
}  // namespace cudf
