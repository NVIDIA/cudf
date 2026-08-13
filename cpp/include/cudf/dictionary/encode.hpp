/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief Dictionary column encode and decode APIs
 */

namespace CUDF_EXPORT cudf {
namespace dictionary {
/**
 * @addtogroup dictionary_encode
 * @{
 */

/**
 * @brief Construct a dictionary column by dictionary encoding an existing column
 *
 * The output column is a DICTIONARY type with a keys column of non-null, unique values
 * that are in a strict, total order. Meaning, `keys[i]` is _ordered before
 * `keys[i+1]` for all `i in [0,n-1)` where `n` is the number of keys.

 * The output column has a child indices column that is of integer type and with
 * the same size as the input column.
 * The indices column will be of type `indices_type`.
 * The result is undefined if the `indices_type` is not large enough for the indices values.
 *
 * The null mask and null count are copied from the input column to the output column.
 *
 * @throw cudf::data_type_error if indices type is not a signed integer type
 * @throw std::invalid_argument if the column to encode is a DICTIONARY or nested type
 *
 * @code{.pseudo}
 * c = [429, 111, 213, 111, 213, 429, 213]
 * d = encode(c)
 * d now has keys [111, 213, 429] and indices [2, 0, 1, 0, 1, 2, 1]
 * @endcode
 *
 * @param column The column to dictionary encode
 * @param indices_type The integer type to use for the indices
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Returns a dictionary column
 */
std::unique_ptr<column> encode(
  column_view const& column,
  data_type indices_type            = data_type{type_id::INT32},
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create a column by gathering the keys from the provided
 * dictionary_column into a new column using the indices from that column.
 *
 * @code{.pseudo}
 * d1 = {["a", "c", "d"], [2, 0, 1, 0]}
 * s = decode(d1)
 * s is now ["d", "a", "c", "a"]
 * @endcode
 *
 * @param dictionary_column Existing dictionary column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column with type matching the dictionary_column's keys
 */
std::unique_ptr<column> decode(
  dictionary_column_view const& dictionary_column,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace dictionary
}  // namespace CUDF_EXPORT cudf
