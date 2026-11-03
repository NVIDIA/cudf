/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace dictionary::detail {
/**
 * @brief Construct a dictionary column by dictionary encoding an existing column.
 *
 * The output column is a DICTIONARY type with a keys column of non-null, unique values
 * that are in a strict, total order. Meaning, `keys[i]` is ordered before
 * `keys[i+1]` for all `i in [0,n-1)` where `n` is the number of keys.

 * The output column has a child indices column that is of integer type and with
 * the same size as the input column.
 *
 * The null_mask and null count are copied from the input column to the output column.
 *
 * @throw cudf::logic_error if indices_type is not INT32
 *
 * ```
 * c = [429,111,213,111,213,429,213]
 * d = make_dictionary_column(c)
 * d now has keys [111,213,429] and indices [2,0,1,0,1,2,1]
 * ```
 *
 * @param column The column to dictionary encode.
 * @param indices_type The integer type to use for the indices.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Memory resources used for temporary allocations and the returned column.
 * @return Returns a dictionary column.
 */
std::unique_ptr<column> encode(column_view const& column,
                               data_type indices_type,
                               cuda::stream_ref stream,
                               memory_resources mr);

/**
 * @brief Create a column by gathering the keys from the provided
 * dictionary_column into a new column using the indices from that column.
 *
 * ```
 * d1 = {["a","c","d"],[2,0,1,0]}
 * s = decode(d1)
 * s is now ["d","a","c","a"]
 * ```
 *
 * @param dictionary_column Existing dictionary column.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Memory resources used for temporary allocations and the returned column.
 * @return New column with type matching the dictionary_column's keys.
 */
std::unique_ptr<column> decode(dictionary_column_view const& dictionary_column,
                               cuda::stream_ref stream,
                               memory_resources mr);

/**
 * @brief Return minimal integer type for the given number of elements.
 *
 * @param keys_size Number of elements in the keys
 * @return Minimal type that can hold `keys_size` values
 */
data_type get_indices_type_for_size(size_type keys_size);

}  // namespace dictionary::detail
}  // namespace cudf
