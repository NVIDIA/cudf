/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/stream>

namespace cudf {
namespace detail {

/**
 * @brief Return a fixed-width value from column.
 *
 * Retrieves the specified value from device memory. This function
 * synchronizes the stream.
 *
 * @throw cudf::logic_error if `col_view` is not a fixed-width column
 * @throw cudf::logic_error if `element_index < 0 or >= col_view.size()`
 *
 * @tparam T Fixed-width type to return.
 * @param col_view The column to retrieve the element from.
 * @param element_index The specific element to retrieve
 * @param stream The stream to use for copying the value to the host.
 * @return Value from the `col_view[element_index]`
 */
template <typename T>
T get_value(column_view const& col_view, size_type element_index, cuda::stream_ref stream)
{
  CUDF_EXPECTS(cudf::is_fixed_width(col_view.type()), "get_value supports only fixed-width types");
  CUDF_EXPECTS(data_type(type_to_id<T>()) == col_view.type(), "get_value data type mismatch");
  CUDF_EXPECTS(element_index >= 0 && element_index < col_view.size(),
               "invalid element_index value");
  return cudf::detail::make_host_vector(device_span<T const>{col_view.data<T>() + element_index, 1},
                                        stream)
    .front();
}

}  // namespace detail
}  // namespace cudf
