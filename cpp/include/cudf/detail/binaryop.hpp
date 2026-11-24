/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/binaryop.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <optional>

namespace cudf {
//! Inner interfaces and implementations
namespace detail {

/**
 * @copydoc cudf::binary_operation(column_view const&, column_view const&,
 * std::string const&, data_type, cuda::stream_ref, cudf::memory_resources)
 */
std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         column_view const& rhs,
                                         std::string const& ptx,
                                         data_type output_type,
                                         cuda::stream_ref stream,
                                         cudf::memory_resources mr);

/**
 * @copydoc cudf::binary_operation(scalar const&, column_view const&, binary_operator,
 * data_type, cuda::stream_ref, cudf::memory_resources)
 */
std::unique_ptr<column> binary_operation(scalar const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         cuda::stream_ref stream,
                                         cudf::memory_resources mr);

/**
 * @copydoc cudf::binary_operation(column_view const&, scalar const&, binary_operator,
 * data_type, cuda::stream_ref, cudf::memory_resources)
 */
std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         scalar const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         cuda::stream_ref stream,
                                         cudf::memory_resources mr);

/**
 * @copydoc cudf::binary_operation(column_view const&, column_view const&,
 * binary_operator, data_type, cuda::stream_ref, cudf::memory_resources)
 */
std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         cuda::stream_ref stream,
                                         cudf::memory_resources mr);
}  // namespace detail

namespace binops::compiled::detail {

/**
 * @brief Selects the common type used by compiled binary operations.
 *
 * @param out Requested output type
 * @param lhs Left input type
 * @param rhs Right input type
 * @return Common type of @p out, @p lhs and @p rhs, falling back to the common
 * input type, or std::nullopt if neither exists
 */
CUDF_EXPORT std::optional<data_type> get_common_type(data_type out, data_type lhs, data_type rhs);

}  // namespace binops::compiled::detail
}  // namespace cudf
