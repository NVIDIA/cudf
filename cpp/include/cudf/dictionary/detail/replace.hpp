/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream_ref>

namespace cudf {
namespace dictionary::detail {

/**
 * @brief Create a new dictionary column by replacing nulls with values
 * from a second dictionary.
 *
 * @throw cudf::logic_error if the keys type of both dictionaries do not match.
 * @throw cudf::logic_error if the column sizes do not match.
 *
 * @param input Column with nulls to replace.
 * @param replacement Column with values to use for replacing.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New dictionary column with null rows replaced.
 */
std::unique_ptr<column> replace_nulls(dictionary_column_view const& input,
                                      dictionary_column_view const& replacement,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @brief Create a new dictionary column by replacing nulls with a
 * specified scalar.
 *
 * @throw cudf::logic_error if the keys type does not match the replacement type.
 *
 * @param input Column with nulls to replace.
 * @param replacement Value to use for replacing.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New dictionary column with null rows replaced.
 */
std::unique_ptr<column> replace_nulls(dictionary_column_view const& input,
                                      scalar const& replacement,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr);

}  // namespace dictionary::detail
}  // namespace cudf
