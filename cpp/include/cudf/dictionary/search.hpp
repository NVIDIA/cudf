/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief APIs for searching for a key value within a dictionary column's keys.
 */

namespace CUDF_EXPORT cudf {
namespace dictionary {
/**
 * @addtogroup dictionary_search
 * @{
 */

/**
 * @brief Return the index value for a given key
 *
 * If the key does not exist in the dictionary the returned scalar will have `is_valid()==false`.
 *
 * @throw std::invalid_argument if `key.type() != dictionary.keys().type()` or dictionary is empty.
 *
 * @param dictionary The dictionary to search for the key
 * @param key The value to search for in the dictionary keyset
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Numeric scalar index value of the key within the dictionary
 */
std::unique_ptr<scalar> get_index(
  dictionary_column_view const& dictionary,
  scalar const& key,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace dictionary
}  // namespace CUDF_EXPORT cudf
