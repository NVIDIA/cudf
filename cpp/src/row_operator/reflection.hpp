/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

#include <cstdint>
#include <string>
#include <string_view>

namespace cudf::row_operators {

/** @brief Supported row hash algorithms. */
enum class algorithm : uint8_t {
  xxhash_32,
  xxhash_64,
  murmurhash3_x86_32,
  murmurhash3_x64_128,
  spark_murmurhash3_x86_32
};

/**
 * @brief Returns a hash algorithm's name for generated code and diagnostics.
 *
 * @param alg Hash algorithm
 * @return Algorithm name
 * @throws std::invalid_argument If the algorithm is unsupported
 */
[[nodiscard]] std::string_view to_string(algorithm alg);

/**
 * @brief Reflects the output policy for a row hash algorithm.
 *
 * Supported names are `murmurhash3_x86_32`, `spark_murmurhash3_x86_32`,
 * `murmurhash3_x64_128`, `xxhash_32`, and `xxhash_64`.
 *
 * @param alg Hash algorithm
 * @return Fully qualified `Hash32`, `Hash64`, or `Hash128` type name
 * @throws std::invalid_argument If the algorithm is unsupported
 */
[[nodiscard]] std::string reflect_hash_output(algorithm alg);

/**
 * @brief Reflects a column's typed plain, list, dictionary, or dictionary-list layout.
 *
 * Supports flat values, lists of flat values, and dictionaries with either key layout.
 * Structs and deeper nesting are unsupported.
 *
 * @param column Input column
 * @return Fully qualified layout template instantiation
 * @throws cudf::data_type_error If the layout or value type is unsupported
 */
[[nodiscard]] std::string reflect_layout(column_view const& column);

/**
 * @brief Reflects the ABI element type after unwrapping a column's layout.
 *
 * Decimal and chrono types retain their logical identities. Dictionary index widths and decimal
 * scales are read from the device views and do not specialize the kernel.
 *
 * @param column Input column
 * @return CUDA C++ spelling of the ABI element type
 * @throws cudf::data_type_error If the layout or value type is unsupported
 */
[[nodiscard]] std::string reflect_element(column_view const& column);

/**
 * @brief Reflects the indexed, typed hasher used by the hash kernel.
 *
 * @param alg Hash algorithm
 * @param column Input column
 * @param index Column's position in the device-view array
 * @return Fully qualified hasher template instantiation
 * @throws cudf::data_type_error If the column layout or value type is unsupported
 * @throws std::invalid_argument If the algorithm or index is unsupported
 */
[[nodiscard]] std::string reflect_hasher(algorithm alg, column_view const& column, size_type index);

}  // namespace cudf::row_operators
