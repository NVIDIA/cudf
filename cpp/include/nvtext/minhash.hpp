/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/hashing.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

namespace CUDF_EXPORT nvtext {
/**
 * @addtogroup nvtext_minhash
 * @{
 * @file
 * @brief APIs for computing MinHash values of strings columns.
 */

/**
 * @brief Returns the minhash values for each string
 *
 * This function uses MurmurHash3_x86_32 for the hash algorithm.
 *
 * The input strings are first hashed using the given `seed` over substrings
 * of `width` characters. These hash values are then combined with the `a`
 * and `b` parameter values using the following formula:
 * ```
 *   max_hash = max of uint32
 *   mp = (1 << 61) - 1
 *   hv[i] = hash value of a substring at i
 *   pv[i] = ((hv[i] * a[i] + b[i]) % mp) & max_hash
 * ```
 *
 * This calculation is performed on each substring and the minimum value is computed
 * as follows:
 * ```
 *   mh[j,i] = min(pv[i]) for all substrings in row j
 *                        and where i=[0,a.size())
 * ```
 *
 * Any null row entries result in corresponding null output rows.
 *
 * @throw std::invalid_argument if the width < 2
 * @throw std::invalid_argument if parameter_a is empty
 * @throw std::invalid_argument if `parameter_b.size() != parameter_a.size()`
 * @throw std::overflow_error if `parameter_a.size() * input.size()` exceeds the column size limit
 *
 * @param input Strings column to compute minhash
 * @param seed Seed value used for the hash algorithm
 * @param parameter_a Values used for the permuted calculation
 * @param parameter_b Values used for the permuted calculation
 * @param width The character width of substrings to hash for each row
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return List column of minhash values for each string per seed
 */
std::unique_ptr<cudf::column> minhash(
  cudf::strings_column_view const& input,
  uint32_t seed,
  cudf::device_span<uint32_t const> parameter_a,
  cudf::device_span<uint32_t const> parameter_b,
  cudf::size_type width,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the minhash values for each string
 *
 * This function uses MurmurHash3_x64_128 for the hash algorithm.
 *
 * The input strings are first hashed using the given `seed` over substrings
 * of `width` characters. These hash values are then combined with the `a`
 * and `b` parameter values using the following formula:
 * ```
 *   max_hash = max of uint64
 *   mp = (1 << 61) - 1
 *   hv[i] = hash value of a substring at i
 *   pv[i] = ((hv[i] * a[i] + b[i]) % mp) & max_hash
 * ```
 *
 * This calculation is performed on each substring and the minimum value is computed
 * as follows:
 * ```
 *   mh[j,i] = min(pv[i]) for all substrings in row j
 *                        and where i=[0,a.size())
 * ```
 *
 * Any null row entries result in corresponding null output rows.
 *
 * @throw std::invalid_argument if the width < 2
 * @throw std::invalid_argument if parameter_a is empty
 * @throw std::invalid_argument if `parameter_b.size() != parameter_a.size()`
 * @throw std::overflow_error if `parameter_a.size() * input.size()` exceeds the column size limit
 *
 * @param input Strings column to compute minhash
 * @param seed Seed value used for the hash algorithm
 * @param parameter_a Values used for the permuted calculation
 * @param parameter_b Values used for the permuted calculation
 * @param width The character width of substrings to hash for each row
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return List column of minhash values for each string per seed
 */
std::unique_ptr<cudf::column> minhash64(
  cudf::strings_column_view const& input,
  uint64_t seed,
  cudf::device_span<uint64_t const> parameter_a,
  cudf::device_span<uint64_t const> parameter_b,
  cudf::size_type width,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the minhash values for each input row
 *
 * This function uses MurmurHash3_x86_32 for the hash algorithm.
 *
 * The input row is first hashed using the given `seed` over a sliding window
 * of `ngrams` of strings. These hash values are then combined with the `a`
 * and `b` parameter values using the following formula:
 * ```
 *   max_hash = max of uint32
 *   mp = (1 << 61) - 1
 *   hv[i] = hash value of a ngrams at i
 *   pv[i] = ((hv[i] * a[i] + b[i]) % mp) & max_hash
 * ```
 *
 * This calculation is performed on each set of ngrams and the minimum value
 * is computed as follows:
 * ```
 *   mh[j,i] = min(pv[i]) for all ngrams in row j
 *                        and where i=[0,a.size())
 * ```
 *
 * Any null row entries result in corresponding null output rows.
 *
 * @throw std::invalid_argument if the ngrams < 2
 * @throw std::invalid_argument if parameter_a is empty
 * @throw std::invalid_argument if `parameter_b.size() != parameter_a.size()`
 * @throw std::overflow_error if `parameter_a.size() * input.size()` exceeds the column size limit
 *
 * @param input Strings column to compute minhash
 * @param ngrams The number of strings to hash within each row
 * @param seed Seed value used for the hash algorithm
 * @param parameter_a Values used for the permuted calculation
 * @param parameter_b Values used for the permuted calculation
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return List column of minhash values for each string per seed
 */
std::unique_ptr<cudf::column> minhash_ngrams(
  cudf::lists_column_view const& input,
  cudf::size_type ngrams,
  uint32_t seed,
  cudf::device_span<uint32_t const> parameter_a,
  cudf::device_span<uint32_t const> parameter_b,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the minhash values for each input row
 *
 * This function uses MurmurHash3_x64_128 for the hash algorithm.
 *
 * The input row is first hashed using the given `seed` over a sliding window
 * of `ngrams` of strings. These hash values are then combined with the `a`
 * and `b` parameter values using the following formula:
 * ```
 *   max_hash = max of uint64
 *   mp = (1 << 61) - 1
 *   hv[i] = hash value of a ngrams at i
 *   pv[i] = ((hv[i] * a[i] + b[i]) % mp) & max_hash
 * ```
 *
 * This calculation is performed on each set of ngrams and the minimum value
 * is computed as follows:
 * ```
 *   mh[j,i] = min(pv[i]) for all ngrams in row j
 *                        and where i=[0,a.size())
 * ```
 *
 * Any null row entries result in corresponding null output rows.
 *
 * @throw std::invalid_argument if the ngrams < 2
 * @throw std::invalid_argument if parameter_a is empty
 * @throw std::invalid_argument if `parameter_b.size() != parameter_a.size()`
 * @throw std::overflow_error if `parameter_a.size() * input.size()` exceeds the column size limit
 *
 * @param input List strings column to compute minhash
 * @param ngrams The number of strings to hash within each row
 * @param seed Seed value used for the hash algorithm
 * @param parameter_a Values used for the permuted calculation
 * @param parameter_b Values used for the permuted calculation
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return List column of minhash values for each string per seed
 */
std::unique_ptr<cudf::column> minhash64_ngrams(
  cudf::lists_column_view const& input,
  cudf::size_type ngrams,
  uint64_t seed,
  cudf::device_span<uint64_t const> parameter_a,
  cudf::device_span<uint64_t const> parameter_b,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT nvtext
