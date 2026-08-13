/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief APIs for computing hash values of columns and tables using various hash algorithms.
 */

namespace CUDF_EXPORT cudf {

/**
 * @brief Type of hash value
 * @ingroup column_hash
 */
using hash_value_type = uint32_t;

/**
 * @brief The default seed value for hash functions
 * @ingroup column_hash
 */
static constexpr uint32_t DEFAULT_HASH_SEED = 0;

//! Hash APIs
namespace hashing {

/**
 * @addtogroup column_hash
 * @{
 */

/**
 * @brief Computes the MurmurHash3 32-bit hash value of each row in the given table
 *
 * This function computes the hash of each column using the `seed` for the first column
 * and the resulting hash as a seed for the next column and so on.
 * The result is a uint32 value for each row.
 *
 * @param input The table of columns to hash
 * @param seed Optional seed value to use for the hash function
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column where each row is the hash of a row from the input
 */
std::unique_ptr<column> murmurhash3_x86_32(
  table_view const& input,
  uint32_t seed                     = DEFAULT_HASH_SEED,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Computes the MurmurHash3 64-bit hash value of each row in the given table
 *
 * This function takes a 64-bit seed value and returns hash values using the
 * MurmurHash3_x64_128 algorithm. The hash produces in two uint64 values per row.
 *
 * @param input The table of columns to hash
 * @param seed Optional seed value to use for the hash function
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A table of two UINT64 columns
 */
std::unique_ptr<table> murmurhash3_x64_128(
  table_view const& input,
  uint64_t seed                     = DEFAULT_HASH_SEED,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Computes the MD5 hash value of each row in the given table
 *
 * @param input The table of columns to hash
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column where each row is the hash of a row from the input
 */
std::unique_ptr<column> md5(
  table_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Computes the SHA-1 hash value of each row in the given table
 *
 * @param input The table of columns to hash
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column where each row is the hash of a row from the input
 */
std::unique_ptr<column> sha1(
  table_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Computes the SHA-224 hash value of each row in the given table
 *
 * @param input The table of columns to hash
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column where each row is the hash of a row from the input
 */
std::unique_ptr<column> sha224(
  table_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Computes the SHA-256 hash value of each row in the given table
 *
 * @param input The table of columns to hash
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column where each row is the hash of a row from the input
 */
std::unique_ptr<column> sha256(
  table_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Computes the SHA-384 hash value of each row in the given table
 *
 * @param input The table of columns to hash
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column where each row is the hash of a row from the input
 */
std::unique_ptr<column> sha384(
  table_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Computes the SHA-512 hash value of each row in the given table
 *
 * @param input The table of columns to hash
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column where each row is the hash of a row from the input
 */
std::unique_ptr<column> sha512(
  table_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Computes the XXHash_32 hash value of each row in the given table
 *
 * This function computes the hash of each column using the `seed` for the first column
 * and the resulting hash as a seed for the next column and so on.
 * The result is a uint32 value for each row.
 *
 * @param input The table of columns to hash
 * @param seed Optional seed value to use for the hash function
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column where each row is the hash of a row from the input
 */
std::unique_ptr<column> xxhash_32(
  table_view const& input,
  uint32_t seed                     = DEFAULT_HASH_SEED,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Computes the XXHash_64 hash value of each row in the given table
 *
 * This function takes a 64-bit seed value and returns a column of type UINT64.
 *
 * @param input The table of columns to hash
 * @param seed Optional seed value to use for the hash function
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column where each row is the hash of a row from the input
 */
std::unique_ptr<column> xxhash_64(
  table_view const& input,
  uint64_t seed                     = DEFAULT_HASH_SEED,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group

}  // namespace hashing

}  // namespace CUDF_EXPORT cudf
