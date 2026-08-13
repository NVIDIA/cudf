/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace CUDF_EXPORT nvtext {

/**
 * @addtogroup nvtext_tokenize
 * @{
 * @file
 * @brief APIs for byte pair encoding (BPE) tokenization of strings columns.
 */

/**
 * @brief The table of merge pairs for the BPE encoder.
 *
 * @deprecated Since 26.04
 *
 * To create an instance, call @ref nvtext::load_merge_pairs
 */
struct bpe_merge_pairs {
  struct bpe_merge_pairs_impl;

  /**
   * @brief Construct a new bpe merge pairs object
   *
   * @param input The input file containing the BPE merge pairs
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the device memory
   */
  bpe_merge_pairs(std::unique_ptr<cudf::column>&& input,
                  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Construct a new bpe merge pairs object
   *
   * @param input The input column of strings
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the device memory
   */
  bpe_merge_pairs(cudf::strings_column_view const& input,
                  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  ~bpe_merge_pairs();
  bpe_merge_pairs();

 private:
  friend bpe_merge_pairs_impl const* get_bpe_merge_pairs_impl(bpe_merge_pairs const&);
  bpe_merge_pairs_impl* impl{};  ///< Implementation of the BPE merge pairs table.
};

/**
 * @brief Create a nvtext::bpe_merge_pairs from a strings column
 *
 * The input column should contain a unique pair of strings per line separated by
 * a single space. An incorrect format or non-unique entries will result in
 * undefined behavior.
 *
 * Example:
 * @code{.pseudo}
 * merge_pairs = ["e n", "i t", "i s", "e s", "en t", "c e", "es t", "en ce", "t est", "s ent"]
 * mps = load_merge_pairs(merge_pairs)
 * // the mps object can be passed to the byte_pair_encoding API
 * @endcode
 *
 * The pairs are expected to be ordered in the file by their rank
 * relative to each other. A pair earlier in the file has priority over
 * any pairs below it.
 *
 * @throw cudf::logic_error if `merge_pairs` is empty or contains nulls
 *
 * @param merge_pairs Column containing the unique merge pairs
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resource to allocate any returned objects
 * @return A nvtext::bpe_merge_pairs object
 */
std::unique_ptr<bpe_merge_pairs> load_merge_pairs(
  cudf::strings_column_view const& merge_pairs,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Byte pair encode the input strings.
 *
 * The encoding algorithm rebuilds each string by matching substrings
 * in the `merge_pairs` table and iteratively removing the minimum ranked pair
 * until no pairs are left. Then, the separator is inserted between the remaining
 * pairs before the result is joined to make the output string.
 *
 * @code{.pseudo}
 * merge_pairs = ["e n", "i t", "i s", "e s", "en t", "c e", "es t", "en ce", "t est", "s ent"]
 * mps = load_merge_pairs(merge_pairs)
 * input = ["test sentence", "thisis test"]
 * result = byte_pair_encoding(input, mps)
 * result is now ["test sent ence", "this is test"]
 * @endcode
 *
 * @throw cudf::logic_error if `merge_pairs` is empty
 * @throw cudf::logic_error if `separator` is invalid
 *
 * @param input Strings to encode.
 * @param merges_pairs Created by a call to @ref nvtext::load_merge_pairs.
 * @param separator String used to build the output after encoding.
 *                  Default is a space.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resource to allocate any returned objects.
 * @return An encoded column of strings.
 */
std::unique_ptr<cudf::column> byte_pair_encoding(
  cudf::strings_column_view const& input,
  bpe_merge_pairs const& merges_pairs,
  cudf::string_scalar const& separator = cudf::string_scalar(" "),
  rmm::cuda_stream_view stream         = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr    = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT nvtext
