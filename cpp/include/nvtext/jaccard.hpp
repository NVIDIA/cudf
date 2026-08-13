/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace CUDF_EXPORT nvtext {
/**
 * @addtogroup nvtext_jaccard
 * @{
 * @file
 * @brief APIs for computing the Jaccard similarity between strings in two columns.
 */

/**
 * @brief Computes the Jaccard similarity between individual rows
 * in two strings columns
 *
 * The similarity is calculated between strings in corresponding rows
 * such that `output[row] = J(input1[row],input2[row])`.
 *
 * The Jaccard index formula is https://en.wikipedia.org/wiki/Jaccard_index
 * ```
 *  J = |A ∩ B| / |A ∪ B|
 *  where |A ∩ B| is number of common values between A and B
 *  and |x| is the number of unique values in x.
 * ```
 *
 * The computation here compares strings columns by treating each string as text (i.e. sentences,
 * paragraphs, articles) instead of individual words or tokens to be compared directly. The
 * algorithm applies a sliding window (size specified by the `width` parameter) to each string to
 * form the set of tokens to compare within each row of the two input columns.
 *
 * These substrings are essentially character ngrams and used as part of the union and intersect
 * calculations for that row. For efficiency, the substrings are hashed using the default
 * MurmurHash32 to identify uniqueness within each row. Once the union and intersect sizes for the
 * row are resolved, the Jaccard index is computed using the above formula and returned as a float32
 * value.
 *
 * @code{.pseudo}
 * input1 = ["the fuzzy dog", "little piggy", "funny bunny", "chatty parrot"]
 * input2 = ["the fuzzy cat", "bitty piggy", "funny bunny", "silent partner"]
 * r = jaccard_index(input1, input2)
 * r is now [0.5, 0.15384616, 1.0, 0]
 * @endcode
 *
 * If either input column's row is null, the output for that row will also be null.
 *
 * @throw std::invalid_argument if the `width < 2` or `input1.size() != input2.size()`
 *
 * @param input1 Strings column to compare with `input2`
 * @param input2 Strings column to compare with `input1`
 * @param width The character width used for apply substrings
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Index calculation values
 */
std::unique_ptr<cudf::column> jaccard_index(
  cudf::strings_column_view const& input1,
  cudf::strings_column_view const& input2,
  cudf::size_type width,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT nvtext
