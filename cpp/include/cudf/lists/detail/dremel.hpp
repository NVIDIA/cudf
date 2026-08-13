/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>

#include <rmm/device_uvector.hpp>

namespace cudf {
namespace detail {

/**
 * @brief Device view for `dremel_data`.
 *
 * @see the `dremel_data` struct for more info.
 */
struct dremel_device_view {
  size_type const* offsets;
  uint8_t const* rep_levels;
  uint8_t const* def_levels;
  size_type leaf_data_size;
  uint8_t max_def_level;
};

/**
 * @brief Dremel data that describes one nested type column
 *
 * @see get_dremel_data() for more info.
 */
struct dremel_data {
  rmm::device_uvector<size_type> dremel_offsets;
  rmm::device_uvector<uint8_t> rep_level;
  rmm::device_uvector<uint8_t> def_level;

  size_type leaf_data_size;
  uint8_t max_def_level;

  operator dremel_device_view() const
  {
    return dremel_device_view{
      dremel_offsets.data(), rep_level.data(), def_level.data(), leaf_data_size, max_def_level};
  }
};

/**
 * @brief Get the dremel offsets, repetition levels, and definition levels for a LIST column
 *
 * Dremel is a query system created by Google for ad hoc data analysis. The Dremel engine is
 * described in depth in the paper "Dremel: Interactive Analysis of Web-Scale
 * Datasets" (https://research.google/pubs/dremel-interactive-analysis-of-web-scale-datasets-2/).
 * One of the key components of Dremel is an encoding that converts record-like data into a columnar
 * store for efficient memory accesses. The Parquet file format uses Dremel encoding to handle
 * nested data, so libcudf requires some facilities for working with this encoding. Furthermore,
 * libcudf leverages Dremel encoding as a means for performing lexicographic comparisons of nested
 * columns.
 *
 * Dremel encoding is built around two concepts, the repetition and definition levels.
 * Since describing them thoroughly is out of scope for this docstring, here are a couple of
 * blogs that provide useful background:
 *
 * http://www.goldsborough.me/distributed-systems/2019/05/18/21-09-00-a_look_at_dremel/
 * https://akshays-blog.medium.com/wrapping-head-around-repetition-and-definition-levels-in-dremel-powering-bigquery-c1a33c9695da
 * https://blog.x.com/engineering/en_us/a/2013/dremel-made-simple-with-parquet
 *
 * The remainder of this documentation assumes familiarity with the Dremel concepts.
 *
 * Dremel offsets are the per row offsets into the repetition and definition level arrays for a
 * column.
 * Example:
 * ```
 * col            = {{1, 2, 3}, { }, {5, 6}}
 * dremel_offsets = { 0,         3,   4,  6}
 * rep_level      = { 0, 1, 1,   0,   0, 1}
 * def_level      = { 1, 1, 1,   0,   1, 1}
 * ```
 *
 * The repetition and definition level values are ideally computed using a recursive call over a
 * nested structure but in order to better utilize GPU resources, this function calculates them
 * with a bottom up merge method.
 *
 * Given a LIST column of type `List<List<int>>` like so:
 * ```
 * col = {
 *    [],
 *    [[], [1, 2, 3], [4, 5]],
 *    [[]]
 * }
 * ```
 * We can represent it in cudf format with two level of offsets like this:
 * ```
 * Level 0 offsets = {0, 0, 3, 4}
 * Level 1 offsets = {0, 0, 3, 5, 5}
 * Values          = {1, 2, 3, 4, 5}
 * ```
 * This function returns the dremel offsets, repetition levels, and definition level
 * values that correspond to the data values:
 * ```
 * col =            {[], [[], [1, 2, 3], [4, 5]], [[]]}
 * dremel_offsets = { 0,  1,                       7, 8}
 * def_levels     = { 0,  1,   2, 2, 2,   2, 2,    1 }
 * rep_levels     = { 0,  0,   1, 2, 2,   1, 2,    0 }
 * ```
 *
 * Since repetition and definition levels arrays contain a value for each empty list, the size of
 * the rep/def level array can be given by
 * ```
 * rep_level.size() = size of leaf column + number of empty lists in level 0
 *                                        + number of empty lists in level 1 ...
 * ```
 *
 * We start with finding the empty lists in the penultimate level and merging it with the indices
 * of the leaf level. The values for the merge are the definition and repetition levels
 * ```
 * empties at level 1 = {0, 5}
 * def values at 1    = {1, 1}
 * rep values at 1    = {1, 1}
 * indices at leaf    = {0, 1, 2, 3, 4}
 * def values at leaf = {2, 2, 2, 2, 2}
 * rep values at leaf = {2, 2, 2, 2, 2}
 * ```
 *
 * merged def values  = {1, 2, 2, 2, 2, 2, 1}
 * merged rep values  = {1, 2, 2, 2, 2, 2, 1}
 *
 * The size of the rep/def values is now larger than the leaf values and the offsets need to be
 * adjusted in order to point to the correct start indices. We do this with an exclusive scan over
 * the indices of offsets of empty lists and adding to existing offsets.
 * ```
 * Level 1 new offsets = {0, 1, 4, 6, 7}
 * ```
 * Repetition values at the beginning of a list need to be decremented. We use the new offsets to
 * scatter the rep value.
 * ```
 * merged rep values  = {1, 2, 2, 2, 2, 2, 1}
 * scatter (1, new offsets)
 * new offsets        = {0, 1,       4,    6, 7}
 * new rep values     = {1, 1, 2, 2, 1, 2, 1}
 * ```
 *
 * Similarly we merge up all the way till level 0 offsets
 *
 * STRUCT COLUMNS :
 * In case of struct columns, we don't have to merge struct levels with their children because a
 * struct is the same size as its children. e.g. for a column `struct<int, float>`, if the row `i`
 * is null, then the children columns `int` and `float` are also null at `i`. They also have the
 * null entry represented in their respective null masks. So for any case of strictly struct based
 * nesting, we can get the definition levels merely by iterating over the nesting for the same row.
 *
 * In case struct and lists are intermixed, the definition levels of all the contiguous struct
 * levels can be constructed using the aforementioned iterative method. Only when we reach a list
 * level, we need to do a merge with the subsequent level.
 *
 * So, for a column like `struct<list<int>>`, we are going to merge between the levels `struct<list`
 * and `int`.
 * For a column like `list<struct<int>>`, we are going to merge between `list` and `struct<int>`.
 *
 * In general, one nesting level is the list level and any struct level that precedes it.
 *
 * A few more examples to visualize the partitioning of column hierarchy into nesting levels:
 * (L is list, S is struct, i is integer(leaf data level), angle brackets omitted)
 * ```
 * 1. LSi     = L   Si
 *              - | --
 *
 * 2. LLSi    = L   L   Si
 *              - | - | --
 *
 * 3. SSLi    = SSL   i
 *              --- | -
 *
 * 4. LLSLSSi = L   L   SL   SSi
 *              - | - | -- | ---
 * ```
 *
 * @param input Column of LIST type
 * @param nullability Pre-determined nullability at each list level. Empty means infer from
 * `input`
 * @param output_as_byte_array if `true`, then any nested list level that has a child of type
 * `uint8_t` will be considered as the last level
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return A struct containing dremel data
 */
dremel_data get_dremel_data(column_view input,
                            std::vector<uint8_t> nullability,
                            bool output_as_byte_array,
                            rmm::cuda_stream_view stream);

/**
 * @brief Get Dremel offsets, repetition levels, and modified definition levels to be used for
 *        lexicographical comparators. The modified definition levels are produced by treating
 *        each nested column in the input as nullable
 *
 * @param input Column of LIST type
 * @param nullability Pre-determined nullability at each list level. Empty means infer from
 * `input`
 * @param output_as_byte_array if `true`, then any nested list level that has a child of type
 * `uint8_t` will be considered as the last level
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return A struct containing dremel data
 */
dremel_data get_comparator_data(column_view input,
                                std::vector<uint8_t> nullability,
                                bool output_as_byte_array,
                                rmm::cuda_stream_view stream);
}  // namespace detail
}  // namespace cudf
