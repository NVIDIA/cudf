/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/types.hpp>

namespace cudf::groupby::detail {
/**
 * @brief Functor to compare two rows of a table in given permutation order
 *
 * This is useful to identify unique elements in a sorted order table, when the permutation order is
 * the sorted order of the table.
 */
template <typename ComparatorT, typename Iterator>
struct permuted_row_equality_comparator {
  /**
   * @brief Constructs a permuted comparator object which compares two rows of the table in given
   * permutation order
   *
   * @param comparator Equality comparator
   * @param permutation The permutation map that specifies the effective ordering of
   * `t`. Must be the same size as `t.num_rows()`
   */
  permuted_row_equality_comparator(ComparatorT const& comparator, Iterator const permutation)
    : _comparator{comparator}, _permutation{permutation}
  {
  }

  /**
   * @brief Returns true if the two rows at the specified indices in the permuted
   * order are equivalent.
   *
   * For example, comparing rows `i` and `j` is equivalent to comparing
   * rows `permutation[i]` and `permutation[j]` in the original table.
   *
   * @param lhs The index of the first row
   * @param rhs The index of the second row
   * @returns true if the two specified rows in the permuted order are equivalent
   */
  __device__ bool operator()(cudf::size_type lhs, cudf::size_type rhs) const
  {
    return _comparator(_permutation[lhs], _permutation[rhs]);
  };

 private:
  ComparatorT const _comparator;
  Iterator const _permutation;
};
}  // namespace cudf::groupby::detail
