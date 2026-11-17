/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/iterator.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <iterator>
#include <memory>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace test {
namespace iterators {

/**
 * @brief Bool iterator for marking (possibly multiple) null elements in a column_wrapper.
 *
 * The returned iterator yields `false` (to mark `null`) at all the specified indices,
 * and yields `true` (to mark valid rows) for all other indices. E.g.
 *
 * @code
 * auto indices = std::vector<size_type>{8,9};
 * auto iter = nulls_at(indices.cbegin(), indices.end());
 * iter[6] == true;  // i.e. Valid row at index 6.
 * iter[7] == true;  // i.e. Valid row at index 7.
 * iter[8] == false; // i.e. Invalid row at index 8.
 * iter[9] == false; // i.e. Invalid row at index 9.
 * @endcode
 *
 * @tparam Iter Iterator type
 * @param index_start Iterator to start of indices for which the validity iterator
 *                    must return `false` (i.e. null)
 * @param index_end   Iterator to end of indices for the validity iterator
 * @return auto Validity iterator
 */
template <typename Iter>
[[maybe_unused]] static auto nulls_at(Iter index_start, Iter index_end)
{
  using index_type = typename std::iterator_traits<Iter>::value_type;

  auto indices = std::make_shared<std::vector<index_type> const>(index_start, index_end);
  return cuda::transform_iterator(
    cuda::counting_iterator<cudf::size_type>{0}, [indices = std::move(indices)](auto i) {
      return std::find(indices->cbegin(), indices->cend(), i) == indices->cend();
    });
}

/**
 * @brief Bool iterator for marking (possibly multiple) null elements in a column_wrapper.
 *
 * The returned iterator yields `false` (to mark `null`) at all the specified indices,
 * and yields `true` (to mark valid rows) for all other indices. E.g.
 *
 * @code
 * auto iter = nulls_at({8,9});
 * iter[6] == true;  // i.e. Valid row at index 6.
 * iter[7] == true;  // i.e. Valid row at index 7.
 * iter[8] == false; // i.e. Invalid row at index 8.
 * iter[9] == false; // i.e. Invalid row at index 9.
 * @endcode
 *
 * @param indices The indices for which the validity iterator must return `false` (i.e. null)
 * @return auto Validity iterator
 */
[[maybe_unused]] static auto nulls_at(std::vector<cudf::size_type> const& indices)
{
  return nulls_at(indices.cbegin(), indices.cend());
}

/**
 * @brief Bool iterator for marking a single null element in a column_wrapper
 *
 * The returned iterator yields `false` (to mark `null`) at the specified index,
 * and yields `true` (to mark valid rows) for all other indices. E.g.
 *
 * @code
 * auto iter = null_at(8);
 * iter[7] == true;  // i.e. Valid row at index 7.
 * iter[8] == false; // i.e. Invalid row at index 8.
 * @endcode
 *
 * @param index The index for which the validity iterator must return `false` (i.e. null)
 * @return auto Validity iterator
 */
[[maybe_unused]] static auto null_at(cudf::size_type index)
{
  return nulls_at(std::vector<cudf::size_type>{index});
}

/**
 * @brief Bool iterator for marking all elements are null
 *
 * @return auto Validity iterator which always yields `false`
 */
[[maybe_unused]] static auto all_nulls() { return cuda::make_constant_iterator(false); }

/**
 * @brief Bool iterator for marking all elements are valid (non-null)
 *
 * @return auto Validity iterator which always yields `true`
 */
[[maybe_unused]] static auto no_nulls() { return cuda::make_constant_iterator(true); }

/**
 * @brief Bool iterator for marking null elements at every multiple of n.
 *
 * The returned iterator yields `false` (to mark `null`) at indices 0, n, 2n, ...,
 * and yields `true` (to mark valid rows) for all other indices. E.g.
 *
 * @code
 * auto iter = nulls_at_multiples_of(3);
 * iter[0] == false; // i.e. Invalid (null) row at index 0.
 * iter[1] == true;  // i.e. Valid row at index 1.
 * iter[2] == true;  // i.e. Valid row at index 2.
 * iter[3] == false; // i.e. Invalid (null) row at index 3.
 * @endcode
 *
 * @param n The period at which nulls occur (nulls at indices 0, n, 2n, ...)
 * @return auto Validity iterator
 */
[[maybe_unused]] static auto nulls_at_multiples_of(cudf::size_type n)
{
  return cudf::detail::make_counting_transform_iterator(0, [n](auto i) { return (i % n) != 0; });
}

/**
 * @brief Bool iterator for marking valid elements only at multiples of n, null elsewhere.
 *
 * The returned iterator yields `true` (to mark valid rows) at indices 0, n, 2n, ...,
 * and yields `false` (to mark `null`) for all other indices. E.g.
 *
 * @code
 * auto iter = valids_at_multiples_of(3);
 * iter[0] == true;  // i.e. Valid row at index 0.
 * iter[1] == false; // i.e. Invalid (null) row at index 1.
 * iter[2] == false; // i.e. Invalid (null) row at index 2.
 * iter[3] == true;  // i.e. Valid row at index 3.
 * @endcode
 *
 * @param n The period at which valid elements occur (valid at indices 0, n, 2n, ...)
 * @return auto Validity iterator
 */
[[maybe_unused]] static auto valids_at_multiples_of(cudf::size_type n)
{
  return cudf::detail::make_counting_transform_iterator(0, [n](auto i) { return (i % n) == 0; });
}

/**
 * @brief Bool iterator for marking null elements from pointers of data
 *
 * The returned iterator yields `false` (to mark `null`) at the indices corresponding to the
 * pointers having `nullptr` values and `true` for the remaining indices.
 *
 * @note The input vector is referenced by the transform iterator, so the
 * lifespan must be just as long as the iterator.
 *
 * @tparam T the data type
 * @param ptrs The data pointers for which the validity iterator is computed
 * @return auto Validity iterator
 */
template <class T>
[[maybe_unused]] static auto nulls_from_nullptrs(std::vector<T const*> const& ptrs)
{
  return cuda::transform_iterator(ptrs.begin(), [](auto ptr) { return ptr != nullptr; });
}

}  // namespace iterators
}  // namespace test
}  // namespace CUDF_EXPORT cudf
