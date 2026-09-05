/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/utilities/accumulate.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/structs/structs_column_device_view.cuh>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {
namespace detail::row::hash {

/**
 * @brief Computes row hashes using Spark's iterative seeding convention.
 *
 * Spark uses the hash of each value as the seed for the next value and ignores
 * null values. Consequently, values of different nested shapes can collide.
 * For example, the integer `1`, the list `[1]`, and a struct containing only
 * `1` have the same hash. Likewise, `[1]`, `[1, null]`, and `[null, 1]` have
 * the same hash. A null element returns its input seed unchanged.
 *
 * The element hash function is responsible for Spark-specific type encodings
 * and hash algorithm behavior.
 *
 * LIST columns whose child is a STRUCT are unsupported and must be rejected
 * before invoking this hasher.
 *
 * @tparam hash_function Seeded element hash functor
 * @tparam Nullate A cudf::nullate type describing whether to check for nulls
 */
template <template <typename> class hash_function, typename Nullate>
class spark_device_row_hasher {
  friend class row_hasher;

 public:
  // `int32_t` is only a probe type used to ask the hasher what it returns.  Every
  // `hash_function` specialization shares one return type, so the choice does not affect the
  // answer; it matches the probe used by `element_hasher` in `hashing.cuh`.
  using result_type = cuda::std::invoke_result_t<hash_function<int32_t>, int32_t>;

  /**
   * @brief Returns the hash value of a row in the table.
   *
   * @param row_index The row index to hash
   * @return The hash value of the row
   */
  __device__ result_type operator()(size_type row_index) const noexcept
  {
    return cudf::detail::accumulate(
      _table.begin(),
      _table.end(),
      _seed,
      cuda::proclaim_return_type<result_type>(
        [row_index, nulls = this->_check_nulls] __device__(auto hash, auto column) {
          return cudf::type_dispatcher(
            column.type(), element_hasher_adapter{nulls, hash}, column, row_index);
        }));
  }

 private:
  /**
   * @brief Computes the hash value of an element in a column.
   *
   * Nested values are flattened and hashed serially, with each output becoming
   * the seed for the next value. A null element returns the input seed.
   */
  class element_hasher_adapter {
    using hash_functor = element_hasher<hash_function, Nullate>;

   public:
    __device__ element_hasher_adapter(Nullate check_nulls, result_type seed) noexcept
      : _check_nulls(check_nulls), _seed(seed)
    {
    }

    template <typename T>
      requires(not cudf::is_nested<T>())
    __device__ result_type operator()(column_device_view const& col,
                                      size_type row_index) const noexcept
    {
      auto const hasher = hash_functor{_check_nulls, _seed, _seed};
      return hasher.template operator()<T>(col, row_index);
    }

    template <typename T>
      requires(cudf::is_nested<T>())
    __device__ result_type operator()(column_device_view const& col,
                                      size_type row_index) const noexcept
    {
      column_device_view curr_col = col.slice(row_index, 1);
      while (curr_col.type().id() == type_id::STRUCT || curr_col.type().id() == type_id::LIST) {
        if (curr_col.type().id() == type_id::STRUCT) {
          if (curr_col.num_child_columns() == 0) { return _seed; }
          // Non-empty structs are decomposed and contain only one child.
          curr_col = structs_column_device_view(curr_col).get_sliced_child(0);
        } else {
          curr_col = lists_column_device_view(curr_col).get_sliced_child();
        }
      }

      return cudf::detail::accumulate(
        cuda::counting_iterator<size_type>{0},
        cuda::counting_iterator<size_type>{curr_col.size()},
        _seed,
        cuda::proclaim_return_type<result_type>(
          [curr_col, nulls = this->_check_nulls] __device__(auto hash, auto element_index) {
            auto const hasher = hash_functor{nulls, hash, hash};
            return cudf::type_dispatcher<cudf::detail::dispatch_void_if_nested>(
              curr_col.type(), hasher, curr_col, element_index);
          }));
    }

   private:
    Nullate const _check_nulls;
    result_type const _seed;
  };

  CUDF_HOST_DEVICE spark_device_row_hasher(Nullate check_nulls,
                                           table_device_view table,
                                           result_type seed = DEFAULT_HASH_SEED) noexcept
    : _check_nulls{check_nulls}, _table{table}, _seed(seed)
  {
  }

  Nullate const _check_nulls;
  table_device_view const _table;
  result_type const _seed;
};

}  // namespace detail::row::hash
}  // namespace CUDF_EXPORT cudf
