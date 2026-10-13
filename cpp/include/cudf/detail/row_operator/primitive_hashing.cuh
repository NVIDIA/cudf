/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/row_operator/preprocessed_table.cuh>
#include <cudf/detail/row_operator/primitive_common.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/hashing/detail/default_hash.cuh>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <memory>

namespace CUDF_EXPORT cudf {
namespace detail::row::primitive {

/**
 * @brief Function object for computing the hash value of a row in a column.
 *
 * @tparam Hash Hash functor to use for hashing elements
 */
template <template <typename> class Hash>
class element_hasher {
 public:
  using result_type = cuda::std::invoke_result_t<Hash<int32_t>, int32_t>;

  /**
   * @brief Returns the hash value of the given element in the given column.
   *
   * @tparam T The type of the element to hash
   * @param seed The seed value to use for hashing
   * @param col The column to hash
   * @param row_index The index of the row to hash
   * @return The hash value of the given element
   */
  template <typename T, CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
  __device__ result_type operator()(result_type seed,
                                    column_device_view const& col,
                                    size_type row_index) const
  {
    return Hash<T>{seed}(col.element<T>(row_index));
  }

  // @cond
  template <typename T, CUDF_ENABLE_IF(not column_device_view::has_element_accessor<T>())>
  __device__ result_type operator()(result_type, column_device_view const&, size_type) const
  {
    CUDF_UNREACHABLE("Unsupported type in hash.");
  }
  // @endcond
};

/**
 * @brief Computes the hash value of a row in the given table.
 *
 * @tparam Hash Hash functor to use for hashing elements.
 */
template <template <typename> class Hash = cudf::hashing::detail::default_hash>
class row_hasher {
 public:
  using result_type = cuda::std::invoke_result_t<Hash<int32_t>, int32_t>;

  row_hasher() = delete;

  /**
   * @brief Constructs a row_hasher object with a seed value.
   *
   * @param has_nulls Indicates if the input column contains nulls
   * @param t A table_device_view to hash
   * @param seed A seed value to use for hashing
   */
  row_hasher(cudf::nullate::DYNAMIC const& has_nulls,
             table_device_view t,
             result_type seed = hashing::detail::DEFAULT_ALGORITHM_HASH_SEED)
    : _has_nulls{has_nulls}, _table{t}, _seed{seed}
  {
  }

  /**
   * @brief Constructs a row_hasher object with a seed value.
   *
   * @param has_nulls Indicates if the input column contains nulls
   * @param t Preprocessed table to hash
   * @param seed A seed value to use for hashing
   */
  row_hasher(cudf::nullate::DYNAMIC const& has_nulls,
             std::shared_ptr<cudf::detail::row::equality::preprocessed_table> t,
             result_type seed = hashing::detail::DEFAULT_ALGORITHM_HASH_SEED)
    : _has_nulls{has_nulls}, _table{*t}, _seed{seed}
  {
  }

  /**
   * @brief Computes the hash value of the row at `row_index` in the `table`
   *
   * @param row_index The index of the row in the `table` to hash
   * @return The hash value of the row at `row_index` in the `table`
   */
  __device__ auto operator()(size_type row_index) const
  {
    element_hasher<Hash> hasher;
    auto hash = cuda::std::numeric_limits<result_type>::max();
    if (!_has_nulls || !_table.column(0).is_null(row_index)) {
      hash = cudf::type_dispatcher<dispatch_primitive_type>(
        _table.column(0).type(), hasher, _seed, _table.column(0), row_index);
    }

    for (size_type i = 1; i < _table.num_columns(); ++i) {
      if (!(_has_nulls && _table.column(i).is_null(row_index))) {
        hash = cudf::hashing::detail::hash_combine(
          hash,
          cudf::type_dispatcher<dispatch_primitive_type>(
            _table.column(i).type(), hasher, _seed, _table.column(i), row_index));
      } else {
        hash =
          cudf::hashing::detail::hash_combine(hash, cuda::std::numeric_limits<result_type>::max());
      }
    }
    return hash;
  }

 private:
  cudf::nullate::DYNAMIC _has_nulls;
  table_device_view _table;
  result_type _seed;
};

}  // namespace detail::row::primitive
}  // namespace CUDF_EXPORT cudf
