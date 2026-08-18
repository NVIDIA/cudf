/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/detail/row_operator/preprocessed_table.cuh>
#include <cudf/detail/utilities/accumulate.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/hashing/detail/default_hash.cuh>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/structs/structs_column_device_view.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/iterator>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <memory>

namespace CUDF_EXPORT cudf {
namespace detail::row::hash {

/**
 * @brief Computes the hash value of an element in the given column.
 *
 * @tparam hash_function Hash functor to use for hashing elements.
 * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
 */
template <template <typename> class hash_function, typename Nullate>
class element_hasher {
 public:
  using result_type = cuda::std::invoke_result_t<hash_function<int32_t>, int32_t>;

  /**
   * @brief Constructs an element_hasher object.
   *
   * @param nulls Indicates whether to check for nulls
   * @param seed  The seed to use for the hash function
   * @param null_hash The hash value to use for nulls
   */
  __device__ element_hasher(
    Nullate nulls,
    result_type seed      = DEFAULT_HASH_SEED,
    result_type null_hash = cuda::std::numeric_limits<result_type>::max()) noexcept
    : _check_nulls(nulls), _seed(seed), _null_hash(null_hash)
  {
  }

  /**
   * @brief Returns the hash value of the given element.
   *
   * @tparam T The type of the element to hash
   * @param col The column to hash
   * @param row_index The index of the row to hash
   * @return The hash value of the given element
   */
  template <typename T>
  __device__ result_type operator()(column_device_view const& col,
                                    size_type row_index) const noexcept
    requires(column_device_view::has_element_accessor<T>())
  {
    if (_check_nulls && col.is_null(row_index)) { return _null_hash; }
    return hash_function<T>{_seed}(col.element<T>(row_index));
  }

  /**
   * @brief Returns the hash value of the given element.
   *
   * @tparam T The type of the element to hash
   * @param col The column to hash
   * @param row_index The index of the row to hash
   * @return The hash value of the given element
   */
  template <typename T>
  __device__ result_type operator()(column_device_view const& col,
                                    size_type row_index) const noexcept
    requires(not column_device_view::has_element_accessor<T>())
  {
    CUDF_UNREACHABLE("Unsupported type in hash.");
  }

  Nullate _check_nulls;
  // Assumes seeds are the same as the result type of the hash function
  result_type _seed;
  result_type _null_hash;
};

/**
 * @brief Computes the hash value of a row in the given table.
 *
 * @tparam hash_function Hash functor to use for hashing elements.
 * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
 */
template <template <typename> class hash_function, typename Nullate>
class device_row_hasher {
  friend class row_hasher;

 public:
  using result_type = cuda::std::invoke_result_t<hash_function<int32_t>, int32_t>;

  /**
   * @brief Return the hash value of a row in the given table.
   *
   * @param row_index The row index to compute the hash value of
   * @return The hash value of the row
   */
  __device__ result_type operator()(size_type row_index) const noexcept
  {
    auto const hasher = [row_index, this](auto const& column) {
      return cudf::type_dispatcher<dispatch_storage_type>(
        column.type(), element_hasher_adapter{_check_nulls, _seed}, column, row_index);
    };

    auto const has_columns = _table.num_columns() > 0;
    auto const init        = has_columns ? hasher(_table.column(0)) : _seed;
    auto const start_col   = static_cast<size_type>(has_columns);

    auto it = cuda::transform_iterator(_table.begin() + start_col, hasher);
    return detail::accumulate(
      it, it + (_table.num_columns() - start_col), init, [](auto hash, auto h) {
        return cudf::hashing::detail::hash_combine(hash, h);
      });
  }

 private:
  /**
   * @brief Computes the hash value of an element in the given column.
   *
   * When the column is non-nested, this is a simple wrapper around the element_hasher.
   * When the column is nested, this uses the element_hasher to hash the shape and values of the
   * column.
   */
  class element_hasher_adapter {
    static constexpr result_type NULL_HASH     = cuda::std::numeric_limits<result_type>::max();
    static constexpr result_type NON_NULL_HASH = 0;

   public:
    __device__ element_hasher_adapter(Nullate check_nulls, result_type seed) noexcept
      : _element_hasher(check_nulls, seed), _check_nulls(check_nulls)
    {
    }

    template <typename T>
    __device__ result_type operator()(column_device_view const& col,
                                      size_type row_index) const noexcept
      requires(not cudf::is_nested<T>() and not cudf::is_dictionary<T>())
    {
      return _element_hasher.template operator()<T>(col, row_index);
    }

    template <typename T>
    __device__ result_type operator()(column_device_view const& col,
                                      size_type row_index) const noexcept
      requires(cudf::is_dictionary<T>())
    {
      if (_check_nulls && col.is_null(row_index)) { return NULL_HASH; }

      auto const keys = col.child(dictionary_column_view::keys_column_index);
      return type_dispatcher<dispatch_storage_type>(
        keys.type(),
        _element_hasher,
        keys,
        static_cast<size_type>(col.element<dictionary32>(row_index)));
    }

    template <typename T>
    __device__ result_type operator()(column_device_view const& col,
                                      size_type row_index) const noexcept
      requires(cudf::is_nested<T>())
    {
      auto hash                   = result_type{0};
      column_device_view curr_col = col.slice(row_index, 1);
      while (curr_col.type().id() == type_id::STRUCT || curr_col.type().id() == type_id::LIST) {
        if (_check_nulls) {
          auto validity_it = detail::make_validity_iterator<true>(curr_col);
          hash             = detail::accumulate(
            validity_it, validity_it + curr_col.size(), hash, [](auto hash, auto is_valid) {
              return cudf::hashing::detail::hash_combine(hash,
                                                         is_valid ? NON_NULL_HASH : NULL_HASH);
            });
        }
        if (curr_col.type().id() == type_id::STRUCT) {
          if (curr_col.num_child_columns() == 0) { return hash; }
          curr_col = structs_column_device_view(curr_col).get_sliced_child(0);
        } else if (curr_col.type().id() == type_id::LIST) {
          auto list_col   = lists_column_device_view(curr_col);
          auto list_sizes = make_list_size_iterator(list_col);
          hash            = detail::accumulate(
            list_sizes, list_sizes + list_col.size(), hash, [](auto hash, auto size) {
              return cudf::hashing::detail::hash_combine(hash, hash_function<size_type>{}(size));
            });
          curr_col = list_col.get_sliced_child();
        }
      }
      for (int i = 0; i < curr_col.size(); ++i) {
        hash = cudf::hashing::detail::hash_combine(
          hash,
          type_dispatcher<dispatch_void_if_nested>(curr_col.type(), _element_hasher, curr_col, i));
      }
      return hash;
    }

    element_hasher<hash_function, Nullate> const _element_hasher;
    Nullate const _check_nulls;
  };

  CUDF_HOST_DEVICE device_row_hasher(Nullate check_nulls,
                                     table_device_view t,
                                     result_type seed = DEFAULT_HASH_SEED) noexcept
    : _check_nulls{check_nulls}, _table{t}, _seed(seed)
  {
  }

  Nullate const _check_nulls;
  table_device_view const _table;
  // Assumes seeds are the same as the result type of the hash function
  result_type const _seed;
};

/**
 * @brief Computes the hash value of a row in the given table.
 *
 */
class row_hasher {
 public:
  /**
   * @brief Construct an owning object for hashing the rows of a table
   *
   * @param t The table containing rows to hash
   * @param stream The stream to construct this object on. Not the stream that will be used for
   * comparisons using this object.
   */
  row_hasher(table_view const& t, rmm::cuda_stream_view stream)
    : d_t(preprocessed_table::create(t, stream))
  {
  }

  /**
   * @brief Construct an owning object for hashing the rows of a table from an existing
   * preprocessed_table
   *
   * This constructor allows independently constructing a `preprocessed_table` and sharing it among
   * multiple `row_hasher` and `equality::self_comparator` objects.
   *
   * @param t A table preprocessed for hashing or equality.
   */
  row_hasher(std::shared_ptr<preprocessed_table> t) : d_t{std::move(t)} {}

  /**
   * @brief Get the hash operator to use on the device
   *
   * Returns a unary callable, `F`, where `F(i)` returns the hash value of row i.
   *
   * @tparam hash_function Hash functor to use for hashing elements
   * @tparam DeviceRowHasher The device row hasher type to use
   * @tparam Nullate A cudf::nullate type describing whether to check for nulls
   *
   * @param nullate Indicates if any input column contains nulls
   * @param seed The seed to use for the hash function
   * @return A hash operator to use on the device
   */
  template <
    template <typename> class hash_function = cudf::hashing::detail::default_hash,
    template <template <typename> class, typename> class DeviceRowHasher = device_row_hasher,
    typename Nullate>
  DeviceRowHasher<hash_function, Nullate> device_hasher(
    Nullate nullate                                                  = {},
    cuda::std::invoke_result_t<hash_function<int32_t>, int32_t> seed = DEFAULT_HASH_SEED) const
  {
    return DeviceRowHasher<hash_function, Nullate>(nullate, *d_t, seed);
  }

 private:
  std::shared_ptr<preprocessed_table> d_t;
};

}  // namespace detail::row::hash
}  // namespace CUDF_EXPORT cudf
