/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>

#include <cuda/std/utility>

namespace cudf {
namespace dictionary {
namespace detail {

/**
 * @brief Accessor functor for returning a dictionary key element in a dictionary iterator.
 *
 * @tparam KeyType The type of the dictionary's key element.
 */
template <typename KeyType>
struct dictionary_access_fn {
  dictionary_access_fn(column_device_view const& d_dictionary) : d_dictionary{d_dictionary} {}

  __device__ KeyType operator()(size_type idx) const
  {
    if (d_dictionary.is_null(idx)) return KeyType{};
    auto keys = d_dictionary.child(dictionary_column_view::keys_column_index);
    return keys.element<KeyType>(
      static_cast<size_type>(d_dictionary.element<dictionary32>(idx).value()));
  };

 private:
  column_device_view const d_dictionary;
};

/**
 * @brief Create dictionary iterator that produces key elements.
 *
 * The iterator returns `keys[indices[i]]` where the `keys` are the dictionary's key
 * elements and the `indices` are the dictionary's index elements.
 *
 * @throw cudf::logic_error if `dictionary_column` is not a dictionary column.
 *
 * @tparam KeyType The type of the dictionary's key element.
 * @param dictionary_column The dictionary device view to iterate.
 * @return Iterator
 */
template <typename KeyType>
auto make_dictionary_iterator(column_device_view const& dictionary_column)
{
  CUDF_EXPECTS(is_dictionary(dictionary_column.type()),
               "Dictionary iterator is only for dictionary columns");
  return cudf::detail::make_counting_transform_iterator(
    size_type{0}, dictionary_access_fn<KeyType>{dictionary_column});
}

/**
 * @brief Accessor functor for returning a dictionary pair iterator.
 *
 * @tparam KeyType The type of the dictionary's key element.
 *
 * @throw cudf::logic_error if `has_nulls==true` and `d_dictionary` is not nullable.
 */
template <typename KeyType>
struct dictionary_access_pair_fn {
  dictionary_access_pair_fn(column_device_view const& d_dictionary, bool has_nulls = true)
    : d_dictionary{d_dictionary}, has_nulls{has_nulls}
  {
    if (has_nulls) { CUDF_EXPECTS(d_dictionary.nullable(), "unexpected non-nullable column"); }
  }

  __device__ cuda::std::pair<KeyType, bool> operator()(size_type idx) const
  {
    if (has_nulls && d_dictionary.is_null(idx)) return {KeyType{}, false};
    auto keys = d_dictionary.child(dictionary_column_view::keys_column_index);
    return {keys.element<KeyType>(
              static_cast<size_type>(d_dictionary.element<dictionary32>(idx).value())),
            true};
  };

 private:
  column_device_view const d_dictionary;
  bool has_nulls;
};

/**
 * @brief Create dictionary iterator that produces key and valid element pair.
 *
 * The iterator returns a pair where the `first` value is
 * `dictionary_column.keys[dictionary_column.indices[i]]`
 * The `second` pair member is a `bool` which is set to
 * `dictionary_column.is_valid(i)`.
 *
 * @throw cudf::logic_error if `dictionary_column` is not a dictionary column.
 *
 * @tparam KeyType The type of the dictionary's key element.
 *
 * @param dictionary_column The dictionary device view to iterate.
 * @param has_nulls Set to `true` if the `dictionary_column` has nulls.
 * @return Pair iterator with `{value,valid}`
 */
template <typename KeyType>
auto make_dictionary_pair_iterator(column_device_view const& dictionary_column,
                                   bool has_nulls = true)
{
  CUDF_EXPECTS(is_dictionary(dictionary_column.type()),
               "Dictionary iterator is only for dictionary columns");
  return cudf::detail::make_counting_transform_iterator(
    0, dictionary_access_pair_fn<KeyType>{dictionary_column, has_nulls});
}

}  // namespace detail
}  // namespace dictionary
}  // namespace cudf
