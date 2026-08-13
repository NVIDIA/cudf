/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/iterator.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/iterator>
#include <cuda/std/utility>
#include <cuda_runtime.h>
#include <thrust/iterator/transform_iterator.h>

namespace CUDF_EXPORT cudf {

/**
 * @brief A non-owning, immutable view of device data that represents
 * a list of elements of arbitrary type (including further nested lists).
 */
class list_device_view {
  using lists_column_device_view = cudf::lists_column_device_view;

 public:
  list_device_view() = default;

  /**
   * @brief Constructs a list_device_view from a list column and index.
   *
   * @param lists_column list column device view containing the list to view
   * @param row_index index of the list row to view
   */
  __device__ inline list_device_view(lists_column_device_view const& lists_column,
                                     size_type const& row_index)
    : lists_column(lists_column), _row_index(row_index)
  {
    column_device_view const& offsets = lists_column.offsets();
    cudf_assert(row_index >= 0 && row_index < lists_column.size() && row_index < offsets.size() &&
                "row_index out of bounds");

    begin_offset = offsets.element<size_type>(row_index + lists_column.offset());
    cudf_assert(begin_offset >= 0 && begin_offset <= lists_column.child().size() &&
                "begin_offset out of bounds.");
    _size = offsets.element<size_type>(row_index + 1 + lists_column.offset()) - begin_offset;
  }

  ~list_device_view() = default;

  /**
   * @brief Fetches the offset in the list column's child that corresponds to
   * the element at the specified list index.
   *
   * Consider the following lists column:
   *  [
   *   [0,1,2],
   *   [3,4,5],
   *   [6,7,8]
   *  ]
   *
   * The list's internals would look like:
   *  offsets: [0, 3, 6, 9]
   *  child  : [0, 1, 2, 3, 4, 5, 6, 7, 8]
   *
   * The second list row (i.e. row_index=1) is [3,4,5].
   * The third element (i.e. idx=2) of the second list row is 5.
   *
   * The offset of this element as stored in the child column (i.e. 5)
   * may be fetched using this method.
   *
   * @param idx The list index of the element to fetch the offset for
   * @return The offset of the element at the specified list index
   */
  [[nodiscard]] __device__ inline size_type element_offset(size_type idx) const
  {
    cudf_assert(idx >= 0 && idx < size() && "idx out of bounds");
    return begin_offset + idx;
  }

  /**
   * @brief Fetches the element at the specified index within the list row.
   *
   * @tparam T The type of the list's element.
   * @param idx The index into the list row
   * @return The element at the specified index of the list row.
   */
  template <typename T>
  __device__ inline T element(size_type idx) const
  {
    return lists_column.child().element<T>(element_offset(idx));
  }

  /**
   * @brief Checks whether the element is null at the specified index in the list
   *
   * @param idx The index into the list row
   * @return `true` if the element is null at the specified index in the list row
   */
  [[nodiscard]] __device__ inline bool is_null(size_type idx) const
  {
    cudf_assert(idx >= 0 && idx < size() && "Index out of bounds.");
    auto element_offset = begin_offset + idx;
    return lists_column.child().is_null(element_offset);
  }

  /**
   * @brief Checks whether this list row is null.
   *
   * @return `true` if this list is null
   */
  [[nodiscard]] __device__ inline bool is_null() const { return lists_column.is_null(_row_index); }

  /**
   * @brief Fetches the number of elements in this list row.
   *
   * @return The number of elements in this list row
   */
  [[nodiscard]] __device__ inline size_type size() const { return _size; }

  /**
   * @brief Returns the row index of this list in the original lists column.
   *
   * @return The row index of this list
   */
  [[nodiscard]] __device__ inline size_type row_index() const { return _row_index; }

  /**
   * @brief Fetches the lists_column_device_view that contains this list.
   *
   * @return The lists_column_device_view that contains this list
   */
  [[nodiscard]] __device__ inline lists_column_device_view const& get_column() const
  {
    return lists_column;
  }

  template <typename T>
  struct pair_accessor;

  template <typename T>
  struct pair_rep_accessor;

  /// const pair iterator for the list
  template <typename T>
  using const_pair_iterator =
    thrust::transform_iterator<pair_accessor<T>, cuda::counting_iterator<cudf::size_type>>;

  /// const pair iterator type for the list
  template <typename T>
  using const_pair_rep_iterator =
    thrust::transform_iterator<pair_rep_accessor<T>, cuda::counting_iterator<cudf::size_type>>;

  /**
   * @brief Fetcher for a pair iterator to the first element in the list_device_view.
   *
   * Dereferencing the returned iterator yields a `cuda::std::pair<T, bool>`.
   *
   * If the element at index `i` is valid, then for `p = iter[i]`,
   *   1. `p.first` is the value of the element at `i`
   *   2. `p.second == true`
   *
   * If the element at index `i` is null,
   *   1. `p.first` is undefined
   *   2. `p.second == false`
   *
   * @return A pair iterator to the first element in the list_device_view and whether or not the
   * element is valid
   */
  template <typename T>
  [[nodiscard]] __device__ inline const_pair_iterator<T> pair_begin() const
  {
    return const_pair_iterator<T>{cuda::counting_iterator<size_type>{0}, pair_accessor<T>{*this}};
  }

  /**
   * @brief Fetcher for a pair iterator to one position past the last element in the
   * list_device_view.
   *
   * @return A pair iterator to one past the last element in the list_device_view and whether or not
   * that element is valid
   */
  template <typename T>
  [[nodiscard]] __device__ inline const_pair_iterator<T> pair_end() const
  {
    return const_pair_iterator<T>{cuda::counting_iterator<size_type>{size()},
                                  pair_accessor<T>{*this}};
  }

  /**
   * @brief Fetcher for a pair iterator to the first element in the list_device_view.
   *
   * Dereferencing the returned iterator yields a `cuda::std::pair<rep_type, bool>`,
   * where `rep_type` is `device_storage_type_t<T>`, the type used to store the value
   * on the device.
   *
   * If the element at index `i` is valid, then for `p = iter[i]`,
   *   1. `p.first` is the value of the element at `i`
   *   2. `p.second == true`
   *
   * If the element at index `i` is null,
   *   1. `p.first` is undefined
   *   2. `p.second == false`
   *
   * @return A pair iterator to the first element in the list_device_view and whether or not that
   * element is valid
   */
  template <typename T>
  [[nodiscard]] __device__ inline const_pair_rep_iterator<T> pair_rep_begin() const
  {
    return const_pair_rep_iterator<T>{cuda::counting_iterator<size_type>{0},
                                      pair_rep_accessor<T>{*this}};
  }

  /**
   * @brief Fetcher for a pair iterator to one position past the last element in the
   * list_device_view.
   *
   * @return A pair iterator one past the last element in the list_device_view and whether or not
   * that element is valid
   */
  template <typename T>
  [[nodiscard]] __device__ inline const_pair_rep_iterator<T> pair_rep_end() const
  {
    return const_pair_rep_iterator<T>{cuda::counting_iterator<size_type>{size()},
                                      pair_rep_accessor<T>{*this}};
  }

 private:
  lists_column_device_view const& lists_column;
  size_type _row_index{};  // Row index in the Lists column vector.
  size_type _size{};       // Number of elements in *this* list row.

  size_type begin_offset;  // Offset in list_column_device_view where this list begins.

  /**
   * @brief pair accessor for elements in a `list_device_view`
   *
   * This unary functor returns a pair of:
   *   1. data element at a specified index
   *   2. boolean validity flag for that element
   *
   * @tparam T The element-type of the list row
   */
  template <typename T>
  struct pair_accessor {
    list_device_view const& list;  ///< The list_device_view to access

    /**
     * @brief constructor
     *
     * @param _list The `list_device_view` whose rows are being accessed.
     */
    explicit CUDF_HOST_DEVICE inline pair_accessor(list_device_view const& _list) : list{_list} {}

    /**
     * @brief Accessor for the {data, validity} pair at the specified index
     *
     * @param i Index into the list_device_view
     * @return A pair of data element and its validity flag.
     */
    __device__ inline cuda::std::pair<T, bool> operator()(cudf::size_type i) const
    {
      return {list.element<T>(i), !list.is_null(i)};
    }
  };

  /**
   * @brief pair rep accessor for elements in a `list_device_view`
   *
   * Returns a `pair<rep_type, bool>`, where `rep_type` = `device_storage_type_t<T>`,
   * the type used to store the value on the device.
   *
   * This unary functor returns a pair of:
   *   1. rep element at a specified index
   *   2. boolean validity flag for that element
   *
   * @tparam T The element-type of the list row
   */
  template <typename T>
  struct pair_rep_accessor {
    list_device_view const& list;  ///< The list_device_view whose rows are being accessed

    using rep_type = device_storage_type_t<T>;  ///< The type used to store the value on the device

    /**
     * @brief constructor
     *
     * @param _list The `list_device_view` whose rows are being accessed.
     */
    explicit CUDF_HOST_DEVICE inline pair_rep_accessor(list_device_view const& _list) : list{_list}
    {
    }

    /**
     * @brief Accessor for the {rep_data, validity} pair at the specified index
     *
     * @param i Index into the list_device_view
     * @return A pair of data element and its validity flag.
     */
    __device__ inline cuda::std::pair<rep_type, bool> operator()(cudf::size_type i) const
    {
      return {get_rep<T>(i), !list.is_null(i)};
    }

   private:
    template <typename R>
    __device__ inline rep_type get_rep(cudf::size_type i) const
      requires(std::is_same_v<R, rep_type>)
    {
      return list.element<R>(i);
    }

    template <typename R>
    __device__ inline rep_type get_rep(cudf::size_type i) const
      requires(not std::is_same_v<R, rep_type>)
    {
      return list.element<R>(i).value();
    }
  };
};

/**
 * @brief Returns the size of the list by row index
 *
 */
struct list_size_functor {
  lists_column_device_view const d_column;  ///< The list column to access
  /**
   * @brief Constructor
   *
   * @param d_col The cudf::lists_column_device_view whose rows are being accessed
   */
  CUDF_HOST_DEVICE inline list_size_functor(lists_column_device_view const& d_col) : d_column(d_col)
  {
  }
  /**
   * @brief Returns size of the list by row index
   *
   * @param idx row index
   * @return size of the list
   */
  __device__ inline size_type operator()(size_type idx)
  {
    if (d_column.is_null(idx)) return size_type{0};
    return d_column.offset_at(idx + 1) - d_column.offset_at(idx);
  }
};

/**
 * @brief Makes an iterator that returns size of the list by row index
 *
 * Example:
 * For a list_column_device_view with 3 rows, `l = {[1, 2, 3], [4, 5], [6, 7, 8, 9]}`,
 * @code{.cpp}
 * auto it = make_list_size_iterator(l);
 * assert(it[0] == 3);
 * assert(it[1] == 2);
 * assert(it[2] == 4);
 * @endcode
 *
 * @param c The list_column_device_view to iterate over
 * @return An iterator that returns the size of the list by row index
 */
CUDF_HOST_DEVICE auto inline make_list_size_iterator(lists_column_device_view const& c)
{
  return detail::make_counting_transform_iterator(0, list_size_functor{c});
}

}  // namespace CUDF_EXPORT cudf
