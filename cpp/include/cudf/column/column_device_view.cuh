/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/alignment.hpp>
#include <cudf/lists/list_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/struct_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/iterator>
#include <cuda/std/utility>
#include <thrust/iterator/transform_iterator.h>

#include <functional>

/**
 * @file column_device_view.cuh
 * @brief Column device view class definitions
 */

namespace CUDF_EXPORT cudf {

/**
 * @brief An immutable, non-owning view of device data as a column of elements
 * that is trivially copyable and usable in CUDA device code.
 *
 * @ingroup column_classes
 */
class alignas(16) column_device_view : public column_device_view_core {
 public:
  using base = column_device_view_core;  ///< Base type

  column_device_view()                          = delete;
  ~column_device_view()                         = default;
  column_device_view(column_device_view const&) = default;  ///< Copy constructor
  column_device_view(column_device_view&&)      = default;  ///< Move constructor
  /**
   * @brief Copy assignment operator
   *
   * @return Reference to this object
   */
  column_device_view& operator=(column_device_view const&) = default;
  /**
   * @brief Move assignment operator
   *
   * @return Reference to this object (after transferring ownership)
   */
  column_device_view& operator=(column_device_view&&) = default;

  /**
   * @brief Creates an instance of this class using the specified host memory
   * pointer (h_ptr) to store child objects and the device memory pointer
   * (d_ptr) as a base for any child object pointers.
   *
   * @param column Column view from which to create this instance.
   * @param h_ptr Host memory pointer on which to place any child data.
   * @param d_ptr Device memory pointer on which to base any child pointers.
   */
  column_device_view(column_view column, void* h_ptr, void* d_ptr);

  /**
   * @brief Get a new column_device_view which is a slice of this column.
   *
   * Example:
   * @code{.cpp}
   * // column = column_device_view([1, 2, 3, 4, 5, 6, 7])
   * auto c = column.slice(1, 3);
   * // c = column_device_view([2, 3, 4])
   * auto c1 = column.slice(2, 3);
   * // c1 = column_device_view([3, 4, 5])
   * @endcode
   *
   * @param offset The index of the first element in the slice
   * @param size The number of elements in the slice
   * @return A slice of this column
   */
  [[nodiscard]] CUDF_HOST_DEVICE column_device_view slice(size_type offset,
                                                          size_type size) const noexcept
  {
    return column_device_view{this->type(),
                              size,
                              this->head(),
                              this->null_count(),
                              this->null_mask(),
                              this->offset() + offset,
                              static_cast<column_device_view*>(_children),
                              this->num_child_columns()};
  }

  /**
   * @brief Returns a copy of the element at the specified index
   *
   * If the element at the specified index is NULL, i.e.,
   * `is_null(element_index) == true`, then any attempt to use the result will
   * lead to undefined behavior.
   *
   * This function accounts for the offset.
   *
   * This function does not participate in overload resolution if `is_rep_layout_compatible<T>` is
   * false. Specializations of this function may exist for types `T` where
   *`is_rep_layout_compatible<T>` is false.
   *
   * @tparam T The element type
   * @param element_index Position of the desired element
   * @return The element at the specified index
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  [[nodiscard]] __device__ T element(size_type element_index) const noexcept
  {
    return base::element<T>(element_index);
  }

  /**
   * @brief Returns `string_view` to the string element at the specified index.
   *
   * If the element at the specified index is NULL, i.e., `is_null(element_index)
   * == true`, then any attempt to use the result will lead to undefined behavior.
   *
   * This function accounts for the offset.
   *
   * @param element_index Position of the desired string element
   * @return string_view instance representing this element at this index
   */
  template <typename T, CUDF_ENABLE_IF(cuda::std::is_same_v<T, string_view>)>
  [[nodiscard]] __device__ T element(size_type element_index) const noexcept
  {
    return base::element<T>(element_index);
  }

  /**
   * @brief Returns a `numeric::fixed_point` element at the specified index for a `fixed_point`
   * column.
   *
   * If the element at the specified index is NULL, i.e., `is_null(element_index) == true`,
   * then any attempt to use the result will lead to undefined behavior.
   *
   * @param element_index Position of the desired element
   * @return numeric::fixed_point representing the element at this index
   */
  template <typename T, CUDF_ENABLE_IF(cudf::is_fixed_point<T>())>
  [[nodiscard]] __device__ T element(size_type element_index) const noexcept
  {
    return base::element<T>(element_index);
  }

 private:
  /**
   * @brief Dispatch functor for resolving the index value for a dictionary element.
   *
   * The basic dictionary elements are the indices which can be any index type.
   */
  struct index_element_fn {
    template <typename IndexType,
              CUDF_ENABLE_IF(is_index_type<IndexType>() and cuda::std::is_signed_v<IndexType>)>
    __device__ size_type operator()(column_device_view const& indices, size_type index)
    {
      return static_cast<size_type>(indices.element<IndexType>(index));
    }

    template <typename IndexType,
              typename... Args,
              CUDF_ENABLE_IF(not(is_index_type<IndexType>() and cuda::std::is_signed_v<IndexType>))>
    __device__ size_type operator()(Args&&... args)
    {
      CUDF_UNREACHABLE("dictionary indices must be a signed integral type");
    }
  };

 public:
  /**
   * @brief Returns `dictionary32` element at the specified index for a
   * dictionary column.
   *
   * `dictionary32` is a strongly typed wrapper around an `int32_t` value that holds the
   * offset into the dictionary keys for the specified element.
   *
   * For example, given a dictionary column `d` with:
   * ```c++
   * keys: {"foo", "bar", "baz"}
   * indices: {2, 0, 2, 1, 0}
   *
   * d.element<dictionary32>(0) == dictionary32{2};
   * d.element<dictionary32>(1) == dictionary32{0};
   * ```
   *
   * If the element at the specified index is NULL, i.e., `is_null(element_index) == true`,
   * then any attempt to use the result will lead to undefined behavior.
   *
   * This function accounts for the offset.
   *
   * @param element_index Position of the desired element
   * @return dictionary32 instance representing this element at this index
   */
  template <typename T, CUDF_ENABLE_IF(cuda::std::is_same_v<T, dictionary32>)>
  [[nodiscard]] __device__ T element(size_type element_index) const noexcept
  {
    size_type index    = element_index + offset();  // account for this view's _offset
    auto const indices = child(0);
    return dictionary32{type_dispatcher(indices.type(), index_element_fn{}, indices, index)};
  }

  /**
   * @brief For a given `T`, indicates if `column_device_view::element<T>()` has a valid overload.
   *
   * @tparam T The element type
   * @return `true` if `column_device_view::element<T>()` has a valid overload, `false` otherwise
   */
  template <typename T>
  CUDF_HOST_DEVICE static constexpr bool has_element_accessor()
  {
    return has_element_accessor_impl<column_device_view, T>::value;
  }

  /// Counting iterator
  using count_it = cuda::counting_iterator<size_type>;
  /**
   * @brief Iterator for navigating this column
   */
  template <typename T>
  using const_iterator = thrust::transform_iterator<detail::value_accessor<T>, count_it>;

  /**
   * @brief Return an iterator to the first element of the column.
   *
   * This iterator only supports columns where `has_nulls() == false`. Using it
   * with columns where `has_nulls() == true` will result in undefined behavior
   * when accessing null elements.
   *
   * This function does not participate in overload resolution if
   * `column_device_view::has_element_accessor<T>()` is false.
   *
   * For columns with null elements, use `make_null_replacement_iterator`.
   *
   * @tparam T Type of the elements in the column
   * @return An iterator to the first element of the column
   */
  template <typename T, CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
  [[nodiscard]] const_iterator<T> begin() const
  {
    return const_iterator<T>{count_it{0}, detail::value_accessor<T>{*this}};
  }

  /**
   * @brief Returns an iterator to the element following the last element of the column.
   *
   * This iterator only supports columns where `has_nulls() == false`. Using it
   * with columns where `has_nulls() == true` will result in undefined behavior
   * when accessing null elements.
   *
   * This function does not participate in overload resolution if
   * `column_device_view::has_element_accessor<T>()` is false.
   *
   * For columns with null elements, use `make_null_replacement_iterator`.
   *
   * @return An iterator to the element following the last element of the column
   */
  template <typename T, CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
  [[nodiscard]] const_iterator<T> end() const
  {
    return const_iterator<T>{count_it{size()}, detail::value_accessor<T>{*this}};
  }

  /**
   * @brief Optional iterator for navigating this column
   */
  template <typename T, typename Nullate>
  using const_optional_iterator =
    thrust::transform_iterator<detail::optional_accessor<T, Nullate>, count_it>;

  /**
   * @brief Pair iterator for navigating this column
   */
  template <typename T, bool has_nulls>
  using const_pair_iterator =
    thrust::transform_iterator<detail::pair_accessor<T, has_nulls>, count_it>;

  /**
   * @brief Pair rep iterator for navigating this column
   *
   * Each row value is accessed in its representative form.
   */
  template <typename T, bool has_nulls>
  using const_pair_rep_iterator =
    thrust::transform_iterator<detail::pair_rep_accessor<T, has_nulls>, count_it>;

  /**
   * @brief Return an optional iterator to the first element of the column.
   *
   * Dereferencing the returned iterator returns a `cuda::std::optional<T>`.
   *
   * The element of this iterator contextually converts to bool. The conversion returns true
   * if the object contains a value and false if it does not contain a value.
   *
   * Calling this method with `nullate::DYNAMIC` defers the assumption of nullability to
   * runtime with the caller indicating if the column has nulls. The `nullate::DYNAMIC` is
   * useful when an algorithm is going to execute on multiple iterators and all the combinations of
   * iterator types are not required at compile time.
   *
   * @code{.cpp}
   * template<typename T>
   * void some_function(cudf::column_view<T> const& col_view){
   *    auto d_col = cudf::column_device_view::create(col_view);
   *    // Create a `DYNAMIC` optional iterator
   *    auto optional_iterator =
   *       d_col->optional_begin<T>(cudf::nullate::DYNAMIC{col_view.has_nulls()});
   * }
   * @endcode
   *
   * Calling this method with `nullate::YES` means that the column supports nulls and
   * the optional returned might not contain a value.
   *
   * Calling this method with `nullate::NO` means that the column has no null values
   * and the optional returned will always contain a value.
   *
   * @code{.cpp}
   * template<typename T, bool has_nulls>
   * void some_function(cudf::column_view<T> const& col_view){
   *    auto d_col = cudf::column_device_view::create(col_view);
   *    if constexpr(has_nulls) {
   *      auto optional_iterator = d_col->optional_begin<T>(cudf::nullate::YES{});
   *      //use optional_iterator
   *    } else {
   *      auto optional_iterator = d_col->optional_begin<T>(cudf::nullate::NO{});
   *      //use optional_iterator
   *    }
   * }
   * @endcode
   *
   * This function does not participate in overload resolution if
   * `column_device_view::has_element_accessor<T>()` is false.
   *
   * @throws cudf::logic_error if the column is not nullable and `has_nulls` evaluates to true.
   * @throws cudf::logic_error if column datatype and Element type mismatch.
   *
   * @tparam T The type of elements in the column
   * @tparam Nullate A cudf::nullate type describing how to check for nulls
   * @param has_nulls  A cudf::nullate type describing how to check for nulls
   * @return An optional iterator to the first element of the column
   */
  template <typename T,
            typename Nullate,
            CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
  auto optional_begin(Nullate has_nulls) const
  {
    return const_optional_iterator<T, Nullate>{
      count_it{0}, detail::optional_accessor<T, Nullate>{*this, has_nulls}};
  }

  /**
   * @brief Return a pair iterator to the first element of the column.
   *
   * Dereferencing the returned iterator returns a `cuda::std::pair<T, bool>`.
   *
   * If an element at position `i` is valid (or `has_nulls == false`), then
   * for `p = *(iter + i)`, `p.first` contains the value of the element at `i`
   * and `p.second == true`.
   *
   * Else, if the element at `i` is null, then the value of `p.first` is
   * undefined and `p.second == false`.
   *
   * This function does not participate in overload resolution if
   * `column_device_view::has_element_accessor<T>()` is false.
   *
   * @throws cudf::logic_error if tparam `has_nulls == true` and
   * `nullable() == false`
   * @throws cudf::logic_error if column datatype and Element type mismatch.
   *
   * @return A pair iterator to the first element of the column
   */
  template <typename T,
            bool has_nulls,
            CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
  [[nodiscard]] const_pair_iterator<T, has_nulls> pair_begin() const
  {
    return const_pair_iterator<T, has_nulls>{count_it{0},
                                             detail::pair_accessor<T, has_nulls>{*this}};
  }

  /**
   * @brief Return a pair iterator to the first element of the column.
   *
   * Dereferencing the returned iterator returns a `cuda::std::pair<rep_type, bool>`,
   * where `rep_type` is `device_storage_type<T>`, the type used to store
   * the value on the device.
   *
   * If an element at position `i` is valid (or `has_nulls == false`), then
   * for `p = *(iter + i)`, `p.first` contains the value of the element at `i`
   * and `p.second == true`.
   *
   * Else, if the element at `i` is null, then the value of `p.first` is
   * undefined and `p.second == false`.
   *
   * This function does not participate in overload resolution if
   * `column_device_view::has_element_accessor<T>()` is false.
   *
   * @throws cudf::logic_error if tparam `has_nulls == true` and
   * `nullable() == false`
   * @throws cudf::logic_error if column datatype and Element type mismatch.
   *
   * @return A pair iterator to the first element of the column
   */
  template <typename T,
            bool has_nulls,
            CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
  [[nodiscard]] const_pair_rep_iterator<T, has_nulls> pair_rep_begin() const
  {
    return const_pair_rep_iterator<T, has_nulls>{count_it{0},
                                                 detail::pair_rep_accessor<T, has_nulls>{*this}};
  }

  /**
   * @brief Return an optional iterator to the element following the last element of the column.
   *
   * The returned iterator represents a `cuda::std::optional<T>` element.
   *
   * This function does not participate in overload resolution if
   * `column_device_view::has_element_accessor<T>()` is false.
   *
   * @throws cudf::logic_error if the column is not nullable and `has_nulls` is true
   * @throws cudf::logic_error if column datatype and Element type mismatch.
   *
   * @tparam T The type of elements in the column
   * @tparam Nullate A cudf::nullate type describing how to check for nulls
   * @param has_nulls  A cudf::nullate type describing how to check for nulls
   * @return An optional iterator to the element following the last element of the column
   */
  template <typename T,
            typename Nullate,
            CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
  auto optional_end(Nullate has_nulls) const
  {
    return const_optional_iterator<T, Nullate>{
      count_it{size()}, detail::optional_accessor<T, Nullate>{*this, has_nulls}};
  }

  /**
   * @brief Return a pair iterator to the element following the last element of the column.
   *
   * This function does not participate in overload resolution if
   * `column_device_view::has_element_accessor<T>()` is false.
   *
   * @throws cudf::logic_error if tparam `has_nulls == true` and
   * `nullable() == false`
   * @throws cudf::logic_error if column datatype and Element type mismatch.
   * @return A pair iterator to the element following the last element of the column
   */
  template <typename T,
            bool has_nulls,
            CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
  [[nodiscard]] const_pair_iterator<T, has_nulls> pair_end() const
  {
    return const_pair_iterator<T, has_nulls>{count_it{size()},
                                             detail::pair_accessor<T, has_nulls>{*this}};
  }

  /**
   * @brief Return a pair iterator to the element following the last element of the column.
   *
   * This function does not participate in overload resolution if
   * `column_device_view::has_element_accessor<T>()` is false.
   *
   * @throws cudf::logic_error if tparam `has_nulls == true` and
   * `nullable() == false`
   * @throws cudf::logic_error if column datatype and Element type mismatch.
   *
   * @return A pair iterator to the element following the last element of the column
   */
  template <typename T,
            bool has_nulls,
            CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
  [[nodiscard]] const_pair_rep_iterator<T, has_nulls> pair_rep_end() const
  {
    return const_pair_rep_iterator<T, has_nulls>{count_it{size()},
                                                 detail::pair_rep_accessor<T, has_nulls>{*this}};
  }

  /**
   * @brief Factory to construct a column view that is usable in device memory.
   *
   * Allocates and copies views of `source_view`'s children to device memory to
   * make them accessible in device code.
   *
   * If `source_view.num_children() == 0`, then no device memory is allocated.
   *
   * Returns a `std::unique_ptr<column_device_view>` with a custom deleter to
   * free the device memory allocated for the children.
   *
   * A `column_device_view` should be passed by value into GPU kernels.
   *
   * @param source_view The `column_view` to make usable in device code
   * @param stream CUDA stream used for device memory operations for children columns.
   * @param mr Device memory resource used to allocate the returned child-view storage
   * @return A `unique_ptr` to a `column_device_view` that makes the data from
   *`source_view` available in device memory.
   */
  static std::unique_ptr<column_device_view, std::function<void(column_device_view*)>> create(
    column_view source_view,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Destroy the `column_device_view` object.
   *
   * @note Does not free the column data, simply frees the device memory
   * allocated to hold the child views.
   */
  void destroy();

  /**
   * @brief Return the size in bytes of the amount of memory needed to hold a
   * device view of the specified column and its children.
   *
   * @param source_view The `column_view` to use for this calculation.
   * @return number of bytes to store device view in GPU memory
   */
  static std::size_t extent(column_view const& source_view);

  /**
   * @brief Returns the specified child
   *
   * @param child_index The index of the desired child
   * @return column_view The requested child `column_view`
   */
  [[nodiscard]] __device__ column_device_view child(size_type child_index) const noexcept
  {
    return static_cast<column_device_view*>(_children)[child_index];
  }

  /**
   * @brief Returns a span containing the children of this column
   *
   * @return A span containing the children of this column
   */
  [[nodiscard]] __device__ device_span<column_device_view const> children() const noexcept
  {
    return {static_cast<column_device_view*>(_children), static_cast<std::size_t>(_num_children)};
  }

  /**
   * @brief Returns the number of child columns
   *
   * @return The number of child columns
   */
  [[nodiscard]] CUDF_HOST_DEVICE size_type num_child_columns() const noexcept
  {
    return _num_children;
  }

 private:
  /**
   * @brief Creates an instance of this class using pre-existing device memory pointers to data,
   * nullmask, and offset.
   *
   * @param type The type of the column
   * @param size The number of elements in the column
   * @param data Pointer to the device memory containing the data
   * @param null_count The number of nulls in the column
   * @param null_mask Pointer to the device memory containing the null bitmask
   * @param offset The index of the first element in the column
   * @param children Pointer to the device memory containing child data
   * @param num_children The number of child columns
   */
  CUDF_HOST_DEVICE column_device_view(data_type type,
                                      size_type size,
                                      void const* data,
                                      size_type null_count,
                                      bitmask_type const* null_mask,
                                      size_type offset,
                                      column_device_view* children,
                                      size_type num_children)
    : column_device_view_core{
        type, size, data, null_count, null_mask, offset, children, num_children}
  {
  }

  /**
   * @brief Construct's a `column_device_view` from a `column_view` populating
   * all but the children.
   *
   * @note This constructor is for internal use only. To create a
   *`column_device_view` from a `column_view`, the
   *`column_device_view::create()` function should be used.
   *
   * @param source The `column_view` to use for this construction
   */
  column_device_view(column_view source);
};

/**
 * @brief A mutable, non-owning view of device data as a column of elements
 * that is trivially copyable and usable in CUDA device code.
 *
 * @ingroup column_classes
 */
class alignas(16) mutable_column_device_view : public mutable_column_device_view_core {
 public:
  using base = mutable_column_device_view_core;  ///< Base class

  mutable_column_device_view()                                  = delete;
  ~mutable_column_device_view()                                 = default;
  mutable_column_device_view(mutable_column_device_view const&) = default;  ///< Copy constructor
  mutable_column_device_view(mutable_column_device_view&&)      = default;  ///< Move constructor
  /**
   * @brief Copy assignment operator
   *
   * @return Reference to this object
   */
  mutable_column_device_view& operator=(mutable_column_device_view const&) = default;
  /**
   * @brief Move assignment operator
   *
   * @return Reference to this object (after transferring ownership)
   */
  mutable_column_device_view& operator=(mutable_column_device_view&&) = default;

  /**
   * @brief Creates an instance of this class using the specified host memory
   * pointer (h_ptr) to store child objects and the device memory pointer
   * (d_ptr) as a base for any child object pointers.
   *
   * @param column Column view from which to create this instance.
   * @param h_ptr Host memory pointer on which to place any child data.
   * @param d_ptr Device memory pointer on which to base any child pointers.
   */
  mutable_column_device_view(mutable_column_view column, void* h_ptr, void* d_ptr);

  /**
   * @brief Factory to construct a column view that is usable in device memory.
   *
   * Allocates and copies views of `source_view`'s children to device memory to
   * make them accessible in device code.
   *
   * If `source_view.num_children() == 0`, then no device memory is allocated.
   *
   * Returns a `std::unique_ptr<mutable_column_device_view>` with a custom
   * deleter to free the device memory allocated for the children.
   *
   * A `mutable_column_device_view` should be passed by value into GPU kernels.
   *
   * @param source_view The `column_view` to make usable in device code
   * @param stream CUDA stream used for device memory operations for children columns.
   * @param mr Device memory resource used to allocate the returned child-view storage
   * @return A `unique_ptr` to a `mutable_column_device_view` that makes the
   * data from `source_view` available in device memory.
   */
  static std::unique_ptr<mutable_column_device_view,
                         std::function<void(mutable_column_device_view*)>>
  create(mutable_column_view source_view,
         rmm::cuda_stream_view stream      = cudf::get_default_stream(),
         rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Returns reference to element at the specified index.
   *
   * If the element at the specified index is NULL, i.e.,
   * `is_null(element_index) == true`, then any attempt to use the result will
   * lead to undefined behavior.
   *
   * This function accounts for the offset.
   *
   * This function does not participate in overload resolution if `is_rep_layout_compatible<T>` is
   * false. Specializations of this function may exist for types `T` where
   *`is_rep_layout_compatible<T>` is false.
   *
   * @tparam T The element type
   * @param element_index Position of the desired element
   * @return Reference to the element at the specified index
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  [[nodiscard]] __device__ T& element(size_type element_index) const noexcept
  {
    return base::element<T>(element_index);
  }

  /**
   * @brief For a given `T`, indicates if `mutable_column_device_view::element<T>()` has a valid
   * overload.
   *
   * @return `true` if `mutable_column_device_view::element<T>()` has a valid overload, `false`
   */
  template <typename T>
  CUDF_HOST_DEVICE static constexpr bool has_element_accessor()
  {
    return has_element_accessor_impl<mutable_column_device_view, T>::value;
  }

  /// Counting iterator
  using count_it = cuda::counting_iterator<size_type>;
  /**
   * @brief Iterator for navigating this column
   */
  template <typename T>
  using iterator = thrust::transform_iterator<detail::mutable_value_accessor<T>, count_it>;

  /**
   * @brief Return first element (accounting for offset) after underlying data
   * is casted to the specified type.
   *
   * This function does not participate in overload resolution if
   * `mutable_column_device_view::has_element_accessor<T>()` is false.
   *
   * @tparam T The desired type
   * @return Pointer to the first element after casting
   */
  template <typename T, CUDF_ENABLE_IF(mutable_column_device_view::has_element_accessor<T>())>
  iterator<T> begin()
  {
    return iterator<T>{count_it{0}, detail::mutable_value_accessor<T>{*this}};
  }

  /**
   * @brief Return one past the last element after underlying data is casted to
   * the specified type.
   *
   * This function does not participate in overload resolution if
   * `mutable_column_device_view::has_element_accessor<T>()` is false.
   *
   * @tparam T The desired type
   * @return Pointer to one past the last element after casting
   */
  template <typename T, CUDF_ENABLE_IF(mutable_column_device_view::has_element_accessor<T>())>
  iterator<T> end()
  {
    return iterator<T>{count_it{size()}, detail::mutable_value_accessor<T>{*this}};
  }

  /**
   * @brief Returns the specified child
   *
   * @param child_index The index of the desired child
   * @return The requested child `column_view`
   */
  [[nodiscard]] __device__ mutable_column_device_view child(size_type child_index) const noexcept
  {
    return static_cast<mutable_column_device_view*>(_children)[child_index];
  }

  /**
   * @brief Return the size in bytes of the amount of memory needed to hold a
   * device view of the specified column and its children.
   *
   * @param source_view The `column_view` to use for this calculation.
   * @return The size in bytes of the amount of memory needed to hold a
   * device view of the specified column and its children
   */
  static std::size_t extent(mutable_column_view source_view);

  /**
   * @brief Factory to construct a mutable column view that is usable in device memory from
   * pre-existing device memory pointers to data, nullmask, and offset.
   *
   * @param type The type of the column
   * @param size The number of elements in the column
   * @param data Pointer to the device memory containing the data
   * @param null_mask Pointer to the device memory containing the null bitmask
   * @param offset The index of the first element in the column
   * @param children Pointer to the device memory containing child data
   * @param num_children The number of child columns
   * @return A `mutable_column_device_view` instance representing the specified column data
   * in device memory
   */
  [[nodiscard]] static auto create(data_type type,
                                   size_type size,
                                   void const* data,
                                   bitmask_type const* null_mask,
                                   size_type offset,
                                   mutable_column_device_view* children,
                                   size_type num_children)
  {
    return mutable_column_device_view{type, size, data, null_mask, offset, children, num_children};
  }

  /**
   * @brief Destroy the `mutable_column_device_view` object.
   *
   * @note Does not free the column data, simply frees the device memory
   * allocated to hold the child views.
   */
  void destroy();

 private:
  /**
   * @brief Construct's a `mutable_column_device_view` from a
   *`mutable_column_view` populating all but the children.
   *
   * @note This constructor is for internal use only. To create a
   *`mutable_column_device_view` from a `column_view`, the
   *`mutable_column_device_view::create()` function should be used.
   */
  mutable_column_device_view(mutable_column_view source);

  /**
   * @brief Creates an instance of this class using pre-existing device memory pointers to data,
   * nullmask, and offset.
   *
   * @param type The type of the column
   * @param size The number of elements in the column
   * @param data Pointer to the device memory containing the data
   * @param null_mask Pointer to the device memory containing the null bitmask
   * @param offset The index of the first element in the column
   * @param children Pointer to the device memory containing child data
   * @param num_children The number of child columns
   */
  CUDF_HOST_DEVICE mutable_column_device_view(data_type type,
                                              size_type size,
                                              void const* data,
                                              bitmask_type const* null_mask,
                                              size_type offset,
                                              mutable_column_device_view* children,
                                              size_type num_children)
    : mutable_column_device_view_core{type, size, data, null_mask, offset, children, num_children}
  {
  }
};

namespace detail {

/**
 * @brief value accessor of column without null bitmask
 *
 * A unary functor returns scalar value at `id`.
 * `operator() (cudf::size_type id)` computes `element`
 * This functor is only allowed for non-nullable columns.
 *
 * the return value for element `i` will return `column[i]`
 *
 * @throws cudf::logic_error if the column is nullable.
 * @throws cudf::logic_error if column datatype and template T type mismatch.
 *
 * @tparam T The type of elements in the column
 */
template <typename T>
struct value_accessor {
  column_device_view const col;  ///< column view of column in device

  /**
   * @brief constructor
   *
   * @param[in] _col column device view of cudf column
   */
  value_accessor(column_device_view const& _col) : col{_col}
  {
    CUDF_EXPECTS(type_id_matches_device_storage_type<T>(col.type().id()), "the data type mismatch");
  }

  /**
   * @brief Returns the value of element at index `i`
   * @param[in] i index of element
   * @return value of element at index `i`
   */
  __device__ T operator()(cudf::size_type i) const { return col.element<T>(i); }
};

/**
 * @brief optional accessor of a column
 *
 *
 * The optional_accessor always returns a `cuda::std::optional` of `column[i]`. The validity
 * of the optional is determined by the `Nullate` parameter which may be one of the following:
 *
 * - `nullate::YES` means that the column supports nulls and the optional returned
 *    might be valid or invalid.
 *
 * - `nullate::NO` means the caller attests that the column has no null values,
 *    no checks will occur and `cuda::std::optional{column[i]}` will be
 *    return for each `i`.
 *
 * - `nullate::DYNAMIC` defers the assumption of nullability to runtime and the caller
 *    specifies if the column has nulls at runtime.
 *    For `DYNAMIC{true}` the return value will be `cuda::std::optional{column[i]}` if
 *      element `i` is not null and `cuda::std::optional{}` if element `i` is null.
 *    For `DYNAMIC{false}` the return value will always be `cuda::std::optional{column[i]}`.
 *
 * @throws cudf::logic_error if column datatype and template T type mismatch.
 * @throws cudf::logic_error if the column is not nullable and `with_nulls` evaluates to true
 *
 * @tparam T The type of elements in the column
 * @tparam Nullate A cudf::nullate type describing how to check for nulls.
 */
template <typename T, typename Nullate>
struct optional_accessor {
  column_device_view const col;  ///< column view of column in device

  /**
   * @brief Constructor
   *
   * @param _col Column on which to iterator over its elements.
   * @param with_nulls Indicates if the `col` should be checked for nulls.
   */
  optional_accessor(column_device_view const& _col, Nullate with_nulls)
    : col{_col}, has_nulls{with_nulls}
  {
    CUDF_EXPECTS(type_id_matches_device_storage_type<T>(col.type().id()), "the data type mismatch");
    if (with_nulls) { CUDF_EXPECTS(_col.nullable(), "Unexpected non-nullable column."); }
  }

  /**
   * @brief Returns a `cuda::std::optional` of `column[i]`.
   *
   * @param i The index of the element to return
   * @return A `cuda::std::optional` that contains the value of `column[i]` is not null. If that
   * element is null, the resulting optional will not contain a value.
   */
  __device__ inline cuda::std::optional<T> operator()(cudf::size_type i) const
  {
    if (has_nulls) {
      return (col.is_valid_nocheck(i)) ? cuda::std::optional<T>{col.element<T>(i)}
                                       : cuda::std::optional<T>{cuda::std::nullopt};
    }
    return cuda::std::optional<T>{col.element<T>(i)};
  }

  Nullate has_nulls{};  ///< Indicates if the `col` should be checked for nulls.
};

/**
 * @brief pair accessor of column with/without null bitmask
 *
 * A unary functor returns pair with scalar value at `id` and boolean validity
 * `operator() (cudf::size_type id)` computes `element`  and
 * returns a `pair(element, validity)`
 *
 * the return value for element `i` will return `pair(column[i], validity)`
 * `validity` is `true` if `has_nulls=false`.
 * `validity` is validity of the element at `i` if `has_nulls=true` and the
 * column is nullable.
 *
 * @throws cudf::logic_error if `has_nulls==true` and the column is not
 * nullable.
 * @throws cudf::logic_error if column datatype and template T type mismatch.
 *
 * @tparam T The type of elements in the column
 * @tparam has_nulls boolean indicating to treat the column is nullable
 */
template <typename T, bool has_nulls = false>
struct pair_accessor {
  column_device_view const col;  ///< column view of column in device

  /**
   * @brief constructor
   *
   * @param[in] _col column device view of cudf column
   */
  pair_accessor(column_device_view const& _col) : col{_col}
  {
    CUDF_EXPECTS(type_id_matches_device_storage_type<T>(col.type().id()), "the data type mismatch");
    if (has_nulls) { CUDF_EXPECTS(_col.nullable(), "Unexpected non-nullable column."); }
  }

  /**
   * @brief Pair accessor
   *
   * @param[in] i index of the element
   * @return pair(element, validity)
   */
  __device__ inline cuda::std::pair<T, bool> operator()(cudf::size_type i) const
  {
    return {col.element<T>(i), (has_nulls ? col.is_valid_nocheck(i) : true)};
  }
};

/**
 * @brief pair accessor of column with/without null bitmask
 *
 * A unary functor returns pair with representative scalar value at `id` and boolean validity
 * `operator() (cudf::size_type id)` computes `element`  and
 * returns a `pair(element, validity)`
 *
 * the return value for element `i` will return `pair(column[i], validity)`
 * `validity` is `true` if `has_nulls=false`.
 * `validity` is validity of the element at `i` if `has_nulls=true` and the
 * column is nullable.
 *
 * @throws cudf::logic_error if `has_nulls==true` and the column is not
 * nullable.
 * @throws cudf::logic_error if column datatype and template T type mismatch.
 *
 * @tparam T The type of elements in the column
 * @tparam has_nulls boolean indicating to treat the column is nullable
 */
template <typename T, bool has_nulls = false>
struct pair_rep_accessor {
  column_device_view const col;  ///< column view of column in device

  using rep_type = device_storage_type_t<T>;  ///< representation type

  /**
   * @brief constructor
   *
   * @param[in] _col column device view of cudf column
   */
  pair_rep_accessor(column_device_view const& _col) : col{_col}
  {
    CUDF_EXPECTS(type_id_matches_device_storage_type<T>(col.type().id()), "the data type mismatch");
    if (has_nulls) { CUDF_EXPECTS(_col.nullable(), "Unexpected non-nullable column."); }
  }

  /**
   * @brief Pair accessor
   *
   * @param[in] i index of element to access
   * @return pair of element and validity
   */
  __device__ inline cuda::std::pair<rep_type, bool> operator()(cudf::size_type i) const
  {
    return {get_rep<T>(i), (has_nulls ? col.is_valid_nocheck(i) : true)};
  }

 private:
  template <typename R>
  [[nodiscard]] __device__ inline auto get_rep(cudf::size_type i) const
    requires(std::is_same_v<R, rep_type>)
  {
    return col.element<R>(i);
  }

  template <typename R>
  [[nodiscard]] __device__ inline auto get_rep(cudf::size_type i) const
    requires(not std::is_same_v<R, rep_type>)
  {
    return col.element<R>(i).value();
  }
};

/**
 * @brief Mutable value accessor of column without null bitmask
 *
 * A unary functor that accepts an index and returns a reference to the element at that index in the
 * column.
 *
 * @throws cudf::logic_error if the column is nullable
 * @throws cudf::logic_error if column datatype and template T type mismatch
 *
 * @tparam T The type of elements in the column
 */
template <typename T>
struct mutable_value_accessor {
  mutable_column_device_view col;  ///< mutable column view of column in device

  /**
   * @brief Constructor
   *
   * @param[in] _col mutable column device view of cudf column
   */
  mutable_value_accessor(mutable_column_device_view& _col) : col{_col}
  {
    CUDF_EXPECTS(type_id_matches_device_storage_type<T>(col.type().id()), "the data type mismatch");
  }

  /**
   * @brief Accessor
   *
   * @param i index of element to access
   * @return reference to element at `i`
   */
  __device__ T& operator()(cudf::size_type i) { return col.element<T>(i); }
};

/**
 * @brief Helper function for use by column_device_view and mutable_column_device_view
 * constructors to build device_views from views.
 *
 * It is used to build the array of child columns in device memory. Since child columns can
 * also have child columns, this uses recursion to build up the flat device buffer to contain
 * all the children and set the member pointers appropriately.
 *
 * This is accomplished by laying out all the children and grand-children into a flat host
 * buffer first but also keep a running device pointer to use when setting the
 * d_children array result.
 *
 * This function is provided both the host pointer in which to insert its children (and
 * by recursion its grand-children) and the device pointer to be used when calculating
 * ultimate device pointer for the d_children member.
 *
 * @tparam ColumnView is either column_view or mutable_column_view
 * @tparam ColumnDeviceView is either column_device_view or mutable_column_device_view
 *
 * @param child_begin Iterator pointing to begin of child columns to make into a device view
 * @param child_end   Iterator pointing to end   of child columns to make into a device view
 * @param h_ptr The host memory where to place any child data
 * @param d_ptr The device pointer for calculating the d_children member of any child data
 * @return The device pointer to be used for the d_children member of the given column
 */
template <typename ColumnDeviceView, typename ColumnViewIterator>
ColumnDeviceView* child_columns_to_device_array(ColumnViewIterator child_begin,
                                                ColumnViewIterator child_end,
                                                void* h_ptr,
                                                void* d_ptr)
{
  ColumnDeviceView* d_children = detail::align_ptr_for_type<ColumnDeviceView>(d_ptr);
  auto num_children            = std::distance(child_begin, child_end);
  if (num_children > 0) {
    // The beginning of the memory must be the fixed-sized ColumnDeviceView
    // struct objects in order for d_children to be used as an array.
    auto h_column = detail::align_ptr_for_type<ColumnDeviceView>(h_ptr);
    auto d_column = d_children;

    // Any child data is assigned past the end of this array: h_end and d_end.
    auto h_end = reinterpret_cast<int8_t*>(h_column + num_children);
    auto d_end = reinterpret_cast<int8_t*>(d_column + num_children);
    std::for_each(child_begin, child_end, [&](auto const& col) {
      // inplace-new each child into host memory
      new (h_column) ColumnDeviceView(col, h_end, d_end);
      h_column++;  // advance to next child
      // update the pointers for holding this child column's child data
      auto col_child_data_size = ColumnDeviceView::extent(col) - sizeof(ColumnDeviceView);
      h_end += col_child_data_size;
      d_end += col_child_data_size;
    });
  }
  return d_children;
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
