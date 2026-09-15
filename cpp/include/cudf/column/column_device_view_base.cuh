/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_child_offsets.hpp>
#include <cudf/detail/offsets_iterator.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/algorithm>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

/**
 * @file column_device_view_base.cuh
 * @brief Column device view class definitions
 */

namespace CUDF_EXPORT cudf {

/**
 * @brief Indicates the presence of nulls at compile-time or runtime.
 *
 * If used at compile-time, this indicator can tell the optimizer
 * to include or exclude any null-checking clauses.
 *
 * @ingroup column_classes
 *
 */
struct nullate {
  struct YES : cuda::std::true_type {};
  struct NO : cuda::std::false_type {};
  /**
   * @brief `nullate::DYNAMIC` defers the determination of nullability to run time rather than
   * compile time. The calling code is responsible for specifying whether or not nulls are
   * present using the constructor parameter at run time.
   */
  struct DYNAMIC {
    DYNAMIC() = delete;
    /**
     * @brief Create a runtime nullate object.
     *
     * @see cudf::column_device_view::optional_begin for example usage
     *
     * @param b True if nulls are expected in the operation in which this
     *          object is applied.
     */
    constexpr explicit DYNAMIC(bool b) noexcept : value{b} {}
    /**
     * @brief Returns true if nulls are expected in the operation in which this object is applied.
     *
     * @return `true` if nulls are expected in the operation in which this object is applied,
     * otherwise false
     */
    CUDF_HOST_DEVICE constexpr operator bool() const noexcept { return value; }
    bool value;  ///< True if nulls are expected
  };
};

/**
 * @brief A type tag to specify that a column should be treated as a dictionary column
 *
 * @tparam IndexType The type of the dictionary indices
 * @tparam KeyType The type of the dictionary keys
 */
template <typename IndexType, typename KeyType>
  requires(is_index_type<IndexType>() && is_relationally_comparable<KeyType, KeyType>())
struct dictionary_element {
  using index_type = IndexType;  ///< The type of the dictionary indices
  using key_type   = KeyType;    ///< The type of the dictionary keys

  key_type key{};  ///< The dictionary key for this element
};

/**
 * @brief A type trait to determine if a type is a dictionary encoded type.
 * @tparam T The type to check
 */
template <typename T>
inline constexpr bool is_dictionary_encoded = false;

/**
 * @brief A type trait to determine if a type is a dictionary encoded type.
 * @tparam IndexType The type of the dictionary indices
 * @tparam KeyType The type of the dictionary keys
 */
template <typename IndexType, typename KeyType>
inline constexpr bool is_dictionary_encoded<dictionary_element<IndexType, KeyType>> = true;

namespace detail {
/**
 * @brief An immutable, non-owning view of device data as a column of elements
 * that is trivially copyable and usable in CUDA device code.
 */
class alignas(16) column_device_view_base {
 public:
  static constexpr size_type offsets_column_index =
    cudf::offsets_column_index;  ///< Child index of the offsets column

  column_device_view_base()                               = delete;
  ~column_device_view_base()                              = default;
  column_device_view_base(column_device_view_base const&) = default;  ///< Copy constructor
  column_device_view_base(column_device_view_base&&)      = default;  ///< Move constructor
  /**
   * @brief Copy assignment operator
   *
   * @return Reference to this object
   */
  column_device_view_base& operator=(column_device_view_base const&) = default;
  /**
   * @brief Move assignment operator
   *
   * @return Reference to this object (after transferring ownership)
   */
  column_device_view_base& operator=(column_device_view_base&&) = default;

  /**
   * @brief Returns pointer to the base device memory allocation casted to
   * the specified type.
   *
   * @note If `offset() == 0`, then `head<T>() == data<T>()`
   *
   * @note It should be rare to need to access the `head<T>()` allocation of
   * a column, and instead, accessing the elements should be done via
   *`data<T>()`.
   *
   * This function will only participate in overload resolution if `is_rep_layout_compatible<T>()`
   * or `std::is_same_v<T,void>` are true.
   *
   * @tparam T The type to cast to
   * @return Typed pointer to underlying data
   */
  template <typename T = void,
            CUDF_ENABLE_IF(cuda::std::is_same_v<T, void> or is_rep_layout_compatible<T>())>
  [[nodiscard]] CUDF_HOST_DEVICE T const* head() const noexcept
  {
    return static_cast<T const*>(_data);
  }

  /**
   * @brief Returns the underlying data casted to the specified type, plus the
   * offset.
   *
   * @note If `offset() == 0`, then `head<T>() == data<T>()`
   *
   * For columns with children, the pointer returned is undefined
   * and should not be used.
   *
   * This function does not participate in overload resolution if `is_rep_layout_compatible<T>` is
   * false.
   *
   * @tparam T The type to cast to
   * @return Typed pointer to underlying data, including the offset
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  [[nodiscard]] CUDF_HOST_DEVICE T const* data() const noexcept
  {
    return head<T>() + _offset;
  }

  /**
   * @brief Returns the number of elements in the column.
   *
   * @return The number of elements in the column
   */
  [[nodiscard]] CUDF_HOST_DEVICE size_type size() const noexcept { return _size; }

  /**
   * @brief Returns the element type
   *
   * @return The element type
   */
  [[nodiscard]] CUDF_HOST_DEVICE data_type type() const noexcept { return _type; }

  /**
   * @brief Indicates whether the column can contain null elements, i.e., if it
   *has an allocated bitmask.
   *
   * @note If `null_count() > 0`, this function must always return `true`.
   *
   * @return true The bitmask is allocated
   * @return false The bitmask is not allocated
   */
  [[nodiscard]] CUDF_HOST_DEVICE bool nullable() const noexcept { return nullptr != _null_mask; }

  /**
   * @brief Returns raw pointer to the underlying bitmask allocation.
   *
   * @note This function does *not* account for the `offset()`.
   *
   * @note If `null_count() == 0`, this may return `nullptr`.
   *
   * @return Raw pointer to the underlying bitmask allocation
   */
  [[nodiscard]] CUDF_HOST_DEVICE bitmask_type const* null_mask() const noexcept
  {
    return _null_mask;
  }

  /**
   * @brief Returns the index of the first element relative to the base memory
   * allocation, i.e., what is returned from `head<T>()`.
   *
   * @return The index of the first element relative to the `head<T>()`
   */
  [[nodiscard]] CUDF_HOST_DEVICE size_type offset() const noexcept { return _offset; }

  /**
   * @brief Returns whether the specified element holds a valid value (i.e., not
   * null).
   *
   * Checks first for the existence of the null bitmask. If `nullable() ==
   * false`, this function always returns true.
   *
   * @note If `nullable() == true` can be guaranteed, then it is more performant
   * to use `is_valid_nocheck()`.
   *
   * @param element_index The index of the element to query
   * @return true The element is valid
   * @return false The element is null
   */
  [[nodiscard]] __device__ bool is_valid(size_type element_index) const noexcept
  {
    return not nullable() or is_valid_nocheck(element_index);
  }

  /**
   * @brief Returns whether the specified element holds a valid value (i.e., not
   * null)
   *
   * This function does *not* verify the existence of the bitmask before
   * attempting to read it. Therefore, it is undefined behavior to call this
   * function if `nullable() == false`.
   *
   * @param element_index The index of the element to query
   * @return true The element is valid
   * @return false The element is null
   */
  [[nodiscard]] __device__ bool is_valid_nocheck(size_type element_index) const noexcept
  {
    return bit_is_set(_null_mask, offset() + element_index);
  }

  /**
   * @brief Returns whether the specified element is null.
   *
   * Checks first for the existence of the null bitmask. If `nullable() ==
   * false`, this function always returns false.
   *
   * @note If `nullable() == true` can be guaranteed, then it is more performant
   * to use `is_null_nocheck()`.
   *
   * @param element_index The index of the element to query
   * @return true The element is null
   * @return false The element is valid
   */
  [[nodiscard]] __device__ bool is_null(size_type element_index) const noexcept
  {
    return not is_valid(element_index);
  }

  /**
   * @brief Returns whether the specified element is null
   *
   * This function does *not* verify the existence of the bitmask before
   * attempting to read it. Therefore, it is undefined behavior to call this
   * function if `nullable() == false`.
   *
   * @param element_index The index of the element to query
   * @return true The element is null
   * @return false The element is valid
   */
  [[nodiscard]] __device__ bool is_null_nocheck(size_type element_index) const noexcept
  {
    return not is_valid_nocheck(element_index);
  }

  /**
   * @brief Returns the specified bitmask word from the `null_mask()`.
   *
   * @note It is undefined behavior to call this function if `nullable() ==
   * false`.
   *
   * @param word_index The index of the word to get
   * @return bitmask word for the given word_index
   */
  [[nodiscard]] __device__ bitmask_type get_mask_word(size_type word_index) const noexcept
  {
    return null_mask()[word_index];
  }

 protected:
  data_type _type{type_id::EMPTY};   ///< Element type
  cudf::size_type _size{};           ///< Number of elements
  void const* _data{};               ///< Pointer to device memory containing elements
  size_type _null_count{};           ///< The number of nulls
  bitmask_type const* _null_mask{};  ///< Pointer to device memory containing
                                     ///< bitmask representing null elements.
  size_type _offset{};               ///< Index position of the first element.
                                     ///< Enables zero-copy slicing
  void* _children{};                 ///< Array of `column_device_view` objects in device memory.
  size_type _num_children{};         ///< The number of child columns

  /**
   * @brief Constructs a column with the specified type, size, data, nullmask and offset.
   *
   * @param type The type of the column
   * @param size The number of elements in the column
   * @param data Pointer to device memory containing elements
   * @param null_count The number of nulls in the column
   * @param null_mask Pointer to device memory containing bitmask representing valid elements
   * @param offset Index position of the first element
   * @param children Pointer to device memory containing child `column_device_view` objects
   * @param num_children The number of child columns
   */
  CUDF_HOST_DEVICE column_device_view_base(data_type type,
                                           size_type size,
                                           void const* data,
                                           size_type null_count,
                                           bitmask_type const* null_mask,
                                           size_type offset,
                                           void* children,
                                           size_type num_children)
    : _type{type},
      _size{size},
      _data{data},
      _null_count{null_count},
      _null_mask{null_mask},
      _offset{offset},
      _children{children},
      _num_children{num_children}
  {
  }

  template <typename C, typename T, typename = void>
  struct has_element_accessor_impl : cuda::std::false_type {};

  template <typename C, typename T>
  struct has_element_accessor_impl<
    C,
    T,
    void_t<decltype(cuda::std::declval<C>().template element<T>(cuda::std::declval<size_type>()))>>
    : cuda::std::true_type {};
};
// @cond
// Forward declaration
template <typename T>
struct value_accessor;
template <typename T, typename Nullate>
struct optional_accessor;
template <typename T, bool has_nulls>
struct pair_accessor;
template <typename T, bool has_nulls>
struct pair_rep_accessor;
template <typename T>
struct mutable_value_accessor;
// @endcond
}  // namespace detail

/**
 * @brief An immutable, non-owning view of device data as a column of elements
 * that is trivially copyable and usable in CUDA device code and offline-compiled code (i.e. NVRTC).
 *
 * @ingroup column_classes
 */
class alignas(16) column_device_view_core : public detail::column_device_view_base {
 public:
  static constexpr bool is_mutable =
    false;  ///< Indicates whether this view allows mutation of the underlying data

  column_device_view_core()                               = delete;
  ~column_device_view_core()                              = default;
  column_device_view_core(column_device_view_core const&) = default;  ///< Copy constructor
  column_device_view_core(column_device_view_core&&)      = default;  ///< Move constructor
  /**
   * @brief Copy assignment operator
   *
   * @return Reference to this object
   */
  column_device_view_core& operator=(column_device_view_core const&) = default;
  /**
   * @brief Move assignment operator
   *
   * @return Reference to this object (after transferring ownership)
   */
  column_device_view_core& operator=(column_device_view_core&&) = default;

  /**
   * @brief Creates an instance of this class using the specified host memory
   * pointer (h_ptr) to store child objects and the device memory pointer
   * (d_ptr) as a base for any child object pointers.
   *
   * @param column Column view from which to create this instance.
   * @param h_ptr Host memory pointer on which to place any child data.
   * @param d_ptr Device memory pointer on which to base any child pointers.
   */
  column_device_view_core(column_view column, void* h_ptr, void* d_ptr);

  /**
   * @brief Get a new raw_column_device_view which is a slice of this column.
   *
   * Example:
   * @code{.cpp}
   * // column = raw_column_device_view([1, 2, 3, 4, 5, 6, 7])
   * auto c = column.slice(1, 3);
   * // c = raw_column_device_view([2, 3, 4])
   * auto c1 = column.slice(2, 3);
   * // c1 = raw_column_device_view([3, 4, 5])
   * @endcode
   *
   * @param offset The index of the first element in the slice
   * @param size The number of elements in the slice
   * @return A slice of this column
   */
  [[nodiscard]] CUDF_HOST_DEVICE column_device_view_core slice(size_type offset,
                                                               size_type size) const noexcept
  {
    return column_device_view_core{this->type(),
                                   size,
                                   this->head(),
                                   this->null_count(),
                                   this->null_mask(),
                                   this->offset() + offset,
                                   static_cast<column_device_view_core*>(_children),
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
    return data<T>()[element_index];
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
    size_type index       = element_index + offset();  // account for this view's _offset
    char const* d_strings = static_cast<char const*>(_data);
    auto const offsets    = child(offsets_column_index);
    auto const itr        = cudf::detail::input_offsetalator(offsets.head(), offsets.type());
    auto const offset     = itr[index];
    return string_view{d_strings + offset, static_cast<cudf::size_type>(itr[index + 1] - offset)};
  }

 public:
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
    using namespace numeric;
    using rep        = typename T::rep;
    auto const scale = scale_type{_type.scale()};
    return T{scaled_integer<rep>{data<rep>()[element_index], scale}};
  }

  /**
   * @brief Returns a decoded copy of the element at the specified index.
   *
   * If the element at the specified index is NULL, i.e.,
   * `is_null(element_index) == true`, then any attempt to use the result will
   * lead to undefined behavior.
   *
   * This function accounts for the offset.
   *
   * This function does not participate in overload resolution if `is_dictionary_encoded<T>` is
   * false.
   *
   * @tparam T The element type, e.g., `dictionary_element<int32_t, float>` for a dictionary column
   * with `int32_t` indices and `float` keys
   * @param element_index Position of the desired element
   * @return The element at the specified index
   */
  template <typename T, CUDF_ENABLE_IF(is_dictionary_encoded<T>)>
  [[nodiscard]] __device__ decltype(auto) element(size_type element_index) const noexcept
  {
    auto const& indices = child(dictionary_indices_column_index);
    auto const& keys    = child(dictionary_keys_column_index);
    auto const index    = indices.template element<typename T::index_type>(
      element_index + offset());  // account for this view's _offset
    return keys.template element<typename T::key_type>(index);
  }

  /**
   * @brief Returns a nullable element at the specified index. If the element is null, returns
   * `nullopt`.
   *
   * @param element_index Position of the desired element
   * @return `optional` containing the element at the specified index, or `nullopt` if the element
   * is null
   */
  template <typename T>
  [[nodiscard]] __device__ cuda::std::optional<T> nullable_element(
    size_type element_index) const noexcept
  {
    if (is_null(element_index)) { return cuda::std::nullopt; }
    return element<T>(element_index);
  }

  /**
   * @brief Returns the specified child
   *
   * @param child_index The index of the desired child
   * @return column_view The requested child `column_view`
   */
  [[nodiscard]] __device__ column_device_view_core child(size_type child_index) const noexcept
  {
    return static_cast<column_device_view_core*>(_children)[child_index];
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

  /**
   * @brief Returns the number of nulls in this column
   *
   * @return The number of nulls
   */
  [[nodiscard]] CUDF_HOST_DEVICE size_type null_count() const noexcept { return _null_count; }

 protected:
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
  CUDF_HOST_DEVICE column_device_view_core(data_type type,
                                           size_type size,
                                           void const* data,
                                           size_type null_count,
                                           bitmask_type const* null_mask,
                                           size_type offset,
                                           column_device_view_core* children,
                                           size_type num_children)
    : column_device_view_base(
        type, size, data, null_count, null_mask, offset, children, num_children)
  {
  }
};

/**
 * @brief A mutable, non-owning view of device data as a column of elements
 * that is trivially copyable and usable in CUDA device code and offline-compiled code (i.e.
 * NVRTC).
 *
 * @ingroup column_classes
 */
class alignas(16) mutable_column_device_view_core : public detail::column_device_view_base {
 public:
  static constexpr bool is_mutable =
    true;  ///< Indicates whether this view allows mutation of the underlying data

  mutable_column_device_view_core()  = delete;
  ~mutable_column_device_view_core() = default;
  mutable_column_device_view_core(mutable_column_device_view_core const&) =
    default;  ///< Copy constructor
  mutable_column_device_view_core(mutable_column_device_view_core&&) =
    default;  ///< Move constructor
  /**
   * @brief Copy assignment operator
   *
   * @return Reference to this object
   */
  mutable_column_device_view_core& operator=(mutable_column_device_view_core const&) = default;
  /**
   * @brief Move assignment operator
   *
   * @return Reference to this object (after transferring ownership)
   */
  mutable_column_device_view_core& operator=(mutable_column_device_view_core&&) = default;

  /**
   * @brief Returns pointer to the base device memory allocation casted to
   * the specified type.
   *
   * This function will only participate in overload resolution if `is_rep_layout_compatible<T>()`
   * or `std::is_same_v<T,void>` are true.
   *
   * @note If `offset() == 0`, then `head<T>() == data<T>()`
   *
   * @note It should be rare to need to access the `head<T>()` allocation of
   * a column, and instead, accessing the elements should be done via
   * `data<T>()`.
   *
   * @tparam T The type to cast to
   * @return Typed pointer to underlying data
   */
  template <typename T = void,
            CUDF_ENABLE_IF(cuda::std::is_same_v<T, void> or is_rep_layout_compatible<T>())>
  [[nodiscard]] CUDF_HOST_DEVICE T* head() const noexcept
  {
    return const_cast<T*>(detail::column_device_view_base::head<T>());
  }

  /**
   * @brief Returns the underlying data casted to the specified type, plus the
   * offset.
   *
   * This function does not participate in overload resolution if `is_rep_layout_compatible<T>` is
   * false.
   *
   * @note If `offset() == 0`, then `head<T>() == data<T>()`
   *
   * @tparam T The type to cast to
   * @return Typed pointer to underlying data, including the offset
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  [[nodiscard]] CUDF_HOST_DEVICE T* data() const noexcept
  {
    return const_cast<T*>(detail::column_device_view_base::data<T>());
  }

  /**
   * @brief Returns reference to element at the specified index.
   *
   * This function accounts for the offset.
   *
   * This function does not participate in overload resolution if `is_rep_layout_compatible<T>` is
   * false. Specializations of this function may exist for types `T` where
   *`is_rep_layout_compatible<T>` is false.
   *
   *
   * @tparam T The element type
   * @param element_index Position of the desired element
   * @return Reference to the element at the specified index
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  [[nodiscard]] __device__ T& element(size_type element_index) const noexcept
  {
    return data<T>()[element_index];
  }

  /**
   * @brief Returns `string_view` to the string element at the specified index.
   *
   * If the element at the specified index is NULL, i.e., `is_null(element_index)
   * == true`, then using the result is undefined behavior.
   *
   * This function accounts for the offset.
   *
   * @param element_index Position of the desired string element
   * @return string_view instance representing this element at this index
   */
  template <typename T, CUDF_ENABLE_IF(cuda::std::is_same_v<T, string_view>)>
  [[nodiscard]] __device__ T element(size_type element_index) const noexcept
  {
    size_type index       = element_index + offset();  // account for this view's _offset
    char const* d_strings = static_cast<char const*>(_data);
    auto const offsets    = child(offsets_column_index);
    auto const itr        = cudf::detail::input_offsetalator(offsets.head(), offsets.type());
    auto const offset     = itr[index];
    return string_view{d_strings + offset, static_cast<cudf::size_type>(itr[index + 1] - offset)};
  }

  /**
   * @brief Returns a `numeric::fixed_point` element at the specified index for a `fixed_point`
   * column.
   *
   * If the element at the specified index is NULL, i.e., `is_null(element_index) == true`,
   * then using the result is undefined behavior.
   *
   * @param element_index Position of the desired element
   * @return numeric::fixed_point representing the element at this index
   */
  template <typename T, CUDF_ENABLE_IF(cudf::is_fixed_point<T>())>
  [[nodiscard]] __device__ T element(size_type element_index) const noexcept
  {
    using namespace numeric;
    using rep        = typename T::rep;
    auto const scale = scale_type{_type.scale()};
    return T{scaled_integer<rep>{data<rep>()[element_index], scale}};
  }

  /**
   * @brief Returns a nullable element at the specified index. If the element is null, returns
   * `nullopt`.
   *
   * @param element_index Position of the desired element
   * @return `optional` containing the element at the specified index, or `nullopt` if the element
   * is null
   */
  template <typename T>
  [[nodiscard]] __device__ cuda::std::optional<T> nullable_element(
    size_type element_index) const noexcept
  {
    if (is_null(element_index)) { return cuda::std::nullopt; }
    return element<T>(element_index);
  }

  /**
   * @brief Assigns `value` to the element at `element_index`
   *
   * @tparam T The element type
   * @param element_index Position of the desired element
   * @param value The value to assign
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  __device__ void assign(size_type element_index, T value) const noexcept
  {
    data<T>()[element_index] = value;
  }

  /**
   * @brief Assigns `value` to the element at `element_index`.
   * @warning Expects that `value` has been scaled to the column's scale
   *
   * @tparam T The element type
   * @param element_index Position of the desired element
   * @param value The value to assign
   */
  template <typename T, CUDF_ENABLE_IF(is_fixed_point<T>())>
  __device__ void assign(size_type element_index, T value) const noexcept
  {
    // consider asserting that the scale matches
    using namespace numeric;
    using rep                  = typename T::rep;
    data<rep>()[element_index] = value.value();
  }

  /**
   * @brief Returns raw pointer to the underlying bitmask allocation.
   *
   * @note This function does *not* account for the `offset()`.
   *
   * @note If `null_count() == 0`, this may return `nullptr`.
   * @return Raw pointer to the underlying bitmask allocation
   */
  [[nodiscard]] CUDF_HOST_DEVICE bitmask_type* null_mask() const noexcept
  {
    return const_cast<bitmask_type*>(detail::column_device_view_base::null_mask());
  }

  /**
   * @brief Returns the specified child
   *
   * @param child_index The index of the desired child
   * @return The requested child `column_view`
   */
  [[nodiscard]] __device__ mutable_column_device_view_core
  child(size_type child_index) const noexcept
  {
    return static_cast<mutable_column_device_view_core*>(_children)[child_index];
  }

#ifdef __CUDACC__  // because set_bit in bit.hpp is wrapped with __CUDACC__
  /**
   * @brief Updates the null mask to indicate that the specified element is
   * valid
   *
   * @note This operation requires a global atomic operation. Therefore, it is
   * not recommended to use this function in performance critical regions. When
   * possible, it is more efficient to compute and update an entire word at
   * once using `set_mask_word`.
   *
   * @note It is undefined behavior to call this function if `nullable() ==
   * false`.
   *
   * @param element_index The index of the element to update
   */
  __device__ void set_valid(size_type element_index) const noexcept
  {
    return set_bit(null_mask(), element_index);
  }

  /**
   * @brief Updates the null mask to indicate that the specified element is null
   *
   * @note This operation requires a global atomic operation. Therefore, it is
   * not recommended to use this function in performance critical regions. When
   * possible, it is more efficient to compute and update an entire word at
   * once using `set_mask_word`.
   *
   * @note It is undefined behavior to call this function if `nullable() ==
   * false`.
   *
   * @param element_index The index of the element to update
   */
  __device__ void set_null(size_type element_index) const noexcept
  {
    return clear_bit(null_mask(), element_index);
  }

#endif

  /**
   * @brief Updates the specified bitmask word in the `null_mask()` with a
   * new word.
   *
   * @note It is undefined behavior to call this function if `nullable() ==
   * false`.
   *
   * @param word_index The index of the word to update
   * @param new_word The new bitmask word
   */
  __device__ void set_mask_word(size_type word_index, bitmask_type new_word) const noexcept
  {
    null_mask()[word_index] = new_word;
  }

 protected:
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
  CUDF_HOST_DEVICE mutable_column_device_view_core(data_type type,
                                                   size_type size,
                                                   void const* data,
                                                   bitmask_type const* null_mask,
                                                   size_type offset,
                                                   mutable_column_device_view_core* children,
                                                   size_type num_children)
    : column_device_view_base(type,
                              size,
                              data,
                              0,  // unused
                              null_mask,
                              offset,
                              children,
                              num_children)
  {
  }
};

}  // namespace CUDF_EXPORT cudf
