/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/span>

#include <limits>
#include <type_traits>
#include <vector>

/**
 * @file column_view.hpp
 * @brief column view class definitions
 */
namespace CUDF_EXPORT cudf {
namespace detail {
/**
 * @brief A non-owning, immutable view of device data as a column of elements,
 * some of which may be null as indicated by a bitmask.
 *
 * A `column_view_base` can be constructed implicitly from a `cudf::column`, or
 *may be constructed explicitly from a pointer to pre-existing device memory.
 *
 * Unless otherwise noted, the memory layout of the `column_view_base`'s data
 *and bitmask is expected to adhere to the Arrow Physical Memory Layout
 * Specification: https://arrow.apache.org/docs/format/Columnar.html#format-columnar
 *
 * Because `column_view_base` is non-owning, no device memory is allocated nor
 *freed when `column_view_base` objects are created or destroyed.
 *
 * To enable zero-copy slicing, a `column_view_base` has an `offset` that
 *indicates the index of the first element in the column relative to the base
 *device memory allocation. By default, `offset()` is zero.
 */
class column_view_base {
 public:
  /**
   * @brief Returns pointer to the base device memory allocation casted to
   * the specified type.
   *
   * @note If `offset() == 0`, then `head<T>() == data<T>()`
   *
   * @note It should be rare to need to access the `head<T>()` allocation of
   *a column, and instead, accessing the elements should be done via
   *`data<T>()`.
   *
   * This function will only participate in overload resolution if `is_rep_layout_compatible<T>()`
   * or `std::is_same_v<T,void>` are true.
   *
   * @tparam T The type to cast to
   * @return Typed pointer to underlying data
   */
  template <typename T = void,
            CUDF_ENABLE_IF(std::is_same_v<T, void> or is_rep_layout_compatible<T>())>
  T const* head() const noexcept
  {
    return static_cast<T const*>(get_data());
  }

  /**
   * @brief Returns the underlying data casted to the specified type, plus the
   * offset.
   *
   * @note If `offset() == 0`, then `head<T>() == data<T>()`
   *
   * This function does not participate in overload resolution if `is_rep_layout_compatible<T>` is
   * false.
   *
   * @tparam T The type to cast to
   * @return Typed pointer to underlying data, including the offset
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  T const* data() const noexcept
  {
    return head<T>() + _offset;
  }

  /**
   * @brief Return first element (accounting for offset) after underlying data
   * is casted to the specified type.
   *
   * This function does not participate in overload resolution if `is_rep_layout_compatible<T>` is
   * false.
   *
   * @tparam T The desired type
   * @return Pointer to the first element after casting
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  T const* begin() const noexcept
  {
    return data<T>();
  }

  /**
   * @brief Return one past the last element after underlying data is casted to
   * the specified type.
   *
   * This function does not participate in overload resolution if `is_rep_layout_compatible<T>` is
   * false.
   *
   * @tparam T The desired type
   * @return Pointer to one past the last element after casting
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  T const* end() const noexcept
  {
    return begin<T>() + size();
  }

  /**
   * @brief Returns the number of elements in the column
   *
   * @return The number of elements in the column
   */
  [[nodiscard]] size_type size() const noexcept { return _size; }

  /**
   * @brief Returns true if `size()` returns zero, or false otherwise
   *
   * @return True if `size()` returns zero, or false otherwise
   */
  [[nodiscard]] bool is_empty() const noexcept { return size() == 0; }

  /**
   * @brief Returns the element `data_type`
   *
   * @return The `data_type` of the elements in the column
   */
  [[nodiscard]] data_type type() const noexcept { return _type; }

  /**
   * @brief Indicates if the column can contain null elements, i.e., if it has
   * an allocated bitmask.
   *
   * @note If `null_count() > 0`, this function must always return `true`.
   *
   * @return true The bitmask is allocated
   * @return false The bitmask is not allocated
   */
  [[nodiscard]] bool nullable() const noexcept { return nullptr != _null_mask; }

  /**
   * @brief Returns the count of null elements
   *
   * @return The count of null elements
   */
  [[nodiscard]] size_type null_count() const { return _null_count; }

  /**
   * @brief Returns the count of null elements in the range [begin, end)
   *
   * @note If `null_count() != 0`, every invocation of `null_count(begin, end)`
   * will recompute the count of null elements indicated by the `null_mask` in
   * the range [begin, end).
   *
   * @throws cudf::logic_error for invalid range (if `begin < 0`,
   * `begin > end`, `begin >= size()`, or `end > size()`).
   *
   * @param[in] begin The starting index of the range (inclusive).
   * @param[in] end The index of the last element in the range (exclusive).
   * @param[in] stream CUDA stream used for device memory operations and kernel launches
   * @return The count of null elements in the given range
   */
  [[nodiscard]] size_type null_count(
    size_type begin,
    size_type end,
    rmm::cuda_stream_view stream = cudf::get_default_stream()) const;

  /**
   * @brief Indicates if the column contains null elements,
   * i.e., `null_count() > 0`
   *
   * @return true One or more elements are null
   * @return false All elements are valid
   */
  [[nodiscard]] bool has_nulls() const { return null_count() > 0; }

  /**
   * @brief Indicates if the column contains null elements in the range
   * [begin, end), i.e., `null_count(begin, end) > 0`
   *
   * @throws cudf::logic_error for invalid range (if `begin < 0`,
   * `begin > end`, `begin >= size()`, or `end > size()`).
   *
   * @param begin The starting index of the range (inclusive).
   * @param end The index of the last element in the range (exclusive).
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return true One or more elements are null in the range [begin, end)
   * @return false All elements are valid in the range [begin, end)
   */
  [[nodiscard]] bool has_nulls(size_type begin,
                               size_type end,
                               rmm::cuda_stream_view stream = cudf::get_default_stream()) const
  {
    return null_count(begin, end, stream) > 0;
  }

  /**
   * @brief Returns raw pointer to the underlying bitmask allocation.
   *
   * @note This function does *not* account for the `offset()`.
   *
   * @note If `null_count() == 0`, this may return `nullptr`.
   * @return Raw pointer to the bitmask
   */
  [[nodiscard]] bitmask_type const* null_mask() const noexcept { return _null_mask; }

  /**
   * @brief Returns the index of the first element relative to the base memory
   * allocation, i.e., what is returned from `head<T>()`.
   *
   * @return The index of the first element relative to `head<T>()`
   */
  [[nodiscard]] size_type offset() const noexcept { return _offset; }

 protected:
  /**
   * @brief Returns pointer to the base device memory allocation.
   *
   * The primary purpose of this function is to allow derived classes to
   * override the fundamental properties of memory accesses without needing to
   * change all of the different accessors for the underlying pointer.
   *
   * @return Typed pointer to underlying data
   */
  [[nodiscard]] virtual void const* get_data() const noexcept { return _data; }

  data_type _type{type_id::EMPTY};   ///< Element type
  size_type _size{};                 ///< Number of elements
  void const* _data{};               ///< Pointer to device memory containing elements
  bitmask_type const* _null_mask{};  ///< Pointer to device memory containing
                                     ///< bitmask representing null elements.
                                     ///< Optional if `null_count() == 0`
  mutable size_type _null_count{};   ///< The number of null elements
  size_type _offset{};               ///< Index position of the first element.
                                     ///< Enables zero-copy slicing

  column_view_base()                        = default;
  virtual ~column_view_base()               = default;
  column_view_base(column_view_base const&) = default;  ///< Copy constructor
  column_view_base(column_view_base&&)      = default;  ///< Move constructor
  /**
   * @brief Copy assignment operator
   *
   * @return Reference to this object
   */
  column_view_base& operator=(column_view_base const&) = default;
  /**
   * @brief Move assignment operator
   *
   * @return Reference to this object (after transferring ownership)
   */
  column_view_base& operator=(column_view_base&&) = default;

  /**
   * @brief Construct a `column_view_base` from pointers to device memory for
   *the elements and bitmask of the column.
   *
   * If `null_count()` is zero, `null_mask` is optional.
   *
   * If `type` is `EMPTY`, the specified `null_count` will be ignored and
   * `null_count()` will always return the same value as `size()`
   *
   * @throws cudf::logic_error if `size < 0`
   * @throws cudf::logic_error if `size > 0` but `data == nullptr`
   * @throws cudf::logic_error if `type.id() == EMPTY` but `data != nullptr`
   *or `null_mask != nullptr`
   * @throws cudf::logic_error if `null_count > 0`, but `null_mask == nullptr`
   * @throws cudf::logic_error if `offset < 0`
   *
   * @param type The element type
   * @param size The number of elements
   * @param data Pointer to device memory containing the column elements
   * @param null_mask Pointer to device memory containing the null
   * indicator bitmask
   * @param null_count The number of null elements.
   * @param offset Optional, index of the first element
   */
  column_view_base(data_type type,
                   size_type size,
                   void const* data,
                   bitmask_type const* null_mask,
                   size_type null_count,
                   size_type offset = 0);
};

}  // namespace detail

/**
 * @brief A non-owning, immutable view of device data as a column of elements,
 * some of which may be null as indicated by a bitmask.
 *
 * @ingroup column_classes
 *
 * A `column_view` can be constructed implicitly from a `cudf::column`, or may
 * be constructed explicitly from a pointer to pre-existing device memory.
 *
 * Unless otherwise noted, the memory layout of the `column_view`'s data and
 * bitmask is expected to adhere to the Arrow Physical Memory Layout
 * Specification: https://arrow.apache.org/docs/format/Columnar.html#format-columnar
 *
 * Because `column_view` is non-owning, no device memory is allocated nor freed
 * when `column_view` objects are created or destroyed.
 *
 * To enable zero-copy slicing, a `column_view` has an `offset` that indicates
 * the index of the first element in the column relative to the base device
 * memory allocation. By default, `offset()` is zero.
 */
class column_view : public detail::column_view_base {
 public:
  column_view() = default;

  // these pragmas work around the nvcc issue where if a column_view is used
  // inside of a __device__ code path, these functions will end up being created
  // as __host__ __device__ because they are explicitly defaulted.  However, if
  // they then end up being called by a simple __host__ function
  // (eg std::vector destructor) you get a compile error because you're trying to
  // call a __host__ __device__ function from a __host__ function.
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  ~column_view() override = default;
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  column_view(column_view const&) = default;  ///< Copy constructor
  column_view(column_view&&)      = default;  ///< Move constructor
  /**
   * @brief Copy assignment operator
   *
   * @return Reference to this object
   */
  column_view& operator=(column_view const&) = default;
  /**
   * @brief Move assignment operator
   *
   * @return Reference to this object
   */
  column_view& operator=(column_view&&) = default;

  /**
   * @brief Construct a `column_view` from pointers to device memory for the
   * elements and bitmask of the column.
   *
   * If `null_count()` is zero, `null_mask` is optional.
   *
   * If `type` is `EMPTY`, the specified `null_count` will be ignored and
   * `null_count()` will always return the same value as `size()`
   *
   * @throws cudf::logic_error if `size < 0`
   * @throws cudf::logic_error if `size > 0` but `data == nullptr`
   * @throws cudf::logic_error if `type.id() == EMPTY` but `data != nullptr`
   *or `null_mask != nullptr`
   * @throws cudf::logic_error if `null_count > 0`, but `null_mask == nullptr`
   * @throws cudf::logic_error if `offset < 0`
   *
   * @param type The element type
   * @param size The number of elements
   * @param data Pointer to device memory containing the column elements
   * @param null_mask Pointer to device memory containing the null
   * indicator bitmask
   * @param null_count The number of null elements.
   * @param offset Optional, index of the first element
   * @param children Optional, depending on the element type, child columns may
   * contain additional data
   */
  column_view(data_type type,
              size_type size,
              void const* data,
              bitmask_type const* null_mask,
              size_type null_count,
              size_type offset                         = 0,
              std::vector<column_view> const& children = {});

  /**
   * @brief Returns the specified child
   *
   * @param child_index The index of the desired child
   * @return The requested child `column_view`
   */
  [[nodiscard]] column_view child(size_type child_index) const noexcept
  {
    return _children[child_index];
  }

  /**
   * @brief Returns the number of child columns.
   *
   * @return The number of child columns
   */
  [[nodiscard]] size_type num_children() const noexcept { return _children.size(); }

  /**
   * @brief Returns iterator to the beginning of the ordered sequence of child column-views.
   *
   * @return An iterator to a `column_view` referencing the first child column
   */
  auto child_begin() const noexcept { return _children.cbegin(); }

  /**
   * @brief Returns iterator to the end of the ordered sequence of child column-views.
   *
   * @return An iterator to a `column_view` one past the end of the child columns
   */
  auto child_end() const noexcept { return _children.cend(); }

  /**
   * @brief Construct a column view from a device_span<T>.
   *
   * Only numeric and chrono types are supported.
   *
   * @tparam T The device span type. Must be const and match the column view's type.
   * @param data A typed device span containing the column view's data.
   */
  template <typename T, CUDF_ENABLE_IF(cudf::is_numeric<T>() or cudf::is_chrono<T>())>
  column_view(device_span<T const> data)
    : column_view(
        cudf::data_type{cudf::type_to_id<T>()}, data.size(), data.data(), nullptr, 0, 0, {})
  {
    CUDF_EXPECTS(
      data.size() <= static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
      "Data exceeds the column size limit",
      std::overflow_error);
  }

  /**
   * @brief Converts a column view into a device span.
   *
   * Only numeric and chrono data types are supported. The column view must not
   * be nullable.
   *
   * @tparam T The device span type. Must be const and match the column view's type.
   * @throws cudf::logic_error if the column view type does not match the span type.
   * @throws cudf::logic_error if the column view is nullable.
   * @return A typed device span of the column view's data.
   */
  template <typename T, CUDF_ENABLE_IF(cudf::is_numeric<T>() or cudf::is_chrono<T>())>
  [[nodiscard]] operator device_span<T const>() const
  {
    CUDF_EXPECTS(type() == cudf::data_type{cudf::type_to_id<T>()},
                 "Device span type must match column view type.");
    CUDF_EXPECTS(!nullable(), "A nullable column view cannot be converted to a device span.");
    return device_span<T const>(data<T>(), size());
  }

 protected:
  /**
   * @brief Returns pointer to the base device memory allocation.
   *
   * The primary purpose of this function is to allow derived classes to
   * override the fundamental properties of memory accesses without needing to
   * change all of the different accessors for the underlying pointer.
   *
   * @return Typed pointer to underlying data
   */
  void const* get_data() const noexcept override;

 private:
  friend column_view bit_cast(column_view const& input, data_type type);

  std::vector<column_view> _children{};  ///< Based on element type, children
                                         ///< may contain additional data
};

/**
 * @brief A non-owning, mutable view of device data as a column of elements,
 * some of which may be null as indicated by a bitmask.
 *
 * @ingroup column_classes
 *
 * A `mutable_column_view` can be constructed implicitly from a `cudf::column`,
 * or may be constructed explicitly from a pointer to pre-existing device memory.
 *
 * Unless otherwise noted, the memory layout of the `mutable_column_view`'s data
 * and bitmask is expected to adhere to the Arrow Physical Memory Layout
 * Specification: https://arrow.apache.org/docs/format/Columnar.html#format-columnar
 *
 * Because `mutable_column_view` is non-owning, no device memory is allocated
 * nor freed when `mutable_column_view` objects are created or destroyed.
 *
 * To enable zero-copy slicing, a `mutable_column_view` has an `offset` that
 * indicates the index of the first element in the column relative to the base
 * device memory allocation. By default, `offset()` is zero.
 */
class mutable_column_view : public detail::column_view_base {
 public:
  mutable_column_view() = default;

  ~mutable_column_view() override {
    // Needed so that the first instance of the implicit destructor for any TU isn't 'constructed'
    // from a host+device function marking the implicit version also as host+device
  };

  mutable_column_view(mutable_column_view const&) = default;  ///< Copy constructor
  mutable_column_view(mutable_column_view&&)      = default;  ///< Move constructor
  /**
   * @brief Copy assignment operator
   *
   * @return Reference to this object
   */
  mutable_column_view& operator=(mutable_column_view const&) = default;
  /**
   * @brief Move assignment operator
   *
   * @return Reference to this object (after transferring ownership)
   */
  mutable_column_view& operator=(mutable_column_view&&) = default;

  /**
   * @brief Construct a `mutable_column_view` from pointers to device memory for
   * the elements and bitmask of the column.

   * If `type` is `EMPTY`, the specified `null_count` will be ignored and
   * `null_count()` will always return the same value as `size()`
   *
   * @throws cudf::logic_error if `size < 0`
   * @throws cudf::logic_error if `size > 0` but `data == nullptr`
   * @throws cudf::logic_error if `type.id() == EMPTY` but `data != nullptr`
   *or `null_mask != nullptr`
   * @throws cudf::logic_error if `null_count > 0`, but `null_mask == nullptr`
   * @throws cudf::logic_error if `offset < 0`
   *
   * @param type The element type
   * @param size The number of elements
   * @param data Pointer to device memory containing the column elements
   * @param null_mask Pointer to device memory containing the null
   indicator
   * bitmask
   * @param null_count The number of null elements.
   * @param offset Optional, index of the first element
   * @param children Optional, depending on the element type, child columns may
   * contain additional data
   */
  mutable_column_view(data_type type,
                      size_type size,
                      void* data,
                      bitmask_type* null_mask,
                      size_type null_count,
                      size_type offset                                 = 0,
                      std::vector<mutable_column_view> const& children = {});

  /**
   * @brief Returns pointer to the base device memory allocation casted to
   * the specified type.
   *
   * This function will only participate in overload resolution if `is_rep_layout_compatible<T>()`
   * or `std::is_same_v<T,void>` are true.
   *
   * @note If `offset() == 0`, then `head<T>() == data<T>()`
   *
   * @note It should be rare to need to access the `head<T>()` allocation of a
   * column, and instead, accessing the elements should be done via `data<T>()`.
   *
   * @tparam T The type to cast to
   * @return Typed pointer to underlying data
   */
  template <typename T = void,
            CUDF_ENABLE_IF(std::is_same_v<T, void> or is_rep_layout_compatible<T>())>
  T* head() const noexcept
  {
    return const_cast<T*>(detail::column_view_base::head<T>());
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
  T* data() const noexcept
  {
    return const_cast<T*>(detail::column_view_base::data<T>());
  }

  /**
   * @brief Return first element (accounting for offset) after underlying data is
   * casted to the specified type.
   *
   * This function does not participate in overload resolution if `is_rep_layout_compatible<T>` is
   * false.
   *
   * @tparam T The desired type
   * @return Pointer to the first element after casting
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  T* begin() const noexcept
  {
    return const_cast<T*>(detail::column_view_base::begin<T>());
  }

  /**
   * @brief Return one past the last element after underlying data is casted to
   * the specified type.
   *
   * This function does not participate in overload resolution if `is_rep_layout_compatible<T>` is
   * false.
   *
   * @tparam T The desired type
   * @return Pointer to one past the last element after casting
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  T* end() const noexcept
  {
    return const_cast<T*>(detail::column_view_base::end<T>());
  }

  /**
   * @brief Returns raw pointer to the underlying bitmask allocation.
   *
   * @note This function does *not* account for the `offset()`.
   *
   * @note If `null_count() == 0`, this may return `nullptr`.
   *
   * @return Raw pointer to the underlying bitmask allocation
   */
  [[nodiscard]] bitmask_type* null_mask() const noexcept
  {
    return const_cast<bitmask_type*>(detail::column_view_base::null_mask());
  }

  /**
   * @brief Set the null count
   *
   * @throws cudf::logic_error if `new_null_count > 0` and `nullable() == false`
   *
   * @param new_null_count The new null count
   */
  void set_null_count(size_type new_null_count);

  /**
   * @brief Returns a reference to the specified child
   *
   * @param child_index The index of the desired child
   * @return The requested child `mutable_column_view`
   */
  [[nodiscard]] mutable_column_view child(size_type child_index) const noexcept
  {
    return mutable_children[child_index];
  }

  /**
   * @brief Returns the number of child columns.
   *
   * @return The number of child columns
   */
  [[nodiscard]] size_type num_children() const noexcept { return mutable_children.size(); }

  /**
   * @brief Returns iterator to the beginning of the ordered sequence of child column-views.
   *
   * @return An iterator to a `mutable_column_view` referencing the first child column
   */
  auto child_begin() const noexcept { return mutable_children.begin(); }

  /**
   * @brief Returns iterator to the end of the ordered sequence of child column-views.
   *
   * @return An iterator to a `mutable_column_view` to the element following the last child column
   */
  auto child_end() const noexcept { return mutable_children.end(); }

  /**
   * @brief Converts a mutable view into an immutable view
   *
   * @return An immutable view of the mutable view's elements
   */
  operator column_view() const;

 protected:
  /**
   * @brief Returns pointer to the base device memory allocation.
   *
   * The primary purpose of this function is to allow derived classes to
   * override the fundamental properties of memory accesses without needing to
   * change all of the different accessors for the underlying pointer.
   *
   * @return Typed pointer to underlying data
   */
  [[nodiscard]] void const* get_data() const noexcept override;

 private:
  friend mutable_column_view bit_cast(mutable_column_view const& input, data_type type);

  std::vector<mutable_column_view> mutable_children;
};

/**
 * @brief Counts the number of descendants of the specified parent.
 *
 * @param parent The parent whose descendants will be counted
 * @return The number of descendants of the parent
 */
size_type count_descendants(column_view parent);

/**
 * @brief Zero-copy cast between types with the same size and compatible underlying representations.
 *
 * This is similar to `reinterpret_cast` or `bit_cast` in that it gives a view of the same raw bits
 * as a different type. Unlike `reinterpret_cast` however, this cast is only allowed on types that
 * have the same width and compatible representations. For example, the way timestamp types are laid
 * out in memory is equivalent to an integer representing a duration since a fixed epoch;
 * bit-casting to the same integer type (INT32 for days, INT64 for others) results in a raw view of
 * the duration count. A FLOAT32 can also be bit-casted into INT32 and treated as an integer value.
 * However, an INT32 column cannot be bit-casted to INT64 as the sizes differ, nor can a string_view
 * column be casted into a numeric type column as their data representations are not compatible.
 *
 * The validity of the conversion can be checked with `cudf::is_bit_castable()`.
 *
 * @throws cudf::logic_error if the specified cast is not possible, i.e.,
 * `is_bit_castable(input.type(), type)` is false.
 *
 * @ingroup utility_types
 *
 * @param input The `column_view` to cast from
 * @param type The `data_type` to cast to
 * @return New `column_view` wrapping the same data as `input` but cast to `type`
 */
column_view bit_cast(column_view const& input, data_type type);

/**
 * @brief Zero-copy cast between types with the same size and compatible underlying representations.
 *
 * This is similar to `reinterpret_cast` or `bit_cast` in that it gives a view of the same raw bits
 * as a different type. Unlike `reinterpret_cast` however, this cast is only allowed on types that
 * have the same width and compatible representations. For example, the way timestamp types are laid
 * out in memory is equivalent to an integer representing a duration since a fixed epoch;
 * bit-casting to the same integer type (INT32 for days, INT64 for others) results in a raw view of
 * the duration count. A FLOAT32 can also be bit-casted into INT32 and treated as an integer value.
 * However, an INT32 column cannot be bit-casted to INT64 as the sizes differ, nor can a string_view
 * column be casted into a numeric type column as their data representations are not compatible.
 *
 * The validity of the conversion can be checked with `cudf::is_bit_castable()`.
 *
 * @throws cudf::logic_error if the specified cast is not possible, i.e.,
 * `is_bit_castable(input.type(), type)` is false.
 *
 * @ingroup utility_types
 *
 * @param input The `mutable_column_view` to cast from
 * @param type The `data_type` to cast to
 * @return New `mutable_column_view` wrapping the same data as `input` but cast to `type`
 */
mutable_column_view bit_cast(mutable_column_view const& input, data_type type);

namespace detail {
/**
 * @brief Computes a hash value from the shallow state of the specified column
 *
 * For any two columns, if `is_shallow_equivalent(c0,c1)` then `shallow_hash(c0) ==
 * shallow_hash(c1)`.
 *
 * The complexity of computing the hash value of `input` is `O( count_descendants(input) )`, i.e.,
 * it is independent of the number of elements in the column.
 *
 * This function does _not_ inspect the elements of `input` nor access any device memory or launch
 * any kernels.
 *
 * @param input The `column_view` to compute hash
 * @return The hash value derived from the shallow state of `input`.
 */
std::size_t shallow_hash(column_view const& input);

/**
 * @brief Uses only shallow state to determine if two `column_view`s view equivalent columns
 *
 *  Two columns are equivalent if for any operation `F` then:
 *   ```
 *    is_shallow_equivalent(c0, c1) ==> The results of F(c0) and F(c1) are equivalent
 *   ```
 * For any two non-empty columns, `is_shallow_equivalent(c0,c1)` is true only if they view the exact
 * same physical column. In other words, two physically independent columns may have exactly
 * equivalent elements but their shallow state would not be equivalent.
 *
 * The complexity of this function is `O( min(count_descendants(lhs), count_descendants(rhs)) )`,
 * i.e., it is independent of the number of elements in either column.
 *
 * This function does _not_ inspect the elements of `lhs` or `rhs` nor access any device memory nor
 * launch any kernels.
 *
 * @param lhs The left `column_view` to compare
 * @param rhs The right `column_view` to compare
 * @return If `lhs` and `rhs` have equivalent shallow state
 */
bool is_shallow_equivalent(column_view const& lhs, column_view const& rhs);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
