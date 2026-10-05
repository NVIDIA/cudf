/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/types.hpp>

/**
 * @file scalar_device_view.cuh
 * @brief Scalar device view class definitions
 */

namespace CUDF_EXPORT cudf {
namespace detail {
/**
 * @brief A non-owning view of scalar from device that is trivially copyable
 * and usable in CUDA device code.
 */
class scalar_device_view_base {
 public:
  ~scalar_device_view_base() = default;

  /**
   * @brief Returns the value type
   *
   * @returns The value type
   */
  [[nodiscard]] __host__ __device__ data_type type() const noexcept { return _type; }

  /**
   * @brief Returns whether the scalar holds a valid value (i.e., not null).
   *
   * @return true The element is valid
   * @return false The element is null
   */
  [[nodiscard]] __device__ bool is_valid() const noexcept { return *_is_valid; }

  /**
   * @brief Updates the validity of the value
   *
   * @param is_valid true: set the value to valid. false: set it to null
   */
  __device__ void set_valid(bool is_valid) noexcept { *_is_valid = is_valid; }

 protected:
  data_type _type{type_id::EMPTY};  ///< Value data type
  bool* _is_valid{};                ///< Pointer to device memory containing
                                    ///< boolean representing validity of the value.

  /**
   * @brief Construct a new scalar device view base object  from a device pointer
   * and a validity boolean.
   *
   * @param type The data type of the scalar
   * @param is_valid Pointer to device memory containing boolean representing
   * validity of the scalar.
   */
  scalar_device_view_base(data_type type, bool* is_valid) : _type(type), _is_valid(is_valid) {}

  scalar_device_view_base() = default;
};

/**
 * @brief A type-erased scalar_device_view where the value is a fixed width type
 */
class fixed_width_scalar_device_view_base : public detail::scalar_device_view_base {
 public:
  /**
   * @brief Returns reference to stored value.
   *
   * @tparam T The desired type
   * @returns Reference to stored value
   */
  template <typename T>
  __device__ T& value() noexcept
  { return *data<T>(); }

  /**
   * @brief Returns const reference to stored value.
   *
   * @tparam T The desired type
   * @returns Const reference to stored value
   */
  template <typename T>
  __device__ T const& value() const noexcept
  { return *data<T>(); }

  /**
   * @brief Stores the value in scalar
   *
   * @tparam T The desired type
   * @param value The value to store in scalar
   */
  template <typename T>
  __device__ void set_value(T value)
  { *static_cast<T*>(_data) = value; }

  /**
   * @brief Returns a raw pointer to the value in device memory
   *
   * @tparam T The desired type
   * @returns Raw pointer to the value in device memory
   */
  template <typename T>
  __device__ T* data() noexcept
  { return static_cast<T*>(_data); }

  /**
   * @brief Returns a const raw pointer to the value in device memory
   *
   * @tparam T The desired type
   * @returns Const raw pointer to the value in device memory
   */
  template <typename T>
  __device__ T const* data() const noexcept
  { return static_cast<T const*>(_data); }

 protected:
  void* _data{};  ///< Pointer to device memory containing the value

  /**
   * @brief Construct a new fixed width scalar device view object
   *
   * This constructor should not be used directly. get_scalar_device_view
   * should be used to get the view of an existing scalar
   *
   * @param type The data type of the value
   * @param data The pointer to the data in device memory
   * @param is_valid The pointer to the bool in device memory that indicates the
   * validity of the stored value
   */
  fixed_width_scalar_device_view_base(data_type type, void* data, bool* is_valid)
    : detail::scalar_device_view_base(type, is_valid), _data(data)
  {
  }
};

/**
 * @brief A type of scalar_device_view where the value is a fixed width type
 */
template <typename T>
class fixed_width_scalar_device_view : public detail::fixed_width_scalar_device_view_base {
 public:
  using value_type = T;  ///< The value type of the scalar

  /**
   * @brief Returns reference to stored value.
   *
   * @returns Reference to stored value
   */
  __device__ T& value() noexcept { return fixed_width_scalar_device_view_base::value<T>(); }

  /**
   * @brief Returns const reference to stored value.
   *
   * @returns Const reference to stored value
   */
  __device__ T const& value() const noexcept
  { return fixed_width_scalar_device_view_base::value<T>(); }

  /**
   * @brief Stores the value in scalar
   *
   * @param value The value to store in scalar
   */
  __device__ void set_value(T value) { fixed_width_scalar_device_view_base::set_value<T>(value); }

  /**
   * @brief Returns a raw pointer to the value in device memory
   *
   * @returns Raw pointer to the value in device memory
   */
  __device__ T* data() noexcept { return fixed_width_scalar_device_view_base::data<T>(); }

  /**
   * @brief Returns a const raw pointer to the value in device memory
   *
   * @returns Const raw pointer to the value in device memory
   */
  __device__ T const* data() const noexcept
  { return fixed_width_scalar_device_view_base::data<T>(); }

 protected:
  /**
   * @brief Construct a new fixed width scalar device view object
   *
   * This constructor should not be used directly. get_scalar_device_view
   * should be used to get the view of an existing scalar
   *
   * @param type The data type of the value
   * @param data The pointer to the data in device memory
   * @param is_valid The pointer to the bool in device memory that indicates the
   * validity of the stored value
   */
  fixed_width_scalar_device_view(data_type type, T* data, bool* is_valid)
    : detail::fixed_width_scalar_device_view_base(type, data, is_valid)
  {
  }
};

}  // namespace detail

/**
 * @brief A type of scalar_device_view that stores a pointer to a numerical value
 */
template <typename T>
class numeric_scalar_device_view : public detail::fixed_width_scalar_device_view<T> {
 public:
  /**
   * @brief Construct a new numeric scalar device view object from data and validity pointers.
   *
   * @param type The data type of the value
   * @param data The pointer to the data in device memory
   * @param is_valid The pointer to the bool in device memory that indicates the
   * validity of the stored value
   */
  numeric_scalar_device_view(data_type type, T* data, bool* is_valid)
    : detail::fixed_width_scalar_device_view<T>(type, data, is_valid)
  {
  }
};

/**
 * @brief A type of scalar_device_view that stores a pointer to a fixed_point value
 */
template <typename T>
class fixed_point_scalar_device_view : public detail::scalar_device_view_base {
 public:
  using rep_type = typename T::rep;  ///< The representation type of the fixed_point value

  /**
   * @brief Construct a new fixed point scalar device view object from data and validity pointers.
   *
   * @param type The data type of the value
   * @param data The pointer to the data in device memory
   * @param is_valid The pointer to the bool in device memory that indicates the
   * validity of the stored value
   */
  fixed_point_scalar_device_view(data_type type, rep_type* data, bool* is_valid)
    : detail::scalar_device_view_base(type, is_valid), _data(data)
  {
  }

  /**
   * @brief Stores the value in scalar
   *
   * @param value The value to store in scalar
   */
  __device__ void set_value(rep_type value) { *_data = value; }

  /**
   * @brief Get the value of the scalar, as a `rep_type`.
   *
   * @returns The value of the scalar, as a `rep_type`
   */
  __device__ rep_type const& rep() const noexcept { return *_data; }

 private:
  rep_type* _data{};
};

/**
 * @brief A type of scalar_device_view that stores a pointer to a string value
 */
class string_scalar_device_view : public detail::scalar_device_view_base {
 public:
  using ValueType = cudf::string_view;  ///< The value type of the string scalar

  /**
   * @brief Construct a new string scalar device view object from string data, size and validity
   * pointers.
   *
   * @param type The data type of the value
   * @param data The pointer to the string data in device memory
   * @param is_valid The pointer to the bool in device memory that indicates the
   * validity of the stored value
   * @param size The pointer to the size of the string in device memory
   */
  string_scalar_device_view(data_type type, char const* data, bool* is_valid, size_type size)
    : detail::scalar_device_view_base(type, is_valid), _data(data), _size(size)
  {
  }

  /**
   * @brief Returns string_view of the value of this scalar.
   *
   * @returns string_view of the value of this scalar
   */
  [[nodiscard]] __device__ ValueType value() const noexcept
  { return ValueType{this->data(), _size}; }

  /**
   * @brief Returns a raw pointer to the value in device memory
   *
   * @returns Raw pointer to the value in device memory
   */
  [[nodiscard]] __device__ char const* data() const noexcept
  { return static_cast<char const*>(_data); }

  /**
   * @brief Returns the size of the string in bytes.
   *
   * @returns The size of the string in bytes
   */
  [[nodiscard]] __device__ size_type size() const noexcept { return _size; }

 private:
  char const* _data{};  ///< Pointer to device memory containing the value
  size_type _size;      ///< Size of the string in bytes
};

/**
 * @brief A type of scalar_device_view that stores a pointer to a timestamp value
 */
template <typename T>
class timestamp_scalar_device_view : public detail::fixed_width_scalar_device_view<T> {
 public:
  /**
   * @brief Construct a new timestamp scalar device view object
   *
   * @param type The data type of the value
   * @param data The pointer to the data in device memory
   * @param is_valid The pointer to the bool in device memory that indicates the
   * validity of the stored value
   */
  timestamp_scalar_device_view(data_type type, T* data, bool* is_valid)
    : detail::fixed_width_scalar_device_view<T>(type, data, is_valid)
  {
  }
};

/**
 * @brief A type of scalar_device_view that stores a pointer to a duration value
 */
template <typename T>
class duration_scalar_device_view : public detail::fixed_width_scalar_device_view<T> {
 public:
  /**
   * @brief Construct a new duration scalar device view object from data and validity pointers.
   *
   * @param type The data type of the value
   * @param data The pointer to the data in device memory
   * @param is_valid The pointer to the bool in device memory that indicates the
   * validity of the stored value
   */
  duration_scalar_device_view(data_type type, T* data, bool* is_valid)
    : detail::fixed_width_scalar_device_view<T>(type, data, is_valid)
  {
  }
};

/**
 * @brief Get the device view of a numeric_scalar
 *
 * @param s The numeric_scalar to get the device view of
 * @return A device view of a numeric_scalar
 */
template <typename T>
auto get_scalar_device_view(numeric_scalar<T>& s)
{ return numeric_scalar_device_view<T>(s.type(), s.data(), s.validity_data()); }

/**
 * @brief Get the device view of a string_scalar
 *
 * @param s The string_scalar to get the device view of
 * @return A device view of a string_scalar
 */
inline auto get_scalar_device_view(string_scalar& s)
{ return string_scalar_device_view(s.type(), s.data(), s.validity_data(), s.size()); }

/**
 * @brief Get the device view of a timestamp_scalar
 *
 * @param s The timestamp_scalar to get the device view of
 * @return A device view of a timestamp_scalar
 */
template <typename T>
auto get_scalar_device_view(timestamp_scalar<T>& s)
{ return timestamp_scalar_device_view<T>(s.type(), s.data(), s.validity_data()); }

/**
 * @brief Get the device view of a duration_scalar
 *
 * @param s The duration_scalar to get the device view of
 * @return A device view of a duration_scalar
 */
template <typename T>
auto get_scalar_device_view(duration_scalar<T>& s)
{ return duration_scalar_device_view<T>(s.type(), s.data(), s.validity_data()); }

/**
 * @brief Get the device view of a fixed_point_scalar
 *
 * @param s The fixed_point_scalar to get the device view of
 * @return The device view of the fixed_point_scalar
 */
template <typename T>
auto get_scalar_device_view(fixed_point_scalar<T>& s)
{ return fixed_point_scalar_device_view<T>(s.type(), s.data(), s.validity_data()); }

}  // namespace CUDF_EXPORT cudf
