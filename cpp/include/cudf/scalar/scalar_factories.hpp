/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <span>

/**
 * @file
 * @brief Scalar factory APIs
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup scalar_factories
 * @{
 */

/**
 * @brief Construct scalar with uninitialized storage to hold a value of the
 * specified numeric `data_type`.
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a numeric type
 *
 * @param type The desired numeric element type
 * @param stream CUDA stream used for device memory operations.
 * @param mr Device memory resource used to allocate the scalar's `data` and `is_valid` bool.
 * @returns An uninitialized numeric scalar
 */
std::unique_ptr<scalar> make_numeric_scalar(
  data_type type,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Construct scalar with uninitialized storage to hold a value of the
 * specified timestamp `data_type`.
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a timestamp type
 *
 * @param type The desired timestamp element type
 * @param stream CUDA stream used for device memory operations.
 * @param mr Device memory resource used to allocate the scalar's `data` and `is_valid` bool.
 * @return An uninitialized timestamp scalar
 */
std::unique_ptr<scalar> make_timestamp_scalar(
  data_type type,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Construct scalar with uninitialized storage to hold a value of the
 * specified duration `data_type`.
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a duration type
 *
 * @param type The desired duration element type
 * @param stream CUDA stream used for device memory operations.
 * @param mr Device memory resource used to allocate the scalar's `data` and `is_valid` bool.
 * @return An uninitialized duration scalar
 */
std::unique_ptr<scalar> make_duration_scalar(
  data_type type,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Construct scalar with uninitialized storage to hold a value of the
 * specified fixed-width `data_type`.
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a fixed-width type
 *
 * @param type The desired fixed-width element type
 * @param stream CUDA stream used for device memory operations.
 * @param mr Device memory resource used to allocate the scalar's `data` and `is_valid` bool.
 * @return An uninitialized fixed-width scalar
 */
std::unique_ptr<scalar> make_fixed_width_scalar(
  data_type type,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Construct STRING type scalar given a `std::string`.
 * The size of the `std::string` must not exceed the maximum size of size_type.
 * The string characters are expected to be UTF-8 encoded sequence of char bytes.
 *
 * @throws std::bad_alloc if device memory allocation fails
 *
 * @param string The `std::string` to copy to device
 * @param stream CUDA stream used for device memory operations.
 * @param mr Device memory resource used to allocate the scalar's `data` and `is_valid` bool.
 * @returns A string scalar with the contents of `string`
 */
std::unique_ptr<scalar> make_string_scalar(
  std::string const& string,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Constructs default constructed scalar of type `type`
 *
 * @throws std::bad_alloc if device memory allocation fails
 *
 * @param type The desired element type
 * @param stream CUDA stream used for device memory operations.
 * @param mr Device memory resource used to allocate the scalar's `data` and `is_valid` bool.
 * @returns A scalar of type `type`
 */
std::unique_ptr<scalar> make_default_constructed_scalar(
  data_type type,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Creates an empty (invalid) scalar of the same type as the `input` column_view.
 *
 * @throw cudf::logic_error if the `input` column is struct type and empty
 *
 * @param input Immutable view of input column to emulate
 * @param stream CUDA stream used for device memory operations.
 * @param mr Device memory resource used to allocate the scalar's `data` and `is_valid` bool.
 * @returns A scalar of type of `input` column
 */
std::unique_ptr<scalar> make_empty_scalar_like(
  column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Construct scalar using the given value of fixed width type
 *
 * @tparam T Datatype of the value to be represented by the scalar
 * @param value The value to store in the scalar object
 * @param stream CUDA stream used for device memory operations.
 * @param mr Device memory resource used to allocate the scalar's `data` and `is_valid` bool.
 * @returns A scalar of type `T`
 */
template <typename T>
std::unique_ptr<scalar> make_fixed_width_scalar(
  T value,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  return std::make_unique<scalar_type_t<T>>(value, true, stream, mr);
}

/**
 * @brief Construct scalar using the given value of fixed_point type
 *
 * @tparam T Datatype of the value to be represented by the scalar
 * @param value The value to store in the scalar object
 * @param scale The scale of the fixed point value
 * @param stream CUDA stream used for device memory operations.
 * @param mr Device memory resource used to allocate the scalar's `data` and `is_valid` bool.
 * @returns A scalar of type `T`
 */
template <typename T>
std::unique_ptr<scalar> make_fixed_point_scalar(
  typename T::rep value,
  numeric::scale_type scale,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  return std::make_unique<scalar_type_t<T>>(value, scale, true, stream, mr);
}

/**
 * @brief Construct scalar using the given column of elements
 *
 * @param elements Elements of the list
 * @param stream CUDA stream used for device memory operations.
 * @param mr Device memory resource used to allocate the scalar's `data` and `is_valid` bool.
 * @returns A list scalar
 */
std::unique_ptr<scalar> make_list_scalar(
  column_view elements,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Construct a struct scalar using the given table_view.
 *
 * The columns must have 1 row.
 *
 * @param data The columnar data to store in the scalar object
 * @param stream CUDA stream used for device memory operations.
 * @param mr Device memory resource used to allocate the scalar's `data` and `is_valid` bool.
 * @returns A struct scalar
 */
std::unique_ptr<scalar> make_struct_scalar(
  table_view const& data,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Construct a struct scalar using the given span of column views.
 *
 * The columns must have 1 row.
 *
 * @param data The columnar data to store in the scalar object
 * @param stream CUDA stream used for device memory operations.
 * @param mr Device memory resource used to allocate the scalar's `data` and `is_valid` bool.
 * @returns A struct scalar
 */
std::unique_ptr<scalar> make_struct_scalar(
  std::span<column_view const> data,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
