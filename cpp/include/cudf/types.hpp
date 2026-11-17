/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef __CUDACC__
/**
 * @brief Indicates that the function or method is usable on host and device
 */
#define CUDF_HOST_DEVICE __host__ __device__
/**
 * @brief Indicates that the function is a CUDA kernel
 */
#define CUDF_KERNEL __global__ static
#else
/**
 * @brief Indicates that the function or method is usable on host and device
 */
#define CUDF_HOST_DEVICE
/**
 * @brief Indicates that the function is a CUDA kernel
 */
#define CUDF_KERNEL static
#endif

#include <cudf/utilities/export.hpp>

#include <cuda/std/iterator>

#include <cassert>
#include <cstddef>
#include <cstdint>

/**
 * @file
 * @brief Type declarations for libcudf.
 */

namespace CUDF_EXPORT cudf {
// Forward declaration
class column;
class column_view;
class mutable_column_view;
class string_view;
class list_view;
class struct_view;
class scalar;

// clang-format off
class list_scalar;
class struct_scalar;
class string_scalar;
template <typename T> class numeric_scalar;
template <typename T> class fixed_point_scalar;
template <typename T> class timestamp_scalar;
template <typename T> class duration_scalar;

class string_scalar_device_view;
template <typename T> class numeric_scalar_device_view;
template <typename T> class fixed_point_scalar_device_view;
template <typename T> class timestamp_scalar_device_view;
template <typename T> class duration_scalar_device_view;
// clang-format on

class table;
class table_view;
class mutable_table_view;

/**
 * @addtogroup utility_types
 * @{
 */

using size_type         = int32_t;   ///< Row index type for columns and tables
using bitmask_type      = uint32_t;  ///< Bitmask type stored as 32-bit unsigned integer
using valid_type        = uint8_t;   ///< Valid type in host memory
using thread_index_type = int64_t;   ///< Thread index type in kernels
using char_utf8         = uint32_t;  ///< UTF-8 characters are 1-4 bytes

/**
 * @brief Similar to `std::distance` but returns `cudf::size_type` and performs `static_cast`
 *
 * @tparam T Iterator type
 * @param f "first" iterator
 * @param l "last" iterator
 * @return The distance between first and last
 */
template <typename T>
size_type distance(T f, T l)
{
  return static_cast<size_type>(cuda::std::distance(f, l));
}

/**
 * @brief Indicates the order in which elements should be sorted.
 */
enum class order : bool {
  ASCENDING,  ///< Elements ordered from small to large
  DESCENDING  ///< Elements ordered from large to small
};

/**
 * @brief Enum to specify whether to include nulls or exclude nulls
 */
enum class null_policy : bool {
  EXCLUDE,  ///< exclude null elements
  INCLUDE   ///< include null elements
};

/**
 * @brief Enum to treat NaN floating point value as null or non-null element
 */
enum class nan_policy : bool {
  NAN_IS_NULL,  ///< treat nans as null elements
  NAN_IS_VALID  ///< treat nans as valid elements (non-null)
};

/**
 * @brief Enum to consider different elements (of floating point types) holding NaN value as equal
 * or unequal
 */
enum class nan_equality /*unspecified*/ {
  ALL_EQUAL,  ///< All NaNs compare equal, regardless of sign
  UNEQUAL     ///< All NaNs compare unequal (IEEE754 behavior)
};

/**
 * @brief Enum to consider two nulls as equal or unequal
 */
enum class null_equality : bool {
  EQUAL,   ///< nulls compare equal
  UNEQUAL  ///< nulls compare unequal
};

/**
 * @brief Indicates how null values compare against all other values.
 */
enum class null_order : bool {
  AFTER,  ///< NULL values ordered *after* all other values
  BEFORE  ///< NULL values ordered *before* all other values
};

/**
 * @brief Indicates whether a collection of values is known to be sorted.
 */
enum class sorted : bool { NO, YES };

/**
 * @brief Indicates how a collection of values has been ordered.
 */
struct order_info {
  sorted is_sorted;          ///< Indicates whether the collection is sorted
  order ordering;            ///< Indicates the order in which the values are sorted
  null_order null_ordering;  ///< Indicates how null values compare against all other values
};

/**
 * @brief Controls the allocation/initialization of a null mask.
 */
enum class mask_state : int32_t {
  UNALLOCATED,    ///< Null mask not allocated, (all elements are valid)
  UNINITIALIZED,  ///< Null mask allocated, but not initialized
  ALL_VALID,      ///< Null mask allocated, initialized to all elements valid
  ALL_NULL        ///< Null mask allocated, initialized to all elements NULL
};

/**
 * @brief Interpolation method to use when the desired quantile lies between
 * two data points i and j
 */
enum class interpolation : int32_t {
  LINEAR,          ///< Linear interpolation between i and j
  LOWER,           ///< Lower data point (i)
  HIGHER,          ///< Higher data point (j)
  MIDPOINT,        ///< (i + j)/2
  NEAREST,         ///< i or j, whichever is nearest (half-to-even rounding)
  NEAREST_HALF_UP  ///< i or j, whichever is nearest (half-away-from-zero rounding)
};

/**
 * @brief Identifies a column's logical element type
 */
enum class type_id : int32_t {
  EMPTY,                   ///< Always null with no underlying data
  INT8,                    ///< 1 byte signed integer
  INT16,                   ///< 2 byte signed integer
  INT32,                   ///< 4 byte signed integer
  INT64,                   ///< 8 byte signed integer
  UINT8,                   ///< 1 byte unsigned integer
  UINT16,                  ///< 2 byte unsigned integer
  UINT32,                  ///< 4 byte unsigned integer
  UINT64,                  ///< 8 byte unsigned integer
  FLOAT32,                 ///< 4 byte floating point
  FLOAT64,                 ///< 8 byte floating point
  BOOL8,                   ///< Boolean using one byte per value, 0 == false, else true
  TIMESTAMP_DAYS,          ///< point in time in days since Unix Epoch in int32
  TIMESTAMP_SECONDS,       ///< point in time in seconds since Unix Epoch in int64
  TIMESTAMP_MILLISECONDS,  ///< point in time in milliseconds since Unix Epoch in int64
  TIMESTAMP_MICROSECONDS,  ///< point in time in microseconds since Unix Epoch in int64
  TIMESTAMP_NANOSECONDS,   ///< point in time in nanoseconds since Unix Epoch in int64
  DURATION_DAYS,           ///< time interval of days in int32
  DURATION_SECONDS,        ///< time interval of seconds in int64
  DURATION_MILLISECONDS,   ///< time interval of milliseconds in int64
  DURATION_MICROSECONDS,   ///< time interval of microseconds in int64
  DURATION_NANOSECONDS,    ///< time interval of nanoseconds in int64
  DICTIONARY32,            ///< Dictionary type using int32 indices
  STRING,                  ///< String elements
  LIST,                    ///< List elements
  DECIMAL32,               ///< Fixed-point type with int32_t
  DECIMAL64,               ///< Fixed-point type with int64_t
  DECIMAL128,              ///< Fixed-point type with __int128_t
  STRUCT,                  ///< Struct elements
  // `NUM_TYPE_IDS` must be last!
  NUM_TYPE_IDS  ///< Total number of type ids
};

/**
 * @brief A function is null-aware if its output value uses the input validity.
 *
 * For example, ADD is not null-aware, but IS_NULL and NULL_LOGICAL_AND are null-aware.
 */
enum class null_aware : bool {
  NO  = 0,  ///< The function is not null-aware
  YES = 1   ///< The function is null-aware
};

/**
 * @brief Indicates the null output policy of a function.
 *
 * This policy determines whether a null-mask should be excluded from the output column.
 *
 * For example, consider a `null-aware` `IS_NULL` function:
 *
 * ```cpp
 * void is_null(optional<bool> * out, optional<int> a){
 *   *out = !a.has_value();
 * }
 *
 * ```
 *
 * using `output_nullability::PRESERVE` a null-mask may be produced.
 * with `output_nullability::ALL_VALID` a null-mask will not be produced and all values are
 * considered valid. It is undefined behaviour to use `ALL_VALID` with a UDF that produces null
 * values.
 *
 *
 */
enum class output_nullability : uint8_t {
  PRESERVE  = 0,  ///< A null-mask may be produced if needed
  ALL_VALID = 1   ///< A null-mask is not produced and all values are considered valid even if
                  ///< null values are produced
};

/**
 * @brief Indicates whether a function nullifies its output on error.
 *
 */
enum class error_policy : uint8_t {
  PROPAGATE = 0,  ///< The function propagates errors
  NULLIFY   = 1   ///< The function nullifies its output on error
};

/**
 * @brief Indicates the source language of a user defined function (UDF) to be used in JIT APIs.
 */
enum class udf_source_type : uint8_t {
  CUDA = 0,  ///< The UDF is a CUDA function
  PTX  = 1   ///< The UDF is a PTX function
};

/**
 * @brief Indicator for the logical data type of an element in a column.
 *
 * Simple types can be entirely described by their `id()`, but some types
 * require additional metadata to fully describe elements of that type.
 */
class data_type {
 public:
  data_type()                 = default;
  ~data_type()                = default;
  data_type(data_type const&) = default;  ///< Copy constructor
  data_type(data_type&&)      = default;  ///< Move constructor

  /**
   * @brief Copy assignment operator for data_type
   *
   * @return Reference to this object
   */
  data_type& operator=(data_type const&) = default;

  /**
   * @brief Move assignment operator for data_type
   *
   * @return Reference to this object
   */
  data_type& operator=(data_type&&) = default;

  /**
   * @brief Construct a new `data_type` object
   *
   * @param id The type's identifier
   */
  CUDF_HOST_DEVICE explicit constexpr data_type(type_id id) : _id{id} {}

  /**
   * @brief Construct a new `data_type` object for `numeric::fixed_point`
   *
   * @param id The `fixed_point`'s identifier
   * @param scale The `fixed_point`'s scale (see `fixed_point::_scale`)
   */
  explicit data_type(type_id id, int32_t scale) : _id{id}, _fixed_point_scale{scale}
  {
    assert(id == type_id::DECIMAL32 || id == type_id::DECIMAL64 || id == type_id::DECIMAL128);
  }

  /**
   * @brief Returns the type identifier
   *
   * @return The type identifier
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr type_id id() const noexcept { return _id; }

  /**
   * @brief Returns the scale (for fixed_point types)
   *
   * @return The scale
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr int32_t scale() const noexcept
  {
    return _fixed_point_scale;
  }

 private:
  type_id _id{type_id::EMPTY};

  // Below is additional type specific metadata. Currently, only _fixed_point_scale is stored.

  int32_t _fixed_point_scale{};  // numeric::scale_type not available here, use int32_t
};

/**
 * @brief Compares two `data_type` objects for equality.
 *
 * // TODO Define exactly what it means for two `data_type`s to be equal. e.g.,
 * are two timestamps with different resolutions equal? How about decimals with
 * different scale/precision?
 *
 * @param lhs The first `data_type` to compare
 * @param rhs The second `data_type` to compare
 * @return true `lhs` is equal to `rhs`
 * @return false `lhs` is not equal to `rhs`
 */
constexpr bool operator==(data_type const& lhs, data_type const& rhs)
{
  // use std::tie in the future, breaks NVRTC currently
  return lhs.id() == rhs.id() && lhs.scale() == rhs.scale();
}

/**
 * @brief Compares two `data_type` objects for inequality.
 *
 * // TODO Define exactly what it means for two `data_type`s to be equal. e.g.,
 * are two timestamps with different resolutions equal? How about decimals with
 * different scale/precision?
 *
 * @param lhs The first `data_type` to compare
 * @param rhs The second `data_type` to compare
 * @return true `lhs` is not equal to `rhs`
 * @return false `lhs` is equal to `rhs`
 */
inline bool operator!=(data_type const& lhs, data_type const& rhs) { return !(lhs == rhs); }

/**
 * @brief Returns the size in bytes of elements of the specified `data_type`
 *
 * @note Only fixed-width types are supported
 *
 * @throws cudf::logic_error if `is_fixed_width(element_type) == false`
 *
 * @param t The `data_type` to get the size of
 * @return Size in bytes of an element of the specified `data_type`
 */
std::size_t size_of(data_type t);

/** @} */
}  // namespace CUDF_EXPORT cudf
