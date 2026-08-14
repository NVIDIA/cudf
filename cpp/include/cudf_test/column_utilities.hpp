/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf_test/default_stream.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream_ref>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>

namespace CUDF_EXPORT cudf {
namespace test {

/**
 * @brief Verbosity level of output from column and table comparison functions.
 */
enum class debug_output_level {
  FIRST_ERROR = 0,  // print first error only
  ALL_ERRORS,       // print all errors
  QUIET             // no debug output
};

constexpr size_type default_ulp = 4;

namespace detail {

/**
 * @brief Verifies the property equality of two columns.
 *
 * @note This function should not be used directly. Use `CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUAL`
 * instead.
 *
 * @param lhs The first column
 * @param rhs The second column
 * @param verbosity Level of debug output verbosity
 * @param stream CUDA stream used for device memory operations
 * @param mr Memory resources used for temporary device allocations
 *
 * @returns True if the column properties are equal, false otherwise
 */
bool expect_column_properties_equal(
  cudf::column_view const& lhs,
  cudf::column_view const& rhs,
  debug_output_level verbosity = debug_output_level::FIRST_ERROR,
  cuda::stream_ref stream      = cudf::test::get_default_stream(),
  cudf::memory_resources mr    = cudf::get_current_device_resource_ref());

/**
 * @brief Verifies the property equivalence of two columns.
 *
 * If the columns don't have nulls, then the nullability equality is relaxed.
 * i.e. the two columns are considered equivalent even if one has a null mask
 * and the other doesn't.
 *
 * @note This function should not be used directly. Use
 * `CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUIVALENT` instead.
 *
 * @param lhs The first column
 * @param rhs The second column
 * @param verbosity Level of debug output verbosity
 * @param stream CUDA stream used for device memory operations
 * @param mr Memory resources used for temporary device allocations
 *
 * @returns True if the column properties are equivalent, false otherwise
 */
bool expect_column_properties_equivalent(
  cudf::column_view const& lhs,
  cudf::column_view const& rhs,
  debug_output_level verbosity = debug_output_level::FIRST_ERROR,
  cuda::stream_ref stream      = cudf::test::get_default_stream(),
  cudf::memory_resources mr    = cudf::get_current_device_resource_ref());

/**
 * @brief Verifies the element-wise equality of two columns.
 *
 * Treats null elements as equivalent.
 *
 * @note This function should not be used directly. Use
 * `CUDF_TEST_EXPECT_COLUMNS_EQUAL` instead.
 *
 * @param lhs The first column
 * @param rhs The second column
 * @param verbosity Level of debug output verbosity
 * @param stream CUDA stream used for device memory operations
 * @param mr Memory resources used for temporary device allocations
 *
 * @returns True if the columns (and their properties) are equal, false otherwise
 */
bool expect_columns_equal(cudf::column_view const& lhs,
                          cudf::column_view const& rhs,
                          debug_output_level verbosity = debug_output_level::FIRST_ERROR,
                          cuda::stream_ref stream      = cudf::test::get_default_stream(),
                          cudf::memory_resources mr    = cudf::get_current_device_resource_ref());

/**
 * @brief Verifies the element-wise equivalence of two columns.
 *
 * Uses machine epsilon to compare floating point types.
 * Treats null elements as equivalent.
 *
 * @note This function should not be used directly. Use `CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT`
 * instead.
 *
 * @param lhs The first column
 * @param rhs The second column
 * @param verbosity Level of debug output verbosity
 * @param fp_ulps # of ulps of tolerance to allow when comparing
 * floating point values
 * @param stream CUDA stream used for device memory operations
 * @param mr Memory resources used for temporary device allocations
 *
 * @returns True if the columns (and their properties) are equivalent, false otherwise
 */
bool expect_columns_equivalent(cudf::column_view const& lhs,
                               cudf::column_view const& rhs,
                               debug_output_level verbosity = debug_output_level::FIRST_ERROR,
                               size_type fp_ulps            = cudf::test::default_ulp,
                               cuda::stream_ref stream      = cudf::test::get_default_stream(),
                               cudf::memory_resources mr = cudf::get_current_device_resource_ref());

/**
 * @brief Verifies the bitwise equality of two device memory buffers.
 *
 * @note This function should not be used directly. Use `CUDF_TEST_EXPECT_EQUAL_BUFFERS` instead.
 *
 * @param lhs The first buffer
 * @param rhs The second buffer
 * @param size_bytes The number of bytes to check for equality
 * @param stream CUDA stream used for device memory operations
 * @param mr Memory resources used for temporary device allocations
 */
void expect_equal_buffers(void const* lhs,
                          void const* rhs,
                          std::size_t size_bytes,
                          cuda::stream_ref stream   = cudf::test::get_default_stream(),
                          cudf::memory_resources mr = cudf::get_current_device_resource_ref());

}  // namespace detail

/**
 * @brief Verifies the given column is empty
 *
 * @param col The column to check
 */
void expect_column_empty(cudf::column_view const& col);

/**
 * @brief Copy the null bitmask from a column view to a host vector
 *
 * @param c      The column view
 * @param stream CUDA stream used for device memory operations
 * @param mr Memory resources used for temporary device allocations
 * @returns      Vector of bitmask_type elements
 */
std::vector<bitmask_type> bitmask_to_host(
  cudf::column_view const& c,
  cuda::stream_ref stream   = cudf::test::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

/**
 * @brief Validates bitmask situated in host as per `number_of_elements`
 *
 * This takes care of padded bits
 *
 * @param        expected_mask A vector representing expected mask
 * @param        got_mask_begin A vector representing mask obtained from column
 * @param        number_of_elements number of elements the mask represent
 *
 * @returns      true if both vector match till the `number_of_elements`
 */
bool validate_host_masks(std::vector<bitmask_type> const& expected_mask,
                         std::vector<bitmask_type> const& got_mask_begin,
                         size_type number_of_elements);

/**
 * @brief Copies the data and bitmask of a `column_view` to the host.
 *
 * @tparam T The data type of the elements of the `column_view`
 * @param c the `column_view` to copy from
 * @param stream CUDA stream used for device memory operations
 * @param mr Memory resources used for temporary device allocations
 * @return std::pair<thrust::host_vector<T>, std::vector<bitmask_type>> first is the
 *  `column_view`'s data, and second is the column's bitmask.
 */
template <typename T, std::enable_if_t<not cudf::is_fixed_point<T>()>* = nullptr>
std::pair<thrust::host_vector<T>, std::vector<bitmask_type>> to_host(
  column_view c,
  cuda::stream_ref stream   = cudf::test::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref())
{
  auto col_span  = cudf::device_span<T const>(c.data<T>(), c.size());
  auto host_data = cudf::detail::make_host_vector(col_span, stream);
  return {std::move(host_data), bitmask_to_host(c, stream, mr)};
}

// This signature is identical to the above overload apart from SFINAE so
// doxygen sees it as a duplicate.
//! @cond Doxygen_Suppress
/**
 * @brief Copies the data and bitmask of a `column_view` to the host.
 *
 * This is the specialization for `fixed_point` that performs construction of a `fixed_point` from
 * the underlying `rep` type that is stored on the device.
 *
 * @tparam T The data type of the elements of the `column_view`
 * @param c the `column_view` to copy from
 * @param stream CUDA stream used for device memory operations
 * @param mr Memory resources used for temporary device allocations
 * @return std::pair<thrust::host_vector<T>, std::vector<bitmask_type>> first is the
 *  `column_view`'s data, and second is the column's bitmask.
 */
template <typename T, std::enable_if_t<cudf::is_fixed_point<T>()>* = nullptr>
CUDF_EXPORT std::pair<thrust::host_vector<T>, std::vector<bitmask_type>> to_host(
  column_view c,
  cuda::stream_ref stream   = cudf::test::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

/**
 * @brief Copies the data and bitmask of a `column_view` of strings
 * column to the host.
 *
 * @throw cudf::logic_error if c is not strings column.
 *
 * @param c the `column_view` of strings to copy from
 * @param stream CUDA stream used for device memory operations
 * @param mr Memory resources used for temporary device allocations
 * @return std::pair first is `std::vector` of `std::string`
 * and second is the column's bitmask.
 */
template <>
CUDF_EXPORT std::pair<thrust::host_vector<std::string>, std::vector<bitmask_type>> to_host(
  column_view c, cuda::stream_ref stream, cudf::memory_resources mr);
//! @endcond

/**
 * @brief For enabling large strings testing in specific tests
 */
struct large_strings_enabler {
  /**
   * @brief Create large strings enable object
   *
   * @param default_enable Default enables large strings support
   */
  large_strings_enabler(bool default_enable = true);
  ~large_strings_enabler();

  /**
   * @brief Enable large strings support
   */
  void enable();

  /**
   * @brief Disable large strings support
   */
  void disable();
};

}  // namespace test
}  // namespace CUDF_EXPORT cudf

// Macros for showing line of failure.
#define CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUAL(lhs, rhs, ...)                              \
  do {                                                                                       \
    SCOPED_TRACE(" <--  line of failure\n");                                                 \
    cudf::test::detail::expect_column_properties_equal(lhs, rhs __VA_OPT__(, ) __VA_ARGS__); \
  } while (0)

#define CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUIVALENT(lhs, rhs, ...)                              \
  do {                                                                                            \
    SCOPED_TRACE(" <--  line of failure\n");                                                      \
    cudf::test::detail::expect_column_properties_equivalent(lhs, rhs __VA_OPT__(, ) __VA_ARGS__); \
  } while (0)

#define CUDF_TEST_EXPECT_COLUMNS_EQUAL(lhs, rhs, ...)                              \
  do {                                                                             \
    SCOPED_TRACE(" <--  line of failure\n");                                       \
    cudf::test::detail::expect_columns_equal(lhs, rhs __VA_OPT__(, ) __VA_ARGS__); \
  } while (0)

#define CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(lhs, rhs, ...)                              \
  do {                                                                                  \
    SCOPED_TRACE(" <--  line of failure\n");                                            \
    cudf::test::detail::expect_columns_equivalent(lhs, rhs __VA_OPT__(, ) __VA_ARGS__); \
  } while (0)

#define CUDF_TEST_EXPECT_EQUAL_BUFFERS(lhs, rhs, size_bytes, ...)                              \
  do {                                                                                         \
    SCOPED_TRACE(" <--  line of failure\n");                                                   \
    cudf::test::detail::expect_equal_buffers(lhs, rhs, size_bytes __VA_OPT__(, ) __VA_ARGS__); \
  } while (0)

#define CUDF_TEST_ENABLE_LARGE_STRINGS() cudf::test::large_strings_enabler ls___
