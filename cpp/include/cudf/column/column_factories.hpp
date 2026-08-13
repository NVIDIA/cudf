/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/std/utility>

/**
 * @file
 * @brief Column factory APIs
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup column_factories
 * @{
 */

/**
 * @brief Creates an empty column of the specified @p type
 *
 * An empty column contains zero elements and no validity mask.
 *
 * @param[in] type The column data type
 * @return Empty column with desired type
 */
std::unique_ptr<column> make_empty_column(data_type type);

/**
 * @brief Creates an empty column of the specified type.
 *
 * An empty column contains zero elements and no validity mask.
 *
 * @param[in] id The column type id
 * @return Empty column with specified type
 */
std::unique_ptr<column> make_empty_column(type_id id);

/**
 * @brief Construct column with sufficient uninitialized storage to hold `size` elements of the
 * specified numeric `data_type` with an optional null mask.
 *
 * @note `null_count()` is determined by the requested null mask `state`
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a numeric type
 * @throws cudf::logic_error if `size < 0`
 *
 * @param[in] type The desired numeric element type
 * @param[in] size The number of elements in the column
 * @param[in] state Optional, controls allocation/initialization of the
 * column's null mask. By default, no null mask is allocated.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 * @return Constructed numeric column
 */
std::unique_ptr<column> make_numeric_column(
  data_type type,
  size_type size,
  mask_state state                  = mask_state::UNALLOCATED,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Construct column with sufficient uninitialized storage to hold `size` elements of the
 * specified numeric `data_type` with a null mask.
 *
 * @note null_count is optional and will be computed if not provided.
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a numeric type
 *
 * @param[in] type The desired numeric element type
 * @param[in] size The number of elements in the column
 * @param[in] null_mask Null mask to use for this column.
 * @param[in] null_count Optional number of nulls in the null_mask.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 * @return Constructed numeric column
 */
template <typename B>
std::unique_ptr<column> make_numeric_column(
  data_type type,
  size_type size,
  B&& null_mask,
  size_type null_count,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  CUDF_EXPECTS(is_numeric(type), "Invalid, non-numeric type.");
  return std::make_unique<column>(type,
                                  size,
                                  rmm::device_buffer{size * cudf::size_of(type), stream, mr},
                                  std::forward<B>(null_mask),
                                  null_count);
}

/**
 * @brief Construct column with sufficient uninitialized storage to hold `size` elements of the
 * specified `fixed_point` `data_type` with an optional null mask.
 *
 * @note The column's null count is determined by the requested null mask `state`.
 *
 * @throws cudf::logic_error if `type` is not a `fixed_point` type.
 * @throws cudf::logic_error if `size < 0`
 *
 * @param[in] type The desired `fixed_point` element type.
 * @param[in] size The number of elements in the column.
 * @param[in] state Optional, controls allocation/initialization of the.
 * column's null mask. By default, no null mask is allocated.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory.
 * @return Constructed fixed-point type column
 */
std::unique_ptr<column> make_fixed_point_column(
  data_type type,
  size_type size,
  mask_state state                  = mask_state::UNALLOCATED,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Construct column with sufficient uninitialized storage to hold `size` elements of the
 * specified `fixed_point` `data_type` with a null mask.
 *
 * @note null_count is optional and will be computed if not provided.
 *
 * @throws cudf::logic_error if `type` is not a `fixed_point` type.
 *
 * @param[in] type The desired `fixed_point` element type.
 * @param[in] size The number of elements in the column.
 * @param[in] null_mask Null mask to use for this column.
 * @param[in] null_count Optional number of nulls in the null_mask.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory.
 * @return Constructed fixed-point type column
 */
template <typename B>
std::unique_ptr<column> make_fixed_point_column(
  data_type type,
  size_type size,
  B&& null_mask,
  size_type null_count,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  CUDF_EXPECTS(is_fixed_point(type), "Invalid, non-fixed_point type.");
  return std::make_unique<column>(type,
                                  size,
                                  rmm::device_buffer{size * cudf::size_of(type), stream, mr},
                                  std::forward<B>(null_mask),
                                  null_count);
}

/**
 * @brief Construct column with sufficient uninitialized storage to hold `size` elements of the
 * specified timestamp `data_type` with an optional null mask.
 *
 * @note `null_count()` is determined by the requested null mask `state`
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a timestamp type
 * @throws cudf::logic_error if `size < 0`
 *
 * @param[in] type The desired timestamp element type
 * @param[in] size The number of elements in the column
 * @param[in] state Optional, controls allocation/initialization of the
 * column's null mask. By default, no null mask is allocated.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 * @return Constructed timestamp type column
 */
std::unique_ptr<column> make_timestamp_column(
  data_type type,
  size_type size,
  mask_state state                  = mask_state::UNALLOCATED,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Construct column with sufficient uninitialized storage to hold `size` elements of the
 * specified timestamp `data_type` with a null mask.
 *
 * @note null_count is optional and will be computed if not provided.
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a timestamp type
 *
 * @param[in] type The desired timestamp element type
 * @param[in] size The number of elements in the column
 * @param[in] null_mask Null mask to use for this column.
 * @param[in] null_count Optional number of nulls in the null_mask.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 * @return Constructed timestamp type column
 */
template <typename B>
std::unique_ptr<column> make_timestamp_column(
  data_type type,
  size_type size,
  B&& null_mask,
  size_type null_count,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  CUDF_EXPECTS(is_timestamp(type), "Invalid, non-timestamp type.");
  return std::make_unique<column>(type,
                                  size,
                                  rmm::device_buffer{size * cudf::size_of(type), stream, mr},
                                  std::forward<B>(null_mask),
                                  null_count);
}

/**
 * @brief Construct column with sufficient uninitialized storage to hold `size` elements of the
 * specified duration `data_type` with an optional null mask.
 *
 * @note `null_count()` is determined by the requested null mask `state`
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a duration type
 * @throws cudf::logic_error if `size < 0`
 *
 * @param[in] type The desired duration element type
 * @param[in] size The number of elements in the column
 * @param[in] state Optional, controls allocation/initialization of the
 * column's null mask. By default, no null mask is allocated.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 * @return Constructed duration type column
 */
std::unique_ptr<column> make_duration_column(
  data_type type,
  size_type size,
  mask_state state                  = mask_state::UNALLOCATED,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Construct column with sufficient uninitialized storage to hold `size` elements of the
 * specified duration `data_type` with a null mask.
 *
 * @note null_count is optional and will be computed if not provided.
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a duration type
 *
 * @param[in] type The desired duration element type
 * @param[in] size The number of elements in the column
 * @param[in] null_mask Null mask to use for this column.
 * @param[in] null_count Optional number of nulls in the null_mask.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 * @return Constructed duration type column
 */
template <typename B>
std::unique_ptr<column> make_duration_column(
  data_type type,
  size_type size,
  B&& null_mask,
  size_type null_count,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  CUDF_EXPECTS(is_duration(type), "Invalid, non-duration type.");
  return std::make_unique<column>(type,
                                  size,
                                  rmm::device_buffer{size * cudf::size_of(type), stream, mr},
                                  std::forward<B>(null_mask),
                                  null_count);
}

/**
 * @brief Construct column with sufficient uninitialized storage to hold `size` elements of the
 * specified fixed width `data_type` with an optional null mask.
 *
 * @note `null_count()` is determined by the requested null mask `state`
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a fixed width type
 * @throws cudf::logic_error if `size < 0`
 *
 * @param[in] type The desired fixed width type
 * @param[in] size The number of elements in the column
 * @param[in] state Optional, controls allocation/initialization of the
 * column's null mask. By default, no null mask is allocated.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 * @return Constructed fixed-width type column
 */
std::unique_ptr<column> make_fixed_width_column(
  data_type type,
  size_type size,
  mask_state state                  = mask_state::UNALLOCATED,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Construct column with sufficient uninitialized storage to hold `size` elements of the
 * specified fixed width `data_type` with a null mask.
 *
 * @note null_count is optional and will be computed if not provided.
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a fixed width type
 *
 * @param[in] type The desired fixed width element type
 * @param[in] size The number of elements in the column
 * @param[in] null_mask Null mask to use for this column.
 * @param[in] null_count Optional number of nulls in the null_mask.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 * @return Constructed fixed-width type column
 */
template <typename B>
std::unique_ptr<column> make_fixed_width_column(
  data_type type,
  size_type size,
  B&& null_mask,
  size_type null_count,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  CUDF_EXPECTS(is_fixed_width(type), "Invalid, non-fixed-width type.");
  if (is_timestamp(type)) {
    return make_timestamp_column(type, size, std::forward<B>(null_mask), null_count, stream, mr);
  } else if (is_duration(type)) {
    return make_duration_column(type, size, std::forward<B>(null_mask), null_count, stream, mr);
  } else if (is_fixed_point(type)) {
    return make_fixed_point_column(type, size, std::forward<B>(null_mask), null_count, stream, mr);
  }
  return make_numeric_column(type, size, std::forward<B>(null_mask), null_count, stream, mr);
}

/**
 * @brief Construct a STRING type column given a device span of pointer/size pairs.
 *
 * The total number of char bytes must not exceed the maximum size of size_type.
 * The string characters are expected to be UTF-8 encoded sequence of char
 * bytes. Use the strings_column_view class to perform strings operations on
 * this type of column.
 *
 * @note `null_count()` and `null_bitmask` are determined if a pair contains
 * a null string. That is, for each pair, if `.first` is null, that string
 * is considered null. Likewise, a string is considered empty (not null)
 * if `.first` is not null and `.second` is 0. Otherwise the `.first` member
 * must be a valid device address pointing to `.second` consecutive bytes.
 *
 * @throws std::bad_alloc if device memory allocation fails
 *
 * @param[in] strings The device span of pointer/size pairs. Each pointer must be a device memory
   address or `nullptr` (indicating a null string). The size must be the number of bytes.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used for allocation of the column's `null_mask` and children
 * columns' device memory.
 * @return Constructed strings column
 */
std::unique_ptr<column> make_strings_column(
  cudf::device_span<cuda::std::pair<char const*, size_type> const> strings,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Construct a batch of STRING type columns given an array of device spans of pointer/size
 * pairs.
 *
 * This function has input/output expectation similar to the `make_strings_column()` API that
 * accepts only one device span of pointer/size pairs. The difference is that, this is designed to
 * create many strings columns at once with minimal overhead of multiple kernel launches and
 * stream synchronizations.
 *
 * @param input Array of device spans of pointer/size pairs, where each pointer is a device memory
 *        address or `nullptr` (indicating a null string), and size is string length (in bytes)
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used for memory allocation of the output columns
 * @return Array of constructed strings columns
 */
std::vector<std::unique_ptr<column>> make_strings_column_batch(
  std::vector<cudf::device_span<cuda::std::pair<char const*, size_type> const>> const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Construct a STRING type column given a device span of string_view.
 *
 * The total number of char bytes must not exceed the maximum size of size_type.
 * The string characters are expected to be UTF-8 encoded sequence of char
 * bytes. Use the strings_column_view class to perform strings operations on
 * this type of column.
 *
 * @note For each string_view, if `.data()` is `null_placeholder.data()`, that
 * string is considered null. Likewise, a string is considered empty (not null)
 * if `.data()` is not `null_placeholder.data()` and `.size_bytes()` is 0.
 * Otherwise the `.data()` must be a valid device address pointing to
 * `.size_bytes()` consecutive bytes. The `null_count()` for the output column
 * will be equal to the number of input `string_view`s that are null.
 *
 * @throws std::bad_alloc if device memory allocation fails
 *
 * @param[in] string_views The span of string_view. Each string_view must point to a device memory
   address or `null_placeholder` (indicating a null string). The size must be the number of bytes.
 * @param[in] null_placeholder string_view indicating null string in given list of
 * string_views.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used for allocation of the column's `null_mask` and children
 * columns' device memory.
  * @return Constructed strings column
 */
std::unique_ptr<column> make_strings_column(
  cudf::device_span<string_view const> string_views,
  string_view const null_placeholder,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Construct a STRING type column given offsets column, chars columns, and null mask and null
 * count.
 *
 * The columns and mask are moved into the resulting strings column.
 *
 * @param num_strings The number of strings the column represents.
 * @param offsets_column The column of offset values for this column. The number of elements is
 *  one more than the total number of strings so the `offset[last] - offset[0]` is the total number
 *  of bytes in the strings vector.
 * @param chars_buffer The buffer of char bytes for all the strings for this column. Individual
 *  strings are identified by the offsets and the nullmask.
 * @param null_count The number of null string entries.
 * @param null_mask The bits specifying the null strings in device memory. Arrow format for
 *  nulls is used for interpreting this bitmask.
 * @return Constructed strings column
 */
std::unique_ptr<column> make_strings_column(size_type num_strings,
                                            std::unique_ptr<column> offsets_column,
                                            rmm::device_buffer&& chars_buffer,
                                            size_type null_count,
                                            rmm::device_buffer&& null_mask);

/**
 * @brief Construct a LIST type column given offsets column, child column, null mask and null
 * count.
 *
 * The columns and mask are moved into the resulting lists column.
 *
 * List columns are structured similarly to strings columns.  They contain
 * a set of offsets which represents the lengths of the lists in each row, and
 * a "child" column of data that is referenced by the offsets.  Since lists
 * are a nested type, the child column may itself be further nested.
 *
 * When child column at depth N+1 is itself a list, the offsets column at
 * depth N references the offsets column for depth N+1.  When the child column at depth
 * N+1 is a leaf type (int, float, etc), the offsets column at depth N references
 * the data for depth N+1.
 *
 * @code{.pseudo}
 * Example:
 * List<int>
 * input:              {{1, 2}, {3, 4, 5}}
 * offsets (depth 0)   {0, 2, 5}
 * data    (depth 0)
 * offsets (depth 1)
 * data    (depth 1)   {1, 2, 3, 4, 5}
 * @endcode
 *
 * @code{.pseudo}
 * Example:
 * List<List<int>>
 * input:              { {{1, 2}}, {{3, 4, 5}, {6, 7}} }
 * offsets (depth 0)   {0, 1, 3}
 * data    (depth 0)
 * offsets (depth 1)   {0, 2, 5, 7}
 * data    (depth 1)
 * offsets (depth 2)
 * data    (depth 2)   {1, 2, 3, 4, 5, 6, 7}
 * @endcode
 *
 * @param num_rows The number of lists the column represents.
 * @param offsets_column The column of offset values for this column. Each value should
 * represent the starting offset into the child elements that corresponds to the beginning of the
 * row, with the first row starting at 0. The length of row N can be determined by subtracting
 * `offsets[N+1] - offsets[N]`. The total number of offsets should be 1 longer than the
 * number of rows in the column.
 * @param child_column The column of nested data referenced by the lists represented by the
 * offsets_column. Note: the child column may itself be further nested.
 * @param null_count The number of null list entries.
 * @param null_mask The bits specifying the null lists in device memory.
 * @return Constructed lists column
 */
std::unique_ptr<cudf::column> make_lists_column(size_type num_rows,
                                                std::unique_ptr<column> offsets_column,
                                                std::unique_ptr<column> child_column,
                                                size_type null_count,
                                                rmm::device_buffer&& null_mask);

/**
 * @brief Create an empty LIST column
 *
 * A list column requires a child type and so cannot be created with `make_empty_column`.
 *
 * @param child_type The type used for the empty child column
 * @return New empty lists column
 */
std::unique_ptr<column> make_empty_lists_column(data_type child_type);

/**
 * @brief Construct a STRUCT column using specified child columns as members.
 *
 * Specified child/member columns and null_mask are adopted by resultant
 * struct column.
 *
 * A struct column requires that all specified child columns have the same
 * number of rows. A struct column's row count equals that of any/all
 * of its child columns. A single struct row at any index is comprised of
 * all the individual child column values at the same index, in the order
 * specified in the list of child columns.
 *
 * The specified null mask governs which struct row has a null value. This
 * is orthogonal to the null values of individual child columns.
 *
 * @param[in] num_rows The number of struct values in the struct column.
 * @param[in] child_columns The list of child/members that the struct is comprised of.
 * @param[in] null_count The number of null values in the struct column.
 * @param[in] null_mask The bits specifying the null struct values in the column.
 * @param[in] stream Optional stream for use with all memory allocation and device kernels.
 * @param[in] mr Optional resource to use for device memory allocation.
 * @return Constructed structs column
 */
std::unique_ptr<cudf::column> make_structs_column(
  size_type num_rows,
  std::vector<std::unique_ptr<column>>&& child_columns,
  size_type null_count,
  rmm::device_buffer&& null_mask,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Construct a STRUCT column using specified child columns as members
 *
 * Specified child/member columns and null_mask are adopted by resultant
 * struct column i.e. the struct column hierarchy is created with the specified child
 * columns as its children. This function does not ensure null consistency with the
 * child columns or the descendant columns i.e. there are no guarantees that if a row of a struct
 * column is null, then the corresponding rows of the descendant columns would also be null.
 *
 * A struct column requires that all specified child columns have the same
 * number of rows. A struct column's row count equals that of any/all
 * of its child columns. A single struct row at any index is comprised of
 * all the individual child column values at the same index, in the order
 * specified in the list of child columns.
 *
 * The specified null mask governs which struct row has a null value. This
 * is orthogonal to the null values of individual child columns.
 *
 * @param[in] num_rows The number of struct values in the struct column.
 * @param[in] child_columns The list of child/members that the struct is comprised of.
 * @param[in] null_count The number of null values in the struct column.
 * @param[in] null_mask The bits specifying the null struct values in the column.
 * @param[in] stream Optional stream for use with all memory allocation and device kernels.
 * @param[in] mr Optional resource to use for device memory allocation.
 * @return Constructed structs column
 */
std::unique_ptr<cudf::column> create_structs_hierarchy(
  size_type num_rows,
  std::vector<std::unique_ptr<column>>&& child_columns,
  size_type null_count,
  rmm::device_buffer&& null_mask,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Construct a column with size elements that are all equal to the given scalar.
 *
 * The output column will have the same type as `s.type()`
 * The output column will contain all null rows if `s.invalid()==false`
 * The output column will be empty if `size==0`. For LIST scalars, the column hierarchy
 * from @p s is preserved.
 *
 * @param[in] s The scalar to use for values in the column.
 * @param[in] size The number of rows for the output column.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory.
 * @return Constructed column whose rows all contain the scalar value
 */
std::unique_ptr<column> make_column_from_scalar(
  scalar const& s,
  size_type size,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Construct a dictionary column with size elements that are all equal to the given scalar.
 *
 * The output column will have keys of type `s.type()`
 * The output column will be empty if `size==0`.
 *
 * @throw cudf::logic_error if `s.is_valid()==false`
 *
 * @param[in] s The scalar to use for values in the column.
 * @param[in] size The number of rows for the output column.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory.
 * @return Constructed dictionary column
 */
std::unique_ptr<column> make_dictionary_from_scalar(
  scalar const& s,
  size_type size,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
