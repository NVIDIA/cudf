/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>
#include <span>
#include <vector>

/**
 * @file
 * @brief Column APIs for gather, scatter, split, slice, etc.
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_copy
 * @{
 */

/**
 * @brief Policy to account for possible out-of-bounds indices
 */
enum class out_of_bounds_policy : bool {
  NULLIFY,    ///< Output values corresponding to out-of-bounds indices are null
  DONT_CHECK  ///< No bounds checking is performed, better performance
};

/**
 * @brief Policy to recognize negative indices as relative to the end of the source
 */
enum class negative_index_policy : bool {
  ALLOWED,     ///< Negative indices are allowed and result in wrapping behavior
  NOT_ALLOWED  ///< Negative indices are not allowed and result in undefined behavior
};

/**
 * @brief Gathers the specified rows (including null values) of a set of columns.
 *
 * @ingroup copy_gather
 *
 * Gathers the rows of the source columns according to `gather_map` such that row "i"
 * in the resulting table's columns will contain row "gather_map[i]" from the source columns.
 * The number of rows in the result table will be equal to the number of elements in
 * `gather_map`.
 *
 * A negative value `i` in the `gather_map` is interpreted as `i+n`, where
 * `n` is the number of rows in the `source_table`.
 *
 * For dictionary columns, the keys column component is copied and not trimmed
 * if the gather results in abandoned key elements.
 *
 * @throws std::invalid_argument if gather_map contains null values.
 *
 * @param source_table The input columns whose rows will be gathered
 * @param gather_map View into a non-nullable column of integral indices that maps the
 * rows in the source columns to rows in the destination columns.
 * @param bounds_policy Policy to apply to account for possible out-of-bounds indices
 * `DONT_CHECK` skips all bounds checking for gather map values. `NULLIFY` coerces rows that
 * corresponds to out-of-bounds indices in the gather map to be null elements. Callers should
 * use `DONT_CHECK` when they are certain that the gather_map contains only valid indices for
 * better performance. If `policy` is set to `DONT_CHECK` and there are out-of-bounds indices
 * in the gather map, the behavior is undefined. Defaults to `DONT_CHECK`.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return Result of the gather
 */
std::unique_ptr<table> gather(
  table_view const& source_table,
  column_view const& gather_map,
  out_of_bounds_policy bounds_policy = out_of_bounds_policy::DONT_CHECK,
  rmm::cuda_stream_view stream       = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr  = cudf::get_current_device_resource_ref());

/**
 * @brief Gathers the specified rows of a set of columns according to a gather map.
 *
 * @ingroup copy_gather
 *
 * Gathers the rows of the source columns according to `gather_map` such that row `i`
 * in the resulting table's columns will contain row `gather_map[i]` from the
 * `source_table` columns.
 *
 * The number of rows in the result table will be equal to the number of elements in
 * `gather_map`.
 *
 * If `neg_indices == ALLOWED` then a negative value `i` in the `gather_map` is interpreted
 * as `i+n`, where `n` is the number of rows in the `source_table`.
 * For better performance, use `DONT_CHECK` when the `gather_map` is known to contain only
 * valid indices.
 *
 * The output of row `i` results in null if `bounds_policy == NULLIFY` and an index exists
 * in `gather_map` outside the range `[-n, n)`, where `n` is the number of rows in the
 * `source table`.
 *
 * @param source_table The input columns whose rows will be gathered
 * @param gather_map View into a non-nullable column of integral indices that maps the
 * rows in the source columns to rows in the destination columns.
 * @param bounds_policy Interpretation of out-of-bounds indices
 * @param neg_indices Interpretation of a negative index `i` in the `gather_map`
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return Result of the gather
 */
std::unique_ptr<table> gather(
  table_view const& source_table,
  column_view const& gather_map,
  out_of_bounds_policy bounds_policy,
  negative_index_policy neg_indices,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Reverses the rows within a table.
 *
 * Creates a new table that is the reverse of @p source_table.
 * Example:
 * ```
 * source = [[4,5,6], [7,8,9], [10,11,12]]
 * return = [[6,5,4], [9,8,7], [12,11,10]]
 * ```
 *
 * @param source_table Table that will be reversed
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return Reversed table
 */
std::unique_ptr<table> reverse(
  table_view const& source_table,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Reverses the elements of a column
 *
 * Creates a new column that is the reverse of @p source_column.
 * Example:
 * ```
 * source = [4,5,6]
 * return = [6,5,4]
 * ```
 *
 * @param source_column Column that will be reversed
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return Reversed column
 */
std::unique_ptr<column> reverse(
  column_view const& source_column,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Scatters the rows of the source table into a copy of the target table
 * according to a scatter map.
 *
 * @ingroup copy_scatter
 *
 * Scatters values from the source table into the target table out-of-place,
 * returning a "destination table". The scatter is performed according to a
 * scatter map such that row `scatter_map[i]` of the destination table gets row
 * `i` of the source table. All other rows of the destination table equal
 * corresponding rows of the target table.
 *
 * The number of columns in source must match the number of columns in target
 * and their corresponding datatypes must be the same.
 *
 * If the same index appears more than once in the scatter map, the result is
 * undefined.
 *
 * If any values in `scatter_map` are outside of the interval [-n, n) where `n`
 * is the number of rows in the `target` table, behavior is undefined.
 *
 * A negative value `i` in the `scatter_map` is interpreted as `i+n`, where `n`
 * is the number of rows in the `target` table.
 *
 * @throws std::invalid_argument if the number of columns in source does not match the number of
 * columns in target
 * @throws std::invalid_argument if the number of rows in source does not match the number of
 * elements in scatter_map
 * @throws cudf::data_type_error if the data types of the source and target columns do not match
 * @throws std::invalid_argument if scatter_map contains null values
 *
 * @param source The input columns containing values to be scattered into the
 * target columns
 * @param scatter_map A non-nullable column of integral indices that maps the
 * rows in the source table to rows in the target table. The size must be equal
 * to or less than the number of elements in the source columns.
 * @param target The set of columns into which values from the source_table
 * are to be scattered
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return Result of scattering values from source to target
 */
std::unique_ptr<table> scatter(
  table_view const& source,
  column_view const& scatter_map,
  table_view const& target,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Scatters a row of scalar values into a copy of the target table
 * according to a scatter map.
 *
 * @ingroup copy_scatter
 *
 * Scatters values from the source row into the target table out-of-place,
 * returning a "destination table". The scatter is performed according to a
 * scatter map such that row `scatter_map[i]` of the destination table is
 * replaced by the source row. All other rows of the destination table equal
 * corresponding rows of the target table.
 *
 * The number of elements in source must match the number of columns in target
 * and their corresponding datatypes must be the same.
 *
 * If the same index appears more than once in the scatter map, the result is
 * undefined.
 *
 * If any values in `scatter_map` are outside of the interval [-n, n) where `n`
 * is the number of rows in the `target` table, behavior is undefined.
 *
 * @throws std::invalid_argument if the number of scalars does not match the number of columns in
 * target
 * @throws std::invalid_argument if indices contains null values
 * @throws cudf::data_type_error if the data types of the scalars and target columns do not match
 *
 * @param source The input scalars containing values to be scattered into the
 * target columns
 * @param indices A non-nullable column of integral indices that indicate
 * the rows in the target table to be replaced by source.
 * @param target The set of columns into which values from the source_table
 * are to be scattered
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return Result of scattering values from source to target
 */
std::unique_ptr<table> scatter(
  std::vector<std::reference_wrapper<scalar const>> const& source,
  column_view const& indices,
  table_view const& target,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Indicates when to allocate a mask, based on an existing mask.
 */
enum class mask_allocation_policy : int32_t {
  NEVER,   ///< Do not allocate a null mask, regardless of input
  RETAIN,  ///< Allocate a null mask if the input contains one
  ALWAYS   ///< Allocate a null mask, regardless of input
};

/**
 * @brief Initializes and returns an empty column of the same type as the `input`.
 *
 * @param[in] input Immutable view of input column to emulate
 * @return An empty column of same type as `input`
 */
std::unique_ptr<column> empty_like(column_view const& input);

/**
 * @brief Initializes and returns an empty column of the same type as the `input`.
 *
 * @param[in] input Scalar to emulate
 * @return An empty column of same type as `input`
 */
std::unique_ptr<column> empty_like(scalar const& input);

/**
 * @brief Creates an uninitialized new column of the same size and type as the `input`.
 *
 * Supports only fixed-width types.
 *
 * If the `mask_alloc` allocates a validity mask that mask is also uninitialized
 * and the validity bits and the null count should be set by the caller.
 *
 * @throws cudf::data_type_error if input type is not of fixed width.
 *
 * @param input Immutable view of input column to emulate
 * @param mask_alloc Optional, Policy for allocating null mask. Defaults to RETAIN
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A column with sufficient uninitialized capacity to hold the same
 * number of elements as `input` of the same type as `input.type()`
 */
std::unique_ptr<column> allocate_like(
  column_view const& input,
  mask_allocation_policy mask_alloc = mask_allocation_policy::RETAIN,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Creates an uninitialized new column of the specified size and same type as the `input`.
 *
 * Supports only fixed-width types.
 *
 * If the `mask_alloc` allocates a validity mask that mask is also uninitialized
 * and the validity bits and the null count should be set by the caller.
 *
 * @param input Immutable view of input column to emulate
 * @param size The desired number of elements that the new column should have capacity for
 * @param mask_alloc Optional, Policy for allocating null mask. Defaults to RETAIN
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A column with sufficient uninitialized capacity to hold the specified number of elements
 * as `input` of the same type as `input.type()`
 */
std::unique_ptr<column> allocate_like(
  column_view const& input,
  size_type size,
  mask_allocation_policy mask_alloc = mask_allocation_policy::RETAIN,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Creates a table of empty columns with the same types as the `input_table`
 *
 * Creates the `cudf::column` objects, but does not allocate any underlying device
 * memory for the column's data or bitmask.
 *
 * @param[in] input_table Immutable view of input table to emulate
 * @return A table of empty columns with the same types as the columns in
 * `input_table`
 */
std::unique_ptr<table> empty_like(table_view const& input_table);

/**
 * @brief Copies a range of elements in-place from one column to another.
 *
 * Overwrites the range of elements in @p target indicated by the indices
 * [@p target_begin, @p target_begin + N) with the elements from @p source
 * indicated by the indices [@p source_begin, @p source_end) (where N =
 * (@p source_end - @p source_begin)). Use the out-of-place copy function
 * returning std::unique_ptr<column> for uses cases requiring memory
 * reallocation. For example for strings columns and other variable-width types.
 *
 * If @p source and @p target refer to the same elements and the ranges overlap,
 * the behavior is undefined.
 *
 * @throws cudf::data_type_error if memory reallocation is required (e.g. for
 * variable width types).
 * @throws std::out_of_range for invalid range (if
 * @p source_begin > @p source_end, @p source_begin < 0,
 * @p source_begin >= @p source.size(), @p source_end > @p source.size(),
 * @p target_begin < 0, target_begin >= @p target.size(), or
 * @p target_begin + (@p source_end - @p source_begin) > @p target.size()).
 * @throws cudf::data_type_error if @p target and @p source have different types.
 * @throws std::invalid_argument if @p source has null values and @p target is not
 * nullable.
 *
 * @param source The column to copy from
 * @param target The preallocated column to copy into
 * @param source_begin The starting index of the source range (inclusive)
 * @param source_end The index of the last element in the source range
 * (exclusive)
 * @param target_begin The starting index of the target range (inclusive)
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void copy_range_in_place(column_view const& source,
                         mutable_column_view& target,
                         size_type source_begin,
                         size_type source_end,
                         size_type target_begin,
                         rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Copies a range of elements out-of-place from one column to another.
 *
 * Creates a new column as if an in-place copy was performed into @p target.
 * A copy of @p target is created first and then the elements indicated by the
 * indices [@p target_begin, @p target_begin + N) were copied from the elements
 * indicated by the indices [@p source_begin, @p source_end) of @p source
 * (where N = (@p source_end - @p source_begin)). Elements outside the range are
 * copied from @p target into the returned new column target.
 *
 * If @p source and @p target refer to the same elements and the ranges overlap,
 * the behavior is undefined.
 *
 * A range is considered invalid if:
 *   - Either the begin or end indices are out of bounds for the corresponding column
 *   - Begin is greater than end for source or target
 *   - The size of the source range would overflow the target column starting at target_begin
 *
 * @throws std::out_of_range for any invalid range.
 * @throws cudf::data_type_error if @p target and @p source have different types.
 * @throws cudf::data_type_error if the data type is not fixed width, string, or dictionary
 *
 * @param source The column to copy from inside the range
 * @param target The column to copy from outside the range
 * @param source_begin The starting index of the source range (inclusive)
 * @param source_end The index of the last element in the source range
 * (exclusive)
 * @param target_begin The starting index of the target range (inclusive)
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return The result target column
 */
std::unique_ptr<column> copy_range(
  column_view const& source,
  column_view const& target,
  size_type source_begin,
  size_type source_end,
  size_type target_begin,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Creates a new column by shifting all values by an offset.
 *
 * @ingroup copy_shift
 *
 * Elements will be determined by `output[idx] = input[idx - offset]`.
 * Some elements in the output may be indeterminable from the input. For those
 * elements, the value will be determined by `fill_values`.
 *
 * @code{.pseudo}
 * Examples
 * -------------------------------------------------
 * input       = [0, 1, 2, 3, 4]
 * offset      = 3
 * fill_values = @
 * return      = [@, @, @, 0, 1]
 * -------------------------------------------------
 * input       = [5, 4, 3, 2, 1]
 * offset      = -2
 * fill_values = 7
 * return      = [3, 2, 1, 7, 7]
 * @endcode
 *
 * @note if the input is nullable, the output will be nullable.
 * @note if the fill value is null, the output will be nullable.
 *
 * @param input      Column to be shifted
 * @param offset     The offset by which to shift the input
 * @param fill_value Fill value for indeterminable outputs
 * @param stream     CUDA stream used for device memory operations and kernel launches
 * @param mr         Device memory resource used to allocate the returned result's device memory
 *
 * @throw cudf::data_type_error if @p input dtype is neither fixed-width nor string type
 * @throw cudf::data_type_error if @p fill_value dtype does not match @p input dtype.
 *
 * @return The shifted column
 */
std::unique_ptr<column> shift(
  column_view const& input,
  size_type offset,
  scalar const& fill_value,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Slices a `column_view` into a set of `column_view`s according to a set of indices.
 *
 * @ingroup copy_slice
 *
 * The returned views of `input` are constructed from an even number indices where
 * the `i`th returned `column_view` views the elements in `input` indicated by the range
 * `[indices[2*i], indices[(2*i)+1])`.
 *
 * For all `i` it is expected `indices[i] <= input.size()`
 * For all `i%2==0`, it is expected that `indices[i] <= indices[i+1]`
 *
 * @note It is the caller's responsibility to ensure that the returned views
 * do not outlive the viewed device memory.
 *
 * @code{.pseudo}
 * input:   {10, 12, 14, 16, 18, 20, 22, 24, 26, 28}
 * indices: {1, 3, 5, 9, 2, 4, 8, 8}
 * output:  {{12, 14}, {20, 22, 24, 26}, {14, 16}, {}}
 * @endcode
 *
 * @throws std::invalid_argument if `indices` size is not even.
 * @throws std::invalid_argument When the values in the pair are strictly decreasing.
 * @throws std::out_of_range When any of the values in the pair don't belong to
 * the range [0, input.size()).
 *
 * @param input View of column to slice
 * @param indices Indices used to take slices of `input`
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Vector of views of `input` indicated by the ranges in `indices`
 */
std::vector<column_view> slice(column_view const& input,
                               std::span<size_type const> indices,
                               rmm::cuda_stream_view stream = cudf::get_default_stream());
/**
 * @ingroup copy_slice
 * @copydoc cudf::slice(column_view const&, std::span<size_type const>, rmm::cuda_stream_view)
 */
std::vector<column_view> slice(column_view const& input,
                               std::initializer_list<size_type> indices,
                               rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Slices a `table_view` into a set of `table_view`s according to a set of indices.
 *
 * @ingroup copy_slice
 *
 * The returned views of `input` are constructed from an even number indices where
 * the `i`th returned `table_view` views the elements in `input` indicated by the range
 * `[indices[2*i], indices[(2*i)+1])`.
 *
 * For all `i` it is expected `indices[i] <= input.size()`
 * For all `i%2==0`, it is expected that `indices[i] <= indices[i+1]`
 *
 * @note It is the caller's responsibility to ensure that the returned views
 * do not outlive the viewed device memory.
 *
 * @code{.pseudo}
 * input:   [{10, 12, 14, 16, 18, 20, 22, 24, 26, 28},
 *           {50, 52, 54, 56, 58, 60, 62, 64, 66, 68}]
 * indices: {1, 3, 5, 9, 2, 4, 8, 8}
 * output:  [{{12, 14}, {20, 22, 24, 26}, {14, 16}, {}},
 *           {{52, 54}, {60, 22, 24, 26}, {14, 16}, {}}]
 * @endcode
 *
 * @throws std::invalid_argument if `indices` size is not even.
 * @throws std::invalid_argument When the values in the pair are strictly decreasing.
 * @throws std::out_of_range When any of the values in the pair don't belong to
 * the range [0, input.size()).
 *
 * @param input View of table to slice
 * @param indices Indices used to take slices of `input`
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Vector of views of `input` indicated by the ranges in `indices`
 */
std::vector<table_view> slice(table_view const& input,
                              std::span<size_type const> indices,
                              rmm::cuda_stream_view stream = cudf::get_default_stream());
/**
 * @ingroup copy_slice
 * @copydoc cudf::slice(table_view const&, std::span<size_type const>, rmm::cuda_stream_view stream)
 */
std::vector<table_view> slice(table_view const& input,
                              std::initializer_list<size_type> indices,
                              rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Splits a `column_view` into a set of `column_view`s according to a set of indices
 * derived from expected splits.
 *
 * @ingroup copy_split
 *
 * The returned view's of `input` are constructed from vector of splits, which indicates
 * where the split should occur. The `i`th returned `column_view` is sliced as
 * `[0, splits[i])` if `i`=0, else `[splits[i], input.size())` if `i` is the last view and
 * `[splits[i-1], splits[i]]` otherwise.
 *
 * For all `i` it is expected `splits[i] <= splits[i+1] <= input.size()`
 * For a `splits` size N, there will always be N+1 splits in the output
 *
 * @note It is the caller's responsibility to ensure that the returned views
 * do not outlive the viewed device memory.
 *
 * @code{.pseudo}
 * Example:
 * input:   {10, 12, 14, 16, 18, 20, 22, 24, 26, 28}
 * splits:  {2, 5, 9}
 * output:  {{10, 12}, {14, 16, 18}, {20, 22, 24, 26}, {28}}
 * @endcode
 *
 * @throws std::out_of_range if `splits` has end index > size of `input`.
 * @throws std::out_of_range When the value in `splits` is not in the range [0, input.size()).
 * @throws std::invalid_argument When the values in the `splits` are 'strictly decreasing'.
 *
 * @param input View of column to split
 * @param splits Indices where the view will be split
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return The set of requested views of `input` indicated by the `splits`
 */
std::vector<column_view> split(column_view const& input,
                               std::span<size_type const> splits,
                               rmm::cuda_stream_view stream = cudf::get_default_stream());
/**
 * @ingroup copy_split
 * @copydoc cudf::split(column_view const&, std::span<size_type const>, rmm::cuda_stream_view)
 */
std::vector<column_view> split(column_view const& input,
                               std::initializer_list<size_type> splits,
                               rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Splits a `table_view` into a set of `table_view`s according to a set of indices
 * derived from expected splits.
 *
 * @ingroup copy_split
 *
 * The returned views of `input` are constructed from vector of splits, which indicates
 * where the split should occur. The `i`th returned `table_view` is sliced as
 * `[0, splits[i])` if `i`=0, else `[splits[i], input.size())` if `i` is the last view and
 * `[splits[i-1], splits[i]]` otherwise.
 *
 * For all `i` it is expected `splits[i] <= splits[i+1] <= input.size()`
 * For a `splits` size N, there will always be N+1 splits in the output
 *
 * @note It is the caller's responsibility to ensure that the returned views
 * do not outlive the viewed device memory.
 *
 * @code{.pseudo}
 * Example:
 * input:   [{10, 12, 14, 16, 18, 20, 22, 24, 26, 28},
 *           {50, 52, 54, 56, 58, 60, 62, 64, 66, 68}]
 * splits:  {2, 5, 9}
 * output:  [{{10, 12}, {14, 16, 18}, {20, 22, 24, 26}, {28}},
 *           {{50, 52}, {54, 56, 58}, {60, 62, 64, 66}, {68}}]
 * @endcode
 *
 * @throws std::out_of_range if `splits` has end index > size of `input`.
 * @throws std::out_of_range When the value in `splits` is not in the range [0, input.size()).
 * @throws std::invalid_argument When the values in the `splits` are 'strictly decreasing'.
 *
 * @param input View of a table to split
 * @param splits Indices where the view will be split
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return The set of requested views of `input` indicated by the `splits`
 */
std::vector<table_view> split(table_view const& input,
                              std::span<size_type const> splits,
                              rmm::cuda_stream_view stream = cudf::get_default_stream());
/**
 * @ingroup copy_split
 * @copydoc cudf::split(table_view const&, std::span<size_type const>, rmm::cuda_stream_view)
 */
std::vector<table_view> split(table_view const& input,
                              std::initializer_list<size_type> splits,
                              rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief   Returns a new column, where each element is selected from either @p lhs or
 *          @p rhs based on the value of the corresponding element in @p boolean_mask
 *
 * Selects each element i in the output column from either @p rhs or @p lhs using the following
 * rule: `output[i] = (boolean_mask.valid(i) and boolean_mask[i]) ? lhs[i] : rhs[i]`
 *
 * @throws cudf::data_type_error if lhs and rhs are not of the same type
 * @throws std::invalid_argument if lhs and rhs are not of the same length
 * @throws cudf::data_type_error if boolean mask is not of type bool
 * @throws std::invalid_argument if boolean mask is not of the same length as lhs and rhs
 * @param lhs left-hand column_view
 * @param rhs right-hand column_view
 * @param boolean_mask column of `type_id::BOOL8` representing "left (true) / right (false)"
 * boolean for each element. Null element represents false.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns new column with the selected elements
 */
std::unique_ptr<column> copy_if_else(
  column_view const& lhs,
  column_view const& rhs,
  column_view const& boolean_mask,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief   Returns a new column, where each element is selected from either @p lhs or
 *          @p rhs based on the value of the corresponding element in @p boolean_mask
 *
 * Selects each element i in the output column from either @p rhs or @p lhs using the following
 * rule: `output[i] = (boolean_mask.valid(i) and boolean_mask[i]) ? lhs : rhs[i]`
 *
 * @throws cudf::data_type_error if lhs and rhs are not of the same type
 * @throws cudf::data_type_error if boolean mask is not of type bool
 * @throws std::invalid_argument if boolean mask is not of the same length as lhs and rhs
 * @param lhs left-hand scalar
 * @param rhs right-hand column_view
 * @param boolean_mask column of `type_id::BOOL8` representing "left (true) / right (false)"
 * boolean for each element. Null element represents false.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns new column with the selected elements
 */
std::unique_ptr<column> copy_if_else(
  scalar const& lhs,
  column_view const& rhs,
  column_view const& boolean_mask,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief   Returns a new column, where each element is selected from either @p lhs or
 *          @p rhs based on the value of the corresponding element in @p boolean_mask
 *
 * Selects each element i in the output column from either @p rhs or @p lhs using the following
 * rule: `output[i] = (boolean_mask.valid(i) and boolean_mask[i]) ? lhs[i] : rhs`
 *
 * @throws cudf::data_type_error if lhs and rhs are not of the same type
 * @throws cudf::data_type_error if boolean mask is not of type bool
 * @throws std::invalid_argument if boolean mask is not of the same length as lhs and rhs
 * @param lhs left-hand column_view
 * @param rhs right-hand scalar
 * @param boolean_mask column of `type_id::BOOL8` representing "left (true) / right (false)"
 * boolean for each element. Null element represents false.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns new column with the selected elements
 */
std::unique_ptr<column> copy_if_else(
  column_view const& lhs,
  scalar const& rhs,
  column_view const& boolean_mask,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief   Returns a new column, where each element is selected from either @p lhs or
 *          @p rhs based on the value of the corresponding element in @p boolean_mask
 *
 * Selects each element i in the output column from either @p rhs or @p lhs using the following
 * rule: `output[i] = (boolean_mask.valid(i) and boolean_mask[i]) ? lhs : rhs`
 *
 * @throws cudf::logic_error if boolean mask is not of type bool
 * @param lhs left-hand scalar
 * @param rhs right-hand scalar
 * @param boolean_mask column of `type_id::BOOL8` representing "left (true) / right (false)"
 * boolean for each element. null element represents false.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns new column with the selected elements
 */
std::unique_ptr<column> copy_if_else(
  scalar const& lhs,
  scalar const& rhs,
  column_view const& boolean_mask,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Scatters rows from the input table to rows of the output corresponding
 * to true values in a boolean mask.
 *
 * @ingroup copy_scatter
 *
 * The `i`th row of `input` will be written to the output table at the location
 * of the `i`th true value in `boolean_mask`. All other rows in the output will
 * equal the same row in `target`.
 *
 * `boolean_mask` should have number of `true`s <= number of rows in `input`.
 * If boolean mask is `true`, corresponding value in target is updated with
 * value from corresponding `input` column, else it is left untouched.
 *
 * @code{.pseudo}
 * Example:
 * input: {{1, 5, 6, 8, 9}}
 * boolean_mask: {true, false, false, false, true, true, false, true, true, false}
 * target:       {{   2,     2,     3,     4,    4,     7,    7,    7,    8,    10}}
 *
 * output:       {{   1,     2,     3,     4,    5,     6,    7,    8,    9,    10}}
 * @endcode
 *
 * @throws std::invalid_argument if input.num_columns() != target.num_columns()
 * @throws cudf::data_type_error if any `i`th input_column type != `i`th target_column type
 * @throws cudf::data_type_error if boolean_mask.type() != bool
 * @throws std::invalid_argument if boolean_mask.size() != target.num_rows()
 * @throws std::invalid_argument if number of `true` in `boolean_mask` > input.num_rows()
 *
 * @param input table_view (set of dense columns) to scatter
 * @param target table_view to modify with scattered values from `input`
 * @param boolean_mask column_view which acts as boolean mask
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the returned table
 *
 * @returns Returns a table by scattering `input` into `target` as per `boolean_mask`
 */
std::unique_ptr<table> boolean_mask_scatter(
  table_view const& input,
  table_view const& target,
  column_view const& boolean_mask,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Scatters scalar values to rows of the output corresponding
 * to true values in a boolean mask.
 *
 * @ingroup copy_scatter
 *
 * The `i`th scalar in `input` will be written to the ith column of the output
 * table at the location of every true value in `boolean_mask`.
 * All other rows in the output will equal the same row in `target`.
 *
 * @code{.pseudo}
 * Example:
 * input: {11}
 * boolean_mask: {true, false, false, false, true, true, false, true, true, false}
 * target:      {{   2,     2,     3,     4,    4,     7,    7,    7,    8,    10}}
 *
 * output:       {{   11,    2,     3,     4,   11,    11,    7,   11,   11,    10}}
 * @endcode
 *
 * @throws std::invalid_argument if input.size() != target.num_columns()
 * @throws cudf::data_type_error if any `i`th input_column type != `i`th target_column type
 * @throws cudf::data_type_error if boolean_mask.type() != bool
 * @throws std::invalid_argument if boolean_mask.size() != target.num_rows()
 *
 * @param input scalars to scatter
 * @param target table_view to modify with scattered values from `input`
 * @param boolean_mask column_view which acts as boolean mask
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the returned table
 *
 * @returns Returns a table by scattering `input` into `target` as per `boolean_mask`
 */
std::unique_ptr<table> boolean_mask_scatter(
  std::vector<std::reference_wrapper<scalar const>> const& input,
  table_view const& target,
  column_view const& boolean_mask,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Get the element at specified index from a column
 *
 * @warning This function is expensive (invokes a kernel launch). So, it is not
 * recommended to be used in performance sensitive code or inside a loop.
 *
 * @throws std::out_of_range if `index` is not within the range `[0, input.size())`
 *
 * @param input Column view to get the element from
 * @param index Index into `input` to get the element at
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Scalar containing the single value
 */
std::unique_ptr<scalar> get_element(
  column_view const& input,
  size_type index,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Indicates whether a row can be sampled more than once.
 */
enum class sample_with_replacement : bool {
  FALSE,  ///< A row can be sampled only once
  TRUE    ///< A row can be sampled more than once
};

/**
 * @brief Gather `n` samples from given `input` randomly
 *
 * @code{.pseudo}
 * Example:
 * input: {col1: {1, 2, 3, 4, 5}, col2: {6, 7, 8, 9, 10}}
 * n: 3
 * replacement: false
 *
 * output:       {col1: {3, 1, 4}, col2: {8, 6, 9}}
 *
 * replacement: true
 *
 * output:       {col1: {3, 1, 1}, col2: {8, 6, 6}}
 * @endcode
 *
 * @throws cudf::logic_error if `n` > `input.num_rows()` and `replacement` == FALSE.
 * @throws cudf::logic_error if `n` < 0.
 *
 * @param input View of a table to sample
 * @param n non-negative number of samples expected from `input`
 * @param replacement Allow or disallow sampling of the same row more than once
 * @param seed Seed value to initiate random number generator
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 *
 * @return Table containing samples from `input`
 */
std::unique_ptr<table> sample(
  table_view const& input,
  size_type const n,
  sample_with_replacement replacement = sample_with_replacement::FALSE,
  int64_t const seed                  = 0,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr   = cudf::get_current_device_resource_ref());

/**
 * @brief Checks if a column or its descendants have non-empty null rows
 *
 * @note This function is exact. If it returns `true`, there exists one or more
 * non-empty null elements.
 *
 * A LIST or STRING column might have non-empty rows that are marked as null.
 * A STRUCT OR LIST column might have child columns that have non-empty null rows.
 * Other types of columns are deemed incapable of having non-empty null rows.
 * E.g. Fixed width columns have no concept of an "empty" row.
 *
 * @param input The column which is (and whose descendants are) to be checked for
 * non-empty null rows.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return true If either the column or its descendants have non-empty null rows
 * @return false If neither the column or its descendants have non-empty null rows
 */
bool has_nonempty_nulls(column_view const& input,
                        rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Approximates if a column or its descendants *may* have non-empty null elements
 *
 * @note This function is approximate.
 * - `true`: Non-empty null elements could exist
 * - `false`: Non-empty null elements definitely do not exist
 *
 * False positives are possible, but false negatives are not.
 *
 * Compared to the exact `has_nonempty_nulls()` function, this function is typically
 * more efficient.
 *
 * Complexity:
 * - Best case: `O(count_descendants(input))`
 * - Worst case: `O(count_descendants(input)) * m`, where `m` is the number of rows in the largest
 * descendant
 *
 * @param input The column which is (and whose descendants are) to be checked for
 * non-empty null rows
 * @return true If either the column or its descendants have null rows
 * @return false If neither the column nor its descendants have null rows
 */
bool may_have_nonempty_nulls(column_view const& input);

/**
 * @brief Copy `input` into output while purging any non-empty null rows in the column or its
 * descendants.
 *
 * If the input column is not of compound type (LIST/STRING/STRUCT/DICTIONARY), the output will be
 * the same as input.
 *
 * The purge operation only applies directly to LIST and STRING columns, but it applies indirectly
 * to STRUCT/DICTIONARY columns as well, since these columns may have child columns that
 * are LIST or STRING.
 *
 * Examples:
 *
 * @code{.pseudo}
 * auto const lists   = lists_column_wrapper<int32_t>{ {0,1}, {2,3}, {4,5} }.release();
 * cudf::set_null_mask(lists->null_mask(), 1, 2, false);
 *
 * lists[1] is now null, but the lists child column still stores `{2,3}`.
 * The lists column contents will be:
 *   Validity: 101
 *   Offsets:  [0, 2, 4, 6]
 *   Child:    [0, 1, 2, 3, 4, 5]
 *
 * After purging the contents of the list's null rows, the column's contents will be:
 *   Validity: 101
 *   Offsets:  [0, 2, 2, 4]
 *   Child:    [0, 1, 4, 5]
 * @endcode
 *
 * @code{.pseudo}
 * auto const strings = strings_column_wrapper{ "AB", "CD", "EF" }.release();
 * cudf::set_null_mask(strings->null_mask(), 1, 2, false);
 *
 * strings[1] is now null, but the strings column still stores `"CD"`.
 * The lists column contents will be:
 *   Validity: 101
 *   Offsets:  [0, 2, 4, 6]
 *   Child:    [A, B, C, D, E, F]
 *
 * After purging the contents of the list's null rows, the column's contents
 * will be:
 *   Validity: 101
 *   Offsets:  [0, 2, 2, 4]
 *   Child:    [A, B, E, F]
 * @endcode
 *
 * @code{.pseudo}
 * auto const lists   = lists_column_wrapper<int32_t>{ {0,1}, {2,3}, {4,5} };
 * auto const structs = structs_column_wrapper{ {lists}, null_at(1) };
 *
 * structs[1].child is now null, but the lists column still stores `{2,3}`.
 * The lists column contents will be:
 *   Validity: 101
 *   Offsets:  [0, 2, 4, 6]
 *   Child:    [0, 1, 2, 3, 4, 5]
 *
 * After purging the contents of the list's null rows, the column's contents
 * will be:
 *   Validity: 101
 *   Offsets:  [0, 2, 2, 4]
 *   Child:    [0, 1, 4, 5]
 * @endcode
 *
 * @param input The column whose null rows are to be checked and purged
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A new column with equivalent contents to `input`, but with null rows purged
 */
std::unique_ptr<column> purge_nonempty_nulls(
  column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */
}  // namespace CUDF_EXPORT cudf
