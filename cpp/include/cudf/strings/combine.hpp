/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file strings/combine.hpp
 * @brief Strings APIs for concatenate and join
 */

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_combine
 * @{
 */

/**
 * @brief Setting for specifying how separators are added with
 * null strings elements.
 */
enum class separator_on_nulls {
  YES,  ///< Always add separators between elements
  NO    ///< Do not add separators if an element is null
};

/**
 * @brief Setting for specifying what will be output from `join_list_elements` when an input list
 * is empty.
 */
enum class output_if_empty_list {
  EMPTY_STRING,  ///< Empty list will result in empty string
  NULL_ELEMENT   ///< Empty list will result in a null
};

/**
 * @brief Concatenates all strings in the column into one new string delimited
 * by an optional separator string.
 *
 * This returns a column with one string. Any null entries are ignored unless
 * the @p narep parameter specifies a replacement string.
 *
 * @code{.pseudo}
 * Example:
 * s = ['aa', null, '', 'zz' ]
 * r = join_strings(s,':','_')
 * r is ['aa:_::zz']
 * @endcode
 *
 * @throw cudf::logic_error if separator is not valid.
 *
 * @param input Strings for this operation
 * @param separator String that should inserted between each string.
 *        Default is an empty string.
 * @param narep String to replace any null strings found.
 *        Default of invalid-scalar will ignore any null entries.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column containing one string.
 */
std::unique_ptr<column> join_strings(
  strings_column_view const& input,
  string_scalar const& separator    = string_scalar(""),
  string_scalar const& narep        = string_scalar("", false),
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Concatenates a list of strings columns using separators for each row
 * and returns the result as a strings column.
 *
 * Each new string is created by concatenating the strings from the same
 * row delimited by the row separator provided for that row. The following rules
 * are applicable:
 *
 * - If row separator for a given row is null, output column for that row is null, unless
 *   there is a valid @p separator_narep
 * - The separator is applied between two output row values if the @p separate_nulls
 *   is `YES` or only between valid rows if @p separate_nulls is `NO`.
 * - If @p separator_narep and @p col_narep are both valid, the output column is always
 *   non nullable
 *
 * @code{.pseudo}
 * Example:
 * c0   = ['aa', null, '',  'ee',  null, 'ff']
 * c1   = [null, 'cc', 'dd', null, null, 'gg']
 * c2   = ['bb', '',   null, null, null, 'hh']
 * sep  = ['::', '%%', '^^', '!',  '*',  null]
 * out = concatenate({c0, c1, c2}, sep)
 * // all rows have at least one null or sep[i]==null
 * out is [null, null, null, null, null, null]
 *
 * sep_rep = '+'
 * out = concatenate({c0, c1, c2}, sep, sep_rep)
 * // all rows with at least one null output as null
 * out is [null, null, null, null, null, 'ff+gg+hh']
 *
 * col_narep = '-'
 * sep_na = non-valid scalar
 * out = concatenate({c0, c1, c2}, sep, sep_na, col_narep)
 * // only the null entry in the sep column produces a null row
 * out is ['aa::-::bb', '-%%cc%%', '^^dd^^-', 'ee!-!-', '-*-*-', null]
 *
 * col_narep = ''
 * out = concatenate({c0, c1, c2}, sep, sep_rep, col_narep, separator_on_nulls:NO)
 * // parameter suppresses separator for null rows
 * out is ['aa::bb', 'cc%%', '^^dd', 'ee', '', 'ff+gg+hh']
 * @endcode
 *
 * @throw cudf::logic_error if no input columns are specified - table view is empty
 * @throw cudf::logic_error if input columns are not all strings columns.
 * @throw cudf::logic_error if the number of rows from @p separators and @p strings_columns
 *                          do not match
 *
 * @param strings_columns List of strings columns to concatenate
 * @param separators Strings column that provides the separator for a given row
 * @param separator_narep String to replace a null separator for a given row.
 *        Default of invalid-scalar means no row separator value replacements.
 * @param col_narep String that should be used in place of any null strings found in any column.
 *        Default of invalid-scalar means no null column value replacements.
 * @param separate_nulls If YES, then the separator is included for null rows
 *        if `col_narep` is valid.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Resource for allocating device memory
 * @return New column with concatenated results
 */
std::unique_ptr<column> concatenate(
  table_view const& strings_columns,
  strings_column_view const& separators,
  string_scalar const& separator_narep = string_scalar("", false),
  string_scalar const& col_narep       = string_scalar("", false),
  separator_on_nulls separate_nulls    = separator_on_nulls::YES,
  rmm::cuda_stream_view stream         = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr    = cudf::get_current_device_resource_ref());

/**
 * @brief Row-wise concatenates the given list of strings columns and
 * returns a single strings column result.
 *
 * Each new string is created by concatenating the strings from the same
 * row delimited by the separator provided.
 *
 * Any row with a null entry will result in the corresponding output
 * row to be null entry unless a narep string is specified to be used
 * in its place.
 *
 * If @p separate_nulls is set to `NO` and @p narep is valid then
 * separators are not added to the output between null elements.
 * Otherwise, separators are always added if @p narep is valid.
 *
 * More than one column must be specified in the input @p strings_columns
 * table.
 *
 * @code{.pseudo}
 * Example:
 * s1 = ['aa', null, '', 'dd']
 * s2 = ['', 'bb', 'cc', null]
 * out = concatenate({s1, s2})
 * out is ['aa', null, 'cc', null]
 *
 * out = concatenate({s1, s2}, ':', '_')
 * out is ['aa:', '_:bb', ':cc', 'dd:_']
 *
 * out = concatenate({s1, s2}, ':', '', separator_on_nulls::NO)
 * out is ['aa:', 'bb', ':cc', 'dd']
 * @endcode
 *
 * @throw cudf::logic_error if input columns are not all strings columns.
 * @throw cudf::logic_error if separator is not valid.
 * @throw cudf::logic_error if only one column is specified
 *
 * @param strings_columns List of string columns to concatenate
 * @param separator String that should inserted between each string from each row.
 *        Default is an empty string.
 * @param narep String to replace any null strings found in any column.
 *        Default of invalid-scalar means any null entry in any column will
 *        produces a null result for that row.
 * @param separate_nulls If YES, then the separator is included for null rows if `narep` is valid
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column with concatenated results
 */
std::unique_ptr<column> concatenate(
  table_view const& strings_columns,
  string_scalar const& separator    = string_scalar(""),
  string_scalar const& narep        = string_scalar("", false),
  separator_on_nulls separate_nulls = separator_on_nulls::YES,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Given a lists column of strings (each row is a list of strings), concatenates the strings
 * within each row and returns a single strings column result.
 *
 * Each new string is created by concatenating the strings from the same row (same list element)
 * delimited by the row separator provided in the @p separators strings column.
 *
 * A null list row will always result in a null string in the output row. Any non-null list row
 * having a null element will result in the corresponding output row to be null unless a valid
 * @p string_narep scalar is provided to be used in its place. Any null row in the @p separators
 * column will also result in a null output row unless a valid @p separator_narep scalar is provided
 * to be used in place of the null separators.
 *
 * If @p separate_nulls is set to `NO` and @p string_narep is valid then separators are not added to
 * the output between null elements. Otherwise, separators are always added if @p string_narep is
 * valid.
 *
 * If @p empty_list_policy is set to `EMPTY_STRING`, any row that is an empty list will result in
 * an empty output string. Otherwise, the output will be a null.
 *
 * In the special case when the input list row contains all null elements, the output will be the
 * same as in case of empty input list regardless of @p string_narep and @p separate_nulls values.
 *
 * @code{.pseudo}
 * Example:
 * s = [ ['aa', 'bb', 'cc'], null, ['', 'dd'], ['ee', null], ['ff', 'gg'] ]
 * sep  = ['::', '%%',  '!',  '*',  null]
 *
 * out = join_list_elements(s, sep)
 * out is ['aa::bb::cc', null, '!dd', null, null]
 *
 * out = join_list_elements(s, sep, ':', '_')
 * out is ['aa::bb::cc', null,  '!dd', 'ee*_', 'ff:gg']
 *
 * out = join_list_elements(s, sep, ':', '', separator_on_nulls::NO)
 * out is ['aa::bb::cc', null,  '!dd', 'ee', 'ff:gg']
 * @endcode
 *
 * @throw cudf::logic_error if input column is not lists of strings column.
 * @throw cudf::logic_error if the number of rows from `separators` and `lists_strings_column` do
 *        not match
 *
 * @param lists_strings_column Column containing lists of strings to concatenate
 * @param separators Strings column that provides separators for concatenation
 * @param separator_narep String that should be used to replace a null separator.
 *        Default is an invalid-scalar denoting that rows containing null separator will result in
 *        a null string in the corresponding output rows.
 * @param string_narep String to replace null strings in any non-null list row.
 *        Default is an invalid-scalar denoting that list rows containing null strings will result
 *        in a null string in the corresponding output rows.
 * @param separate_nulls If YES, then the separator is included for null rows if `narep` is valid
 * @param empty_list_policy If set to EMPTY_STRING, any input row that is an empty list will
 *        result in an empty string. Otherwise, it will result in a null.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column with concatenated results
 */
std::unique_ptr<column> join_list_elements(
  lists_column_view const& lists_strings_column,
  strings_column_view const& separators,
  string_scalar const& separator_narep   = string_scalar("", false),
  string_scalar const& string_narep      = string_scalar("", false),
  separator_on_nulls separate_nulls      = separator_on_nulls::YES,
  output_if_empty_list empty_list_policy = output_if_empty_list::EMPTY_STRING,
  rmm::cuda_stream_view stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref());

/**
 * @brief Given a lists column of strings (each row is a list of strings), concatenates the strings
 * within each row and returns a single strings column result.
 *
 * Each new string is created by concatenating the strings from the same row (same list element)
 * delimited by the @p separator provided.
 *
 * A null list row will always result in a null string in the output row. Any non-null list row
 * having a null element will result in the corresponding output row to be null unless a
 * @p narep string is specified to be used in its place.
 *
 * If @p separate_nulls is set to `NO` and @p narep is valid then separators are not added to the
 * output between null elements. Otherwise, separators are always added if @p narep is valid.
 *
 * If @p empty_list_policy is set to `EMPTY_STRING`, any row that is an empty list will result in
 * an empty output string. Otherwise, the output will be a null.
 *
 * In the special case when the input list row contains all null elements, the output will be the
 * same as in case of empty input list regardless of @p narep and @p separate_nulls values.
 *
 * @code{.pseudo}
 * Example:
 * s = [ ['aa', 'bb', 'cc'], null, ['', 'dd'], ['ee', null], ['ff'] ]
 *
 * out = join_list_elements(s)
 * out is ['aabbcc', null, 'dd', null, 'ff']
 *
 * out = join_list_elements(s, ':', '_')
 * out is ['aa:bb:cc', null,  ':dd', 'ee:_', 'ff']
 *
 * out = join_list_elements(s, ':', '', separator_on_nulls::NO)
 * out is ['aa:bb:cc', null,  ':dd', 'ee', 'ff']
 * @endcode
 *
 * @throw cudf::logic_error if input column is not lists of strings column.
 * @throw cudf::logic_error if separator is not valid.
 *
 * @param lists_strings_column Column containing lists of strings to concatenate
 * @param separator String to insert between strings of each list row.
 *        Default is an empty string.
 * @param narep String to replace null strings in any non-null list row.
 *        Default is an invalid-scalar denoting that list rows containing null strings will result
 *        in a null string in the corresponding output rows.
 * @param separate_nulls If YES, then the separator is included for null rows if `narep` is valid
 * @param empty_list_policy If set to EMPTY_STRING, any input row that is an empty list will result
 *        in an empty string. Otherwise, it will result in a null.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column with concatenated results
 */
std::unique_ptr<column> join_list_elements(
  lists_column_view const& lists_strings_column,
  string_scalar const& separator         = string_scalar(""),
  string_scalar const& narep             = string_scalar("", false),
  separator_on_nulls separate_nulls      = separator_on_nulls::YES,
  output_if_empty_list empty_list_policy = output_if_empty_list::EMPTY_STRING,
  rmm::cuda_stream_view stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
