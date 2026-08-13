/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief APIs for querying JSON values from a strings column using `get_json_object()`
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup json_object
 * @{
 */

/**
 * @brief Settings for `get_json_object()`.
 */
class get_json_object_options {
  // allow single quotes to represent strings in JSON
  bool allow_single_quotes = false;

  // individual string values are returned with quotes stripped.
  bool strip_quotes_from_single_strings = true;

  // Whether to return nulls when an object does not contain the requested field.
  bool missing_fields_as_nulls = false;

 public:
  /**
   * @brief Default constructor.
   */
  explicit get_json_object_options() = default;

  /**
   * @brief Returns true/false depending on whether single-quotes for representing strings
   * are allowed.
   *
   * @return true if single-quotes are allowed, false otherwise.
   */
  [[nodiscard]] CUDF_HOST_DEVICE inline bool get_allow_single_quotes() const
  {
    return allow_single_quotes;
  }

  /**
   * @brief Returns true/false depending on whether individually returned string values have
   * their quotes stripped.
   *
   * When set to true, if the return value for a given row is an individual string
   * (not an object, or an array of strings), strip the quotes from the string and return only the
   * contents of the string itself.  Example:
   *
   * @code{.pseudo}
   *
   * With strip_quotes_from_single_strings OFF:
   * Input  = {"a" : "b"}
   * Query  = $.a
   * Output = "b"
   *
   * With strip_quotes_from_single_strings ON:
   * Input  = {"a" : "b"}
   * Query  = $.a
   * Output = b
   *
   * @endcode
   *
   * @return true if individually returned string values have their quotes stripped.
   */
  [[nodiscard]] CUDF_HOST_DEVICE inline bool get_strip_quotes_from_single_strings() const
  {
    return strip_quotes_from_single_strings;
  }

  /**
   * @brief Whether a field not contained by an object is to be interpreted as null.
   *
   * When set to true, if an object is queried for a field it does not contain, a null is returned.
   *
   * @code{.pseudo}
   *
   * With missing_fields_as_nulls OFF:
   * Input  = {"a" : [{"x": "1", "y": "2"}, {"x": "3"}]}
   * Query  = $.a[*].y
   * Output = ["2"]
   *
   * With missing_fields_as_nulls ON:
   * Input  = {"a" : [{"x": "1", "y": "2"}, {"x": "3"}]}
   * Query  = $.a[*].y
   * Output = ["2", null]
   *
   * @endcode
   *
   * @return true if missing fields are interpreted as null.
   */
  [[nodiscard]] CUDF_HOST_DEVICE inline bool get_missing_fields_as_nulls() const
  {
    return missing_fields_as_nulls;
  }

  /**
   * @brief Set whether single-quotes for strings are allowed.
   *
   * @param _allow_single_quotes bool indicating desired behavior.
   */
  void set_allow_single_quotes(bool _allow_single_quotes)
  {
    allow_single_quotes = _allow_single_quotes;
  }

  /**
   * @brief Set whether individually returned string values have their quotes stripped.
   *
   * @param _strip_quotes_from_single_strings bool indicating desired behavior.
   */
  void set_strip_quotes_from_single_strings(bool _strip_quotes_from_single_strings)
  {
    strip_quotes_from_single_strings = _strip_quotes_from_single_strings;
  }

  /**
   * @brief Set whether missing fields are interpreted as null.
   *
   * @param _missing_fields_as_nulls bool indicating desired behavior.
   */
  void set_missing_fields_as_nulls(bool _missing_fields_as_nulls)
  {
    missing_fields_as_nulls = _missing_fields_as_nulls;
  }
};

/**
 * @brief Apply a JSONPath string to all rows in an input strings column.
 *
 * Applies a JSONPath string to an incoming strings column where each row in the column
 * is a valid json string.  The output is returned by row as a strings column.
 *
 * https://datatracker.ietf.org/doc/id/draft-goessner-dispatch-jsonpath-00.html
 * Implements only the operators: $ . [] *
 *
 * @throw std::invalid_argument if provided an invalid operator or an empty name
 *
 * @param col The input strings column. Each row must contain a valid json string
 * @param json_path The JSONPath string to be applied to each row
 * @param options Options for controlling the behavior of the function
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Resource for allocating device memory
 * @return New strings column containing the retrieved json object strings
 */
std::unique_ptr<cudf::column> get_json_object(
  cudf::strings_column_view const& col,
  cudf::string_scalar const& json_path,
  get_json_object_options options   = get_json_object_options{},
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace CUDF_EXPORT cudf
