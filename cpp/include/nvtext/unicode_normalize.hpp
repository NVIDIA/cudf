/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace CUDF_EXPORT nvtext {
/**
 * @addtogroup nvtext_normalize
 * @{
 * @file
 * @brief APIs for Unicode TR15 normalization of strings columns.
 */

/**
 * @brief Unicode normalization form per Unicode Standard Annex #15.
 *
 * https://unicode.org/reports/tr15/
 */
enum class unicode_normalization_form {
  NFD,   ///< Canonical Decomposition
  NFC,   ///< Canonical Decomposition followed by Canonical Composition
  NFKD,  ///< Compatibility Decomposition
  NFKC   ///< Compatibility Decomposition followed by Canonical Composition
};

/**
 * @brief Normalizer object for Unicode TR15 normalization.
 *
 * The normalizer is constructed from the contents of the Unicode Character
 * Database `UnicodeData.txt` file, loaded by the caller as a cudf table
 * (e.g. via `cudf::io::read_csv`).
 *
 * The file is published by the Unicode Consortium and can be downloaded from:
 * https://unicode.org/Public/15.1.0/ucd/UnicodeData.txt
 * It is a semicolon-delimited file with 15 fields per row.  The three fields
 * required by this API are field 0 (code point hex), field 3 (CCC), and
 * field 5 (decomposition mapping).
 *
 * The `unicode_data` table must contain exactly three columns in the following
 * order, corresponding to the fields above:
 *   - column[0]: STRING  Code point values as uppercase hex strings (e.g. "00C9")
 *   - column[1]: INT32   Canonical_Combining_Class (CCC) values in range [0, 254]
 *   - column[2]: STRING  Decomposition_Mapping field; empty string for identity
 *                        mappings, optionally prefixed with a compatibility tag
 *                        such as `<compat>`, `<font>`, `<wide>`, etc.
 *
 * @code{.cpp}
 * // typical workflow
 * cudf::io::csv_reader_options in_opts =
 *   cudf::io::csv_reader_options::builder(cudf::io::source_info("UnicodeData.txt"))
 *    .delimiter(';').header(-1)
 *    .use_cols_indexes({0, 3, 5})
 *    .dtypes({cudf::data_type{cudf::type_id::STRING}, cudf::data_type{cudf::type_id::INT32},
 *             cudf::data_type{cudf::type_id::STRING}});
 * auto const ud = cudf::io::read_csv(in_opts);
 * auto normalizer = nvtext::create_unicode_normalizer(ud.tbl->view(), NFKC);
 * // the following can be called repeatedly on different strings columns with the same normalizer
 * auto result = nvtext::normalize_unicode(input_strings, *normalizer);
 * @endcode
 *
 * Decomposition of Hangul syllables (U+AC00..U+D7A3) is performed
 * algorithmically per the Unicode standard and does not require entries
 * in the provided table.
 *
 * Composition exclusions (singletons, non-starter decompositions, and the
 * ~70 Unicode-specified explicit exclusions) are computed internally.
 */
struct unicode_normalizer {
  /**
   * @brief Construct a unicode_normalizer from UnicodeData.txt columns.
   *
   * @param unicode_data Table with three columns as described above
   * @param form Normalization form to apply
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate internal tables
   */
  unicode_normalizer(cudf::table_view const& unicode_data,
                     unicode_normalization_form form,
                     rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                     rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
  ~unicode_normalizer();

  struct unicode_normalizer_impl;
  std::unique_ptr<unicode_normalizer_impl> _impl;
};

/**
 * @brief Create a unicode_normalizer object.
 *
 * @see nvtext::unicode_normalizer
 *
 * @param unicode_data Table with three columns parsed from UnicodeData.txt
 * @param form Normalization form to apply
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate internal tables
 * @return Normalizer object to be reused across calls to nvtext::normalize_unicode
 */
std::unique_ptr<unicode_normalizer> create_unicode_normalizer(
  cudf::table_view const& unicode_data,
  unicode_normalization_form form,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Normalize a strings column using Unicode TR15 normalization.
 *
 * Input is UTF-8 encoded and output is UTF-8 encoded. Each string is
 * normalized independently. Null entries produce null output entries.
 *
 * @code{.pseudo}
 * cn = create_unicode_normalizer(unicode_table, NFKC)
 * s  = ["é", "ﬁ", "가", "hello"]
 * r  = normalize_unicode(s, cn)
 * // NFD:  ["é", "ﬁ",   "가", "hello"]
 * // NFC:  ["é",       "ﬁ",   "가",            "hello"]
 * // NFKD: ["é", "fi",  "가", "hello"]
 * // NFKC: ["é",       "fi",  "가",            "hello"]
 * @endcode
 *
 * During each call to nvtext::normalize_unicode, the amount of temporary
 * memory required is approximately 16x the input data size.
 *
 * @param input Strings column to normalize
 * @param normalizer Normalizer object created by nvtext::create_unicode_normalizer
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column of normalized UTF-8 strings
 */
std::unique_ptr<cudf::column> normalize_unicode(
  cudf::strings_column_view const& input,
  unicode_normalizer const& normalizer,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT nvtext
