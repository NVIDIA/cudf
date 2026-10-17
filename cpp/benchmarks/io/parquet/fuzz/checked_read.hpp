/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

/**
 * @brief Parquet readers
 */
enum class parquet_reader_kind {
  WHOLE,       ///< `read_parquet`
  CHUNKED,     ///< `chunked_parquet_reader`
  HYBRID_SCAN  ///< `hybrid_scan_multifile`, in passes of row groups, each read in chunks
};

/**
 * @brief Settings of a read of Parquet files
 */
struct read_settings {
  parquet_reader_kind reader;
  std::size_t chunk_read_limit = 0;         ///< Output chunk size limit of the chunked readers
  std::size_t pass_read_limit  = 0;         ///< Pass memory limit of the chunked readers
  cudf::size_type skip_rows    = 0;         ///< Rows to skip, for the whole and chunked readers
  std::optional<cudf::size_type> num_rows;  ///< Rows to read, for the whole and chunked readers
};

/**
 * @brief Describes `read`, for failure messages
 */
std::string to_string(read_settings const& read);

/**
 * @brief Reads `paths` with `read`, and checks that the result is the rows of `expected` that the
 * read selects, failing the current test where it is not
 *
 * Compares each chunk that the chunked readers return with its rows of `expected`, so that the
 * null counts of each chunk are checked too. Columns are compared with
 * `CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT`, with no ULP tolerance for floating point values.
 *
 * @throws cudf::logic_error if a chunked reader returns an empty chunk after its first one (of a
 * pass, for the hybrid scan reader)
 */
void checked_read(read_settings const& read,
                  std::vector<std::string> const& paths,
                  cudf::table_view const& expected);
