/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/span.hpp>

#include <random>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

template <typename T, typename SourceElementT = T>
using column_wrapper =
  std::conditional_t<std::is_same_v<T, cudf::string_view>,
                     cudf::test::strings_column_wrapper,
                     cudf::test::fixed_width_column_wrapper<T, SourceElementT>>;
using column     = cudf::column;
using table      = cudf::table;
using table_view = cudf::table_view;

// Global environment for temporary files
extern cudf::test::TempDirTestEnvironment* const temp_env;

// Writes `tbl` to a temp Parquet file. If `column_names` is non-empty, it sets the top-level
// column names in the file metadata.
[[nodiscard]] std::string write_parquet_temp_file(cudf::table_view const& tbl,
                                                  std::string_view const filename,
                                                  std::vector<std::string> column_names = {},
                                                  std::vector<int> field_ids            = {});

// TODO: Replace with `NumericTypes` when unsigned support is added. Issue #5352
using SupportedTypes = cudf::test::Types<int8_t, int16_t, int32_t, int64_t, bool, float, double>;

using ComparableAndFixedTypes =
  cudf::test::Concat<cudf::test::ComparableTypes, cudf::test::FixedPointTypes>;

using SupportedTimestampTypes =
  cudf::test::Types<cudf::timestamp_ms, cudf::timestamp_us, cudf::timestamp_ns>;

using ByteLikeTypes = cudf::test::Types<int8_t, char, uint8_t, unsigned char, std::byte>;

// These chrono types are not supported because parquet writer does not have a type to represent
// them.
using UnsupportedChronoTypes =
  cudf::test::Types<cudf::timestamp_s, cudf::duration_D, cudf::duration_s>;

// Support types for AST expression evaluator
using SupportedTestTypesAST =
  cudf::test::RemoveIf<cudf::test::ContainedIn<UnsupportedChronoTypes>, ComparableAndFixedTypes>;

// JIT does not yet support fixed point types
using SupportedTestTypesJIT =
  cudf::test::RemoveIf<cudf::test::ContainedIn<cudf::test::FixedPointTypes>, SupportedTestTypesAST>;

// removing duration_D, duration_s, and timestamp_s as they don't appear to be supported properly.
// see definition of UnsupportedChronoTypes above.
using DeltaDecimalTypes = cudf::test::Types<numeric::decimal32, numeric::decimal64>;
using DeltaBinaryTypes =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::ChronoTypes, DeltaDecimalTypes>;
using SupportedDeltaTestTypes =
  cudf::test::RemoveIf<cudf::test::ContainedIn<UnsupportedChronoTypes>, DeltaBinaryTypes>;

//////////////////////////////////////////////////////////////////////
// Test fixtures

// Base test fixture for tests
struct ParquetWriterTest : public cudf::test::BaseFixture {};

// Base test fixture for tests
struct ParquetReaderTest : public cudf::test::BaseFixture {};

////////////////////////////////////////////////////////////////////

// Generates a vector of uniform random values of type T
template <typename T>
std::vector<T> random_values(size_t size);

template <typename T>
std::unique_ptr<cudf::table> create_random_fixed_table(cudf::size_type num_columns,
                                                       cudf::size_type num_rows,
                                                       bool include_validity);

template <typename T>
std::unique_ptr<cudf::table> create_compressible_fixed_table(cudf::size_type num_columns,
                                                             cudf::size_type num_rows,
                                                             cudf::size_type period,
                                                             bool include_validity);

// this function replicates the "list_gen" function in
// python/cudf/cudf/tests/test_parquet.py
template <typename T>
std::unique_ptr<cudf::column> make_parquet_list_list_col(
  int skip_rows, int num_rows, int lists_per_row, int list_size, bool include_validity);

// given a datasource pointing to a parquet file, read the footer
// of the file to populate the FileMetaData pointed to by file_meta_data.
// throws cudf::logic_error if the file or metadata is invalid.
void read_footer(std::unique_ptr<cudf::io::datasource> const& source,
                 cudf::io::parquet::FileMetaData* file_meta_data);

// returns the number of bits used for dictionary encoding data at the given page location.
// this assumes the data is uncompressed.
// throws cudf::logic_error if the page_loc data is invalid.
int read_dict_bits(std::unique_ptr<cudf::io::datasource> const& source,
                   cudf::io::parquet::PageLocation const& page_loc);

// read column index from datasource at location indicated by chunk,
// parse and return as a ColumnIndex struct.
// throws cudf::logic_error if the chunk data is invalid.
cudf::io::parquet::ColumnIndex read_column_index(
  std::unique_ptr<cudf::io::datasource> const& source, cudf::io::parquet::ColumnChunk const& chunk);

// read offset index from datasource at location indicated by chunk,
// parse and return as an OffsetIndex struct.
// throws cudf::logic_error if the chunk data is invalid.
cudf::io::parquet::OffsetIndex read_offset_index(
  std::unique_ptr<cudf::io::datasource> const& source, cudf::io::parquet::ColumnChunk const& chunk);

// Return as a Statistics from the column chunk
cudf::io::parquet::Statistics const& get_statistics(cudf::io::parquet::ColumnChunk const& chunk);

// read page header from datasource at location indicated by page_loc,
// parse and return as a PageHeader struct.
// throws cudf::logic_error if the page_loc data is invalid.
cudf::io::parquet::PageHeader read_page_header(std::unique_ptr<cudf::io::datasource> const& source,
                                               cudf::io::parquet::PageLocation const& page_loc);

// make a random validity iterator
inline auto random_validity(std::mt19937& engine)
{
  static std::bernoulli_distribution bn(0.7f);
  return cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(engine); });
}

// make a random list<T> column
template <typename T>
std::unique_ptr<cudf::column> make_parquet_list_col(std::mt19937& engine,
                                                    int num_rows,
                                                    int max_vals_per_row,
                                                    bool include_validity);

// return vector of random strings
std::vector<std::string> string_values(std::mt19937& engine, int num_rows, int max_string_len);

// make a random list<string> column, with random string lengths of 0..max_string_len,
// and up to max_vals_per_row strings in each list.
std::unique_ptr<cudf::column> make_parquet_string_list_col(std::mt19937& engine,
                                                           int num_rows,
                                                           int max_vals_per_row,
                                                           int max_string_len,
                                                           bool include_validity);

template <typename T>
std::pair<cudf::table, std::string> create_parquet_typed_with_stats(std::string const& filename);

// A source for row-group output ordering tests: a parquet file plus the in-memory ground-truth
// table used for slicing. The single int32 column is laid out so each row group holds a disjoint
// value range (value = file_id * 10000 + rg_idx * rows_per_row_group + row), so any reorder by the
// reader is detectable via table equality on the concatenated output.
struct ordered_rg_source {
  std::unique_ptr<cudf::table> table;
  std::string filepath;
  cudf::size_type num_row_groups;
  cudf::size_type rows_per_row_group;
};

// write a parquet file with num_row_groups row groups of rows_per_row_group int32 rows each,
// emitting row-group-level statistics so predicate pushdown can prune.
ordered_rg_source make_ordered_rg_source(int file_id,
                                         cudf::size_type num_row_groups,
                                         cudf::size_type rows_per_row_group = 100);

// return a view over the rows of row group rg_idx in src.table.
cudf::table_view get_row_group_slice(ordered_rg_source const& src, cudf::size_type rg_idx);

// build the expected reader output for a per-source row-group selection: source-major, and in the
// given order within each source (repeated indices kept, an empty inner vector contributes
// nothing). sources[i] pairs with rg_selection[i]; pointers must be non-null.
std::unique_ptr<cudf::table> build_expected_ordered_table(
  cudf::host_span<ordered_rg_source const* const> sources,
  cudf::host_span<std::vector<cudf::size_type> const> rg_selection);

int32_t compare_binary(std::vector<uint8_t> const& v1,
                       std::vector<uint8_t> const& v2,
                       cudf::io::parquet::Type ptype,
                       std::optional<cudf::io::parquet::ConvertedType> const& ctype);

void expect_compression_stats_empty(std::shared_ptr<cudf::io::writer_compression_statistics> stats);

// =============================================================================
// ---- test data for stats sort order tests
// need at least 3 pages, and min page count is 5000, so need at least 15000 values.
// use 20000 to be safe.
static constexpr int num_ordered_rows            = 20000;
static constexpr int page_size_for_ordered_tests = 5000;

namespace testdata {

// ----- most numerics
template <typename T>
std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>,
                 cudf::test::fixed_width_column_wrapper<T>>
ascending();

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>,
                 cudf::test::fixed_width_column_wrapper<T>>
descending();

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>,
                 cudf::test::fixed_width_column_wrapper<T>>
unordered();

// ----- bool

template <typename T>
std::enable_if_t<std::is_same_v<T, bool>, cudf::test::fixed_width_column_wrapper<bool>> ascending();

template <typename T>
std::enable_if_t<std::is_same_v<T, bool>, cudf::test::fixed_width_column_wrapper<bool>>
descending();

template <typename T>
std::enable_if_t<std::is_same_v<T, bool>, cudf::test::fixed_width_column_wrapper<bool>> unordered();

// ----- fixed point types

template <typename T>
std::enable_if_t<cudf::is_fixed_point<T>(), cudf::test::fixed_width_column_wrapper<T>> ascending();

template <typename T>
std::enable_if_t<cudf::is_fixed_point<T>(), cudf::test::fixed_width_column_wrapper<T>> descending();

template <typename T>
std::enable_if_t<cudf::is_fixed_point<T>(), cudf::test::fixed_width_column_wrapper<T>> unordered();

// ----- chrono types
// ----- timstamp

template <typename T>
std::enable_if_t<cudf::is_timestamp<T>(), cudf::test::fixed_width_column_wrapper<T>> ascending();

template <typename T>
std::enable_if_t<cudf::is_timestamp<T>(), cudf::test::fixed_width_column_wrapper<T>> descending();

template <typename T>
std::enable_if_t<cudf::is_timestamp<T>(), cudf::test::fixed_width_column_wrapper<T>> unordered();

// ----- duration

template <typename T>
std::enable_if_t<cudf::is_duration<T>(), cudf::test::fixed_width_column_wrapper<T>> ascending();

template <typename T>
std::enable_if_t<cudf::is_duration<T>(), cudf::test::fixed_width_column_wrapper<T>> descending();

template <typename T>
std::enable_if_t<cudf::is_duration<T>(), cudf::test::fixed_width_column_wrapper<T>> unordered();

// ----- string_view

template <typename T>
std::enable_if_t<std::is_same_v<T, cudf::string_view>, cudf::test::strings_column_wrapper>
ascending();

template <typename T>
std::enable_if_t<std::is_same_v<T, cudf::string_view>, cudf::test::strings_column_wrapper>
descending();

template <typename T>
std::enable_if_t<std::is_same_v<T, cudf::string_view>, cudf::test::strings_column_wrapper>
unordered();

}  // namespace testdata
