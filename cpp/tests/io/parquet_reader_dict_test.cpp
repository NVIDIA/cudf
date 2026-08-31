/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <array>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr cudf::size_type num_rows              = 5000;
constexpr cudf::size_type cardinality           = num_rows / 10;
constexpr cudf::size_type row_group_size        = 1000;
constexpr unsigned int seed                     = 0xcece;
constexpr unsigned int list_strings_seed        = seed ^ 0xA5701DUL;
constexpr cudf::size_type max_elements_per_list = 8;
constexpr double null_probability               = 0.1;

// Per-distinct-value prefixes deliberately mixing ASCII with multi-byte UTF-8 (accented Latin,
// Greek, CJK, and an emoji) so the transcode/fallback paths are exercised on non-ASCII keys. The
// numeric suffix in `make_value_string` keeps every distinct value a distinct string, preserving
// the intended cardinality.
std::array<char const*, 6> const utf8_prefixes{
  "str", "café", "naïve", "Ωμέγα", "日本語", "🚀rocket"};

// Map a dictionary value to a UTF-8 string. Distinct values map to distinct
// strings via the numeric suffix.
std::string make_value_string(int value)
{
  return std::string{utf8_prefixes[value % utf8_prefixes.size()]} + "_" + std::to_string(value);
}

cudf::test::strings_column_wrapper make_low_cardinality_strings()
{
  std::mt19937 engine(seed);
  std::uniform_int_distribution<int> value_dist(0, cardinality - 1);
  std::bernoulli_distribution null_dist(null_probability);

  std::vector<std::string> strings(num_rows);
  std::vector<bool> valids(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    strings[i] = make_value_string(value_dist(engine));
    valids[i]  = not null_dist(engine);
  }

  return cudf::test::strings_column_wrapper(strings.begin(), strings.end(), valids.begin());
}

std::unique_ptr<cudf::column> make_low_cardinality_lists_of_strings()
{
  std::mt19937 engine(list_strings_seed);
  std::uniform_int_distribution<int> value_dist(0, cardinality - 1);
  std::uniform_int_distribution<int> len_dist(0, max_elements_per_list);

  std::vector<cudf::size_type> offsets;
  offsets.reserve(num_rows + 1);
  offsets.push_back(0);
  std::vector<std::string> child_strings;
  for (cudf::size_type row = 0; row < num_rows; ++row) {
    auto const len = len_dist(engine);
    for (int e = 0; e < len; ++e) {
      child_strings.push_back(make_value_string(value_dist(engine)));
    }
    offsets.push_back(offsets.back() + static_cast<cudf::size_type>(len));
  }

  auto child = cudf::test::strings_column_wrapper(child_strings.begin(), child_strings.end());
  auto offsets_col =
    cudf::test::fixed_width_column_wrapper<int32_t>(offsets.begin(), offsets.end()).release();

  return cudf::make_lists_column(
    num_rows, std::move(offsets_col), child.release(), 0, rmm::device_buffer{});
}

void write_parquet(cudf::table_view const& input, std::string const& filepath)
{
  // Produce row groups consisting of `row_group_size` rows, with a single (non-chunked) write. Row
  // groups are built from whole page fragments, so `max_page_fragment_size` must also be lowered to
  // `row_group_size`
  // -- otherwise the default 5000-row fragment would force row groups to snap to multiples of 5000
  // instead of the requested size.
  auto const options =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input)
      .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
      .compression(cudf::io::compression_type::NONE)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .row_group_size_rows(row_group_size)
      .max_page_fragment_size(row_group_size)
      .build();
  cudf::io::write_parquet(options);
}

cudf::io::table_with_metadata read_parquet_as_dict(std::string const& filepath)
{
  auto const read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .output_dict_columns(true)
                           .build();
  return cudf::io::read_parquet(read_opts);
}

// A simple INT32 iota column, used as a non-string / filter key column alongside the strings.
cudf::test::fixed_width_column_wrapper<int32_t> make_int_key_column()
{
  std::vector<int32_t> keys(num_rows);
  std::iota(keys.begin(), keys.end(), 0);
  return cudf::test::fixed_width_column_wrapper<int32_t>(keys.begin(), keys.end());
}

// Build a string column whose first half is low-cardinality (a small dictionary that fits the
// writer's dictionary budget) and whose second half is all-distinct (a dictionary too large for
// the budget). cuDF chooses dictionary use per column-chunk, so under an ADAPTIVE policy with a
// small `max_dictionary_size` the resulting column carries a mix of dictionary-encoded and
// PLAIN-encoded chunks -- which makes it ineligible for the direct transcode fast path.
cudf::test::strings_column_wrapper make_mixed_encoding_strings()
{
  std::vector<std::string> strings(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    bool const low_cardinality_region = i < num_rows / 2;
    int const value                   = low_cardinality_region ? (i % 16) : i;
    strings[i]                        = make_value_string(value);
  }
  return cudf::test::strings_column_wrapper(strings.begin(), strings.end());
}

// Like `write_parquet`, but with an ADAPTIVE dictionary policy and a caller-supplied dictionary
// budget so the writer falls back to PLAIN encoding for row groups whose dictionary exceeds it.
void write_parquet_adaptive(cudf::table_view const& input,
                            std::string const& filepath,
                            size_t max_dict_size)
{
  auto const options =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input)
      .dictionary_policy(cudf::io::dictionary_policy::ADAPTIVE)
      .max_dictionary_size(max_dict_size)
      .compression(cudf::io::compression_type::NONE)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .row_group_size_rows(row_group_size)
      .max_page_fragment_size(row_group_size)
      .build();
  cudf::io::write_parquet(options);
}

}  // namespace

struct ParquetReaderDictTest : public cudf::test::BaseFixture {};

// A flat string column that is fully dictionary-encoded in every row group should be returned
// as a DICTIONARY32 column when `output_dict_columns` is enabled, and the decoded keys
// should match the original input.
TEST_F(ParquetReaderDictTest, FlatStringDictTranscode)
{
  auto input_col = make_low_cardinality_strings();

  auto const input_tbl = cudf::table_view{{input_col}};
  auto const filepath  = temp_env->get_temp_filepath("FlatStringDictTranscode.parquet");
  write_parquet(input_tbl, filepath);

  auto const dict_input      = cudf::dictionary::encode(input_col);
  auto const dict_input_view = cudf::dictionary_column_view(dict_input->view());
  auto const decoded_input   = cudf::dictionary::decode(dict_input_view);

  auto const read_table = read_parquet_as_dict(filepath).tbl;
  ASSERT_EQ(read_table->num_rows(), num_rows);
  ASSERT_EQ(read_table->num_columns(), 1);

  auto const read_col = read_table->view().column(0);
  ASSERT_EQ(read_col.type().id(), cudf::type_id::DICTIONARY32)
    << "Expected the reader to produce a DICTIONARY32 column when output_dict_columns is on";

  cudf::dictionary_column_view dict_read_view(read_col);
  auto const decoded_read = cudf::dictionary::decode(dict_read_view);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input_col, decoded_read->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded_input->view(), decoded_read->view());
}

// When the option is not set, the reader should still produce a plain STRING column, regardless
// of whether the source file is fully dictionary-encoded.
TEST_F(ParquetReaderDictTest, FlatStringNoTranscodeByDefault)
{
  auto input_col = make_low_cardinality_strings();

  auto const input_tbl = cudf::table_view{{input_col}};
  auto const filepath  = temp_env->get_temp_filepath("FlatStringNoTranscodeByDefault.parquet");
  write_parquet(input_tbl, filepath);

  auto const read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).build();
  auto const read_table = cudf::io::read_parquet(read_opts).tbl;

  ASSERT_EQ(read_table->num_columns(), 1);
  auto const read_col = read_table->view().column(0);
  ASSERT_EQ(read_col.type().id(), cudf::type_id::STRING);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input_col, read_col);
}

// List<string> is not eligible for Parquet-dictionary → DICTIONARY32 transcode (flat string columns
// only). With `output_dict_columns` enabled, the reader still round-trips as LIST<STRING>.
TEST_F(ParquetReaderDictTest, ListOfStringsDictEncodedWithTryOutputDictOption)
{
  auto list_col = make_low_cardinality_lists_of_strings();

  auto const input_tbl = cudf::table_view{{list_col->view()}};
  auto const filepath =
    temp_env->get_temp_filepath("ListOfStringsDictEncodedWithTryOutputDictOption.parquet");
  write_parquet(input_tbl, filepath);

  auto const read_table = read_parquet_as_dict(filepath).tbl;
  ASSERT_EQ(read_table->num_rows(), input_tbl.num_rows());
  ASSERT_EQ(read_table->num_columns(), 1);

  auto const read_col = read_table->view().column(0);
  ASSERT_EQ(read_col.type().id(), cudf::type_id::LIST)
    << "List<string> must remain LIST when output_dict_columns is on (transcode is flat-only)";
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(list_col->view(), read_col);
}

// Edge case: empty input. A zero-row flat STRING column must round-trip through the transcode
// path without error and reproduce the empty input (whether it comes back as STRING or as an
// empty DICTIONARY32 via the best-effort fallback encode).
TEST_F(ParquetReaderDictTest, EmptyFlatStringDictTranscode)
{
  std::vector<std::string> const empty;
  auto const input_col = cudf::test::strings_column_wrapper(empty.begin(), empty.end());

  auto const input_tbl = cudf::table_view{{input_col}};
  auto const filepath  = temp_env->get_temp_filepath("EmptyFlatStringDictTranscode.parquet");

  // Write directly: the chunked row-group loop in `write_parquet` would skip a zero-row table.
  auto const write_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input_tbl)
      .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
      .compression(cudf::io::compression_type::NONE)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .build();
  cudf::io::write_parquet(write_opts);

  auto const read_table = read_parquet_as_dict(filepath).tbl;
  ASSERT_EQ(read_table->num_rows(), 0);
  ASSERT_EQ(read_table->num_columns(), 1);

  auto const read_col = read_table->view().column(0);
  if (read_col.type().id() == cudf::type_id::DICTIONARY32) {
    // An empty DICTIONARY32 has no keys, so `cudf::dictionary::decode` returns a type-EMPTY empty
    // column (there is no key type to recover) rather than an empty STRING. Just confirm the
    // round-trip stays empty.
    auto const decoded = cudf::dictionary::decode(cudf::dictionary_column_view(read_col));
    EXPECT_EQ(decoded->size(), 0);
  } else {
    ASSERT_EQ(read_col.type().id(), cudf::type_id::STRING);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(input_col, read_col);
  }
}

// Edge case: sliced input. Writing a sliced (non-zero offset, reduced size) flat STRING column
// must transcode correctly -- the reader's DICTIONARY32 output must decode back to exactly the
// sliced rows (including nulls), not the underlying full column.
TEST_F(ParquetReaderDictTest, SlicedFlatStringDictTranscode)
{
  auto const full_col = make_low_cardinality_strings();

  // Interior slice so the view carries a non-zero offset and a reduced size, spanning multiple
  // row groups to also exercise the per-row-group key concatenation / index remapping path.
  auto const slice_start = row_group_size + 7;
  auto const slice_end   = num_rows - 13;
  auto const sliced =
    cudf::slice(static_cast<cudf::column_view>(full_col), {slice_start, slice_end}).front();

  auto const input_tbl = cudf::table_view{{sliced}};
  auto const filepath  = temp_env->get_temp_filepath("SlicedFlatStringDictTranscode.parquet");
  write_parquet(input_tbl, filepath);

  auto const read_table = read_parquet_as_dict(filepath).tbl;
  ASSERT_EQ(read_table->num_rows(), slice_end - slice_start);
  ASSERT_EQ(read_table->num_columns(), 1);

  auto const read_col = read_table->view().column(0);
  ASSERT_EQ(read_col.type().id(), cudf::type_id::DICTIONARY32)
    << "Expected a DICTIONARY32 column for a fully dict-encoded sliced string input";

  auto const decoded_read = cudf::dictionary::decode(cudf::dictionary_column_view(read_col));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sliced, decoded_read->view());
}

// Non-happy test (Fast path ineligible): a filter combined with `output_dict_columns`. A filter
// forces the direct transcode fast path off (predicates evaluate on materialized STRING columns);
// `finalize_output` still encodes the surviving rows to DICTIONARY32 after the filter is applied.
// The key column is projected through unchanged.
TEST_F(ParquetReaderDictTest, FilterWithOutputDictColumns)
{
  auto key_col = make_int_key_column();
  auto str_col = make_low_cardinality_strings();

  auto const input_tbl = cudf::table_view{{key_col, str_col}};
  auto const filepath  = temp_env->get_temp_filepath("FilterWithOutputDictColumns.parquet");
  write_parquet(input_tbl, filepath);

  // Filter: key column (col 0) >= num_rows / 2.
  auto literal_value = cudf::numeric_scalar<int32_t>(num_rows / 2);
  auto literal       = cudf::ast::literal(literal_value);
  auto col_ref       = cudf::ast::column_reference(0);
  auto filter_expr = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref, literal);

  // Expected result: apply the same predicate to the input table on host-visible data.
  auto const predicate = cudf::compute_column(input_tbl, filter_expr);
  auto const expected  = cudf::apply_retention_mask(input_tbl, predicate->view());
  ASSERT_LT(expected->num_rows(), num_rows) << "filter must remove some rows to be meaningful";

  auto const read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .output_dict_columns(true)
                           .filter(filter_expr)
                           .build();
  auto const read_table = cudf::io::read_parquet(read_opts).tbl;
  ASSERT_EQ(read_table->num_columns(), 2);
  ASSERT_EQ(read_table->num_rows(), expected->num_rows());

  // Key column: unchanged INT32.
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view().column(0), read_table->view().column(0));

  // String column: DICTIONARY32 via the post-filter fallback encode; decodes to the surviving rows.
  auto const read_str = read_table->view().column(1);
  ASSERT_EQ(read_str.type().id(), cudf::type_id::DICTIONARY32);
  auto const decoded = cudf::dictionary::decode(cudf::dictionary_column_view(read_str));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view().column(1), decoded->view());
}

// Non-happy test (Fast path ineligible): a column whose chunks mix dictionary and PLAIN data pages.
// cuDF decides dictionary use per column-chunk, so a small dictionary budget over mixed-cardinality
// data yields some dictionary-encoded row groups and some PLAIN-encoded ones. The PLAIN pages
// disqualify the column from the fast path; `output_dict_columns` still produces a correct
// DICTIONARY32 via the fallback encode.
TEST_F(ParquetReaderDictTest, MixedDictAndPlainPagesFallback)
{
  auto input_col = make_mixed_encoding_strings();

  auto const input_tbl = cudf::table_view{{input_col}};
  auto const filepath  = temp_env->get_temp_filepath("MixedDictAndPlainPagesFallback.parquet");
  write_parquet_adaptive(input_tbl, filepath, /*max_dict_size=*/4 * 1024);

  auto const read_table = read_parquet_as_dict(filepath).tbl;
  ASSERT_EQ(read_table->num_rows(), num_rows);
  ASSERT_EQ(read_table->num_columns(), 1);

  auto const read_col = read_table->view().column(0);
  ASSERT_EQ(read_col.type().id(), cudf::type_id::DICTIONARY32)
    << "output_dict_columns must still yield DICTIONARY32 via the fallback encode";
  auto const decoded = cudf::dictionary::decode(cudf::dictionary_column_view(read_col));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input_col, decoded->view());
}

// Non-happy test (Fast path ineligible): `skip_rows` / `num_rows`. Custom row bounds force the fast
// path off; the fallback still emits a DICTIONARY32 that must decode to exactly the requested row
// window.
TEST_F(ParquetReaderDictTest, SkipRowsNumRowsDictTranscode)
{
  auto input_col = make_low_cardinality_strings();

  auto const input_tbl = cudf::table_view{{input_col}};
  auto const filepath  = temp_env->get_temp_filepath("SkipRowsNumRowsDictTranscode.parquet");
  write_parquet(input_tbl, filepath);

  cudf::size_type const skip = row_group_size + 25;
  cudf::size_type const rows = 2 * row_group_size + 40;

  auto const read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .output_dict_columns(true)
                           .skip_rows(skip)
                           .num_rows(rows)
                           .build();
  auto const read_table = cudf::io::read_parquet(read_opts).tbl;
  ASSERT_EQ(read_table->num_columns(), 1);
  ASSERT_EQ(read_table->num_rows(), rows);

  auto const read_col = read_table->view().column(0);
  ASSERT_EQ(read_col.type().id(), cudf::type_id::DICTIONARY32);
  auto const decoded = cudf::dictionary::decode(cudf::dictionary_column_view(read_col));

  auto const expected =
    cudf::slice(static_cast<cudf::column_view>(input_col), {skip, skip + rows}).front();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, decoded->view());
}

// Non-happy test (Fast path ineligible): `chunked_parquet_reader`. A chunked read sets an
// output-chunk byte limit, which disables the fast path. Each chunk must come back as DICTIONARY32
// via the fallback; reassembling the decoded chunks must reproduce the original column.
TEST_F(ParquetReaderDictTest, ChunkedReadDictTranscode)
{
  auto input_col = make_low_cardinality_strings();

  auto const input_tbl = cudf::table_view{{input_col}};
  auto const filepath  = temp_env->get_temp_filepath("ChunkedReadDictTranscode.parquet");
  write_parquet(input_tbl, filepath);

  auto const read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .output_dict_columns(true)
                           .build();
  // Small byte limit so the read is split across multiple output chunks.
  auto reader = cudf::io::chunked_parquet_reader(/*chunk_read_limit=*/16 * 1024, read_opts);

  std::vector<std::unique_ptr<cudf::column>> decoded_chunks;
  cudf::size_type total_rows = 0;
  int num_chunks             = 0;
  while (reader.has_next()) {
    auto chunk = reader.read_chunk();
    ASSERT_EQ(chunk.tbl->num_columns(), 1);
    auto const read_col = chunk.tbl->view().column(0);
    if (read_col.size() == 0) { continue; }
    ASSERT_EQ(read_col.type().id(), cudf::type_id::DICTIONARY32);
    decoded_chunks.push_back(cudf::dictionary::decode(cudf::dictionary_column_view(read_col)));
    total_rows += read_col.size();
    ++num_chunks;
  }
  ASSERT_EQ(total_rows, num_rows);
  EXPECT_GT(num_chunks, 1) << "byte limit should split the read into multiple chunks";

  std::vector<cudf::column_view> views;
  views.reserve(decoded_chunks.size());
  for (auto const& c : decoded_chunks) {
    views.push_back(c->view());
  }
  auto const combined = cudf::concatenate(views);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input_col, combined->view());
}

// Non-happy test (Fast path ineligible): a multi-column table mixing eligible and ineligible
// columns. Only the flat string column is transcoded to DICTIONARY32; the LIST<STRING> column stays
// LIST (flat-only transcode) and the INT32 column is untouched (non-string). This also exercises
// the output-buffer indexing when an eligible column is preceded/followed by columns of differing
// nesting.
TEST_F(ParquetReaderDictTest, MultiColumnMixedEligibility)
{
  auto str_col  = make_low_cardinality_strings();           // eligible  -> DICTIONARY32
  auto list_col = make_low_cardinality_lists_of_strings();  // ineligible -> LIST<STRING>
  auto key_col  = make_int_key_column();                    // non-string -> INT32

  auto const input_tbl = cudf::table_view{{str_col, list_col->view(), key_col}};
  auto const filepath  = temp_env->get_temp_filepath("MultiColumnMixedEligibility.parquet");
  write_parquet(input_tbl, filepath);

  auto const read_table = read_parquet_as_dict(filepath).tbl;
  ASSERT_EQ(read_table->num_rows(), num_rows);
  ASSERT_EQ(read_table->num_columns(), 3);

  // Flat string column: transcoded to DICTIONARY32.
  auto const read_str = read_table->view().column(0);
  ASSERT_EQ(read_str.type().id(), cudf::type_id::DICTIONARY32);
  auto const decoded_str = cudf::dictionary::decode(cudf::dictionary_column_view(read_str));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(str_col, decoded_str->view());

  // List<string> column: not eligible, remains LIST<STRING>.
  auto const read_list = read_table->view().column(1);
  ASSERT_EQ(read_list.type().id(), cudf::type_id::LIST)
    << "List<string> must remain LIST when output_dict_columns is on (transcode is flat-only)";
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(list_col->view(), read_list);

  // All-unique INT32 column: the writer does not dictionary-encode it (data pages are PLAIN),
  // so it is ineligible for transcode and stays a plain INT32 column.
  auto const read_key = read_table->view().column(2);
  ASSERT_EQ(read_key.type().id(), cudf::type_id::INT32);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(key_col, read_key);
}

namespace {

/// A low-cardinality nullable INT32 column.
cudf::test::fixed_width_column_wrapper<int32_t> make_low_cardinality_ints()
{
  std::mt19937 engine(seed ^ 0xF17ED0UL);
  std::uniform_int_distribution<int> value_dist(0, cardinality - 1);
  std::bernoulli_distribution null_dist(null_probability);
  std::vector<int32_t> values(num_rows);
  std::vector<bool> valids(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    values[i] = 1'000'000 + value_dist(engine);
    valids[i] = not null_dist(engine);
  }
  return cudf::test::fixed_width_column_wrapper<int32_t>(
    values.begin(), values.end(), valids.begin());
}

/// A low-cardinality INT64 column.
cudf::test::fixed_width_column_wrapper<int64_t> make_low_cardinality_int64s()
{
  std::mt19937 engine(seed ^ 0x64B175UL);
  std::uniform_int_distribution<int> value_dist(0, cardinality - 1);
  std::vector<int64_t> values(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    values[i] = 3'000'000'000LL + value_dist(engine);
  }
  return cudf::test::fixed_width_column_wrapper<int64_t>(values.begin(), values.end());
}

/// A low-cardinality date (TIMESTAMP_DAYS, physical INT32) column.
cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, int32_t> make_low_cardinality_dates()
{
  std::mt19937 engine(seed ^ 0xDA7E5UL);
  std::uniform_int_distribution<int> value_dist(0, cardinality - 1);
  std::vector<int32_t> values(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    values[i] = 8000 + value_dist(engine);
  }
  return cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, int32_t>(values.begin(),
                                                                            values.end());
}

}  // namespace

// Flat fixed-width columns (INT32 with nulls, INT64, DATE) written with dictionary encoding must
// transcode to DICTIONARY32 columns whose keys carry the logical type, across several row groups,
// and decode back to exactly the input.
TEST_F(ParquetReaderDictTest, FlatFixedWidthDictTranscode)
{
  auto const int32_col = make_low_cardinality_ints();
  auto const int64_col = make_low_cardinality_int64s();
  auto const date_col  = make_low_cardinality_dates();
  auto const input     = cudf::table_view{{int32_col, int64_col, date_col}};

  auto const filepath = temp_env->get_temp_filepath("FlatFixedWidthDictTranscode.parquet");
  write_parquet(input, filepath);

  auto const read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .output_dict_columns(true)
                           .build();
  auto const read_table = cudf::io::read_parquet(read_opts).tbl;

  std::array<cudf::type_id, 3> const key_types{
    cudf::type_id::INT32, cudf::type_id::INT64, cudf::type_id::TIMESTAMP_DAYS};
  for (cudf::size_type i = 0; i < read_table->num_columns(); ++i) {
    auto const read_col = read_table->view().column(i);
    ASSERT_EQ(read_col.type().id(), cudf::type_id::DICTIONARY32) << "column " << i;
    cudf::dictionary_column_view const dict{read_col};
    EXPECT_EQ(dict.keys().type().id(), key_types[i]) << "column " << i;
    // `cardinality` distinct keys fit INT16 indices (get_indices_type_for_size)
    EXPECT_EQ(dict.indices().type().id(), cudf::type_id::INT16) << "column " << i;
    auto const decoded = cudf::dictionary::decode(dict);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(input.column(i), decoded->view());
  }
}

// Fixed-width columns stay plain by default (option off).
TEST_F(ParquetReaderDictTest, FlatFixedWidthNoTranscodeByDefault)
{
  auto const int32_col = make_low_cardinality_ints();
  auto const input     = cudf::table_view{{int32_col}};
  auto const filepath  = temp_env->get_temp_filepath("FlatFixedWidthNoTranscode.parquet");
  write_parquet(input, filepath);

  auto const read_table =
    cudf::io::read_parquet(
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).build())
      .tbl;
  auto const read_col = read_table->view().column(0);
  ASSERT_EQ(read_col.type().id(), cudf::type_id::INT32);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(int32_col, read_col);
}

// Under an AST filter the direct fast path is off: string columns are still delivered as
// DICTIONARY32 via the post-hoc encode, while fixed-width columns fall back to plain.
TEST_F(ParquetReaderDictTest, FixedWidthFilterFallsBackToPlain)
{
  auto const int32_col  = make_low_cardinality_ints();
  auto const string_col = make_low_cardinality_strings();
  auto const input      = cudf::table_view{{int32_col, string_col}};
  auto const filepath   = temp_env->get_temp_filepath("FixedWidthFilterFallback.parquet");
  write_parquet(input, filepath);

  auto const ref     = cudf::ast::column_reference(0);
  auto literal_value = cudf::numeric_scalar<int32_t>(1'000'000 + cardinality / 2);
  auto const literal = cudf::ast::literal(literal_value);
  auto const expr    = cudf::ast::operation(cudf::ast::ast_operator::LESS, ref, literal);

  auto const read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .filter(expr)
                           .output_dict_columns(true)
                           .build();
  auto const read_table = cudf::io::read_parquet(read_opts).tbl;

  auto const read_int = read_table->view().column(0);
  ASSERT_EQ(read_int.type().id(), cudf::type_id::INT32);
  auto const read_str = read_table->view().column(1);
  ASSERT_EQ(read_str.type().id(), cudf::type_id::DICTIONARY32);

  // Cross-check the surviving rows against a plain filtered read.
  auto const plain_table = cudf::io::read_parquet(cudf::io::parquet_reader_options::builder(
                                                    cudf::io::source_info{filepath})
                                                    .filter(expr)
                                                    .build())
                             .tbl;
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(plain_table->view().column(0), read_int);
  auto const decoded_str = cudf::dictionary::decode(cudf::dictionary_column_view(read_str));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(plain_table->view().column(1), decoded_str->view());
}

namespace {

/// A nullable INT32 column with `distinct` distinct values over `rows` rows.
cudf::test::fixed_width_column_wrapper<int32_t> make_int_column_with_cardinality(
  cudf::size_type rows, cudf::size_type distinct, unsigned int local_seed)
{
  std::mt19937 engine(local_seed);
  std::uniform_int_distribution<int> value_dist(0, distinct - 1);
  std::vector<int32_t> values(rows);
  for (cudf::size_type i = 0; i < rows; ++i) {
    values[i] = 5'000'000 + value_dist(engine);
  }
  return cudf::test::fixed_width_column_wrapper<int32_t>(values.begin(), values.end());
}

}  // namespace

// The emitted index width follows the largest row-group dictionary: <= 127 keys use INT8 and
// > 32767 keys use INT32 (the INT16 middle case is covered by FlatFixedWidthDictTranscode).
TEST_F(ParquetReaderDictTest, FixedWidthIndexWidths)
{
  // INT8: 100 distinct keys across the default row groups.
  {
    auto const col      = make_int_column_with_cardinality(num_rows, 100, seed ^ 0x1D8);
    auto const filepath = temp_env->get_temp_filepath("FixedWidthIndexWidthInt8.parquet");
    write_parquet(cudf::table_view{{col}}, filepath);
    auto const read_table = read_parquet_as_dict(filepath).tbl;
    auto const read_col   = read_table->view().column(0);
    ASSERT_EQ(read_col.type().id(), cudf::type_id::DICTIONARY32);
    cudf::dictionary_column_view const dict{read_col};
    EXPECT_EQ(dict.indices().type().id(), cudf::type_id::INT8);
    auto const decoded = cudf::dictionary::decode(dict);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(col, decoded->view());
  }
  // INT32: a single row group whose dictionary exceeds 32767 keys.
  {
    constexpr cudf::size_type wide_rows     = 136'000;
    constexpr cudf::size_type wide_distinct = 34'000;
    // deterministic: each value appears four times in adjacent runs, so the dictionary holds
    // `wide_distinct` keys and dictionary encoding beats PLAIN in size (the writer skips the
    // dictionary, even with policy ALWAYS, when it would not be smaller)
    std::vector<int32_t> wide_values(wide_rows);
    for (cudf::size_type i = 0; i < wide_rows; ++i) {
      wide_values[i] = 5'000'000 + (i / 4) % wide_distinct;
    }
    cudf::test::fixed_width_column_wrapper<int32_t> const col(wide_values.begin(),
                                                              wide_values.end());
    auto const filepath = temp_env->get_temp_filepath("FixedWidthIndexWidthInt32.parquet");
    auto const options  = cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath},
                                                                   cudf::table_view{{col}})
                           .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
                           .row_group_size_rows(wide_rows)
                           .build();
    cudf::io::write_parquet(options);
    auto const read_table = read_parquet_as_dict(filepath).tbl;
    auto const read_col   = read_table->view().column(0);
    ASSERT_EQ(read_col.type().id(), cudf::type_id::DICTIONARY32);
    cudf::dictionary_column_view const dict{read_col};
    EXPECT_EQ(dict.indices().type().id(), cudf::type_id::INT32);
    EXPECT_GT(dict.keys_size(), 32767);
    auto const decoded = cudf::dictionary::decode(dict);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(col, decoded->view());
  }
}

// Bounded reads (skip_rows / num_rows) disable the direct fast path; fixed-width columns fall
// back to their plain type and match a plain bounded read.
TEST_F(ParquetReaderDictTest, FixedWidthBoundedReadFallsBackToPlain)
{
  auto const col      = make_low_cardinality_ints();
  auto const filepath = temp_env->get_temp_filepath("FixedWidthBoundedRead.parquet");
  write_parquet(cudf::table_view{{col}}, filepath);

  auto const bounded = [&](bool output_dict) {
    return cudf::io::read_parquet(
             cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
               .skip_rows(row_group_size / 2)
               .num_rows(row_group_size)
               .output_dict_columns(output_dict)
               .build())
      .tbl;
  };
  auto const read_table = bounded(true);
  auto const read_col   = read_table->view().column(0);
  ASSERT_EQ(read_col.type().id(), cudf::type_id::INT32);
  auto const plain_table = bounded(false);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(plain_table->view().column(0), read_col);
}

// Chunked reads disable the direct fast path; fixed-width columns come back plain in every
// output chunk and reassemble to the input.
TEST_F(ParquetReaderDictTest, FixedWidthChunkedReadFallsBackToPlain)
{
  auto const col      = make_low_cardinality_ints();
  auto const filepath = temp_env->get_temp_filepath("FixedWidthChunkedRead.parquet");
  write_parquet(cudf::table_view{{col}}, filepath);

  auto const read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .output_dict_columns(true)
                           .build();
  auto reader = cudf::io::chunked_parquet_reader(/*chunk_read_limit=*/8 * 1024, read_opts);

  std::vector<std::unique_ptr<cudf::table>> chunks;
  std::vector<cudf::column_view> views;
  cudf::size_type total_rows = 0;
  int num_chunks             = 0;
  while (reader.has_next()) {
    auto chunk          = reader.read_chunk();
    auto const read_col = chunk.tbl->view().column(0);
    if (read_col.size() == 0) { continue; }
    ASSERT_EQ(read_col.type().id(), cudf::type_id::INT32);
    total_rows += read_col.size();
    ++num_chunks;
    chunks.push_back(std::move(chunk.tbl));
    views.push_back(chunks.back()->view().column(0));
  }
  ASSERT_EQ(total_rows, num_rows);
  EXPECT_GT(num_chunks, 1) << "byte limit should split the read into multiple chunks";
  auto const combined = cudf::concatenate(views);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col, combined->view());
}
