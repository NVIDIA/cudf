/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Targeted correctness tests for the chunked RLE staging path introduced by
// Tasks D+E+F in cpp/src/io/parquet/rle_stream.cuh (rle_stream::decode_next_chunked)
// and the double-buffered staging in preprocess_levels_kernel.
//
// The chunked path is exercised whenever we decode definition/repetition levels
// for nested (LIST/STRUCT) columns. These tests deliberately construct pages
// that stress the fragile boundary/skip conditions of that path:
//   1. RleStagingExactBoundary   - page rows == N * chunk_size (4096)
//   2. RleStagingSingleLongLiteralRun - one long RLE literal run spanning many chunks
//   3. RleStagingSkipAhead       - skip_rows before decode_next_chunked
//   4. RleStagingZeroLengthPage  - a page with 0 rows after row-group filtering

#include "parquet_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table_view.hpp>

#include <cuda/iterator>

#include <memory>
#include <vector>

namespace {

// Chunk size used by rle_stream::decode_next_chunked; see
// cpp/src/io/parquet/rle_stream.cuh (constexpr int chunk_size = 4096).
constexpr int kChunkSize = 4096;

// Build a LIST<INT32> column with `num_rows` rows, `list_size` values per row,
// and every other row null. Nested types force preprocess_levels_kernel
// (which uses the chunked staging path) to run for definition/repetition levels.
std::unique_ptr<cudf::column> make_list_int_col(int num_rows, int list_size)
{
  auto const valids = cudf::test::iterators::valids_at_multiples_of(2);

  std::vector<cudf::size_type> offsets(num_rows + 1);
  int running = 0;
  for (int i = 0; i < num_rows; ++i) {
    offsets[i] = running;
    if (valids[i]) { running += list_size; }
  }
  offsets[num_rows] = running;

  int const child_size = running;
  auto value_iter      = cuda::counting_iterator<int>{0};
  cudf::test::fixed_width_column_wrapper<int> child(value_iter, value_iter + child_size);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets_col(offsets.begin(),
                                                                      offsets.end());

  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(valids, valids + num_rows);
  return cudf::make_lists_column(
    num_rows, offsets_col.release(), child.release(), null_count, std::move(null_mask));
}

// Build a LIST<INT32> column with `num_rows` rows and no nulls, list_size per row.
// Absence of nulls makes the definition-level stream a single long literal-encoded run
// (bit-packed run spanning many decode chunks).
std::unique_ptr<cudf::column> make_dense_list_int_col(int num_rows, int list_size)
{
  std::vector<cudf::size_type> offsets(num_rows + 1);
  for (int i = 0; i <= num_rows; ++i) {
    offsets[i] = i * list_size;
  }
  int const child_size = num_rows * list_size;
  auto value_iter      = cuda::counting_iterator<int>{0};
  cudf::test::fixed_width_column_wrapper<int> child(value_iter, value_iter + child_size);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets_col(offsets.begin(),
                                                                      offsets.end());
  return cudf::make_lists_column(
    num_rows, offsets_col.release(), child.release(), 0, rmm::device_buffer{});
}

}  // namespace

// -----------------------------------------------------------------------------
// Test 1: page rows == N * chunk_size. Exercises the outer-loop chunk-transition
// boundary in rle_stream::decode_next_chunked (page must not read off-by-one
// across a chunk boundary that exactly aligns with page end).
// -----------------------------------------------------------------------------
TEST_F(ParquetReaderTest, RleStagingExactBoundary)
{
  // Try two page-size multiples: N=2 (8192) and N=3 (12288).
  for (int n : {2, 3}) {
    int const num_rows  = n * kChunkSize;
    int const list_size = 3;
    auto col            = make_list_int_col(num_rows, list_size);
    cudf::table_view tbl({*col});

    auto filepath =
      temp_env->get_temp_filepath("RleStagingExactBoundary_" + std::to_string(n) + ".parquet");
    cudf::io::parquet_writer_options out_args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl)
        .max_page_size_rows(num_rows)  // one page for the whole row group
        .row_group_size_rows(num_rows);
    cudf::io::write_parquet(out_args);

    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    auto result = cudf::io::read_parquet(read_args);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, tbl);
  }
}

// -----------------------------------------------------------------------------
// Test 2: page encoded as one long RLE literal (bit-packed) run spanning many
// decode chunks (>32K rows in a single page, no nulls -> single-run def-level
// stream). Exercises partial_run_meta stash/resume across every chunk boundary.
// -----------------------------------------------------------------------------
TEST_F(ParquetReaderTest, RleStagingSingleLongLiteralRun)
{
  // Two variants: dense (no nulls) and sparse (with nulls). Dense builds a
  // rep-level stream of one long literal run; sparse forces def-level decoding
  // (preprocess_levels_kernel only decodes def-levels when the column has nulls,
  // so a null-free page never exercises the def-level chunked staging path).
  constexpr int num_rows  = 40 * 1024;  // > 32K, spans ~10 chunk_size chunks
  constexpr int list_size = 2;
  for (bool with_nulls : {false, true}) {
    auto col = with_nulls ? make_list_int_col(num_rows, list_size)
                          : make_dense_list_int_col(num_rows, list_size);
    cudf::table_view tbl({*col});

    auto filepath = temp_env->get_temp_filepath(std::string("RleStagingSingleLongLiteralRun_") +
                                                (with_nulls ? "nulls" : "dense") + ".parquet");
    cudf::io::parquet_writer_options out_args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl)
        .max_page_size_rows(num_rows)  // one giant page
        .row_group_size_rows(num_rows);
    cudf::io::write_parquet(out_args);

    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    auto result = cudf::io::read_parquet(read_args);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, tbl);
  }
}

// -----------------------------------------------------------------------------
// Test 3: skip_rows before decode_next_chunked. Ensures skip_decode plus the
// staged double-buffered load align correctly. Skips cross chunk boundaries.
// -----------------------------------------------------------------------------
TEST_F(ParquetReaderTest, RleStagingSkipAhead)
{
  constexpr int num_rows  = 20 * 1024;
  constexpr int list_size = 4;
  auto col                = make_list_int_col(num_rows, list_size);
  cudf::table_view tbl({*col});

  auto filepath = temp_env->get_temp_filepath("RleStagingSkipAhead.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl)
      .max_page_size_rows(4000)  // multiple pages per row group
      .row_group_size_rows(num_rows);
  cudf::io::write_parquet(out_args);

  // Skip values chosen to land inside a chunk (not at chunk_size boundary),
  // across a chunk_size boundary, and across a page boundary.
  std::vector<std::pair<int, int>> params{
    {kChunkSize / 2, -1},    // mid-chunk skip
    {kChunkSize - 1, -1},    // just before chunk boundary
    {kChunkSize + 3, -1},    // just after chunk boundary
    {2 * kChunkSize, 4096},  // exactly on chunk boundary, bounded num_rows
    {4001, 5000},            // cross page boundary
  };
  for (auto [skip, take] : params) {
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    read_args.set_skip_rows(skip);
    if (take >= 0) { read_args.set_num_rows(take); }
    auto result = cudf::io::read_parquet(read_args);

    int const end = take >= 0 ? skip + take : num_rows;
    std::vector<cudf::size_type> slice_indices{skip, end};
    auto expected = cudf::slice(tbl, slice_indices);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, expected[0]);
  }
}

// -----------------------------------------------------------------------------
// Test 4: page with 0 rows after row-group filtering. skip_rows == total row
// count -> zero rows to decode; must not touch SMEM staging or panic.
// -----------------------------------------------------------------------------
TEST_F(ParquetReaderTest, RleStagingZeroLengthPage)
{
  constexpr int num_rows  = 8 * 1024;
  constexpr int list_size = 3;
  auto col                = make_list_int_col(num_rows, list_size);
  cudf::table_view tbl({*col});

  auto filepath = temp_env->get_temp_filepath("RleStagingZeroLengthPage.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl)
      .max_page_size_rows(2048)
      .row_group_size_rows(num_rows);
  cudf::io::write_parquet(out_args);

  // Read past the end: skip == num_rows yields zero rows.
  cudf::io::parquet_reader_options read_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  read_args.set_skip_rows(num_rows);
  auto result = cudf::io::read_parquet(read_args);

  EXPECT_EQ(result.tbl->num_rows(), 0);
  EXPECT_EQ(result.tbl->num_columns(), 1);

  // Also: request 0 rows explicitly at start.
  cudf::io::parquet_reader_options zero_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  zero_args.set_num_rows(0);
  auto zero_result = cudf::io::read_parquet(zero_args);
  EXPECT_EQ(zero_result.tbl->num_rows(), 0);
  EXPECT_EQ(zero_result.tbl->num_columns(), 1);
}
