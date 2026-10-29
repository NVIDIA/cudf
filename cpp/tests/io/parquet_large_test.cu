/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet_chunked_test_utils.cuh"
#include "parquet_common.hpp"

#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/io/parquet.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/copy.h>

#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using parquet_test::bool_gen;
using parquet_test::offset_gen;

TEST_F(ParquetReaderTest, TableTooLargeOverflows)
{
  using T                             = bool;
  constexpr int64_t per_file_num_rows = std::numeric_limits<cudf::size_type>::max() / 2 + 1000;
  static_assert(per_file_num_rows <= std::numeric_limits<cudf::size_type>::max(),
                "Number of rows per file should be less than size_type::max()");
  static_assert(2 * per_file_num_rows > std::numeric_limits<cudf::size_type>::max(),
                "Twice number of rows per file should be greather than size_type::max()");
  auto value  = cuda::make_constant_iterator(true);
  auto column = cudf::test::fixed_width_column_wrapper<T>(value, value + per_file_num_rows);

  auto filepath = temp_env->get_temp_filepath("TableTooLargeOverflows.parquet");
  {
    auto sink = cudf::io::sink_info{filepath};
    auto options =
      cudf::io::parquet_writer_options::builder(sink, cudf::table_view{{column}}).build();
    std::ignore = cudf::io::write_parquet(options);
  }
  std::vector<std::string> files{{filepath, filepath}};
  auto source                 = cudf::io::source_info(files);
  auto metadata               = cudf::io::read_parquet_metadata(source);
  auto const num_rows_to_read = metadata.num_rows() - 1000;
  EXPECT_EQ(metadata.num_rows(), per_file_num_rows * 2);
  auto options = cudf::io::parquet_reader_options::builder(source)
                   .num_rows(num_rows_to_read)
                   .skip_rows(10)
                   .build();

  EXPECT_THROW(cudf::io::read_parquet(options), std::overflow_error);
  auto reader = cudf::io::chunked_parquet_reader(0, 0, options);
  int64_t num_rows_read{0};
  while (reader.has_next()) {
    auto chunk = reader.read_chunk();
    num_rows_read += chunk.tbl->num_rows();
  }
  EXPECT_EQ(num_rows_read, num_rows_to_read);
}

TEST_F(ParquetReaderTest, ManyLargeLists)
{
  // this test runs over 3 hours when racecheck is used
  if (getenv("LIBCUDF_RACECHECK_ENABLED")) { GTEST_SKIP(); }

  auto const stream = cudf::get_default_stream();

  // Generate a large list<bool> column
  constexpr cudf::size_type num_rows = 10'000'000;

  // Helper to test chunked-read a parquet table with a large list<bool> column
  auto const test_chunked_read_many_large_lists = [&](cudf::size_type bools_per_row,
                                                      cudf::mask_state bools_col_state) {
    // Offsets column
    auto offsets_iter =
      cudf::detail::make_counting_transform_iterator(0, offset_gen{bools_per_row});
    auto offsets_col = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT32}, num_rows + 1, cudf::mask_state::UNALLOCATED);
    thrust::copy(rmm::exec_policy_nosync(stream),
                 offsets_iter,
                 offsets_iter + num_rows + 1,
                 offsets_col->mutable_view().begin<int>());

    // Booleans column
    auto bools_iter = cudf::detail::make_counting_transform_iterator(0, bool_gen{});
    auto bools_col  = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::BOOL8}, num_rows * bools_per_row, bools_col_state);
    thrust::copy(rmm::exec_policy_nosync(stream),
                 bools_iter,
                 bools_iter + (num_rows * bools_per_row),
                 bools_col->mutable_view().begin<bool>());

    stream.sync();

    // list<bool> column
    auto list_col =
      cudf::make_lists_column(num_rows,
                              std::move(offsets_col),
                              std::move(bools_col),
                              0,
                              cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

    auto const table    = cudf::table_view({*list_col});
    auto const filepath = temp_env->get_temp_filepath("ManyLargeLists.parquet");
    auto const out_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, table).build();
    // Write the table to parquet
    cudf::io::write_parquet(out_opts);

    // Times to concat filepath to (just slightly) overflow cudf column size limits
    auto const reads_to_overflow = static_cast<cudf::size_type>(
      (std::numeric_limits<cudf::size_type>::max() / (num_rows * bools_per_row)) + 1);

    auto const in_opts =
      cudf::io::parquet_reader_options::builder(
        cudf::io::source_info{std::vector<std::string>(reads_to_overflow, filepath)})
        .build();

    // Expect an overflow error when reading the files
    EXPECT_THROW(cudf::io::read_parquet(in_opts), std::overflow_error);

    // Now read the files with the chunked reader (no limits) without the overflow error
    auto const reader = cudf::io::chunked_parquet_reader(0, 0, in_opts);
    auto num_chunks   = 0;
    while (reader.has_next()) {
      ASSERT_NO_THROW(std::ignore = reader.read_chunk());
      num_chunks++;
    }
    // We will end up with exactly two chunks as the total number of leaf rows is just above 2B
    // rows per table chunk limit and we haven't set any chunk or pass read limits
    EXPECT_EQ(num_chunks, 2);
  };

  // Test the case where the number of top-level (list) rows does not exceed the cudf column size
  // limit but the number of leaf (bool) values (not rows) does
  auto constexpr bools_per_row_for_excess_leaf_rows = 4;
  test_chunked_read_many_large_lists(bools_per_row_for_excess_leaf_rows,
                                     cudf::mask_state::UNALLOCATED);

  // Test the case where the number of top-level (list) rows exceeds the cudf column size
  // limit but the number of leaf (bool) values (not rows) does not
  auto constexpr bools_per_row_for_excess_top_level_rows = 1;
  test_chunked_read_many_large_lists(bools_per_row_for_excess_top_level_rows,
                                     cudf::mask_state::ALL_NULL);

  // Note: The chunked reader will always fail if the number of leaf (bool) rows (values + nulls)
  // exceeds the cudf column size limit unless a smaller `pass_read_limit` is specified.
}
