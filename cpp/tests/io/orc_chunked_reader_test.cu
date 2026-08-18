/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compression_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/io_metadata_utilities.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/io/orc_metadata.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>

namespace {
enum class output_limit : std::size_t {};
enum class input_limit : std::size_t {};
enum class output_row_granularity : cudf::size_type {};

// Global environment for temporary files
auto const temp_env = reinterpret_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

using int32s_col       = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64s_col       = cudf::test::fixed_width_column_wrapper<int64_t>;
using doubles_col      = cudf::test::fixed_width_column_wrapper<double>;
using strings_col      = cudf::test::strings_column_wrapper;
using structs_col      = cudf::test::structs_column_wrapper;
using int32s_lists_col = cudf::test::lists_column_wrapper<int32_t>;

auto write_file(std::vector<std::unique_ptr<cudf::column>>& input_columns,
                std::string const& filename,
                bool nullable                    = false,
                std::size_t stripe_size_bytes    = cudf::io::default_stripe_size_bytes,
                cudf::size_type stripe_size_rows = cudf::io::default_stripe_size_rows)
{
  if (nullable) {
    // Generate deterministic bitmask instead of random bitmask for easy computation of data size.
    auto const valid_iter = cudf::detail::make_counting_transform_iterator(
      0, [](cudf::size_type i) { return i % 4 != 3; });
    cudf::size_type offset{0};
    for (auto& col : input_columns) {
      auto const [null_mask, null_count] =
        cudf::test::detail::make_null_mask(valid_iter + offset, valid_iter + col->size() + offset);
      col = cudf::structs::detail::superimpose_and_sanitize_nulls(
        static_cast<cudf::bitmask_type const*>(null_mask.data()),
        null_count,
        std::move(col),
        cudf::get_default_stream(),
        cudf::get_current_device_resource_ref());

      // Shift nulls of the next column by one position, to avoid having all nulls
      // in the same table rows.
      ++offset;
    }
  }

  auto input_table = std::make_unique<cudf::table>(std::move(input_columns));
  auto filepath =
    temp_env->get_temp_filepath(nullable ? filename + "_nullable.orc" : filename + ".orc");

  auto const write_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, *input_table)
      .stripe_size_bytes(stripe_size_bytes)
      .stripe_size_rows(stripe_size_rows)
      .build();
  cudf::io::write_orc(write_opts);

  return std::pair{std::move(input_table), std::move(filepath)};
}

// NOTE: By default, output_row_granularity=10'000 rows.
// This means if the input file has more than 10k rows then the output chunk will never
// have less than 10k rows.
auto chunked_read(std::string const& filepath,
                  output_limit output_limit_bytes,
                  input_limit input_limit_bytes             = input_limit{0},
                  output_row_granularity output_granularity = output_row_granularity{10'000})
{
  auto const read_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath}).build();
  auto reader = cudf::io::chunked_orc_reader(static_cast<std::size_t>(output_limit_bytes),
                                             static_cast<std::size_t>(input_limit_bytes),
                                             static_cast<cudf::size_type>(output_granularity),
                                             read_opts);

  auto num_chunks = 0;
  auto out_tables = std::vector<std::unique_ptr<cudf::table>>{};

  // TODO: remove this scope, when we get rid of mem stat in the reader.
  // This is to avoid use-after-free of memory resource created by the mem stat object.
  auto mr = cudf::get_current_device_resource_ref();

  do {
    auto chunk = reader.read_chunk();
    // If the input file is empty, the first call to `read_chunk` will return an empty table.
    // Thus, we only check for non-empty output table from the second call.
    if (num_chunks > 0) {
      CUDF_EXPECTS(chunk.tbl->num_rows() != 0, "Number of rows in the new chunk is zero.");
    }
    ++num_chunks;
    out_tables.emplace_back(std::move(chunk.tbl));
  } while (reader.has_next());

  if (num_chunks > 1) {
    CUDF_EXPECTS(out_tables.front()->num_rows() != 0, "Number of rows in the new chunk is zero.");
  }

  auto out_tviews = std::vector<cudf::table_view>{};
  for (auto const& tbl : out_tables) {
    out_tviews.emplace_back(tbl->view());
  }

  // return std::pair(cudf::concatenate(out_tviews), num_chunks);

  // TODO: remove this
  return std::pair(cudf::concatenate(out_tviews, cudf::get_default_stream(), mr), num_chunks);
}

auto chunked_read(std::string const& filepath,
                  output_limit output_limit_bytes,
                  output_row_granularity output_granularity)
{
  return chunked_read(filepath, output_limit_bytes, input_limit{0UL}, output_granularity);
}

void expect_selected_nested_empty_struct_table(cudf::table_view expected, cudf::table_view actual)
{
  ASSERT_EQ(1, actual.num_columns());
  ASSERT_EQ(1, actual.column(0).num_children());
  ASSERT_EQ(cudf::type_id::STRUCT, actual.column(0).child(0).type().id());
  ASSERT_EQ(0, actual.column(0).child(0).num_children());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, actual);
}

void expect_selected_nested_empty_struct_chunked_round_trip(
  std::unique_ptr<cudf::column> input_column, std::string const& filename)
{
  std::vector<std::unique_ptr<cudf::column>> input_columns;
  input_columns.emplace_back(std::move(input_column));

  auto expected = std::make_unique<cudf::table>(std::move(input_columns));
  cudf::io::table_input_metadata metadata(expected->view());
  metadata.column_metadata[0].set_name("name");
  metadata.column_metadata[0].child(0).set_name("empty");

  auto const filepath = temp_env->get_temp_filepath(filename);
  auto const write_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, *expected)
      .metadata(std::move(metadata))
      .build();
  cudf::io::write_orc(write_opts);

  auto const make_read_opts = [&filepath] {
    return cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath})
      .columns({"name"})
      .build();
  };

  auto reader = cudf::io::chunked_orc_reader(1'000, 0, 10'000, make_read_opts());
  auto chunk  = reader.read_chunk();

  EXPECT_FALSE(reader.has_next());
  expect_selected_nested_empty_struct_table(expected->view(), chunk.tbl->view());
}

}  // namespace

struct OrcChunkedReaderTest : public cudf::test::BaseFixture {};

using OrcChunkedDecompressionTest = DecompressionTest<OrcChunkedReaderTest>;

TEST_F(OrcChunkedReaderTest, TestChunkedReadNoData)
{
  std::vector<std::unique_ptr<cudf::column>> input_columns;
  input_columns.emplace_back(int32s_col{}.release());
  input_columns.emplace_back(int64s_col{}.release());

  auto const [expected, filepath] = write_file(input_columns, "chunked_read_empty");
  auto const [result, num_chunks] = chunked_read(filepath, output_limit{1'000});
  EXPECT_EQ(num_chunks, 1);
  EXPECT_EQ(result->num_rows(), 0);
  EXPECT_EQ(result->num_columns(), 2);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
}

TEST_F(OrcChunkedReaderTest, NestedEmptyStructColumnSelection)
{
  auto const validity = std::vector<bool>{true, false};
  auto const num_rows = static_cast<cudf::size_type>(validity.size());
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(validity.begin(), validity.end());

  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.emplace_back(cudf::make_structs_column(num_rows, {}, 0, {}));
  auto input_column = cudf::make_structs_column(
    num_rows, std::move(struct_children), null_count, std::move(null_mask));
  ASSERT_TRUE(input_column->nullable());
  ASSERT_EQ(null_count, input_column->null_count());
  // Struct parent nulls are superimposed on their children by make_structs_column.
  ASSERT_TRUE(input_column->child(0).nullable());
  ASSERT_EQ(null_count, input_column->child(0).null_count());

  expect_selected_nested_empty_struct_chunked_round_trip(std::move(input_column),
                                                         "nested_empty_struct_outer_nullable.orc");
}

TEST_F(OrcChunkedReaderTest, NullableEmptyStructChildColumnSelection)
{
  auto const validity = std::vector<bool>{true, false};
  auto const num_rows = static_cast<cudf::size_type>(validity.size());
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(validity.begin(), validity.end());

  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.emplace_back(
    cudf::make_structs_column(num_rows, {}, null_count, std::move(null_mask)));
  auto input_column = cudf::make_structs_column(num_rows, std::move(struct_children), 0, {});
  ASSERT_FALSE(input_column->nullable());
  ASSERT_TRUE(input_column->child(0).nullable());
  ASSERT_EQ(null_count, input_column->child(0).null_count());

  expect_selected_nested_empty_struct_chunked_round_trip(std::move(input_column),
                                                         "nested_empty_struct_inner_nullable.orc");
}

TEST_F(OrcChunkedReaderTest, TestChunkedReadInvalidParameter)
{
  std::vector<std::unique_ptr<cudf::column>> input_columns;
  input_columns.emplace_back(int32s_col{}.release());
  input_columns.emplace_back(int64s_col{}.release());

  auto const [expected, filepath] = write_file(input_columns, "chunked_read_invalid");
  EXPECT_THROW(
    chunked_read(filepath, output_limit{1'000}, output_row_granularity{-1} /*invalid value*/),
    cudf::logic_error);
}

TEST_F(OrcChunkedReaderTest, TestChunkedReadSimpleData)
{
  auto constexpr num_rows = 40'000;

  auto const generate_input = [num_rows](bool nullable, std::size_t stripe_rows) {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    auto const value_iter = cuda::counting_iterator<int32_t>{0};
    input_columns.emplace_back(int32s_col(value_iter, value_iter + num_rows).release());
    input_columns.emplace_back(int64s_col(value_iter, value_iter + num_rows).release());

    return write_file(input_columns,
                      "chunked_read_simple",
                      nullable,
                      cudf::io::default_stripe_size_bytes,
                      stripe_rows);
  };

  {
    auto const [expected, filepath] = generate_input(false, 1'000);
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{245'000});
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
  {
    auto const [expected, filepath] = generate_input(false, cudf::io::default_stripe_size_rows);
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{245'000});
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  {
    auto const [expected, filepath] = generate_input(true, 1'000);
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{245'000});
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
  {
    auto const [expected, filepath] = generate_input(true, cudf::io::default_stripe_size_rows);
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{245'000});
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
}

TEST_F(OrcChunkedReaderTest, TestChunkedReadBoundaryCases)
{
  // Tests some specific boundary conditions in the split calculations.

  auto constexpr num_rows = 40'000;

  auto const [expected, filepath] = [num_rows]() {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    auto const value_iter = cuda::counting_iterator<int32_t>{0};
    input_columns.emplace_back(int32s_col(value_iter, value_iter + num_rows).release());
    return write_file(input_columns, "chunked_read_simple_boundary");
  }();

  // Test with zero limit: everything will be read in one chunk.
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{0UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a very small limit: 1 byte.
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{1UL});
    // Number of chunks is 4 because of using default `output_row_granularity = 10k`.
    EXPECT_EQ(num_chunks, 4);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a very small limit: 1 byte, and small value of `output_row_granularity`.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{1UL}, output_row_granularity{1'000});
    EXPECT_EQ(num_chunks, 40);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a very small limit: 1 byte, and large value of `output_row_granularity`.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{1UL}, output_row_granularity{30'000});
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
  // Test with a very large limit
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{2L << 40});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
  // Test with a limit slightly less than one granularity segment of data
  // (output_row_granularity = 10k rows = 40'000 bytes).
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{39'000UL});
    EXPECT_EQ(num_chunks, 4);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a limit exactly the size one granularity segment of data
  // (output_row_granularity = 10k rows = 40'000 bytes).
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{40'000UL});
    EXPECT_EQ(num_chunks, 4);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a limit slightly more than one granularity segment of data
  // (output_row_granularity = 10k rows = 40'000 bytes).
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{41'000UL});
    EXPECT_EQ(num_chunks, 4);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a limit slightly less than two granularity segments of data
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{79'000UL});
    EXPECT_EQ(num_chunks, 4);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a limit exactly the size of two granularity segments of data minus 1 byte.
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{79'999UL});
    EXPECT_EQ(num_chunks, 4);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a limit exactly the size of two granularity segments of data.
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{80'000UL});
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a limit slightly more the size two granularity segments of data.
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{81'000});
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a limit exactly the size of the input minus 1 byte.
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{159'999UL});
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a limit exactly the size of the input.
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{160'000UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
}

TEST_F(OrcChunkedReaderTest, TestChunkedReadWithString)
{
  auto constexpr num_rows           = 60'000;
  auto constexpr output_granularity = output_row_granularity{20'000};

  auto const generate_input = [num_rows](bool nullable) {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    auto const value_iter = cuda::counting_iterator<int32_t>{0};

    // ints                               Granularity Segment  total bytes   cumulative bytes
    // 20000 rows of 4 bytes each               = A0           80000         80000
    // 20000 rows of 4 bytes each               = A1           80000         160000
    // 20000 rows of 4 bytes each               = A2           80000         240000
    input_columns.emplace_back(int32s_col(value_iter, value_iter + num_rows).release());

    // strings                            Granularity Segment  total bytes   cumulative bytes
    // 20000 rows of 1 char each    (20000  + 80004) = B0      100004        100004
    // 20000 rows of 4 chars each   (80000  + 80004) = B1      160004        260008
    // 20000 rows of 16 chars each  (320000 + 80004) = B2      400004        660012
    auto const strings  = std::vector<std::string>{"a", "bbbb", "cccccccccccccccc"};
    auto const str_iter = cudf::detail::make_counting_transform_iterator(0, [&](int32_t i) {
      if (i < 20000) { return strings[0]; }
      if (i < 40000) { return strings[1]; }
      return strings[2];
    });
    input_columns.emplace_back(strings_col(str_iter, str_iter + num_rows).release());

    // Cumulative sizes:
    // A0 + B0 :  180004
    // A1 + B1 :  420008
    // A2 + B2 :  900012
    //                                    skip_rows / num_rows
    // byte_limit==500000  should give 2 chunks: {0, 40000}, {40000, 20000}
    // byte_limit==1000000 should give 1 chunks: {0, 60000},
    return write_file(input_columns, "chunked_read_with_strings", nullable);
  };

  auto const [expected_no_null, filepath_no_null]       = generate_input(false);
  auto const [expected_with_nulls, filepath_with_nulls] = generate_input(true);

  // Test with zero limit: everything will be read in one chunk.
  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{0UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{0UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Test with a very small limit: 1 byte.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath_no_null, output_limit{1UL}, output_granularity);
    EXPECT_EQ(num_chunks, 3);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] =
      chunked_read(filepath_with_nulls, output_limit{1UL}, output_granularity);
    EXPECT_EQ(num_chunks, 3);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Test with a very large limit.
  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{2L << 40});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{2L << 40});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Other tests:

  {
    auto const [result, num_chunks] =
      chunked_read(filepath_no_null, output_limit{500'000UL}, output_granularity);
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] =
      chunked_read(filepath_with_nulls, output_limit{500'000UL}, output_granularity);
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{1'000'000UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{1'000'000UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }
}

TEST_F(OrcChunkedReaderTest, TestChunkedReadWithStructs)
{
  auto constexpr num_rows           = 100'000;
  auto constexpr output_granularity = output_row_granularity{20'000};

  auto const generate_input = [num_rows](bool nullable) {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    auto const int_iter = cuda::counting_iterator<int32_t>{0};
    input_columns.emplace_back(int32s_col(int_iter, int_iter + num_rows).release());
    input_columns.emplace_back([=] {
      auto child1 = int32s_col(int_iter, int_iter + num_rows);
      auto child2 = int32s_col(int_iter + num_rows, int_iter + num_rows * 2);

      auto const str_iter = cudf::detail::make_counting_transform_iterator(
        0, [&](int32_t i) { return std::to_string(i); });
      auto child3 = strings_col{str_iter, str_iter + num_rows};

      return structs_col{{child1, child2, child3}}.release();
    }());

    return write_file(input_columns, "chunked_read_with_structs", nullable);
  };

  auto const [expected_no_null, filepath_no_null]       = generate_input(false);
  auto const [expected_with_nulls, filepath_with_nulls] = generate_input(true);

  // Test with zero limit: everything will be read in one chunk.
  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{0UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{0UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Test with a very small limit: 1 byte.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath_no_null, output_limit{1UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] =
      chunked_read(filepath_with_nulls, output_limit{1UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Test with a very large limit.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath_no_null, output_limit{2L << 40}, output_granularity);
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] =
      chunked_read(filepath_with_nulls, output_limit{2L << 40}, output_granularity);
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Other tests:

  {
    auto const [result, num_chunks] =
      chunked_read(filepath_no_null, output_limit{500'000UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] =
      chunked_read(filepath_with_nulls, output_limit{500'000UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }
}

TEST_F(OrcChunkedReaderTest, TestChunkedReadWithListsNoNulls)
{
  auto constexpr num_rows           = 100'000;
  auto constexpr output_granularity = output_row_granularity{20'000};

  auto const [expected, filepath] = [num_rows]() {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    // 20000 rows in 1 segment consist of:
    //
    // 20001 offsets :   80004  bytes
    // 30000 ints    :   120000 bytes
    // total         :   200004 bytes
    //
    // However, `segmented_row_bit_count` used in chunked reader returns 200000,
    // thus we consider as having only 200000 bytes in total.
    auto const template_lists = int32s_lists_col{
      int32s_lists_col{}, int32s_lists_col{0}, int32s_lists_col{1, 2}, int32s_lists_col{3, 4, 5}};

    auto const gather_iter =
      cudf::detail::make_counting_transform_iterator(0, [&](int32_t i) { return i % 4; });
    auto const gather_map = int32s_col(gather_iter, gather_iter + num_rows);
    input_columns.emplace_back(
      std::move(cudf::gather(cudf::table_view{{template_lists}}, gather_map)->release().front()));

    return write_file(input_columns, "chunked_read_with_lists_no_null");
  }();

  // Test with zero limit: everything will be read in one chunk.
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{0UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a very small limit: 1 byte.
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{1UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a very large limit.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{2L << 40UL}, output_granularity);
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Chunk size slightly less than 1 row segment (forcing it to be at least 1 segment per read).
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{199'999UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Chunk size exactly 1 row segment.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{200'000UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Chunk size == size of 2 segments. Totally have 3 chunks.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{400'000UL}, output_granularity);
    EXPECT_EQ(num_chunks, 3);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Chunk size == size of 2 segment minus one byte: each chunk will be just one segment.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{399'999UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
}

TEST_F(OrcChunkedReaderTest, TestChunkedReadWithListsHavingNulls)
{
  auto constexpr num_rows           = 100'000;
  auto constexpr output_granularity = output_row_granularity{20'000};

  auto const [expected, filepath] = [num_rows]() {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    // 20000 rows in 1 page consist of:
    //
    // 625 validity words :   2500 bytes   (a null every 4 rows: null at indices [3, 7, 11, ...])
    // 20001 offsets      :   80004  bytes
    // 15000 ints         :   60000 bytes
    // total              :   142504 bytes
    //
    // However, `segmented_row_bit_count` used in chunked reader returns 142500,
    // thus we consider as having only 142500 bytes in total.
    auto const template_lists =
      int32s_lists_col{// these will all be null
                       int32s_lists_col{},
                       int32s_lists_col{0},
                       int32s_lists_col{1, 2},
                       int32s_lists_col{3, 4, 5, 6, 7, 8, 9} /* this list will be nullified out */};
    auto const gather_iter =
      cudf::detail::make_counting_transform_iterator(0, [&](int32_t i) { return i % 4; });
    auto const gather_map = int32s_col(gather_iter, gather_iter + num_rows);
    input_columns.emplace_back(
      std::move(cudf::gather(cudf::table_view{{template_lists}}, gather_map)->release().front()));

    return write_file(input_columns, "chunked_read_with_lists_nulls", true /*nullable*/);
  }();

  // Test with zero limit: everything will be read in one chunk.
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{0UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a very small limit: 1 byte.
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{1UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a very large limit.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{2L << 40}, output_granularity);
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Chunk size slightly less than 1 row segment (forcing it to be at least 1 segment per read).
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{142'499UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Chunk size exactly 1 row segment.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{142'500UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Chunk size == size of 2 segments. Totally have 3 chunks.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{285'000UL}, output_granularity);
    EXPECT_EQ(num_chunks, 3);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Chunk size == size of 2 segment minus one byte: each chunk will be just one segment.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{284'999UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
}

TEST_F(OrcChunkedReaderTest, TestChunkedReadWithStructsOfLists)
{
  auto constexpr num_rows = 100'000;

  // Size of each segment (10k row by default) is from 537k to 560k bytes (no nulls)
  // and from 456k to 473k (with nulls).
  auto const generate_input = [num_rows](bool nullable) {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    auto const int_iter = cuda::counting_iterator<int32_t>{0};
    input_columns.emplace_back(int32s_col(int_iter, int_iter + num_rows).release());
    input_columns.emplace_back([=] {
      std::vector<std::unique_ptr<cudf::column>> child_columns;
      child_columns.emplace_back(int32s_col(int_iter, int_iter + num_rows).release());
      child_columns.emplace_back(
        int32s_col(int_iter + num_rows, int_iter + num_rows * 2).release());

      auto const str_iter = cudf::detail::make_counting_transform_iterator(0, [&](int32_t i) {
        return std::to_string(i) + "++++++++++++++++++++" + std::to_string(i);
      });
      child_columns.emplace_back(strings_col{str_iter, str_iter + num_rows}.release());

      auto const template_lists = int32s_lists_col{
        int32s_lists_col{}, int32s_lists_col{0}, int32s_lists_col{0, 1}, int32s_lists_col{0, 1, 2}};
      auto const gather_iter =
        cudf::detail::make_counting_transform_iterator(0, [&](int32_t i) { return i % 4; });
      auto const gather_map = int32s_col(gather_iter, gather_iter + num_rows);
      child_columns.emplace_back(
        std::move(cudf::gather(cudf::table_view{{template_lists}}, gather_map)->release().front()));

      return structs_col(std::move(child_columns)).release();
    }());

    return write_file(input_columns, "chunked_read_with_structs_of_lists", nullable);
  };

  auto const [expected_no_null, filepath_no_null]       = generate_input(false);
  auto const [expected_with_nulls, filepath_with_nulls] = generate_input(true);

  // Test with zero limit: everything will be read in one chunk.
  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{0UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{0UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Test with a very small limit: 1 byte.
  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{1UL});
    EXPECT_EQ(num_chunks, 10);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{1UL});
    EXPECT_EQ(num_chunks, 10);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Test with a very large limit.
  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{2L << 40});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{2L << 40});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Other tests:

  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{1'000'000UL});
    EXPECT_EQ(num_chunks, 10);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{1'500'000UL});
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{2'000'000UL});
    EXPECT_EQ(num_chunks, 4);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{5'000'000UL});
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{1'000'000UL});
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{1'500'000UL});
    EXPECT_EQ(num_chunks, 4);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{2'000'000UL});
    EXPECT_EQ(num_chunks, 3);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{5'000'000UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }
}

TEST_F(OrcChunkedReaderTest, TestChunkedReadWithListsOfStructs)
{
  auto constexpr num_rows = 100'000;

  // Size of each segment (10k row by default) is from 450k to 530k bytes (no nulls)
  // and from 330k to 380k (with nulls).
  auto const generate_input = [num_rows](bool nullable) {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    auto const int_iter = cuda::counting_iterator<int32_t>{0};
    input_columns.emplace_back(int32s_col(int_iter, int_iter + num_rows).release());

    auto offsets = std::vector<cudf::size_type>{};
    offsets.reserve(num_rows * 2);
    cudf::size_type num_structs = 0;
    for (int i = 0; i < num_rows; ++i) {
      offsets.push_back(num_structs);
      auto const new_list_size = i % 4;
      num_structs += new_list_size;
    }
    offsets.push_back(num_structs);

    auto const make_structs_col = [=] {
      auto child1 = int32s_col(int_iter, int_iter + num_structs);
      auto child2 = int32s_col(int_iter + num_structs, int_iter + num_structs * 2);

      auto const str_iter = cudf::detail::make_counting_transform_iterator(
        0, [&](int32_t i) { return std::to_string(i) + std::to_string(i) + std::to_string(i); });
      auto child3 = strings_col{str_iter, str_iter + num_structs};

      return structs_col{{child1, child2, child3}}.release();
    };

    input_columns.emplace_back(
      cudf::make_lists_column(static_cast<cudf::size_type>(offsets.size() - 1),
                              int32s_col(offsets.begin(), offsets.end()).release(),
                              make_structs_col(),
                              0,
                              rmm::device_buffer{}));

    return write_file(input_columns, "chunked_read_with_lists_of_structs", nullable);
  };

  auto const [expected_no_null, filepath_no_null]       = generate_input(false);
  auto const [expected_with_nulls, filepath_with_nulls] = generate_input(true);

  // Test with zero limit: everything will be read in one chunk.
  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{0UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{0UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Test with a very small limit: 1 byte.
  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{1UL});
    EXPECT_EQ(num_chunks, 10);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{1UL});
    EXPECT_EQ(num_chunks, 10);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Test with a very large limit.
  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{2L << 40});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{2L << 40});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Other tests.

  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{1'000'000UL});
    EXPECT_EQ(num_chunks, 7);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{1'500'000UL});
    EXPECT_EQ(num_chunks, 4);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{2'000'000UL});
    EXPECT_EQ(num_chunks, 3);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{5'000'000UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{1'000'000UL});
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{1'500'000UL});
    EXPECT_EQ(num_chunks, 3);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{2'000'000UL});
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{5'000'000UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }
}

TEST_F(OrcChunkedReaderTest, TestChunkedReadNullCount)
{
  auto constexpr num_rows = 100'000;

  auto const sequence = cuda::constant_iterator(1);
  auto const validity =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 4 != 3; });
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(int32s_col{sequence, sequence + num_rows, validity}.release());
  auto const expected = std::make_unique<cudf::table>(std::move(cols));

  auto const filepath          = temp_env->get_temp_filepath("chunked_reader_null_count.orc");
  auto const stripe_limit_rows = num_rows / 5;
  auto const write_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, *expected)
      .stripe_size_rows(stripe_limit_rows)
      .build();
  cudf::io::write_orc(write_opts);

  auto const byte_limit = stripe_limit_rows * sizeof(int);
  auto const read_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath}).build();
  auto reader =
    cudf::io::chunked_orc_reader(byte_limit, 0UL /*read_limit*/, stripe_limit_rows, read_opts);

  do {
    // Every fourth row is null.
    EXPECT_EQ(reader.read_chunk().tbl->get_column(0).null_count(), stripe_limit_rows / 4UL);
  } while (reader.has_next());
}

namespace {

std::size_t constexpr input_limit_expected_file_count = 3;

std::vector<std::string> input_limit_get_test_names(std::string const& base_filename)
{
  return {base_filename + "_a.orc", base_filename + "_b.orc", base_filename + "_c.orc"};
}

void input_limit_test_write_one(std::string const& filepath,
                                cudf::table_view const& input,
                                cudf::size_type stripe_size_rows,
                                cudf::io::compression_type compression)
{
  auto const out_opts = cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, input)
                          .compression(compression)
                          .stripe_size_rows(stripe_size_rows)
                          .build();
  cudf::io::write_orc(out_opts);
}

void input_limit_test_write(
  std::vector<std::string> const& test_files,
  cudf::table_view const& input,
  cudf::size_type stripe_size_rows = 20'000 /*write relatively small stripes by default*/)
{
  CUDF_EXPECTS(test_files.size() == input_limit_expected_file_count,
               "Unexpected count of test filenames.");

  // ZSTD yields a very small decompression size, can be much smaller than SNAPPY.
  // However, ORC reader typically over-estimates the decompression size of data
  // compressed by ZSTD to be very large, can be much larger than that of SNAPPY.
  // That is because ZSTD may use a lot of scratch space at decode time
  // (2.5x the total decompressed buffer size).
  // As such, we may see smaller output chunks for the input data compressed by ZSTD.
  input_limit_test_write_one(
    test_files[0], input, stripe_size_rows, cudf::io::compression_type::NONE);
  input_limit_test_write_one(
    test_files[1], input, stripe_size_rows, cudf::io::compression_type::ZSTD);
  input_limit_test_write_one(
    test_files[2], input, stripe_size_rows, cudf::io::compression_type::SNAPPY);
}

void input_limit_test_read(int test_location,
                           std::vector<std::string> const& test_files,
                           cudf::table_view const& input,
                           output_limit output_limit_bytes,
                           input_limit input_limit_bytes,
                           int const* expected_chunk_counts)
{
  CUDF_EXPECTS(test_files.size() == input_limit_expected_file_count,
               "Unexpected count of test filenames.");

  for (size_t idx = 0; idx < test_files.size(); ++idx) {
    SCOPED_TRACE("Original line of failure: " + std::to_string(test_location) +
                 ", file idx: " + std::to_string(idx));
    auto const [result, num_chunks] =
      chunked_read(test_files[idx], output_limit_bytes, input_limit_bytes);
    // Skipping the chunk count check because compression can impact it
    // EXPECT_EQ(expected_chunk_counts[idx], num_chunks);
    // TODO: equal
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*result, input);
  }
}

}  // namespace

struct OrcChunkedReaderInputLimitTest : public cudf::test::BaseFixture {};

TEST_F(OrcChunkedReaderInputLimitTest, SingleFixedWidthColumn)
{
  auto constexpr num_rows = 1'000'000;
  auto const iter1        = cuda::make_constant_iterator(15);
  auto const col1         = doubles_col(iter1, iter1 + num_rows);

  auto const filename   = std::string{"single_col_fixed_width"};
  auto const test_files = input_limit_get_test_names(temp_env->get_temp_filepath(filename));
  auto const input      = cudf::table_view{{col1}};
  input_limit_test_write(test_files, input);

  {
    int constexpr expected[] = {50, 50, 50};
    input_limit_test_read(
      __LINE__, test_files, input, output_limit{0UL}, input_limit{1UL}, expected);
  }

  {
    int constexpr expected[] = {17, 13, 10};
    input_limit_test_read(
      __LINE__, test_files, input, output_limit{0UL}, input_limit{2 * 1024 * 1024UL}, expected);
  }
}

TEST_F(OrcChunkedReaderInputLimitTest, MixedColumns)
{
  auto constexpr num_rows = 1'000'000;

  auto const iter1 = cuda::counting_iterator<int>{0};
  auto const col1  = int32s_col(iter1, iter1 + num_rows);

  auto const iter2 =
    cudf::detail::make_counting_transform_iterator(0, [](int i) { return static_cast<double>(i); });
  auto const col2 = doubles_col(iter2, iter2 + num_rows);

  auto const strings  = std::vector<std::string>{"abc", "de", "fghi"};
  auto const str_iter = cudf::detail::make_counting_transform_iterator(0, [&](int32_t i) {
    if (i < 250000) { return strings[0]; }
    if (i < 750000) { return strings[1]; }
    return strings[2];
  });
  auto const col3     = strings_col(str_iter, str_iter + num_rows);

  auto const filename   = std::string{"mixed_columns"};
  auto const test_files = input_limit_get_test_names(temp_env->get_temp_filepath(filename));
  auto const input      = cudf::table_view{{col1, col2, col3}};
  input_limit_test_write(test_files, input);

  {
    int constexpr expected[] = {50, 50, 50};
    input_limit_test_read(
      __LINE__, test_files, input, output_limit{0UL}, input_limit{1UL}, expected);
  }

  {
    int constexpr expected[] = {17, 50, 17};
    input_limit_test_read(
      __LINE__, test_files, input, output_limit{0UL}, input_limit{2 * 1024 * 1024UL}, expected);
  }
}

namespace {

struct offset_gen {
  int const group_size;
  __device__ int operator()(int i) const { return i * group_size; }
};

template <typename T>
struct value_gen {
  __device__ T operator()(int i) const { return i % 1024; }
};

struct char_values {
  __device__ int8_t operator()(int i) const
  {
    int const index = (i / 2) % 3;
    // Generate repeating 3-runs of 2 values each: "aabbccaabbcc...".
    return index == 0 ? 'a' : (index == 1 ? 'b' : 'c');
  }
};

}  // namespace

TEST_F(OrcChunkedReaderInputLimitTest, ListType)
{
  // this test runs over 3 hours when racecheck is used
  if (getenv("LIBCUDF_RACECHECK_ENABLED")) { GTEST_SKIP(); }

  int constexpr num_rows  = 50'000'000;
  int constexpr list_size = 4;

  auto const stream = cudf::get_default_stream();
  auto const iter   = cuda::counting_iterator<int32_t>{0};

  auto offset_col = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows + 1, cudf::mask_state::UNALLOCATED);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    iter,
                    iter + num_rows + 1,
                    offset_col->mutable_view().begin<int>(),
                    offset_gen{list_size});

  int constexpr num_ints = num_rows * list_size;
  auto value_col         = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, num_ints, cudf::mask_state::UNALLOCATED);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    iter,
                    iter + num_ints,
                    value_col->mutable_view().begin<int>(),
                    value_gen<int>{});

  auto const lists_col =
    cudf::make_lists_column(num_rows, std::move(offset_col), std::move(value_col), 0, {});

  auto const filename   = std::string{"list_type"};
  auto const test_files = input_limit_get_test_names(temp_env->get_temp_filepath(filename));
  auto const input      = cudf::table_view{{*lists_col}};

  // Although we set `stripe_size_rows` to be very large, the writer only write
  // 250k rows (top level) per stripe due to having nested type.
  // Thus, we have 200 stripes in total.
  input_limit_test_write(test_files, input, cudf::io::default_stripe_size_rows);

  {
    int constexpr expected[] = {3, 40, 3};
    input_limit_test_read(
      __LINE__, test_files, input, output_limit{0UL}, input_limit{5 * 1024 * 1024UL}, expected);
  }

  {
    int constexpr expected[] = {8, 40, 9};
    input_limit_test_read(__LINE__,
                          test_files,
                          input,
                          output_limit{128 * 1024 * 1024UL},
                          input_limit{5 * 1024 * 1024UL},
                          expected);
  }
}

TEST_F(OrcChunkedReaderInputLimitTest, MixedColumnsHavingList)
{
  // this test runs over 3 hours when racecheck is used
  if (getenv("LIBCUDF_RACECHECK_ENABLED")) { GTEST_SKIP(); }

  int constexpr num_rows  = 50'000'000;
  int constexpr list_size = 4;
  int constexpr str_size  = 3;

  auto const stream = cudf::get_default_stream();
  auto const iter   = cuda::counting_iterator<int32_t>{0};

  // list<int>
  auto offset_col = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows + 1, cudf::mask_state::UNALLOCATED);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    iter,
                    iter + num_rows + 1,
                    offset_col->mutable_view().begin<int>(),
                    offset_gen{list_size});

  int constexpr num_ints = num_rows * list_size;
  auto value_col         = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, num_ints, cudf::mask_state::UNALLOCATED);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    iter,
                    iter + num_ints,
                    value_col->mutable_view().begin<int>(),
                    value_gen<int>{});

  auto const lists_col =
    cudf::make_lists_column(num_rows, std::move(offset_col), std::move(value_col), 0, {});

  // strings
  int constexpr num_chars = num_rows * str_size;
  auto str_offset_col     = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows + 1, cudf::mask_state::UNALLOCATED);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    iter,
                    iter + num_rows + 1,
                    str_offset_col->mutable_view().begin<int>(),
                    offset_gen{str_size});
  rmm::device_buffer str_chars(num_chars, stream);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    iter,
                    iter + num_chars,
                    static_cast<int8_t*>(str_chars.data()),
                    char_values{});
  auto const str_col =
    cudf::make_strings_column(num_rows, std::move(str_offset_col), std::move(str_chars), 0, {});

  // doubles
  auto const double_col = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::FLOAT64}, num_rows, cudf::mask_state::UNALLOCATED);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    iter,
                    iter + num_rows,
                    double_col->mutable_view().begin<double>(),
                    value_gen<double>{});

  auto const filename   = std::string{"mixed_cols_having_list"};
  auto const test_files = input_limit_get_test_names(temp_env->get_temp_filepath(filename));
  auto const input      = cudf::table_view{{*lists_col, *str_col, *double_col}};

  // Although we set `stripe_size_rows` to be very large, the writer only write
  // 250k rows (top level) per stripe due to having nested type.
  // Thus, we have 200 stripes in total.
  input_limit_test_write(test_files, input, cudf::io::default_stripe_size_rows);

  {
    int constexpr expected[] = {13, 8, 6};
    input_limit_test_read(
      __LINE__, test_files, input, output_limit{0UL}, input_limit{128 * 1024 * 1024UL}, expected);
  }

  {
    int constexpr expected[] = {13, 15, 17};
    input_limit_test_read(__LINE__,
                          test_files,
                          input,
                          output_limit{128 * 1024 * 1024UL},
                          input_limit{128 * 1024 * 1024UL},
                          expected);
  }
}

TEST_F(OrcChunkedReaderInputLimitTest, ReadWithRowSelection)
{
  // `num_rows` should not be divisible by `stripe_size_rows`, to test the correctness of row
  // selections.
  int64_t constexpr num_rows    = 100'517'687l;
  int constexpr rows_per_stripe = 100'000;
  static_assert(num_rows % rows_per_stripe != 0,
                "`num_rows` should not be divisible by `stripe_size_rows`.");

  auto const it    = cuda::counting_iterator<int32_t>{0};
  auto const col   = int32s_col(it, it + num_rows);
  auto const input = cudf::table_view{{col}};

  auto const filepath = temp_env->get_temp_filepath("chunk_read_with_row_selection.orc");
  auto const write_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, input)
      .stripe_size_rows(rows_per_stripe)
      .build();
  cudf::io::write_orc(write_opts);

  // Verify metadata.
  auto const metadata = cudf::io::read_orc_metadata(cudf::io::source_info{filepath});
  EXPECT_EQ(metadata.num_rows(), num_rows);
  EXPECT_EQ(metadata.num_stripes(), num_rows / rows_per_stripe + 1);

  int constexpr random_val = 123456;

  // Read some random number or rows that is not stripe size.
  int constexpr num_rows_to_read = rows_per_stripe * 5 + random_val;

  // Just shift the read data region back by a random offset.
  auto const num_rows_to_skip = num_rows - num_rows_to_read - random_val;

  auto const sequence_start = num_rows_to_skip % num_rows;
  auto const skipped_col = int32s_col(it + sequence_start, it + sequence_start + num_rows_to_read);
  auto const expected    = cudf::table_view{{skipped_col}};

  auto const read_opts = cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath})
                           .use_index(false)
                           .skip_rows(num_rows_to_skip)
                           .num_rows(num_rows_to_read)
                           .build();

  auto reader = cudf::io::chunked_orc_reader(
    60'000UL * sizeof(int) /*output limit, equal to 60k rows, less than rows in 1 stripe*/,
    rows_per_stripe * sizeof(int) /*input limit, around size of 1 stripe's decoded data*/,
    50'000 /*output granularity, or minimum number of rows for the output chunk*/,
    read_opts);

  auto num_chunks  = 0;
  auto read_tables = std::vector<std::unique_ptr<cudf::table>>{};
  auto tviews      = std::vector<cudf::table_view>{};

  do {
    auto chunk = reader.read_chunk();
    // Each output chunk should have either exactly 50k rows, or num_rows_to_read % 50k.
    // Skipping the chunk count check because compression can impact it
    // EXPECT_TRUE(chunk.tbl->num_rows() == 50000 ||
    //             chunk.tbl->num_rows() == num_rows_to_read % 50000);

    tviews.emplace_back(chunk.tbl->view());
    read_tables.emplace_back(std::move(chunk.tbl));
    ++num_chunks;
  } while (reader.has_next());

  auto const read_result = cudf::concatenate(tviews);
  // Skipping the chunk count check because compression can impact it
  // EXPECT_EQ(num_chunks, 13);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, read_result->view());
}

TEST_F(OrcChunkedReaderInputLimitTest, SizeTypeRowsOverflow)
{
  // this test runs over 3 hours when racecheck is used
  if (getenv("LIBCUDF_RACECHECK_ENABLED")) { GTEST_SKIP(); }

  using data_type = int16_t;
  using data_col  = cudf::test::fixed_width_column_wrapper<data_type, int64_t>;

  int64_t constexpr num_rows    = 500'000'000l;
  int constexpr rows_per_stripe = 1'000'000;
  int constexpr num_reps        = 10;
  int64_t constexpr total_rows  = num_rows * num_reps;
  static_assert(total_rows > std::numeric_limits<cudf::size_type>::max());

  auto const it =
    thrust::make_transform_iterator(cuda::counting_iterator<int64_t>{0}, [num_rows](int64_t i) {
      return (i % num_rows) % static_cast<int64_t>(std::numeric_limits<data_type>::max() / 2);
    });
  auto const col         = data_col(it, it + num_rows);
  auto const chunk_table = cudf::table_view{{col}};

  std::vector<char> data_buffer;
  {
    auto const write_opts =
      cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{&data_buffer})
        .stripe_size_rows(rows_per_stripe)
        .build();

    auto writer = cudf::io::orc_chunked_writer(write_opts);
    for (int i = 0; i < num_reps; ++i) {
      writer.write(chunk_table);
    }
  }

  // Verify metadata.
  auto const metadata =
    cudf::io::read_orc_metadata(cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(data_buffer.data()), data_buffer.size()}});
  EXPECT_EQ(metadata.num_rows(), total_rows);
  EXPECT_EQ(metadata.num_stripes(), total_rows / rows_per_stripe);

  // Read with row selections and memory limit.
  {
    int64_t constexpr num_rows_to_read = 5'000'000l;
    int64_t const num_rows_to_skip =
      static_cast<int64_t>(metadata.num_rows()) - num_rows_to_read -
      123456l /*just shift the read data region back by a random offset*/;

    // Check validity of the last 5 million rows.
    auto const sequence_start = num_rows_to_skip % num_rows;
    auto const skipped_col = data_col(it + sequence_start, it + sequence_start + num_rows_to_read);
    auto const expected    = cudf::table_view{{skipped_col}};

    auto const read_opts =
      cudf::io::orc_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(data_buffer.data()), data_buffer.size()}})
        .use_index(false)
        .skip_rows(num_rows_to_skip)
        .num_rows(num_rows_to_read)
        .build();
    auto reader = cudf::io::chunked_orc_reader(
      600'000UL * sizeof(data_type) /* output limit, equal to 600k rows */,
      rows_per_stripe * sizeof(data_type) /* input limit, around size of 1 stripe's decoded data */,
      rows_per_stripe / 2 /* output granularity, or minimum number of rows for the output chunk */,
      read_opts);

    auto num_chunks  = 0;
    auto read_tables = std::vector<std::unique_ptr<cudf::table>>{};
    auto tviews      = std::vector<cudf::table_view>{};

    do {
      auto chunk = reader.read_chunk();
      ++num_chunks;
      tviews.emplace_back(chunk.tbl->view());
      read_tables.emplace_back(std::move(chunk.tbl));
    } while (reader.has_next());

    auto const read_result = cudf::concatenate(tviews);
    // Skipping the chunk count check because compression can impact it
    // EXPECT_EQ(num_chunks, 11);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, read_result->view());
  }

  // The test below requires a huge amount of memory, thus it is disabled by default.
#ifdef LOCAL_TEST
  // Read with only output limit -- there is no limit on the memory usage.
  // However, the reader should be able to detect and load only enough stripes each time
  // to avoid decoding a table having number of rows that exceeds the column size limit.
  {
    auto const read_opts =
      cudf::io::orc_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(data_buffer.data()), data_buffer.size()}})
        .use_index(false)
        .build();
    auto reader = cudf::io::chunked_orc_reader(
      static_cast<std::size_t>(rows_per_stripe * 5.7) *
        sizeof(data_type) /* output limit, equal to 5.7M rows */,
      0UL /* no input limit */,
      rows_per_stripe / 2 /* output granularity, or minimum number of rows for the output chunk */,
      read_opts);

    int num_chunks          = 0;
    int64_t num_read_rows   = 0;
    int64_t test_rows_start = 0;
    auto test_chunk         = std::unique_ptr<cudf::table>{};

    do {
      auto chunk            = reader.read_chunk();
      auto const chunk_rows = chunk.tbl->num_rows();

      // Just randomly select one output chunk to verify.
      if (num_chunks == 123) {
        test_rows_start = num_read_rows;
        test_chunk      = std::move(chunk.tbl);
      }

      ++num_chunks;
      num_read_rows += chunk_rows;
    } while (reader.has_next());

    EXPECT_EQ(num_read_rows, total_rows);

    // Typically, we got a chunk having 5M rows.
    // However, since the reader internally splits file stripes that are not multiple of 5 stripes,
    // we may have some extra chunks that have less than 5M rows.
    // Skipping the chunk count check because compression can impact it
    // EXPECT_EQ(num_chunks, 1002);

    // Verify the selected chunk.
    using namespace cudf::test::iterators;
    auto const skipped_col =
      data_col(it + test_rows_start, it + test_rows_start + test_chunk->num_rows(), no_nulls());
    auto const expected = cudf::table_view{{skipped_col}};
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, test_chunk->view());
  }

#endif  // LOCAL_TEST
}

TEST_P(OrcChunkedDecompressionTest, RoundTripBasic)
{
  auto const compression_type = std::get<1>(GetParam());

  auto const num_rows = 12'345;

  std::vector<std::unique_ptr<cudf::column>> input_columns;
  auto value_iter = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i / 4; });
  input_columns.emplace_back(int32s_col(value_iter, value_iter + num_rows).release());
  input_columns.emplace_back(int64s_col(value_iter, value_iter + num_rows).release());
  auto expected = std::make_unique<cudf::table>(std::move(input_columns));

  auto const filepath = temp_env->get_temp_filepath("chunked_read_compressions.orc");
  auto const write_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, *expected)
      .compression(compression_type)
      .stripe_size_rows(2'000)
      .row_index_stride(1'000)
      .build();
  cudf::io::write_orc(write_opts);

  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{0}, input_limit{2'400'000});
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{0}, input_limit{240'000});
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{0}, input_limit{24'000});
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
}

INSTANTIATE_TEST_CASE_P(Nvcomp,
                        OrcChunkedDecompressionTest,
                        ::testing::Combine(::testing::Values("NVCOMP"),
                                           ::testing::Values(cudf::io::compression_type::AUTO,
                                                             cudf::io::compression_type::SNAPPY,
                                                             cudf::io::compression_type::LZ4,
                                                             cudf::io::compression_type::ZLIB,
                                                             cudf::io::compression_type::ZSTD)));

INSTANTIATE_TEST_CASE_P(DeviceInternal,
                        OrcChunkedDecompressionTest,
                        ::testing::Combine(::testing::Values("DEVICE_INTERNAL"),
                                           ::testing::Values(cudf::io::compression_type::AUTO,
                                                             cudf::io::compression_type::SNAPPY)));

INSTANTIATE_TEST_CASE_P(Host,
                        OrcChunkedDecompressionTest,
                        ::testing::Combine(::testing::Values("HOST"),
                                           ::testing::Values(cudf::io::compression_type::AUTO,
                                                             cudf::io::compression_type::SNAPPY,
                                                             cudf::io::compression_type::ZSTD)));
