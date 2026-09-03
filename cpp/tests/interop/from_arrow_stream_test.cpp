/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/nanoarrow_utils.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/interop.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <numeric>
#include <vector>

struct FromArrowStreamTest : public cudf::test::BaseFixture {};

std::pair<std::unique_ptr<cudf::column>, ArrowArrayStream> get_nanoarrow_chunked_stream(
  int num_copies, cudf::size_type length)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  std::vector<nanoarrow::UniqueArray> arrays;
  for (auto i = 0; i < 3; ++i) {
    auto [tbl, sch, arr] = get_nanoarrow_host_tables(length);
    // just use the first column
    columns.push_back(std::move(tbl->release().front()));
    arrays.push_back(std::move(arr->children[0]));
  }
  std::vector<cudf::column_view> views;
  for (auto const& col : columns) {
    views.push_back(col->view());
  }
  auto expected = cudf::concatenate(views);

  nanoarrow::UniqueSchema schema;
  ArrowSchemaInit(schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(schema.get(), NANOARROW_TYPE_INT64));

  ArrowArrayStream stream;
  makeStreamFromArrays(std::move(arrays), std::move(schema), &stream);
  return std::make_pair(std::move(expected), stream);
}

TEST_F(FromArrowStreamTest, BasicTest)
{
  constexpr auto num_copies = 3;
  auto [tbl, stream]        = get_nanoarrow_stream(num_copies);

  auto result = cudf::from_arrow_stream(&stream);
  CUDF_TEST_EXPECT_TABLES_EQUAL(tbl->view(), result->view());
}

TEST_F(FromArrowStreamTest, EmptyTest)
{
  auto [tbl, sch, arr] = get_nanoarrow_host_tables(0);
  std::vector<cudf::table_view> table_views{tbl->view()};
  auto expected = cudf::concatenate(table_views);

  ArrowArrayStream stream;
  makeStreamFromArrays({}, std::move(sch), &stream);
  auto result = cudf::from_arrow_stream(&stream);
  cudf::have_same_types(expected->view(), result->view());
}

TEST_F(FromArrowStreamTest, ChunkedTest)
{
  constexpr auto num_copies = 3;
  constexpr auto length     = 3;
  auto [expected, stream]   = get_nanoarrow_chunked_stream(num_copies, length);

  auto result = cudf::from_arrow_stream_column(&stream);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view());
}

TEST_F(FromArrowStreamTest, EmptyChunkedTest)
{
  constexpr auto num_copies = 3;
  constexpr auto length     = 0;
  auto [expected, stream]   = get_nanoarrow_chunked_stream(num_copies, length);

  auto result = cudf::from_arrow_stream_column(&stream);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected->view());
}

// exercises make_empty_column_from_schema, which builds the column from the schema alone
TEST_F(FromArrowStreamTest, FixedSizeListEmptyTest)
{
  auto schema = make_struct_fixed_size_list_int64_schema(3);

  ArrowArrayStream stream;
  makeStreamFromArrays({}, std::move(schema), &stream);

  auto result = cudf::from_arrow_stream(&stream);
  EXPECT_EQ(result->num_rows(), 0);
  EXPECT_EQ(result->get_column(0).type(), cudf::data_type{cudf::type_id::LIST});
}

// exercises concatenate over columns whose offsets were synthesized rather than copied
TEST_F(FromArrowStreamTest, FixedSizeListChunkedTest)
{
  constexpr int32_t width = 2;
  auto schema             = make_struct_fixed_size_list_int64_schema(width);

  std::vector<nanoarrow::UniqueArray> arrays;
  for (auto i = 0; i < 3; ++i) {
    auto base = static_cast<int64_t>(i * 4);
    arrays.push_back(make_struct_fixed_size_list_int64_array(
      schema.get(), {base + 1, base + 2, base + 3, base + 4}, /*num_rows=*/2));
  }

  auto expected_col =
    cudf::test::lists_column_wrapper<int64_t>{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}};
  cudf::table_view expected_table_view({expected_col});

  ArrowArrayStream stream;
  makeStreamFromArrays(std::move(arrays), std::move(schema), &stream);

  auto result = cudf::from_arrow_stream(&stream);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_table_view, result->view());
}

TEST_F(FromArrowStreamTest, FixedSizeListChunkedNullsTest)
{
  constexpr cudf::size_type num_rows = 4;
  auto schema = make_struct_fixed_size_list_int64_schema(2, /*nullable=*/true);

  std::vector<nanoarrow::UniqueArray> arrays;
  arrays.push_back(
    make_struct_fixed_size_list_int64_array(schema.get(), {1, 2, 3, 4}, 2, /*validity=*/{1, 0}));
  arrays.push_back(
    make_struct_fixed_size_list_int64_array(schema.get(), {5, 6, 7, 8}, 2, /*validity=*/{0, 1}));

  auto child   = cudf::test::fixed_width_column_wrapper<int64_t>{1, 2, 3, 4, 5, 6, 7, 8}.release();
  auto offsets = cudf::test::fixed_width_column_wrapper<int32_t>{0, 2, 4, 6, 8}.release();
  std::vector<uint8_t> validity{1, 0, 0, 1};
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(validity.begin(), validity.end());
  auto expected = cudf::make_lists_column(
    num_rows, std::move(offsets), std::move(child), null_count, std::move(null_mask));

  ArrowArrayStream stream;
  makeStreamFromArrays(std::move(arrays), std::move(schema), &stream);

  auto result       = cudf::from_arrow_stream(&stream);
  auto result_lists = cudf::lists_column_view{result->get_column(0)};
  EXPECT_EQ(result_lists.null_count(), 2);

  auto expected_logical = cudf::purge_nonempty_nulls(expected->view());
  auto result_logical   = cudf::purge_nonempty_nulls(result_lists.parent());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected_logical, *result_logical);
}

TEST_F(FromArrowStreamTest, FixedSizeListSlicedTest)
{
  constexpr int32_t width = 2;
  auto schema             = make_struct_fixed_size_list_int64_schema(width);

  std::vector<nanoarrow::UniqueArray> arrays;
  arrays.push_back(make_struct_fixed_size_list_int64_array(
    schema.get(), {1, 2, 3, 4, 5, 6, 7, 8}, /*num_rows=*/4));
  arrays.front()->length              = 2;
  arrays.front()->children[0]->offset = 1;
  arrays.front()->children[0]->length = 2;

  auto expected = cudf::test::lists_column_wrapper<int64_t>{{3, 4}, {5, 6}};

  ArrowArrayStream stream;
  makeStreamFromArrays(std::move(arrays), std::move(schema), &stream);

  auto result = cudf::from_arrow_stream(&stream);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->get_column(0));
}

TEST_F(FromArrowStreamTest, FixedSizeListBoundaryAndMultiBlockTest)
{
  constexpr int32_t width = 2;
  auto schema             = make_struct_fixed_size_list_int64_schema(width);

  std::vector<nanoarrow::UniqueArray> arrays;
  std::vector<int64_t> expected_values;
  std::vector<int32_t> expected_offsets{0};
  for (auto const num_rows : {cudf::size_type{1024}, cudf::size_type{1025}}) {
    std::vector<int64_t> values(num_rows * width);
    std::iota(values.begin(), values.end(), static_cast<int64_t>(expected_values.size()));
    expected_values.insert(expected_values.end(), values.begin(), values.end());
    arrays.push_back(make_struct_fixed_size_list_int64_array(schema.get(), values, num_rows));
    for (cudf::size_type i = 0; i < num_rows; ++i) {
      expected_offsets.push_back(expected_offsets.back() + width);
    }
  }

  auto expected_offsets_col = cudf::test::fixed_width_column_wrapper<int32_t>(
    expected_offsets.begin(), expected_offsets.end());
  auto expected_child =
    cudf::test::fixed_width_column_wrapper<int64_t>(expected_values.begin(), expected_values.end());

  ArrowArrayStream stream;
  makeStreamFromArrays(std::move(arrays), std::move(schema), &stream);

  auto result       = cudf::from_arrow_stream(&stream);
  auto result_lists = cudf::lists_column_view{result->get_column(0)};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_offsets_col, result_lists.offsets());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_child, result_lists.child());
}
