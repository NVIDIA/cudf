/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
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

void makeStreamFromArrays(std::vector<nanoarrow::UniqueArray> arrays,
                          nanoarrow::UniqueSchema schema,
                          ArrowArrayStream* out)
{
  auto* private_data  = new VectorOfArrays{std::move(arrays), std::move(schema)};
  out->get_schema     = VectorOfArrays::get_schema;
  out->get_next       = VectorOfArrays::get_next;
  out->get_last_error = VectorOfArrays::get_last_error;
  out->release        = VectorOfArrays::release;
  out->private_data   = private_data;
}

std::tuple<std::unique_ptr<cudf::table>, nanoarrow::UniqueSchema, ArrowArrayStream>
get_nanoarrow_stream(int num_copies)
{
  std::vector<std::unique_ptr<cudf::table>> tables;
  // The schema is unique across all tables.
  nanoarrow::UniqueSchema schema;
  std::vector<nanoarrow::UniqueArray> arrays;
  for (auto i = 0; i < num_copies; ++i) {
    auto [tbl, sch, arr] = get_nanoarrow_host_tables(3);
    tables.push_back(std::move(tbl));
    arrays.push_back(std::move(arr));
    if (i == 0) { sch.move(schema.get()); }
  }
  std::vector<cudf::table_view> table_views;
  for (auto const& table : tables) {
    table_views.push_back(table->view());
  }
  auto expected = cudf::concatenate(table_views);

  ArrowArrayStream stream;
  makeStreamFromArrays(std::move(arrays), std::move(schema), &stream);
  return std::make_tuple(std::move(expected), std::move(schema), stream);
}

std::tuple<std::unique_ptr<cudf::column>, nanoarrow::UniqueSchema, ArrowArrayStream>
get_nanoarrow_chunked_stream(int num_copies, cudf::size_type length)
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
  return std::make_tuple(std::move(expected), std::move(schema), stream);
}

TEST_F(FromArrowStreamTest, BasicTest)
{
  constexpr auto num_copies = 3;
  auto [tbl, sch, stream]   = get_nanoarrow_stream(num_copies);

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
  constexpr auto num_copies       = 3;
  constexpr auto length           = 3;
  auto [expected, schema, stream] = get_nanoarrow_chunked_stream(num_copies, length);

  auto result = cudf::from_arrow_stream_column(&stream);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view());
}

TEST_F(FromArrowStreamTest, EmptyChunkedTest)
{
  constexpr auto num_copies       = 3;
  constexpr auto length           = 0;
  auto [expected, schema, stream] = get_nanoarrow_chunked_stream(num_copies, length);

  auto result = cudf::from_arrow_stream_column(&stream);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected->view());
}

namespace {

// Builds a struct schema with one fixed_size_list<int64>[width] child. ArrowSchemaInitFromType
// does not support NANOARROW_TYPE_FIXED_SIZE_LIST, so ArrowSchemaSetTypeFixedSize is used and
// the allocated "item" child gets its type set explicitly.
nanoarrow::UniqueSchema make_fixed_size_list_stream_schema(int32_t width, bool nullable = false)
{
  nanoarrow::UniqueSchema schema;
  ArrowSchemaInit(schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(schema.get(), 1));

  NANOARROW_THROW_NOT_OK(
    ArrowSchemaSetTypeFixedSize(schema->children[0], NANOARROW_TYPE_FIXED_SIZE_LIST, width));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[0], "a"));
  schema->children[0]->flags = nullable ? ARROW_FLAG_NULLABLE : 0;

  NANOARROW_THROW_NOT_OK(
    ArrowSchemaSetType(schema->children[0]->children[0], NANOARROW_TYPE_INT64));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[0]->children[0], "element"));
  schema->children[0]->children[0]->flags = 0;

  return schema;
}

nanoarrow::UniqueArray make_fixed_size_list_chunk(ArrowSchema* schema,
                                                  std::vector<int64_t> const& values,
                                                  int64_t num_rows,
                                                  std::vector<uint8_t> const& validity = {})
{
  nanoarrow::UniqueArray array;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(array.get(), schema, nullptr));
  array->length     = num_rows;
  array->null_count = 0;

  auto* list_array       = array->children[0];
  list_array->length     = num_rows;
  list_array->null_count = 0;
  if (!validity.empty()) {
    ArrowBitmap bitmap;
    ArrowBitmapInit(&bitmap);
    NANOARROW_THROW_NOT_OK(ArrowBitmapReserve(&bitmap, validity.size()));
    ArrowBitmapAppendInt8Unsafe(
      &bitmap, reinterpret_cast<int8_t const*>(validity.data()), validity.size());
    ArrowArraySetValidityBitmap(list_array, &bitmap);
    list_array->null_count =
      num_rows -
      ArrowBitCountSet(ArrowArrayValidityBitmap(list_array)->buffer.data, 0, validity.size());
  }

  auto* values_array = list_array->children[0];
  NANOARROW_THROW_NOT_OK(ArrowBufferAppend(ArrowArrayBuffer(values_array, 1),
                                           reinterpret_cast<void const*>(values.data()),
                                           values.size() * sizeof(int64_t)));
  values_array->length     = values.size();
  values_array->null_count = 0;

  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(array.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr));
  return array;
}

}  // namespace

// exercises make_empty_column_from_schema, which builds the column from the schema alone
TEST_F(FromArrowStreamTest, FixedSizeListEmptyTest)
{
  auto schema = make_fixed_size_list_stream_schema(3);

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
  auto schema             = make_fixed_size_list_stream_schema(width);

  std::vector<nanoarrow::UniqueArray> arrays;
  for (auto i = 0; i < 3; ++i) {
    auto base = static_cast<int64_t>(i * 4);
    arrays.push_back(make_fixed_size_list_chunk(
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
  auto schema                        = make_fixed_size_list_stream_schema(2, /*nullable=*/true);

  std::vector<nanoarrow::UniqueArray> arrays;
  arrays.push_back(make_fixed_size_list_chunk(schema.get(), {1, 2, 3, 4}, 2, /*validity=*/{1, 0}));
  arrays.push_back(make_fixed_size_list_chunk(schema.get(), {5, 6, 7, 8}, 2, /*validity=*/{0, 1}));

  auto child   = cudf::test::fixed_width_column_wrapper<int64_t>{1, 2, 3, 4, 5, 6, 7, 8}.release();
  auto offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2, 4, 6, 8}.release();
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
  constexpr cudf::size_type width = 2;
  auto schema                     = make_fixed_size_list_stream_schema(width);

  std::vector<nanoarrow::UniqueArray> arrays;
  arrays.push_back(
    make_fixed_size_list_chunk(schema.get(), {1, 2, 3, 4, 5, 6, 7, 8}, /*num_rows=*/4));
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
  constexpr cudf::size_type width = 2;
  auto schema                     = make_fixed_size_list_stream_schema(width);

  std::vector<nanoarrow::UniqueArray> arrays;
  std::vector<int64_t> expected_values;
  std::vector<cudf::size_type> expected_offsets{0};
  for (auto const num_rows : {cudf::size_type{1024}, cudf::size_type{1025}}) {
    std::vector<int64_t> values(num_rows * width);
    std::iota(values.begin(), values.end(), static_cast<int64_t>(expected_values.size()));
    expected_values.insert(expected_values.end(), values.begin(), values.end());
    arrays.push_back(make_fixed_size_list_chunk(schema.get(), values, num_rows));
    for (cudf::size_type i = 0; i < num_rows; ++i) {
      expected_offsets.push_back(expected_offsets.back() + width);
    }
  }

  auto expected_offsets_col = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
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
