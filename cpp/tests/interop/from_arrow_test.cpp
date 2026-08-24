/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "arrow_utils.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/nanoarrow_utils.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cuda/iterator>

#include <arrow/c/bridge.h>

std::unique_ptr<cudf::table> get_cudf_table()
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.emplace_back(cudf::test::fixed_width_column_wrapper<int32_t>(
                         {1, 2, 5, 2, 7}, {true, false, true, true, true})
                         .release());
  columns.emplace_back(cudf::test::fixed_width_column_wrapper<int64_t>({1, 2, 3, 4, 5}).release());
  columns.emplace_back(cudf::test::strings_column_wrapper({"fff", "aaa", "", "fff", "ccc"},
                                                          {true, true, true, false, true})
                         .release());

  auto keys    = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 5, 7});
  auto indices = cudf::test::fixed_width_column_wrapper<int32_t>({0, 1, 2, 1, 3}, {1, 0, 1, 1, 1});
  columns.emplace_back(cudf::make_dictionary_column(keys, indices));

  columns.emplace_back(cudf::test::fixed_width_column_wrapper<bool>(
                         {true, false, true, false, true}, {true, false, true, true, false})
                         .release());
  columns.emplace_back(cudf::test::strings_column_wrapper(
                         {
                           "",
                           "abc",
                           "def",
                           "1",
                           "2",
                         },
                         {0, 1, 1, 1, 1})
                         .release());
  // columns.emplace_back(cudf::test::lists_column_wrapper<int>({{1, 2}, {3, 4}, {}, {6}, {7, 8,
  // 9}}).release());
  return std::make_unique<cudf::table>(std::move(columns));
}

std::shared_ptr<arrow::LargeStringArray> get_arrow_large_string_array(
  std::vector<std::string> const& data, std::vector<uint8_t> const& mask = {})
{
  std::shared_ptr<arrow::LargeStringArray> large_string_array;
  arrow::LargeStringBuilder large_string_builder;

  auto const append_status = large_string_builder.AppendValues(data, mask.data());
  CUDF_EXPECTS(append_status.ok(), "Failed to append values to string builder");
  auto const finish_status = large_string_builder.Finish(&large_string_array);
  CUDF_EXPECTS(finish_status.ok(), "Failed to create arrow string array");

  return large_string_array;
}

struct FromArrowTest : public cudf::test::BaseFixture {};

template <typename T>
struct FromArrowTestDurationsTest : public cudf::test::BaseFixture {};

template <typename T>
struct FromArrowTestDecimalsTest : public cudf::test::BaseFixture {};

std::optional<std::unique_ptr<cudf::table>> export_table(std::shared_ptr<arrow::Table> arrow_table)
{
  ArrowSchema schema;
  if (!arrow::ExportSchema(*arrow_table->schema(), &schema).ok()) { return std::nullopt; }
  auto batch = arrow_table->CombineChunksToBatch().ValueOrDie();
  ArrowArray arr;
  if (!arrow::ExportRecordBatch(*batch, &arr).ok()) { return std::nullopt; }
  auto ret = cudf::from_arrow(&schema, &arr);
  arr.release(&arr);
  schema.release(&schema);
  return {std::move(ret)};
}

std::optional<std::unique_ptr<cudf::scalar>> export_scalar(arrow::Scalar const& arrow_scalar)
{
  auto maybe_array = arrow::MakeArrayFromScalar(arrow_scalar, 1);
  if (!maybe_array.ok()) { return std::nullopt; }
  auto array = *maybe_array;

  ArrowSchema schema;
  if (!arrow::ExportType(*array->type(), &schema).ok()) { return std::nullopt; }

  ArrowArray arr;
  if (!arrow::ExportArray(*array, &arr).ok()) { return std::nullopt; }

  auto col = cudf::from_arrow_column(&schema, &arr);
  auto ret = cudf::get_element(col->view(), 0);

  arr.release(&arr);
  schema.release(&schema);
  return {std::move(ret)};
}

std::optional<std::unique_ptr<cudf::scalar>> export_scalar(
  std::shared_ptr<arrow::Scalar> const arrow_scalar)
{
  return export_scalar(*arrow_scalar);
}

TYPED_TEST_SUITE(FromArrowTestDurationsTest, cudf::test::DurationTypes);
using FixedPointTypes = cudf::test::Types<int32_t, int64_t, __int128_t>;
TYPED_TEST_SUITE(FromArrowTestDecimalsTest, FixedPointTypes);

TEST_F(FromArrowTest, EmptyTable)
{
  auto tables = get_tables(0);

  auto expected_cudf_table = tables.first->view();
  auto arrow_table         = tables.second;

  auto got_cudf_table = export_table(arrow_table);
  ASSERT_TRUE(got_cudf_table.has_value());

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_cudf_table, got_cudf_table.value()->view());
}

TEST_F(FromArrowTest, DateTimeTable)
{
  auto data = std::vector<int64_t>{1, 2, 3, 4, 5, 6};

  auto col = cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>(
    data.begin(), data.end(), cuda::constant_iterator<bool>(true));

  cudf::table_view expected_table_view({col});

  std::shared_ptr<arrow::Array> arr;
  arrow::TimestampBuilder timestamp_builder(arrow::timestamp(arrow::TimeUnit::type::MILLI),
                                            arrow::default_memory_pool());
  ASSERT_TRUE(timestamp_builder.AppendValues(data).ok());
  ASSERT_TRUE(timestamp_builder.Finish(&arr).ok());

  std::vector<std::shared_ptr<arrow::Field>> schema_vector({arrow::field("a", arr->type())});
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  auto arrow_table = arrow::Table::Make(schema, {arr});

  auto got_cudf_table = export_table(arrow_table);
  ASSERT_TRUE(got_cudf_table.has_value());

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table_view, got_cudf_table.value()->view());
}

TYPED_TEST(FromArrowTestDurationsTest, DurationTable)
{
  using T = TypeParam;

  auto data = {T{1}, T{2}, T{3}, T{4}, T{5}, T{6}};
  auto col  = cudf::test::fixed_width_column_wrapper<T>(data, cuda::constant_iterator<bool>(true));

  std::shared_ptr<arrow::Array> arr;
  cudf::table_view expected_table_view({col});
  arrow::TimeUnit::type arrow_unit;

  switch (cudf::type_to_id<TypeParam>()) {
    case cudf::type_id::DURATION_SECONDS: arrow_unit = arrow::TimeUnit::type::SECOND; break;
    case cudf::type_id::DURATION_MILLISECONDS: arrow_unit = arrow::TimeUnit::type::MILLI; break;
    case cudf::type_id::DURATION_MICROSECONDS: arrow_unit = arrow::TimeUnit::type::MICRO; break;
    case cudf::type_id::DURATION_NANOSECONDS: arrow_unit = arrow::TimeUnit::type::NANO; break;
    case cudf::type_id::DURATION_DAYS: return;
    default: CUDF_FAIL("Unsupported duration unit in arrow");
  }
  arrow::DurationBuilder duration_builder(duration(arrow_unit), arrow::default_memory_pool());
  ASSERT_TRUE(duration_builder.AppendValues(std::vector<int64_t>{1, 2, 3, 4, 5, 6}).ok());
  ASSERT_TRUE(duration_builder.Finish(&arr).ok());

  std::vector<std::shared_ptr<arrow::Field>> schema_vector({arrow::field("a", arr->type())});
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  auto arrow_table = arrow::Table::Make(schema, {arr});

  auto got_cudf_table = export_table(arrow_table);
  ASSERT_TRUE(got_cudf_table.has_value());

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table_view, got_cudf_table.value()->view());
}

TEST_F(FromArrowTest, NestedList)
{
  auto valids = cudf::test::iterators::nulls_at_multiples_of(3);
  auto col    = cudf::test::lists_column_wrapper<int64_t>(
    {{{{{1, 2}, valids}, {{3, 4}, valids}, {5}}, {{6}, {{7, 8, 9}, valids}}}, valids});
  cudf::table_view expected_table_view({col});

  auto list_arr = get_arrow_list_array<int64_t>({6, 7, 8, 9}, {0, 1, 4}, {1, 0, 1, 1});
  std::vector<int32_t> offset{0, 0, 2};
  auto mask_buffer     = arrow::internal::BytesToBits(std::vector<uint8_t>({0, 1})).ValueOrDie();
  auto nested_list_arr = std::make_shared<arrow::ListArray>(arrow::list(list(arrow::int64())),
                                                            offset.size() - 1,
                                                            arrow::Buffer::Wrap(offset),
                                                            list_arr,
                                                            mask_buffer);

  std::vector<std::shared_ptr<arrow::Field>> schema_vector(
    {arrow::field("a", nested_list_arr->type())});
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  auto arrow_table = arrow::Table::Make(schema, {nested_list_arr});

  auto got_cudf_table = export_table(arrow_table);
  ASSERT_TRUE(got_cudf_table.has_value());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_table_view, got_cudf_table.value()->view());
}

TEST_F(FromArrowTest, StructColumn)
{
  using vector_of_columns = std::vector<std::unique_ptr<cudf::column>>;

  // Create cudf table
  auto nested_type_field_names =
    std::vector<std::vector<std::string>>{{"string", "integral", "bool", "nested_list", "struct"}};
  auto str_col =
    cudf::test::strings_column_wrapper{
      "Samuel Vimes", "Carrot Ironfoundersson", "Angua von Überwald"}
      .release();
  auto str_col2 =
    cudf::test::strings_column_wrapper{{"", "ROCKS", ""}, {false, true, false}}.release();
  int num_rows{str_col->size()};
  auto int_col = cudf::test::fixed_width_column_wrapper<int32_t, int32_t>{{48, 27, 25}}.release();
  auto int_col2 =
    cudf::test::fixed_width_column_wrapper<int32_t, int32_t>{{12, 24, 47}, {true, false, true}}
      .release();
  auto bool_col = cudf::test::fixed_width_column_wrapper<bool>{{true, true, false}}.release();
  auto list_col = cudf::test::lists_column_wrapper<int64_t>(
                    {{{1, 2}, {3, 4}, {5}}, {{{6}}}, {{7}, {8, 9}}})  // NOLINT
                    .release();
  vector_of_columns cols2;
  cols2.push_back(std::move(str_col2));
  cols2.push_back(std::move(int_col2));
  auto [null_mask, null_count] =
    cudf::bools_to_mask(cudf::test::fixed_width_column_wrapper<bool>{{true, true, false}});
  auto sub_struct_col =
    cudf::make_structs_column(num_rows, std::move(cols2), null_count, std::move(*null_mask));
  vector_of_columns cols;
  cols.push_back(std::move(str_col));
  cols.push_back(std::move(int_col));
  cols.push_back(std::move(bool_col));
  cols.push_back(std::move(list_col));
  cols.push_back(std::move(sub_struct_col));

  auto struct_col = cudf::make_structs_column(num_rows, std::move(cols), 0, {});
  cudf::table_view expected_cudf_table({struct_col->view()});

  // Create Arrow table
  std::vector<std::string> str{"Samuel Vimes", "Carrot Ironfoundersson", "Angua von Überwald"};
  std::vector<std::string> str2{"CUDF", "ROCKS", "EVERYWHERE"};
  auto str_array  = get_arrow_array<cudf::string_view>(str);
  auto int_array  = get_arrow_array<int32_t>({48, 27, 25});
  auto str2_array = get_arrow_array<cudf::string_view>(str2, {0, 1, 0});
  auto int2_array = get_arrow_array<int32_t>({12, 24, 47}, {1, 0, 1});
  auto bool_array = get_arrow_array<bool>({true, true, false});
  auto list_arr = get_arrow_list_array<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 2, 4, 5, 6, 7, 9});
  std::vector<int32_t> offset{0, 3, 4, 6};
  auto nested_list_arr = std::make_shared<arrow::ListArray>(
    arrow::list(list(arrow::field("element", arrow::int64(), false))),
    offset.size() - 1,
    arrow::Buffer::Wrap(offset),
    list_arr);

  std::vector<std::shared_ptr<arrow::Array>> child_arrays2({str2_array, int2_array});
  auto fields2 = std::vector<std::shared_ptr<arrow::Field>>{
    std::make_shared<arrow::Field>("string2", str2_array->type(), str2_array->null_count() > 0),
    std::make_shared<arrow::Field>("integral2", int2_array->type(), int2_array->null_count() > 0)};
  std::shared_ptr<arrow::Buffer> mask_buffer =
    arrow::internal::BytesToBits(std::vector<uint8_t>({1, 1, 0})).ValueOrDie();
  auto dtype2        = std::make_shared<arrow::StructType>(fields2);
  auto struct_array2 = std::make_shared<arrow::StructArray>(
    dtype2, static_cast<int64_t>(expected_cudf_table.num_rows()), child_arrays2, mask_buffer);

  std::vector<std::shared_ptr<arrow::Array>> child_arrays(
    {str_array, int_array, bool_array, nested_list_arr, struct_array2});
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::transform(child_arrays.cbegin(),
                 child_arrays.cend(),
                 nested_type_field_names[0].cbegin(),
                 std::back_inserter(fields),
                 [](auto const array, auto const name) {
                   return std::make_shared<arrow::Field>(
                     name, array->type(), array->null_count() > 0);
                 });
  auto dtype = std::make_shared<arrow::StructType>(fields);

  auto struct_array = std::make_shared<arrow::StructArray>(
    dtype, static_cast<int64_t>(expected_cudf_table.num_rows()), child_arrays);
  std::vector<std::shared_ptr<arrow::Field>> schema_vector(
    {arrow::field("a", struct_array->type())});
  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  auto input  = arrow::Table::Make(schema, {struct_array});

  auto got_cudf_table = export_table(input);
  ASSERT_TRUE(got_cudf_table.has_value());

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_cudf_table, got_cudf_table.value()->view());
}

TEST_F(FromArrowTest, DictionaryIndicesType)
{
  auto keys     = std::initializer_list<int64_t>{1, 2, 5, 7};
  auto indices1 = std::initializer_list<int8_t>{0, 1, 2, 1, 3};
  auto indices2 = std::initializer_list<int16_t>{0, 1, 2, 1, 3};
  auto indices3 = std::initializer_list<int64_t>{0, 1, 2, 1, 3};
  auto valids   = std::initializer_list<uint8_t>{1, 0, 1, 1, 1};
  auto bvalids  = std::initializer_list<bool>{true, false, true, true, true};

  auto array1 = get_arrow_dict_array<int64_t, int8_t>(keys, indices1, valids);
  auto array2 = get_arrow_dict_array<int64_t, int16_t>(keys, indices2, valids);
  auto array3 = get_arrow_dict_array<int64_t, int64_t>(keys, indices3, valids);

  std::vector<std::shared_ptr<arrow::Field>> schema_vector({arrow::field("a", array1->type()),
                                                            arrow::field("b", array2->type()),
                                                            arrow::field("c", array3->type())});
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  auto arrow_table = arrow::Table::Make(schema, {array1, array2, array3});

  std::vector<std::unique_ptr<cudf::column>> columns;
  auto keys_cw     = cudf::test::fixed_width_column_wrapper<int64_t>(keys);
  auto indices1_cw = cudf::test::fixed_width_column_wrapper<int8_t>(indices1, bvalids);
  auto indices2_cw = cudf::test::fixed_width_column_wrapper<int16_t>(indices2, bvalids);
  auto indices3_cw = cudf::test::fixed_width_column_wrapper<int64_t>(indices3, bvalids);
  columns.emplace_back(cudf::make_dictionary_column(keys_cw, indices1_cw));
  columns.emplace_back(cudf::make_dictionary_column(keys_cw, indices2_cw));
  columns.emplace_back(cudf::make_dictionary_column(keys_cw, indices3_cw));

  cudf::table expected_table(std::move(columns));

  auto got_cudf_table = export_table(arrow_table);
  ASSERT_TRUE(got_cudf_table.has_value());

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table.view(), got_cudf_table.value()->view());
}

TEST_F(FromArrowTest, ChunkedArray)
{
  auto int64array     = get_arrow_array<int64_t>({1, 2, 3, 4, 5});
  auto int32array_1   = get_arrow_array<int32_t>({1, 2}, {1, 0});
  auto int32array_2   = get_arrow_array<int32_t>({5, 2, 7}, {1, 1, 1});
  auto string_array_1 = get_arrow_array<cudf::string_view>({
    "fff",
    "aaa",
    "",
  });
  auto string_array_2 = get_arrow_array<cudf::string_view>(
    {
      "fff",
      "ccc",
    },
    {0, 1});
  auto large_string_array_1 = get_arrow_large_string_array(
    {
      "",
      "abc",
      "def",
      "1",
      "2",
    },
    {0, 1, 1, 1, 1});
  auto dict_array1 = get_arrow_dict_array({1, 2, 5, 7}, {0, 1, 2}, {1, 0, 1});
  auto dict_array2 = get_arrow_dict_array({1, 2, 5, 7}, {1, 3});

  auto int64_chunked_array = std::make_shared<arrow::ChunkedArray>(int64array);
  auto int32_chunked_array = std::make_shared<arrow::ChunkedArray>(
    std::vector<std::shared_ptr<arrow::Array>>{int32array_1, int32array_2});
  auto string_chunked_array = std::make_shared<arrow::ChunkedArray>(
    std::vector<std::shared_ptr<arrow::Array>>{string_array_1, string_array_2});
  auto dict_chunked_array = std::make_shared<arrow::ChunkedArray>(
    std::vector<std::shared_ptr<arrow::Array>>{dict_array1, dict_array2});
  auto boolean_array =
    get_arrow_array<bool>({true, false, true, false, true}, {true, false, true, true, false});
  auto boolean_chunked_array      = std::make_shared<arrow::ChunkedArray>(boolean_array);
  auto large_string_chunked_array = std::make_shared<arrow::ChunkedArray>(
    std::vector<std::shared_ptr<arrow::Array>>{large_string_array_1});

  std::vector<std::shared_ptr<arrow::Field>> schema_vector(
    {arrow::field("a", int32_chunked_array->type()),
     arrow::field("b", int64array->type()),
     arrow::field("c", string_array_1->type()),
     arrow::field("d", dict_chunked_array->type()),
     arrow::field("e", boolean_chunked_array->type()),
     arrow::field("f", large_string_array_1->type())});
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  auto arrow_table = arrow::Table::Make(schema,
                                        {int32_chunked_array,
                                         int64_chunked_array,
                                         string_chunked_array,
                                         dict_chunked_array,
                                         boolean_chunked_array,
                                         large_string_chunked_array});

  auto expected_cudf_table = get_cudf_table();

  auto got_cudf_table = export_table(arrow_table);
  ASSERT_TRUE(got_cudf_table.has_value());

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_cudf_table->view(), got_cudf_table.value()->view());
}

struct FromArrowTestSlice
  : public FromArrowTest,
    public ::testing::WithParamInterface<std::tuple<cudf::size_type, cudf::size_type>> {};

TEST_P(FromArrowTestSlice, SliceTest)
{
  auto tables             = get_tables(10000);
  auto cudf_table_view    = tables.first->view();
  auto arrow_table        = tables.second;
  auto const [start, end] = GetParam();

  auto sliced_cudf_table   = cudf::slice(cudf_table_view, {start, end})[0];
  auto expected_cudf_table = cudf::table{sliced_cudf_table};
  auto sliced_arrow_table  = arrow_table->Slice(start, end - start);
  auto got_cudf_table      = export_table(sliced_arrow_table);
  ASSERT_TRUE(got_cudf_table.has_value());

  // This has been added to take-care of empty string column issue with no children
  if (got_cudf_table.value()->num_rows() == 0 and expected_cudf_table.num_rows() == 0) {
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_cudf_table.view(), got_cudf_table.value()->view());
  } else {
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_cudf_table.view(), got_cudf_table.value()->view());
  }
}

template <typename T>
using fp_wrapper = cudf::test::fixed_point_column_wrapper<T>;

TYPED_TEST(FromArrowTestDecimalsTest, FixedPointTable)
{
  using T = TypeParam;
  using namespace numeric;

  auto const precision = get_decimal_precision<T>();
  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const data = std::vector<T>{1, 2, 3, 4, 5, 6};
    auto const col  = fp_wrapper<T>(
      data.cbegin(), data.cend(), cuda::constant_iterator<bool>(true), scale_type{scale});
    auto const expected = cudf::table_view({col});

    auto const arr = get_decimal_arrow_array(data, std::nullopt, precision, scale);

    auto const field         = arrow::field("a", arr->type());
    auto const schema_vector = std::vector<std::shared_ptr<arrow::Field>>({field});
    auto const schema        = std::make_shared<arrow::Schema>(schema_vector);
    auto const arrow_table   = arrow::Table::Make(schema, {arr});

    auto got_cudf_table = export_table(arrow_table);
    ASSERT_TRUE(got_cudf_table.has_value());

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got_cudf_table.value()->view());
  }
}

TYPED_TEST(FromArrowTestDecimalsTest, FixedPointTableLarge)
{
  using T = TypeParam;
  using namespace numeric;

  auto const precision        = get_decimal_precision<T>();
  auto constexpr NUM_ELEMENTS = 1000;

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto data = std::vector<T>(NUM_ELEMENTS);
    std::iota(data.begin(), data.end(), T{1});
    auto const col = fp_wrapper<T>(
      data.begin(), data.end(), cuda::constant_iterator<bool>(true), scale_type{scale});
    auto const expected = cudf::table_view({col});

    auto const arr = get_decimal_arrow_array(data, std::nullopt, precision, scale);

    auto const field         = arrow::field("a", arr->type());
    auto const schema_vector = std::vector<std::shared_ptr<arrow::Field>>({field});
    auto const schema        = std::make_shared<arrow::Schema>(schema_vector);
    auto const arrow_table   = arrow::Table::Make(schema, {arr});

    auto got_cudf_table = export_table(arrow_table);
    ASSERT_TRUE(got_cudf_table.has_value());

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got_cudf_table.value()->view());
  }
}

TYPED_TEST(FromArrowTestDecimalsTest, FixedPointTableNulls)
{
  using T = TypeParam;
  using namespace numeric;

  auto const precision = get_decimal_precision<T>();
  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const data     = std::vector<T>{1, 2, 3, 4, 5, 6, 0, 0};
    auto const validity = std::vector<uint8_t>{1, 1, 1, 1, 1, 1, 0, 0};
    auto const col      = fp_wrapper<T>({1, 2, 3, 4, 5, 6, 0, 0},
                                        {true, true, true, true, true, true, false, false},
                                   scale_type{scale});
    auto const expected = cudf::table_view({col});

    auto const arr = get_decimal_arrow_array(data, validity, precision, scale);

    auto const field         = arrow::field("a", arr->type());
    auto const schema_vector = std::vector<std::shared_ptr<arrow::Field>>({field});
    auto const schema        = std::make_shared<arrow::Schema>(schema_vector);
    auto const arrow_table   = arrow::Table::Make(schema, {arr});

    auto got_cudf_table = export_table(arrow_table);
    ASSERT_TRUE(got_cudf_table.has_value());

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got_cudf_table.value()->view());
  }
}

TYPED_TEST(FromArrowTestDecimalsTest, FixedPointTableNullsLarge)
{
  using T = TypeParam;
  using namespace numeric;

  auto const precision        = get_decimal_precision<T>();
  auto constexpr NUM_ELEMENTS = 1000;

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto every_other = [](auto i) { return i % 2 ? 0 : 1; };
    auto validity    = cudf::detail::make_counting_transform_iterator(0, every_other);
    auto iota       = cudf::detail::make_counting_transform_iterator(1, [](int i) { return T{i}; });
    auto const data = std::vector<T>(iota, iota + NUM_ELEMENTS);
    auto const col  = fp_wrapper<T>(iota, iota + NUM_ELEMENTS, validity, scale_type{scale});
    auto const expected = cudf::table_view({col});

    auto const arr = get_decimal_arrow_array(
      data, std::vector<uint8_t>(validity, validity + NUM_ELEMENTS), precision, scale);

    auto const field         = arrow::field("a", arr->type());
    auto const schema_vector = std::vector<std::shared_ptr<arrow::Field>>({field});
    auto const schema        = std::make_shared<arrow::Schema>(schema_vector);
    auto const arrow_table   = arrow::Table::Make(schema, {arr});

    auto got_cudf_table = export_table(arrow_table);
    ASSERT_TRUE(got_cudf_table.has_value());

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got_cudf_table.value()->view());
  }
}

INSTANTIATE_TEST_CASE_P(FromArrowTest,
                        FromArrowTestSlice,
                        ::testing::Values(std::make_tuple(0, 10000),
                                          std::make_tuple(2912, 2915),
                                          std::make_tuple(100, 3000),
                                          std::make_tuple(0, 0),
                                          std::make_tuple(0, 3000),
                                          std::make_tuple(10000, 10000)));

template <typename T>
struct FromArrowNumericScalarTest : public cudf::test::BaseFixture {};

using NumericTypesNotBool =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;
TYPED_TEST_SUITE(FromArrowNumericScalarTest, NumericTypesNotBool);

TYPED_TEST(FromArrowNumericScalarTest, Basic)
{
  TypeParam const value{42};
  auto const arrow_scalar = arrow::MakeScalar(value);

  auto const cudf_scalar = export_scalar(arrow_scalar);
  ASSERT_TRUE(cudf_scalar.has_value());

  auto const cudf_numeric_scalar =
    dynamic_cast<cudf::numeric_scalar<TypeParam>*>(cudf_scalar.value().get());
  CUDF_EXPECTS(cudf_numeric_scalar != nullptr, "Attempted to test with a non-numeric type.");
  EXPECT_EQ(cudf_numeric_scalar->type(), cudf::data_type(cudf::type_to_id<TypeParam>()));
  EXPECT_EQ(cudf_numeric_scalar->value(), value);
}

struct FromArrowDecimalScalarTest : public cudf::test::BaseFixture {};

template <typename ScalarType, typename DecimalType>
void check_decimal_scalar(int const value, ScalarType const& arrow_scalar)
{
  auto const scale{4};
  auto const cudf_scalar = export_scalar(arrow_scalar);
  ASSERT_TRUE(cudf_scalar.has_value());

  auto const cudf_decimal_scalar =
    dynamic_cast<cudf::fixed_point_scalar<DecimalType>*>(cudf_scalar.value().get());
  EXPECT_EQ(cudf_decimal_scalar->type(), cudf::data_type(cudf::type_to_id<DecimalType>(), scale));
  EXPECT_EQ(cudf_decimal_scalar->value(), value);
}

TEST_F(FromArrowDecimalScalarTest, Basic)
{
  auto const value{42};
  auto const precision{8};
  auto const scale{4};
  auto arrow_scalar32  = arrow::Decimal32Scalar(value, arrow::decimal32(precision, -scale));
  auto arrow_scalar64  = arrow::Decimal64Scalar(value, arrow::decimal64(precision, -scale));
  auto arrow_scalar128 = arrow::Decimal128Scalar(value, arrow::decimal128(precision, -scale));

  check_decimal_scalar<arrow::Decimal32Scalar, numeric::decimal32>(value, arrow_scalar32);
  check_decimal_scalar<arrow::Decimal64Scalar, numeric::decimal64>(value, arrow_scalar64);
  check_decimal_scalar<arrow::Decimal128Scalar, numeric::decimal128>(value, arrow_scalar128);
}

struct FromArrowStringScalarTest : public cudf::test::BaseFixture {};

TEST_F(FromArrowStringScalarTest, Basic)
{
  auto const value        = std::string("hello world");
  auto const arrow_scalar = arrow::StringScalar(value);
  auto const cudf_scalar  = export_scalar(arrow_scalar);
  ASSERT_TRUE(cudf_scalar.has_value());

  auto const cudf_string_scalar = dynamic_cast<cudf::string_scalar*>(cudf_scalar.value().get());
  EXPECT_EQ(cudf_string_scalar->type(), cudf::data_type(cudf::type_id::STRING));
  EXPECT_EQ(cudf_string_scalar->to_string(), value);
}

struct FromArrowListScalarTest : public cudf::test::BaseFixture {};

TEST_F(FromArrowListScalarTest, Basic)
{
  std::vector<int64_t> host_values = {1, 2, 3, 5, 6, 7, 8};
  std::vector<bool> host_validity  = {true, true, true, false, true, true, true};

  arrow::Int64Builder builder;
  auto const status      = builder.AppendValues(host_values, host_validity);
  auto const maybe_array = builder.Finish();
  auto const array       = *maybe_array;

  auto const arrow_scalar = arrow::ListScalar(array);
  auto const cudf_scalar  = export_scalar(arrow_scalar);
  ASSERT_TRUE(cudf_scalar.has_value());

  auto const cudf_list_scalar = dynamic_cast<cudf::list_scalar*>(cudf_scalar.value().get());
  EXPECT_EQ(cudf_list_scalar->type(), cudf::data_type(cudf::type_id::LIST));

  cudf::test::fixed_width_column_wrapper<int64_t> const lhs(
    host_values.begin(), host_values.end(), host_validity.begin());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(lhs, cudf_list_scalar->view());
}

struct FromArrowStructScalarTest : public cudf::test::BaseFixture {};

TEST_F(FromArrowStructScalarTest, Basic)
{
  int64_t const value{42};
  auto const underlying_arrow_scalar = arrow::MakeScalar(value);

  auto const field        = arrow::field("", underlying_arrow_scalar->type);
  auto const arrow_type   = arrow::struct_({field});
  auto const arrow_scalar = arrow::StructScalar({underlying_arrow_scalar}, arrow_type);
  auto const cudf_scalar  = export_scalar(arrow_scalar);
  ASSERT_TRUE(cudf_scalar.has_value());

  auto const cudf_struct_scalar = dynamic_cast<cudf::struct_scalar*>(cudf_scalar.value().get());
  EXPECT_EQ(cudf_struct_scalar->type(), cudf::data_type(cudf::type_id::STRUCT));

  cudf::test::fixed_width_column_wrapper<int64_t> const col({value}, {true});
  cudf::table_view const lhs({col});

  CUDF_TEST_EXPECT_TABLES_EQUAL(lhs, cudf_struct_scalar->view());
}
