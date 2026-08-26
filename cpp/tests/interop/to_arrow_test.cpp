/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/interop/arrow_utils.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/interop.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cuda/iterator>

#include <arrow/c/bridge.h>

using vector_of_columns = std::vector<std::unique_ptr<cudf::column>>;

std::pair<std::unique_ptr<cudf::table>, std::shared_ptr<arrow::Table>> get_tables(
  cudf::size_type length)
{
  std::vector<int64_t> int64_data(length);
  std::vector<bool> bool_data(length);
  std::vector<std::string> string_data(length);
  std::vector<uint8_t> validity(length);
  std::vector<bool> bool_validity(length);
  std::vector<uint8_t> bool_data_validity;

  std::vector<std::unique_ptr<cudf::column>> columns;

  auto validity_generator = []() { return rand() % 7 != 0; };

  std::generate(int64_data.begin(), int64_data.end(), []() { return rand() % 500000; });
  std::generate(bool_data.begin(), bool_data.end(), validity_generator);
  std::generate(
    string_data.begin(), string_data.end(), []() { return rand() % 7 != 0 ? "CUDF" : "Rocks"; });
  std::generate(validity.begin(), validity.end(), validity_generator);
  std::generate(bool_validity.begin(), bool_validity.end(), validity_generator);

  std::transform(bool_validity.cbegin(),
                 bool_validity.cend(),
                 std::back_inserter(bool_data_validity),
                 [](auto val) { return static_cast<uint8_t>(val); });

  std::vector<uint8_t> list_validity = bool_data_validity;
  std::vector<int32_t> list_offsets(length + 1);
  list_offsets[0] = 0;

  constexpr cudf::size_type length_of_individual_list = 3;
  for (cudf::size_type i = 0; i < length; ++i) {
    list_offsets[i + 1] = list_offsets[i] + (list_validity[i] ? length_of_individual_list : 0);
  }
  std::vector<uint8_t> list_int64_data_validity(list_offsets.back());
  std::vector<int64_t> list_int64_data(list_offsets.back());
  std::generate(list_int64_data.begin(), list_int64_data.end(), []() { return rand() % 500000; });
  std::generate(
    list_int64_data_validity.begin(), list_int64_data_validity.end(), validity_generator);

  columns.emplace_back(cudf::test::fixed_width_column_wrapper<int64_t>(
                         int64_data.begin(), int64_data.end(), validity.begin())
                         .release());
  columns.emplace_back(
    cudf::test::strings_column_wrapper(string_data.begin(), string_data.end(), validity.begin())
      .release());
  auto dict_column = cudf::test::dictionary_column_wrapper<int64_t>(
    int64_data.begin(), int64_data.end(), validity.begin());
  // make a copy of dict_column since it is used again below
  columns.emplace_back(std::make_unique<cudf::column>(dict_column));
  columns.emplace_back(cudf::test::fixed_width_column_wrapper<bool>(
                         bool_data.begin(), bool_data.end(), bool_validity.begin())
                         .release());
  auto list_child_column = cudf::test::fixed_width_column_wrapper<int64_t>(
    list_int64_data.begin(), list_int64_data.end(), list_int64_data_validity.begin());
  auto list_offsets_column =
    cudf::test::fixed_width_column_wrapper<int32_t>(list_offsets.begin(), list_offsets.end());
  auto [list_mask, list_nulls] = cudf::bools_to_mask(
    cudf::test::fixed_width_column_wrapper<bool>(list_validity.begin(), list_validity.end()));
  columns.emplace_back(cudf::make_lists_column(length,
                                               list_offsets_column.release(),
                                               list_child_column.release(),
                                               list_nulls,
                                               std::move(*list_mask)));
  auto int_column = cudf::test::fixed_width_column_wrapper<int64_t>(
                      int64_data.begin(), int64_data.end(), validity.begin())
                      .release();
  auto str_column =
    cudf::test::strings_column_wrapper(string_data.begin(), string_data.end(), validity.begin())
      .release();
  vector_of_columns cols;
  cols.push_back(std::move(int_column));
  cols.push_back(std::move(str_column));
  auto [null_mask, null_count] = cudf::bools_to_mask(cudf::test::fixed_width_column_wrapper<bool>(
    bool_data_validity.begin(), bool_data_validity.end()));
  columns.emplace_back(
    cudf::make_structs_column(length, std::move(cols), null_count, std::move(*null_mask)));

  auto int64array = get_arrow_array<int64_t>(int64_data, validity);
  auto boolarray  = get_arrow_array<bool>(bool_data, bool_validity);
  auto list_array = get_arrow_list_array<int64_t>(
    list_int64_data, list_offsets, list_int64_data_validity, list_validity);

  cudf::dictionary_column_view view(dict_column);
  auto keys       = cudf::test::to_host<int64_t>(view.keys()).first;
  auto indices    = cudf::test::to_host<int32_t>(view.indices()).first;
  auto dict_array = get_arrow_dict_array(std::vector<int64_t>(keys.begin(), keys.end()),
                                         std::vector<int32_t>(indices.begin(), indices.end()),
                                         validity);

  auto string_array = get_arrow_array<cudf::string_view>(string_data, validity);
  arrow::ArrayVector child_arrays({int64array, string_array});
  std::vector<std::shared_ptr<arrow::Field>> fields = {
    arrow::field("integral", int64array->type(), int64array->null_count() > 0),
    arrow::field("string", string_array->type(), string_array->null_count() > 0)};
  auto dtype = std::make_shared<arrow::StructType>(fields);
  std::shared_ptr<arrow::Buffer> mask_buffer =
    arrow::internal::BytesToBits(static_cast<std::vector<uint8_t>>(bool_data_validity))
      .ValueOrDie();
  auto struct_array =
    std::make_shared<arrow::StructArray>(dtype, length, child_arrays, mask_buffer);

  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
    arrow::field("a", int64array->type()),
    arrow::field("b", string_array->type()),
    arrow::field("c", dict_array->type()),
    arrow::field("d", boolarray->type()),
    arrow::field("e", list_array->type()),
    arrow::field("f", struct_array->type())};

  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  return std::pair(
    std::make_unique<cudf::table>(std::move(columns)),
    arrow::Table::Make(
      schema, {int64array, string_array, dict_array, boolarray, list_array, struct_array}));
}

struct ToArrowTest : public cudf::test::BaseFixture {};

template <typename T>
struct ToArrowTestDurationsTest : public cudf::test::BaseFixture {};

auto is_equal(cudf::table_view const& table,
              cudf::host_span<cudf::column_metadata const> metadata,
              std::shared_ptr<arrow::Table> expected_arrow_table)
{
  auto got_arrow_schema = cudf::to_arrow_schema(table, metadata);
  auto got_arrow_table  = cudf::to_arrow_host(table);

  for (auto i = 0; i < got_arrow_schema->n_children; ++i) {
    auto arr = arrow::ImportArray(got_arrow_table->array.children[i], got_arrow_schema->children[i])
                 .ValueOrDie();
    if (!expected_arrow_table->column(i)->Equals(arrow::ChunkedArray(arr))) { return false; }
  }
  return true;
}

TYPED_TEST_SUITE(ToArrowTestDurationsTest, cudf::test::DurationTypes);

TEST_F(ToArrowTest, EmptyTable)
{
  auto tables = get_tables(0);

  auto cudf_table_view      = tables.first->view();
  auto expected_arrow_table = tables.second;
  auto struct_meta          = cudf::column_metadata{"f"};
  struct_meta.children_meta = {{"integral"}, {"string"}};

  std::vector<cudf::column_metadata> const metadata = {
    {"a"}, {"b"}, {"c"}, {"d"}, {"e"}, struct_meta};
  ASSERT_TRUE(is_equal(cudf_table_view, metadata, expected_arrow_table));
}

TEST_F(ToArrowTest, DateTimeTable)
{
  auto data = {1, 2, 3, 4, 5, 6};

  auto col =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>(data);

  cudf::table_view input_view({col});

  std::shared_ptr<arrow::Array> arr;
  arrow::TimestampBuilder timestamp_builder(timestamp(arrow::TimeUnit::type::MILLI),
                                            arrow::default_memory_pool());
  ASSERT_TRUE(timestamp_builder.AppendValues(std::vector<int64_t>{1, 2, 3, 4, 5, 6}).ok());
  ASSERT_TRUE(timestamp_builder.Finish(&arr).ok());

  std::vector<std::shared_ptr<arrow::Field>> schema_vector({arrow::field("a", arr->type())});
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  auto expected_arrow_table = arrow::Table::Make(schema, {arr});

  std::vector<cudf::column_metadata> const metadata = {{"a"}};
  ASSERT_TRUE(is_equal(input_view, metadata, expected_arrow_table));
}

TYPED_TEST(ToArrowTestDurationsTest, DurationTable)
{
  using T = TypeParam;

  auto data = {T{1}, T{2}, T{3}, T{4}, T{5}, T{6}};
  auto col  = cudf::test::fixed_width_column_wrapper<T>(data);

  cudf::table_view input_view({col});

  std::shared_ptr<arrow::Array> arr;
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

  auto expected_arrow_table = arrow::Table::Make(schema, {arr});

  std::vector<cudf::column_metadata> const metadata = {{"a"}};
  ASSERT_TRUE(is_equal(input_view, metadata, expected_arrow_table));
}

TEST_F(ToArrowTest, NestedList)
{
  auto valids = cudf::test::iterators::nulls_at_multiples_of(3);
  auto col    = cudf::test::lists_column_wrapper<int64_t>(
    {{{{{1, 2}, valids}, {{3, 4}, valids}, {5}}, {{6}, {{7, 8, 9}, valids}}}, valids});
  cudf::table_view input_view({col});

  auto list_arr = get_arrow_list_array<int64_t>({6, 7, 8, 9}, {0, 1, 4}, {1, 0, 1, 1});
  std::vector<int32_t> offset{0, 0, 2};
  auto mask_buffer     = arrow::internal::BytesToBits(std::vector<uint8_t>({0, 1})).ValueOrDie();
  auto nested_list_arr = std::make_shared<arrow::ListArray>(
    arrow::list(arrow::field("element", arrow::list(arrow::int64()), false)),
    offset.size() - 1,
    arrow::Buffer::Wrap(offset),
    list_arr,
    mask_buffer);

  std::vector<std::shared_ptr<arrow::Field>> schema_vector(
    {arrow::field("a", nested_list_arr->type(), false)});
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  auto expected_arrow_table                         = arrow::Table::Make(schema, {nested_list_arr});
  std::vector<cudf::column_metadata> const metadata = {{"a"}};
  ASSERT_TRUE(is_equal(input_view, metadata, expected_arrow_table));
}

TEST_F(ToArrowTest, StructColumn)
{
  // Create cudf table
  auto nested_type_field_names =
    std::vector<std::vector<std::string>>{{"string", "integral", "bool", "nested_list", "struct"}};
  auto str_col =
    cudf::test::strings_column_wrapper{
      "Samuel Vimes", "Carrot Ironfoundersson", "Angua von Überwald"}
      .release();
  auto str_col2 =
    cudf::test::strings_column_wrapper{{"CUDF", "ROCKS", "EVERYWHERE"}, {0, 1, 0}}.release();
  int num_rows{str_col->size()};
  auto int_col = cudf::test::fixed_width_column_wrapper<int32_t, int32_t>{{48, 27, 25}}.release();
  auto int_col2 =
    cudf::test::fixed_width_column_wrapper<int32_t, int32_t>{{12, 24, 47}, {1, 0, 1}}.release();
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
  cudf::table_view input_view({struct_col->view()});

  // Create name metadata
  auto sub_metadata          = cudf::column_metadata{"struct"};
  sub_metadata.children_meta = {{"string2"}, {"integral2"}};
  auto metadata              = cudf::column_metadata{"a"};
  metadata.children_meta     = {{"string"}, {"integral"}, {"bool"}, {"nested_list"}, sub_metadata};

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
    arrow::list(arrow::field("a", arrow::list(arrow::field("a", arrow::int64(), false)), false)),
    offset.size() - 1,
    arrow::Buffer::Wrap(offset),
    list_arr);

  std::vector<std::shared_ptr<arrow::Array>> child_arrays2({str2_array, int2_array});
  auto fields2 = std::vector<std::shared_ptr<arrow::Field>>{
    std::make_shared<arrow::Field>("string2", str2_array->type(), str2_array->null_count() > 0),
    std::make_shared<arrow::Field>("integral2", int2_array->type(), int2_array->null_count() > 0)};
  auto dtype2 = std::make_shared<arrow::StructType>(fields2);
  std::shared_ptr<arrow::Buffer> mask_buffer =
    arrow::internal::BytesToBits(std::vector<uint8_t>({1, 1, 0})).ValueOrDie();
  auto struct_array2 = std::make_shared<arrow::StructArray>(
    dtype2, static_cast<int64_t>(input_view.num_rows()), child_arrays2, mask_buffer);

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
    dtype, static_cast<int64_t>(input_view.num_rows()), child_arrays);
  std::vector<std::shared_ptr<arrow::Field>> schema_vector(
    {arrow::field("a", struct_array->type())});
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  auto expected_arrow_table = arrow::Table::Make(schema, {struct_array});

  std::vector<cudf::column_metadata> const meta = {metadata};
  ASSERT_TRUE(is_equal(input_view, meta, expected_arrow_table));
}

template <typename T>
using fp_wrapper = cudf::test::fixed_point_column_wrapper<T>;

TEST_F(ToArrowTest, FixedPoint64Table)
{
  using namespace numeric;

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const col         = fp_wrapper<int64_t>({-1, 2, 3, 4, 5, 6}, scale_type{scale});
    auto const input       = cudf::table_view({col});
    auto const expect_data = std::vector<int64_t>{-1, 2, 3, 4, 5, 6};

    auto const arr = get_decimal_arrow_array(expect_data, std::nullopt, 18, scale);

    auto const field                = arrow::field("a", arr->type());
    auto const schema_vector        = std::vector<std::shared_ptr<arrow::Field>>({field});
    auto const schema               = std::make_shared<arrow::Schema>(schema_vector);
    auto const expected_arrow_table = arrow::Table::Make(schema, {arr});

    std::vector<cudf::column_metadata> const metadata = {{"a"}};
    ASSERT_TRUE(is_equal(input, metadata, expected_arrow_table));
  }
}

TEST_F(ToArrowTest, FixedPoint128Table)
{
  using namespace numeric;

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const col         = fp_wrapper<__int128_t>({-1, 2, 3, 4, 5, 6}, scale_type{scale});
    auto const input       = cudf::table_view({col});
    auto const expect_data = std::vector<__int128_t>{-1, 2, 3, 4, 5, 6};

    auto const arr = get_decimal_arrow_array(expect_data, std::nullopt, 38, scale);

    auto const field                = arrow::field("a", arr->type());
    auto const schema_vector        = std::vector<std::shared_ptr<arrow::Field>>({field});
    auto const schema               = std::make_shared<arrow::Schema>(schema_vector);
    auto const expected_arrow_table = arrow::Table::Make(schema, {arr});

    std::vector<cudf::column_metadata> const metadata = {{"a"}};
    ASSERT_TRUE(is_equal(input, metadata, expected_arrow_table));
  }
}

TEST_F(ToArrowTest, FixedPoint64TableLarge)
{
  using namespace numeric;
  auto constexpr NUM_ELEMENTS = 1000;

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const iota        = cuda::counting_iterator<int64_t>{1};
    auto const col         = fp_wrapper<int64_t>(iota, iota + NUM_ELEMENTS, scale_type{scale});
    auto const input       = cudf::table_view({col});
    auto const expect_data = std::vector<int64_t>{iota, iota + NUM_ELEMENTS};

    auto const arr = get_decimal_arrow_array(expect_data, std::nullopt, 18, scale);

    auto const field                = arrow::field("a", arr->type());
    auto const schema_vector        = std::vector<std::shared_ptr<arrow::Field>>({field});
    auto const schema               = std::make_shared<arrow::Schema>(schema_vector);
    auto const expected_arrow_table = arrow::Table::Make(schema, {arr});

    std::vector<cudf::column_metadata> const metadata = {{"a"}};  // NOLINT
    ASSERT_TRUE(is_equal(input, metadata, expected_arrow_table));
  }
}

TEST_F(ToArrowTest, FixedPoint128TableLarge)
{
  using namespace numeric;
  auto constexpr NUM_ELEMENTS = 1000;

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const iota        = cuda::counting_iterator<__int128_t>{1};
    auto const col         = fp_wrapper<__int128_t>(iota, iota + NUM_ELEMENTS, scale_type{scale});
    auto const input       = cudf::table_view({col});
    auto const expect_data = std::vector<__int128_t>{iota, iota + NUM_ELEMENTS};

    auto const arr = get_decimal_arrow_array(expect_data, std::nullopt, 38, scale);

    auto const field                = arrow::field("a", arr->type());
    auto const schema_vector        = std::vector<std::shared_ptr<arrow::Field>>({field});
    auto const schema               = std::make_shared<arrow::Schema>(schema_vector);
    auto const expected_arrow_table = arrow::Table::Make(schema, {arr});

    std::vector<cudf::column_metadata> const metadata = {{"a"}};
    ASSERT_TRUE(is_equal(input, metadata, expected_arrow_table));
  }
}

TEST_F(ToArrowTest, FixedPoint64TableNullsSimple)
{
  using namespace numeric;

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const data     = std::vector<int64_t>{1, 2, 3, 4, 5, 6, 0, 0};
    auto const validity = std::vector<uint8_t>{1, 1, 1, 1, 1, 1, 0, 0};
    auto const col =
      fp_wrapper<int64_t>({1, 2, 3, 4, 5, 6, 0, 0}, {1, 1, 1, 1, 1, 1, 0, 0}, scale_type{scale});
    auto const input = cudf::table_view({col});

    auto const arr = get_decimal_arrow_array(data, validity, 18, scale);

    auto const field         = arrow::field("a", arr->type());
    auto const schema_vector = std::vector<std::shared_ptr<arrow::Field>>({field});
    auto const schema        = std::make_shared<arrow::Schema>(schema_vector);
    auto const arrow_table   = arrow::Table::Make(schema, {arr});

    std::vector<cudf::column_metadata> const metadata = {{"a"}};
    ASSERT_TRUE(is_equal(input, metadata, arrow_table));
  }
}

TEST_F(ToArrowTest, FixedPoint128TableNullsSimple)
{
  using namespace numeric;

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const data     = std::vector<__int128_t>{1, 2, 3, 4, 5, 6, 0, 0};
    auto const validity = std::vector<uint8_t>{1, 1, 1, 1, 1, 1, 0, 0};
    auto const col =
      fp_wrapper<__int128_t>({1, 2, 3, 4, 5, 6, 0, 0}, {1, 1, 1, 1, 1, 1, 0, 0}, scale_type{scale});
    auto const input = cudf::table_view({col});

    auto const arr = get_decimal_arrow_array(data, validity, 38, scale);

    auto const field         = arrow::field("a", arr->type());
    auto const schema_vector = std::vector<std::shared_ptr<arrow::Field>>({field});
    auto const schema        = std::make_shared<arrow::Schema>(schema_vector);
    auto const arrow_table   = arrow::Table::Make(schema, {arr});

    std::vector<cudf::column_metadata> const metadata = {{"a"}};
    ASSERT_TRUE(is_equal(input, metadata, arrow_table));
  }
}

TEST_F(ToArrowTest, FixedPoint64TableNulls)
{
  using namespace numeric;

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const col = fp_wrapper<int64_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, scale_type{scale});
    auto const input = cudf::table_view({col});

    auto const expect_data = std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto const validity    = std::vector<uint8_t>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0};

    auto const arr = get_decimal_arrow_array(expect_data, validity, 18, scale);

    auto const field                = arrow::field("a", arr->type());
    auto const schema_vector        = std::vector<std::shared_ptr<arrow::Field>>({field});
    auto const schema               = std::make_shared<arrow::Schema>(schema_vector);
    auto const expected_arrow_table = arrow::Table::Make(schema, {arr});

    std::vector<cudf::column_metadata> const metadata = {{"a"}};
    ASSERT_TRUE(is_equal(input, metadata, expected_arrow_table));
  }
}

TEST_F(ToArrowTest, FixedPoint128TableNulls)
{
  using namespace numeric;

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const col = fp_wrapper<__int128_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, scale_type{scale});
    auto const input = cudf::table_view({col});

    auto const expect_data = std::vector<__int128_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto const validity    = std::vector<uint8_t>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0};

    auto const arr = get_decimal_arrow_array(expect_data, validity, 38, scale);

    auto const field                = arrow::field("a", arr->type());
    auto const schema_vector        = std::vector<std::shared_ptr<arrow::Field>>({field});
    auto const schema               = std::make_shared<arrow::Schema>(schema_vector);
    auto const expected_arrow_table = arrow::Table::Make(schema, {arr});

    std::vector<cudf::column_metadata> const metadata = {{"a"}};
    ASSERT_TRUE(is_equal(input, metadata, expected_arrow_table));
  }
}

struct ToArrowTestSlice
  : public ToArrowTest,
    public ::testing::WithParamInterface<std::tuple<cudf::size_type, cudf::size_type>> {};

TEST_P(ToArrowTestSlice, SliceTest)
{
  auto tables             = get_tables(10000);
  auto cudf_table_view    = tables.first->view();
  auto arrow_table        = tables.second;
  auto const [start, end] = GetParam();

  auto sliced_cudf_table    = cudf::slice(cudf_table_view, {start, end})[0];
  auto expected_arrow_table = arrow_table->Slice(start, end - start);
  auto struct_meta          = cudf::column_metadata{"f"};
  struct_meta.children_meta = {{"integral"}, {"string"}};

  std::vector<cudf::column_metadata> const metadata = {
    {"a"}, {"b"}, {"c"}, {"d"}, {"e"}, struct_meta};
  ASSERT_TRUE(is_equal(sliced_cudf_table, metadata, expected_arrow_table));
}

INSTANTIATE_TEST_CASE_P(ToArrowTest,
                        ToArrowTestSlice,
                        ::testing::Values(std::make_tuple(0, 10000),
                                          std::make_tuple(100, 3000),
                                          std::make_tuple(0, 0),
                                          std::make_tuple(0, 3000)));

template <typename T>
struct ToArrowNumericScalarTest : public cudf::test::BaseFixture {};

using NumericTypesNotBool =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;
TYPED_TEST_SUITE(ToArrowNumericScalarTest, NumericTypesNotBool);

auto col_to_arrow_type(cudf::column_view const& col)
{
  switch (col.type().id()) {
    case cudf::type_id::BOOL8: return arrow::boolean();
    case cudf::type_id::INT8: return arrow::int8();
    case cudf::type_id::INT16: return arrow::int16();
    case cudf::type_id::INT32: return arrow::int32();
    case cudf::type_id::INT64: return arrow::int64();
    case cudf::type_id::UINT8: return arrow::uint8();
    case cudf::type_id::UINT16: return arrow::uint16();
    case cudf::type_id::UINT32: return arrow::uint32();
    case cudf::type_id::UINT64: return arrow::uint64();
    case cudf::type_id::FLOAT32: return arrow::float32();
    case cudf::type_id::FLOAT64: return arrow::float64();
    case cudf::type_id::TIMESTAMP_DAYS: return arrow::date32();
    case cudf::type_id::STRING: return arrow::utf8();
    case cudf::type_id::LIST:
      return arrow::list(col_to_arrow_type(col.child(cudf::lists_column_view::child_column_index)));
    case cudf::type_id::DECIMAL32: return arrow::decimal32(9, -col.type().scale());
    case cudf::type_id::DECIMAL64: return arrow::decimal64(18, -col.type().scale());
    case cudf::type_id::DECIMAL128: return arrow::decimal128(38, -col.type().scale());
    default: CUDF_FAIL("Unsupported type_id conversion to arrow type", cudf::data_type_error);
  }
}

std::optional<std::shared_ptr<arrow::Scalar>> cudf_scalar_to_arrow(
  cudf::scalar const& scalar, std::optional<cudf::column_metadata> metadata = std::nullopt)
{
  auto const cudf_column   = cudf::make_column_from_scalar(scalar, 1);
  auto const c_arrow_array = cudf::to_arrow_host(*cudf_column);
  auto const arrow_array   = [&]() {
    if (metadata.has_value()) {
      auto const table = cudf::table_view({cudf_column->view()});
      std::vector<cudf::column_metadata> const table_metadata = {metadata.value()};
      auto const arrow_schema = cudf::to_arrow_schema(table, table_metadata);
      return arrow::ImportArray(&c_arrow_array->array, arrow_schema->children[0]).ValueOrDie();
    } else {
      auto const arrow_type = col_to_arrow_type(cudf_column->view());
      return arrow::ImportArray(&c_arrow_array->array, arrow_type).ValueOrDie();
    }
  }();
  auto const maybe_scalar = arrow_array->GetScalar(0);
  if (!maybe_scalar.ok()) { return std::nullopt; }
  return maybe_scalar.ValueOrDie();
}

TYPED_TEST(ToArrowNumericScalarTest, Basic)
{
  TypeParam const value{42};
  auto const cudf_scalar = cudf::make_fixed_width_scalar<TypeParam>(value);

  auto const maybe_scalar = cudf_scalar_to_arrow(*cudf_scalar);
  ASSERT_TRUE(maybe_scalar.has_value());
  auto const arrow_scalar = *maybe_scalar;

  auto const ref_arrow_scalar = arrow::MakeScalar(value);
  EXPECT_TRUE(arrow_scalar->Equals(*ref_arrow_scalar));
}

struct ToArrowDecimalScalarTest : public cudf::test::BaseFixture {};

template <typename DecimalType>
void check_decimal_scalar(int const value, arrow::Scalar const& ref_scalar, int32_t const scale)
{
  auto const cudf_scalar =
    cudf::make_fixed_point_scalar<DecimalType>(value, numeric::scale_type{scale});

  auto const maybe_scalar = cudf_scalar_to_arrow(*cudf_scalar);
  ASSERT_TRUE(maybe_scalar.has_value());
  auto const arrow_scalar = *maybe_scalar;
  EXPECT_TRUE(arrow_scalar->Equals(ref_scalar));
}

TEST_F(ToArrowDecimalScalarTest, Basic)
{
  auto const value{42};
  int32_t const scale{4};

  auto const get_ref_scalar = [&](std::shared_ptr<arrow::DataType> type) {
    auto const maybe_ref_scalar = arrow::MakeScalar(type, value);
    CUDF_EXPECTS(maybe_ref_scalar.ok(), "Failed to construct reference scalar");
    return *maybe_ref_scalar;
  };

  auto const decimal32_scalar  = get_ref_scalar(arrow::decimal32(9, -scale));
  auto const decimal64_scalar  = get_ref_scalar(arrow::decimal64(18, -scale));
  auto const decimal128_scalar = get_ref_scalar(arrow::decimal128(38, -scale));

  check_decimal_scalar<numeric::decimal32>(value, *decimal32_scalar, scale);
  check_decimal_scalar<numeric::decimal64>(value, *decimal64_scalar, scale);
  check_decimal_scalar<numeric::decimal128>(value, *decimal128_scalar, scale);
}

struct ToArrowStringScalarTest : public cudf::test::BaseFixture {};

TEST_F(ToArrowStringScalarTest, Basic)
{
  std::string const value{"hello world"};
  auto const cudf_scalar  = cudf::make_string_scalar(value);
  auto const maybe_scalar = cudf_scalar_to_arrow(*cudf_scalar);
  ASSERT_TRUE(maybe_scalar.has_value());
  auto const arrow_scalar = *maybe_scalar;

  auto const ref_arrow_scalar = arrow::MakeScalar(value);
  EXPECT_TRUE(arrow_scalar->Equals(*ref_arrow_scalar));
}

struct ToArrowListScalarTest : public cudf::test::BaseFixture {};

TEST_F(ToArrowListScalarTest, Basic)
{
  std::vector<int64_t> const host_values = {1, 2, 3, 5, 6, 7, 8};
  std::vector<bool> const host_validity  = {true, true, true, false, true, true, true};

  cudf::test::fixed_width_column_wrapper<int64_t> const col(
    host_values.begin(), host_values.end(), host_validity.begin());

  auto const cudf_scalar = cudf::make_list_scalar(col);

  auto const maybe_scalar = cudf_scalar_to_arrow(*cudf_scalar);
  ASSERT_TRUE(maybe_scalar.has_value());
  auto const arrow_scalar = *maybe_scalar;

  arrow::Int64Builder builder;
  auto const status      = builder.AppendValues(host_values, host_validity);
  auto const maybe_array = builder.Finish();
  auto const array       = *maybe_array;

  auto const ref_arrow_scalar = arrow::ListScalar(array);

  EXPECT_TRUE(arrow_scalar->Equals(ref_arrow_scalar));
}

struct ToArrowStructScalarTest : public cudf::test::BaseFixture {};

TEST_F(ToArrowStructScalarTest, Basic)
{
  int64_t const value{42};
  auto const field_name{"a"};

  cudf::test::fixed_width_column_wrapper<int64_t> const col{value};
  cudf::table_view const tbl({col});
  auto const cudf_scalar = cudf::make_struct_scalar(tbl);

  cudf::column_metadata metadata{""};
  metadata.children_meta.emplace_back(field_name);

  auto const maybe_scalar = cudf_scalar_to_arrow(*cudf_scalar, metadata);
  ASSERT_TRUE(maybe_scalar.has_value());
  auto const arrow_scalar = *maybe_scalar;

  auto const underlying_arrow_scalar = arrow::MakeScalar(value);
  auto const field            = arrow::field(field_name, underlying_arrow_scalar->type, false);
  auto const arrow_type       = arrow::struct_({field});
  auto const ref_arrow_scalar = arrow::StructScalar({underlying_arrow_scalar}, arrow_type);

  EXPECT_TRUE(arrow_scalar->Equals(ref_arrow_scalar));
}

CUDF_TEST_PROGRAM_MAIN()
