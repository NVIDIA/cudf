/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/nanoarrow_utils.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/interop.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cuda/iterator>

struct FromArrowDeviceTest : public cudf::test::BaseFixture {};

template <typename T>
struct FromArrowDeviceTestDurationsTest : public cudf::test::BaseFixture {};

template <typename T>
struct FromArrowDeviceTestDecimalsTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(FromArrowDeviceTestDurationsTest, cudf::test::DurationTypes);
using FixedPointTypes = cudf::test::Types<int32_t, int64_t, __int128_t>;
TYPED_TEST_SUITE(FromArrowDeviceTestDecimalsTest, FixedPointTypes);

TEST_F(FromArrowDeviceTest, FailConditions)
{
  // can't pass null for schema or device array
  EXPECT_THROW(cudf::from_arrow_device(nullptr, nullptr), std::invalid_argument);
  // can't pass null for device array
  ArrowSchema schema;
  EXPECT_THROW(cudf::from_arrow_device(&schema, nullptr), std::invalid_argument);
  // device_type must be CUDA/CUDA_HOST/CUDA_MANAGED
  // should fail with ARROW_DEVICE_CPU
  ArrowDeviceArray arr;
  arr.device_type = ARROW_DEVICE_CPU;
  EXPECT_THROW(cudf::from_arrow_device(&schema, &arr), std::invalid_argument);

  // can't pass null for schema or device array
  EXPECT_THROW(cudf::from_arrow_device_column(nullptr, nullptr), std::invalid_argument);
  // can't pass null for device array
  EXPECT_THROW(cudf::from_arrow_device_column(&schema, nullptr), std::invalid_argument);
  // device_type must be CUDA/CUDA_HOST/CUDA_MANAGED
  // should fail with ARROW_DEVICE_CPU
  EXPECT_THROW(cudf::from_arrow_device_column(&schema, &arr), std::invalid_argument);
}

TEST_F(FromArrowDeviceTest, EmptyTable)
{
  auto const [table, schema, arr] = get_nanoarrow_tables(0);

  auto expected_cudf_table = table->view();

  ArrowDeviceArray input;
  memcpy(&input.array, arr.get(), sizeof(ArrowArray));
  input.device_id   = rmm::get_current_cuda_device().value();
  input.device_type = ARROW_DEVICE_CUDA;
  input.sync_event  = nullptr;

  auto got_cudf_table = cudf::from_arrow_device(schema.get(), &input);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_cudf_table, *got_cudf_table);

  auto got_cudf_col = cudf::from_arrow_device_column(schema.get(), &input);
  EXPECT_EQ(got_cudf_col->type(), cudf::data_type{cudf::type_id::STRUCT});
  cudf::table_view from_struct{
    std::vector<cudf::column_view>(got_cudf_col->child_begin(), got_cudf_col->child_end())};
  CUDF_TEST_EXPECT_TABLES_EQUAL(*got_cudf_table, from_struct);
}

TEST_F(FromArrowDeviceTest, ZeroColumnsWithRows)
{
  constexpr cudf::size_type num_rows = 5;

  nanoarrow::UniqueSchema input_schema;
  ArrowSchemaInit(input_schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(input_schema.get(), 0));

  nanoarrow::UniqueArray input_array;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(input_array.get(), input_schema.get(), nullptr));
  input_array->length     = num_rows;
  input_array->null_count = 0;
  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(input_array.get(), NANOARROW_VALIDATION_LEVEL_MINIMAL, nullptr));

  ArrowDeviceArray input;
  memcpy(&input.array, input_array.get(), sizeof(ArrowArray));
  input.device_id   = rmm::get_current_cuda_device().value();
  input.device_type = ARROW_DEVICE_CUDA;
  input.sync_event  = nullptr;

  auto got_cudf_table = cudf::from_arrow_device(input_schema.get(), &input);
  EXPECT_EQ(got_cudf_table->num_columns(), 0);
  EXPECT_EQ(got_cudf_table->num_rows(), num_rows);
}

TEST_F(FromArrowDeviceTest, DateTimeTable)
{
  auto data = std::vector<int64_t>{1, 2, 3, 4, 5, 6};
  auto col  = cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>(
    data.begin(), data.end());

  cudf::table_view expected_table_view({col});

  nanoarrow::UniqueSchema input_schema;
  ArrowSchemaInit(input_schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(input_schema.get(), 1));
  ArrowSchemaInit(input_schema->children[0]);
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeDateTime(
    input_schema->children[0], NANOARROW_TYPE_TIMESTAMP, NANOARROW_TIME_UNIT_MILLI, nullptr));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(input_schema->children[0], "a"));

  nanoarrow::UniqueArray input_array;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(input_array.get(), input_schema.get(), nullptr));
  input_array->length                  = 6;
  input_array->null_count              = 0;
  input_array->children[0]->length     = 6;
  input_array->children[0]->null_count = 0;
  NANOARROW_THROW_NOT_OK(
    ArrowBufferSetAllocator(ArrowArrayBuffer(input_array->children[0], 1), noop_alloc));
  ArrowArrayBuffer(input_array->children[0], 1)->data =
    const_cast<uint8_t*>(cudf::column_view(col).data<uint8_t>());
  ArrowArrayBuffer(input_array->children[0], 1)->size_bytes =
    sizeof(int64_t) * cudf::column_view(col).size();
  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(input_array.get(), NANOARROW_VALIDATION_LEVEL_MINIMAL, nullptr));

  ArrowDeviceArray input_device_array;
  input_device_array.device_id   = rmm::get_current_cuda_device().value();
  input_device_array.device_type = ARROW_DEVICE_CUDA;
  input_device_array.sync_event  = nullptr;
  memcpy(&input_device_array.array, input_array.get(), sizeof(ArrowArray));

  auto got_cudf_table_view = cudf::from_arrow_device(input_schema.get(), &input_device_array);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table_view, *got_cudf_table_view);

  auto got_cudf_col = cudf::from_arrow_device_column(input_schema.get(), &input_device_array);
  EXPECT_EQ(got_cudf_col->type(), cudf::data_type{cudf::type_id::STRUCT});
  cudf::table_view from_struct{
    std::vector<cudf::column_view>(got_cudf_col->child_begin(), got_cudf_col->child_end())};
  CUDF_TEST_EXPECT_TABLES_EQUAL(*got_cudf_table_view, from_struct);
}

TYPED_TEST(FromArrowDeviceTestDurationsTest, DurationTable)
{
  using T = TypeParam;

  if (cudf::type_to_id<TypeParam>() == cudf::type_id::DURATION_DAYS) { return; }

  auto data = {T{1}, T{2}, T{3}, T{4}, T{5}, T{6}};
  auto col  = cudf::test::fixed_width_column_wrapper<T>(data);

  cudf::table_view expected_table_view({col});
  ArrowTimeUnit const time_unit = [&] {
    switch (cudf::type_to_id<TypeParam>()) {
      case cudf::type_id::DURATION_SECONDS: return NANOARROW_TIME_UNIT_SECOND;
      case cudf::type_id::DURATION_MILLISECONDS: return NANOARROW_TIME_UNIT_MILLI;
      case cudf::type_id::DURATION_MICROSECONDS: return NANOARROW_TIME_UNIT_MICRO;
      case cudf::type_id::DURATION_NANOSECONDS: return NANOARROW_TIME_UNIT_NANO;
      default: CUDF_FAIL("Unsupported duration unit in arrow");
    }
  }();

  nanoarrow::UniqueSchema input_schema;
  ArrowSchemaInit(input_schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(input_schema.get(), 1));

  ArrowSchemaInit(input_schema->children[0]);
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeDateTime(
    input_schema->children[0], NANOARROW_TYPE_DURATION, time_unit, nullptr));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(input_schema->children[0], "a"));

  auto data_ptr  = expected_table_view.column(0).data<uint8_t>();
  auto data_size = expected_table_view.column(0).size();
  nanoarrow::UniqueArray input_array;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(input_array.get(), input_schema.get(), nullptr));
  input_array->length                  = expected_table_view.num_rows();
  input_array->null_count              = 0;
  input_array->children[0]->length     = expected_table_view.num_rows();
  input_array->children[0]->null_count = 0;
  NANOARROW_THROW_NOT_OK(
    ArrowBufferSetAllocator(ArrowArrayBuffer(input_array->children[0], 1), noop_alloc));
  ArrowArrayBuffer(input_array->children[0], 1)->data       = const_cast<uint8_t*>(data_ptr);
  ArrowArrayBuffer(input_array->children[0], 1)->size_bytes = sizeof(T) * data_size;
  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(input_array.get(), NANOARROW_VALIDATION_LEVEL_MINIMAL, nullptr));

  ArrowDeviceArray input_device_array;
  input_device_array.device_id   = rmm::get_current_cuda_device().value();
  input_device_array.device_type = ARROW_DEVICE_CUDA;
  input_device_array.sync_event  = nullptr;
  memcpy(&input_device_array.array, input_array.get(), sizeof(ArrowArray));

  auto got_cudf_table_view = cudf::from_arrow_device(input_schema.get(), &input_device_array);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table_view, *got_cudf_table_view);

  auto got_cudf_col = cudf::from_arrow_device_column(input_schema.get(), &input_device_array);
  EXPECT_EQ(got_cudf_col->type(), cudf::data_type{cudf::type_id::STRUCT});
  cudf::table_view from_struct{
    std::vector<cudf::column_view>(got_cudf_col->child_begin(), got_cudf_col->child_end())};
  CUDF_TEST_EXPECT_TABLES_EQUAL(*got_cudf_table_view, from_struct);
}

TEST_F(FromArrowDeviceTest, NestedList)
{
  auto valids = cudf::test::iterators::nulls_at_multiples_of(3);
  auto col    = cudf::test::lists_column_wrapper<int64_t>(
    {{{{{1, 2}, valids}, {{3, 4}, valids}, {5}}, {{6}, {{7, 8, 9}, valids}}}, valids});
  cudf::table_view expected_table_view({col});

  nanoarrow::UniqueSchema input_schema;
  ArrowSchemaInit(input_schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(input_schema.get(), 1));

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(input_schema->children[0], NANOARROW_TYPE_LIST));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(input_schema->children[0], "a"));
  input_schema->children[0]->flags = ARROW_FLAG_NULLABLE;

  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(input_schema->children[0]->children[0], NANOARROW_TYPE_LIST));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(input_schema->children[0]->children[0], "element"));
  input_schema->children[0]->children[0]->flags = 0;

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(
    input_schema->children[0]->children[0]->children[0], NANOARROW_TYPE_INT64));
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaSetName(input_schema->children[0]->children[0]->children[0], "element"));
  input_schema->children[0]->children[0]->children[0]->flags = ARROW_FLAG_NULLABLE;

  nanoarrow::UniqueArray input_array;
  EXPECT_EQ(NANOARROW_OK, ArrowArrayInitFromSchema(input_array.get(), input_schema.get(), nullptr));
  input_array->length = expected_table_view.num_rows();
  auto top_list       = input_array->children[0];
  cudf::lists_column_view lview{expected_table_view.column(0)};
  populate_list_from_col(top_list, lview);
  cudf::lists_column_view nested_view{lview.child()};
  populate_list_from_col(top_list->children[0], nested_view);
  populate_from_col<int64_t>(top_list->children[0]->children[0], nested_view.child());
  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(input_array.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr));

  ArrowDeviceArray input_device_array;
  input_device_array.device_id   = rmm::get_current_cuda_device().value();
  input_device_array.device_type = ARROW_DEVICE_CUDA;
  input_device_array.sync_event  = nullptr;
  memcpy(&input_device_array.array, input_array.get(), sizeof(ArrowArray));

  auto got_cudf_table_view = cudf::from_arrow_device(input_schema.get(), &input_device_array);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table_view, *got_cudf_table_view);

  auto got_cudf_col = cudf::from_arrow_device_column(input_schema.get(), &input_device_array);
  EXPECT_EQ(got_cudf_col->type(), cudf::data_type{cudf::type_id::STRUCT});
  cudf::table_view from_struct{
    std::vector<cudf::column_view>(got_cudf_col->child_begin(), got_cudf_col->child_end())};
  CUDF_TEST_EXPECT_TABLES_EQUAL(*got_cudf_table_view, from_struct);
}

TEST_F(FromArrowDeviceTest, StructColumn)
{
  using vector_of_columns = std::vector<std::unique_ptr<cudf::column>>;

  // Create cudf table
  auto nested_type_field_names =
    std::vector<std::vector<std::string>>{{"string", "integral", "bool", "nested_list", "struct"}};
  auto str_col =
    cudf::test::strings_column_wrapper{
      "Samuel Vimes", "Carrot Ironfoundersson", "Angua von Überwald"}
      .release();
  auto str_col2 = cudf::test::strings_column_wrapper{{"", "ROCKS", ""}, {0, 1, 0}}.release();
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
  cudf::table_view expected_table_view({struct_col->view()});

  // Create name metadata
  auto sub_metadata          = cudf::column_metadata{"struct"};
  sub_metadata.children_meta = {{"string2"}, {"integral2"}};
  auto metadata              = cudf::column_metadata{"a"};
  metadata.children_meta     = {{"string"}, {"integral"}, {"bool"}, {"nested_list"}, sub_metadata};

  nanoarrow::UniqueSchema input_schema;
  ArrowSchemaInit(input_schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(input_schema.get(), 1));

  ArrowSchemaInit(input_schema->children[0]);
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(input_schema->children[0], 5));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(input_schema->children[0], "a"));
  input_schema->children[0]->flags = 0;

  auto child = input_schema->children[0];
  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(child->children[0], NANOARROW_TYPE_STRING));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(child->children[0], "string"));
  child->children[0]->flags = 0;

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(child->children[1], NANOARROW_TYPE_INT32));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(child->children[1], "integral"));
  child->children[1]->flags = 0;

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(child->children[2], NANOARROW_TYPE_BOOL));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(child->children[2], "bool"));
  child->children[2]->flags = 0;

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(child->children[3], NANOARROW_TYPE_LIST));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(child->children[3], "nested_list"));
  child->children[3]->flags = 0;
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(child->children[3]->children[0], NANOARROW_TYPE_LIST));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(child->children[3]->children[0], "element"));
  child->children[3]->children[0]->flags = 0;
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(child->children[3]->children[0]->children[0], NANOARROW_TYPE_INT64));
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaSetName(child->children[3]->children[0]->children[0], "element"));
  child->children[3]->children[0]->children[0]->flags = 0;

  ArrowSchemaInit(child->children[4]);
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(child->children[4], 2));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(child->children[4], "struct"));

  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(child->children[4]->children[0], NANOARROW_TYPE_STRING));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(child->children[4]->children[0], "string2"));
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(child->children[4]->children[1], NANOARROW_TYPE_INT32));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(child->children[4]->children[1], "integral2"));

  nanoarrow::UniqueArray input_array;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(input_array.get(), input_schema.get(), nullptr));

  input_array->length = expected_table_view.num_rows();

  auto array_a        = input_array->children[0];
  auto view_a         = expected_table_view.column(0);
  array_a->length     = view_a.size();
  array_a->null_count = view_a.null_count();

  NANOARROW_THROW_NOT_OK(ArrowBufferSetAllocator(ArrowArrayBuffer(array_a, 0), noop_alloc));
  ArrowArrayValidityBitmap(array_a)->buffer.data =
    const_cast<uint8_t*>(reinterpret_cast<uint8_t const*>(view_a.null_mask()));

  populate_from_col<cudf::string_view>(array_a->children[0], view_a.child(0));
  populate_from_col<int32_t>(array_a->children[1], view_a.child(1));
  populate_from_col<bool>(array_a->children[2], view_a.child(2));
  populate_list_from_col(array_a->children[3], cudf::lists_column_view{view_a.child(3)});
  populate_list_from_col(array_a->children[3]->children[0],
                         cudf::lists_column_view{view_a.child(3).child(1)});
  populate_from_col<int64_t>(array_a->children[3]->children[0]->children[0],
                             view_a.child(3).child(1).child(1));

  auto array_struct        = array_a->children[4];
  auto view_struct         = view_a.child(4);
  array_struct->length     = view_struct.size();
  array_struct->null_count = view_struct.null_count();

  NANOARROW_THROW_NOT_OK(ArrowBufferSetAllocator(ArrowArrayBuffer(array_struct, 0), noop_alloc));
  ArrowArrayValidityBitmap(array_struct)->buffer.data =
    const_cast<uint8_t*>(reinterpret_cast<uint8_t const*>(view_struct.null_mask()));

  populate_from_col<cudf::string_view>(array_struct->children[0], view_struct.child(0));
  populate_from_col<int32_t>(array_struct->children[1], view_struct.child(1));

  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(input_array.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr));

  ArrowDeviceArray input_device_array;
  input_device_array.device_id   = rmm::get_current_cuda_device().value();
  input_device_array.device_type = ARROW_DEVICE_CUDA;
  input_device_array.sync_event  = nullptr;
  memcpy(&input_device_array.array, input_array.get(), sizeof(ArrowArray));

  auto got_cudf_table_view = cudf::from_arrow_device(input_schema.get(), &input_device_array);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table_view, *got_cudf_table_view);

  {
    // there's one boolean column so we should have one "owned_mem" column in the
    // returned unique_ptr's custom deleter
    cudf::custom_view_deleter<cudf::table_view> const& deleter = got_cudf_table_view.get_deleter();
    EXPECT_EQ(deleter.owned_mem_.size(), 1);
  }

  auto got_cudf_col = cudf::from_arrow_device_column(input_schema.get(), &input_device_array);
  EXPECT_EQ(got_cudf_col->type(), cudf::data_type{cudf::type_id::STRUCT});
  cudf::table_view from_struct{
    std::vector<cudf::column_view>(got_cudf_col->child_begin(), got_cudf_col->child_end())};
  CUDF_TEST_EXPECT_TABLES_EQUAL(*got_cudf_table_view, from_struct);

  {
    // there's one boolean column so we should have one "owned_mem" column in the
    // returned unique_ptr's custom deleter
    cudf::custom_view_deleter<cudf::column_view> const& deleter = got_cudf_col.get_deleter();
    EXPECT_EQ(deleter.owned_mem_.size(), 1);
  }
}

TEST_F(FromArrowDeviceTest, DictionaryIndicesType)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  auto col = cudf::test::fixed_width_column_wrapper<int64_t>({1, 2, 5, 2, 7}, {1, 0, 1, 1, 1});
  columns.emplace_back(cudf::dictionary::encode(col, cudf::data_type(cudf::type_id::INT8)));
  columns.emplace_back(cudf::dictionary::encode(col, cudf::data_type(cudf::type_id::INT16)));
  columns.emplace_back(cudf::dictionary::encode(col, cudf::data_type(cudf::type_id::INT64)));

  cudf::table expected_table(std::move(columns));
  cudf::table_view expected_table_view = expected_table.view();

  nanoarrow::UniqueSchema input_schema;
  ArrowSchemaInit(input_schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(input_schema.get(), 3));

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(input_schema->children[0], NANOARROW_TYPE_INT8));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(input_schema->children[0], "a"));
  NANOARROW_THROW_NOT_OK(ArrowSchemaAllocateDictionary(input_schema->children[0]));
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(input_schema->children[0]->dictionary, NANOARROW_TYPE_INT64));

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(input_schema->children[1], NANOARROW_TYPE_INT16));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(input_schema->children[1], "b"));
  NANOARROW_THROW_NOT_OK(ArrowSchemaAllocateDictionary(input_schema->children[1]));
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(input_schema->children[1]->dictionary, NANOARROW_TYPE_INT64));

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(input_schema->children[2], NANOARROW_TYPE_INT64));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(input_schema->children[2], "c"));
  NANOARROW_THROW_NOT_OK(ArrowSchemaAllocateDictionary(input_schema->children[2]));
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(input_schema->children[2]->dictionary, NANOARROW_TYPE_INT64));

  nanoarrow::UniqueArray input_array;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(input_array.get(), input_schema.get(), nullptr));
  input_array->length     = expected_table.num_rows();
  input_array->null_count = 0;

  populate_from_col<int8_t>(
    input_array->children[0],
    cudf::dictionary_column_view{expected_table_view.column(0)}.get_indices_annotated());
  populate_from_col<int64_t>(input_array->children[0]->dictionary,
                             cudf::dictionary_column_view{expected_table_view.column(0)}.keys());

  populate_from_col<int16_t>(
    input_array->children[1],
    cudf::dictionary_column_view{expected_table_view.column(1)}.get_indices_annotated());
  populate_from_col<int64_t>(input_array->children[1]->dictionary,
                             cudf::dictionary_column_view{expected_table_view.column(1)}.keys());

  populate_from_col<int64_t>(
    input_array->children[2],
    cudf::dictionary_column_view{expected_table_view.column(2)}.get_indices_annotated());
  populate_from_col<int64_t>(input_array->children[2]->dictionary,
                             cudf::dictionary_column_view{expected_table_view.column(2)}.keys());

  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(input_array.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr));

  ArrowDeviceArray input_device_array;
  input_device_array.device_id   = rmm::get_current_cuda_device().value();
  input_device_array.device_type = ARROW_DEVICE_CUDA;
  input_device_array.sync_event  = nullptr;
  memcpy(&input_device_array.array, input_array.get(), sizeof(ArrowArray));

  auto got_cudf_table_view = cudf::from_arrow_device(input_schema.get(), &input_device_array);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table_view, *got_cudf_table_view);

  {
    cudf::custom_view_deleter<cudf::table_view> const& deleter = got_cudf_table_view.get_deleter();
    EXPECT_EQ(deleter.owned_mem_.size(), 0);
  }

  auto got_cudf_col = cudf::from_arrow_device_column(input_schema.get(), &input_device_array);
  EXPECT_EQ(got_cudf_col->type(), cudf::data_type{cudf::type_id::STRUCT});
  cudf::table_view from_struct{
    std::vector<cudf::column_view>(got_cudf_col->child_begin(), got_cudf_col->child_end())};
  CUDF_TEST_EXPECT_TABLES_EQUAL(*got_cudf_table_view, from_struct);

  {
    cudf::custom_view_deleter<cudf::column_view> const& deleter = got_cudf_col.get_deleter();
    EXPECT_EQ(deleter.owned_mem_.size(), 0);
  }
}

TEST_F(FromArrowDeviceTest, StringViewType)
{
  auto data = std::vector<std::string>({"hello",
                                        "worldy",
                                        "much longer string",
                                        "",
                                        "another even longer string",
                                        "",
                                        "other string"});

  auto validity = std::vector<bool>{true, true, true, false, true, true, true};
  auto expected_col =
    cudf::test::strings_column_wrapper(data.begin(), data.end(), validity.begin());
  auto expected_view = cudf::column_view(expected_col);

  ArrowArray input;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(&input, NANOARROW_TYPE_STRING_VIEW));
  NANOARROW_THROW_NOT_OK(ArrowArrayStartAppending(&input));
  for (size_t i = 0; i < data.size(); ++i) {
    auto item = ArrowStringView{data[i].c_str(), static_cast<int64_t>(data[i].size())};
    NANOARROW_THROW_NOT_OK(ArrowArrayAppendString(&input, item));
  }
  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(&input, NANOARROW_VALIDATION_LEVEL_NONE, nullptr));

  ArrowSchema schema;
  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(&schema, NANOARROW_TYPE_STRING_VIEW));
  ArrowArrayView view;
  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&view, &schema, nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&view, &input, nullptr));

  auto stream  = cudf::get_default_stream();
  auto items   = view.buffer_views[1].data.as_binary_view;
  auto d_items = rmm::device_uvector<ArrowBinaryView>(input.length, stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_items.data(),
                                items,
                                input.length * sizeof(ArrowBinaryView),
                                cudaMemcpyDefault,
                                stream.value()));
  auto variadics     = std::vector<rmm::device_buffer>();
  auto variadic_ptrs = std::vector<char*>();
  for (auto i = 0L; i < view.n_variadic_buffers; ++i) {
    variadics.emplace_back(view.variadic_buffers[i], view.variadic_buffer_sizes[i], stream);
    variadic_ptrs.push_back(static_cast<char*>(variadics.back().data()));
  }

  stream.synchronize();

  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(&schema, 1));
  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(schema.children[0], NANOARROW_TYPE_STRING_VIEW));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema.children[0], "a"));

  auto variadic_sizes = std::vector<int64_t>();
  for (auto const& buf : variadics) {
    variadic_sizes.push_back(static_cast<int64_t>(buf.size()));
  }

  // Arrow C Data STRING_VIEW layout: [validity, views, variadic0..N-1, variadic_sizes]
  auto child_buffers =
    std::vector<void const*>(NANOARROW_BINARY_VIEW_FIXED_BUFFERS + variadic_ptrs.size() + 1);
  child_buffers[0] = expected_view.null_mask();
  child_buffers[1] = d_items.data();
  for (std::size_t i = 0; i < variadic_ptrs.size(); ++i) {
    child_buffers[i + NANOARROW_BINARY_VIEW_FIXED_BUFFERS] = variadic_ptrs[i];
  }
  child_buffers.back() = variadic_sizes.data();

  ArrowArray child_array{};
  child_array.length     = input.length;
  child_array.null_count = expected_view.null_count();
  child_array.offset     = 0;
  child_array.n_buffers  = static_cast<int64_t>(child_buffers.size());
  child_array.n_children = 0;
  child_array.buffers    = child_buffers.data();
  child_array.children   = nullptr;
  child_array.dictionary = nullptr;
  child_array.release    = nullptr;

  ArrowArray* children[]       = {&child_array};
  void const* struct_buffers[] = {nullptr};
  ArrowArray struct_array{};
  struct_array.length     = input.length;
  struct_array.null_count = 0;
  struct_array.offset     = 0;
  struct_array.n_buffers  = 1;
  struct_array.n_children = 1;
  struct_array.buffers    = struct_buffers;
  struct_array.children   = children;
  struct_array.dictionary = nullptr;
  struct_array.release    = nullptr;

  ArrowDeviceArray input_device_array;
  input_device_array.device_id   = rmm::get_current_cuda_device().value();
  input_device_array.device_type = ARROW_DEVICE_CUDA;
  input_device_array.sync_event  = nullptr;
  input_device_array.array       = struct_array;

  auto got_cudf_table_view = cudf::from_arrow_device(&schema, &input_device_array);
  CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view({expected_col}), *got_cudf_table_view);
}

TEST_F(FromArrowDeviceTest, StringViewTypeWithProducerOwnedPrivateData)
{
  auto data = std::vector<std::string>({"hello",
                                        "worldy",
                                        "much longer string",
                                        "",
                                        "another even longer string",
                                        "",
                                        "other string"});

  auto validity = std::vector<bool>{true, true, true, false, true, true, true};
  auto expected_col =
    cudf::test::strings_column_wrapper(data.begin(), data.end(), validity.begin());
  auto expected_view = cudf::column_view(expected_col);

  nanoarrow::UniqueArray input;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(input.get(), NANOARROW_TYPE_STRING_VIEW));
  NANOARROW_THROW_NOT_OK(ArrowArrayStartAppending(input.get()));
  for (auto const& str : data) {
    auto item = ArrowStringView{str.c_str(), static_cast<int64_t>(str.size())};
    NANOARROW_THROW_NOT_OK(ArrowArrayAppendString(input.get(), item));
  }
  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(input.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr));

  nanoarrow::UniqueSchema schema;
  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(schema.get(), NANOARROW_TYPE_STRING_VIEW));
  ArrowArrayView view;
  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&view, schema.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&view, input.get(), nullptr));
  ASSERT_GT(view.n_variadic_buffers, 0);

  auto stream  = cudf::get_default_stream();
  auto items   = view.buffer_views[1].data.as_binary_view;
  auto d_items = rmm::device_uvector<ArrowBinaryView>(input->length, stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_items.data(),
                                items,
                                input->length * sizeof(ArrowBinaryView),
                                cudaMemcpyDefault,
                                stream.value()));
  auto variadics     = std::vector<rmm::device_buffer>();
  auto variadic_ptrs = std::vector<char*>();
  for (auto i = 0L; i < view.n_variadic_buffers; ++i) {
    variadics.emplace_back(view.variadic_buffers[i], view.variadic_buffer_sizes[i], stream);
    variadic_ptrs.push_back(static_cast<char*>(variadics.back().data()));
  }
  stream.synchronize();

  auto variadic_sizes = std::vector<int64_t>();
  for (auto i = 0L; i < view.n_variadic_buffers; ++i) {
    variadic_sizes.push_back(view.variadic_buffer_sizes[i]);
  }

  // Arrow C Data STRING_VIEW layout: [validity, views, variadic0..N-1, variadic_sizes]
  auto device_buffers =
    std::vector<void const*>(NANOARROW_BINARY_VIEW_FIXED_BUFFERS + variadic_ptrs.size() + 1);
  device_buffers[0] = expected_view.null_mask();
  device_buffers[1] = d_items.data();
  for (std::size_t i = 0; i < variadic_ptrs.size(); ++i) {
    device_buffers[i + NANOARROW_BINARY_VIEW_FIXED_BUFFERS] = variadic_ptrs[i];
  }
  device_buffers.back() = variadic_sizes.data();

  int producer_private_data = 0;
  ArrowArray device_array{};
  device_array.length       = input->length;
  device_array.null_count   = expected_view.null_count();
  device_array.offset       = 0;
  device_array.n_buffers    = static_cast<int64_t>(device_buffers.size());
  device_array.n_children   = 0;
  device_array.buffers      = device_buffers.data();
  device_array.children     = nullptr;
  device_array.dictionary   = nullptr;
  device_array.release      = nullptr;
  device_array.private_data = &producer_private_data;

  ArrowDeviceArray input_device_array;
  input_device_array.device_id   = rmm::get_current_cuda_device().value();
  input_device_array.device_type = ARROW_DEVICE_CUDA;
  input_device_array.sync_event  = nullptr;
  input_device_array.array       = device_array;

  // Mirrors external Arrow C Device producers where ArrowArray.private_data is producer-owned
  // and not Nanoarrow state.
  auto got_cudf_col = cudf::from_arrow_device_column(schema.get(), &input_device_array);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_view, *got_cudf_col);
}

void slice_nanoarrow(ArrowArray* arr, int64_t start, int64_t end)
{
  auto op = [&](ArrowArray* array) {
    array->offset = start;
    array->length = end - start;
    if (array->null_count != 0) {
      array->null_count =
        cudf::null_count(reinterpret_cast<cudf::bitmask_type const*>(array->buffers[0]),
                         start,
                         end,
                         cudf::get_default_stream());
    }
  };

  if (arr->n_children == 0) {
    op(arr);
    return;
  }

  arr->length = end - start;
  for (int64_t i = 0; i < arr->n_children; ++i) {
    op(arr->children[i]);
  }
}

struct FromArrowDeviceTestSlice
  : public FromArrowDeviceTest,
    public ::testing::WithParamInterface<std::tuple<cudf::size_type, cudf::size_type>> {};

TEST_P(FromArrowDeviceTestSlice, SliceTest)
{
  auto [table, schema, array] = get_nanoarrow_tables(10000);
  auto cudf_table_view        = table->view();
  auto const [start, end]     = GetParam();

  auto sliced_cudf_table = cudf::slice(cudf_table_view, {start, end})[0];
  slice_nanoarrow(array.get(), start, end);

  ArrowDeviceArray input_device_array;
  input_device_array.device_id   = rmm::get_current_cuda_device().value();
  input_device_array.device_type = ARROW_DEVICE_CUDA;
  input_device_array.sync_event  = nullptr;
  memcpy(&input_device_array.array, array.get(), sizeof(ArrowArray));

  auto got_cudf_table_view = cudf::from_arrow_device(schema.get(), &input_device_array);
  if (got_cudf_table_view->num_rows() == 0 and sliced_cudf_table.num_rows() == 0) {
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sliced_cudf_table, *got_cudf_table_view);

    auto got_cudf_col = cudf::from_arrow_device_column(schema.get(), &input_device_array);
    EXPECT_EQ(got_cudf_col->type(), cudf::data_type{cudf::type_id::STRUCT});
    cudf::table_view from_struct{
      std::vector<cudf::column_view>(got_cudf_col->child_begin(), got_cudf_col->child_end())};
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*got_cudf_table_view, from_struct);

  } else {
    CUDF_TEST_EXPECT_TABLES_EQUAL(sliced_cudf_table, *got_cudf_table_view);

    auto got_cudf_col = cudf::from_arrow_device_column(schema.get(), &input_device_array);
    EXPECT_EQ(got_cudf_col->type(), cudf::data_type{cudf::type_id::STRUCT});
    cudf::table_view from_struct{
      std::vector<cudf::column_view>(got_cudf_col->child_begin(), got_cudf_col->child_end())};
    CUDF_TEST_EXPECT_TABLES_EQUAL(*got_cudf_table_view, from_struct);
  }
}

INSTANTIATE_TEST_CASE_P(FromArrowDeviceTest,
                        FromArrowDeviceTestSlice,
                        ::testing::Values(std::make_tuple(0, 10000),
                                          std::make_tuple(2912, 2915),
                                          std::make_tuple(100, 3000),
                                          std::make_tuple(0, 0),
                                          std::make_tuple(0, 3000),
                                          std::make_tuple(10000, 10000)));

template <typename T>
using fp_wrapper = cudf::test::fixed_point_column_wrapper<T>;

TYPED_TEST(FromArrowDeviceTestDecimalsTest, FixedPointTable)
{
  using T = TypeParam;
  using namespace numeric;

  auto const precision = get_decimal_precision<T>();

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const data     = std::vector<T>{1, 2, 3, 4, 5, 6};
    auto const col      = fp_wrapper<T>(data.cbegin(), data.cend(), scale_type{scale});
    auto const expected = cudf::table_view({col});

    nanoarrow::UniqueSchema input_schema;
    ArrowSchemaInit(input_schema.get());
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(input_schema.get(), 1));
    ArrowSchemaInit(input_schema->children[0]);
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeDecimal(
      input_schema->children[0], nanoarrow_decimal_type<T>::type, precision, -scale));
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(input_schema->children[0], "a"));

    nanoarrow::UniqueArray input_array;
    NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromSchema(input_array.get(), input_schema.get(), nullptr));
    input_array->length = expected.num_rows();

    populate_from_col<T>(input_array->children[0], expected.column(0));
    NANOARROW_THROW_NOT_OK(
      ArrowArrayFinishBuilding(input_array.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr));

    ArrowDeviceArray input_device_array;
    input_device_array.device_id   = rmm::get_current_cuda_device().value();
    input_device_array.device_type = ARROW_DEVICE_CUDA;
    input_device_array.sync_event  = nullptr;
    memcpy(&input_device_array.array, input_array.get(), sizeof(ArrowArray));

    auto got_cudf_table_view = cudf::from_arrow_device(input_schema.get(), &input_device_array);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *got_cudf_table_view);

    auto got_cudf_col = cudf::from_arrow_device_column(input_schema.get(), &input_device_array);
    EXPECT_EQ(got_cudf_col->type(), cudf::data_type{cudf::type_id::STRUCT});
    cudf::table_view from_struct{
      std::vector<cudf::column_view>(got_cudf_col->child_begin(), got_cudf_col->child_end())};
    CUDF_TEST_EXPECT_TABLES_EQUAL(*got_cudf_table_view, from_struct);
  }
}

TYPED_TEST(FromArrowDeviceTestDecimalsTest, FixedPointTableLarge)
{
  using T = TypeParam;
  using namespace numeric;

  auto const precision        = get_decimal_precision<T>();
  auto constexpr NUM_ELEMENTS = 1000;

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto iota       = cudf::detail::make_counting_transform_iterator(1, [](int i) { return T{i}; });
    auto const data = std::vector<T>(iota, iota + NUM_ELEMENTS);
    auto const col  = fp_wrapper<T>(iota, iota + NUM_ELEMENTS, scale_type{scale});
    auto const expected = cudf::table_view({col});

    nanoarrow::UniqueSchema input_schema;
    ArrowSchemaInit(input_schema.get());
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(input_schema.get(), 1));
    ArrowSchemaInit(input_schema->children[0]);
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeDecimal(
      input_schema->children[0], nanoarrow_decimal_type<T>::type, precision, -scale));
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(input_schema->children[0], "a"));

    nanoarrow::UniqueArray input_array;
    NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromSchema(input_array.get(), input_schema.get(), nullptr));
    input_array->length = expected.num_rows();

    populate_from_col<T>(input_array->children[0], expected.column(0));
    NANOARROW_THROW_NOT_OK(
      ArrowArrayFinishBuilding(input_array.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr));

    ArrowDeviceArray input_device_array;
    input_device_array.device_id   = rmm::get_current_cuda_device().value();
    input_device_array.device_type = ARROW_DEVICE_CUDA;
    input_device_array.sync_event  = nullptr;
    memcpy(&input_device_array.array, input_array.get(), sizeof(ArrowArray));

    auto got_cudf_table_view = cudf::from_arrow_device(input_schema.get(), &input_device_array);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *got_cudf_table_view);

    auto got_cudf_col = cudf::from_arrow_device_column(input_schema.get(), &input_device_array);
    EXPECT_EQ(got_cudf_col->type(), cudf::data_type{cudf::type_id::STRUCT});
    cudf::table_view from_struct{
      std::vector<cudf::column_view>(got_cudf_col->child_begin(), got_cudf_col->child_end())};
    CUDF_TEST_EXPECT_TABLES_EQUAL(*got_cudf_table_view, from_struct);
  }
}

TYPED_TEST(FromArrowDeviceTestDecimalsTest, FixedPointTableNulls)
{
  using T = TypeParam;
  using namespace numeric;

  auto const precision = get_decimal_precision<T>();

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const data     = std::vector<T>{1, 2, 3, 4, 5, 6, 0, 0};
    auto const validity = std::vector<int32_t>{1, 1, 1, 1, 1, 1, 0, 0};
    auto const col =
      fp_wrapper<T>({1, 2, 3, 4, 5, 6, 0, 0}, {1, 1, 1, 1, 1, 1, 0, 0}, scale_type{scale});
    auto const expected = cudf::table_view({col});

    nanoarrow::UniqueSchema input_schema;
    ArrowSchemaInit(input_schema.get());
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(input_schema.get(), 1));
    ArrowSchemaInit(input_schema->children[0]);
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeDecimal(
      input_schema->children[0], nanoarrow_decimal_type<T>::type, precision, -scale));
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(input_schema->children[0], "a"));

    nanoarrow::UniqueArray input_array;
    NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromSchema(input_array.get(), input_schema.get(), nullptr));
    input_array->length = expected.num_rows();

    populate_from_col<T>(input_array->children[0], expected.column(0));
    NANOARROW_THROW_NOT_OK(
      ArrowArrayFinishBuilding(input_array.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr));

    ArrowDeviceArray input_device_array;
    input_device_array.device_id   = rmm::get_current_cuda_device().value();
    input_device_array.device_type = ARROW_DEVICE_CUDA;
    input_device_array.sync_event  = nullptr;
    memcpy(&input_device_array.array, input_array.get(), sizeof(ArrowArray));

    auto got_cudf_table_view = cudf::from_arrow_device(input_schema.get(), &input_device_array);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *got_cudf_table_view);

    auto got_cudf_col = cudf::from_arrow_device_column(input_schema.get(), &input_device_array);
    EXPECT_EQ(got_cudf_col->type(), cudf::data_type{cudf::type_id::STRUCT});
    cudf::table_view from_struct{
      std::vector<cudf::column_view>(got_cudf_col->child_begin(), got_cudf_col->child_end())};
    CUDF_TEST_EXPECT_TABLES_EQUAL(*got_cudf_table_view, from_struct);
  }
}

TYPED_TEST(FromArrowDeviceTestDecimalsTest, FixedPointTableNullsLarge)
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

    nanoarrow::UniqueSchema input_schema;
    ArrowSchemaInit(input_schema.get());
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(input_schema.get(), 1));
    ArrowSchemaInit(input_schema->children[0]);
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeDecimal(
      input_schema->children[0], nanoarrow_decimal_type<T>::type, precision, -scale));
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(input_schema->children[0], "a"));

    nanoarrow::UniqueArray input_array;
    NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromSchema(input_array.get(), input_schema.get(), nullptr));
    input_array->length = expected.num_rows();

    populate_from_col<T>(input_array->children[0], expected.column(0));
    NANOARROW_THROW_NOT_OK(
      ArrowArrayFinishBuilding(input_array.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr));

    ArrowDeviceArray input_device_array;
    input_device_array.device_id   = rmm::get_current_cuda_device().value();
    input_device_array.device_type = ARROW_DEVICE_CUDA;
    input_device_array.sync_event  = nullptr;
    memcpy(&input_device_array.array, input_array.get(), sizeof(ArrowArray));

    auto got_cudf_table_view = cudf::from_arrow_device(input_schema.get(), &input_device_array);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *got_cudf_table_view);

    auto got_cudf_col = cudf::from_arrow_device_column(input_schema.get(), &input_device_array);
    EXPECT_EQ(got_cudf_col->type(), cudf::data_type{cudf::type_id::STRUCT});
    cudf::table_view from_struct{
      std::vector<cudf::column_view>(got_cudf_col->child_begin(), got_cudf_col->child_end())};
    CUDF_TEST_EXPECT_TABLES_EQUAL(*got_cudf_table_view, from_struct);
  }
}
