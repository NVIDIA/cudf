/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/nanoarrow_utils.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/detail/interop.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

std::tuple<std::unique_ptr<cudf::table>, nanoarrow::UniqueSchema, generated_test_data>
get_nanoarrow_cudf_table(cudf::size_type length)
{
  generated_test_data test_data(length);

  std::vector<std::unique_ptr<cudf::column>> columns;

  columns.emplace_back(cudf::test::fixed_width_column_wrapper<int64_t>(test_data.int64_data.begin(),
                                                                       test_data.int64_data.end(),
                                                                       test_data.validity.begin())
                         .release());
  columns.emplace_back(cudf::test::strings_column_wrapper(test_data.string_data.begin(),
                                                          test_data.string_data.end(),
                                                          test_data.validity.begin())
                         .release());
  auto col4 = cudf::test::fixed_width_column_wrapper<int64_t>(
    test_data.int64_data.begin(), test_data.int64_data.end(), test_data.validity.begin());
  columns.emplace_back(cudf::dictionary::encode(col4));
  columns.emplace_back(cudf::test::fixed_width_column_wrapper<bool>(test_data.bool_data.begin(),
                                                                    test_data.bool_data.end(),
                                                                    test_data.bool_validity.begin())
                         .release());
  auto list_child_column =
    cudf::test::fixed_width_column_wrapper<int64_t>(test_data.list_int64_data.begin(),
                                                    test_data.list_int64_data.end(),
                                                    test_data.list_int64_data_validity.begin());
  auto list_offsets_column = cudf::test::fixed_width_column_wrapper<int32_t>(
    test_data.list_offsets.begin(), test_data.list_offsets.end());
  auto [list_mask, list_nulls] = cudf::bools_to_mask(cudf::test::fixed_width_column_wrapper<bool>(
    test_data.list_validity.begin(), test_data.list_validity.end()));
  columns.emplace_back(cudf::make_lists_column(length,
                                               list_offsets_column.release(),
                                               list_child_column.release(),
                                               list_nulls,
                                               std::move(*list_mask)));
  auto int_column =
    cudf::test::fixed_width_column_wrapper<int64_t>(
      test_data.int64_data.begin(), test_data.int64_data.end(), test_data.validity.begin())
      .release();
  auto str_column =
    cudf::test::strings_column_wrapper(
      test_data.string_data.begin(), test_data.string_data.end(), test_data.validity.begin())
      .release();
  vector_of_columns cols;
  cols.push_back(std::move(int_column));
  cols.push_back(std::move(str_column));
  auto [null_mask, null_count] = cudf::bools_to_mask(cudf::test::fixed_width_column_wrapper<bool>(
    test_data.bool_data_validity.begin(), test_data.bool_data_validity.end()));
  columns.emplace_back(
    cudf::make_structs_column(length, std::move(cols), null_count, std::move(*null_mask)));

  nanoarrow::UniqueSchema schema;
  ArrowSchemaInit(schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(schema.get(), 6));

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(schema->children[0], NANOARROW_TYPE_INT64));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[0], "a"));
  if (columns[0]->null_count() > 0) {
    schema->children[0]->flags |= ARROW_FLAG_NULLABLE;
  } else {
    schema->children[0]->flags = 0;
  }

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(schema->children[1], NANOARROW_TYPE_STRING));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[1], "b"));
  if (columns[1]->null_count() > 0) {
    schema->children[1]->flags |= ARROW_FLAG_NULLABLE;
  } else {
    schema->children[1]->flags = 0;
  }

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(schema->children[2], NANOARROW_TYPE_INT32));
  NANOARROW_THROW_NOT_OK(ArrowSchemaAllocateDictionary(schema->children[2]));
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(schema->children[2]->dictionary, NANOARROW_TYPE_INT64));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[2], "c"));
  if (columns[2]->null_count() > 0) {
    schema->children[2]->flags |= ARROW_FLAG_NULLABLE;
  } else {
    schema->children[2]->flags = 0;
  }

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(schema->children[3], NANOARROW_TYPE_BOOL));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[3], "d"));
  if (columns[3]->null_count() > 0) {
    schema->children[3]->flags |= ARROW_FLAG_NULLABLE;
  } else {
    schema->children[3]->flags = 0;
  }

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(schema->children[4], NANOARROW_TYPE_LIST));
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(schema->children[4]->children[0], NANOARROW_TYPE_INT64));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[4]->children[0], "element"));
  if (columns[4]->child(1).null_count() > 0) {
    schema->children[4]->children[0]->flags |= ARROW_FLAG_NULLABLE;
  } else {
    schema->children[4]->children[0]->flags = 0;
  }

  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[4], "e"));
  if (columns[4]->has_nulls()) {
    schema->children[4]->flags |= ARROW_FLAG_NULLABLE;
  } else {
    schema->children[4]->flags = 0;
  }

  ArrowSchemaInit(schema->children[5]);
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(schema->children[5], 2));
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(schema->children[5]->children[0], NANOARROW_TYPE_INT64));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[5]->children[0], "integral"));
  if (columns[5]->child(0).has_nulls()) {
    schema->children[5]->children[0]->flags |= ARROW_FLAG_NULLABLE;
  } else {
    schema->children[5]->children[0]->flags = 0;
  }

  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(schema->children[5]->children[1], NANOARROW_TYPE_STRING));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[5]->children[1], "string"));
  if (columns[5]->child(1).has_nulls()) {
    schema->children[5]->children[1]->flags |= ARROW_FLAG_NULLABLE;
  } else {
    schema->children[5]->children[1]->flags = 0;
  }

  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[5], "f"));
  if (columns[5]->has_nulls()) {
    schema->children[5]->flags |= ARROW_FLAG_NULLABLE;
  } else {
    schema->children[5]->flags = 0;
  }

  return std::make_tuple(
    std::make_unique<cudf::table>(std::move(columns)), std::move(schema), std::move(test_data));
}

std::tuple<std::unique_ptr<cudf::table>, nanoarrow::UniqueSchema, nanoarrow::UniqueArray>
get_nanoarrow_tables(cudf::size_type length)
{
  auto [table, schema, test_data] = get_nanoarrow_cudf_table(length);

  nanoarrow::UniqueArray arrow;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(arrow.get(), schema.get(), nullptr));
  arrow->length = length;

  populate_from_col<int64_t>(arrow->children[0], table->get_column(0).view());
  populate_from_col<cudf::string_view>(arrow->children[1], table->get_column(1).view());
  populate_dict_from_col<int64_t, int32_t>(
    arrow->children[2], cudf::dictionary_column_view(table->get_column(2).view()));

  populate_from_col<bool>(arrow->children[3], table->get_column(3).view());
  cudf::lists_column_view list_view{table->get_column(4).view()};
  populate_list_from_col(arrow->children[4], list_view);
  populate_from_col<int64_t>(arrow->children[4]->children[0], list_view.child());

  cudf::structs_column_view struct_view{table->get_column(5).view()};
  populate_from_col<int64_t>(arrow->children[5]->children[0], struct_view.child(0));
  populate_from_col<cudf::string_view>(arrow->children[5]->children[1], struct_view.child(1));
  arrow->children[5]->length     = struct_view.size();
  arrow->children[5]->null_count = struct_view.null_count();
  NANOARROW_THROW_NOT_OK(
    ArrowBufferSetAllocator(ArrowArrayBuffer(arrow->children[5], 0), noop_alloc));
  ArrowArrayValidityBitmap(arrow->children[5])->buffer.size_bytes =
    cudf::bitmask_allocation_size_bytes(struct_view.size());
  ArrowArrayValidityBitmap(arrow->children[5])->buffer.data =
    const_cast<uint8_t*>(reinterpret_cast<uint8_t const*>(struct_view.null_mask()));

  ArrowError error;
  if (ArrowArrayFinishBuilding(arrow.get(), NANOARROW_VALIDATION_LEVEL_MINIMAL, &error) !=
      NANOARROW_OK) {
    std::cerr << ArrowErrorMessage(&error) << std::endl;
    CUDF_FAIL("failed to build example arrays");
  }

  return std::make_tuple(std::move(table), std::move(schema), std::move(arrow));
}

// populate an ArrowArray list array from device buffers using a no-op
// allocator so that the ArrowArray doesn't have ownership of the buffers
void populate_list_from_col(ArrowArray* arr, cudf::lists_column_view view)
{
  arr->length     = view.size();
  arr->null_count = view.null_count();

  NANOARROW_THROW_NOT_OK(ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 0), noop_alloc));
  ArrowArrayValidityBitmap(arr)->buffer.size_bytes =
    cudf::bitmask_allocation_size_bytes(view.size());
  ArrowArrayValidityBitmap(arr)->buffer.data =
    const_cast<uint8_t*>(reinterpret_cast<uint8_t const*>(view.null_mask()));

  NANOARROW_THROW_NOT_OK(ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 1), noop_alloc));
  ArrowArrayBuffer(arr, 1)->size_bytes = sizeof(int32_t) * view.offsets().size();
  ArrowArrayBuffer(arr, 1)->data       = const_cast<uint8_t*>(view.offsets().data<uint8_t>());
}

struct BaseArrowFixture : public cudf::test::BaseFixture {
  void compare_schemas(ArrowSchema const* expected, ArrowSchema const* actual)
  {
    EXPECT_STREQ(expected->format, actual->format);
    EXPECT_STREQ(expected->name, actual->name);
    EXPECT_STREQ(expected->metadata, actual->metadata);
    EXPECT_EQ(expected->flags, actual->flags);
    EXPECT_EQ(expected->n_children, actual->n_children);

    if (expected->n_children == 0) {
      EXPECT_EQ(nullptr, actual->children);
    } else {
      for (int i = 0; i < expected->n_children; ++i) {
        SCOPED_TRACE(expected->children[i]->name);
        compare_schemas(expected->children[i], actual->children[i]);
      }
    }

    if (expected->dictionary != nullptr) {
      EXPECT_NE(nullptr, actual->dictionary);
      SCOPED_TRACE("dictionary");
      compare_schemas(expected->dictionary, actual->dictionary);
    } else {
      EXPECT_EQ(nullptr, actual->dictionary);
    }
  }

  void compare_device_buffers(size_t const nbytes,
                              int const buffer_idx,
                              ArrowArray const* expected,
                              ArrowArray const* actual)
  {
    auto const dt       = cudf::data_type{cudf::type_id::UINT8};
    auto actual_bytes   = cudf::column_view(dt, nbytes, actual->buffers[buffer_idx], nullptr, 0);
    auto expected_bytes = cudf::column_view(dt, nbytes, expected->buffers[buffer_idx], nullptr, 0);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual_bytes, expected_bytes);
  }

  void compare_arrays(ArrowSchema const* schema,
                      ArrowArray const* expected,
                      ArrowArray const* actual)
  {
    ArrowSchemaView schema_view;
    NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&schema_view, schema, nullptr));

    EXPECT_EQ(expected->length, actual->length);
    EXPECT_EQ(expected->null_count, actual->null_count);
    EXPECT_EQ(expected->offset, actual->offset);
    EXPECT_EQ(expected->n_buffers, actual->n_buffers);
    EXPECT_EQ(expected->n_children, actual->n_children);

    if (expected->length > 0) {
      EXPECT_EQ(expected->buffers[0], actual->buffers[0]);
      if (schema_view.type == NANOARROW_TYPE_BOOL) {
        size_t const nbytes = (expected->length + 7) >> 3;
        compare_device_buffers(nbytes, 1, expected, actual);
      } else if (schema_view.type == NANOARROW_TYPE_DECIMAL128) {
        size_t const nbytes = (expected->length * sizeof(__int128_t));
        compare_device_buffers(nbytes, 1, expected, actual);
      } else {
        for (int i = 1; i < expected->n_buffers; ++i) {
          EXPECT_EQ(expected->buffers[i], actual->buffers[i]);
        }
      }
    }

    if (expected->n_children == 0) {
      EXPECT_EQ(nullptr, actual->children);
    } else {
      for (int i = 0; i < expected->n_children; ++i) {
        SCOPED_TRACE(schema->children[i]->name);
        compare_arrays(schema->children[i], expected->children[i], actual->children[i]);
      }
    }

    if (expected->dictionary != nullptr) {
      EXPECT_NE(nullptr, actual->dictionary);
      SCOPED_TRACE("dictionary");
      compare_arrays(schema->dictionary, expected->dictionary, actual->dictionary);
    } else {
      EXPECT_EQ(nullptr, actual->dictionary);
    }
  }
};

struct ToArrowDeviceTest : public BaseArrowFixture {};

template <typename T>
struct ToArrowDeviceTestDurationsTest : public BaseArrowFixture {};

TYPED_TEST_SUITE(ToArrowDeviceTestDurationsTest, cudf::test::DurationTypes);

TEST_F(ToArrowDeviceTest, EmptyTable)
{
  auto const [table, schema, arr] = get_nanoarrow_tables(0);

  auto struct_meta          = cudf::column_metadata{"f"};
  struct_meta.children_meta = {{"integral"}, {"string"}};

  cudf::dictionary_column_view dview{table->view().column(2)};

  std::vector<cudf::column_metadata> meta{{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, struct_meta};
  auto got_arrow_schema = cudf::to_arrow_schema(table->view(), meta);

  compare_schemas(schema.get(), got_arrow_schema.get());

  auto got_arrow_device = cudf::to_arrow_device(table->view());
  EXPECT_EQ(rmm::get_current_cuda_device().value(), got_arrow_device->device_id);
  EXPECT_EQ(ARROW_DEVICE_CUDA, got_arrow_device->device_type);
  ASSERT_CUDA_SUCCEEDED(
    cudaEventSynchronize(*reinterpret_cast<cudaEvent_t*>(got_arrow_device->sync_event)));
  compare_arrays(schema.get(), arr.get(), &got_arrow_device->array);

  got_arrow_device = cudf::to_arrow_device(std::move(*table));
  EXPECT_EQ(rmm::get_current_cuda_device().value(), got_arrow_device->device_id);
  EXPECT_EQ(ARROW_DEVICE_CUDA, got_arrow_device->device_type);
  ASSERT_CUDA_SUCCEEDED(
    cudaEventSynchronize(*reinterpret_cast<cudaEvent_t*>(got_arrow_device->sync_event)));
  compare_arrays(schema.get(), arr.get(), &got_arrow_device->array);
}

TEST_F(ToArrowDeviceTest, EmptyDictionary)
{
  auto empty = cudf::make_empty_column(cudf::type_id::DICTIONARY32);
  auto meta  = std::vector<cudf::column_metadata>({cudf::column_metadata{"d"}});

  auto got_arrow_schema = cudf::to_arrow_schema(cudf::table_view({empty->view()}), meta);
  ASSERT_EQ(got_arrow_schema->n_children, 1);
  auto dictionary_schema = got_arrow_schema->children[0]->dictionary;
  ASSERT_NE(dictionary_schema, nullptr);
  EXPECT_EQ(dictionary_schema->n_children, 0);

  auto got_arrow_device = cudf::to_arrow_device(cudf::table_view({empty->view()}));
  ASSERT_EQ(got_arrow_device->array.n_children, 1);
  auto dictionary = got_arrow_device->array.children[0]->dictionary;
  ASSERT_NE(dictionary, nullptr);
  EXPECT_EQ(dictionary->n_children, 0);
}

TEST_F(ToArrowDeviceTest, DateTimeTable)
{
  auto data = {1, 2, 3, 4, 5, 6};
  auto col =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>(data);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.emplace_back(col.release());
  cudf::table input(std::move(cols));

  auto got_arrow_schema =
    cudf::to_arrow_schema(input.view(), std::vector<cudf::column_metadata>{{"a"}});
  nanoarrow::UniqueSchema expected_schema;
  ArrowSchemaInit(expected_schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(expected_schema.get(), 1));
  ArrowSchemaInit(expected_schema->children[0]);
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeDateTime(
    expected_schema->children[0], NANOARROW_TYPE_TIMESTAMP, NANOARROW_TIME_UNIT_MILLI, nullptr));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(expected_schema->children[0], "a"));
  expected_schema->children[0]->flags = 0;

  compare_schemas(expected_schema.get(), got_arrow_schema.get());

  auto data_ptr        = input.get_column(0).view().data<int64_t>();
  auto got_arrow_array = cudf::to_arrow_device(input.view());
  EXPECT_EQ(rmm::get_current_cuda_device().value(), got_arrow_array->device_id);
  EXPECT_EQ(ARROW_DEVICE_CUDA, got_arrow_array->device_type);
  ASSERT_CUDA_SUCCEEDED(
    cudaEventSynchronize(*reinterpret_cast<cudaEvent_t*>(got_arrow_array->sync_event)));

  EXPECT_EQ(data.size(), got_arrow_array->array.length);
  EXPECT_EQ(0, got_arrow_array->array.null_count);
  EXPECT_EQ(0, got_arrow_array->array.offset);
  EXPECT_EQ(1, got_arrow_array->array.n_children);
  EXPECT_EQ(nullptr, got_arrow_array->array.buffers[0]);

  EXPECT_EQ(data.size(), got_arrow_array->array.children[0]->length);
  EXPECT_EQ(0, got_arrow_array->array.children[0]->null_count);
  EXPECT_EQ(0, got_arrow_array->array.children[0]->offset);
  EXPECT_EQ(nullptr, got_arrow_array->array.children[0]->buffers[0]);
  EXPECT_EQ(data_ptr, got_arrow_array->array.children[0]->buffers[1]);

  got_arrow_array = cudf::to_arrow_device(std::move(input));
  EXPECT_EQ(rmm::get_current_cuda_device().value(), got_arrow_array->device_id);
  EXPECT_EQ(ARROW_DEVICE_CUDA, got_arrow_array->device_type);
  ASSERT_CUDA_SUCCEEDED(
    cudaEventSynchronize(*reinterpret_cast<cudaEvent_t*>(got_arrow_array->sync_event)));

  EXPECT_EQ(data.size(), got_arrow_array->array.length);
  EXPECT_EQ(0, got_arrow_array->array.null_count);
  EXPECT_EQ(0, got_arrow_array->array.offset);
  EXPECT_EQ(1, got_arrow_array->array.n_children);
  EXPECT_EQ(nullptr, got_arrow_array->array.buffers[0]);

  EXPECT_EQ(data.size(), got_arrow_array->array.children[0]->length);
  EXPECT_EQ(0, got_arrow_array->array.children[0]->null_count);
  EXPECT_EQ(0, got_arrow_array->array.children[0]->offset);
  EXPECT_EQ(nullptr, got_arrow_array->array.children[0]->buffers[0]);
  EXPECT_EQ(data_ptr, got_arrow_array->array.children[0]->buffers[1]);
}

TYPED_TEST(ToArrowDeviceTestDurationsTest, DurationTable)
{
  using T = TypeParam;

  if (cudf::type_to_id<TypeParam>() == cudf::type_id::DURATION_DAYS) { return; }

  auto data = {T{1}, T{2}, T{3}, T{4}, T{5}, T{6}};
  auto col  = cudf::test::fixed_width_column_wrapper<T>(data);

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.emplace_back(col.release());
  cudf::table input(std::move(cols));

  nanoarrow::UniqueSchema expected_schema;
  ArrowSchemaInit(expected_schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(expected_schema.get(), 1));

  ArrowSchemaInit(expected_schema->children[0]);
  ArrowTimeUnit const arrow_unit = [&] {
    switch (cudf::type_to_id<TypeParam>()) {
      case cudf::type_id::DURATION_SECONDS: return NANOARROW_TIME_UNIT_SECOND;
      case cudf::type_id::DURATION_MILLISECONDS: return NANOARROW_TIME_UNIT_MILLI;
      case cudf::type_id::DURATION_MICROSECONDS: return NANOARROW_TIME_UNIT_MICRO;
      case cudf::type_id::DURATION_NANOSECONDS: return NANOARROW_TIME_UNIT_NANO;
      default: CUDF_FAIL("Unsupported duration unit in arrow");
    }
  }();
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeDateTime(
    expected_schema->children[0], NANOARROW_TYPE_DURATION, arrow_unit, nullptr));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(expected_schema->children[0], "a"));
  expected_schema->children[0]->flags = 0;

  auto got_arrow_schema =
    cudf::to_arrow_schema(input.view(), std::vector<cudf::column_metadata>{{"a"}});
  BaseArrowFixture::compare_schemas(expected_schema.get(), got_arrow_schema.get());

  auto data_ptr        = input.get_column(0).view().data<int64_t>();
  auto got_arrow_array = cudf::to_arrow_device(input.view());
  EXPECT_EQ(rmm::get_current_cuda_device().value(), got_arrow_array->device_id);
  EXPECT_EQ(ARROW_DEVICE_CUDA, got_arrow_array->device_type);
  ASSERT_CUDA_SUCCEEDED(
    cudaEventSynchronize(*reinterpret_cast<cudaEvent_t*>(got_arrow_array->sync_event)));

  EXPECT_EQ(data.size(), got_arrow_array->array.length);
  EXPECT_EQ(0, got_arrow_array->array.null_count);
  EXPECT_EQ(0, got_arrow_array->array.offset);
  EXPECT_EQ(1, got_arrow_array->array.n_children);
  EXPECT_EQ(nullptr, got_arrow_array->array.buffers[0]);

  EXPECT_EQ(data.size(), got_arrow_array->array.children[0]->length);
  EXPECT_EQ(0, got_arrow_array->array.children[0]->null_count);
  EXPECT_EQ(0, got_arrow_array->array.children[0]->offset);
  EXPECT_EQ(nullptr, got_arrow_array->array.children[0]->buffers[0]);
  EXPECT_EQ(data_ptr, got_arrow_array->array.children[0]->buffers[1]);

  got_arrow_array = cudf::to_arrow_device(std::move(input));
  EXPECT_EQ(rmm::get_current_cuda_device().value(), got_arrow_array->device_id);
  EXPECT_EQ(ARROW_DEVICE_CUDA, got_arrow_array->device_type);
  ASSERT_CUDA_SUCCEEDED(
    cudaEventSynchronize(*reinterpret_cast<cudaEvent_t*>(got_arrow_array->sync_event)));

  EXPECT_EQ(data.size(), got_arrow_array->array.length);
  EXPECT_EQ(0, got_arrow_array->array.null_count);
  EXPECT_EQ(0, got_arrow_array->array.offset);
  EXPECT_EQ(1, got_arrow_array->array.n_children);
  EXPECT_EQ(nullptr, got_arrow_array->array.buffers[0]);

  EXPECT_EQ(data.size(), got_arrow_array->array.children[0]->length);
  EXPECT_EQ(0, got_arrow_array->array.children[0]->null_count);
  EXPECT_EQ(0, got_arrow_array->array.children[0]->offset);
  EXPECT_EQ(nullptr, got_arrow_array->array.children[0]->buffers[0]);
  EXPECT_EQ(data_ptr, got_arrow_array->array.children[0]->buffers[1]);
}

TEST_F(ToArrowDeviceTest, NestedList)
{
  auto valids = cudf::test::iterators::nulls_at_multiples_of(3);
  auto col    = cudf::test::lists_column_wrapper<int64_t>(
    {{{{{1, 2}, valids}, {{3, 4}, valids}, {5}}, {{6}, {{7, 8, 9}, valids}}}, valids});

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.emplace_back(col.release());
  cudf::table input(std::move(cols));

  nanoarrow::UniqueSchema expected_schema;
  ArrowSchemaInit(expected_schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(expected_schema.get(), 1));

  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(expected_schema->children[0], NANOARROW_TYPE_LIST));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(expected_schema->children[0], "a"));
  expected_schema->children[0]->flags = ARROW_FLAG_NULLABLE;

  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(expected_schema->children[0]->children[0], NANOARROW_TYPE_LIST));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(expected_schema->children[0]->children[0], "element"));
  expected_schema->children[0]->children[0]->flags = 0;

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(
    expected_schema->children[0]->children[0]->children[0], NANOARROW_TYPE_INT64));
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaSetName(expected_schema->children[0]->children[0]->children[0], "element"));
  expected_schema->children[0]->children[0]->children[0]->flags = ARROW_FLAG_NULLABLE;

  auto got_arrow_schema =
    cudf::to_arrow_schema(input.view(), std::vector<cudf::column_metadata>{{"a"}});
  compare_schemas(expected_schema.get(), got_arrow_schema.get());

  nanoarrow::UniqueArray expected_array;
  EXPECT_EQ(NANOARROW_OK,
            ArrowArrayInitFromSchema(expected_array.get(), expected_schema.get(), nullptr));
  expected_array->length = input.num_rows();
  auto top_list          = expected_array->children[0];
  cudf::lists_column_view lview{input.get_column(0).view()};
  populate_list_from_col(top_list, lview);
  cudf::lists_column_view nested_view{lview.child()};
  populate_list_from_col(top_list->children[0], nested_view);
  populate_from_col<int64_t>(top_list->children[0]->children[0], nested_view.child());

  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(expected_array.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr));

  auto got_arrow_array = cudf::to_arrow_device(input.view());
  EXPECT_EQ(rmm::get_current_cuda_device().value(), got_arrow_array->device_id);
  EXPECT_EQ(ARROW_DEVICE_CUDA, got_arrow_array->device_type);
  ASSERT_CUDA_SUCCEEDED(
    cudaEventSynchronize(*reinterpret_cast<cudaEvent_t*>(got_arrow_array->sync_event)));
  compare_arrays(expected_schema.get(), expected_array.get(), &got_arrow_array->array);

  got_arrow_array = cudf::to_arrow_device(std::move(input));
  EXPECT_EQ(rmm::get_current_cuda_device().value(), got_arrow_array->device_id);
  EXPECT_EQ(ARROW_DEVICE_CUDA, got_arrow_array->device_type);
  compare_arrays(expected_schema.get(), expected_array.get(), &got_arrow_array->array);
}

TEST_F(ToArrowDeviceTest, StructColumn)
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
  std::vector<std::unique_ptr<cudf::column>> table_cols;
  table_cols.emplace_back(struct_col.release());
  cudf::table input(std::move(table_cols));

  // Create name metadata
  auto sub_metadata          = cudf::column_metadata{"struct"};
  sub_metadata.children_meta = {{"string2"}, {"integral2"}};
  auto metadata              = cudf::column_metadata{"a"};
  metadata.children_meta     = {{"string"}, {"integral"}, {"bool"}, {"nested_list"}, sub_metadata};

  nanoarrow::UniqueSchema expected_schema;
  ArrowSchemaInit(expected_schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(expected_schema.get(), 1));

  ArrowSchemaInit(expected_schema->children[0]);
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(expected_schema->children[0], 5));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(expected_schema->children[0], "a"));
  expected_schema->children[0]->flags = 0;

  auto child = expected_schema->children[0];
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

  auto got_arrow_schema =
    cudf::to_arrow_schema(input.view(), std::vector<cudf::column_metadata>{metadata});
  compare_schemas(expected_schema.get(), got_arrow_schema.get());

  nanoarrow::UniqueArray expected_array;
  NANOARROW_THROW_NOT_OK(
    ArrowArrayInitFromSchema(expected_array.get(), expected_schema.get(), nullptr));

  expected_array->length = input.num_rows();

  auto array_a        = expected_array->children[0];
  auto view_a         = input.view().column(0);
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
    ArrowArrayFinishBuilding(expected_array.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr));

  auto got_arrow_array = cudf::to_arrow_device(input.view());
  EXPECT_EQ(rmm::get_current_cuda_device().value(), got_arrow_array->device_id);
  EXPECT_EQ(ARROW_DEVICE_CUDA, got_arrow_array->device_type);
  ASSERT_CUDA_SUCCEEDED(
    cudaEventSynchronize(*reinterpret_cast<cudaEvent_t*>(got_arrow_array->sync_event)));
  compare_arrays(expected_schema.get(), expected_array.get(), &got_arrow_array->array);

  got_arrow_array = cudf::to_arrow_device(std::move(input));
  EXPECT_EQ(rmm::get_current_cuda_device().value(), got_arrow_array->device_id);
  EXPECT_EQ(ARROW_DEVICE_CUDA, got_arrow_array->device_type);
  ASSERT_CUDA_SUCCEEDED(
    cudaEventSynchronize(*reinterpret_cast<cudaEvent_t*>(got_arrow_array->sync_event)));
  compare_arrays(expected_schema.get(), expected_array.get(), &got_arrow_array->array);
}

template <typename T>
using fp_wrapper = cudf::test::fixed_point_column_wrapper<T>;

TEST_F(ToArrowDeviceTest, FixedPoint32Table)
{
  using namespace numeric;

  for (auto const scale : {6, 4, 2, 0, -1, -3, -5}) {
    auto const expect_data = std::vector<int32_t>{-1000, 2400, -3456, 4650, 5154, 6800};
    auto col = fp_wrapper<int32_t>({-1000, 2400, -3456, 4650, 5154, 6800}, scale_type{scale});
    std::vector<std::unique_ptr<cudf::column>> table_cols;
    table_cols.emplace_back(col.release());
    auto input = cudf::table(std::move(table_cols));

    nanoarrow::UniqueSchema expected_schema;
    ArrowSchemaInit(expected_schema.get());
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(expected_schema.get(), 1));
    ArrowSchemaInit(expected_schema->children[0]);
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeDecimal(expected_schema->children[0],
                                                     NANOARROW_TYPE_DECIMAL32,
                                                     get_decimal_precision<int32_t>(),
                                                     -scale));
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(expected_schema->children[0], "a"));
    expected_schema->children[0]->flags = 0;

    auto got_arrow_schema =
      cudf::to_arrow_schema(input.view(), std::vector<cudf::column_metadata>{{"a"}});
    compare_schemas(expected_schema.get(), got_arrow_schema.get());

    nanoarrow::UniqueArray expected_array;
    NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromSchema(expected_array.get(), expected_schema.get(), nullptr));
    expected_array->length = input.num_rows();

    populate_from_col<int64_t>(expected_array->children[0], input.view().column(0));
    NANOARROW_THROW_NOT_OK(
      ArrowArrayFinishBuilding(expected_array.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr));

    auto got_arrow_array = cudf::to_arrow_device(input.view());
    ASSERT_EQ(rmm::get_current_cuda_device().value(), got_arrow_array->device_id);
    ASSERT_EQ(ARROW_DEVICE_CUDA, got_arrow_array->device_type);
    ASSERT_CUDA_SUCCEEDED(
      cudaEventSynchronize(*reinterpret_cast<cudaEvent_t*>(got_arrow_array->sync_event)));
    compare_arrays(expected_schema.get(), expected_array.get(), &got_arrow_array->array);

    got_arrow_array = cudf::to_arrow_device(std::move(input));
    ASSERT_EQ(rmm::get_current_cuda_device().value(), got_arrow_array->device_id);
    ASSERT_EQ(ARROW_DEVICE_CUDA, got_arrow_array->device_type);
    ASSERT_CUDA_SUCCEEDED(
      cudaEventSynchronize(*reinterpret_cast<cudaEvent_t*>(got_arrow_array->sync_event)));
    compare_arrays(expected_schema.get(), expected_array.get(), &got_arrow_array->array);
  }
}

TEST_F(ToArrowDeviceTest, FixedPoint64Table)
{
  using namespace numeric;

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const expect_data = std::vector<int64_t>{-1, -1, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0};
    auto col               = fp_wrapper<int64_t>({-1, 2, 3, 4, 5, 6}, scale_type{scale});
    std::vector<std::unique_ptr<cudf::column>> table_cols;
    table_cols.emplace_back(col.release());
    auto input = cudf::table(std::move(table_cols));

    nanoarrow::UniqueSchema expected_schema;
    ArrowSchemaInit(expected_schema.get());
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(expected_schema.get(), 1));
    ArrowSchemaInit(expected_schema->children[0]);
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeDecimal(
      expected_schema->children[0], NANOARROW_TYPE_DECIMAL64, 18, -scale));
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(expected_schema->children[0], "a"));
    expected_schema->children[0]->flags = 0;

    auto got_arrow_schema =
      cudf::to_arrow_schema(input.view(), std::vector<cudf::column_metadata>{{"a"}});
    compare_schemas(expected_schema.get(), got_arrow_schema.get());

    nanoarrow::UniqueArray expected_array;
    NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromSchema(expected_array.get(), expected_schema.get(), nullptr));
    expected_array->length = input.num_rows();

    populate_from_col<int32_t>(expected_array->children[0], input.view().column(0));
    NANOARROW_THROW_NOT_OK(
      ArrowArrayFinishBuilding(expected_array.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr));

    auto got_arrow_array = cudf::to_arrow_device(input.view());
    ASSERT_EQ(rmm::get_current_cuda_device().value(), got_arrow_array->device_id);
    ASSERT_EQ(ARROW_DEVICE_CUDA, got_arrow_array->device_type);
    ASSERT_CUDA_SUCCEEDED(
      cudaEventSynchronize(*reinterpret_cast<cudaEvent_t*>(got_arrow_array->sync_event)));
    compare_arrays(expected_schema.get(), expected_array.get(), &got_arrow_array->array);

    got_arrow_array = cudf::to_arrow_device(std::move(input));
    ASSERT_EQ(rmm::get_current_cuda_device().value(), got_arrow_array->device_id);
    ASSERT_EQ(ARROW_DEVICE_CUDA, got_arrow_array->device_type);
    ASSERT_CUDA_SUCCEEDED(
      cudaEventSynchronize(*reinterpret_cast<cudaEvent_t*>(got_arrow_array->sync_event)));
    compare_arrays(expected_schema.get(), expected_array.get(), &got_arrow_array->array);
  }
}

TEST_F(ToArrowDeviceTest, FixedPoint128Table)
{
  using namespace numeric;

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const expect_data = std::vector<__int128_t>{-1, 2, 3, 4, 5, 6};
    auto col               = fp_wrapper<__int128_t>({-1, 2, 3, 4, 5, 6}, scale_type{scale});
    std::vector<std::unique_ptr<cudf::column>> table_cols;
    table_cols.emplace_back(col.release());
    auto input = cudf::table(std::move(table_cols));

    nanoarrow::UniqueSchema expected_schema;
    ArrowSchemaInit(expected_schema.get());
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(expected_schema.get(), 1));
    ArrowSchemaInit(expected_schema->children[0]);
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeDecimal(expected_schema->children[0],
                                                     NANOARROW_TYPE_DECIMAL128,
                                                     get_decimal_precision<__int128_t>(),
                                                     -scale));
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(expected_schema->children[0], "a"));
    expected_schema->children[0]->flags = 0;

    auto got_arrow_schema =
      cudf::to_arrow_schema(input.view(), std::vector<cudf::column_metadata>{{"a"}});
    compare_schemas(expected_schema.get(), got_arrow_schema.get());

    nanoarrow::UniqueArray expected_array;
    NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromSchema(expected_array.get(), expected_schema.get(), nullptr));
    expected_array->length = input.num_rows();

    populate_from_col<__int128_t>(expected_array->children[0], input.view().column(0));
    NANOARROW_THROW_NOT_OK(
      ArrowArrayFinishBuilding(expected_array.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr));

    auto got_arrow_array = cudf::to_arrow_device(input.view());
    EXPECT_EQ(rmm::get_current_cuda_device().value(), got_arrow_array->device_id);
    EXPECT_EQ(ARROW_DEVICE_CUDA, got_arrow_array->device_type);
    ASSERT_CUDA_SUCCEEDED(
      cudaEventSynchronize(*reinterpret_cast<cudaEvent_t*>(got_arrow_array->sync_event)));
    compare_arrays(expected_schema.get(), expected_array.get(), &got_arrow_array->array);

    got_arrow_array = cudf::to_arrow_device(std::move(input));
    EXPECT_EQ(rmm::get_current_cuda_device().value(), got_arrow_array->device_id);
    EXPECT_EQ(ARROW_DEVICE_CUDA, got_arrow_array->device_type);
    ASSERT_CUDA_SUCCEEDED(
      cudaEventSynchronize(*reinterpret_cast<cudaEvent_t*>(got_arrow_array->sync_event)));
    compare_arrays(expected_schema.get(), expected_array.get(), &got_arrow_array->array);
  }
}
