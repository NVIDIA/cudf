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
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/interop.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cuda/iterator>

#include <numeric>

using vector_of_columns = std::vector<std::unique_ptr<cudf::column>>;

struct BaseToArrowHostFixture : public cudf::test::BaseFixture {
  template <typename T>
  void compare_subset(ArrowArrayView const* expected,
                      int64_t start_offset_expected,
                      ArrowArrayView const* actual,
                      int64_t start_offset_actual,
                      int64_t length)
    requires(cudf::is_fixed_width<T>() and !std::is_same_v<T, bool>)
  {
    for (int64_t i = 0; i < length; ++i) {
      bool const is_null = ArrowArrayViewIsNull(expected, start_offset_expected + i);
      EXPECT_EQ(is_null, ArrowArrayViewIsNull(actual, start_offset_actual + i));
      if (is_null) continue;

      auto const expected_val = ArrowArrayViewGetIntUnsafe(expected, start_offset_expected + i);
      auto const actual_val   = ArrowArrayViewGetIntUnsafe(actual, start_offset_actual + i);

      EXPECT_EQ(expected_val, actual_val);
    }
  }

  template <typename T>
  void compare_subset(ArrowArrayView const* expected,
                      int64_t start_offset_expected,
                      ArrowArrayView const* actual,
                      int64_t start_offset_actual,
                      int64_t length)
    requires(std::is_same_v<T, cudf::string_view>)
  {
    for (int64_t i = 0; i < length; ++i) {
      bool const is_null = ArrowArrayViewIsNull(expected, start_offset_expected + i);
      EXPECT_EQ(is_null, ArrowArrayViewIsNull(actual, start_offset_actual + i));
      if (is_null) continue;

      auto const expected_view = ArrowArrayViewGetBytesUnsafe(expected, start_offset_expected + i);
      auto const actual_view   = ArrowArrayViewGetBytesUnsafe(actual, start_offset_actual + i);

      EXPECT_EQ(expected_view.size_bytes, actual_view.size_bytes);
      EXPECT_TRUE(
        0 == std::memcmp(expected_view.data.data, actual_view.data.data, expected_view.size_bytes));
    }
  }

  void compare_child_subset(ArrowArrayView const* expected,
                            int64_t exp_start_offset,
                            ArrowArrayView const* actual,
                            int64_t act_start_offset,
                            int64_t length)
  {
    EXPECT_EQ(expected->storage_type, actual->storage_type);
    EXPECT_EQ(expected->n_children, actual->n_children);

    switch (expected->storage_type) {
      case NANOARROW_TYPE_LIST:
        for (int64_t i = 0; i < length; ++i) {
          auto const expected_start = exp_start_offset + i;
          auto const actual_start   = act_start_offset + i;

          // ArrowArrayViewIsNull accounts for the array offset, so we can properly
          // compare the validity of indexes
          bool const is_null = ArrowArrayViewIsNull(expected, expected_start);
          EXPECT_EQ(is_null, ArrowArrayViewIsNull(actual, actual_start));
          if (is_null) continue;

          // ArrowArrayViewListChildOffset does not account for array offset, so we need
          // to add the offset to the index in order to get the correct offset into the list
          int64_t const start_offset_expected =
            ArrowArrayViewListChildOffset(expected, expected->offset + expected_start);
          int64_t const start_offset_actual =
            ArrowArrayViewListChildOffset(actual, actual->offset + actual_start);

          int64_t const end_offset_expected =
            ArrowArrayViewListChildOffset(expected, expected->offset + expected_start + 1);
          int64_t const end_offset_actual =
            ArrowArrayViewListChildOffset(actual, actual->offset + actual_start + 1);

          // verify the list lengths are the same
          EXPECT_EQ(end_offset_expected - start_offset_expected,
                    end_offset_actual - start_offset_actual);
          // compare the list values
          compare_child_subset(expected->children[0],
                               start_offset_expected,
                               actual->children[0],
                               start_offset_actual,
                               end_offset_expected - start_offset_expected);
        }
        break;
      case NANOARROW_TYPE_STRUCT:
        for (int64_t i = 0; i < length; ++i) {
          SCOPED_TRACE("idx: " + std::to_string(i));
          auto const expected_start = exp_start_offset + i;
          auto const actual_start   = act_start_offset + i;

          bool const is_null = ArrowArrayViewIsNull(expected, expected_start);
          EXPECT_EQ(is_null, ArrowArrayViewIsNull(actual, actual_start));
          if (is_null) continue;

          for (int64_t child = 0; child < expected->n_children; ++child) {
            SCOPED_TRACE("child: " + std::to_string(child));
            compare_child_subset(expected->children[child],
                                 expected_start + expected->offset,
                                 actual->children[child],
                                 actual_start + actual->offset,
                                 1);
          }
        }
        break;
      case NANOARROW_TYPE_STRING:
      case NANOARROW_TYPE_LARGE_STRING:
      case NANOARROW_TYPE_BINARY:
      case NANOARROW_TYPE_LARGE_BINARY:
        compare_subset<cudf::string_view>(
          expected, exp_start_offset, actual, act_start_offset, length);
        break;
      default:
        compare_subset<int64_t>(expected, exp_start_offset, actual, act_start_offset, length);
        break;
    }
  }

  void compare_arrays(ArrowArrayView const* expected, ArrowArrayView const* actual)
  {
    EXPECT_EQ(expected->length, actual->length);
    EXPECT_EQ(expected->null_count, actual->null_count);
    EXPECT_EQ(expected->offset, actual->offset);
    EXPECT_EQ(expected->n_children, actual->n_children);
    EXPECT_EQ(expected->storage_type, actual->storage_type);

    // cudf automatically pushes down nulls and purges non-empty, non-zero nulls
    // from the children columns. So while we can memcmp the buffers for top
    // level arrays, we need to do an "equivalence" comparison for nested
    // arrays (lists and structs) by checking each index for null and skipping
    // comparisons for children if null.
    switch (expected->storage_type) {
      case NANOARROW_TYPE_STRUCT:
        // if we're a struct with no children, then we just skip
        // attempting to compare the children
        if (expected->n_children == 0) {
          EXPECT_EQ(nullptr, actual->children);
          break;
        }
        // otherwise we can fallthrough and do the same thing we do for lists
      case NANOARROW_TYPE_LIST:
        compare_child_subset(expected, 0, actual, 0, expected->length);
        break;
      default:
        for (int64_t i = 0; i < actual->array->n_buffers; ++i) {
          SCOPED_TRACE("buffer " + std::to_string(i));
          auto expected_buf = expected->buffer_views[i];
          auto actual_buf   = actual->buffer_views[i];

          EXPECT_TRUE(0 == std::memcmp(expected_buf.data.data,
                                       actual_buf.data.data,
                                       expected_buf.size_bytes));
        }
    }

    if (expected->dictionary != nullptr) {
      EXPECT_NE(nullptr, actual->dictionary);
      SCOPED_TRACE("dictionary");
      compare_arrays(expected->dictionary, actual->dictionary);
    } else {
      EXPECT_EQ(nullptr, actual->dictionary);
    }
  }
};

struct ToArrowHostDeviceTest : public BaseToArrowHostFixture {};

TEST_F(ToArrowHostDeviceTest, EmptyTable)
{
  auto [tbl, schema, arr] = get_nanoarrow_host_tables(0);

  auto got_arrow_host = cudf::to_arrow_host(tbl->view());
  EXPECT_EQ(ARROW_DEVICE_CPU, got_arrow_host->device_type);
  EXPECT_EQ(-1, got_arrow_host->device_id);
  EXPECT_EQ(nullptr, got_arrow_host->sync_event);

  ArrowArrayView expected, actual;
  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&expected, schema.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&expected, arr.get(), nullptr));

  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&actual, schema.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
  compare_arrays(&expected, &actual);

  ArrowArrayViewReset(&expected);
  ArrowArrayViewReset(&actual);
}

TEST_F(ToArrowHostDeviceTest, Nullable)
{
  auto const input = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3, 4}, {1, 1, 1, 1});
  auto const tv    = cudf::table_view{{input}};

  auto schema       = cudf::to_arrow_schema(tv, cudf::interop::get_table_metadata(tv));
  auto arrow        = cudf::to_arrow_host(tv);
  auto roundtripped = cudf::from_arrow_host(schema.get(), arrow.get());
  auto const after  = roundtripped->view().column(0);

  EXPECT_TRUE(after.nullable());
}

TEST_F(ToArrowHostDeviceTest, EmptyDictionary)
{
  auto empty = cudf::make_empty_column(cudf::type_id::DICTIONARY32);

  auto got_arrow_host = cudf::to_arrow_host(cudf::table_view({empty->view()}));
  ASSERT_EQ(got_arrow_host->array.n_children, 1);
  auto dictionary = got_arrow_host->array.children[0]->dictionary;
  ASSERT_NE(dictionary, nullptr);
  EXPECT_EQ(dictionary->n_children, 0);
}

TEST_F(ToArrowHostDeviceTest, DateTimeTable)
{
  auto data = std::initializer_list<int64_t>{1, 2, 3, 4, 5, 6};
  auto col =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>(data);
  cudf::table_view input_view({col});

  nanoarrow::UniqueSchema expected_schema;
  ArrowSchemaInit(expected_schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(expected_schema.get(), 1));
  ArrowSchemaInit(expected_schema->children[0]);
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeDateTime(
    expected_schema->children[0], NANOARROW_TYPE_TIMESTAMP, NANOARROW_TIME_UNIT_MILLI, nullptr));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(expected_schema->children[0], "a"));
  expected_schema->children[0]->flags = 0;

  auto got_arrow_host = cudf::to_arrow_host(input_view);
  EXPECT_EQ(ARROW_DEVICE_CPU, got_arrow_host->device_type);
  EXPECT_EQ(-1, got_arrow_host->device_id);
  EXPECT_EQ(nullptr, got_arrow_host->sync_event);

  ArrowArrayView expected, actual;
  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&expected, expected_schema.get(), nullptr));
  expected.length              = data.size();
  expected.children[0]->length = data.size();
  ArrowArrayViewSetLength(expected.children[0], data.size());
  expected.children[0]->buffer_views[0].data.data  = nullptr;
  expected.children[0]->buffer_views[0].size_bytes = 0;
  expected.children[0]->buffer_views[1].data.data  = data.begin();

  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&actual, expected_schema.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
  compare_arrays(&expected, &actual);
  ArrowArrayViewReset(&actual);

  got_arrow_host = cudf::to_arrow_host(input_view.column(0));
  NANOARROW_THROW_NOT_OK(
    ArrowArrayViewInitFromSchema(&actual, expected_schema->children[0], nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
  BaseToArrowHostFixture::compare_arrays(expected.children[0], &actual);
  ArrowArrayViewReset(&actual);

  ArrowArrayViewReset(&expected);
  ArrowArrayViewReset(&actual);
}

TEST_F(ToArrowHostDeviceTest, StringView)
{
  ArrowSchema schema;
  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(&schema, NANOARROW_TYPE_STRING_VIEW));

  auto data = cudf::test::strings_column_wrapper(
    {"hello", "worldy", "much longer string", "", "another even longer string", "", "other string"},
    {1, 1, 1, 0, 1, 1, 1});
  auto result   = cudf::to_arrow_host_stringview(cudf::strings_column_view(data));
  auto expected = cudf::from_arrow_column(&schema, &result->array);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected->view(), data);

  auto sliced = cudf::split(data, {3}).front();
  result      = cudf::to_arrow_host_stringview(cudf::strings_column_view(sliced));
  expected    = cudf::from_arrow_column(&schema, &result->array);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected->view(), sliced);

  data =
    cudf::test::strings_column_wrapper({"all of", "these", "strings", "will", "fit", "inline"});
  result   = cudf::to_arrow_host_stringview(cudf::strings_column_view(data));
  expected = cudf::from_arrow_column(&schema, &result->array);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected->view(), data);

  data = cudf::test::strings_column_wrapper(
    {"all of these strings", "are longer and will", "not fit as inline ones"});
  result   = cudf::to_arrow_host_stringview(cudf::strings_column_view(data));
  expected = cudf::from_arrow_column(&schema, &result->array);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected->view(), data);

  data     = cudf::test::strings_column_wrapper();  // empty column test
  result   = cudf::to_arrow_host_stringview(cudf::strings_column_view(data));
  expected = cudf::from_arrow_column(&schema, &result->array);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected->view(), data);
}

template <typename T>
struct ToArrowHostDeviceTestDurationsTest : public BaseToArrowHostFixture {};

TYPED_TEST_SUITE(ToArrowHostDeviceTestDurationsTest, cudf::test::DurationTypes);

TYPED_TEST(ToArrowHostDeviceTestDurationsTest, DurationTable)
{
  using T = TypeParam;

  if (cudf::type_to_id<TypeParam>() == cudf::type_id::DURATION_DAYS) { return; }

  auto data = {T{1}, T{2}, T{3}, T{4}, T{5}, T{6}};
  auto col  = cudf::test::fixed_width_column_wrapper<T>(data);

  cudf::table_view input_view({col});

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

  auto got_arrow_host = cudf::to_arrow_host(input_view);
  EXPECT_EQ(ARROW_DEVICE_CPU, got_arrow_host->device_type);
  EXPECT_EQ(-1, got_arrow_host->device_id);
  EXPECT_EQ(nullptr, got_arrow_host->sync_event);

  ArrowArrayView expected, actual;
  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&expected, expected_schema.get(), nullptr));

  expected.length              = data.size();
  expected.children[0]->length = data.size();
  ArrowArrayViewSetLength(expected.children[0], data.size());
  expected.children[0]->buffer_views[0].data.data  = nullptr;
  expected.children[0]->buffer_views[0].size_bytes = 0;
  expected.children[0]->buffer_views[1].data.data  = data.begin();

  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&actual, expected_schema.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
  BaseToArrowHostFixture::compare_arrays(&expected, &actual);
  ArrowArrayViewReset(&actual);

  got_arrow_host = cudf::to_arrow_host(input_view.column(0));
  NANOARROW_THROW_NOT_OK(
    ArrowArrayViewInitFromSchema(&actual, expected_schema->children[0], nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
  BaseToArrowHostFixture::compare_arrays(expected.children[0], &actual);
  ArrowArrayViewReset(&actual);

  ArrowArrayViewReset(&expected);
}

TEST_F(ToArrowHostDeviceTest, NestedList)
{
  auto valids = cudf::test::iterators::nulls_at_multiples_of(3);
  auto col    = cudf::test::lists_column_wrapper<int64_t>(
    {{{{{1, 2}, valids}, {{3, 4}, valids}, {5}}, {{6}, {{7, 8, 9}, valids}}}, valids});
  cudf::table_view input_view({col});

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

  auto got_arrow_host = cudf::to_arrow_host(input_view);
  EXPECT_EQ(ARROW_DEVICE_CPU, got_arrow_host->device_type);
  EXPECT_EQ(-1, got_arrow_host->device_id);
  EXPECT_EQ(nullptr, got_arrow_host->sync_event);

  auto list_arr = get_nanoarrow_list_array<int64_t>({6, 7, 8, 9}, {0, 1, 4}, {1, 0, 1, 1});
  std::vector<int32_t> offset{0, 0, 2};

  ArrowBitmap mask;
  ArrowBitmapInit(&mask);
  NANOARROW_THROW_NOT_OK(ArrowBitmapReserve(&mask, 2));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(&mask, 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(&mask, 1, 1));

  nanoarrow::UniqueArray expected_arr;
  EXPECT_EQ(NANOARROW_OK,
            ArrowArrayInitFromSchema(expected_arr.get(), expected_schema.get(), nullptr));
  expected_arr->length     = input_view.num_rows();
  expected_arr->null_count = 0;

  ArrowArraySetValidityBitmap(expected_arr->children[0], &mask);
  expected_arr->children[0]->length     = input_view.num_rows();
  expected_arr->children[0]->null_count = 1;
  auto offset_buf                       = ArrowArrayBuffer(expected_arr->children[0], 1);
  EXPECT_EQ(
    NANOARROW_OK,
    ArrowBufferAppend(
      offset_buf, reinterpret_cast<void const*>(offset.data()), offset.size() * sizeof(int32_t)));
  list_arr.move(expected_arr->children[0]->children[0]);
  NANOARROW_THROW_NOT_OK(ArrowArrayFinishBuildingDefault(expected_arr.get(), nullptr));

  ArrowArrayView expected, actual;
  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&expected, expected_schema.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&expected, expected_arr.get(), nullptr));

  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&actual, expected_schema.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
  compare_arrays(&expected, &actual);
  ArrowArrayViewReset(&actual);

  got_arrow_host = cudf::to_arrow_host(input_view.column(0));
  NANOARROW_THROW_NOT_OK(
    ArrowArrayViewInitFromSchema(&actual, expected_schema->children[0], nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
  compare_arrays(expected.children[0], &actual);
  ArrowArrayViewReset(&actual);

  ArrowArrayViewReset(&expected);
}

TEST_F(ToArrowHostDeviceTest, StructColumn)
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

  // create nanoarrow table
  // first our underlying arrays
  std::vector<std::string> str{"Samuel Vimes", "Carrot Ironfoundersson", "Angua von Überwald"};
  std::vector<std::string> str2{"CUDF", "ROCKS", "EVERYWHERE"};
  auto str_array  = get_nanoarrow_array<cudf::string_view>(str);
  auto int_array  = get_nanoarrow_array<int32_t>({48, 27, 25});
  auto str2_array = get_nanoarrow_array<cudf::string_view>(str2, {0, 1, 0});
  // struct null will get pushed down and superimposed on this array
  auto int2_array = get_nanoarrow_array<int32_t, uint8_t>({12, 24, 47}, {1, 0, 0});
  auto bool_array = get_nanoarrow_array<bool>({true, true, false});
  auto list_arr =
    get_nanoarrow_list_array<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 2, 4, 5, 6, 7, 9});
  std::vector<int32_t> offset{0, 3, 4, 6};

  nanoarrow::UniqueArray expected_arr;
  NANOARROW_THROW_NOT_OK(
    ArrowArrayInitFromSchema(expected_arr.get(), expected_schema.get(), nullptr));
  expected_arr->length = input_view.num_rows();

  auto array_a        = expected_arr->children[0];
  auto view_a         = input_view.column(0);
  array_a->length     = view_a.size();
  array_a->null_count = view_a.null_count();

  str_array.move(array_a->children[0]);
  int_array.move(array_a->children[1]);
  bool_array.move(array_a->children[2]);

  array_a->children[3]->length     = input_view.num_rows();
  array_a->children[3]->null_count = 0;

  auto offset_buf = ArrowArrayBuffer(array_a->children[3], 1);
  EXPECT_EQ(
    NANOARROW_OK,
    ArrowBufferAppend(
      offset_buf, reinterpret_cast<void const*>(offset.data()), offset.size() * sizeof(int32_t)));
  list_arr.move(array_a->children[3]->children[0]);

  ArrowBitmap mask;
  ArrowBitmapInit(&mask);
  NANOARROW_THROW_NOT_OK(ArrowBitmapReserve(&mask, 3));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(&mask, 1, 2));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(&mask, 0, 1));

  auto array_struct = array_a->children[4];
  auto view_struct  = view_a.child(4);
  ArrowArraySetValidityBitmap(array_struct, &mask);
  array_struct->null_count = view_struct.null_count();
  array_struct->length     = view_struct.size();

  str2_array.move(array_struct->children[0]);
  int2_array.move(array_struct->children[1]);

  NANOARROW_THROW_NOT_OK(ArrowArrayFinishBuildingDefault(expected_arr.get(), nullptr));

  auto got_arrow_host = cudf::to_arrow_host(input_view);
  EXPECT_EQ(ARROW_DEVICE_CPU, got_arrow_host->device_type);
  EXPECT_EQ(-1, got_arrow_host->device_id);
  EXPECT_EQ(nullptr, got_arrow_host->sync_event);

  ArrowArrayView expected, actual;
  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&expected, expected_schema.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&expected, expected_arr.get(), nullptr));

  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&actual, expected_schema.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
  compare_arrays(&expected, &actual);
  ArrowArrayViewReset(&actual);

  got_arrow_host = cudf::to_arrow_host(input_view.column(0));
  NANOARROW_THROW_NOT_OK(
    ArrowArrayViewInitFromSchema(&actual, expected_schema->children[0], nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
  compare_arrays(expected.children[0], &actual);
  ArrowArrayViewReset(&actual);

  ArrowArrayViewReset(&expected);
}

template <typename T>
using fp_wrapper = cudf::test::fixed_point_column_wrapper<T>;

TEST_F(ToArrowHostDeviceTest, FixedPoint32Table)
{
  using namespace numeric;

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const col   = fp_wrapper<int32_t>({-1, 2, 3, 4, 5, 6}, scale_type{scale});
    auto const input = cudf::table_view({col});

    auto const data = std::vector<int32_t>{-1, 2, 3, 4, 5, 6};
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

    nanoarrow::UniqueArray expected_array;
    NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromSchema(expected_array.get(), expected_schema.get(), nullptr));
    expected_array->length = input.num_rows();

    get_nanoarrow_array<int32_t>(data).move(expected_array->children[0]);
    NANOARROW_THROW_NOT_OK(ArrowArrayFinishBuildingDefault(expected_array.get(), nullptr));

    auto got_arrow_host = cudf::to_arrow_host(input);
    EXPECT_EQ(ARROW_DEVICE_CPU, got_arrow_host->device_type);
    EXPECT_EQ(-1, got_arrow_host->device_id);
    EXPECT_EQ(nullptr, got_arrow_host->sync_event);

    ArrowArrayView expected, actual;
    NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&expected, expected_schema.get(), nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&expected, expected_array.get(), nullptr));

    NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&actual, expected_schema.get(), nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
    compare_arrays(&expected, &actual);
    ArrowArrayViewReset(&actual);

    got_arrow_host = cudf::to_arrow_host(input.column(0));
    NANOARROW_THROW_NOT_OK(
      ArrowArrayViewInitFromSchema(&actual, expected_schema->children[0], nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
    compare_arrays(expected.children[0], &actual);
    ArrowArrayViewReset(&actual);

    ArrowArrayViewReset(&expected);
  }
}

TEST_F(ToArrowHostDeviceTest, FixedPoint64Table)
{
  using namespace numeric;

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const col   = fp_wrapper<int64_t>({-1, 2, 3, 4, 5, 6}, scale_type{scale});
    auto const input = cudf::table_view({col});

    auto const data = std::vector<int64_t>{-1, 2, 3, 4, 5, 6};
    nanoarrow::UniqueSchema expected_schema;
    ArrowSchemaInit(expected_schema.get());
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(expected_schema.get(), 1));
    ArrowSchemaInit(expected_schema->children[0]);
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeDecimal(expected_schema->children[0],
                                                     NANOARROW_TYPE_DECIMAL64,
                                                     get_decimal_precision<int64_t>(),
                                                     -scale));
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(expected_schema->children[0], "a"));
    expected_schema->children[0]->flags = 0;

    nanoarrow::UniqueArray expected_array;
    NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromSchema(expected_array.get(), expected_schema.get(), nullptr));
    expected_array->length = input.num_rows();

    get_nanoarrow_array<int64_t>(data).move(expected_array->children[0]);
    NANOARROW_THROW_NOT_OK(ArrowArrayFinishBuildingDefault(expected_array.get(), nullptr));

    auto got_arrow_host = cudf::to_arrow_host(input);
    EXPECT_EQ(ARROW_DEVICE_CPU, got_arrow_host->device_type);
    EXPECT_EQ(-1, got_arrow_host->device_id);
    EXPECT_EQ(nullptr, got_arrow_host->sync_event);

    ArrowArrayView expected, actual;
    NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&expected, expected_schema.get(), nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&expected, expected_array.get(), nullptr));

    NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&actual, expected_schema.get(), nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
    compare_arrays(&expected, &actual);
    ArrowArrayViewReset(&actual);

    got_arrow_host = cudf::to_arrow_host(input.column(0));
    NANOARROW_THROW_NOT_OK(
      ArrowArrayViewInitFromSchema(&actual, expected_schema->children[0], nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
    compare_arrays(expected.children[0], &actual);
    ArrowArrayViewReset(&actual);

    ArrowArrayViewReset(&expected);
  }
}

TEST_F(ToArrowHostDeviceTest, FixedPoint128Table)
{
  using namespace numeric;

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const col   = fp_wrapper<__int128_t>({-1, 2, 3, 4, 5, 6}, scale_type{scale});
    auto const input = cudf::table_view({col});

    auto const data = std::vector<__int128_t>{-1, 2, 3, 4, 5, 6};

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

    nanoarrow::UniqueArray expected_array;
    NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromSchema(expected_array.get(), expected_schema.get(), nullptr));
    expected_array->length = input.num_rows();

    get_nanoarrow_array<__int128_t>(data).move(expected_array->children[0]);
    NANOARROW_THROW_NOT_OK(ArrowArrayFinishBuildingDefault(expected_array.get(), nullptr));

    auto got_arrow_host = cudf::to_arrow_host(input);
    EXPECT_EQ(ARROW_DEVICE_CPU, got_arrow_host->device_type);
    EXPECT_EQ(-1, got_arrow_host->device_id);
    EXPECT_EQ(nullptr, got_arrow_host->sync_event);

    ArrowArrayView expected, actual;
    NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&expected, expected_schema.get(), nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&expected, expected_array.get(), nullptr));

    NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&actual, expected_schema.get(), nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
    compare_arrays(&expected, &actual);
    ArrowArrayViewReset(&actual);

    got_arrow_host = cudf::to_arrow_host(input.column(0));
    NANOARROW_THROW_NOT_OK(
      ArrowArrayViewInitFromSchema(&actual, expected_schema->children[0], nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
    compare_arrays(expected.children[0], &actual);
    ArrowArrayViewReset(&actual);

    ArrowArrayViewReset(&expected);
  }
}

TEST_F(ToArrowHostDeviceTest, FixedPoint32TableLarge)
{
  using namespace numeric;
  auto constexpr NUM_ELEMENTS = 1000;

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const iota  = cuda::counting_iterator<int32_t>{1};
    auto const col   = fp_wrapper<int32_t>(iota, iota + NUM_ELEMENTS, scale_type{scale});
    auto const input = cudf::table_view({col});

    auto expect_data = std::vector<int32_t>(NUM_ELEMENTS);
    std::iota(expect_data.begin(), expect_data.end(), 1);

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

    nanoarrow::UniqueArray expected_array;
    NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromSchema(expected_array.get(), expected_schema.get(), nullptr));
    expected_array->length = input.num_rows();

    get_nanoarrow_array<int32_t>(expect_data).move(expected_array->children[0]);
    NANOARROW_THROW_NOT_OK(ArrowArrayFinishBuildingDefault(expected_array.get(), nullptr));

    auto got_arrow_host = cudf::to_arrow_host(input);
    EXPECT_EQ(ARROW_DEVICE_CPU, got_arrow_host->device_type);
    EXPECT_EQ(-1, got_arrow_host->device_id);
    EXPECT_EQ(nullptr, got_arrow_host->sync_event);

    ArrowArrayView expected, actual;
    NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&expected, expected_schema.get(), nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&expected, expected_array.get(), nullptr));

    NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&actual, expected_schema.get(), nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
    compare_arrays(&expected, &actual);
    ArrowArrayViewReset(&actual);

    got_arrow_host = cudf::to_arrow_host(input.column(0));
    NANOARROW_THROW_NOT_OK(
      ArrowArrayViewInitFromSchema(&actual, expected_schema->children[0], nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
    compare_arrays(expected.children[0], &actual);
    ArrowArrayViewReset(&actual);

    ArrowArrayViewReset(&expected);
  }
}

TEST_F(ToArrowHostDeviceTest, FixedPoint64TableLarge)
{
  using namespace numeric;
  auto constexpr NUM_ELEMENTS = 1000;

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const iota  = cuda::counting_iterator<int64_t>{1};
    auto const col   = fp_wrapper<int64_t>(iota, iota + NUM_ELEMENTS, scale_type{scale});
    auto const input = cudf::table_view({col});

    auto expect_data = std::vector<int64_t>(NUM_ELEMENTS);
    std::iota(expect_data.begin(), expect_data.end(), 1);

    nanoarrow::UniqueSchema expected_schema;
    ArrowSchemaInit(expected_schema.get());
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(expected_schema.get(), 1));
    ArrowSchemaInit(expected_schema->children[0]);
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeDecimal(expected_schema->children[0],
                                                     NANOARROW_TYPE_DECIMAL64,
                                                     get_decimal_precision<int64_t>(),
                                                     -scale));
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(expected_schema->children[0], "a"));
    expected_schema->children[0]->flags = 0;

    nanoarrow::UniqueArray expected_array;
    NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromSchema(expected_array.get(), expected_schema.get(), nullptr));
    expected_array->length = input.num_rows();

    get_nanoarrow_array<int64_t>(expect_data).move(expected_array->children[0]);
    NANOARROW_THROW_NOT_OK(ArrowArrayFinishBuildingDefault(expected_array.get(), nullptr));

    auto got_arrow_host = cudf::to_arrow_host(input);
    EXPECT_EQ(ARROW_DEVICE_CPU, got_arrow_host->device_type);
    EXPECT_EQ(-1, got_arrow_host->device_id);
    EXPECT_EQ(nullptr, got_arrow_host->sync_event);

    ArrowArrayView expected, actual;
    NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&expected, expected_schema.get(), nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&expected, expected_array.get(), nullptr));

    NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&actual, expected_schema.get(), nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
    compare_arrays(&expected, &actual);
    ArrowArrayViewReset(&actual);

    got_arrow_host = cudf::to_arrow_host(input.column(0));
    NANOARROW_THROW_NOT_OK(
      ArrowArrayViewInitFromSchema(&actual, expected_schema->children[0], nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
    compare_arrays(expected.children[0], &actual);
    ArrowArrayViewReset(&actual);

    ArrowArrayViewReset(&expected);
  }
}

TEST_F(ToArrowHostDeviceTest, FixedPoint128TableLarge)
{
  using namespace numeric;
  auto constexpr NUM_ELEMENTS = 1000;

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const iota  = cuda::counting_iterator<__int128_t>{1};
    auto const col   = fp_wrapper<__int128_t>(iota, iota + NUM_ELEMENTS, scale_type{scale});
    auto const input = cudf::table_view({col});

    auto expect_data = std::vector<__int128_t>(NUM_ELEMENTS);
    std::iota(expect_data.begin(), expect_data.end(), 1);

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

    nanoarrow::UniqueArray expected_array;
    NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromSchema(expected_array.get(), expected_schema.get(), nullptr));
    expected_array->length = input.num_rows();

    get_nanoarrow_array<__int128_t>(expect_data).move(expected_array->children[0]);
    NANOARROW_THROW_NOT_OK(ArrowArrayFinishBuildingDefault(expected_array.get(), nullptr));

    auto got_arrow_host = cudf::to_arrow_host(input);
    EXPECT_EQ(ARROW_DEVICE_CPU, got_arrow_host->device_type);
    EXPECT_EQ(-1, got_arrow_host->device_id);
    EXPECT_EQ(nullptr, got_arrow_host->sync_event);

    ArrowArrayView expected, actual;
    NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&expected, expected_schema.get(), nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&expected, expected_array.get(), nullptr));

    NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&actual, expected_schema.get(), nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
    compare_arrays(&expected, &actual);
    ArrowArrayViewReset(&actual);

    got_arrow_host = cudf::to_arrow_host(input.column(0));
    NANOARROW_THROW_NOT_OK(
      ArrowArrayViewInitFromSchema(&actual, expected_schema->children[0], nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
    compare_arrays(expected.children[0], &actual);
    ArrowArrayViewReset(&actual);

    ArrowArrayViewReset(&expected);
  }
}

TEST_F(ToArrowHostDeviceTest, FixedPoint32TableNullsSimple)
{
  using namespace numeric;

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const data     = std::vector<int32_t>{1, 2, 3, 4, 5, 6, 0, 0};
    auto const validity = std::vector<uint8_t>{1, 1, 1, 1, 1, 1, 0, 0};
    auto const col =
      fp_wrapper<int32_t>({1, 2, 3, 4, 5, 6, 0, 0}, {1, 1, 1, 1, 1, 1, 0, 0}, scale_type{scale});
    auto const input = cudf::table_view({col});

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

    nanoarrow::UniqueArray expected_array;
    NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromSchema(expected_array.get(), expected_schema.get(), nullptr));
    expected_array->length = input.num_rows();

    get_nanoarrow_array<int32_t>(data, validity).move(expected_array->children[0]);
    NANOARROW_THROW_NOT_OK(ArrowArrayFinishBuildingDefault(expected_array.get(), nullptr));

    auto got_arrow_host = cudf::to_arrow_host(input);
    EXPECT_EQ(ARROW_DEVICE_CPU, got_arrow_host->device_type);
    EXPECT_EQ(-1, got_arrow_host->device_id);
    EXPECT_EQ(nullptr, got_arrow_host->sync_event);

    ArrowArrayView expected, actual;
    NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&expected, expected_schema.get(), nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&expected, expected_array.get(), nullptr));

    NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&actual, expected_schema.get(), nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
    compare_arrays(&expected, &actual);
    ArrowArrayViewReset(&actual);

    got_arrow_host = cudf::to_arrow_host(input.column(0));
    NANOARROW_THROW_NOT_OK(
      ArrowArrayViewInitFromSchema(&actual, expected_schema->children[0], nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
    compare_arrays(expected.children[0], &actual);
    ArrowArrayViewReset(&actual);

    ArrowArrayViewReset(&expected);
  }
}

TEST_F(ToArrowHostDeviceTest, FixedPoint64TableNullsSimple)
{
  using namespace numeric;

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const data     = std::vector<int64_t>{1, 2, 3, 4, 5, 6, 0, 0};
    auto const validity = std::vector<uint8_t>{1, 1, 1, 1, 1, 1, 0, 0};
    auto const col =
      fp_wrapper<int64_t>({1, 2, 3, 4, 5, 6, 0, 0}, {1, 1, 1, 1, 1, 1, 0, 0}, scale_type{scale});
    auto const input = cudf::table_view({col});

    nanoarrow::UniqueSchema expected_schema;
    ArrowSchemaInit(expected_schema.get());
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(expected_schema.get(), 1));
    ArrowSchemaInit(expected_schema->children[0]);
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeDecimal(expected_schema->children[0],
                                                     NANOARROW_TYPE_DECIMAL64,
                                                     get_decimal_precision<int64_t>(),
                                                     -scale));
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(expected_schema->children[0], "a"));
    expected_schema->children[0]->flags = 0;

    nanoarrow::UniqueArray expected_array;
    NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromSchema(expected_array.get(), expected_schema.get(), nullptr));
    expected_array->length = input.num_rows();

    get_nanoarrow_array<int64_t>(data, validity).move(expected_array->children[0]);
    NANOARROW_THROW_NOT_OK(ArrowArrayFinishBuildingDefault(expected_array.get(), nullptr));

    auto got_arrow_host = cudf::to_arrow_host(input);
    EXPECT_EQ(ARROW_DEVICE_CPU, got_arrow_host->device_type);
    EXPECT_EQ(-1, got_arrow_host->device_id);
    EXPECT_EQ(nullptr, got_arrow_host->sync_event);

    ArrowArrayView expected, actual;
    NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&expected, expected_schema.get(), nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&expected, expected_array.get(), nullptr));

    NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&actual, expected_schema.get(), nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
    compare_arrays(&expected, &actual);
    ArrowArrayViewReset(&actual);

    got_arrow_host = cudf::to_arrow_host(input.column(0));
    NANOARROW_THROW_NOT_OK(
      ArrowArrayViewInitFromSchema(&actual, expected_schema->children[0], nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
    compare_arrays(expected.children[0], &actual);
    ArrowArrayViewReset(&actual);

    ArrowArrayViewReset(&expected);
  }
}

TEST_F(ToArrowHostDeviceTest, FixedPoint128TableNullsSimple)
{
  using namespace numeric;

  for (auto const scale : {3, 2, 1, 0, -1, -2, -3}) {
    auto const data     = std::vector<__int128_t>{1, 2, 3, 4, 5, 6, 0, 0};
    auto const validity = std::vector<uint8_t>{1, 1, 1, 1, 1, 1, 0, 0};
    auto const col =
      fp_wrapper<__int128_t>({1, 2, 3, 4, 5, 6, 0, 0}, {1, 1, 1, 1, 1, 1, 0, 0}, scale_type{scale});
    auto const input = cudf::table_view({col});

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

    nanoarrow::UniqueArray expected_array;
    NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromSchema(expected_array.get(), expected_schema.get(), nullptr));
    expected_array->length = input.num_rows();

    get_nanoarrow_array<__int128_t>(data, validity).move(expected_array->children[0]);
    NANOARROW_THROW_NOT_OK(ArrowArrayFinishBuildingDefault(expected_array.get(), nullptr));

    auto got_arrow_host = cudf::to_arrow_host(input);
    EXPECT_EQ(ARROW_DEVICE_CPU, got_arrow_host->device_type);
    EXPECT_EQ(-1, got_arrow_host->device_id);
    EXPECT_EQ(nullptr, got_arrow_host->sync_event);

    ArrowArrayView expected, actual;
    NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&expected, expected_schema.get(), nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&expected, expected_array.get(), nullptr));

    NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&actual, expected_schema.get(), nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
    compare_arrays(&expected, &actual);
    ArrowArrayViewReset(&actual);

    got_arrow_host = cudf::to_arrow_host(input.column(0));
    NANOARROW_THROW_NOT_OK(
      ArrowArrayViewInitFromSchema(&actual, expected_schema->children[0], nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
    compare_arrays(expected.children[0], &actual);
    ArrowArrayViewReset(&actual);

    ArrowArrayViewReset(&expected);
  }
}

struct ToArrowHostDeviceTestSlice
  : public ToArrowHostDeviceTest,
    public ::testing::WithParamInterface<std::tuple<cudf::size_type, cudf::size_type>> {};

TEST_P(ToArrowHostDeviceTestSlice, SliceTest)
{
  auto [table, expected_schema, expected_array] = get_nanoarrow_host_tables(10000);
  auto cudf_table_view                          = table->view();
  auto const [start, end]                       = GetParam();

  slice_host_nanoarrow(expected_array.get(), start, end);
  auto sliced_cudf_table = cudf::slice(cudf_table_view, {start, end})[0];
  auto got_arrow_host    = cudf::to_arrow_host(sliced_cudf_table);
  EXPECT_EQ(ARROW_DEVICE_CPU, got_arrow_host->device_type);
  EXPECT_EQ(-1, got_arrow_host->device_id);
  EXPECT_EQ(nullptr, got_arrow_host->sync_event);

  ArrowArrayView expected, actual;
  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&expected, expected_schema.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&expected, expected_array.get(), nullptr));

  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&actual, expected_schema.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
  compare_arrays(&expected, &actual);
  ArrowArrayViewReset(&actual);

  ArrowArrayViewReset(&expected);
}

INSTANTIATE_TEST_CASE_P(ToArrowHostDeviceTest,
                        ToArrowHostDeviceTestSlice,
                        ::testing::Values(std::make_tuple(0, 10000),
                                          std::make_tuple(100, 3000),
                                          std::make_tuple(0, 0),
                                          std::make_tuple(0, 3000)));
