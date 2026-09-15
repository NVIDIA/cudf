/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/nanoarrow_utils.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/interop.hpp>

#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.h>

#include <memory>
#include <utility>

struct ArrowColumnTest : public cudf::test::BaseFixture {};

namespace {

void set_fixed_size_list_type(ArrowSchema* schema)
{
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeFixedSize(schema, NANOARROW_TYPE_FIXED_SIZE_LIST, 3));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_INT64));
}

nanoarrow::UniqueSchema make_fixed_size_list_schema(bool wrap_in_struct)
{
  nanoarrow::UniqueSchema schema;
  ArrowSchemaInit(schema.get());

  auto* list_schema = schema.get();
  if (wrap_in_struct) {
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(schema.get(), 1));
    list_schema = schema->children[0];
  }

  set_fixed_size_list_type(list_schema);
  return schema;
}

nanoarrow::UniqueSchema make_fixed_size_list_dictionary_schema()
{
  nanoarrow::UniqueSchema schema;
  ArrowSchemaInit(schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetType(schema.get(), NANOARROW_TYPE_INT32));
  NANOARROW_THROW_NOT_OK(ArrowSchemaAllocateDictionary(schema.get()));
  ArrowSchemaInit(schema->dictionary);
  set_fixed_size_list_type(schema->dictionary);
  return schema;
}

ArrowDeviceArray make_empty_device_array()
{
  return ArrowDeviceArray{
    .array = {}, .device_id = 0, .device_type = ARROW_DEVICE_CUDA, .sync_event = nullptr};
}

}  // namespace

template <typename T>
auto export_to_arrow(T& obj, ArrowDeviceType device_type = ARROW_DEVICE_CUDA)
{
  // Now we can extract an ArrowDeviceArray from the arrow_column
  auto schema = std::make_unique<ArrowSchema>();
  obj.to_arrow_schema(schema.get());
  auto array = std::make_unique<ArrowDeviceArray>();
  obj.to_arrow(array.get(), device_type);
  return std::make_pair(std::move(schema), std::move(array));
}

TEST_F(ArrowColumnTest, TwoWayConversion)
{
  cudf::test::fixed_width_column_wrapper<int32_t> int_col{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
  auto col = cudf::column(int_col);
  auto arrow_column_from_cudf_column =
    cudf::interop::arrow_column(std::move(col), cudf::interop::get_column_metadata(int_col));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(int_col, arrow_column_from_cudf_column.view());

  auto [arrow_schema_from_arrow_column, arrow_array_from_arrow_column] =
    export_to_arrow(arrow_column_from_cudf_column);
  arrow_column_from_cudf_column.to_arrow_schema(arrow_schema_from_arrow_column.get());
  arrow_column_from_cudf_column.to_arrow(arrow_array_from_arrow_column.get(), ARROW_DEVICE_CUDA);

  auto arrow_column_from_arrow_array = cudf::interop::arrow_column(
    std::move(*arrow_schema_from_arrow_column), std::move(*arrow_array_from_arrow_column));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(int_col, arrow_column_from_arrow_array.view());
}

TEST_F(ArrowColumnTest, LifetimeManagement)
{
  cudf::test::fixed_width_column_wrapper<int32_t> int_col{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
  auto col                           = std::make_unique<cudf::column>(int_col);
  auto arrow_column_from_cudf_column = std::make_unique<cudf::interop::arrow_column>(
    std::move(*col), cudf::interop::get_column_metadata(int_col));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(int_col, arrow_column_from_cudf_column->view());

  auto [schema1, array1] = export_to_arrow(*arrow_column_from_cudf_column);
  auto [schema2, array2] = export_to_arrow(*arrow_column_from_cudf_column);

  // Delete the original owner of the data, then reimport and ensure that we
  // are still referencing the same valid original data.
  arrow_column_from_cudf_column.reset();
  auto col1 =
    std::make_unique<cudf::interop::arrow_column>(std::move(*schema1), std::move(*array1));
  auto col2 =
    std::make_unique<cudf::interop::arrow_column>(std::move(*schema2), std::move(*array2));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(int_col, col1->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col1->view(), col2->view());
}

TEST_F(ArrowColumnTest, ComplexNanoarrowDeviceTables)
{
  auto [tbl, schema, arr] = get_nanoarrow_tables(100);
  for (auto i = 0; i < tbl->num_columns(); i++) {
    auto& col = tbl->get_column(i);

    ArrowDeviceArray device_arr{
      .array       = {},
      .device_id   = 0,
      .device_type = ARROW_DEVICE_CUDA,
    };
    ArrowArrayMove(arr->children[i], &device_arr.array);
    auto arrow_column_from_nanoarrow_array =
      cudf::interop::arrow_column(std::move(*schema->children[i]), std::move(device_arr));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(col.view(), arrow_column_from_nanoarrow_array.view());

    auto [arrow_schema_from_nanoarrow_array, arrow_array_from_arrow_column] =
      export_to_arrow(arrow_column_from_nanoarrow_array);
    arrow_column_from_nanoarrow_array.to_arrow_schema(arrow_schema_from_nanoarrow_array.get());
    arrow_column_from_nanoarrow_array.to_arrow(arrow_array_from_arrow_column.get(),
                                               ARROW_DEVICE_CUDA);

    auto arrow_column_from_arrow_array = cudf::interop::arrow_column(
      std::move(*arrow_schema_from_nanoarrow_array), std::move(*arrow_array_from_arrow_column));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(col.view(), arrow_column_from_arrow_array.view());
  }
}

TEST_F(ArrowColumnTest, ComplexNanoarrowHostTables)
{
  auto [tbl, schema, arr] = get_nanoarrow_host_tables(100);
  for (auto i = 0; i < tbl->num_columns(); i++) {
    auto& col = tbl->get_column(i);

    ArrowDeviceArray device_arr{
      .array       = {},
      .device_id   = -1,
      .device_type = ARROW_DEVICE_CPU,
    };
    ArrowArrayMove(arr->children[i], &device_arr.array);
    auto arrow_column_from_nanoarrow_array =
      cudf::interop::arrow_column(std::move(*schema->children[i]), std::move(device_arr));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(col.view(), arrow_column_from_nanoarrow_array.view());

    auto [arrow_schema_from_nanoarrow_array, arrow_array_from_arrow_column] =
      export_to_arrow(arrow_column_from_nanoarrow_array);
    arrow_column_from_nanoarrow_array.to_arrow_schema(arrow_schema_from_nanoarrow_array.get());
    arrow_column_from_nanoarrow_array.to_arrow(arrow_array_from_arrow_column.get(),
                                               ARROW_DEVICE_CUDA);

    auto arrow_column_from_arrow_array = cudf::interop::arrow_column(
      std::move(*arrow_schema_from_nanoarrow_array), std::move(*arrow_array_from_arrow_column));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(col.view(), arrow_column_from_arrow_array.view());
  }
}

TEST_F(ArrowColumnTest, ComplexNanoarrowHostArrowArrayTables)
{
  auto [tbl, schema, arr] = get_nanoarrow_host_tables(100);
  for (auto i = 0; i < tbl->num_columns(); i++) {
    auto& col = tbl->get_column(i);

    auto arrow_column_from_nanoarrow_array =
      cudf::interop::arrow_column(std::move(*schema->children[i]), std::move(*arr->children[i]));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(col.view(), arrow_column_from_nanoarrow_array.view());

    auto [arrow_schema_from_nanoarrow_array, arrow_array_from_arrow_column] =
      export_to_arrow(arrow_column_from_nanoarrow_array);
    arrow_column_from_nanoarrow_array.to_arrow_schema(arrow_schema_from_nanoarrow_array.get());
    arrow_column_from_nanoarrow_array.to_arrow(arrow_array_from_arrow_column.get(),
                                               ARROW_DEVICE_CUDA);

    auto arrow_column_from_arrow_array = cudf::interop::arrow_column(
      std::move(*arrow_schema_from_nanoarrow_array), std::move(*arrow_array_from_arrow_column));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(col.view(), arrow_column_from_arrow_array.view());
  }
}

TEST_F(ArrowColumnTest, ToFromHost)
{
  cudf::test::fixed_width_column_wrapper<int32_t> int_col{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
  auto col = cudf::column(int_col);
  auto arrow_column_from_cudf_column =
    cudf::interop::arrow_column(std::move(col), cudf::interop::get_column_metadata(int_col));

  auto [arrow_schema_from_arrow_column, arrow_array_from_arrow_column] =
    export_to_arrow(arrow_column_from_cudf_column, ARROW_DEVICE_CPU);
  arrow_column_from_cudf_column.to_arrow_schema(arrow_schema_from_arrow_column.get());
  arrow_column_from_cudf_column.to_arrow(arrow_array_from_arrow_column.get(), ARROW_DEVICE_CPU);

  auto arrow_column_from_arrow_array = cudf::interop::arrow_column(
    std::move(*arrow_schema_from_arrow_column), std::move(*arrow_array_from_arrow_column));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(int_col, arrow_column_from_arrow_array.view());
}

TEST_F(ArrowColumnTest, FixedSizeListDeviceInputRejected)
{
  auto schema = make_fixed_size_list_schema(false);
  auto array  = make_empty_device_array();

  EXPECT_THROW(
    { static_cast<void>(cudf::interop::arrow_column(std::move(*schema.get()), std::move(array))); },
    cudf::data_type_error);
}

TEST_F(ArrowColumnTest, DictionaryFixedSizeListDeviceInputRejected)
{
  auto schema = make_fixed_size_list_dictionary_schema();
  auto array  = make_empty_device_array();

  EXPECT_THROW(
    { static_cast<void>(cudf::interop::arrow_column(std::move(*schema.get()), std::move(array))); },
    cudf::data_type_error);
}

struct ArrowTableTest : public cudf::test::BaseFixture {};

TEST_F(ArrowTableTest, TwoWayConversion)
{
  cudf::test::fixed_width_column_wrapper<int32_t> int_col{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
  cudf::test::fixed_width_column_wrapper<float> float_col{
    {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.}};
  auto original_view = cudf::table_view{{int_col, float_col}};
  cudf::table table{cudf::table_view{{int_col, float_col}}};
  auto arrow_table_from_cudf_table =
    cudf::interop::arrow_table(std::move(table), cudf::interop::get_table_metadata(original_view));

  CUDF_TEST_EXPECT_TABLES_EQUAL(original_view, arrow_table_from_cudf_table.view());

  auto [arrow_schema_from_arrow_table, arrow_array_from_arrow_table] =
    export_to_arrow(arrow_table_from_cudf_table);
  arrow_table_from_cudf_table.to_arrow_schema(arrow_schema_from_arrow_table.get());
  arrow_table_from_cudf_table.to_arrow(arrow_array_from_arrow_table.get(), ARROW_DEVICE_CUDA);

  auto arrow_table_from_arrow_array = cudf::interop::arrow_table(
    std::move(*arrow_schema_from_arrow_table), std::move(*arrow_array_from_arrow_table));
  CUDF_TEST_EXPECT_TABLES_EQUAL(original_view, arrow_table_from_arrow_array.view());
}

TEST_F(ArrowTableTest, ComplexNanoarrowDeviceTables)
{
  auto [tbl, schema, arr] = get_nanoarrow_tables(100);
  ArrowDeviceArray device_arr{
    .array       = {},
    .device_id   = 0,
    .device_type = ARROW_DEVICE_CUDA,
  };
  ArrowArrayMove(arr.get(), &device_arr.array);
  auto arrow_table_from_nanoarrow_array =
    cudf::interop::arrow_table(std::move(*schema.get()), std::move(device_arr));

  CUDF_TEST_EXPECT_TABLES_EQUAL(tbl->view(), arrow_table_from_nanoarrow_array.view());

  auto [arrow_schema_from_nanoarrow_array, arrow_array_from_arrow_table] =
    export_to_arrow(arrow_table_from_nanoarrow_array);
  arrow_table_from_nanoarrow_array.to_arrow_schema(arrow_schema_from_nanoarrow_array.get());
  arrow_table_from_nanoarrow_array.to_arrow(arrow_array_from_arrow_table.get(), ARROW_DEVICE_CUDA);

  auto arrow_table_from_arrow_array = cudf::interop::arrow_table(
    std::move(*arrow_schema_from_nanoarrow_array), std::move(*arrow_array_from_arrow_table));
  CUDF_TEST_EXPECT_TABLES_EQUAL(tbl->view(), arrow_table_from_arrow_array.view());
}

TEST_F(ArrowTableTest, ComplexNanoarrowHostTables)
{
  auto [tbl, schema, arr] = get_nanoarrow_host_tables(100);
  ArrowDeviceArray device_arr{
    .array       = {},
    .device_id   = -1,
    .device_type = ARROW_DEVICE_CPU,
  };
  ArrowArrayMove(arr.get(), &device_arr.array);
  auto arrow_table_from_nanoarrow_array =
    cudf::interop::arrow_table(std::move(*schema.get()), std::move(device_arr));

  CUDF_TEST_EXPECT_TABLES_EQUAL(tbl->view(), arrow_table_from_nanoarrow_array.view());

  auto [arrow_schema_from_nanoarrow_array, arrow_array_from_arrow_table] =
    export_to_arrow(arrow_table_from_nanoarrow_array);
  arrow_table_from_nanoarrow_array.to_arrow_schema(arrow_schema_from_nanoarrow_array.get());
  arrow_table_from_nanoarrow_array.to_arrow(arrow_array_from_arrow_table.get(), ARROW_DEVICE_CUDA);

  auto arrow_table_from_arrow_array = cudf::interop::arrow_table(
    std::move(*arrow_schema_from_nanoarrow_array), std::move(*arrow_array_from_arrow_table.get()));
  CUDF_TEST_EXPECT_TABLES_EQUAL(tbl->view(), arrow_table_from_arrow_array.view());
}

TEST_F(ArrowTableTest, ComplexNanoarrowHostArrowArrayTables)
{
  auto [tbl, schema, arr] = get_nanoarrow_host_tables(100);
  auto arrow_table_from_nanoarrow_array =
    cudf::interop::arrow_table(std::move(*schema.get()), std::move(*arr.get()));

  CUDF_TEST_EXPECT_TABLES_EQUAL(tbl->view(), arrow_table_from_nanoarrow_array.view());

  auto [arrow_schema_from_nanoarrow_array, arrow_array_from_arrow_table] =
    export_to_arrow(arrow_table_from_nanoarrow_array);
  arrow_table_from_nanoarrow_array.to_arrow_schema(arrow_schema_from_nanoarrow_array.get());
  arrow_table_from_nanoarrow_array.to_arrow(arrow_array_from_arrow_table.get(), ARROW_DEVICE_CUDA);

  auto arrow_table_from_arrow_array = cudf::interop::arrow_table(
    std::move(*arrow_schema_from_nanoarrow_array), std::move(*arrow_array_from_arrow_table));
  CUDF_TEST_EXPECT_TABLES_EQUAL(tbl->view(), arrow_table_from_arrow_array.view());
}

TEST_F(ArrowTableTest, ToFromHost)
{
  cudf::test::fixed_width_column_wrapper<int32_t> int_col{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
  cudf::test::fixed_width_column_wrapper<float> float_col{
    {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.}};
  auto original_view = cudf::table_view{{int_col, float_col}};
  cudf::table table{cudf::table_view{{int_col, float_col}}};
  auto arrow_table_from_cudf_table =
    cudf::interop::arrow_table(std::move(table), cudf::interop::get_table_metadata(original_view));

  auto [arrow_schema_from_arrow_table, arrow_array_from_arrow_table] =
    export_to_arrow(arrow_table_from_cudf_table, ARROW_DEVICE_CPU);
  arrow_table_from_cudf_table.to_arrow_schema(arrow_schema_from_arrow_table.get());
  arrow_table_from_cudf_table.to_arrow(arrow_array_from_arrow_table.get(), ARROW_DEVICE_CPU);

  auto arrow_table_from_arrow_array = cudf::interop::arrow_table(
    std::move(*arrow_schema_from_arrow_table), std::move(*arrow_array_from_arrow_table));
  CUDF_TEST_EXPECT_TABLES_EQUAL(original_view, arrow_table_from_arrow_array.view());
}

TEST_F(ArrowTableTest, FromArrowArrayStream)
{
  auto num_copies    = 3;
  auto [tbl, stream] = get_nanoarrow_stream(num_copies);

  auto result = cudf::interop::arrow_table(std::move(stream));
  CUDF_TEST_EXPECT_TABLES_EQUAL(tbl->view(), result.view());
}

TEST_F(ArrowTableTest, NestedFixedSizeListDeviceInputRejected)
{
  auto schema = make_fixed_size_list_schema(true);
  auto array  = make_empty_device_array();

  EXPECT_THROW(
    { static_cast<void>(cudf::interop::arrow_table(std::move(*schema.get()), std::move(array))); },
    cudf::data_type_error);
}
