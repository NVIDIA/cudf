/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/nanoarrow_utils.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

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
get_nanoarrow_stream(int num_copies, cuda::stream_ref stream, cudf::memory_resources mr)
{
  auto const temporary_mr        = mr.get_temporary_mr();
  auto const temporary_resources = cudf::memory_resources{temporary_mr, temporary_mr};
  std::vector<std::unique_ptr<cudf::table>> tables;
  // The schema is unique across all tables.
  nanoarrow::UniqueSchema schema;
  std::vector<nanoarrow::UniqueArray> arrays;
  for (auto i = 0; i < num_copies; ++i) {
    auto [tbl, sch, arr] = get_nanoarrow_host_tables(3, stream, temporary_resources);
    tables.push_back(std::move(tbl));
    arrays.push_back(std::move(arr));
    if (i == 0) { sch.move(schema.get()); }
  }
  std::vector<cudf::table_view> table_views;
  for (auto const& table : tables) {
    table_views.push_back(table->view());
  }
  auto expected = cudf::concatenate(table_views, stream, mr.get_output_mr());

  ArrowArrayStream arrow_stream;
  makeStreamFromArrays(std::move(arrays), std::move(schema), &arrow_stream);
  return std::make_tuple(std::move(expected), std::move(schema), arrow_stream);
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

TEST_F(FromArrowStreamTest, TestUtilityMemoryResourceControl)
{
  auto upstream     = this->mr();
  auto output_mr    = rmm::mr::statistics_resource_adaptor(upstream);
  auto temporary_mr = rmm::mr::statistics_resource_adaptor(upstream);
  auto resources    = cudf::memory_resources{output_mr, temporary_mr};
  auto stream       = cudf::get_default_stream();

  {
    auto direct_table = get_cudf_table(stream, resources);
    auto [generated_table, generated_schema, test_data] =
      get_nanoarrow_cudf_table(3, stream, resources);
    auto [device_table, device_schema, device_array] = get_nanoarrow_tables(0, stream, resources);
    auto [host_table, host_schema, host_array] = get_nanoarrow_host_tables(3, stream, resources);
    auto [stream_table, stream_schema, arrow_stream] = get_nanoarrow_stream(2, stream, resources);

    stream.synchronize();
    EXPECT_GT(output_mr.get_bytes_counter().value, 0);
    EXPECT_EQ(temporary_mr.get_bytes_counter().value, 0);
    EXPECT_GT(temporary_mr.get_bytes_counter().total, 0);

    if (arrow_stream.release != nullptr) { arrow_stream.release(&arrow_stream); }
  }

  stream.synchronize();
  EXPECT_EQ(output_mr.get_bytes_counter().value, 0);
  EXPECT_EQ(temporary_mr.get_bytes_counter().value, 0);
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
