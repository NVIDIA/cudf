/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/nanoarrow_utils.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <iostream>

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

std::tuple<std::unique_ptr<cudf::table>, nanoarrow::UniqueSchema, generated_test_data>
get_nanoarrow_cudf_table(cudf::size_type length, cuda::stream_ref stream, cudf::memory_resources mr)
{
  auto const temporary_mr = mr.get_temporary_mr();
  generated_test_data test_data(length);

  std::vector<std::unique_ptr<cudf::column>> columns;

  columns.emplace_back(cudf::test::fixed_width_column_wrapper<int64_t>(test_data.int64_data.begin(),
                                                                       test_data.int64_data.end(),
                                                                       test_data.validity.begin(),
                                                                       stream,
                                                                       mr)
                         .release());
  columns.emplace_back(cudf::test::strings_column_wrapper(test_data.string_data.begin(),
                                                          test_data.string_data.end(),
                                                          test_data.validity.begin(),
                                                          stream,
                                                          mr)
                         .release());
  auto col4 = cudf::test::fixed_width_column_wrapper<int64_t>(test_data.int64_data.begin(),
                                                              test_data.int64_data.end(),
                                                              test_data.validity.begin(),
                                                              stream,
                                                              temporary_mr);
  columns.emplace_back(cudf::dictionary::encode(
    col4, cudf::data_type{cudf::type_id::INT32}, stream, mr.get_output_mr()));
  columns.emplace_back(cudf::test::fixed_width_column_wrapper<bool>(test_data.bool_data.begin(),
                                                                    test_data.bool_data.end(),
                                                                    test_data.bool_validity.begin(),
                                                                    stream,
                                                                    mr)
                         .release());
  auto list_child_column =
    cudf::test::fixed_width_column_wrapper<int64_t>(test_data.list_int64_data.begin(),
                                                    test_data.list_int64_data.end(),
                                                    test_data.list_int64_data_validity.begin(),
                                                    stream,
                                                    mr);
  auto list_offsets_column = cudf::test::fixed_width_column_wrapper<int32_t>(
    test_data.list_offsets.begin(), test_data.list_offsets.end(), stream, mr);
  auto list_validity = cudf::test::fixed_width_column_wrapper<bool>(
    test_data.list_validity.begin(), test_data.list_validity.end(), stream, temporary_mr);
  auto [list_mask, list_nulls] = cudf::bools_to_mask(list_validity, stream, mr.get_output_mr());
  columns.emplace_back(cudf::make_lists_column(length,
                                               list_offsets_column.release(),
                                               list_child_column.release(),
                                               list_nulls,
                                               std::move(*list_mask)));
  auto int_column = cudf::test::fixed_width_column_wrapper<int64_t>(test_data.int64_data.begin(),
                                                                    test_data.int64_data.end(),
                                                                    test_data.validity.begin(),
                                                                    stream,
                                                                    mr)
                      .release();
  auto str_column = cudf::test::strings_column_wrapper(test_data.string_data.begin(),
                                                       test_data.string_data.end(),
                                                       test_data.validity.begin(),
                                                       stream,
                                                       mr)
                      .release();
  vector_of_columns cols;
  cols.push_back(std::move(int_column));
  cols.push_back(std::move(str_column));
  auto struct_validity = cudf::test::fixed_width_column_wrapper<bool>(
    test_data.bool_data_validity.begin(), test_data.bool_data_validity.end(), stream, temporary_mr);
  auto [null_mask, null_count] = cudf::bools_to_mask(struct_validity, stream, mr.get_output_mr());
  columns.emplace_back(cudf::make_structs_column(
    length, std::move(cols), null_count, std::move(*null_mask), stream, mr.get_output_mr()));

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
get_nanoarrow_tables(cudf::size_type length, cuda::stream_ref stream, cudf::memory_resources mr)
{
  auto [table, schema, test_data] = get_nanoarrow_cudf_table(length, stream, mr);

  nanoarrow::UniqueArray arrow;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(arrow.get(), schema.get(), nullptr));
  arrow->length = length;

  populate_from_col<int64_t>(arrow->children[0], table->get_column(0).view());
  populate_from_col<cudf::string_view>(arrow->children[1], table->get_column(1).view(), stream, mr);
  populate_dict_from_col<int64_t, int32_t>(
    arrow->children[2], cudf::dictionary_column_view(table->get_column(2).view()), stream, mr);

  populate_from_col<bool>(arrow->children[3], table->get_column(3).view(), stream, mr);
  cudf::lists_column_view list_view{table->get_column(4).view()};
  populate_list_from_col(arrow->children[4], list_view);
  populate_from_col<int64_t>(arrow->children[4]->children[0], list_view.child());

  cudf::structs_column_view struct_view{table->get_column(5).view()};
  populate_from_col<int64_t>(arrow->children[5]->children[0], struct_view.child(0));
  populate_from_col<cudf::string_view>(
    arrow->children[5]->children[1], struct_view.child(1), stream, mr);
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

std::tuple<std::unique_ptr<cudf::table>, nanoarrow::UniqueSchema, nanoarrow::UniqueArray>
get_nanoarrow_host_tables(cudf::size_type length,
                          cuda::stream_ref stream,
                          cudf::memory_resources mr)
{
  auto [table, schema, test_data] = get_nanoarrow_cudf_table(length, stream, mr);

  auto int64_array = get_nanoarrow_array<int64_t>(test_data.int64_data, test_data.validity);
  auto string_array =
    get_nanoarrow_array<cudf::string_view>(test_data.string_data, test_data.validity);
  cudf::dictionary_column_view view(table->get_column(2).view());
  auto keys       = cudf::test::to_host<int64_t>(view.keys(), stream, mr).first;
  auto indices    = cudf::test::to_host<uint32_t>(view.indices(), stream, mr).first;
  auto dict_array = get_nanoarrow_dict_array(std::vector<int64_t>(keys.begin(), keys.end()),
                                             std::vector<int32_t>(indices.begin(), indices.end()),
                                             test_data.validity);
  auto boolarray  = get_nanoarrow_array<bool>(test_data.bool_data, test_data.bool_validity);
  auto list_array = get_nanoarrow_list_array<int64_t>(test_data.list_int64_data,
                                                      test_data.list_offsets,
                                                      test_data.list_int64_data_validity,
                                                      test_data.list_validity);

  nanoarrow::UniqueArray arrow;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(arrow.get(), schema.get(), nullptr));
  arrow->length = length;

  int64_array.move(arrow->children[0]);
  string_array.move(arrow->children[1]);
  dict_array.move(arrow->children[2]);
  boolarray.move(arrow->children[3]);
  list_array.move(arrow->children[4]);

  int64_array  = get_nanoarrow_array<int64_t>(test_data.int64_data, test_data.validity);
  string_array = get_nanoarrow_array<cudf::string_view>(test_data.string_data, test_data.validity);
  int64_array.move(arrow->children[5]->children[0]);
  string_array.move(arrow->children[5]->children[1]);

  ArrowBitmap struct_validity;
  ArrowBitmapInit(&struct_validity);
  NANOARROW_THROW_NOT_OK(ArrowBitmapReserve(&struct_validity, length));
  ArrowBitmapAppendInt8Unsafe(
    &struct_validity, reinterpret_cast<int8_t const*>(test_data.bool_data_validity.data()), length);
  arrow->children[5]->length = length;
  ArrowArraySetValidityBitmap(arrow->children[5], &struct_validity);
  arrow->children[5]->null_count =
    length - ArrowBitCountSet(ArrowArrayValidityBitmap(arrow->children[5])->buffer.data, 0, length);

  ArrowError error;
  if (ArrowArrayFinishBuilding(arrow.get(), NANOARROW_VALIDATION_LEVEL_MINIMAL, &error) !=
      NANOARROW_OK) {
    std::cerr << ArrowErrorMessage(&error) << std::endl;
    CUDF_FAIL("failed to build example arrays");
  }

  return std::make_tuple(std::move(table), std::move(schema), std::move(arrow));
}

void slice_host_nanoarrow(ArrowArray* arr, int64_t start, int64_t end)
{
  auto op = [&](ArrowArray* array) {
    // slicing only needs to happen at the top level of an array
    array->offset = start;
    array->length = end - start;
    if (array->null_count != 0) {
      array->null_count =
        array->length -
        ArrowBitCountSet(ArrowArrayValidityBitmap(array)->buffer.data, start, end - start);
    }
  };

  if (arr->n_children == 0) {
    op(arr);
    return;
  }

  // since we want to simulate a sliced table where the children are sliced,
  // we slice each individual child of the record batch
  arr->length = end - start;
  for (int64_t i = 0; i < arr->n_children; ++i) {
    op(arr->children[i]);
  }
}

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

std::pair<std::unique_ptr<cudf::table>, ArrowArrayStream> get_nanoarrow_stream(
  int num_copies, cuda::stream_ref stream, cudf::memory_resources mr)
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
  return std::make_pair(std::move(expected), arrow_stream);
}
