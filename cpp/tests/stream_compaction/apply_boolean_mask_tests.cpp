/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>

struct ApplyBooleanMask : public cudf::test::BaseFixture {};

TEST_F(ApplyBooleanMask, ZeroColumnsPreservesRowCount)
{
  cudf::table_view input{std::vector<cudf::column_view>{}, 4};
  cudf::test::fixed_width_column_wrapper<bool> boolean_mask{{true, false, true, true}};
  auto got = cudf::apply_boolean_mask(input, boolean_mask);
  EXPECT_EQ(got->num_columns(), 0);
  EXPECT_EQ(got->num_rows(), 3);
}

TEST_F(ApplyBooleanMask, NonNullBooleanMask)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::table_view input{{col1, col2, col3}};
  cudf::test::fixed_width_column_wrapper<bool> boolean_mask{
    {true, false, true, false, true, false}};
  cudf::test::fixed_width_column_wrapper<int16_t> col1_expected{{true, true, true}, {1, 0, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{{10, 70, 2}, {1, 0, 1}};
  cudf::test::fixed_width_column_wrapper<double> col3_expected{{10, 70, 2}, {1, 0, 1}};
  cudf::table_view expected{{col1_expected, col2_expected, col3_expected}};

  auto got = cudf::apply_boolean_mask(input, boolean_mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(ApplyBooleanMask, NullBooleanMask)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::table_view input{{col1, col2, col3}};
  cudf::test::fixed_width_column_wrapper<bool> boolean_mask{{true, false, true, false, true, false},
                                                            {0, 1, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int16_t> col1_expected{{true, true}, {0, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{{70, 2}, {0, 1}};
  cudf::test::fixed_width_column_wrapper<double> col3_expected{{70, 2}, {0, 1}};
  cudf::table_view expected{{col1_expected, col2_expected, col3_expected}};

  auto got = cudf::apply_boolean_mask(input, boolean_mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(ApplyBooleanMask, EmptyMask)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::table_view input{{col1, col2, col3}};
  cudf::test::fixed_width_column_wrapper<bool> boolean_mask{};
  cudf::test::fixed_width_column_wrapper<int16_t> col1_expected{};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{};
  cudf::test::fixed_width_column_wrapper<double> col3_expected{};
  cudf::table_view expected{{col1_expected, col2_expected, col3_expected}};

  auto got = cudf::apply_boolean_mask(input, boolean_mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(ApplyBooleanMask, WrongMaskType)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::table_view input{{col1, col2, col3}};
  cudf::test::fixed_width_column_wrapper<int16_t> boolean_mask{
    {true, false, true, false, true, false}};

  EXPECT_THROW(cudf::apply_boolean_mask(input, boolean_mask), cudf::logic_error);
}

TEST_F(ApplyBooleanMask, MaskAndInputSizeMismatch)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::table_view input{{col1, col2, col3}};
  cudf::test::fixed_width_column_wrapper<bool> boolean_mask{{true, false, true, false, true}};

  EXPECT_THROW(cudf::apply_boolean_mask(input, boolean_mask), cudf::logic_error);
}

TEST_F(ApplyBooleanMask, StringColumnTest)
{
  cudf::test::strings_column_wrapper col1{
    {"This", "is", "the", "a", "k12", "string", "table", "column"}, {1, 1, 1, 1, 1, 0, 1, 1}};
  cudf::table_view input{{col1}};
  cudf::test::fixed_width_column_wrapper<bool> boolean_mask{
    {true, true, true, true, false, true, false, true}, {1, 1, 0, 1, 1, 1, 1, 1}};
  cudf::test::strings_column_wrapper col1_expected{{"This", "is", "a", "string", "column"},
                                                   {1, 1, 1, 0, 1}};
  cudf::table_view expected{{col1_expected}};

  auto got = cudf::apply_boolean_mask(input, boolean_mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(ApplyBooleanMask, withoutNullString)
{
  cudf::test::strings_column_wrapper col1({"d", "e", "a", "d", "k", "d", "l"});
  cudf::table_view cudf_table_in_view{{col1}};

  cudf::test::fixed_width_column_wrapper<bool> bool_filter{{1, 1, 0, 0, 1, 0, 0}};
  cudf::column_view bool_filter_col(bool_filter);

  std::unique_ptr<cudf::table> filteredTable =
    cudf::apply_boolean_mask(cudf_table_in_view, bool_filter_col);
  cudf::table_view tableView = filteredTable->view();

  cudf::test::strings_column_wrapper expect_col1({"d", "e", "k"});
  cudf::table_view expect_cudf_table_view{{expect_col1}};

  CUDF_TEST_EXPECT_TABLES_EQUAL(expect_cudf_table_view, tableView);
}

TEST_F(ApplyBooleanMask, FixedPointColumnTest)
{
  using namespace numeric;
  using decimal32_wrapper = cudf::test::fixed_point_column_wrapper<int32_t>;
  using decimal64_wrapper = cudf::test::fixed_point_column_wrapper<int64_t>;

  auto const col1 = decimal32_wrapper{{10, 40, 70, 5, 2, 10, -123}, scale_type{-1}};
  auto const col2 = decimal64_wrapper{{10, 40, 70, 5, 2, 10, -123}, scale_type{-10}};
  cudf::table_view cudf_table_in_view{{col1, col2}};

  cudf::test::fixed_width_column_wrapper<bool> bool_filter{{1, 1, 0, 0, 1, 0, 0}};
  cudf::column_view bool_filter_col(bool_filter);

  std::unique_ptr<cudf::table> filteredTable =
    cudf::apply_boolean_mask(cudf_table_in_view, bool_filter_col);
  cudf::table_view tableView = filteredTable->view();

  auto const expect_col1 = decimal32_wrapper{{10, 40, 2}, scale_type{-1}};
  auto const expect_col2 = decimal64_wrapper{{10, 40, 2}, scale_type{-10}};
  cudf::table_view expect_cudf_table_view{{expect_col1, expect_col2}};

  CUDF_TEST_EXPECT_TABLES_EQUAL(expect_cudf_table_view, tableView);
}

TEST_F(ApplyBooleanMask, FixedPointLargeColumnTest)
{
  cudf::size_type const num_rows = 10000;

  using decimal32_wrapper = cudf::test::fixed_point_column_wrapper<int32_t>;
  using decimal64_wrapper = cudf::test::fixed_point_column_wrapper<int64_t>;

  std::vector<int32_t> dec32_data(num_rows);
  std::vector<int64_t> dec64_data(num_rows);
  std::vector<bool> mask_data(num_rows);

  cudf::test::UniformRandomGenerator<int32_t> rng32(-10000000, 10000000);
  cudf::test::UniformRandomGenerator<int64_t> rng64(-1000000000000, 1000000000000);
  cudf::test::UniformRandomGenerator<bool> rbg;
  std::generate(dec32_data.begin(), dec32_data.end(), [&rng32]() { return rng32.generate(); });
  std::generate(dec64_data.begin(), dec64_data.end(), [&rng64]() { return rng64.generate(); });
  std::generate(mask_data.begin(), mask_data.end(), [&rbg]() { return rbg.generate(); });

  decimal32_wrapper col32(dec32_data.begin(), dec32_data.end(), numeric::scale_type{-3});
  decimal64_wrapper col64(dec64_data.begin(), dec64_data.end(), numeric::scale_type{-10});
  cudf::table_view cudf_table_in_view{{col32, col64}};

  cudf::test::fixed_width_column_wrapper<bool> bool_filter(mask_data.begin(), mask_data.end());
  cudf::column_view bool_filter_col(bool_filter);

  std::unique_ptr<cudf::table> filteredTable =
    cudf::apply_boolean_mask(cudf_table_in_view, bool_filter_col);
  cudf::table_view tableView = filteredTable->view();

  std::vector<int32_t> expect_dec32_data;
  std::vector<int64_t> expect_dec64_data;
  thrust::copy_if(thrust::seq,
                  dec32_data.cbegin(),
                  dec32_data.cend(),
                  mask_data.cbegin(),
                  std::back_inserter(expect_dec32_data),
                  cuda::std::identity{});
  thrust::copy_if(thrust::seq,
                  dec64_data.cbegin(),
                  dec64_data.cend(),
                  mask_data.cbegin(),
                  std::back_inserter(expect_dec64_data),
                  cuda::std::identity{});

  decimal32_wrapper expect_col32(
    expect_dec32_data.begin(), expect_dec32_data.end(), numeric::scale_type{-3});
  decimal64_wrapper expect_col64(
    expect_dec64_data.begin(), expect_dec64_data.end(), numeric::scale_type{-10});
  cudf::table_view expect_cudf_table_view{{expect_col32, expect_col64}};

  CUDF_TEST_EXPECT_TABLES_EQUAL(expect_cudf_table_view, tableView);
}

TEST_F(ApplyBooleanMask, NoNullInput)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col(
    {9668, 9590, 9526, 9205, 9434, 9347, 9160, 9569, 9143, 9807, 9606, 9446, 9279, 9822, 9691});
  cudf::test::fixed_width_column_wrapper<bool> mask({false,
                                                     false,
                                                     true,
                                                     false,
                                                     false,
                                                     true,
                                                     false,
                                                     true,
                                                     false,
                                                     true,
                                                     false,
                                                     false,
                                                     true,
                                                     false,
                                                     true});
  cudf::table_view input({col});
  cudf::test::fixed_width_column_wrapper<int32_t> col_expected(
    {9526, 9347, 9569, 9807, 9279, 9691});
  cudf::table_view expected({col_expected});
  auto got = cudf::apply_boolean_mask(input, mask);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(ApplyBooleanMask, CorrectNullCount)
{
  cudf::size_type inputRows = 471234;

  auto seq1       = cuda::counting_iterator{0};
  auto valid_seq1 = cudf::test::iterators::no_nulls();
  cudf::test::fixed_width_column_wrapper<int64_t, typename decltype(seq1)::value_type> col1(
    seq1, seq1 + inputRows, valid_seq1);

  cudf::table_view input{{col1}};

  auto seq3 =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i % 277) == 0; });
  cudf::test::fixed_width_column_wrapper<bool> boolean_mask(seq3, seq3 + inputRows);

  auto got                 = cudf::apply_boolean_mask(input, boolean_mask);
  auto out_col             = got->get_column(0).view();
  auto expected_null_count = cudf::null_count(out_col.null_mask(), 0, out_col.size());

  ASSERT_EQ(out_col.null_count(), expected_null_count);
}

TEST_F(ApplyBooleanMask, StructFiltering)
{
  using namespace cudf::test;

  auto int_member = fixed_width_column_wrapper<int32_t>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                        {1, 1, 1, 1, 0, 1, 1, 1, 1, 0}};

  auto struct_column = structs_column_wrapper{{int_member}, {0, 1, 1, 1, 1, 0, 1, 1, 1, 1}};

  auto filter_mask = fixed_width_column_wrapper<bool>{{1, 1, 1, 1, 1, 0, 0, 0, 0, 0}};

  auto filtered_table = cudf::apply_boolean_mask(cudf::table_view({struct_column}), filter_mask);
  auto filtered_struct_column = filtered_table->get_column(0);

  // Compare against expected results.
  auto expected_int_member =
    fixed_width_column_wrapper<int32_t>{{-1, 1, 2, 3, -1}, {0, 1, 1, 1, 0}};

  auto expected_struct_column = structs_column_wrapper{{expected_int_member}, {1, 1, 1, 1, 0}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(filtered_struct_column, expected_struct_column);
}

TEST_F(ApplyBooleanMask, ListOfStructsFiltering)
{
  using namespace cudf::test;

  auto key_member = fixed_width_column_wrapper<int32_t>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                        {1, 1, 1, 1, 0, 1, 1, 1, 1, 0}};

  auto value_member = fixed_width_column_wrapper<int32_t>{{0, 10, 20, 30, 40, 50, 60, 70, 80, 90},
                                                          {1, 1, 1, 1, 0, 1, 1, 1, 1, 0}};

  auto struct_column =
    structs_column_wrapper{{key_member, value_member}, {0, 1, 1, 1, 1, 0, 1, 1, 1, 1}};

  auto list_of_structs_column =
    cudf::make_lists_column(5,
                            fixed_width_column_wrapper<int32_t>{0, 2, 4, 6, 8, 10}.release(),
                            struct_column.release(),
                            0,
                            {});

  auto filter_mask = fixed_width_column_wrapper<bool>{{1, 0, 1, 0, 1}};

  auto filtered_table =
    cudf::apply_boolean_mask(cudf::table_view({list_of_structs_column->view()}), filter_mask);
  auto filtered_list_column = filtered_table->get_column(0);

  // Compare against expected values.
  auto expected_key_column =
    fixed_width_column_wrapper<int32_t>{{0, 1, 4, 5, 8, 9}, {0, 1, 0, 0, 1, 0}};
  auto expected_value_column =
    fixed_width_column_wrapper<int32_t>{{0, 10, 40, 50, 80, 90}, {0, 1, 0, 0, 1, 0}};

  auto expected_struct_column =
    structs_column_wrapper{{expected_key_column, expected_value_column}, {0, 1, 1, 0, 1, 1}};

  auto expected_list_of_structs_column =
    cudf::make_lists_column(3,
                            fixed_width_column_wrapper<int32_t>{0, 2, 4, 6}.release(),
                            expected_struct_column.release(),
                            0,
                            {});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(filtered_list_column,
                                      expected_list_of_structs_column->view());
}

TEST_F(ApplyBooleanMask, StructOfListsFiltering)
{
  using namespace cudf::test;

  auto lists_column =
    lists_column_wrapper<int32_t>{{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}}, iterators::null_at(2)};

  auto structs_column = structs_column_wrapper{{lists_column}};

  auto filter_mask    = fixed_width_column_wrapper<bool>{{1, 0, 1, 0, 1}};
  auto filtered_table = cudf::apply_boolean_mask(cudf::table_view({structs_column}), filter_mask);

  auto filtered_lists_column = filtered_table->get_column(0);

  // Compare against expected values;

  auto expected_lists_column =
    lists_column_wrapper<int32_t>{{{0, 0}, {2, 2}, {4, 4}}, iterators::null_at(1)};

  auto expected_structs_column = structs_column_wrapper{{expected_lists_column}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(filtered_lists_column, expected_structs_column);
}

struct ApplyDeletionMask : public cudf::test::BaseFixture {};

TEST_F(ApplyDeletionMask, NonNullDeletionMask)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::table_view input{{col1, col2, col3}};
  cudf::test::fixed_width_column_wrapper<bool> deletion_mask{
    {true, false, true, false, true, false}};
  cudf::test::fixed_width_column_wrapper<int16_t> col1_expected{{false, false, false}, {1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{{40, 5, 10}, {1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<double> col3_expected{{40, 5, 10}, {1, 1, 0}};
  cudf::table_view expected{{col1_expected, col2_expected, col3_expected}};

  auto got = cudf::apply_deletion_mask(input, deletion_mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(ApplyDeletionMask, NullDeletionMask)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::table_view input{{col1, col2, col3}};
  cudf::test::fixed_width_column_wrapper<bool> deletion_mask{
    {true, false, true, false, true, false}, {0, 1, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int16_t> col1_expected{{false, false, false}, {1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{{40, 5, 10}, {1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<double> col3_expected{{40, 5, 10}, {1, 1, 0}};
  cudf::table_view expected{{col1_expected, col2_expected, col3_expected}};

  auto got = cudf::apply_deletion_mask(input, deletion_mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(ApplyDeletionMask, EmptyMask)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::table_view input{{col1, col2, col3}};
  cudf::test::fixed_width_column_wrapper<bool> deletion_mask{};
  cudf::test::fixed_width_column_wrapper<int16_t> col1_expected{};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{};
  cudf::test::fixed_width_column_wrapper<double> col3_expected{};
  cudf::table_view expected{{col1_expected, col2_expected, col3_expected}};

  auto got = cudf::apply_deletion_mask(input, deletion_mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(ApplyDeletionMask, WrongMaskType)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {1, 1, 0, 1, 1, 0}};
  cudf::table_view input{{col1}};
  cudf::test::fixed_width_column_wrapper<int16_t> deletion_mask{
    {true, false, true, false, true, false}};

  EXPECT_THROW(cudf::apply_deletion_mask(input, deletion_mask), cudf::logic_error);
}

TEST_F(ApplyDeletionMask, MaskAndInputSizeMismatch)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {1, 1, 0, 1, 1, 0}};
  cudf::table_view input{{col1}};
  cudf::test::fixed_width_column_wrapper<bool> deletion_mask{{true, false, true, false, true}};

  EXPECT_THROW(cudf::apply_deletion_mask(input, deletion_mask), cudf::logic_error);
}

TEST_F(ApplyDeletionMask, AllTrue)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col{1, 2, 3, 4, 5};
  cudf::table_view input{{col}};
  cudf::test::fixed_width_column_wrapper<bool> deletion_mask{true, true, true, true, true};
  cudf::test::fixed_width_column_wrapper<int32_t> col_expected{};
  cudf::table_view expected{{col_expected}};

  auto got = cudf::apply_deletion_mask(input, deletion_mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(ApplyDeletionMask, AllFalse)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col{1, 2, 3, 4, 5};
  cudf::table_view input{{col}};
  cudf::test::fixed_width_column_wrapper<bool> deletion_mask{false, false, false, false, false};
  cudf::test::fixed_width_column_wrapper<int32_t> col_expected{1, 2, 3, 4, 5};
  cudf::table_view expected{{col_expected}};

  auto got = cudf::apply_deletion_mask(input, deletion_mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(ApplyDeletionMask, AllDeletedMixedNullAndTrue)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col{1, 2, 3, 4, 5};
  cudf::table_view input{{col}};
  cudf::test::fixed_width_column_wrapper<bool> deletion_mask{{true, true, true, true, true},
                                                             {1, 1, 1, 0, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col_expected{};
  cudf::table_view expected{{col_expected}};

  auto got = cudf::apply_deletion_mask(input, deletion_mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(ApplyDeletionMask, StringColumnTest)
{
  cudf::test::strings_column_wrapper col1{
    {"This", "is", "the", "a", "k12", "string", "table", "column"}, {1, 1, 1, 1, 1, 0, 1, 1}};
  cudf::table_view input{{col1}};
  cudf::test::fixed_width_column_wrapper<bool> deletion_mask{
    {true, true, false, true, false, true, false, true}, {1, 1, 0, 1, 1, 1, 1, 1}};
  cudf::test::strings_column_wrapper col1_expected(
    std::initializer_list<std::string>{"k12", "table"}, std::initializer_list<bool>{1, 1});
  cudf::table_view expected{{col1_expected}};

  auto got = cudf::apply_deletion_mask(input, deletion_mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(ApplyDeletionMask, NoNullInput)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col(
    {9668, 9590, 9526, 9205, 9434, 9347, 9160, 9569, 9143, 9807, 9606, 9446, 9279, 9822, 9691});
  cudf::test::fixed_width_column_wrapper<bool> mask({false,
                                                     false,
                                                     true,
                                                     false,
                                                     false,
                                                     true,
                                                     false,
                                                     true,
                                                     false,
                                                     true,
                                                     false,
                                                     false,
                                                     true,
                                                     false,
                                                     true});
  cudf::table_view input({col});
  cudf::test::fixed_width_column_wrapper<int32_t> col_expected(
    {9668, 9590, 9205, 9434, 9160, 9143, 9606, 9446, 9822});
  cudf::table_view expected({col_expected});
  auto got = cudf::apply_deletion_mask(input, mask);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(ApplyDeletionMask, StructFiltering)
{
  using namespace cudf::test;

  auto int_member = fixed_width_column_wrapper<int32_t>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                        {1, 1, 1, 1, 0, 1, 1, 1, 1, 0}};

  auto struct_column = structs_column_wrapper{{int_member}, {0, 1, 1, 1, 1, 0, 1, 1, 1, 1}};

  auto deletion_mask = fixed_width_column_wrapper<bool>{{1, 1, 1, 1, 1, 0, 0, 0, 0, 0}};

  auto filtered_table = cudf::apply_deletion_mask(cudf::table_view({struct_column}), deletion_mask);
  auto filtered_struct_column = filtered_table->get_column(0);

  auto expected_int_member =
    fixed_width_column_wrapper<int32_t>{{-1, 6, 7, 8, -1}, {0, 1, 1, 1, 0}};

  auto expected_struct_column = structs_column_wrapper{{expected_int_member}, {0, 1, 1, 1, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(filtered_struct_column, expected_struct_column);
}

CUDF_TEST_PROGRAM_MAIN()
