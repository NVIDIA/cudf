/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/transform/udf.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/udf.hpp>

#include <array>

class TransformTest : public cudf::test::BaseFixture {};

template <class dtype, class Data>
void test_udf(cudf::transform_udf udf, Data data_init, cudf::size_type size)
{
  auto all_valid = cudf::test::iterators::no_nulls();
  auto data_iter = cudf::detail::make_counting_transform_iterator(0, data_init);
  cudf::test::fixed_width_column_wrapper<dtype, typename decltype(data_iter)::value_type> in(
    data_iter, data_iter + size, all_valid);

  cudf::transform_input inputs[] = {in};

  cudf::transform(udf,
                  inputs,
                  std::array{cudf::transform_output{cudf::data_type(cudf::type_to_id<dtype>()),
                                                    cudf::output_nullability::PRESERVE}},
                  {},
                  std::nullopt,
                  cudf::test::get_default_stream());
}

TEST_F(TransformTest, Transform)
{
  auto data_init = [](cudf::size_type row) { return row % 3; };
  test_udf<float>(cudf::test::get_transform_udf("fourth_power"), data_init, 500);
  test_udf<float>(cudf::test::get_ptx_transform_udf("fourth_power"), data_init, 500);
}

TEST_F(TransformTest, ComputeColumn)
{
  auto c_0        = cudf::test::fixed_width_column_wrapper<cudf::size_type>{3, 20, 1, 50};
  auto c_1        = cudf::test::fixed_width_column_wrapper<cudf::size_type>{10, 7, 20, 0};
  auto table      = cudf::table_view{{c_0, c_1}};
  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);
  cudf::compute_column(table, expression, cudf::test::get_default_stream());
}

TEST_F(TransformTest, ComputeColumnJIT)
{
  auto c_0        = cudf::test::fixed_width_column_wrapper<cudf::size_type>{3, 20, 1, 50};
  auto c_1        = cudf::test::fixed_width_column_wrapper<cudf::size_type>{10, 7, 20, 0};
  auto table      = cudf::table_view{{c_0, c_1}};
  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);
  cudf::compute_column_jit(table, expression, {}, cudf::test::get_default_stream());
}

TEST_F(TransformTest, BoolsToMask)
{
  std::vector<bool> input({1, 0, 1, 0, 1, 0, 1, 0});
  cudf::test::fixed_width_column_wrapper<bool> input_column(input.begin(), input.end());
  cudf::bools_to_mask(input_column, cudf::test::get_default_stream());
}

TEST_F(TransformTest, MaskToBools)
{
  cudf::mask_to_bools(nullptr, 0, 0, cudf::test::get_default_stream());
}

TEST_F(TransformTest, Encode)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> input{{1, 2, 3, 2, 3, 2, 1}};
  cudf::encode(cudf::table_view({input}), cudf::test::get_default_stream());
}

TEST_F(TransformTest, OneHotEncode)
{
  auto input    = cudf::test::fixed_width_column_wrapper<cudf::size_type>{8, 8, 8, 9, 9};
  auto category = cudf::test::fixed_width_column_wrapper<cudf::size_type>{8, 9};
  cudf::one_hot_encode(input, category, cudf::test::get_default_stream());
}

TEST_F(TransformTest, NaNsToNulls)
{
  std::vector<float> input = {1, 2, 3, 4, 5};
  std::vector<bool> mask   = {true, true, true, true, false, false};
  auto input_column =
    cudf::test::fixed_width_column_wrapper<float>(input.begin(), input.end(), mask.begin());
  cudf::column_nans_to_nulls(input_column, cudf::test::get_default_stream());
}

TEST_F(TransformTest, RowBitCount)
{
  std::vector<std::string> strings{"abc", "ï", "", "z", "bananas", "warp", "", "zing"};
  cudf::test::strings_column_wrapper col(strings.begin(), strings.end());
  cudf::row_bit_count(cudf::table_view({col}), cudf::test::get_default_stream());
}

TEST_F(TransformTest, SegmentedRowBitCount)
{
  // clang-format off
  std::vector<std::string> const strings { "daïs", "def", "", "z", "bananas", "warp", "", "zing" };
  std::vector<bool>        const valids  {  1,      0,    0,  1,   0,          1,      1,  1 };
  // clang-format on
  cudf::test::strings_column_wrapper const col(strings.begin(), strings.end(), valids.begin());
  auto const input              = cudf::table_view({col});
  auto constexpr segment_length = 2;
  cudf::segmented_row_bit_count(input, segment_length, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
