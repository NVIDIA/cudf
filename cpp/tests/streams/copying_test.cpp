/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>

#include <cuda/iterator>

#include <limits>

class CopyingTest : public cudf::test::BaseFixture {};

TEST_F(CopyingTest, Gather)
{
  constexpr cudf::size_type source_size{1000};

  auto data = cuda::counting_iterator{0};
  cudf::test::fixed_width_column_wrapper<int32_t> source_column(data, data + source_size);
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(data, data + source_size);

  cudf::table_view source_table({source_column});

  cudf::gather(source_table,
               gather_map,
               cudf::out_of_bounds_policy::DONT_CHECK,
               cudf::test::get_default_stream());
}

TEST_F(CopyingTest, ReverseTable)
{
  constexpr cudf::size_type num_values{10};

  auto input = cudf::test::fixed_width_column_wrapper<int32_t, int32_t>(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{0} + num_values);

  auto input_table = cudf::table_view{{input}};
  cudf::reverse(input_table, cudf::test::get_default_stream());
}

TEST_F(CopyingTest, ReverseColumn)
{
  constexpr cudf::size_type num_values{10};

  auto input = cudf::test::fixed_width_column_wrapper<int32_t, int32_t>(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{0} + num_values);

  cudf::reverse(input, cudf::test::get_default_stream());
}

TEST_F(CopyingTest, ScatterTable)
{
  cudf::test::fixed_width_column_wrapper<int32_t> source({1, 2, 3, 4, 5, 6});
  cudf::test::fixed_width_column_wrapper<int32_t> target({10, 20, 30, 40, 50, 60, 70, 80});
  cudf::test::fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1});

  auto const source_table = cudf::table_view({source, source});
  auto const target_table = cudf::table_view({target, target});

  cudf::scatter(source_table, scatter_map, target_table, cudf::test::get_default_stream());
}

TEST_F(CopyingTest, ScatterScalars)
{
  auto const source = cudf::scalar_type_t<int32_t>(100, true, cudf::test::get_default_stream());
  std::reference_wrapper<cudf::scalar const> slr_ref{source};
  std::vector<std::reference_wrapper<cudf::scalar const>> source_vector{slr_ref};

  cudf::test::fixed_width_column_wrapper<int32_t> target({10, 20, 30, 40, 50, 60, 70, 80});
  cudf::test::fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1});

  auto const target_table = cudf::table_view({target});

  cudf::scatter(source_vector, scatter_map, target_table, cudf::test::get_default_stream());
}

TEST_F(CopyingTest, AllocateLike)
{
  // For same size as input
  cudf::size_type size = 10;

  auto input = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<int32_t>()},
                                         size,
                                         cudf::mask_state::UNALLOCATED,
                                         cudf::test::get_default_stream());
  cudf::allocate_like(
    input->view(), cudf::mask_allocation_policy::RETAIN, cudf::test::get_default_stream());
}

TEST_F(CopyingTest, AllocateLikeSize)
{
  // For same size as input
  cudf::size_type size     = 10;
  cudf::size_type new_size = 10;

  auto input = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<int32_t>()},
                                         size,
                                         cudf::mask_state::UNALLOCATED,
                                         cudf::test::get_default_stream());
  cudf::allocate_like(input->view(),
                      new_size,
                      cudf::mask_allocation_policy::RETAIN,
                      cudf::test::get_default_stream());
}

TEST_F(CopyingTest, CopyRangeInPlace)
{
  constexpr cudf::size_type size{1000};

  cudf::test::fixed_width_column_wrapper<int32_t, int32_t> target(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{0} + size);

  auto source_elements =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });
  cudf::test::fixed_width_column_wrapper<int32_t, typename decltype(source_elements)::value_type>
    source(source_elements, source_elements + size);

  cudf::mutable_column_view target_view{target};

  constexpr cudf::size_type source_begin{9};
  constexpr cudf::size_type source_end{size - 50};
  constexpr cudf::size_type target_begin{30};
  cudf::copy_range_in_place(
    source, target_view, source_begin, source_end, target_begin, cudf::test::get_default_stream());
}

TEST_F(CopyingTest, CopyRange)
{
  constexpr cudf::size_type size{1000};

  cudf::test::fixed_width_column_wrapper<int32_t, int32_t> target(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{0} + size);

  auto source_elements =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });
  cudf::test::fixed_width_column_wrapper<int32_t, typename decltype(source_elements)::value_type>
    source(source_elements, source_elements + size);

  cudf::mutable_column_view target_view{target};
  cudf::column_view const immutable_view{target_view};

  cudf::size_type source_begin{9};
  cudf::size_type source_end{size - 50};
  cudf::size_type target_begin{30};
  cudf::copy_range(source,
                   immutable_view,
                   source_begin,
                   source_end,
                   target_begin,
                   cudf::test::get_default_stream());
}

TEST_F(CopyingTest, Shift)
{
  auto input =
    cudf::test::fixed_width_column_wrapper<int32_t>{std::numeric_limits<int32_t>::min(),
                                                    cudf::test::make_type_param_scalar<int32_t>(1),
                                                    cudf::test::make_type_param_scalar<int32_t>(2),
                                                    cudf::test::make_type_param_scalar<int32_t>(3),
                                                    cudf::test::make_type_param_scalar<int32_t>(4),
                                                    cudf::test::make_type_param_scalar<int32_t>(5),
                                                    std::numeric_limits<int32_t>::max()};
  auto fill = cudf::scalar_type_t<int32_t>(
    cudf::test::make_type_param_scalar<int32_t>(7), true, cudf::test::get_default_stream());
  cudf::shift(input, 2, fill, cudf::test::get_default_stream());
}

TEST_F(CopyingTest, SliceColumn)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 2, 3, 4, 5};

  std::vector<cudf::size_type> indices{1, 3, 2, 2, 2, 5};
  cudf::slice(col, indices, cudf::test::get_default_stream());
  cudf::slice(col, {1, 3, 2, 2, 2, 5}, cudf::test::get_default_stream());
}

TEST_F(CopyingTest, SliceTable)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 2, 3, 4, 5};

  std::vector<cudf::size_type> indices{1, 3, 2, 2, 2, 5};
  cudf::table_view tbl({col});
  cudf::slice(tbl, indices, cudf::test::get_default_stream());
  cudf::slice(tbl, {1, 3, 2, 2, 2, 5}, cudf::test::get_default_stream());
}

TEST_F(CopyingTest, SplitColumn)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 2, 3, 4, 5};

  std::vector<cudf::size_type> indices{1, 3, 5};
  cudf::split(col, indices, cudf::test::get_default_stream());
  cudf::split(col, {1, 3, 5}, cudf::test::get_default_stream());
}

TEST_F(CopyingTest, SplitTable)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 2, 3, 4, 5};

  std::vector<cudf::size_type> indices{1, 3, 5};
  cudf::table_view tbl({col});
  cudf::split(tbl, indices, cudf::test::get_default_stream());
  cudf::split(tbl, {1, 3, 5}, cudf::test::get_default_stream());
}

TEST_F(CopyingTest, CopyIfElseColumnColumn)
{
  cudf::test::fixed_width_column_wrapper<bool> mask_w{1, 0, 0, 0};
  cudf::test::fixed_width_column_wrapper<int32_t, int32_t> lhs_w{5, 5, 5, 5};
  cudf::test::fixed_width_column_wrapper<int32_t, int32_t> rhs_w{6, 6, 6, 6};
  cudf::copy_if_else(lhs_w, rhs_w, mask_w, cudf::test::get_default_stream());
}

TEST_F(CopyingTest, CopyIfElseScalarColumn)
{
  auto scalar = cudf::scalar_type_t<int32_t>(
    cudf::test::make_type_param_scalar<int32_t>(7), true, cudf::test::get_default_stream());
  cudf::test::fixed_width_column_wrapper<int32_t, int32_t> column{5, 5, 5, 5};
  cudf::test::fixed_width_column_wrapper<bool> mask_w{1, 0, 0, 0};
  cudf::copy_if_else(scalar, column, mask_w, cudf::test::get_default_stream());
}

TEST_F(CopyingTest, CopyIfElseColumnScalar)
{
  auto scalar = cudf::scalar_type_t<int32_t>(
    cudf::test::make_type_param_scalar<int32_t>(7), true, cudf::test::get_default_stream());
  cudf::test::fixed_width_column_wrapper<int32_t, int32_t> column{5, 5, 5, 5};
  cudf::test::fixed_width_column_wrapper<bool> mask_w{1, 0, 0, 0};
  cudf::copy_if_else(column, scalar, mask_w, cudf::test::get_default_stream());
}

TEST_F(CopyingTest, CopyIfElseScalarScalar)
{
  auto lhs = cudf::scalar_type_t<int32_t>(
    cudf::test::make_type_param_scalar<int32_t>(7), true, cudf::test::get_default_stream());
  auto rhs = cudf::scalar_type_t<int32_t>(
    cudf::test::make_type_param_scalar<int32_t>(6), true, cudf::test::get_default_stream());
  cudf::test::fixed_width_column_wrapper<bool> mask_w{1, 0, 0, 0};
  cudf::copy_if_else(lhs, rhs, mask_w, cudf::test::get_default_stream());
}

TEST_F(CopyingTest, BooleanMaskScatter)
{
  cudf::test::fixed_width_column_wrapper<int32_t, int32_t> source({1, 5, 6, 8, 9});
  cudf::test::fixed_width_column_wrapper<int32_t, int32_t> target(
    {2, 2, 3, 4, 11, 12, 7, 7, 10, 10});
  cudf::test::fixed_width_column_wrapper<bool> mask(
    {true, false, false, false, true, true, false, true, true, false});

  auto source_table = cudf::table_view({source});
  auto target_table = cudf::table_view({target});

  cudf::boolean_mask_scatter(source_table, target_table, mask, cudf::test::get_default_stream());
}

TEST_F(CopyingTest, BooleanMaskScatterScalars)
{
  std::vector<std::reference_wrapper<cudf::scalar const>> scalars;
  auto s = cudf::scalar_type_t<int32_t>(1, true, cudf::test::get_default_stream());
  scalars.emplace_back(s);
  cudf::test::fixed_width_column_wrapper<int32_t, int32_t> target(
    {2, 2, 3, 4, 11, 12, 7, 7, 10, 10});
  cudf::test::fixed_width_column_wrapper<bool> mask(
    {true, false, false, false, true, true, false, true, true, false});

  auto target_table = cudf::table_view({target});

  cudf::boolean_mask_scatter(scalars, target_table, mask, cudf::test::get_default_stream());
}

TEST_F(CopyingTest, GetElement)
{
  cudf::test::fixed_width_column_wrapper<int32_t> _col{1, 2};
  cudf::get_element(_col, 0, cudf::test::get_default_stream());
}

TEST_F(CopyingTest, Sample)
{
  cudf::size_type const table_size = 1024;
  auto const n_samples             = 10;
  auto const multi_smpl            = cudf::sample_with_replacement::FALSE;

  auto data = cuda::counting_iterator{0};
  cudf::test::fixed_width_column_wrapper<int16_t> col1(data, data + table_size);

  cudf::table_view input({col1});
  cudf::sample(input, n_samples, multi_smpl, 0, cudf::test::get_default_stream());
}

template <typename T>
using LCW = cudf::test::lists_column_wrapper<T, int32_t>;

TEST_F(CopyingTest, HasNonemptyNulls)
{
  auto const input =
    LCW<int32_t>{{{{1, 2, 3, 4}, cudf::test::iterators::null_at(2)}, {5}, {}, {8, 9, 10}},
                 cudf::test::iterators::no_nulls()}
      .release();
  cudf::has_nonempty_nulls(*input, cudf::test::get_default_stream());
}

TEST_F(CopyingTest, PurgeNonEmptyNulls)
{
  auto const input = LCW<int32_t>{{{{1, 2, 3, 4}, cudf::test::iterators::null_at(2)},
                                   {5},
                                   {6, 7},  // <--- Will be set to NULL. Unsanitized row.
                                   {8, 9, 10}},
                                  cudf::test::iterators::no_nulls()}
                       .release();

  // Set nullmask, post construction.
  cudf::set_null_mask(
    input->mutable_view().null_mask(), 2, 3, false, cudf::test::get_default_stream());
  input->set_null_count(1);

  cudf::purge_nonempty_nulls(*input, cudf::test::get_default_stream());
}

TEST_F(CopyingTest, ContiguousSplit)
{
  std::vector<cudf::size_type> splits{
    2, 16, 31, 35, 64, 97, 158, 190, 638, 899, 900, 901, 996, 4200, 7131, 8111};

  cudf::size_type size = 10002;
  auto iter            = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return static_cast<double>(i); });

  std::vector<std::string> base_strings(
    {"banana", "pear", "apple", "pecans", "vanilla", "cat", "mouse", "green"});
  auto string_randomizer = cuda::transform_iterator(
    cuda::counting_iterator<cudf::size_type>{0},
    [&base_strings](cudf::size_type i) { return base_strings[rand() % base_strings.size()]; });

  cudf::test::fixed_width_column_wrapper<double> col(iter, iter + size);
  std::vector<std::string> strings(string_randomizer, string_randomizer + size);
  cudf::test::strings_column_wrapper col2(strings.begin(), strings.end());
  cudf::table_view tbl({col, col2});
  auto result = cudf::contiguous_split(tbl, splits, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
