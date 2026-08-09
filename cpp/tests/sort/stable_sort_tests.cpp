/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <type_traits>
#include <vector>

void run_stable_sort_test(cudf::table_view input,
                          cudf::column_view expected_sorted_indices,
                          std::vector<cudf::order> column_order         = {},
                          std::vector<cudf::null_order> null_precedence = {},
                          bool by_key                                   = true)
{
  auto got      = by_key ? cudf::stable_sort_by_key(input, input, column_order, null_precedence)
                         : cudf::stable_sort(input, column_order, null_precedence);
  auto expected = cudf::gather(input, expected_sorted_indices);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), got->view());
}

using TestTypes = cudf::test::Concat<cudf::test::NumericTypes,  // include integers, floats and bool
                                     cudf::test::ChronoTypes>;  // include timestamps and durations

template <typename T>
struct StableSort : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(StableSort, TestTypes);

TYPED_TEST(StableSort, MixedNullOrder)
{
  using T = TypeParam;
  using R = cudf::size_type;

  cudf::test::fixed_width_column_wrapper<T> col1({0, 1, 1, 0, 0, 1, 0, 1},
                                                 {0, 1, 1, 1, 1, 1, 1, 1});
  cudf::test::strings_column_wrapper col2({"2", "a", "b", "x", "k", "a", "x", "a"},
                                          {true, true, true, true, false, true, true, true});

  cudf::test::fixed_width_column_wrapper<R> expected{{4, 3, 6, 1, 5, 7, 2, 0}};

  auto got = cudf::stable_sorted_order(cudf::table_view({col1, col2}),
                                       {cudf::order::ASCENDING, cudf::order::ASCENDING},
                                       {cudf::null_order::AFTER, cudf::null_order::BEFORE});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(StableSort, WithNullMax)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8, 5}, {1, 1, 0, 1, 1, 1}};
  cudf::test::strings_column_wrapper col2({"d", "e", "a", "d", "k", "d"},
                                          {true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<T> col3{{10, 40, 70, 10, 2, 10}, {1, 1, 0, 1, 1, 1}};
  cudf::table_view input{{col1, col2, col3}};

  std::vector<cudf::order> column_order{
    cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::DESCENDING};
  std::vector<cudf::null_order> null_precedence{
    cudf::null_order::AFTER, cudf::null_order::AFTER, cudf::null_order::AFTER};
  auto expected = std::is_same_v<T, bool>
                    // All the bools are true, and therefore don't affect sort order,
                    // so this is just the sort order of the nullable string column
                    ? cudf::test::fixed_width_column_wrapper<cudf::size_type>{{0, 3, 5, 1, 4, 2}}
                    : cudf::test::fixed_width_column_wrapper<cudf::size_type>{{1, 0, 3, 5, 4, 2}};

  auto got = cudf::stable_sorted_order(input, column_order, null_precedence);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  run_stable_sort_test(input, expected, column_order, null_precedence, false);
  run_stable_sort_test(input, expected, column_order, null_precedence, true);
}

TYPED_TEST(StableSort, SingleColumnNoNull)
{
  // This test exercises the "fast-path" single column sort.
  using T = TypeParam;
  //                                             0  1   2  3  4  5  6   7  8  9
  cudf::test::fixed_width_column_wrapper<T> col{{7, 1, -2, 5, 1, 0, 1, -2, 0, 5}};
  cudf::table_view input{{col}};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};
  auto expected =
    std::is_same_v<T, bool>
      ? cudf::test::fixed_width_column_wrapper<cudf::size_type>{{8, 5, 0, 1, 2, 3, 4, 6, 7, 9}}
    : std::is_unsigned_v<T>
      ? cudf::test::fixed_width_column_wrapper<cudf::size_type>{{5, 8, 1, 4, 6, 3, 9, 0, 2, 7}}
      : cudf::test::fixed_width_column_wrapper<cudf::size_type>{{2, 7, 5, 8, 1, 4, 6, 3, 9, 0}};
  run_stable_sort_test(input, expected, column_order, {}, false);
  run_stable_sort_test(input, expected, column_order, {}, true);
}

TYPED_TEST(StableSort, SingleColumnWithNull)
{
  using T = TypeParam;
  //                                             0  1   2  3  4  5  6   7  8  9
  cudf::test::fixed_width_column_wrapper<T> col{{7, 1, -2, 5, 1, 0, 1, -2, 0, 5},
                                                {1, 1, 0, 0, 1, 0, 1, 0, 1, 0}};
  cudf::table_view input{{col}};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::BEFORE};
  auto expected =
    std::is_same_v<T, bool>
      ? cudf::test::fixed_width_column_wrapper<cudf::size_type>{{5, 2, 3, 7, 9, 8, 0, 1, 4, 6}}
    : std::is_unsigned_v<T>
      ? cudf::test::fixed_width_column_wrapper<cudf::size_type>{{5, 3, 9, 2, 7, 8, 1, 4, 6, 0}}
      : cudf::test::fixed_width_column_wrapper<cudf::size_type>{{2, 7, 5, 3, 9, 8, 1, 4, 6, 0}};
  run_stable_sort_test(input, expected, column_order, {}, false);
  run_stable_sort_test(input, expected, column_order, {}, true);
}

TYPED_TEST(StableSort, WithNullMin)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8}, {1, 1, 0, 1, 1}};
  cudf::test::strings_column_wrapper col2({"d", "e", "a", "d", "k"},
                                          {true, true, false, true, true});
  cudf::test::fixed_width_column_wrapper<T> col3{{10, 40, 70, 10, 2}, {1, 1, 0, 1, 1}};
  cudf::table_view input{{col1, col2, col3}};

  std::vector<cudf::order> column_order{
    cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::DESCENDING};
  auto expected = std::is_same_v<T, bool>
                    // All the bools are true, and therefore don't affect sort order,
                    // so this is just the sort order of the string column
                    ? cudf::test::fixed_width_column_wrapper<cudf::size_type>{{2, 0, 3, 1, 4}}
                    : cudf::test::fixed_width_column_wrapper<cudf::size_type>{{2, 1, 0, 3, 4}};
  auto got      = cudf::stable_sorted_order(input, column_order);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  run_stable_sort_test(input, expected, column_order, {}, false);
  run_stable_sort_test(input, expected, column_order, {}, true);
}

TYPED_TEST(StableSort, WithAllValid)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8}};
  cudf::test::strings_column_wrapper col2({"d", "e", "a", "d", "k"});
  cudf::test::fixed_width_column_wrapper<T> col3{{10, 40, 70, 10, 2}};
  cudf::table_view input{{col1, col2, col3}};

  std::vector<cudf::order> column_order{
    cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::DESCENDING};
  auto expected = std::is_same_v<T, bool>
                    // All the bools are true, and therefore don't affect sort order,
                    // so this is just the sort order of the string column
                    ? cudf::test::fixed_width_column_wrapper<cudf::size_type>{{2, 0, 3, 1, 4}}
                    : cudf::test::fixed_width_column_wrapper<cudf::size_type>{{2, 1, 0, 3, 4}};
  auto got      = cudf::stable_sorted_order(input, column_order);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  run_stable_sort_test(input, expected, column_order, {}, false);
  run_stable_sort_test(input, expected, column_order, {}, true);
}

TYPED_TEST(StableSort, MisMatchInColumnOrderSize)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8}};
  cudf::test::strings_column_wrapper col2({"d", "e", "a", "d", "k"});
  cudf::test::fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2}};
  cudf::table_view input{{col1, col2, col3}};

  std::vector<cudf::order> column_order{cudf::order::ASCENDING, cudf::order::DESCENDING};

  EXPECT_THROW(cudf::stable_sorted_order(input, column_order), cudf::logic_error);
  EXPECT_THROW(cudf::stable_sort_by_key(input, input, column_order), cudf::logic_error);
}

TYPED_TEST(StableSort, MisMatchInNullPrecedenceSize)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8}};
  cudf::test::strings_column_wrapper col2({"d", "e", "a", "d", "k"});
  cudf::test::fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2}};
  cudf::table_view input{{col1, col2, col3}};

  std::vector<cudf::order> column_order{
    cudf::order::ASCENDING, cudf::order::DESCENDING, cudf::order::DESCENDING};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::AFTER, cudf::null_order::BEFORE};

  EXPECT_THROW(cudf::stable_sorted_order(input, column_order, null_precedence), cudf::logic_error);
  EXPECT_THROW(cudf::stable_sort_by_key(input, input, column_order, null_precedence),
               cudf::logic_error);
}

TYPED_TEST(StableSort, ZeroSizedColumns)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col1{};
  cudf::table_view input{{col1}};

  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected{};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};

  auto got = cudf::stable_sorted_order(input, column_order);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  run_stable_sort_test(input, expected, column_order);
}

struct StableSortByKey : public cudf::test::BaseFixture {};

TEST_F(StableSortByKey, ValueKeysSizeMismatch)
{
  using T = int64_t;

  cudf::test::fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8}};
  cudf::test::strings_column_wrapper col2({"d", "e", "a", "d", "k"});
  cudf::test::fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2}};
  cudf::table_view values{{col1, col2, col3}};

  cudf::test::fixed_width_column_wrapper<T> key_col{{5, 4, 3, 5}};
  cudf::table_view keys{{key_col}};

  EXPECT_THROW(cudf::stable_sort_by_key(values, keys), cudf::logic_error);
}

template <typename T>
struct StableSortFixedPoint : public cudf::test::BaseFixture {};

template <typename T>
using wrapper = cudf::test::fixed_width_column_wrapper<T>;
TYPED_TEST_SUITE(StableSortFixedPoint, cudf::test::FixedPointTypes);

TYPED_TEST(StableSortFixedPoint, FixedPointSortedOrderGather)
{
  using namespace numeric;
  using decimalXX = TypeParam;

  auto const ZERO  = decimalXX{0, scale_type{0}};
  auto const ONE   = decimalXX{1, scale_type{0}};
  auto const TWO   = decimalXX{2, scale_type{0}};
  auto const THREE = decimalXX{3, scale_type{0}};
  auto const FOUR  = decimalXX{4, scale_type{0}};

  auto const input_vec  = std::vector<decimalXX>{THREE, TWO, ONE, ZERO, FOUR, THREE};
  auto const index_vec  = std::vector<cudf::size_type>{3, 2, 1, 0, 5, 4};
  auto const sorted_vec = std::vector<decimalXX>{ZERO, ONE, TWO, THREE, THREE, FOUR};

  auto const input_col  = wrapper<decimalXX>(input_vec.begin(), input_vec.end());
  auto const index_col  = wrapper<cudf::size_type>(index_vec.begin(), index_vec.end());
  auto const sorted_col = wrapper<decimalXX>(sorted_vec.begin(), sorted_vec.end());

  auto const sorted_table = cudf::table_view{{sorted_col}};
  auto const input_table  = cudf::table_view{{input_col}};

  auto const indices = cudf::sorted_order(input_table);
  auto const sorted  = cudf::gather(input_table, indices->view());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(index_col, indices->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted_table, sorted->view());
}

using StableSortDouble = StableSort<double>;
TEST_F(StableSortDouble, InfinityAndNaN)
{
  auto constexpr NaN = std::numeric_limits<double>::quiet_NaN();
  auto constexpr Inf = std::numeric_limits<double>::infinity();

  auto input = cudf::test::fixed_width_column_wrapper<double>(
    {-0.0, -NaN, -NaN, NaN, Inf, -Inf, 7.0, 5.0, 6.0, NaN, Inf, -Inf, -NaN, -NaN, -0.0});
  auto expected =  // -inf,-inf,-0,-0,5,6,7,inf,inf,-nan,-nan,nan,nan,-nan,-nan
    cudf::test::fixed_width_column_wrapper<cudf::size_type>(
      {5, 11, 0, 14, 7, 8, 6, 4, 10, 1, 2, 3, 9, 12, 13});
  auto results = stable_sorted_order(cudf::table_view({input}));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}
