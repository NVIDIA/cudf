/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cuda/iterator>

#include <numeric>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

template <typename T>
class GatherTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(GatherTest, cudf::test::NumericTypes);

struct GatherZeroColumnTest : public cudf::test::BaseFixture {};

TEST_F(GatherZeroColumnTest, PreservesRowCount)
{
  cudf::table_view source{std::vector<cudf::column_view>{}, 5};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{{0, 2, 4, 1}};
  auto result = cudf::gather(source, gather_map);
  EXPECT_EQ(result->num_columns(), 0);
  EXPECT_EQ(result->num_rows(), 4);
}

TYPED_TEST(GatherTest, IdentityTest)
{
  constexpr cudf::size_type source_size{1000};

  auto data = cuda::counting_iterator{0};
  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(data, data + source_size);
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(data, data + source_size);

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

TYPED_TEST(GatherTest, GatherEveryIdentityTest)
{
  constexpr cudf::size_type source_size{1000};

  auto data = cuda::counting_iterator{0};
  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(data, data + source_size);

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = cudf::gather_every(source_table, 1);

  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

TYPED_TEST(GatherTest, GatherEveryFencepostTest)
{
  constexpr cudf::size_type source_size{1000};

  // (step, start) pairs chosen to exercise the last selected row: coprime values where the last
  // row lands on, just before, or just after the end of the table, plus steps near the table size.
  std::vector<std::pair<cudf::size_type, cudf::size_type>> const cases{
    {2, 0}, {3, 1}, {7, 5}, {13, 12}, {7, 999}, {999, 0}, {999, 1}, {1000, 0}, {1001, 0}};

  auto data = cuda::counting_iterator{0};
  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(data, data + source_size);
  cudf::table_view source_table({source_column});

  for (auto const& test_case : cases) {
    auto const step        = test_case.first;
    auto const start       = test_case.second;
    auto const result_size = (source_size - start + step - 1) / step;
    auto expected_data     = cudf::detail::make_counting_transform_iterator(
      0, [step, start](auto i) { return start + i * step; });
    cudf::test::fixed_width_column_wrapper<TypeParam> expected_column(expected_data,
                                                                      expected_data + result_size);

    auto const result = cudf::gather_every(source_table, step, start);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column, result->view().column(0));
  }
}

TYPED_TEST(GatherTest, GatherEveryNegativeStepTest)
{
  constexpr cudf::size_type source_size{1000};

  auto data = cuda::counting_iterator{0};
  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(data, data + source_size);
  cudf::table_view source_table({source_column});

  for (auto const step : {-1, -3, -7, -999, -1000}) {
    auto const result_size = (source_size + -step - 1) / -step;
    auto expected_data     = cudf::detail::make_counting_transform_iterator(
      0, [step](auto i) { return source_size - 1 + i * step; });
    cudf::test::fixed_width_column_wrapper<TypeParam> expected_column(expected_data,
                                                                      expected_data + result_size);

    auto const result = cudf::gather_every(source_table, step);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column, result->view().column(0));
  }
}

struct GatherEveryTest : public cudf::test::BaseFixture {};

TEST_F(GatherEveryTest, StartStop)
{
  cudf::test::fixed_width_column_wrapper<int32_t> source_column{0, 1, 2, 3, 4, 5, 6, 7};
  cudf::table_view source_table({source_column});

  // Each case matches Python's `list(range(8))[start:stop:step]`.
  auto const check = [&](cudf::size_type step,
                         std::optional<cudf::size_type> start,
                         std::optional<cudf::size_type> stop,
                         std::vector<int32_t> const& expected) {
    cudf::test::fixed_width_column_wrapper<int32_t> expected_column(expected.begin(),
                                                                    expected.end());
    auto const result = cudf::gather_every(source_table, step, start, stop);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column, result->view().column(0));
  };

  check(2, 1, 6, {1, 3, 5});
  check(2, 1, 7, {1, 3, 5});
  check(3, std::nullopt, 6, {0, 3});
  check(1, -3, std::nullopt, {5, 6, 7});
  check(1, 2, -2, {2, 3, 4, 5});
  check(2, -100, 100, {0, 2, 4, 6});
  check(-1, std::nullopt, std::nullopt, {7, 6, 5, 4, 3, 2, 1, 0});
  check(-2, -2, 1, {6, 4, 2});
  check(-3, 6, std::nullopt, {6, 3, 0});
  check(-1, 100, -100, {7, 6, 5, 4, 3, 2, 1, 0});
  check(-2, 3, -1, {});
}

TEST_F(GatherEveryTest, EmptyResult)
{
  cudf::test::fixed_width_column_wrapper<int32_t> int_column{1, 2, 3};
  cudf::test::strings_column_wrapper string_column{"a", "b", "c"};
  cudf::table_view source_table({int_column, string_column});

  // start at or past the end, stop at or before start, and a reverse slice from before the start
  for (auto const& [step, start, stop] :
       std::vector<std::tuple<cudf::size_type,
                              std::optional<cudf::size_type>,
                              std::optional<cudf::size_type>>>{{1, 3, std::nullopt},
                                                               {1, 100, std::nullopt},
                                                               {1, 2, 2},
                                                               {1, 2, 1},
                                                               {-1, 0, 0},
                                                               {-1, -100, std::nullopt}}) {
    auto const result = cudf::gather_every(source_table, step, start, stop);
    EXPECT_EQ(result->num_rows(), 0);
    CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::empty_like(source_table)->view(), result->view());
  }

  cudf::test::fixed_width_column_wrapper<int32_t> empty_column{};
  cudf::table_view empty_table({empty_column});
  EXPECT_EQ(cudf::gather_every(empty_table, 2)->num_rows(), 0);
  EXPECT_EQ(cudf::gather_every(empty_table, -2)->num_rows(), 0);
}

TEST_F(GatherEveryTest, InvalidArguments)
{
  cudf::test::fixed_width_column_wrapper<int32_t> source_column{1, 2, 3};
  cudf::table_view source_table({source_column});

  EXPECT_THROW(cudf::gather_every(source_table, 0), std::invalid_argument);
  // A zero step is rejected even when the slice would otherwise be empty
  EXPECT_THROW(cudf::gather_every(source_table, 0, 3), std::invalid_argument);

  cudf::test::fixed_width_column_wrapper<int32_t> empty_column{};
  EXPECT_THROW(cudf::gather_every(cudf::table_view({empty_column}), 0), std::invalid_argument);
}

TYPED_TEST(GatherTest, ReverseIdentityTest)
{
  constexpr cudf::size_type source_size{1000};

  auto data = cuda::counting_iterator{0};
  auto reversed_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return source_size - 1 - i; });

  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(data, data + source_size);
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(reversed_data,
                                                             reversed_data + source_size);

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(reversed_data,
                                                                  reversed_data + source_size);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_column, result->view().column(i));
  }
}

TYPED_TEST(GatherTest, EveryOtherNullOdds)
{
  constexpr cudf::size_type source_size{1000};

  // Every other element is valid
  auto data     = cuda::counting_iterator{0};
  auto validity = cudf::test::iterators::nulls_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(
    data, data + source_size, validity);

  // Gather odd-valued indices
  auto map_data = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(map_data,
                                                             map_data + (source_size / 2));

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  auto expect_data  = cuda::constant_iterator{0};
  auto expect_valid = cudf::test::iterators::all_nulls();
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(
    expect_data, expect_data + source_size / 2, expect_valid);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_column, result->view().column(i));
  }
}

TYPED_TEST(GatherTest, EveryOtherNullEvens)
{
  constexpr cudf::size_type source_size{1000};

  // Every other element is valid
  auto data     = cuda::counting_iterator{0};
  auto validity = cudf::test::iterators::nulls_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(
    data, data + source_size, validity);

  // Gather even-valued indices
  auto map_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2 + 1; });

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(map_data,
                                                             map_data + (source_size / 2));

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  auto expect_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2 + 1; });
  auto expect_valid = cudf::test::iterators::no_nulls();
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(
    expect_data, expect_data + source_size / 2, expect_valid);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_column, result->view().column(i));
  }
}

TYPED_TEST(GatherTest, AllNull)
{
  constexpr cudf::size_type source_size{1000};

  // Every element is invalid
  auto data     = cuda::counting_iterator{0};
  auto validity = cudf::test::iterators::all_nulls();

  // Create a gather map that gathers to random locations
  std::vector<cudf::size_type> host_map_data(source_size);
  std::iota(host_map_data.begin(), host_map_data.end(), 0);
  std::mt19937 g(0);
  std::shuffle(host_map_data.begin(), host_map_data.end(), g);

  cudf::test::fixed_width_column_wrapper<TypeParam> source_column{
    data, data + source_size, validity};
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(host_map_data.begin(),
                                                             host_map_data.end());

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  // Check that the result is also all invalid
  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

TYPED_TEST(GatherTest, MultiColReverseIdentityTest)
{
  constexpr cudf::size_type source_size{1000};

  constexpr cudf::size_type n_cols = 3;

  auto data = cuda::counting_iterator{0};
  auto reversed_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return source_size - 1 - i; });

  std::vector<cudf::test::fixed_width_column_wrapper<TypeParam>> source_column_wrappers;
  std::vector<cudf::column_view> source_columns;

  for (int i = 0; i < n_cols; ++i) {
    source_column_wrappers.push_back(
      cudf::test::fixed_width_column_wrapper<TypeParam>(data, data + source_size));
    source_columns.push_back(source_column_wrappers[i]);
  }

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(reversed_data,
                                                             reversed_data + source_size);

  cudf::table_view source_table{source_columns};

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(reversed_data,
                                                                  reversed_data + source_size);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_column, result->view().column(i));
  }
}

TYPED_TEST(GatherTest, MultiColNulls)
{
  constexpr cudf::size_type source_size{1000};

  static_assert(0 == source_size % 2, "Size of source data must be a multiple of 2.");

  constexpr cudf::size_type n_cols = 3;

  auto data     = cuda::counting_iterator{0};
  auto validity = cudf::test::iterators::nulls_at_multiples_of(2);

  std::vector<cudf::test::fixed_width_column_wrapper<TypeParam>> source_column_wrappers;
  std::vector<cudf::column_view> source_columns;

  for (int i = 0; i < n_cols; ++i) {
    source_column_wrappers.push_back(
      cudf::test::fixed_width_column_wrapper<TypeParam>(data, data + source_size, validity));
    source_columns.push_back(source_column_wrappers[i]);
  }

  auto reversed_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return source_size - 1 - i; });

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(reversed_data,
                                                             reversed_data + source_size);

  cudf::table_view source_table{source_columns};

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  // Expected data
  auto expect_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return source_size - i - 1; });
  auto expect_valid = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(
    expect_data, expect_data + source_size, expect_valid);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_column, result->view().column(i));
  }
}

class GatherNullableTest : public cudf::test::BaseFixture {};

TEST_F(GatherNullableTest, NullableNoNulls)
{
  constexpr cudf::size_type source_size{1000};
  auto source_zero                            = cudf::make_fixed_width_scalar<int32_t>(0);
  std::unique_ptr<cudf::column> source_column = cudf::sequence(source_size, *source_zero);

  auto valid_mask = cudf::create_null_mask(source_size, cudf::mask_state::ALL_VALID);
  source_column->set_null_mask(std::move(valid_mask), 0);
  cudf::table_view source_table({source_column->view(), source_column->view()});

  auto gather_zero = cudf::make_fixed_width_scalar<int32_t>(0);

  std::unique_ptr<cudf::column> gather_map = cudf::sequence(source_size, *gather_zero);
  std::unique_ptr<cudf::table> result =
    cudf::gather(source_table, gather_map->view(), cudf::out_of_bounds_policy::DONT_CHECK);

  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}
