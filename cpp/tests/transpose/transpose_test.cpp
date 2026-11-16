/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/transpose.hpp>

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

template <typename T, typename F>
auto generate_vectors(size_t ncols, size_t nrows, F generator)
{
  std::vector<std::vector<T>> values(ncols);

  std::for_each(values.begin(), values.end(), [generator, nrows](std::vector<T>& col) {
    col.resize(nrows);
    std::generate(col.begin(), col.end(), generator);
  });

  return values;
}

template <typename T>
auto transpose_vectors(std::vector<std::vector<T>> const& input)
{
  if (input.empty()) { return input; }
  size_t ncols = input.size();
  size_t nrows = input.front().size();

  std::vector<std::vector<T>> transposed(nrows);
  std::for_each(
    transposed.begin(), transposed.end(), [=](std::vector<T>& col) { col.resize(ncols); });

  for (size_t col = 0; col < input.size(); ++col) {
    for (size_t row = 0; row < nrows; ++row) {
      transposed[row][col] = input[col][row];
    }
  }

  return transposed;
}

template <typename T>
auto flatten_vectors(std::vector<std::vector<T>> const& input)
{
  std::vector<T> flattened;
  if (not input.empty()) { flattened.reserve(input.size() * input.front().size()); }
  for (auto const& column : input) {
    flattened.insert(flattened.end(), column.begin(), column.end());
  }
  return flattened;
}

// Owner equality implies slice equality only when every view references the correct buffers.
void expect_slice_view(cudf::column_view const& actual,
                       cudf::column_view const& owner,
                       cudf::size_type offset,
                       cudf::size_type size,
                       cudf::size_type null_count)
{
  EXPECT_EQ(actual.type(), owner.type());
  EXPECT_EQ(actual.size(), size);
  EXPECT_EQ(actual.offset(), offset);
  EXPECT_EQ(actual.null_count(), null_count);
  EXPECT_EQ(actual.head(), owner.head());
  EXPECT_EQ(actual.null_mask(), owner.null_mask());
  ASSERT_EQ(actual.num_children(), owner.num_children());
  for (cudf::size_type i = 0; i < owner.num_children(); ++i) {
    SCOPED_TRACE("child " + std::to_string(i));
    auto const child = owner.child(i);
    expect_slice_view(actual.child(i), child, child.offset(), child.size(), child.null_count());
  }
}

template <typename T, typename ColumnWrapper>
auto make_columns(std::vector<std::vector<T>> const& values)
{
  std::vector<ColumnWrapper> columns;
  columns.reserve(values.size());

  for (auto const& value_col : values) {
    columns.emplace_back(value_col.begin(), value_col.end());
  }

  return columns;
}

template <typename T, typename ColumnWrapper>
auto make_columns(std::vector<std::vector<T>> const& values,
                  std::vector<std::vector<cudf::size_type>> const& valids)
{
  std::vector<ColumnWrapper> columns;
  columns.reserve(values.size());

  for (size_t col = 0; col < values.size(); ++col) {
    columns.emplace_back(values[col].begin(), values[col].end(), valids[col].begin());
  }

  return columns;
}

template <typename ColumnWrapper>
auto make_table_view(std::vector<ColumnWrapper> const& cols)
{
  std::vector<cudf::column_view> views(cols.size());

  std::transform(cols.begin(), cols.end(), views.begin(), [](auto const& col) {
    return static_cast<cudf::column_view>(col);
  });

  return cudf::table_view(views);
}

template <typename T>
void run_test(size_t ncols, size_t nrows, bool add_nulls)
{
  using ColumnWrapper = std::conditional_t<std::is_same_v<T, std::string>,
                                           cudf::test::strings_column_wrapper,
                                           cudf::test::fixed_width_column_wrapper<T>>;

  std::mt19937 rng(1);

  // Generate values as vector of vectors
  auto const values = generate_vectors<T>(
    ncols, nrows, [&rng]() { return cudf::test::make_type_param_scalar<T>(rng()); });
  auto const valuesT         = transpose_vectors(values);
  auto const expected_values = flatten_vectors(valuesT);

  std::vector<ColumnWrapper> input_cols;
  std::vector<cudf::size_type> expected_nulls(nrows);

  auto const expected = [&] {
    if (add_nulls) {
      // Generate null mask as vector of vectors
      auto const valids = generate_vectors<cudf::size_type>(
        ncols, nrows, [&rng]() { return static_cast<cudf::size_type>(rng() % 3 > 0 ? 1 : 0); });
      auto const validsT = transpose_vectors(valids);

      // Compute the null counts over each transposed column
      std::transform(validsT.begin(),
                     validsT.end(),
                     expected_nulls.begin(),
                     [ncols](std::vector<cudf::size_type> const& vec) {
                       // num nulls = num elems - num valids
                       return ncols - std::accumulate(vec.begin(), vec.end(), 0);
                     });

      auto const expected_valids = flatten_vectors(validsT);
      input_cols                 = make_columns<T, ColumnWrapper>(values, valids);
      return ColumnWrapper(expected_values.begin(), expected_values.end(), expected_valids.begin());
    }
    input_cols = make_columns<T, ColumnWrapper>(values);
    return ColumnWrapper(expected_values.begin(), expected_values.end());
  }();

  auto input_view = make_table_view(input_cols);

  auto result      = cudf::transpose(input_view);
  auto result_view = std::get<1>(result);

  ASSERT_EQ(result_view.num_columns(), valuesT.size());
  if (result_view.num_columns() == 0) {
    EXPECT_EQ(result.first->size(), 0);
    return;
  }

  // disable checking logic during a racecheck run
  if (not getenv("LIBCUDF_RACECHECK_ENABLED")) {
    // Comparing many tiny slices separately repeats GPU launches and synchronization.
    auto const owner = result.first->view();
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(owner, expected);
    for (cudf::size_type i = 0; i < result_view.num_columns(); ++i) {
      SCOPED_TRACE("column " + std::to_string(i));
      expect_slice_view(result_view.column(i),
                        owner,
                        i * static_cast<cudf::size_type>(ncols),
                        static_cast<cudf::size_type>(ncols),
                        expected_nulls[i]);
    }
  }
}

}  // namespace

template <typename T>
class TransposeTest : public cudf::test::BaseFixture {};

// Using std::string here instead of cudf::test::StringTypes allows us to
// use std::vector<T> utilities in this file just like the fixed-width types.
// Should consider changing cudf::test::StringTypes to std::string instead of cudf::string_view.
using StdStringType  = cudf::test::Types<std::string>;
using TransposeTypes = cudf::test::Concat<cudf::test::FixedWidthTypes, StdStringType>;

TYPED_TEST_SUITE(TransposeTest, TransposeTypes);

TYPED_TEST(TransposeTest, SingleValue) { run_test<TypeParam>(1, 1, false); }

TYPED_TEST(TransposeTest, SingleColumn) { run_test<TypeParam>(1, 1000, false); }

TYPED_TEST(TransposeTest, SingleColumnNulls) { run_test<TypeParam>(1, 1000, true); }

TYPED_TEST(TransposeTest, Square) { run_test<TypeParam>(100, 100, false); }

TYPED_TEST(TransposeTest, SquareNulls) { run_test<TypeParam>(100, 100, true); }

TYPED_TEST(TransposeTest, Slim) { run_test<TypeParam>(10, 1000, false); }

TYPED_TEST(TransposeTest, SlimNulls) { run_test<TypeParam>(10, 1000, true); }

TYPED_TEST(TransposeTest, Fat) { run_test<TypeParam>(1000, 10, false); }

TYPED_TEST(TransposeTest, FatNulls) { run_test<TypeParam>(1000, 10, true); }

TYPED_TEST(TransposeTest, EmptyTable) { run_test<TypeParam>(0, 0, false); }

TYPED_TEST(TransposeTest, EmptyColumns) { run_test<TypeParam>(10, 0, false); }

class TransposeTestError : public cudf::test::BaseFixture {};

TEST_F(TransposeTestError, MismatchedColumns)
{
  cudf::test::fixed_width_column_wrapper<uint32_t, int32_t> col1({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int8_t> col2{{4, 5, 6}};
  cudf::test::fixed_width_column_wrapper<float> col3{{7, 8, 9}};
  cudf::table_view input{{col1, col2, col3}};
  EXPECT_THROW(cudf::transpose(input), cudf::logic_error);
}

CUDF_TEST_PROGRAM_MAIN()
