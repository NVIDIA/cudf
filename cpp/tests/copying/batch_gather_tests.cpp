/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/batch_gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ============================================================================
// Helpers (global namespace per the cudf test conventions).
// ============================================================================

std::unique_ptr<cudf::table> run_batch_gather(
  cudf::table_view const& source,
  cudf::column_view const& map,
  cudf::out_of_bounds_policy bounds = cudf::out_of_bounds_policy::DONT_CHECK,
  cudf::negative_index_policy neg   = cudf::negative_index_policy::NOT_ALLOWED)
{
  return cudf::detail::batch_gather(
    source,
    map,
    bounds,
    neg,
    cudf::get_default_stream(),
    cudf::memory_resources{cudf::get_current_device_resource_ref()});
}

std::unique_ptr<cudf::table> run_reference(
  cudf::table_view const& source,
  cudf::column_view const& map,
  cudf::out_of_bounds_policy bounds = cudf::out_of_bounds_policy::DONT_CHECK,
  cudf::negative_index_policy neg   = cudf::negative_index_policy::NOT_ALLOWED)
{
  return cudf::gather(source,
                      map,
                      bounds,
                      neg,
                      cudf::get_default_stream(),
                      cudf::memory_resources{cudf::get_current_device_resource_ref()});
}

// Builds a column of any FixedWidthTypes member (decimals ride their representation type at
// scale 0) with element i = value_of(i) and validity valid_of(i).
template <typename T, typename ValueFn, typename ValidFn>
std::unique_ptr<cudf::column> make_fixed_column(cudf::size_type size,
                                                ValueFn value_of,
                                                ValidFn valid_of)
{
  auto validity = cudf::detail::make_counting_transform_iterator(
    0, [valid_of](cudf::size_type i) { return static_cast<bool>(valid_of(i)); });
  if constexpr (cudf::is_fixed_point<T>()) {
    using Rep = typename T::rep;
    std::vector<Rep> reps;
    reps.reserve(size);
    for (cudf::size_type i = 0; i < size; ++i) {
      reps.push_back(static_cast<Rep>(value_of(i)));
    }
    return cudf::test::fixed_point_column_wrapper<Rep>(
             reps.begin(), reps.end(), validity, numeric::scale_type{0})
      .release();
  } else {
    std::vector<T> values;
    values.reserve(size);
    for (cudf::size_type i = 0; i < size; ++i) {
      if constexpr (cudf::is_timestamp<T>()) {
        values.push_back(T{typename T::duration{static_cast<typename T::rep>(value_of(i))}});
      } else if constexpr (cudf::is_duration<T>()) {
        values.push_back(T{static_cast<typename T::rep>(value_of(i))});
      } else {
        values.push_back(static_cast<T>(value_of(i)));
      }
    }
    return cudf::test::fixed_width_column_wrapper<T>(values.begin(), values.end(), validity)
      .release();
  }
}

// Non-nullable variant of `make_fixed_column`: no validity iterator reaches the wrapper, so no
// mask is allocated (the validity-iterator ctor always allocates one, even for all-true input).
template <typename T, typename ValueFn>
std::unique_ptr<cudf::column> make_fixed_column_without_validity(cudf::size_type size,
                                                                 ValueFn value_of)
{
  if constexpr (cudf::is_fixed_point<T>()) {
    using Rep = typename T::rep;
    std::vector<Rep> reps;
    reps.reserve(size);
    for (cudf::size_type i = 0; i < size; ++i) {
      reps.push_back(static_cast<Rep>(value_of(i)));
    }
    return cudf::test::fixed_point_column_wrapper<Rep>(
             reps.begin(), reps.end(), numeric::scale_type{0})
      .release();
  } else {
    std::vector<T> values;
    values.reserve(size);
    for (cudf::size_type i = 0; i < size; ++i) {
      values.push_back(static_cast<T>(value_of(i)));
    }
    return cudf::test::fixed_width_column_wrapper<T>(values.begin(), values.end()).release();
  }
}

std::vector<cudf::size_type> shuffled_iota(cudf::size_type size, unsigned seed = 0)
{
  std::vector<cudf::size_type> result(size);
  std::iota(result.begin(), result.end(), 0);
  std::mt19937 rng(seed);
  std::shuffle(result.begin(), result.end(), rng);
  return result;
}

// ============================================================================
// §5.1 Typed happy path over all fixed-width types (incl. fixed-point).
// ============================================================================

template <typename T>
class BatchGatherTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(BatchGatherTest, cudf::test::FixedWidthTypes);

TYPED_TEST(BatchGatherTest, Identity)
{
  constexpr cudf::size_type source_size{1000};
  auto source_column = make_fixed_column<TypeParam>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  auto gather_map = make_fixed_column<cudf::size_type>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({source_column->view()});
  auto result = run_batch_gather(source_table, gather_map->view());

  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

TYPED_TEST(BatchGatherTest, ReverseIdentity)
{
  constexpr cudf::size_type source_size{1000};
  auto source_column = make_fixed_column<TypeParam>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  auto gather_map = make_fixed_column<cudf::size_type>(
    source_size,
    [](cudf::size_type i) { return source_size - 1 - i; },
    [](cudf::size_type) { return true; });
  auto expected = make_fixed_column<TypeParam>(
    source_size,
    [](cudf::size_type i) { return source_size - 1 - i; },
    [](cudf::size_type) { return true; });

  cudf::table_view source_table({source_column->view()});
  auto result = run_batch_gather(source_table, gather_map->view());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view().column(0));
}

TYPED_TEST(BatchGatherTest, ShuffledMap)
{
  constexpr cudf::size_type source_size{1000};
  auto const host_map = shuffled_iota(source_size);

  auto source_column = make_fixed_column<TypeParam>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());
  cudf::table_view source_table({source_column->view()});

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

TYPED_TEST(BatchGatherTest, NullablePartialNulls)
{
  constexpr cudf::size_type source_size{1000};  // nulls at even rows, valid at odd rows
  auto source_column = make_fixed_column<TypeParam>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type i) { return i % 2 != 0; });
  auto gather_map = make_fixed_column<cudf::size_type>(
    source_size / 2,
    [](cudf::size_type i) { return i * 2 + 1; },
    [](cudf::size_type) { return true; });
  auto expected = make_fixed_column<TypeParam>(
    source_size / 2,
    [](cudf::size_type i) { return i * 2 + 1; },
    [](cudf::size_type) { return true; });

  cudf::table_view source_table({source_column->view()});
  auto result = run_batch_gather(source_table, gather_map->view());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view().column(0));
}

TYPED_TEST(BatchGatherTest, AllNullSource)
{
  constexpr cudf::size_type source_size{1000};
  auto const host_map = shuffled_iota(source_size);

  auto source_column = make_fixed_column<TypeParam>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return false; });
  auto expected = make_fixed_column<TypeParam>(
    source_size, [](cudf::size_type) { return 0; }, [](cudf::size_type) { return false; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  cudf::table_view source_table({source_column->view()});
  auto result = run_batch_gather(source_table, gather_map);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view().column(0));
}

TYPED_TEST(BatchGatherTest, NullableNoNullsSource)
{
  constexpr cudf::size_type source_size{1000};
  // Validity iterator over all-true values: nullable column with null_count() == 0.
  auto source_column = make_fixed_column<TypeParam>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  ASSERT_EQ(source_column->null_count(), 0);
  auto gather_map = make_fixed_column<cudf::size_type>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({source_column->view()});
  auto result = run_batch_gather(source_table, gather_map->view());

  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

TYPED_TEST(BatchGatherTest, NullifyOutOfBounds)
{
  constexpr cudf::size_type source_size{1000};
  auto source_column = make_fixed_column<TypeParam>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{
    {0, source_size, source_size + 5, 2}};
  auto expected = make_fixed_column<TypeParam>(
    4,
    [](cudf::size_type i) { return i == 3 ? 2 : 0; },
    [](cudf::size_type i) { return i == 0 or i == 3; });

  cudf::table_view source_table({source_column->view()});
  auto result = run_batch_gather(source_table, gather_map, cudf::out_of_bounds_policy::NULLIFY);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view().column(0));
}

TYPED_TEST(BatchGatherTest, NullifyNegativeWrap)
{
  constexpr cudf::size_type source_size{1000};
  auto source_column = make_fixed_column<TypeParam>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{
    {-1, -source_size, 0, source_size - 1, -source_size - 1}};
  auto expected = make_fixed_column<TypeParam>(
    5,
    [](cudf::size_type i) { return (i == 0 or i == 3) ? source_size - 1 : 0; },
    [](cudf::size_type i) { return i != 4; });

  cudf::table_view source_table({source_column->view()});
  auto result = run_batch_gather(source_table,
                                 gather_map,
                                 cudf::out_of_bounds_policy::NULLIFY,
                                 cudf::negative_index_policy::ALLOWED);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view().column(0));
}

// ============================================================================
// §5.2 Sliced sources (non-zero offset): the data<T>()/mask-offset regression class.
// ============================================================================

class BatchGatherSlicedTest : public cudf::test::BaseFixture {};

TEST_F(BatchGatherSlicedTest, SlicedDataAndMask)
{
  constexpr cudf::size_type source_size{1000};
  auto source_column = make_fixed_column<int32_t>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type i) { return i % 2 != 0; });

  auto sliced = cudf::slice(source_column->view(), {100, 600});
  ASSERT_EQ(sliced.size(), 1);
  auto gather_map = make_fixed_column<cudf::size_type>(
    500, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({sliced[0]});
  auto result = run_batch_gather(source_table, gather_map->view());

  auto expected = make_fixed_column<int32_t>(
    500,
    [](cudf::size_type i) { return 100 + i; },
    [](cudf::size_type i) { return (100 + i) % 2 != 0; });
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view().column(0));
}

TEST_F(BatchGatherSlicedTest, SlicedMixedWidths)
{
  constexpr cudf::size_type source_size{100};
  constexpr cudf::size_type slice_begin{10};
  constexpr cudf::size_type slice_end{60};

  auto col_i8 = make_fixed_column<int8_t>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  auto col_i64 = make_fixed_column<int64_t>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  auto col_dec = make_fixed_column<numeric::decimal128>(
    source_size, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  auto sliced_i8  = cudf::slice(col_i8->view(), {slice_begin, slice_end});
  auto sliced_i64 = cudf::slice(col_i64->view(), {slice_begin, slice_end});
  auto sliced_dec = cudf::slice(col_dec->view(), {slice_begin, slice_end});
  cudf::table_view source_table({sliced_i8[0], sliced_i64[0], sliced_dec[0]});

  auto gather_map = make_fixed_column<cudf::size_type>(
    slice_end - slice_begin,
    [](cudf::size_type i) { return i; },
    [](cudf::size_type) { return true; });
  auto result = run_batch_gather(source_table, gather_map->view());

  auto expected_i8 = make_fixed_column<int8_t>(
    slice_end - slice_begin,
    [](cudf::size_type i) { return slice_begin + i; },
    [](cudf::size_type) { return true; });
  auto expected_i64 = make_fixed_column<int64_t>(
    slice_end - slice_begin,
    [](cudf::size_type i) { return slice_begin + i; },
    [](cudf::size_type) { return true; });
  auto expected_dec = make_fixed_column<numeric::decimal128>(
    slice_end - slice_begin,
    [](cudf::size_type i) { return slice_begin + i; },
    [](cudf::size_type) { return true; });

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_i8->view(), result->view().column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_i64->view(), result->view().column(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_dec->view(), result->view().column(2));
}

// ============================================================================
// §5.3 Degenerate shapes.
// ============================================================================

class BatchGatherDegenerateTest : public cudf::test::BaseFixture {};

TEST_F(BatchGatherDegenerateTest, ZeroRowMap)
{
  auto source_column = make_fixed_column<int32_t>(
    10, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{};

  cudf::table_view source_table({source_column->view()});
  auto result = run_batch_gather(source_table, gather_map);

  EXPECT_EQ(result->num_rows(), 0);
  EXPECT_EQ(result->num_columns(), 1);
  EXPECT_EQ(result->view().column(0).type(), source_column->view().type());
}

TEST_F(BatchGatherDegenerateTest, ZeroColumnTable)
{
  cudf::table_view source{std::vector<cudf::column_view>{}, 5};
  auto gather_map = make_fixed_column<cudf::size_type>(
    4, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  auto result = run_batch_gather(source, gather_map->view());

  EXPECT_EQ(result->num_columns(), 0);
  EXPECT_EQ(result->num_rows(), 4);
}

TEST_F(BatchGatherDegenerateTest, OneRowTable)
{
  auto source_column = make_fixed_column<int32_t>(
    1, [](cudf::size_type) { return 42; }, [](cudf::size_type) { return true; });
  cudf::table_view source_table({source_column->view()});

  cudf::test::fixed_width_column_wrapper<cudf::size_type> identity_map{{0}};
  auto result = run_batch_gather(source_table, identity_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());

  cudf::test::fixed_width_column_wrapper<cudf::size_type> wrap_map{{-1}};
  auto wrapped = run_batch_gather(source_table,
                                  wrap_map,
                                  cudf::out_of_bounds_policy::NULLIFY,
                                  cudf::negative_index_policy::ALLOWED);
  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, wrapped->view());
}

// ============================================================================
// §5.4 Width/shape scaling.
// ============================================================================

class BatchGatherShapeTest : public cudf::test::BaseFixture {};

TEST_F(BatchGatherShapeTest, WideTable512Columns)
{
  constexpr cudf::size_type num_rows{1000};
  constexpr cudf::size_type num_cols{512};

  std::vector<std::unique_ptr<cudf::column>> source_columns;
  source_columns.reserve(num_cols);
  for (cudf::size_type c = 0; c < num_cols; ++c) {
    source_columns.push_back(make_fixed_column<int32_t>(
      num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; }));
  }
  std::vector<cudf::column_view> views;
  views.reserve(num_cols);
  for (auto const& col : source_columns) {
    views.push_back(col->view());
  }
  cudf::table_view source_table(views);

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

TEST_F(BatchGatherShapeTest, YCapColumnLoop)
{
  // 66'001 columns span 8'251 tiles per block.y and force a PARTIAL final tile (1 column);
  // the y-stride wrap past the 65535 gridDim.y cap only engages above 524'280 columns
  // (65'535 tiles x 8), which is not host-feasible to build here — the wrap shares the
  // x-stride code shape and its cap math is exercised via the grid-y computation.
  constexpr cudf::size_type num_rows{100};
  constexpr cudf::size_type num_cols{66'001};

  std::vector<std::unique_ptr<cudf::column>> source_columns;
  source_columns.reserve(num_cols);
  for (cudf::size_type c = 0; c < num_cols; ++c) {
    source_columns.push_back(make_fixed_column<int32_t>(
      num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; }));
  }
  std::vector<cudf::column_view> views;
  views.reserve(num_cols);
  for (auto const& col : source_columns) {
    views.push_back(col->view());
  }
  cudf::table_view source_table(views);

  auto gather_map = make_fixed_column<cudf::size_type>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  auto result = run_batch_gather(source_table, gather_map->view());

  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

TEST_F(BatchGatherShapeTest, TallTable)
{
  constexpr cudf::size_type num_rows{1'048'576};
  constexpr cudf::size_type num_cols{8};

  std::vector<std::unique_ptr<cudf::column>> source_columns;
  source_columns.reserve(num_cols);
  for (cudf::size_type c = 0; c < num_cols; ++c) {
    source_columns.push_back(make_fixed_column<int32_t>(
      num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; }));
  }
  std::vector<cudf::column_view> views;
  views.reserve(num_cols);
  for (auto const& col : source_columns) {
    views.push_back(col->view());
  }
  cudf::table_view source_table(views);

  auto identity_map = make_fixed_column<cudf::size_type>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  auto identity_result = run_batch_gather(source_table, identity_map->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, identity_result->view());

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> shuffled_map(host_map.begin(),
                                                                       host_map.end());
  auto shuffled_result = run_batch_gather(source_table, shuffled_map);
  auto reference       = run_reference(source_table, shuffled_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), shuffled_result->view());
}

// ============================================================================
// §5.5 Bounds and policies.
// ============================================================================

class BatchGatherPolicyTest : public cudf::test::BaseFixture {};

TEST_F(BatchGatherPolicyTest, DontCheckPolicyAllValid)
{
  constexpr cudf::size_type num_rows{100};
  // Non-nullable source: DONT_CHECK must not fabricate a mask (§3.4 row 1).
  auto source_column =
    make_fixed_column_without_validity<int32_t>(num_rows, [](cudf::size_type i) { return i; });
  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  cudf::table_view source_table({source_column->view()});
  auto result = run_batch_gather(source_table, gather_map);

  ASSERT_FALSE(source_column->nullable());
  EXPECT_FALSE(result->view().column(0).nullable());
  auto expected = make_fixed_column_without_validity<int32_t>(
    num_rows, [&host_map](cudf::size_type i) { return host_map[i]; });
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view().column(0));
}

TEST_F(BatchGatherPolicyTest, DontCheckPolicyNullableSource)
{
  constexpr cudf::size_type num_rows{100};
  auto source_column = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type i) { return i % 2 != 0; });
  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  cudf::table_view source_table({source_column->view()});
  auto result = run_batch_gather(source_table, gather_map);

  EXPECT_TRUE(result->view().column(0).nullable());
  auto expected = make_fixed_column<int32_t>(
    num_rows,
    [&host_map](cudf::size_type i) { return host_map[i]; },
    [&host_map](cudf::size_type i) { return host_map[i] % 2 != 0; });
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view().column(0));

  auto reference = run_reference(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

TEST_F(BatchGatherPolicyTest, NullifyMakesNullable)
{
  constexpr cudf::size_type num_rows{100};
  auto source_column =
    make_fixed_column_without_validity<int32_t>(num_rows, [](cudf::size_type i) { return i; });
  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  cudf::table_view source_table({source_column->view()});
  auto result = run_batch_gather(source_table, gather_map, cudf::out_of_bounds_policy::NULLIFY);

  EXPECT_TRUE(result->view().column(0).nullable());
  EXPECT_EQ(result->view().column(0).null_count(), 0);
  auto expected = make_fixed_column<int32_t>(
    num_rows,
    [&host_map](cudf::size_type i) { return host_map[i]; },
    [](cudf::size_type) { return true; });
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view().column(0));
}

// ============================================================================
// §5.6 Mixed tables: fused + fallback must agree with the stock gather.
// ============================================================================

class BatchGatherMixedTest : public cudf::test::BaseFixture {};

TEST_F(BatchGatherMixedTest, MixedFixedWidthAndStrings)
{
  constexpr cudf::size_type num_rows{100};
  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  std::vector<std::string> strings;
  strings.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    strings.push_back("s" + std::to_string(i));
  }
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end());
  auto col_i64 = make_fixed_column<int64_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  cudf::table_view source_table({col_i32->view(), col_str, col_i64->view()});

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

TEST_F(BatchGatherMixedTest, MixedWithListsAndStructs)
{
  constexpr cudf::size_type num_rows{10};
  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  cudf::test::lists_column_wrapper<int32_t> col_list{
    {1, 2}, {}, {3}, {4, 5}, {}, {6}, {7, 8}, {}, {9}, {10}};
  auto child_i64 = cudf::test::fixed_width_column_wrapper<int64_t>{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
  auto child_str =
    cudf::test::strings_column_wrapper{{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}};
  cudf::test::structs_column_wrapper col_struct{
    {child_i64, child_str}, {true, true, false, true, true, true, false, true, true, true}};

  cudf::table_view source_table({col_i32->view(), col_list, col_struct});

  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{
    {9, 0, 3, 7, 1, 5, 2, 8, 4, 6}};

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

TEST_F(BatchGatherMixedTest, MixedNullifyOOB)
{
  constexpr cudf::size_type num_rows{10};
  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  cudf::test::lists_column_wrapper<int32_t> col_list{
    {1, 2}, {}, {3}, {4, 5}, {}, {6}, {7, 8}, {}, {9}, {10}};
  auto child_i64 = cudf::test::fixed_width_column_wrapper<int64_t>{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
  auto child_str =
    cudf::test::strings_column_wrapper{{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}};
  cudf::test::structs_column_wrapper col_struct{
    {child_i64, child_str}, {true, true, false, true, true, true, false, true, true, true}};

  cudf::table_view source_table({col_i32->view(), col_list, col_struct});

  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{
    {0, num_rows, 3, num_rows + 5, 1, 5, 2, 8, 4, 6}};

  auto result    = run_batch_gather(source_table, gather_map, cudf::out_of_bounds_policy::NULLIFY);
  auto reference = run_reference(source_table, gather_map, cudf::out_of_bounds_policy::NULLIFY);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

TEST_F(BatchGatherMixedTest, MixedNullableStrings)
{
  constexpr cudf::size_type num_rows{100};
  auto col_i32 = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type i) { return i % 3 != 0; });
  std::vector<std::string> strings;
  std::vector<bool> valid;
  strings.reserve(num_rows);
  valid.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    strings.push_back(i % 5 == 0 ? "" : "v" + std::to_string(i));  // empty string != null string
    valid.push_back(i % 4 != 0);
  }
  auto validity = cudf::detail::make_counting_transform_iterator(
    0, [&valid](cudf::size_type i) { return valid[i]; });
  cudf::test::strings_column_wrapper col_str(strings.begin(), strings.end(), validity);

  cudf::table_view source_table({col_i32->view(), col_str});

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());

  auto result    = run_batch_gather(source_table, gather_map);
  auto reference = run_reference(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
}

// ============================================================================
// §5.7 Randomized differential against the stock gather (reference oracle).
// ============================================================================

class BatchGatherRandomizedTest : public cudf::test::BaseFixture {};

std::vector<cudf::type_id> const& random_candidate_types()
{
  static std::vector<cudf::type_id> const candidates{cudf::type_id::INT8,
                                                     cudf::type_id::INT16,
                                                     cudf::type_id::INT32,
                                                     cudf::type_id::INT64,
                                                     cudf::type_id::UINT32,
                                                     cudf::type_id::FLOAT32,
                                                     cudf::type_id::FLOAT64,
                                                     cudf::type_id::BOOL8,
                                                     cudf::type_id::DECIMAL32,
                                                     cudf::type_id::DECIMAL64,
                                                     cudf::type_id::DECIMAL128,
                                                     cudf::type_id::TIMESTAMP_NANOSECONDS,
                                                     cudf::type_id::DURATION_MICROSECONDS,
                                                     cudf::type_id::STRING};
  return candidates;
}

template <typename T>
std::unique_ptr<cudf::column> make_random_fixed(cudf::size_type num_rows,
                                                double null_probability,
                                                std::mt19937& rng)
{
  auto value_of = [&rng](cudf::size_type) {
    return std::uniform_int_distribution<int64_t>(-1000, 1000)(rng);
  };
  auto valid_of = [&rng, null_probability](cudf::size_type) {
    return std::uniform_real_distribution<double>(0.0, 1.0)(rng) >= null_probability;
  };
  return make_fixed_column<T>(num_rows, value_of, valid_of);
}

std::unique_ptr<cudf::column> make_random_column(cudf::type_id id,
                                                 cudf::size_type num_rows,
                                                 double null_probability,
                                                 std::mt19937& rng)
{
  switch (id) {
    case cudf::type_id::INT8: return make_random_fixed<int8_t>(num_rows, null_probability, rng);
    case cudf::type_id::INT16: return make_random_fixed<int16_t>(num_rows, null_probability, rng);
    case cudf::type_id::INT32: return make_random_fixed<int32_t>(num_rows, null_probability, rng);
    case cudf::type_id::INT64: return make_random_fixed<int64_t>(num_rows, null_probability, rng);
    case cudf::type_id::UINT32: return make_random_fixed<uint32_t>(num_rows, null_probability, rng);
    case cudf::type_id::FLOAT32: return make_random_fixed<float>(num_rows, null_probability, rng);
    case cudf::type_id::FLOAT64: return make_random_fixed<double>(num_rows, null_probability, rng);
    case cudf::type_id::BOOL8: return make_random_fixed<bool>(num_rows, null_probability, rng);
    case cudf::type_id::DECIMAL32:
      return make_random_fixed<numeric::decimal32>(num_rows, null_probability, rng);
    case cudf::type_id::DECIMAL64:
      return make_random_fixed<numeric::decimal64>(num_rows, null_probability, rng);
    case cudf::type_id::DECIMAL128:
      return make_random_fixed<numeric::decimal128>(num_rows, null_probability, rng);
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return make_random_fixed<cudf::timestamp_ns>(num_rows, null_probability, rng);
    case cudf::type_id::DURATION_MICROSECONDS:
      return make_random_fixed<cudf::duration_us>(num_rows, null_probability, rng);
    case cudf::type_id::STRING: {
      std::vector<std::string> strings;
      std::vector<bool> valid;
      strings.reserve(num_rows);
      valid.reserve(num_rows);
      for (cudf::size_type i = 0; i < num_rows; ++i) {
        strings.push_back(std::to_string(std::uniform_int_distribution<int>(-1000, 1000)(rng)));
        valid.push_back(std::uniform_real_distribution<double>(0.0, 1.0)(rng) >= null_probability);
      }
      auto validity = cudf::detail::make_counting_transform_iterator(
        0, [&valid](cudf::size_type i) { return valid[i]; });
      return cudf::test::strings_column_wrapper(strings.begin(), strings.end(), validity).release();
    }
    default: CUDF_FAIL("Unsupported type id in randomized test");
  }
}

TEST_F(BatchGatherRandomizedTest, RandomizedDifferential)
{
  for (int seed = 0; seed < 10; ++seed) {
    SCOPED_TRACE("iteration seed=" + std::to_string(seed));
    std::mt19937 rng(static_cast<unsigned>(seed));

    auto const num_rows = std::uniform_int_distribution<cudf::size_type>(1, 5000)(rng);
    auto const num_cols = std::uniform_int_distribution<cudf::size_type>(1, 64)(rng);
    auto const null_density =
      std::vector<double>{0.0, 0.1, 0.5, 1.0}[std::uniform_int_distribution<int>(0, 3)(rng)];

    std::vector<std::unique_ptr<cudf::column>> source_columns;
    source_columns.reserve(num_cols);
    for (cudf::size_type c = 0; c < num_cols; ++c) {
      auto const type = random_candidate_types()[std::uniform_int_distribution<std::size_t>(
        0, random_candidate_types().size() - 1)(rng)];
      source_columns.push_back(make_random_column(type, num_rows, null_density, rng));
    }
    std::vector<cudf::column_view> views;
    views.reserve(num_cols);
    for (auto const& col : source_columns) {
      views.push_back(col->view());
    }
    cudf::table_view source_table(views);

    auto const map_mode = std::uniform_int_distribution<int>(0, 2)(rng);
    std::vector<cudf::size_type> host_map(num_rows);
    switch (map_mode) {
      case 0: std::iota(host_map.begin(), host_map.end(), 0); break;
      case 1: {
        std::iota(host_map.begin(), host_map.end(), 0);
        std::shuffle(host_map.begin(), host_map.end(), rng);
        break;
      }
      default: {
        std::uniform_int_distribution<cudf::size_type> uniform_idx(0, num_rows - 1);
        std::generate(host_map.begin(), host_map.end(), [&]() { return uniform_idx(rng); });
        break;
      }
    }
    cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                       host_map.end());

    auto const bounds = std::uniform_int_distribution<int>(0, 1)(rng) == 0
                          ? cudf::out_of_bounds_policy::DONT_CHECK
                          : cudf::out_of_bounds_policy::NULLIFY;
    auto const neg    = std::uniform_int_distribution<int>(0, 1)(rng) == 0
                          ? cudf::negative_index_policy::NOT_ALLOWED
                          : cudf::negative_index_policy::ALLOWED;

    auto result    = run_batch_gather(source_table, gather_map, bounds, neg);
    auto reference = run_reference(source_table, gather_map, bounds, neg);

    CUDF_TEST_EXPECT_TABLES_EQUAL(reference->view(), result->view());
  }
}

// ============================================================================
// §5.8 Map types: input_indexalator normalization.
// ============================================================================

class BatchGatherMapTypesTest : public cudf::test::BaseFixture {};

TEST_F(BatchGatherMapTypesTest, MapTypesInt8ToInt64)
{
  constexpr cudf::size_type num_rows{50};
  auto source_column = make_fixed_column<int32_t>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  cudf::table_view source_table({source_column->view()});

  // Same values in every index type; batch_gather must normalize them identically.
  auto iota_map = [](auto type_tag) {
    using MapType = decltype(type_tag);
    std::vector<MapType> values(num_rows);
    std::iota(values.begin(), values.end(), 0);
    return cudf::test::fixed_width_column_wrapper<MapType>(values.begin(), values.end()).release();
  };

  auto first  = run_batch_gather(source_table, iota_map(int8_t{})->view());
  auto second = run_batch_gather(source_table, iota_map(int16_t{})->view());
  auto third  = run_batch_gather(source_table, iota_map(int32_t{})->view());
  auto fourth = run_batch_gather(source_table, iota_map(int64_t{})->view());
  auto fifth  = run_batch_gather(source_table, iota_map(uint32_t{})->view());

  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, first->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(first->view(), second->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(first->view(), third->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(first->view(), fourth->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(first->view(), fifth->view());
}

// ============================================================================
// §5.9 Launch-count verification bodies (profiled externally with nsys).
// ============================================================================

class BatchGatherLaunchCount : public cudf::test::BaseFixture {};

namespace {
constexpr cudf::size_type launch_count_rows{10'000};
constexpr cudf::size_type launch_count_cols{512};

std::vector<cudf::column_view> wide_fixed_width_views(
  std::vector<std::unique_ptr<cudf::column>>& columns)
{
  for (cudf::size_type c = 0; c < launch_count_cols; ++c) {
    columns.push_back(make_fixed_column<int32_t>(
      launch_count_rows,
      [](cudf::size_type i) { return i; },
      [](cudf::size_type) { return true; }));
  }
  auto views = std::vector<cudf::column_view>{};
  views.reserve(columns.size());
  for (auto const& col : columns) {
    views.push_back(col->view());
  }
  return views;
}
}  // namespace

TEST_F(BatchGatherLaunchCount, WideFixedWidth)
{
  // Profiled recipe: nsys must show exactly one batch_gather_kernel launch and no thrust kernels.
  std::vector<std::unique_ptr<cudf::column>> columns;
  auto views        = wide_fixed_width_views(columns);
  auto source_table = cudf::table_view(views);
  auto gather_map   = make_fixed_column<cudf::size_type>(
    launch_count_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  auto result = run_batch_gather(source_table, gather_map->view());

  ASSERT_EQ(result->num_rows(), launch_count_rows);
  ASSERT_EQ(result->num_columns(), launch_count_cols);
  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

TEST_F(BatchGatherLaunchCount, GatherLaunchCount)
{
  // Stock-gather twin: nsys must show num_cols + 1 kernel launches on the same input.
  std::vector<std::unique_ptr<cudf::column>> columns;
  auto views        = wide_fixed_width_views(columns);
  auto source_table = cudf::table_view(views);
  auto gather_map   = make_fixed_column<cudf::size_type>(
    launch_count_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });

  auto result = run_reference(source_table, gather_map->view());

  ASSERT_EQ(result->num_rows(), launch_count_rows);
  ASSERT_EQ(result->num_columns(), launch_count_cols);
  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

// ============================================================================
// §5.10 Map-locality routing: classifier decisions + routed-path behavior parity.
// ============================================================================

class BatchGatherRoutingTest : public cudf::test::BaseFixture {};

std::unique_ptr<cudf::table> run_batch_gather_tile(cudf::table_view const& source,
                                                   cudf::column_view const& map,
                                                   cudf::size_type tile_cols)
{
  return cudf::detail::batch_gather(source,
                                    map,
                                    cudf::out_of_bounds_policy::DONT_CHECK,
                                    cudf::negative_index_policy::NOT_ALLOWED,
                                    cudf::get_default_stream(),
                                    cudf::memory_resources{cudf::get_current_device_resource_ref()},
                                    tile_cols);
}

TEST_F(BatchGatherRoutingTest, ClassifierRoutesAscendingMapToWideTile)
{
  constexpr cudf::size_type num_rows{65536};
  auto gather_map = make_fixed_column<cudf::size_type>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  // >= kMinSampledCols fused columns: the sampler runs and routes ascending maps wide.
  EXPECT_EQ(
    cudf::detail::select_gather_map_tile_cols(gather_map->view(), 16, cudf::get_default_stream()),
    8);
  // Exactly kMinSampledCols - 1 fused columns: the sync-dominated skip regime (measured
  // fatal on 30-90us kernels) — safe K=1 without sampling.
  EXPECT_EQ(
    cudf::detail::select_gather_map_tile_cols(gather_map->view(), 8, cudf::get_default_stream()),
    1);
}

TEST_F(BatchGatherRoutingTest, ClassifierRoutesShuffledMapToSingleColumn)
{
  constexpr cudf::size_type num_rows{65536};
  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map.begin(),
                                                                     host_map.end());
  EXPECT_EQ(cudf::detail::select_gather_map_tile_cols(gather_map, 8, cudf::get_default_stream()),
            1);
}

TEST_F(BatchGatherRoutingTest, ClassifierSkipsInsensitiveRegimes)
{
  // Small table: the sampler is skipped and the safe K=1 returned regardless of map shape.
  constexpr cudf::size_type small_rows{100};
  auto small_ascending = make_fixed_column<cudf::size_type>(
    small_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  EXPECT_EQ(cudf::detail::select_gather_map_tile_cols(
              small_ascending->view(), 8, cudf::get_default_stream()),
            1);

  // Few fused columns: sampler skipped even with enough rows.
  constexpr cudf::size_type num_rows{65536};
  auto long_map = make_fixed_column<cudf::size_type>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  EXPECT_EQ(
    cudf::detail::select_gather_map_tile_cols(long_map->view(), 4, cudf::get_default_stream()), 1);
}

TEST_F(BatchGatherRoutingTest, RoutedPathMatchesExplicitTileWidths)
{
  // 8 fixed-width columns x 65536 rows: the routed path samples (ascending -> wide tile,
  // shuffled -> single column) and must agree with BOTH explicit widths, since results are
  // tile-width independent.
  constexpr cudf::size_type num_rows{65536};
  constexpr cudf::size_type num_cols{8};
  std::vector<std::unique_ptr<cudf::column>> source_columns;
  source_columns.reserve(num_cols);
  for (cudf::size_type c = 0; c < num_cols; ++c) {
    source_columns.push_back(make_fixed_column<int32_t>(
      num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; }));
  }
  std::vector<cudf::column_view> views;
  views.reserve(num_cols);
  for (auto const& col : source_columns) {
    views.push_back(col->view());
  }
  cudf::table_view source_table(views);

  auto ascending_map = make_fixed_column<cudf::size_type>(
    num_rows, [](cudf::size_type i) { return i; }, [](cudf::size_type) { return true; });
  auto routed_ascending = run_batch_gather(source_table, ascending_map->view());
  auto explicit_wide    = run_batch_gather_tile(source_table, ascending_map->view(), 8);
  auto explicit_single  = run_batch_gather_tile(source_table, ascending_map->view(), 1);
  CUDF_TEST_EXPECT_TABLES_EQUAL(routed_ascending->view(), explicit_wide->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(routed_ascending->view(), explicit_single->view());

  auto const host_map = shuffled_iota(num_rows);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> shuffled_map(host_map.begin(),
                                                                       host_map.end());
  auto routed_shuffled    = run_batch_gather(source_table, shuffled_map);
  auto wide_on_shuffled   = run_batch_gather_tile(source_table, shuffled_map, 8);
  auto single_on_shuffled = run_batch_gather_tile(source_table, shuffled_map, 1);
  CUDF_TEST_EXPECT_TABLES_EQUAL(routed_shuffled->view(), wide_on_shuffled->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(routed_shuffled->view(), single_on_shuffled->view());
}
