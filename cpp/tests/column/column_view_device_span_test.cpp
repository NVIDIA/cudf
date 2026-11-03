/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/iterator>
#include <cuda/std/span>

#include <memory>

template <typename T, CUDF_ENABLE_IF(cudf::is_numeric<T>() or cudf::is_chrono<T>())>
std::unique_ptr<cudf::column> example_column()
{
  auto begin = cuda::counting_iterator<int32_t>{1};
  auto end   = cuda::counting_iterator<int32_t>{16};
  return cudf::test::fixed_width_column_wrapper<T>(begin, end).release();
}

template <typename T>
struct ColumnViewSpanTests : public cudf::test::BaseFixture {};

using DeviceSpanTypes = cudf::test::FixedWidthTypesWithoutFixedPoint;
TYPED_TEST_SUITE(ColumnViewSpanTests, DeviceSpanTypes);

TYPED_TEST(ColumnViewSpanTests, device_span_conversion_round_trip)
{
  auto col      = example_column<TypeParam>();
  auto col_view = cudf::column_view{*col};

  // Test implicit conversion, round trip
  cudf::device_span<TypeParam const> device_span_from_col_view = col_view;
  cudf::column_view col_view_from_device_span                  = device_span_from_col_view;
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col_view, col_view_from_device_span);
}

struct ColumnViewSpanErrorTests : public cudf::test::BaseFixture {};

TEST_F(ColumnViewSpanErrorTests, device_span_type_mismatch)
{
  auto col      = example_column<int32_t>();
  auto col_view = cudf::column_view{*col};
  EXPECT_THROW((void)cudf::device_span<float const>{col_view}, cudf::logic_error);
}

TEST_F(ColumnViewSpanErrorTests, device_span_nullable_column)
{
  auto col = example_column<int32_t>();
  col->set_null_mask(cudf::create_null_mask(col->size(), cudf::mask_state::ALL_NULL), col->size());
  auto col_view = cudf::column_view{*col};
  EXPECT_THROW((void)cudf::device_span<int32_t const>{col_view}, cudf::logic_error);
}

TYPED_TEST(ColumnViewSpanTests, std_span_conversion_to_span)
{
  auto col      = example_column<TypeParam>();
  auto col_view = cudf::column_view{*col};

  // Test implicit conversion to cuda::std::span
  cuda::std::span<TypeParam const> cuda_span_from_col_view = col_view;

  // Verify span properties match column view
  EXPECT_EQ(cuda_span_from_col_view.size(), static_cast<std::size_t>(col_view.size()));
  EXPECT_EQ(cuda_span_from_col_view.data(), col_view.data<TypeParam>());
  EXPECT_FALSE(cuda_span_from_col_view.empty());
}

TYPED_TEST(ColumnViewSpanTests, std_span_explicit_conversion_to_span)
{
  auto col      = example_column<TypeParam>();
  auto col_view = cudf::column_view{*col};

  // Test explicit conversion to cuda::std::span
  auto cuda_span_from_col_view = static_cast<cuda::std::span<TypeParam const>>(col_view);

  // Verify span properties match column view
  EXPECT_EQ(cuda_span_from_col_view.size(), static_cast<std::size_t>(col_view.size()));
  EXPECT_EQ(cuda_span_from_col_view.data(), col_view.data<TypeParam>());
}

TYPED_TEST(ColumnViewSpanTests, std_span_empty_column_to_span)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> empty_col{};
  auto col_view = cudf::column_view{empty_col};

  // Test conversion of empty column to cuda::std::span
  cuda::std::span<TypeParam const> cuda_span_from_col_view = col_view;

  // Verify span properties for empty column
  EXPECT_EQ(cuda_span_from_col_view.size(), 0u);
  EXPECT_TRUE(cuda_span_from_col_view.empty());
}

TEST_F(ColumnViewSpanErrorTests, std_span_type_mismatch)
{
  auto col      = example_column<int32_t>();
  auto col_view = cudf::column_view{*col};
  EXPECT_THROW((void)cuda::std::span<float const>{col_view}, cudf::logic_error);
}

TEST_F(ColumnViewSpanErrorTests, std_span_nullable_column)
{
  auto col = example_column<int32_t>();
  col->set_null_mask(cudf::create_null_mask(col->size(), cudf::mask_state::ALL_NULL), col->size());
  auto col_view = cudf::column_view{*col};
  EXPECT_THROW((void)cuda::std::span<int32_t const>{col_view}, cudf::logic_error);
}
