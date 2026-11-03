/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/is_element_valid.hpp>
#include <cudf/detail/iterator.cuh>

#include <cuda/iterator>

struct IsElementValidTest : public cudf::test::BaseFixture {};

TEST_F(IsElementValidTest, IsElementValidBasic)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 1, 1, 1, 1},
                                                      {true, false, false, false, true});
  EXPECT_TRUE(cudf::detail::is_element_valid_sync(col, 0, cudf::get_default_stream()));
  EXPECT_FALSE(cudf::detail::is_element_valid_sync(col, 1, cudf::get_default_stream()));
  EXPECT_FALSE(cudf::detail::is_element_valid_sync(col, 2, cudf::get_default_stream()));
  EXPECT_FALSE(cudf::detail::is_element_valid_sync(col, 3, cudf::get_default_stream()));
  EXPECT_TRUE(cudf::detail::is_element_valid_sync(col, 4, cudf::get_default_stream()));
}

TEST_F(IsElementValidTest, IsElementValidLarge)
{
  auto filter              = [](auto i) { return static_cast<bool>(i % 3); };
  auto val                 = cuda::counting_iterator<int32_t>{0};
  auto valid               = cudf::detail::make_counting_transform_iterator(0, filter);
  cudf::size_type num_rows = 1000;

  cudf::test::fixed_width_column_wrapper<int32_t> col(val, val + num_rows, valid);

  for (int i = 0; i < num_rows; i++) {
    EXPECT_EQ(cudf::detail::is_element_valid_sync(col, i, cudf::get_default_stream()), filter(i));
  }
}

TEST_F(IsElementValidTest, IsElementValidOffset)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 1, 1, 1, 1},
                                                      {true, false, false, false, true});
  {
    auto offset_col = cudf::slice(col, {1, 5}).front();
    EXPECT_FALSE(cudf::detail::is_element_valid_sync(offset_col, 0, cudf::get_default_stream()));
    EXPECT_FALSE(cudf::detail::is_element_valid_sync(offset_col, 1, cudf::get_default_stream()));
    EXPECT_FALSE(cudf::detail::is_element_valid_sync(offset_col, 2, cudf::get_default_stream()));
    EXPECT_TRUE(cudf::detail::is_element_valid_sync(offset_col, 3, cudf::get_default_stream()));
  }
  {
    auto offset_col = cudf::slice(col, {2, 5}).front();
    EXPECT_FALSE(cudf::detail::is_element_valid_sync(offset_col, 0, cudf::get_default_stream()));
    EXPECT_FALSE(cudf::detail::is_element_valid_sync(offset_col, 1, cudf::get_default_stream()));
    EXPECT_TRUE(cudf::detail::is_element_valid_sync(offset_col, 2, cudf::get_default_stream()));
  }
}

TEST_F(IsElementValidTest, IsElementValidOffsetLarge)
{
  auto filter              = [](auto i) { return static_cast<bool>(i % 3); };
  cudf::size_type offset   = 37;
  auto val                 = cuda::counting_iterator<int32_t>{0};
  auto valid               = cudf::detail::make_counting_transform_iterator(0, filter);
  cudf::size_type num_rows = 1000;

  cudf::test::fixed_width_column_wrapper<int32_t> col(val, val + num_rows, valid);
  auto offset_col = cudf::slice(col, {offset, num_rows}).front();

  for (int i = 0; i < offset_col.size(); i++) {
    EXPECT_EQ(cudf::detail::is_element_valid_sync(offset_col, i, cudf::get_default_stream()),
              filter(i + offset));
  }
}
