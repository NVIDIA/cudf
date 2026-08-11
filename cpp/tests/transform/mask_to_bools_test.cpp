/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

struct MaskToBools : public cudf::test::BaseFixture {};

TEST_F(MaskToBools, NullDataWithZeroLength)
{
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({});
  auto out      = cudf::mask_to_bools(nullptr, 0, 0);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, out->view());
}

TEST_F(MaskToBools, NullDataWithNonZeroLength)
{
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({});

  EXPECT_THROW(cudf::mask_to_bools(nullptr, 0, 2), cudf::logic_error);
}

TEST_F(MaskToBools, ImproperBitRange)
{
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({});

  EXPECT_THROW(cudf::mask_to_bools(nullptr, 2, 1), cudf::logic_error);
}

struct MaskToBoolsTest
  : public MaskToBools,
    public ::testing::WithParamInterface<std::tuple<cudf::size_type, cudf::size_type>> {};

TEST_P(MaskToBoolsTest, LargeDataSizeTest)
{
  auto data                       = std::vector<bool>(10000);
  auto const [begin_bit, end_bit] = GetParam();
  std::transform(
    data.cbegin(), data.cend(), data.begin(), [](auto val) { return rand() % 2 == 0; });

  auto col      = cudf::test::fixed_width_column_wrapper<bool>(data.begin(), data.end());
  auto expected = cudf::slice(static_cast<cudf::column_view>(col), {begin_bit, end_bit}).front();

  auto mask = cudf::bools_to_mask(col);

  auto out = cudf::mask_to_bools(
    static_cast<cudf::bitmask_type const*>(mask.first->data()), begin_bit, end_bit);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, out->view());
}

INSTANTIATE_TEST_CASE_P(MaskToBools,
                        MaskToBoolsTest,
                        ::testing::Values(std::make_tuple(0, 0),
                                          std::make_tuple(0, 500),
                                          std::make_tuple(500, 7456),
                                          std::make_tuple(7456, 10000),
                                          std::make_tuple(0, 10000)));
