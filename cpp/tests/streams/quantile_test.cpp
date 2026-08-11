/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/detail/tdigest/tdigest.hpp>
#include <cudf/quantiles.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <stdexcept>

struct QuantileTest : public cudf::test::BaseFixture {};

TEST_F(QuantileTest, TestMultiColumnUnsorted)
{
  auto input_a = cudf::test::strings_column_wrapper(
    {"C", "B", "A", "A", "D", "B", "D", "B", "D", "C", "C", "C",
     "D", "B", "D", "B", "C", "C", "A", "D", "B", "A", "A", "A"},
    {true, true, true, true, true, true, true, true, true, true, true, true,
     true, true, true, true, true, true, true, true, true, true, true, true});

  cudf::test::fixed_width_column_wrapper<numeric::decimal32, int32_t> input_b(
    {4, 3, 5, 0, 1, 0, 4, 1, 5, 3, 0, 5, 2, 4, 3, 2, 1, 2, 3, 0, 5, 1, 4, 2},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

  auto input = cudf::table_view({input_a, input_b});

  auto actual = cudf::quantiles(input,
                                {0.0f, 0.5f, 0.7f, 0.25f, 1.0f},
                                cudf::interpolation::NEAREST,
                                cudf::sorted::NO,
                                {cudf::order::ASCENDING, cudf::order::DESCENDING},
                                {},
                                cudf::test::get_default_stream());
}

TEST_F(QuantileTest, TestEmpty)
{
  auto input = cudf::test::fixed_width_column_wrapper<numeric::decimal32>({});
  cudf::quantile(
    input, {0.5, 0.25}, cudf::interpolation::LINEAR, {}, true, cudf::test::get_default_stream());
}

TEST_F(QuantileTest, EmptyInput)
{
  auto empty_ = cudf::tdigest::detail::make_empty_tdigests_column(
    1, cudf::test::get_default_stream(), cudf::get_current_device_resource_ref());
  cudf::test::fixed_width_column_wrapper<double> percentiles{0.0, 0.25, 0.3};

  std::vector<cudf::column_view> input;
  input.push_back(*empty_);
  input.push_back(*empty_);
  input.push_back(*empty_);
  auto empty = cudf::concatenate(input, cudf::test::get_default_stream());

  cudf::tdigest::tdigest_column_view tdv(*empty);
  auto result = cudf::percentile_approx(tdv, percentiles, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
