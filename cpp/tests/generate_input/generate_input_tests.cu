/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "generate_input.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <thrust/host_vector.h>

#include <vector>

// Tests for data_profile::set_distribution_params routing. A call that passes bounds whose type
// does not match the column type must still take effect instead of being silently ignored, which
// previously made the generator fall back to the default distribution for that type.

struct GenerateInputTest : public cudf::test::BaseFixture {};

TEST_F(GenerateInputTest, IntBoundsOnFloatingType)
{
  data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_to_id<double>(), distribution_id::UNIFORM, 0, 100);

  auto const params = profile.get_distribution_params<double>();
  EXPECT_EQ(params.id, distribution_id::UNIFORM);
  EXPECT_DOUBLE_EQ(params.lower_bound, 0.);
  EXPECT_DOUBLE_EQ(params.upper_bound, 100.);

  auto const column = create_random_column(cudf::type_to_id<double>(), row_count{100}, profile);
  auto const values = cudf::test::to_host<double>(column->view()).first;
  ASSERT_EQ(values.size(), 100);
  for (auto const value : values) {
    EXPECT_GE(value, 0.);
    EXPECT_LE(value, 100.);
  }
}

TEST_F(GenerateInputTest, FloatBoundsOnIntegralType)
{
  data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_to_id<int32_t>(), distribution_id::UNIFORM, 0.0, 100.0);

  auto const params = profile.get_distribution_params<int32_t>();
  EXPECT_EQ(params.id, distribution_id::UNIFORM);
  EXPECT_EQ(params.lower_bound, 0);
  EXPECT_EQ(params.upper_bound, 100);

  auto const column = create_random_column(cudf::type_to_id<int32_t>(), row_count{100}, profile);
  auto const values = cudf::test::to_host<int32_t>(column->view()).first;
  ASSERT_EQ(values.size(), 100);
  for (auto const value : values) {
    EXPECT_GE(value, 0);
    EXPECT_LE(value, 100);
  }
}

TEST_F(GenerateInputTest, FloatBoundsOnFixedPointType)
{
  data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_to_id<numeric::decimal64>(), distribution_id::UNIFORM, 0.0, 1000.0);

  auto const params = profile.get_distribution_params<numeric::decimal64>();
  EXPECT_EQ(params.id, distribution_id::UNIFORM);
  EXPECT_EQ(params.lower_bound, 0);
  EXPECT_EQ(params.upper_bound, 1000);
}

TEST_F(GenerateInputTest, IntBoundsOnFixedPointType)
{
  data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_to_id<numeric::decimal64>(), distribution_id::UNIFORM, 0, 1000);

  auto const params = profile.get_distribution_params<numeric::decimal64>();
  EXPECT_EQ(params.id, distribution_id::UNIFORM);
  EXPECT_EQ(params.lower_bound, 0);
  EXPECT_EQ(params.upper_bound, 1000);
}

TEST_F(GenerateInputTest, FloatBoundsOnTimestampType)
{
  data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_to_id<cudf::timestamp_s>(), distribution_id::UNIFORM, -100.0, 100.0);

  auto const params = profile.get_distribution_params<cudf::timestamp_s>();
  EXPECT_EQ(params.id, distribution_id::UNIFORM);
  EXPECT_EQ(params.lower_bound, -100);
  EXPECT_EQ(params.upper_bound, 100);
}

TEST_F(GenerateInputTest, FloatBoundsOnStringLengths)
{
  data_profile const profile = data_profile_builder().no_validity().distribution(
    cudf::type_id::STRING, distribution_id::GEOMETRIC, 0.0, 64.0);

  auto const params = profile.get_distribution_params<cudf::string_view>().length_params;
  EXPECT_EQ(params.id, distribution_id::GEOMETRIC);
  EXPECT_EQ(params.lower_bound, 0);
  EXPECT_EQ(params.upper_bound, 64);
}

TEST_F(GenerateInputTest, FloatBoundsOnListLengths)
{
  data_profile const profile = data_profile_builder().no_validity().distribution(
    cudf::type_id::LIST, distribution_id::GEOMETRIC, 5.0, 128.0);

  auto const params = profile.get_distribution_params<cudf::list_view>().length_params;
  EXPECT_EQ(params.id, distribution_id::GEOMETRIC);
  EXPECT_EQ(params.lower_bound, 5);
  EXPECT_EQ(params.upper_bound, 128);
}

TEST_F(GenerateInputTest, IntBoundsOnNumericGroup)
{
  data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
    type_group_id::NUMERIC, distribution_id::UNIFORM, -50, 50);

  auto const float_params = profile.get_distribution_params<double>();
  EXPECT_EQ(float_params.id, distribution_id::UNIFORM);
  EXPECT_DOUBLE_EQ(float_params.lower_bound, -50.);
  EXPECT_DOUBLE_EQ(float_params.upper_bound, 50.);

  auto const int_params = profile.get_distribution_params<int32_t>();
  EXPECT_EQ(int_params.id, distribution_id::UNIFORM);
  EXPECT_EQ(int_params.lower_bound, -50);
  EXPECT_EQ(int_params.upper_bound, 50);

  auto const float_column =
    create_random_column(cudf::type_to_id<double>(), row_count{100}, profile);
  for (auto const value : cudf::test::to_host<double>(float_column->view()).first) {
    EXPECT_GE(value, -50.);
    EXPECT_LE(value, 50.);
  }

  auto const int_column =
    create_random_column(cudf::type_to_id<int32_t>(), row_count{100}, profile);
  for (auto const value : cudf::test::to_host<int32_t>(int_column->view()).first) {
    EXPECT_GE(value, -50);
    EXPECT_LE(value, 50);
  }
}
