/*
 * SPDX-FileCopyrightText: Copyright 2018 BlazingDB, Inc.
 * SPDX-FileCopyrightText: Copyright 2018 Alexander Ocsa <cristhian@blazingdb.com>
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Alexander Ocsa <cristhian@blazingdb.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/dictionary/encode.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/iterator>

using namespace cudf::test::iterators;

struct ReplaceErrorTest : public cudf::test::BaseFixture {};

// Error: old-values and new-values size mismatch
TEST_F(ReplaceErrorTest, SizeMismatch)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_column{{7, 5, 6, 3, 1, 2, 8, 4},
                                                               {0, 0, 1, 1, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> values_to_replace_column{{10, 11, 12, 13}};

  ASSERT_THROW(cudf::replace_nulls(input_column, values_to_replace_column), cudf::logic_error);
}

// Error: column type mismatch
TEST_F(ReplaceErrorTest, TypeMismatch)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_column{{7, 5, 6, 3, 1, 2, 8, 4},
                                                               {0, 0, 1, 1, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<float> values_to_replace_column{
    {10, 11, 12, 13, 14, 15, 16, 17}};

  EXPECT_THROW(cudf::replace_nulls(input_column, values_to_replace_column), cudf::data_type_error);
}

// Error: column type mismatch
TEST_F(ReplaceErrorTest, TypeMismatchScalar)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_column{{7, 5, 6, 3, 1, 2, 8, 4},
                                                               {0, 0, 1, 1, 1, 1, 1, 1}};
  cudf::numeric_scalar<float> replacement(1);

  EXPECT_THROW(cudf::replace_nulls(input_column, replacement), cudf::data_type_error);
}

struct ReplaceNullsStringsTest : public cudf::test::BaseFixture {};

TEST_F(ReplaceNullsStringsTest, SimpleReplace)
{
  std::vector<std::string> input{"", "", "", "", "", "", "", ""};
  std::vector<cudf::valid_type> input_v{0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<std::string> replacement{"a", "b", "c", "d", "e", "f", "g", "h"};
  std::vector<cudf::valid_type> replacement_v{1, 1, 1, 1, 1, 1, 1, 1};

  cudf::test::strings_column_wrapper input_w{input.begin(), input.end(), input_v.begin()};
  cudf::test::strings_column_wrapper replacement_w{
    replacement.begin(), replacement.end(), replacement_v.begin()};
  cudf::test::strings_column_wrapper expected_w{
    replacement.begin(), replacement.end(), replacement_v.begin()};

  std::unique_ptr<cudf::column> result;
  ASSERT_NO_THROW(result = cudf::replace_nulls(input_w, replacement_w));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected_w);
}

TEST_F(ReplaceNullsStringsTest, ReplaceWithNulls)
{
  std::vector<std::string> input{"", "", "", "", "", "", "", ""};
  std::vector<cudf::valid_type> input_v{0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<std::string> replacement{"", "", "c", "d", "e", "f", "g", "h"};
  std::vector<cudf::valid_type> replacement_v{0, 0, 1, 1, 1, 1, 1, 1};

  cudf::test::strings_column_wrapper input_w{input.begin(), input.end(), input_v.begin()};
  cudf::test::strings_column_wrapper replacement_w{
    replacement.begin(), replacement.end(), replacement_v.begin()};
  cudf::test::strings_column_wrapper expected_w{
    replacement.begin(), replacement.end(), replacement_v.begin()};

  std::unique_ptr<cudf::column> result;
  ASSERT_NO_THROW(result = cudf::replace_nulls(input_w, replacement_w));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected_w);
}

TEST_F(ReplaceNullsStringsTest, ReplaceWithAllNulls)
{
  std::vector<std::string> input{"", "", "", "", "", "", "", ""};
  std::vector<cudf::valid_type> input_v{0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<std::string> replacement{"", "", "", "", "", "", "", ""};
  std::vector<cudf::valid_type> replacement_v{0, 0, 0, 0, 0, 0, 0, 0};

  cudf::test::strings_column_wrapper input_w{input.begin(), input.end(), input_v.begin()};
  cudf::test::strings_column_wrapper replacement_w{
    replacement.begin(), replacement.end(), replacement_v.begin()};
  cudf::test::strings_column_wrapper expected_w{input.begin(), input.end(), input_v.begin()};

  std::unique_ptr<cudf::column> result;
  ASSERT_NO_THROW(result = cudf::replace_nulls(input_w, replacement_w));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected_w);
}

TEST_F(ReplaceNullsStringsTest, ReplaceWithAllEmpty)
{
  std::vector<std::string> input{"", "", "", "", "", "", "", ""};
  std::vector<cudf::valid_type> input_v{0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<std::string> replacement{"", "", "", "", "", "", "", ""};
  std::vector<cudf::valid_type> replacement_v{1, 1, 1, 1, 1, 1, 1, 1};

  cudf::test::strings_column_wrapper input_w{input.begin(), input.end(), input_v.begin()};
  cudf::test::strings_column_wrapper replacement_w{
    replacement.begin(), replacement.end(), replacement_v.begin()};
  cudf::test::strings_column_wrapper expected_w{input.begin(), input.end(), replacement_v.begin()};

  std::unique_ptr<cudf::column> result;
  ASSERT_NO_THROW(result = cudf::replace_nulls(input_w, replacement_w));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected_w);
}

TEST_F(ReplaceNullsStringsTest, ReplaceNone)
{
  std::vector<std::string> input{"a", "b", "c", "d", "e", "f", "g", "h"};
  std::vector<cudf::valid_type> input_v{1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<std::string> replacement{"z", "a", "c", "d", "e", "f", "g", "h"};
  std::vector<cudf::valid_type> replacement_v{0, 0, 1, 1, 1, 1, 1, 1};

  cudf::test::strings_column_wrapper input_w{input.begin(), input.end(), input_v.begin()};
  cudf::test::strings_column_wrapper replacement_w{
    replacement.begin(), replacement.end(), replacement_v.begin()};
  cudf::test::strings_column_wrapper expected_w{input.begin(), input.end()};

  std::unique_ptr<cudf::column> result;
  ASSERT_NO_THROW(result = cudf::replace_nulls(input_w, replacement_w));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected_w);
}

TEST_F(ReplaceNullsStringsTest, SimpleReplaceScalar)
{
  std::vector<std::string> input{"", "", "", "", "", "", "", ""};
  std::vector<cudf::valid_type> input_v{0, 0, 0, 0, 0, 0, 0, 0};
  std::unique_ptr<cudf::scalar> repl = cudf::make_string_scalar("rep");
  repl->set_valid_async(true, cudf::get_default_stream());
  std::vector<std::string> expected{"rep", "rep", "rep", "rep", "rep", "rep", "rep", "rep"};

  cudf::test::strings_column_wrapper input_w{input.begin(), input.end(), input_v.begin()};
  cudf::test::strings_column_wrapper expected_w{expected.begin(), expected.end()};

  std::unique_ptr<cudf::column> result;
  ASSERT_NO_THROW(result = cudf::replace_nulls(input_w, *repl));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected_w);
}

struct ReplaceNullsPolicyStringTest : public cudf::test::BaseFixture {};

TEST_F(ReplaceNullsPolicyStringTest, PrecedingFill)
{
  cudf::test::strings_column_wrapper input({"head", "", "", "mid", "mid", "", "tail"},
                                           {1, 0, 0, 1, 1, 0, 1});

  cudf::test::strings_column_wrapper expected({"head", "head", "head", "mid", "mid", "mid", "tail"},
                                              no_nulls());

  auto result = cudf::replace_nulls(input, cudf::replace_policy::PRECEDING);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(ReplaceNullsPolicyStringTest, FollowingFill)
{
  cudf::test::strings_column_wrapper input({"head", "", "", "mid", "mid", "", "tail"},
                                           {1, 0, 0, 1, 1, 0, 1});

  cudf::test::strings_column_wrapper expected({"head", "mid", "mid", "mid", "mid", "tail", "tail"},
                                              no_nulls());

  auto result = cudf::replace_nulls(input, cudf::replace_policy::FOLLOWING);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(ReplaceNullsPolicyStringTest, PrecedingFillLeadingNulls)
{
  cudf::test::strings_column_wrapper input({"", "", "", "mid", "mid", "", "tail"},
                                           {0, 0, 0, 1, 1, 0, 1});

  cudf::test::strings_column_wrapper expected({"", "", "", "mid", "mid", "mid", "tail"},
                                              {0, 0, 0, 1, 1, 1, 1});

  auto result = cudf::replace_nulls(input, cudf::replace_policy::PRECEDING);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(ReplaceNullsPolicyStringTest, FollowingFillTrailingNulls)
{
  cudf::test::strings_column_wrapper input({"head", "", "", "mid", "mid", "", ""},
                                           {1, 0, 0, 1, 1, 0, 0});

  cudf::test::strings_column_wrapper expected({"head", "mid", "mid", "mid", "mid", "", ""},
                                              {1, 1, 1, 1, 1, 0, 0});

  auto result = cudf::replace_nulls(input, cudf::replace_policy::FOLLOWING);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

template <typename T>
struct ReplaceNullsTest : public cudf::test::BaseFixture {};

using test_types = cudf::test::NumericTypes;

TYPED_TEST_SUITE(ReplaceNullsTest, test_types);

template <typename T>
void ReplaceNullsColumn(cudf::test::fixed_width_column_wrapper<T> input,
                        cudf::test::fixed_width_column_wrapper<T> replacement_values,
                        cudf::test::fixed_width_column_wrapper<T> expected)
{
  std::unique_ptr<cudf::column> result;
  ASSERT_NO_THROW(result = cudf::replace_nulls(input, replacement_values));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

template <typename T>
void ReplaceNullsScalar(cudf::test::fixed_width_column_wrapper<T> input,
                        cudf::scalar const& replacement_value,
                        cudf::test::fixed_width_column_wrapper<T> expected)
{
  std::unique_ptr<cudf::column> result;
  ASSERT_NO_THROW(result = cudf::replace_nulls(input, replacement_value));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TYPED_TEST(ReplaceNullsTest, ReplaceColumn)
{
  auto const inputColumn =
    cudf::test::make_type_param_vector<TypeParam>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto const inputValid =
    cudf::test::make_type_param_vector<cudf::valid_type>({0, 0, 0, 0, 0, 1, 1, 1, 1, 1});
  auto const replacementColumn =
    cudf::test::make_type_param_vector<TypeParam>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

  ReplaceNullsColumn<TypeParam>(cudf::test::fixed_width_column_wrapper<TypeParam>(
                                  inputColumn.begin(), inputColumn.end(), inputValid.begin()),
                                cudf::test::fixed_width_column_wrapper<TypeParam>(
                                  replacementColumn.begin(), replacementColumn.end()),
                                cudf::test::fixed_width_column_wrapper<TypeParam>(
                                  replacementColumn.begin(), replacementColumn.end()));
}

TYPED_TEST(ReplaceNullsTest, ReplaceColumn_Empty)
{
  ReplaceNullsColumn<TypeParam>(cudf::test::fixed_width_column_wrapper<TypeParam>{},
                                cudf::test::fixed_width_column_wrapper<TypeParam>{},
                                cudf::test::fixed_width_column_wrapper<TypeParam>{});
}

TYPED_TEST(ReplaceNullsTest, ReplaceScalar)
{
  auto const inputColumn =
    cudf::test::make_type_param_vector<TypeParam>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto const inputValid =
    cudf::test::make_type_param_vector<cudf::valid_type>({0, 0, 0, 0, 0, 1, 1, 1, 1, 1});
  auto const expectedColumn =
    cudf::test::make_type_param_vector<TypeParam>({1, 1, 1, 1, 1, 5, 6, 7, 8, 9});
  cudf::numeric_scalar<TypeParam> replacement(1);

  ReplaceNullsScalar<TypeParam>(cudf::test::fixed_width_column_wrapper<TypeParam>(
                                  inputColumn.begin(), inputColumn.end(), inputValid.begin()),
                                replacement,
                                cudf::test::fixed_width_column_wrapper<TypeParam>(
                                  expectedColumn.begin(), expectedColumn.end()));
}

TYPED_TEST(ReplaceNullsTest, ReplacementHasNulls)
{
  using T = TypeParam;

  auto const input_column   = cudf::test::make_type_param_vector<T>({7, 5, 6, 3, 1, 2, 8, 4});
  auto const replace_column = cudf::test::make_type_param_vector<T>({4, 5, 6, 7, 8, 9, 0, 1});
  auto const result_column  = cudf::test::make_type_param_vector<T>({4, 5, 6, 3, 1, 2, 8, 4});

  auto const input_valid =
    cudf::test::make_type_param_vector<cudf::valid_type>({0, 0, 1, 1, 1, 1, 1, 1});
  auto const replace_valid =
    cudf::test::make_type_param_vector<cudf::valid_type>({1, 0, 1, 1, 1, 1, 1, 1});
  auto const result_valid =
    cudf::test::make_type_param_vector<cudf::valid_type>({1, 0, 1, 1, 1, 1, 1, 1});

  ReplaceNullsColumn<T>(cudf::test::fixed_width_column_wrapper<T>(
                          input_column.begin(), input_column.end(), input_valid.begin()),
                        cudf::test::fixed_width_column_wrapper<T>(
                          replace_column.begin(), replace_column.end(), replace_valid.begin()),
                        cudf::test::fixed_width_column_wrapper<T>(
                          result_column.begin(), result_column.end(), result_valid.begin()));
}

TYPED_TEST(ReplaceNullsTest, LargeScale)
{
  std::vector<TypeParam> inputColumn(10000);
  for (size_t i = 0; i < inputColumn.size(); i++)
    inputColumn[i] = i % 2;
  std::vector<cudf::valid_type> inputValid(10000);
  for (size_t i = 0; i < inputValid.size(); i++)
    inputValid[i] = i % 2;
  std::vector<TypeParam> expectedColumn(10000);
  for (size_t i = 0; i < expectedColumn.size(); i++)
    expectedColumn[i] = 1;

  ReplaceNullsColumn<TypeParam>(
    cudf::test::fixed_width_column_wrapper<TypeParam>(
      inputColumn.begin(), inputColumn.end(), inputValid.begin()),
    cudf::test::fixed_width_column_wrapper<TypeParam>(expectedColumn.begin(), expectedColumn.end()),
    cudf::test::fixed_width_column_wrapper<TypeParam>(expectedColumn.begin(),
                                                      expectedColumn.end()));
}

TYPED_TEST(ReplaceNullsTest, LargeScaleScalar)
{
  std::vector<TypeParam> inputColumn(10000);
  for (size_t i = 0; i < inputColumn.size(); i++)
    inputColumn[i] = i % 2;
  std::vector<cudf::valid_type> inputValid(10000);
  for (size_t i = 0; i < inputValid.size(); i++)
    inputValid[i] = i % 2;
  std::vector<TypeParam> expectedColumn(10000);
  for (size_t i = 0; i < expectedColumn.size(); i++)
    expectedColumn[i] = 1;
  cudf::numeric_scalar<TypeParam> replacement(1);

  ReplaceNullsScalar<TypeParam>(cudf::test::fixed_width_column_wrapper<TypeParam>(
                                  inputColumn.begin(), inputColumn.end(), inputValid.begin()),
                                replacement,
                                cudf::test::fixed_width_column_wrapper<TypeParam>(
                                  expectedColumn.begin(), expectedColumn.end()));
}

template <typename T>
struct ReplaceNullsPolicyTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ReplaceNullsPolicyTest, test_types);

template <typename T>
void TestReplaceNullsWithPolicy(cudf::test::fixed_width_column_wrapper<T> input,
                                cudf::test::fixed_width_column_wrapper<T> expected,
                                cudf::replace_policy policy)
{
  auto result = cudf::replace_nulls(input, policy);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TYPED_TEST(ReplaceNullsPolicyTest, PrecedingFill)
{
  auto const col  = cudf::test::make_type_param_vector<TypeParam>({42, 2, 1, -10, 20, -30});
  auto const mask = cudf::test::make_type_param_vector<cudf::valid_type>({1, 0, 0, 1, 0, 1});
  auto const expect_col =
    cudf::test::make_type_param_vector<TypeParam>({42, 42, 42, -10, -10, -30});

  TestReplaceNullsWithPolicy(
    cudf::test::fixed_width_column_wrapper<TypeParam>(col.begin(), col.end(), mask.begin()),
    cudf::test::fixed_width_column_wrapper<TypeParam>(
      expect_col.begin(), expect_col.end(), no_nulls()),
    cudf::replace_policy::PRECEDING);
}

TYPED_TEST(ReplaceNullsPolicyTest, FollowingFill)
{
  auto const col  = cudf::test::make_type_param_vector<TypeParam>({42, 2, 1, -10, 20, -30});
  auto const mask = cudf::test::make_type_param_vector<cudf::valid_type>({1, 0, 0, 1, 0, 1});
  auto const expect_col =
    cudf::test::make_type_param_vector<TypeParam>({42, -10, -10, -10, -30, -30});

  TestReplaceNullsWithPolicy(
    cudf::test::fixed_width_column_wrapper<TypeParam>(col.begin(), col.end(), mask.begin()),
    cudf::test::fixed_width_column_wrapper<TypeParam>(
      expect_col.begin(), expect_col.end(), no_nulls()),
    cudf::replace_policy::FOLLOWING);
}

TYPED_TEST(ReplaceNullsPolicyTest, PrecedingFillLeadingNulls)
{
  auto const col         = cudf::test::make_type_param_vector<TypeParam>({1, 2, 3, 4, 5});
  auto const mask        = cudf::test::make_type_param_vector<cudf::valid_type>({0, 0, 1, 0, 1});
  auto const expect_col  = cudf::test::make_type_param_vector<TypeParam>({1, 2, 3, 3, 5});
  auto const expect_mask = cudf::test::make_type_param_vector<cudf::valid_type>({0, 0, 1, 1, 1});

  TestReplaceNullsWithPolicy(
    cudf::test::fixed_width_column_wrapper<TypeParam>(col.begin(), col.end(), mask.begin()),
    cudf::test::fixed_width_column_wrapper<TypeParam>(
      expect_col.begin(), expect_col.end(), expect_mask.begin()),
    cudf::replace_policy::PRECEDING);
}

TYPED_TEST(ReplaceNullsPolicyTest, FollowingFillTrailingNulls)
{
  auto const col         = cudf::test::make_type_param_vector<TypeParam>({1, 2, 3, 4, 5});
  auto const mask        = cudf::test::make_type_param_vector<cudf::valid_type>({1, 0, 1, 0, 0});
  auto const expect_col  = cudf::test::make_type_param_vector<TypeParam>({1, 3, 3, 4, 5});
  auto const expect_mask = cudf::test::make_type_param_vector<cudf::valid_type>({1, 1, 1, 0, 0});

  TestReplaceNullsWithPolicy(
    cudf::test::fixed_width_column_wrapper<TypeParam>(col.begin(), col.end(), mask.begin()),
    cudf::test::fixed_width_column_wrapper<TypeParam>(
      expect_col.begin(), expect_col.end(), expect_mask.begin()),
    cudf::replace_policy::FOLLOWING);
}

TYPED_TEST(ReplaceNullsPolicyTest, PrecedingFillLargeArray)
{
  cudf::size_type const sz = 1000;

  // Source: 0, null, null...
  auto src_begin = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type i) { return static_cast<TypeParam>(i); });
  auto src_end         = src_begin + sz;
  auto nulls_idx_begin = cuda::counting_iterator<cudf::size_type>{1};
  auto nulls_idx_end   = nulls_idx_begin + sz - 1;

  // Expected: 0, 0, 0, ...
  auto expected_begin = cuda::make_constant_iterator(0);
  auto expected_end   = expected_begin + sz;

  TestReplaceNullsWithPolicy(
    cudf::test::fixed_width_column_wrapper<TypeParam>(
      src_begin, src_end, nulls_at(nulls_idx_begin, nulls_idx_end)),
    cudf::test::fixed_width_column_wrapper<TypeParam>(expected_begin, expected_end, no_nulls()),
    cudf::replace_policy::PRECEDING);
}

TYPED_TEST(ReplaceNullsPolicyTest, FollowingFillLargeArray)
{
  cudf::size_type const sz = 1000;

  // Source: null, ... null, 999
  auto src_begin = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type i) { return static_cast<TypeParam>(i); });
  auto src_end         = src_begin + sz;
  auto nulls_idx_begin = cuda::counting_iterator<cudf::size_type>{0};
  auto nulls_idx_end   = nulls_idx_begin + sz - 1;

  // Expected: 999, 999, 999, ...
  auto expected_begin = cuda::make_constant_iterator(sz - 1);
  auto expected_end   = expected_begin + sz;

  TestReplaceNullsWithPolicy(
    cudf::test::fixed_width_column_wrapper<TypeParam>(
      src_begin, src_end, nulls_at(nulls_idx_begin, nulls_idx_end)),
    cudf::test::fixed_width_column_wrapper<TypeParam>(expected_begin, expected_end, no_nulls()),
    cudf::replace_policy::FOLLOWING);
}

template <typename T>
struct ReplaceNullsFixedPointTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ReplaceNullsFixedPointTest, cudf::test::FixedPointTypes);

TYPED_TEST(ReplaceNullsFixedPointTest, ReplaceColumn)
{
  auto const scale = numeric::scale_type{0};
  auto const sz    = std::size_t{1000};
  auto data_begin =
    cudf::detail::make_counting_transform_iterator(0, [&](auto i) { return TypeParam{i, scale}; });
  auto valid_begin = cudf::test::iterators::nulls_at_multiples_of(3);
  auto replace_begin =
    cudf::detail::make_counting_transform_iterator(0, [&](auto i) { return TypeParam{-2, scale}; });
  auto expected_begin = cudf::detail::make_counting_transform_iterator(0, [&](auto i) {
    int val = i % 3 ? static_cast<int>(i) : -2;
    return TypeParam{val, scale};
  });

  ReplaceNullsColumn<TypeParam>(
    cudf::test::fixed_width_column_wrapper<TypeParam>(data_begin, data_begin + sz, valid_begin),
    cudf::test::fixed_width_column_wrapper<TypeParam>(replace_begin, replace_begin + sz),
    cudf::test::fixed_width_column_wrapper<TypeParam>(expected_begin, expected_begin + sz));
}

TYPED_TEST(ReplaceNullsFixedPointTest, ReplaceColumn_Empty)
{
  ReplaceNullsColumn<TypeParam>(cudf::test::fixed_width_column_wrapper<TypeParam>{},
                                cudf::test::fixed_width_column_wrapper<TypeParam>{},
                                cudf::test::fixed_width_column_wrapper<TypeParam>{});
}

TYPED_TEST(ReplaceNullsFixedPointTest, ReplaceScalar)
{
  auto const scale = numeric::scale_type{0};
  auto const sz    = std::size_t{1000};
  auto data_begin =
    cudf::detail::make_counting_transform_iterator(0, [&](auto i) { return TypeParam{i, scale}; });
  auto valid_begin    = cudf::test::iterators::nulls_at_multiples_of(3);
  auto expected_begin = cudf::detail::make_counting_transform_iterator(0, [&](auto i) {
    int val = i % 3 ? static_cast<int>(i) : -2;
    return TypeParam{val, scale};
  });

  cudf::fixed_point_scalar<TypeParam> replacement{-2, scale};

  ReplaceNullsScalar<TypeParam>(
    cudf::test::fixed_width_column_wrapper<TypeParam>(data_begin, data_begin + sz, valid_begin),
    replacement,
    cudf::test::fixed_width_column_wrapper<TypeParam>(expected_begin, expected_begin + sz));
}

TYPED_TEST(ReplaceNullsFixedPointTest, ReplacementHasNulls)
{
  auto const scale = numeric::scale_type{0};
  auto const sz    = std::size_t{1000};
  auto data_begin =
    cudf::detail::make_counting_transform_iterator(0, [&](auto i) { return TypeParam{i, scale}; });
  auto data_valid_begin = cudf::test::iterators::nulls_at_multiples_of(3);
  auto replace_begin =
    cudf::detail::make_counting_transform_iterator(0, [&](auto i) { return TypeParam{-2, scale}; });
  auto replace_valid_begin  = cudf::test::iterators::nulls_at_multiples_of(2);
  auto expected_begin       = cudf::detail::make_counting_transform_iterator(0, [&](auto i) {
    int val = i % 3 ? static_cast<int>(i) : -2;
    return TypeParam{val, scale};
  });
  auto expected_valid_begin = cudf::test::iterators::nulls_at_multiples_of(6);

  ReplaceNullsColumn<TypeParam>(cudf::test::fixed_width_column_wrapper<TypeParam>(
                                  data_begin, data_begin + sz, data_valid_begin),
                                cudf::test::fixed_width_column_wrapper<TypeParam>(
                                  replace_begin, replace_begin + sz, replace_valid_begin),
                                cudf::test::fixed_width_column_wrapper<TypeParam>(
                                  expected_begin, expected_begin + sz, expected_valid_begin));
}

template <typename T>
struct ReplaceNullsPolicyFixedPointTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ReplaceNullsPolicyFixedPointTest, cudf::test::FixedPointTypes);

TYPED_TEST(ReplaceNullsPolicyFixedPointTest, PrecedingFill)
{
  using fp     = TypeParam;
  auto const s = numeric::scale_type{0};
  auto col     = cudf::test::fixed_width_column_wrapper<TypeParam>(
    {fp{42, s}, fp{2, s}, fp{1, s}, fp{-10, s}, fp{20, s}, fp{-30, s}}, {1, 0, 0, 1, 0, 1});
  auto expect_col = cudf::test::fixed_width_column_wrapper<TypeParam>(
    {fp{42, s}, fp{42, s}, fp{42, s}, fp{-10, s}, fp{-10, s}, fp{-30, s}}, {1, 1, 1, 1, 1, 1});

  TestReplaceNullsWithPolicy(
    std::move(col), std::move(expect_col), cudf::replace_policy::PRECEDING);
}

TYPED_TEST(ReplaceNullsPolicyFixedPointTest, FollowingFill)
{
  using fp     = TypeParam;
  auto const s = numeric::scale_type{0};
  auto col     = cudf::test::fixed_width_column_wrapper<TypeParam>(
    {fp{42, s}, fp{2, s}, fp{1, s}, fp{-10, s}, fp{20, s}, fp{-30, s}}, {1, 0, 0, 1, 0, 1});
  auto expect_col = cudf::test::fixed_width_column_wrapper<TypeParam>(
    {fp{42, s}, fp{-10, s}, fp{-10, s}, fp{-10, s}, fp{-30, s}, fp{-30, s}}, {1, 1, 1, 1, 1, 1});

  TestReplaceNullsWithPolicy(
    std::move(col), std::move(expect_col), cudf::replace_policy::FOLLOWING);
}

TYPED_TEST(ReplaceNullsPolicyFixedPointTest, PrecedingFillLeadingNulls)
{
  using fp     = TypeParam;
  auto const s = numeric::scale_type{0};
  auto col     = cudf::test::fixed_width_column_wrapper<TypeParam>(
    {fp{1, s}, fp{2, s}, fp{3, s}, fp{4, s}, fp{5, s}}, {0, 0, 1, 0, 1});
  auto expect_col = cudf::test::fixed_width_column_wrapper<TypeParam>(
    {fp{1, s}, fp{2, s}, fp{3, s}, fp{3, s}, fp{5, s}}, {0, 0, 1, 1, 1});

  TestReplaceNullsWithPolicy(
    std::move(col), std::move(expect_col), cudf::replace_policy::PRECEDING);
}

TYPED_TEST(ReplaceNullsPolicyFixedPointTest, FollowingFillTrailingNulls)
{
  using fp     = TypeParam;
  auto const s = numeric::scale_type{0};
  auto col     = cudf::test::fixed_width_column_wrapper<TypeParam>(
    {fp{1, s}, fp{2, s}, fp{3, s}, fp{4, s}, fp{5, s}}, {1, 0, 1, 0, 0});
  auto expect_col = cudf::test::fixed_width_column_wrapper<TypeParam>(
    {fp{1, s}, fp{3, s}, fp{3, s}, fp{4, s}, fp{5, s}}, {1, 1, 1, 0, 0});

  TestReplaceNullsWithPolicy(
    std::move(col), std::move(expect_col), cudf::replace_policy::FOLLOWING);
}

struct ReplaceDictionaryTest : public cudf::test::BaseFixture {};

TEST_F(ReplaceDictionaryTest, ReplaceNulls)
{
  auto input_w       = cudf::test::strings_column_wrapper({"c", "", "", "a", "d", "d", "", ""},
                                                          {1, 0, 0, 1, 1, 1, 0, 0});
  auto input         = cudf::dictionary::encode(input_w);
  auto replacement_w = cudf::test::strings_column_wrapper({"c", "c", "", "a", "d", "d", "b", ""},
                                                          {1, 1, 0, 1, 1, 1, 1, 0});
  auto replacement   = cudf::dictionary::encode(replacement_w);
  auto expected      = cudf::test::strings_column_wrapper({"c", "c", "", "a", "d", "d", "b", ""},
                                                          {1, 1, 0, 1, 1, 1, 1, 0});

  auto result  = cudf::replace_nulls(input->view(), replacement->view());
  auto decoded = cudf::dictionary::decode(result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), expected);
}

TEST_F(ReplaceDictionaryTest, ReplaceNullsWithScalar)
{
  auto input_w  = cudf::test::strings_column_wrapper({"c", "", "", "a", "d", "d", "", ""},
                                                     {1, 0, 0, 1, 1, 1, 0, 0});
  auto input    = cudf::dictionary::encode(input_w);
  auto result   = cudf::replace_nulls(input->view(), cudf::string_scalar("b"));
  auto decoded  = cudf::dictionary::decode(result->view());
  auto expected = cudf::test::strings_column_wrapper({"c", "b", "b", "a", "d", "d", "b", "b"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), expected);
}

TEST_F(ReplaceDictionaryTest, ReplaceNullsError)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_w({1, 1, 2, 2}, {1, 0, 0, 1});
  auto input = cudf::dictionary::encode(input_w);
  cudf::test::fixed_width_column_wrapper<int64_t> replacement_w({1, 2, 3, 4});
  auto replacement = cudf::dictionary::encode(replacement_w);

  EXPECT_THROW(cudf::replace_nulls(input->view(), replacement->view()), cudf::data_type_error);
  EXPECT_THROW(cudf::replace_nulls(input->view(), cudf::string_scalar("x")), cudf::data_type_error);

  cudf::test::fixed_width_column_wrapper<int64_t> input_one_w({1}, {0});
  auto input_one  = cudf::dictionary::encode(input_one_w);
  auto dict_input = cudf::dictionary_column_view(input_one->view());
  auto dict_repl  = cudf::dictionary_column_view(replacement->view());
  EXPECT_THROW(cudf::replace_nulls(input->view(), replacement->view()), cudf::data_type_error);
}

TEST_F(ReplaceDictionaryTest, ReplaceNullsEmpty)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input_empty_w({});
  auto input_empty = cudf::dictionary::encode(input_empty_w);
  auto result      = cudf::replace_nulls(input_empty->view(), input_empty->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), input_empty->view());
}

TEST_F(ReplaceDictionaryTest, ReplaceNullsNoNulls)
{
  cudf::test::fixed_width_column_wrapper<int8_t> input_w({1, 1, 1});
  auto input  = cudf::dictionary::encode(input_w);
  auto result = cudf::replace_nulls(input->view(), input->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), input->view());

  result = cudf::replace_nulls(input->view(), cudf::numeric_scalar<int8_t>(0, false));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), input->view());
}

struct ReplaceNullsPolicyDictionaryTest : public cudf::test::BaseFixture {};

TEST_F(ReplaceNullsPolicyDictionaryTest, PrecedingFill)
{
  cudf::test::strings_column_wrapper input_w({"head", "", "", "mid1", "mid2", "tail", "", ""},
                                             {1, 0, 0, 1, 1, 1, 0, 0});
  auto input = cudf::dictionary::encode(input_w);

  cudf::test::strings_column_wrapper expected_w(
    {"head", "head", "head", "mid1", "mid2", "tail", "tail", "tail"}, no_nulls());
  auto expected = cudf::dictionary::encode(expected_w);

  auto result = cudf::replace_nulls(*input, cudf::replace_policy::PRECEDING);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected->view());
}

TEST_F(ReplaceNullsPolicyDictionaryTest, FollowingFill)
{
  cudf::test::strings_column_wrapper input_w({"head", "", "", "mid1", "mid2", "", "", "tail"},
                                             {1, 0, 0, 1, 1, 0, 0, 1});
  auto input = cudf::dictionary::encode(input_w);

  cudf::test::strings_column_wrapper expected_w(
    {"head", "mid1", "mid1", "mid1", "mid2", "tail", "tail", "tail"}, no_nulls());
  auto expected = cudf::dictionary::encode(expected_w);

  auto result = cudf::replace_nulls(*input, cudf::replace_policy::FOLLOWING);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected->view());
}

TEST_F(ReplaceNullsPolicyDictionaryTest, PrecedingFillLeadingNulls)
{
  cudf::test::strings_column_wrapper input_w({"", "", "", "mid1", "mid2", "", "", "tail"},
                                             {0, 0, 0, 1, 1, 0, 0, 1});
  auto input = cudf::dictionary::encode(input_w);

  cudf::test::strings_column_wrapper expected_w(
    {"", "", "", "mid1", "mid2", "mid2", "mid2", "tail"}, {0, 0, 0, 1, 1, 1, 1, 1});
  auto expected = cudf::dictionary::encode(expected_w);

  auto result = cudf::replace_nulls(*input, cudf::replace_policy::PRECEDING);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected->view());
}

TEST_F(ReplaceNullsPolicyDictionaryTest, FollowingFillTrailingNulls)
{
  cudf::test::strings_column_wrapper input_w({"head", "", "", "mid", "tail", "", "", ""},
                                             {1, 0, 0, 1, 1, 0, 0, 0});
  auto input = cudf::dictionary::encode(input_w);

  cudf::test::strings_column_wrapper expected_w({"head", "mid", "mid", "mid", "tail", "", "", ""},
                                                {1, 1, 1, 1, 1, 0, 0, 0});
  auto expected = cudf::dictionary::encode(expected_w);

  auto result = cudf::replace_nulls(*input, cudf::replace_policy::FOLLOWING);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected->view());
}

CUDF_TEST_PROGRAM_MAIN()
