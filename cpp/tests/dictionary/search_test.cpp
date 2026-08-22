/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/search.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar_factories.hpp>

struct DictionarySearchTest : public cudf::test::BaseFixture {};

TEST_F(DictionarySearchTest, StringsColumn)
{
  cudf::test::dictionary_column_wrapper<std::string> dictionary(
    {"fff", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "", ""},
    {true, true, true, true, true, true, true, true, false});

  auto result = cudf::dictionary::get_index(dictionary, cudf::string_scalar("ccc"));
  EXPECT_TRUE(result->is_valid());

  result = cudf::dictionary::get_index(dictionary, cudf::string_scalar("eee"));
  EXPECT_FALSE(result->is_valid());
}

TEST_F(DictionarySearchTest, WithNulls)
{
  cudf::test::dictionary_column_wrapper<int64_t> dictionary({9, 8, 7, 6, 4},
                                                            {false, true, true, false, true});

  auto result = cudf::dictionary::get_index(dictionary, cudf::numeric_scalar<int64_t>(4));
  EXPECT_TRUE(result->is_valid());

  result = cudf::dictionary::get_index(dictionary, cudf::numeric_scalar<int64_t>(5));
  EXPECT_FALSE(result->is_valid());
}

TEST_F(DictionarySearchTest, FixedPointColumn)
{
  using decimal_type = numeric::decimal64;
  auto const scale   = numeric::scale_type{-2};
  auto const keys    = cudf::test::fixed_point_column_wrapper<int64_t>{{100, 123, 250}, scale};
  auto const dictionary = cudf::dictionary::encode(keys);
  auto const present    = cudf::make_fixed_point_scalar<decimal_type>(123, scale);
  auto const missing    = cudf::make_fixed_point_scalar<decimal_type>(124, scale);
  auto const dictionary_view = cudf::dictionary_column_view{dictionary->view()};

  EXPECT_TRUE(cudf::dictionary::get_index(dictionary_view, *present)->is_valid());
  EXPECT_FALSE(cudf::dictionary::get_index(dictionary_view, *missing)->is_valid());
}

TEST_F(DictionarySearchTest, EmptyColumn)
{
  cudf::test::dictionary_column_wrapper<int64_t> dictionary{};
  cudf::numeric_scalar<int64_t> key(7);
  auto result = cudf::dictionary::get_index(dictionary, key);
  EXPECT_FALSE(result->is_valid());
}

TEST_F(DictionarySearchTest, Errors)
{
  cudf::test::dictionary_column_wrapper<int64_t> dictionary({1, 2, 3});
  cudf::numeric_scalar<double> key(7);
  EXPECT_THROW(cudf::dictionary::get_index(dictionary, key), std::invalid_argument);
}
