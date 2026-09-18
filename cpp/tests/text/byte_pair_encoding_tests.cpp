/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/byte_pair_encoding.hpp>

struct TextBytePairEncodingTest : public cudf::test::BaseFixture {};

TEST_F(TextBytePairEncodingTest, EmptyCharacterRows)
{
  cudf::test::strings_column_wrapper merge_pair_strings({"a b"});
  auto merge_pairs = nvtext::load_merge_pairs(cudf::strings_column_view(merge_pair_strings));

  cudf::test::strings_column_wrapper input({"", "", ""}, {1, 0, 1});
  auto results = nvtext::byte_pair_encoding(cudf::strings_column_view(input), *merge_pairs);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, input);
}

TEST_F(TextBytePairEncodingTest, InvalidSeparator)
{
  cudf::test::strings_column_wrapper merge_pair_strings({"a b"});
  auto merge_pairs = nvtext::load_merge_pairs(cudf::strings_column_view(merge_pair_strings));
  cudf::test::strings_column_wrapper input({"", ""});

  EXPECT_THROW(nvtext::byte_pair_encoding(
                 cudf::strings_column_view(input), *merge_pairs, cudf::string_scalar("", false)),
               cudf::logic_error);
}
