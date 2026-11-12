/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/strings/attributes.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cuda/iterator>

struct StringsAttributesTest : public cudf::test::BaseFixture {};

TEST_F(StringsAttributesTest, CodePoints)
{
  std::vector<char const*> h_strings{"eee", "bb", nullptr, "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::code_points(strings_view, cudf::test::get_default_stream());
}

TEST_F(StringsAttributesTest, CountCharacters)
{
  std::vector<std::string> h_strings(
    40000, "something a bit longer than 32 bytes ééé ééé ééé ééé ééé ééé ééé");
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::count_characters(strings_view, cudf::test::get_default_stream());
}

TEST_F(StringsAttributesTest, CountBytes)
{
  std::vector<char const*> h_strings{
    "eee", "bb", nullptr, "", "aa", "ééé", "something a bit longer than 32 bytes"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::count_bytes(strings_view, cudf::test::get_default_stream());
}
