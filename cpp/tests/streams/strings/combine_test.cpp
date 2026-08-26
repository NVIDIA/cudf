/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/lists_column_wrapper.hpp>

#include <cudf/strings/combine.hpp>
#include <cudf/strings/repeat_strings.hpp>

#include <string>

class StringsCombineTest : public cudf::test::BaseFixture {};

TEST_F(StringsCombineTest, Concatenate)
{
  auto input = cudf::test::strings_column_wrapper({"Héllo", "thesé", "tést"});
  auto view  = cudf::table_view({input, input});

  auto separators      = cudf::test::strings_column_wrapper({"_", ".", " "});
  auto separators_view = cudf::strings_column_view(separators);
  auto sep_on_null     = cudf::strings::separator_on_nulls::YES;

  auto const separator = cudf::string_scalar(" ", true, cudf::test::get_default_stream());
  auto const narep     = cudf::string_scalar("n/a", true, cudf::test::get_default_stream());
  cudf::strings::concatenate(view, separator, narep, sep_on_null, cudf::test::get_default_stream());
  cudf::strings::concatenate(
    view, separators_view, narep, narep, sep_on_null, cudf::test::get_default_stream());
}

TEST_F(StringsCombineTest, Join)
{
  auto input = cudf::test::strings_column_wrapper({"Héllo", "thesé", "tést"});
  auto view  = cudf::strings_column_view(input);

  auto const separator = cudf::string_scalar(" ", true, cudf::test::get_default_stream());
  auto const narep     = cudf::string_scalar("n/a", true, cudf::test::get_default_stream());
  cudf::strings::join_strings(view, separator, narep, cudf::test::get_default_stream());
}

TEST_F(StringsCombineTest, JoinLists)
{
  using STR_LISTS  = cudf::test::lists_column_wrapper<cudf::string_view>;
  auto const input = STR_LISTS{
    STR_LISTS{"a", "bb", "ccc"}, STR_LISTS{"ddd", "efgh", "ijk"}, STR_LISTS{"zzz", "xxxxx"}};
  auto view = cudf::lists_column_view(input);

  auto separators      = cudf::test::strings_column_wrapper({"_", ".", " "});
  auto separators_view = cudf::strings_column_view(separators);
  auto sep_on_null     = cudf::strings::separator_on_nulls::YES;
  auto if_empty        = cudf::strings::output_if_empty_list::EMPTY_STRING;

  auto const separator = cudf::string_scalar(" ", true, cudf::test::get_default_stream());
  auto const narep     = cudf::string_scalar("n/a", true, cudf::test::get_default_stream());
  cudf::strings::join_list_elements(
    view, separator, narep, sep_on_null, if_empty, cudf::test::get_default_stream());
  cudf::strings::join_list_elements(
    view, separators_view, narep, narep, sep_on_null, if_empty, cudf::test::get_default_stream());
}

TEST_F(StringsCombineTest, Repeat)
{
  auto input = cudf::test::strings_column_wrapper({"Héllo", "thesé", "tést"});
  auto view  = cudf::strings_column_view(input);
  cudf::strings::repeat_strings(view, 0, cudf::test::get_default_stream());
  cudf::strings::repeat_strings(view, 1, cudf::test::get_default_stream());
  cudf::strings::repeat_strings(view, 10, cudf::test::get_default_stream());

  auto counts = cudf::test::fixed_width_column_wrapper<cudf::size_type>({9, 8, 7});
  cudf::strings::repeat_strings(view, counts, cudf::test::get_default_stream());
  cudf::strings::repeat_strings(view, counts, cudf::test::get_default_stream());

  auto const str = cudf::string_scalar("X", true, cudf::test::get_default_stream());
  cudf::strings::repeat_string(str, 0, cudf::test::get_default_stream());
  cudf::strings::repeat_string(str, 1, cudf::test::get_default_stream());
  cudf::strings::repeat_string(str, 10, cudf::test::get_default_stream());

  auto const invalid = cudf::string_scalar("", false, cudf::test::get_default_stream());
  cudf::strings::repeat_string(invalid, 10, cudf::test::get_default_stream());
}
