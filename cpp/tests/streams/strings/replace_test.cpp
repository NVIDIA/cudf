/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/replace_re.hpp>

#include <string>

class StringsReplaceTest : public cudf::test::BaseFixture {};

TEST_F(StringsReplaceTest, Replace)
{
  auto input = cudf::test::strings_column_wrapper({"Héllo", "thesé", "tést strings", ""});
  auto view  = cudf::strings_column_view(input);

  auto const target = cudf::string_scalar("é", true, cudf::test::get_default_stream());
  auto const repl   = cudf::string_scalar(" ", true, cudf::test::get_default_stream());
  cudf::strings::replace(view, target, repl, -1, cudf::test::get_default_stream());
  cudf::strings::replace_multiple(view, view, view, cudf::test::get_default_stream());
  cudf::strings::replace_slice(view, repl, 1, 2, cudf::test::get_default_stream());
}

TEST_F(StringsReplaceTest, ReplaceRegex)
{
  auto input = cudf::test::strings_column_wrapper({"Héllo", "thesé", "tést strings", ""});
  auto view  = cudf::strings_column_view(input);

  auto const repl    = cudf::string_scalar(" ", true, cudf::test::get_default_stream());
  auto const pattern = std::string("[a-z]");
  auto const prog    = cudf::strings::regex_program::create(pattern);
  cudf::strings::replace_re(view, *prog, repl, 1, cudf::test::get_default_stream());
}

TEST_F(StringsReplaceTest, ReplaceRegexBackref)
{
  auto input = cudf::test::strings_column_wrapper({"Héllo thesé", "tést strings"});
  auto view  = cudf::strings_column_view(input);

  auto const repl_template = std::string("\\2-\\1");
  auto const pattern       = std::string("(\\w) (\\w)");
  auto const prog          = cudf::strings::regex_program::create(pattern);
  cudf::strings::replace_with_backrefs(
    view, *prog, repl_template, cudf::test::get_default_stream());
}
