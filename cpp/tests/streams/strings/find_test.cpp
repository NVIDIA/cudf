/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/strings/find.hpp>
#include <cudf/strings/find_multiple.hpp>
#include <cudf/strings/findall.hpp>
#include <cudf/strings/regex/regex_program.hpp>

#include <string>

class StringsFindTest : public cudf::test::BaseFixture {};

TEST_F(StringsFindTest, Find)
{
  auto input = cudf::test::strings_column_wrapper({"Héllo", "thesé", "tést strings", ""});
  auto view  = cudf::strings_column_view(input);

  auto const target = cudf::string_scalar("é", true, cudf::test::get_default_stream());
  cudf::strings::find(view, target, 0, -1, cudf::test::get_default_stream());
  cudf::strings::rfind(view, target, 0, -1, cudf::test::get_default_stream());
  cudf::strings::find(view, view, 0, cudf::test::get_default_stream());
  cudf::strings::find_multiple(view, view, cudf::test::get_default_stream());
  cudf::strings::contains(view, target, cudf::test::get_default_stream());
  cudf::strings::starts_with(view, target, cudf::test::get_default_stream());
  cudf::strings::starts_with(view, view, cudf::test::get_default_stream());
  cudf::strings::ends_with(view, target, cudf::test::get_default_stream());
  cudf::strings::ends_with(view, view, cudf::test::get_default_stream());
  cudf::strings::find_instance(view, target, 0, cudf::test::get_default_stream());
  cudf::strings::count(view, target, cudf::test::get_default_stream());

  auto const pattern = std::string("[a-z]");
  auto const prog    = cudf::strings::regex_program::create(pattern);
  cudf::strings::findall(view, *prog, cudf::test::get_default_stream());
  cudf::strings::find_re(view, *prog, cudf::test::get_default_stream());
}
