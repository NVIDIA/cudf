/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/strings/extract.hpp>
#include <cudf/strings/regex/regex_program.hpp>

#include <string>

class StringsExtractTest : public cudf::test::BaseFixture {};

TEST_F(StringsExtractTest, Extract)
{
  auto input = cudf::test::strings_column_wrapper({"Joe Schmoe", "John Smith", "Jane Smith"});
  auto view  = cudf::strings_column_view(input);

  auto const pattern = std::string("([A-Z][a-z]+)");
  auto const prog    = cudf::strings::regex_program::create(pattern);
  cudf::strings::extract(view, *prog, cudf::test::get_default_stream());
  cudf::strings::extract_all_record(view, *prog, cudf::test::get_default_stream());
  cudf::strings::extract_single(view, *prog, 0, cudf::test::get_default_stream());
}
