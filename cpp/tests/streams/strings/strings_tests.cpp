/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/strings/padding.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/strings/wrap.hpp>

#include <optional>
#include <string>

class StringsTest : public cudf::test::BaseFixture {};

TEST_F(StringsTest, Strip)
{
  auto input = cudf::test::strings_column_wrapper({"  aBc  ", "   ", "aaaa ", "\tb"});
  auto view  = cudf::strings_column_view(input);

  auto const strip = cudf::string_scalar(" ", true, cudf::test::get_default_stream());
  auto const side  = cudf::strings::side_type::BOTH;
  cudf::strings::strip(view, side, strip, cudf::test::get_default_stream());
}

TEST_F(StringsTest, Pad)
{
  auto input = cudf::test::strings_column_wrapper({"333", "", "4444", "1"});
  auto view  = cudf::strings_column_view(input);

  auto const side = cudf::strings::side_type::BOTH;
  cudf::strings::pad(view, 6, side, " ", cudf::test::get_default_stream());
}

TEST_F(StringsTest, Zfill)
{
  auto input = cudf::test::strings_column_wrapper({"333", "", "4444", "1"});
  auto view  = cudf::strings_column_view(input);

  cudf::strings::zfill(view, 6, cudf::test::get_default_stream());

  auto widths = cudf::test::fixed_width_column_wrapper<cudf::size_type>({6, 7, 8, 8});
  cudf::strings::zfill_by_widths(view, widths, cudf::test::get_default_stream());
}

TEST_F(StringsTest, Wrap)
{
  auto input = cudf::test::strings_column_wrapper({"the quick brown fox jumped"});
  auto view  = cudf::strings_column_view(input);

  cudf::strings::wrap(view, 6, cudf::test::get_default_stream());
}

TEST_F(StringsTest, Slice)
{
  auto input = cudf::test::strings_column_wrapper({"hello", "these", "are test strings"});
  auto view  = cudf::strings_column_view(input);

  auto start = std::optional<cudf::size_type>(2);
  auto stop  = std::optional<cudf::size_type>(5);
  auto step  = std::optional<cudf::size_type>(1);
  cudf::strings::slice_strings(view, start, stop, step, cudf::test::get_default_stream());

  auto starts = cudf::test::fixed_width_column_wrapper<cudf::size_type>({1, 2, 3});
  auto stops  = cudf::test::fixed_width_column_wrapper<cudf::size_type>({4, 5, 6});
  cudf::strings::slice_strings(view, starts, stops, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
