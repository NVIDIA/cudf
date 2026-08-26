/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/lists_column_wrapper.hpp>

#include <cudf/strings/convert/convert_booleans.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/convert/convert_durations.hpp>
#include <cudf/strings/convert/convert_fixed_point.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/convert/convert_ipv4.hpp>
#include <cudf/strings/convert/convert_lists.hpp>
#include <cudf/strings/convert/convert_urls.hpp>
#include <cudf/strings/convert/int_cast.hpp>

#include <string>

class StringsConvertTest : public cudf::test::BaseFixture {};

TEST_F(StringsConvertTest, Booleans)
{
  auto input = cudf::test::strings_column_wrapper({"true", "false", "True", ""});
  auto view  = cudf::strings_column_view(input);

  auto true_scalar  = cudf::string_scalar("true", true, cudf::test::get_default_stream());
  auto false_scalar = cudf::string_scalar("false", true, cudf::test::get_default_stream());

  auto bools = cudf::strings::to_booleans(view, true_scalar, cudf::test::get_default_stream());
  cudf::strings::from_booleans(
    bools->view(), true_scalar, false_scalar, cudf::test::get_default_stream());
}

TEST_F(StringsConvertTest, Timestamps)
{
  auto input = cudf::test::strings_column_wrapper({"2019-03-20T12:34:56Z", "2020-02-29T00:00:00Z"});
  auto view  = cudf::strings_column_view(input);

  std::string format = "%Y-%m-%dT%H:%M:%SZ";
  auto dtype         = cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS};

  cudf::strings::is_timestamp(view, format, cudf::test::get_default_stream());
  auto timestamps =
    cudf::strings::to_timestamps(view, dtype, format, cudf::test::get_default_stream());

  auto empty = cudf::test::strings_column_wrapper();
  cudf::strings::from_timestamps(
    timestamps->view(), format, cudf::strings_column_view(empty), cudf::test::get_default_stream());
}

TEST_F(StringsConvertTest, Durations)
{
  auto input = cudf::test::strings_column_wrapper({"17975 days 12:34:56", "18321 days 00:00:00"});
  auto view  = cudf::strings_column_view(input);

  std::string format = "%D days %H:%M:%S";
  auto dtype         = cudf::data_type{cudf::type_id::DURATION_SECONDS};

  auto durations =
    cudf::strings::to_durations(view, dtype, format, cudf::test::get_default_stream());
  cudf::strings::from_durations(durations->view(), format, cudf::test::get_default_stream());
}

TEST_F(StringsConvertTest, FixedPoint)
{
  auto input = cudf::test::strings_column_wrapper({"1.234E3", "-876", "543.2"});
  auto view  = cudf::strings_column_view(input);

  auto dtype = cudf::data_type{cudf::type_id::DECIMAL64, numeric::scale_type{-3}};

  auto values = cudf::strings::to_fixed_point(view, dtype, cudf::test::get_default_stream());
  cudf::strings::from_fixed_point(values->view(), cudf::test::get_default_stream());
}

TEST_F(StringsConvertTest, Floats)
{
  auto input = cudf::test::strings_column_wrapper({"1.234E3", "-876", "543.2"});
  auto view  = cudf::strings_column_view(input);

  auto dtype = cudf::data_type{cudf::type_id::FLOAT32};

  auto values = cudf::strings::to_floats(view, dtype, cudf::test::get_default_stream());
  cudf::strings::from_floats(values->view(), cudf::test::get_default_stream());
  cudf::strings::is_float(view, cudf::test::get_default_stream());
}

TEST_F(StringsConvertTest, Integers)
{
  auto input = cudf::test::strings_column_wrapper({"1234", "-876", "5432"});
  auto view  = cudf::strings_column_view(input);

  auto dtype = cudf::data_type{cudf::type_id::INT32};

  auto values = cudf::strings::to_integers(view, dtype, cudf::test::get_default_stream());
  cudf::strings::from_integers(values->view(), cudf::test::get_default_stream());
  cudf::strings::is_integer(view, cudf::test::get_default_stream());
  cudf::strings::is_hex(view, cudf::test::get_default_stream());
  cudf::strings::hex_to_integers(view, dtype, cudf::test::get_default_stream());
  cudf::strings::integers_to_hex(values->view(), cudf::test::get_default_stream());
}

TEST_F(StringsConvertTest, IntegerCast)
{
  auto input    = cudf::test::strings_column_wrapper({"aaa", "bbb", "c", "d", "", "f"});
  auto view     = cudf::strings_column_view(input);
  auto stream   = cudf::test::get_default_stream();
  auto otype    = cudf::strings::integer_cast_type(view, stream);
  auto swap     = cudf::strings::endian::LITTLE;
  auto integers = cudf::strings::cast_to_integer(view, otype.value(), swap, stream);
  cudf::strings::cast_from_integer(integers->view(), swap, stream);
}

TEST_F(StringsConvertTest, IPv4)
{
  auto input = cudf::test::strings_column_wrapper({"192.168.0.1", "10.0.0.1"});
  auto view  = cudf::strings_column_view(input);

  auto values = cudf::strings::ipv4_to_integers(view, cudf::test::get_default_stream());
  cudf::strings::integers_to_ipv4(values->view(), cudf::test::get_default_stream());
  cudf::strings::is_ipv4(view, cudf::test::get_default_stream());
}

TEST_F(StringsConvertTest, URLs)
{
  auto input = cudf::test::strings_column_wrapper({"www.nvidia.com/rapids?p=é", "/_file-7.txt"});
  auto view  = cudf::strings_column_view(input);

  auto values = cudf::strings::url_encode(view, cudf::test::get_default_stream());
  cudf::strings::url_decode(values->view(), cudf::test::get_default_stream());
}

TEST_F(StringsConvertTest, ListsFormat)
{
  using STR_LISTS = cudf::test::lists_column_wrapper<cudf::string_view>;
  auto const input =
    STR_LISTS{{STR_LISTS{"a", "bb", "ccc"}, STR_LISTS{}, STR_LISTS{"ddd", "ee", "f"}},
              {STR_LISTS{"gg", "hhh"}, STR_LISTS{"i", "", "", "jj"}}};
  auto view        = cudf::lists_column_view(input);
  auto null_scalar = cudf::string_scalar("NULL", true, cudf::test::get_default_stream());
  auto separators  = cudf::strings_column_view(cudf::test::strings_column_wrapper());
  cudf::strings::format_list_column(
    view, null_scalar, separators, cudf::test::get_default_stream());
}
