/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/strip.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

std::tuple<std::unique_ptr<cudf::column>, std::vector<int32_t>> transform(
  cudf::table_view const& table)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto country_code    = table.column(2);
  auto area_code       = table.column(3);
  auto phone_number    = table.column(4);
  auto age             = table.column(5);
  auto transformed     = std::vector<int32_t>{2, 3, 4, 5};
  auto min_visible_age = cudf::numeric_scalar<int32_t>(21, true, stream, mr);

  auto const num_rows = country_code.size();

  // remove all leading zeros from country code and any dashes
  auto stripped_country_code = cudf::strings::strip(country_code,
                                                    cudf::strings::side_type::LEFT,
                                                    cudf::string_scalar("0", true, stream, mr),
                                                    stream,
                                                    mr);

  stripped_country_code = cudf::strings::replace(cudf::strings_column_view(*stripped_country_code),
                                                 cudf::string_scalar("-", true, stream, mr),
                                                 cudf::string_scalar("", true, stream, mr),
                                                 -1,  // maxrepl: replace all occurrences
                                                 stream,
                                                 mr);

  // remove any dashes from phone number
  auto stripped_phone_number = cudf::strings::replace(cudf::strings_column_view(phone_number),
                                                      cudf::string_scalar("-", true, stream, mr),
                                                      cudf::string_scalar("", true, stream, mr),
                                                      -1,  // maxrepl: replace all occurrences
                                                      stream,
                                                      mr);

  // only show the full phone number if the age is above the minimum visible age
  auto should_show = cudf::binary_operation(age,
                                            min_visible_age,
                                            cudf::binary_operator::GREATER_EQUAL,
                                            cudf::data_type{cudf::type_id::BOOL8},
                                            stream,
                                            mr);

  auto num_phone_digits =
    cudf::strings::count_characters(stripped_phone_number->view(), stream, mr);

  auto hidden_char = cudf::string_scalar("*", true, stream, mr);

  auto hidden_char_column = cudf::make_column_from_scalar(hidden_char, num_rows, stream, mr);

  auto hidden_phone_number =
    cudf::strings::repeat_strings(hidden_char_column->view(), num_phone_digits->view(), stream, mr);

  auto redacted_phone_number =
    cudf::copy_if_else(*stripped_phone_number, *hidden_phone_number, *should_show, stream, mr);

  auto country_symbol =
    cudf::make_column_from_scalar(cudf::string_scalar("+", true, stream, mr), num_rows, stream, mr);

  std::vector<cudf::column_view> concat_cols{
    *country_symbol, *stripped_country_code, area_code, *redacted_phone_number};

  auto formatted = cudf::strings::concatenate(cudf::table_view(concat_cols),
                                              cudf::string_scalar("", true, stream, mr),
                                              cudf::string_scalar("", true, stream, mr),
                                              cudf::strings::separator_on_nulls::YES,
                                              stream,
                                              mr);

  return {std::move(formatted), std::move(transformed)};
}
