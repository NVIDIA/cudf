/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

std::tuple<std::unique_ptr<cudf::column>, std::vector<int32_t>> transform(
  cudf::table_view const& table)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto country_code   = table.column(2);
  auto area_code      = table.column(3);
  auto phone_number   = table.column(4);
  auto transformed    = std::vector<int32_t>{2, 3, 4};
  auto const num_rows = country_code.size();

  // create "n/a" column for unrecognised numbers
  auto na_column = cudf::make_column_from_scalar(
    cudf::string_scalar("n/a", true, stream, mr), num_rows, stream, mr);

  auto left_paren =
    cudf::make_column_from_scalar(cudf::string_scalar("(", true, stream, mr), num_rows, stream, mr);

  auto right_paren_space = cudf::make_column_from_scalar(
    cudf::string_scalar(") ", true, stream, mr), num_rows, stream, mr);

  auto space =
    cudf::make_column_from_scalar(cudf::string_scalar(" ", true, stream, mr), num_rows, stream, mr);

  // create boolean mask for US numbers (country_code == "1")
  auto us_scalar = cudf::string_scalar("1", true, stream, mr);
  auto us_mask   = cudf::binary_operation(country_code,
                                        us_scalar,
                                        cudf::binary_operator::EQUAL,
                                        cudf::data_type{cudf::type_id::BOOL8},
                                        stream,
                                        mr);

  // create boolean mask for UK numbers (country_code == "44")
  auto uk_scalar = cudf::string_scalar("44", true, stream, mr);
  auto uk_mask   = cudf::binary_operation(country_code,
                                        uk_scalar,
                                        cudf::binary_operator::EQUAL,
                                        cudf::data_type{cudf::type_id::BOOL8},
                                        stream,
                                        mr);

  // create boolean mask for Ireland numbers (country_code == "353")
  auto ie_scalar = cudf::string_scalar("353", true, stream, mr);
  auto ie_mask   = cudf::binary_operation(country_code,
                                        ie_scalar,
                                        cudf::binary_operator::EQUAL,
                                        cudf::data_type{cudf::type_id::BOOL8},
                                        stream,
                                        mr);

  // create boolean mask for New-Zealand numbers (country_code == "64")
  auto nz_scalar = cudf::string_scalar("64", true, stream, mr);
  auto nz_mask   = cudf::binary_operation(country_code,
                                        nz_scalar,
                                        cudf::binary_operator::EQUAL,
                                        cudf::data_type{cudf::type_id::BOOL8},
                                        stream,
                                        mr);

  // <------- create formatted US numbers: "(area_code) phone" ----------->

  // concatenate "(area_code) phone" for US format
  std::vector<cudf::column_view> us_format_parts{
    *left_paren, area_code, *right_paren_space, phone_number};

  auto us_formatted = cudf::strings::concatenate(cudf::table_view(us_format_parts),
                                                 cudf::string_scalar("", true, stream, mr),
                                                 cudf::string_scalar("", true, stream, mr),
                                                 cudf::strings::separator_on_nulls::YES,
                                                 stream,
                                                 mr);

  // <------- create formatted UK numbers: "area_code phone" ----------->

  // replace dashes with spaces in phone numbers for UK format
  auto uk_phone_formatted = cudf::strings::replace(phone_number,
                                                   cudf::string_scalar("-", true, stream, mr),
                                                   cudf::string_scalar(" ", true, stream, mr),
                                                   -1,
                                                   stream,
                                                   mr);

  // concatenate "area_code phone" for UK format
  std::vector<cudf::column_view> uk_format_parts{area_code, *space, *uk_phone_formatted};

  auto uk_formatted = cudf::strings::concatenate(cudf::table_view(uk_format_parts),
                                                 cudf::string_scalar("", true, stream, mr),
                                                 cudf::string_scalar("", true, stream, mr),
                                                 cudf::strings::separator_on_nulls::YES,
                                                 stream,
                                                 mr);

  auto uk_or_ie_mask = cudf::binary_operation(*uk_mask,
                                              *ie_mask,
                                              cudf::binary_operator::LOGICAL_OR,
                                              cudf::data_type{cudf::type_id::BOOL8},
                                              stream,
                                              mr);

  // UK, Ireland, and New-Zealand share similar local phone number formats
  auto uk_ie_nz_mask = cudf::binary_operation(*uk_or_ie_mask,
                                              *nz_mask,
                                              cudf::binary_operator::LOGICAL_OR,
                                              cudf::data_type{cudf::type_id::BOOL8},
                                              stream,
                                              mr);

  // first, select between UK format and n/a based on UK mask
  auto uk_or_na_or_ca = cudf::copy_if_else(*uk_formatted, *na_column, *uk_ie_nz_mask, stream, mr);

  // then, select between US format and the result of UK/n/a based on US mask
  return std::make_tuple(cudf::copy_if_else(*us_formatted, *uk_or_na_or_ca, *us_mask, stream, mr),
                         transformed);
}
