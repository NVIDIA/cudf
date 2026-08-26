/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/replace.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

#include <cuda/stream>

std::tuple<std::unique_ptr<cudf::column>, std::vector<int32_t>> transform(
  cudf::table_view const& table)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto emails      = table.column(1);
  auto transformed = std::vector<int32_t>{1};

  auto const index_max =
    cudf::numeric_scalar<cudf::size_type>{std::numeric_limits<cudf::size_type>::max(), stream, mr};
  auto const npos = cudf::numeric_scalar<cudf::size_type>{-1, stream, mr};
  auto const one  = cudf::numeric_scalar<cudf::size_type>{1, stream, mr};
  auto const alt  = cudf::string_scalar(cudf::string_view{"(unknown)", 9}, true, stream, mr);

  // find the positions of the "@" in the email, -1 is returned if not found
  auto at_positions =
    cudf::strings::find(emails, cudf::string_scalar("@", true, stream, mr), 0, -1, stream, mr);

  // find the positions of the "." in the email, -1 is returned if not found. Note that we start the
  // search from the end of the string
  auto dot_positions =
    cudf::strings::rfind(emails, cudf::string_scalar(".", true, stream, mr), 0, -1, stream, mr);

  // if "@" was found
  auto is_at_found = cudf::binary_operation(*at_positions,
                                            npos,
                                            cudf::binary_operator::NOT_EQUAL,
                                            cudf::data_type{cudf::type_id::BOOL8},
                                            stream,
                                            mr);

  // if "." was found
  auto is_dot_found = cudf::binary_operation(*dot_positions,
                                             npos,
                                             cudf::binary_operator::NOT_EQUAL,
                                             cudf::data_type{cudf::type_id::BOOL8},
                                             stream,
                                             mr);

  // if the "." is after "@"
  auto is_dot_after_at = cudf::binary_operation(*at_positions,
                                                *dot_positions,
                                                cudf::binary_operator::LESS,
                                                cudf::data_type{cudf::type_id::BOOL8},
                                                stream,
                                                mr);

  auto dot_and_at_found = cudf::binary_operation(*is_at_found,
                                                 *is_dot_found,
                                                 cudf::binary_operator::LOGICAL_AND,
                                                 cudf::data_type{cudf::type_id::BOOL8},
                                                 stream,
                                                 mr);

  // if the sequence is valid
  auto is_valid = cudf::binary_operation(*is_dot_after_at,
                                         *dot_and_at_found,
                                         cudf::binary_operator::LOGICAL_AND,
                                         cudf::data_type{cudf::type_id::BOOL8},
                                         stream,
                                         mr);

  // filter out the emails whose provider can not be determined
  auto provider_begin = cudf::binary_operation(*at_positions,
                                               one,
                                               cudf::binary_operator::ADD,
                                               cudf::data_type{cudf::type_id::INT32},
                                               stream,
                                               mr);
  provider_begin      = cudf::copy_if_else(*provider_begin, index_max, *is_valid, stream, mr);
  auto provider_end   = cudf::copy_if_else(*dot_positions, npos, *is_valid, stream, mr);

  auto providers = cudf::strings::slice_strings(emails, *provider_begin, *provider_end, stream, mr);

  providers = cudf::copy_if_else(*providers, alt, *is_valid, stream, mr);

  return std::make_tuple(std::move(providers), transformed);
}
