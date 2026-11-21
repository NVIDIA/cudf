/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/transform.hpp>

#include <cuda/stream>

#include <array>
#include <utility>

std::tuple<std::unique_ptr<cudf::column>, std::vector<int32_t>> transform(
  cudf::table_view const& table)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto alt = cudf::make_column_from_scalar(
    cudf::string_scalar(cudf::string_view{"(unknown)", 9}, true, stream, mr), 1, stream, mr);

  auto transformed               = std::vector<int32_t>{1};
  auto emails                    = table.column(1);
  cudf::transform_input inputs[] = {emails, cudf::scalar_column_view(*alt)};

  auto providers = std::move(
    cudf::transform(get_transform_udf("email_provider"),
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type{cudf::type_id::STRING},
                                                      cudf::output_nullability::PRESERVE}},
                    {},
                    std::nullopt,
                    stream,
                    mr)
      ->release()
      .front());

  return {std::move(providers), std::move(transformed)};
}
