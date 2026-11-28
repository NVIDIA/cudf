/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/transform.hpp>

#include <cuda/buffer>
#include <cuda/stream>

#include <array>
#include <utility>

std::tuple<std::unique_ptr<cudf::column>, std::vector<int32_t>> transform(
  cudf::table_view const& table)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  constexpr cudf::size_type maximum_size = 20;
  auto const num_rows                    = table.num_rows();

  cuda::device_buffer<char> scratch(
    stream, mr, maximum_size * static_cast<std::size_t>(num_rows), cuda::no_init);

  auto size = cudf::make_column_from_scalar(
    cudf::numeric_scalar<int32_t>(maximum_size, true, stream, mr), 1, stream, mr);

  auto country_code    = table.column(2);
  auto area_code       = table.column(3);
  auto phone_code      = table.column(4);
  auto age             = table.column(5);
  auto transformed     = std::vector<int32_t>{2, 3, 4, 5};
  auto min_visible_age = cudf::make_column_from_scalar(
    cudf::numeric_scalar<int32_t>(21, true, stream, mr), 1, stream, mr);
  cudf::transform_input inputs[] = {cudf::user_data{scratch.data()},
                                    country_code,
                                    area_code,
                                    phone_code,
                                    age,
                                    cudf::scalar_column_view(*min_visible_age),
                                    cudf::scalar_column_view(*size)};

  auto formatted = std::move(
    cudf::transform(get_transform_udf("e164_format", true),
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type{cudf::type_id::STRING},
                                                      cudf::output_nullability::PRESERVE}},
                    {},
                    std::nullopt,
                    stream,
                    mr)
      ->release()
      .front());

  return std::make_tuple(std::move(formatted), transformed);
}
