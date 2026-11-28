/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
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

  constexpr cudf::size_type MAX_ENTRY_LENGTH = 24;  // Enough space for "(123) 123-4567" or "n/a"

  auto const num_rows = table.num_rows();
  // allocate scratch space for the outputs
  cuda::device_buffer<char> scratch(
    stream, mr, MAX_ENTRY_LENGTH * static_cast<std::size_t>(num_rows), cuda::no_init);

  auto size = cudf::make_column_from_scalar(
    cudf::numeric_scalar<int32_t>(MAX_ENTRY_LENGTH, true, stream, mr), 1, stream, mr);

  auto country_code              = table.column(2);
  auto area_code                 = table.column(3);
  auto phone_number              = table.column(4);
  auto transformed               = std::vector<int32_t>{2, 3, 4};
  cudf::transform_input inputs[] = {cudf::user_data{scratch.data()},
                                    country_code,
                                    area_code,
                                    phone_number,
                                    cudf::scalar_column_view(*size)};

  auto result = std::move(
    cudf::transform(get_transform_udf("format_phone", true),
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type{cudf::type_id::STRING},
                                                      cudf::output_nullability::PRESERVE}},
                    {},
                    std::nullopt,
                    stream,
                    mr)
      ->release()
      .front());

  return {std::move(result), std::move(transformed)};
}
