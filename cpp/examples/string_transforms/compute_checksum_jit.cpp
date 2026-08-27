/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.hpp"

#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

#include <cuda/stream>

#include <array>
#include <utility>

std::tuple<std::unique_ptr<cudf::column>, std::vector<int32_t>> transform(
  cudf::table_view const& table)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto udf = R"***(
 __device__ void checksum(uint16_t* out,
                          cudf::string_view const name,
                          cudf::string_view const email)
 {
   auto fletcher16 = [](cudf::string_view str) -> uint16_t {
     uint16_t sum1 = 0;
     uint16_t sum2 = 0;
     for (cudf::size_type i = 0; i < str.size_bytes(); ++i) {
       sum1 = (sum1 + str.data()[i]) % 255;
       sum2 = (sum2 + sum1) % 255;
     }
     return (sum2 << 8) | sum1;
   };
   *out = fletcher16(name) ^ fletcher16(email);
 }
   )***";

  auto transformed               = std::vector<int32_t>{0, 1};
  auto name                      = table.column(0);
  auto email                     = table.column(1);
  cudf::transform_input inputs[] = {name, email};

  auto result = std::move(
    cudf::transform(udf,
                    cudf::udf_source_type::CUDA,
                    cudf::null_aware::NO,
                    std::nullopt,
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type{cudf::type_id::UINT16},
                                                      cudf::output_nullability::PRESERVE}},
                    {},
                    std::nullopt,
                    stream,
                    mr)
      ->release()
      .front());

  return std::make_tuple(std::move(result), transformed);
}
