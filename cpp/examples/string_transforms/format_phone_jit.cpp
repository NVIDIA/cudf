/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/transform.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <array>
#include <utility>

std::tuple<std::unique_ptr<cudf::column>, std::vector<int32_t>> transform(
  cudf::table_view const& table)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  /// Convert a phone number to E.164 international phone format
  /// (https://en.wikipedia.org/wiki/E.164)
  auto udf = R"***(
__device__ void e164_format(void* scratch,
                            cudf::size_type row,
                            cudf::string_view* out,
                            cudf::string_view const country_code,
                            cudf::string_view const area_code,
                            cudf::string_view const phone_number,
                            int32_t age,
                            int32_t min_visible_age,
                            int32_t scratch_size)
{
  auto const begin = static_cast<char*>(scratch) +
                     static_cast<ptrdiff_t>(row) * static_cast<ptrdiff_t>(scratch_size);
  auto const end = begin + scratch_size;
  auto it        = begin;

  auto push = [&](cudf::string_view str) {
    auto const size = str.size_bytes();

    if ((it + size) > end) { return; }

    memcpy(it, str.data(), size);
    it += size;
  };

  auto push_digits = [&](cudf::string_view str) {
    auto iter      = str.data();
    auto const end = str.data() + str.size_bytes();
    while (iter != end) {
      if (*iter != '-') { push(cudf::string_view{iter, 1}); }
      iter++;
    }

    return iter;
  };

  auto push_hidden_digits = [&](cudf::string_view str) {
    auto iter      = str.data();
    auto const end = str.data() + str.size_bytes();
    while (iter != end) {
      if (*iter != '-') { push(cudf::string_view{"*", 1}); }
      iter++;
    }

    return iter;
  };

  auto country_iter      = country_code.data();
  auto const country_end = country_iter + country_code.size_bytes();
  auto phone_iter        = phone_number.data();
  auto const phone_end   = phone_iter + phone_number.size_bytes();
  auto const should_hide = age < min_visible_age;

  push(cudf::string_view{"+", 1});

  // skip leading zeros in country code and push non-dash digits
  while (country_iter != country_end && *country_iter == '0') {
    country_iter++;
  }

  push_digits(
    cudf::string_view{country_iter, static_cast<cudf::size_type>(country_end - country_iter)});

  push(area_code);

  // push non-dash digits from phone number if the age is above the minimum visible age

  if (should_hide) {
    push_hidden_digits(
      cudf::string_view{phone_iter, static_cast<cudf::size_type>(phone_end - phone_iter)});
  } else {
    push_digits(
      cudf::string_view{phone_iter, static_cast<cudf::size_type>(phone_end - phone_iter)});
  }

  *out = cudf::string_view{begin, static_cast<cudf::size_type>(it - static_cast<char*>(begin))};
}
  )***";

  constexpr cudf::size_type maximum_size = 20;
  auto const num_rows                    = table.num_rows();

  rmm::device_uvector<char> scratch(maximum_size * static_cast<std::size_t>(num_rows), stream, mr);

  auto size = cudf::make_column_from_scalar(
    cudf::numeric_scalar<int32_t>(maximum_size, true, stream, mr), 1, stream, mr);

  auto country_code    = table.column(2);
  auto area_code       = table.column(3);
  auto phone_code      = table.column(4);
  auto age             = table.column(5);
  auto transformed     = std::vector<int32_t>{2, 3, 4, 5};
  auto min_visible_age = cudf::make_column_from_scalar(
    cudf::numeric_scalar<int32_t>(21, true, stream, mr), 1, stream, mr);
  cudf::transform_input inputs[] = {country_code,
                                    area_code,
                                    phone_code,
                                    age,
                                    cudf::scalar_column_view(*min_visible_age),
                                    cudf::scalar_column_view(*size)};

  auto formatted = std::move(
    cudf::transform(udf,
                    cudf::udf_source_type::CUDA,
                    cudf::null_aware::NO,
                    scratch.data(),
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
