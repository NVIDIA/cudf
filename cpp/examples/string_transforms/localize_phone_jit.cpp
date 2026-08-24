/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/transform.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <array>
#include <utility>

std::tuple<std::unique_ptr<cudf::column>, std::vector<int32_t>> transform(
  cudf::table_view const& table)
{
  auto stream = rmm::cuda_stream_default;
  auto mr     = cudf::get_current_device_resource_ref();

  auto const udf = R"***(
__device__ void format_phone(void* scratch,
                             cudf::size_type row,
                             cudf::string_view* out,
                             cudf::string_view const country_code,
                             cudf::string_view const area_code,
                             cudf::string_view const phone_number,
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

  auto push_digits = [&](cudf::string_view str, cudf::size_type max) {
    cudf::size_type digits_pushed = 0;
    auto iter                     = str.data();
    auto const end                = str.data() + str.size_bytes();
    while (iter != end && digits_pushed < max) {
      if (*iter != '-') {
        push(cudf::string_view{iter, 1});
        digits_pushed++;
      }
      iter++;
    }

    return iter;
  };

  auto country_iter      = country_code.data();
  auto const country_end = country_iter + country_code.size_bytes();
  auto area_iter         = area_code.data();
  auto const area_end    = area_iter + area_code.size_bytes();
  auto phone_iter        = phone_number.data();
  auto const phone_end   = phone_iter + phone_number.size_bytes();

  // check if it's a US number (country code = 1)
  if (country_code == cudf::string_view{"1", 1}) {
    // push opening parenthesis
    push(cudf::string_view{"(", 1});

    // skip leading zeros in area code and push remaining digits
    while (area_iter != area_end && *area_iter == '0') {
      area_iter++;
    }

    push(cudf::string_view{area_iter, static_cast<cudf::size_type>(area_end - area_iter)});

    // push ") "
    push(cudf::string_view{") ", 2});

    // push first 3 non-dash digits from phone number
    phone_iter = push_digits(
      cudf::string_view{phone_iter, static_cast<cudf::size_type>(phone_end - phone_iter)}, 3);

    // push "-"
    push(cudf::string_view{"-", 1});

    // push remaining 4 non-dash digits
    phone_iter = push_digits(
      cudf::string_view{phone_iter, static_cast<cudf::size_type>(phone_end - phone_iter)}, 4);
  }
  // check if it's a United Kingdom number (country code = 44) or Ireland number (country_code =
  // 353) or New-Zealand (country_code = 64)
  else if (country_code == cudf::string_view{"44", 2} ||
           country_code == cudf::string_view{"353", 3} ||
           country_code == cudf::string_view{"64", 2}) {
    // push area code with leading zeros
    push(area_code);

    // push space
    push(cudf::string_view{" ", 1});

    // count non-dash digits
    cudf::size_type total_digits = 0;

    for (auto iter = phone_iter; iter != phone_end; iter++) {
      if (*iter != '-') { total_digits++; }
    }

    // push digits before the last 4
    phone_iter = push_digits(
      cudf::string_view{phone_iter, static_cast<cudf::size_type>(phone_end - phone_iter)},
      total_digits - 4);

    // push space before last 4 digits
    push(cudf::string_view{" ", 1});

    phone_iter = push_digits(
      cudf::string_view{phone_iter, static_cast<cudf::size_type>(phone_end - phone_iter)}, 4);

  } else {
    push(cudf::string_view{"n/a", 3});
  }

  *out = cudf::string_view{begin, static_cast<cudf::size_type>(it - begin)};
}
  )***";

  constexpr cudf::size_type MAX_ENTRY_LENGTH = 24;  // Enough space for "(123) 123-4567" or "n/a"

  auto const num_rows = table.num_rows();
  rmm::device_uvector<char> scratch(MAX_ENTRY_LENGTH * static_cast<std::size_t>(num_rows),
                                    stream,
                                    mr);  // allocate scratch space for the outputs

  auto size = cudf::make_column_from_scalar(
    cudf::numeric_scalar<int32_t>(MAX_ENTRY_LENGTH, true, stream, mr), 1, stream, mr);

  auto country_code              = table.column(2);
  auto area_code                 = table.column(3);
  auto phone_number              = table.column(4);
  auto transformed               = std::vector<int32_t>{2, 3, 4};
  cudf::transform_input inputs[] = {
    country_code, area_code, phone_number, cudf::scalar_column_view(*size)};

  auto result = std::move(
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

  return {std::move(result), std::move(transformed)};
}
