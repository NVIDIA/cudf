/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/udf_includes.hpp>

#include <cstddef>
#include <cstdint>

extern "C" __device__ cudf::errc checksum(uint16_t* out,
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

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc email_provider(cudf::string_view* out,
                                                cudf::string_view const email,
                                                cudf::string_view const alt)
{
  auto at_pos = email.find('@');

  if (at_pos == cudf::string_view::npos) {
    // malformed email, return alt
    *out = alt;
    return cudf::errc::SUCCESS;
  }

  auto provider_begin = at_pos + 1;

  auto provider = email.substr(provider_begin, email.length() - provider_begin);

  // find the position of '.' in the provider
  auto dot_pos = provider.find('.');

  if (dot_pos == cudf::string_view::npos) {
    // malformed email, return alt
    *out = alt;
  } else {
    // return only the part before the dot
    *out = provider.substr(0, dot_pos);
  }

  return cudf::errc::SUCCESS;
}

// Convert a phone number to E.164 international phone format.
// https://en.wikipedia.org/wiki/E.164
extern "C" __device__ cudf::errc e164_format(cudf::size_type row,
                                             cudf::string_view* out,
                                             void* scratch,
                                             cudf::string_view const country_code,
                                             cudf::string_view const area_code,
                                             cudf::string_view const phone_number,
                                             int64_t age,
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

    cuda::std::memcpy(it, str.data(), size);
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

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc format_phone(cudf::size_type row,
                                              cudf::string_view* out,
                                              void* scratch,
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
    cuda::std::memcpy(it, str.data(), size);
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

  return cudf::errc::SUCCESS;
}
