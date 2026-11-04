/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/parquet/experimental/variant_json_number.cuh"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

namespace pqed = cudf::io::parquet::experimental::detail;

struct classified {
  pqed::scalar_kind kind;
  int width;
  int scale;
  __int128_t unscaled;
};

CUDF_KERNEL void classify_kernel(char const* chars,
                                 int const* offsets,
                                 int num_inputs,
                                 classified* out)
{
  auto const i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(num_inputs)) { return; }
  auto const begin = offsets[i];
  auto const c     = pqed::classify_scalar(chars + begin, offsets[i + 1] - begin);
  out[i]           = {c.kind, c.width, c.scale, c.unscaled};
}

CUDF_KERNEL void parse_double_kernel(char const* chars,
                                     int const* offsets,
                                     int num_inputs,
                                     double* out)
{
  auto const i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(num_inputs)) { return; }
  auto const begin = offsets[i];
  out[i]           = pqed::parse_double(chars + begin, offsets[i + 1] - begin);
}

/// Packs the inputs into one device buffer with offsets
struct device_inputs {
  rmm::device_uvector<char> chars;
  rmm::device_uvector<int> offsets;

  explicit device_inputs(std::vector<std::string> const& inputs)
    : chars(0, cudf::get_default_stream()), offsets(0, cudf::get_default_stream())
  {
    std::string all;
    std::vector<int> offs{0};
    for (auto const& s : inputs) {
      all += s;
      offs.push_back(static_cast<int>(all.size()));
    }
    chars = cudf::detail::make_device_uvector(cudf::host_span<char const>{all.data(), all.size()},
                                              cudf::get_default_stream(),
                                              cudf::get_current_device_resource_ref());
    offsets =
      cudf::detail::make_device_uvector(cudf::host_span<int const>{offs.data(), offs.size()},
                                        cudf::get_default_stream(),
                                        cudf::get_current_device_resource_ref());
  }
};

std::vector<classified> classify(std::vector<std::string> const& inputs)
{
  auto const stream = cudf::get_default_stream();
  device_inputs d{inputs};
  rmm::device_uvector<classified> out(inputs.size(), stream);
  classify_kernel<<<cudf::util::div_rounding_up_safe<int>(inputs.size(), 128),
                    128,
                    0,
                    stream.get()>>>(
    d.chars.data(), d.offsets.data(), static_cast<int>(inputs.size()), out.data());
  auto const host = cudf::detail::make_std_vector(
    cudf::device_span<classified const>{out.data(), out.size()}, stream);
  return {host.begin(), host.end()};
}

std::vector<double> parse_doubles(std::vector<std::string> const& inputs)
{
  auto const stream = cudf::get_default_stream();
  device_inputs d{inputs};
  rmm::device_uvector<double> out(inputs.size(), stream);
  parse_double_kernel<<<cudf::util::div_rounding_up_safe<int>(inputs.size(), 128),
                        128,
                        0,
                        stream.get()>>>(
    d.chars.data(), d.offsets.data(), static_cast<int>(inputs.size()), out.data());
  auto const host =
    cudf::detail::make_std_vector(cudf::device_span<double const>{out.data(), out.size()}, stream);
  return {host.begin(), host.end()};
}

}  // namespace

struct VariantJsonNumberTest : public cudf::test::BaseFixture {};

TEST_F(VariantJsonNumberTest, Literals)
{
  auto const out = classify({"null", "true", "false", "nul", "True", "", "+1", "x"});
  EXPECT_EQ(out[0].kind, pqed::scalar_kind::NULL_VALUE);
  EXPECT_EQ(out[1].kind, pqed::scalar_kind::TRUE_VALUE);
  EXPECT_EQ(out[2].kind, pqed::scalar_kind::FALSE_VALUE);
  for (std::size_t i = 3; i < out.size(); ++i) {
    EXPECT_EQ(out[i].kind, pqed::scalar_kind::INVALID) << "input " << i;
  }
}

TEST_F(VariantJsonNumberTest, IntegerWidthBoundaries)
{
  // Spark uses the narrowest integer type that holds the value
  std::vector<std::string> const inputs{"0",
                                        "127",
                                        "128",
                                        "-128",
                                        "-129",
                                        "32767",
                                        "32768",
                                        "-32768",
                                        "-32769",
                                        "2147483647",
                                        "2147483648",
                                        "-2147483648",
                                        "-2147483649",
                                        "9223372036854775807",
                                        "-9223372036854775808"};
  std::vector<int> const expected_width{1, 1, 2, 1, 2, 2, 4, 2, 4, 4, 8, 4, 8, 8, 8};

  auto const out = classify(inputs);
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    EXPECT_EQ(out[i].kind, pqed::scalar_kind::INT) << inputs[i];
    EXPECT_EQ(out[i].width, expected_width[i]) << inputs[i];
    EXPECT_EQ(static_cast<int64_t>(out[i].unscaled), std::stoll(inputs[i])) << inputs[i];
  }
}

TEST_F(VariantJsonNumberTest, IntegerOverflowBecomesDecimal)
{
  // One past int64 range, so the value is carried as a decimal with scale 0
  auto const out = classify({"9223372036854775808", "-9223372036854775809"});
  EXPECT_EQ(out[0].kind, pqed::scalar_kind::DECIMAL);
  EXPECT_EQ(out[0].scale, 0);
  EXPECT_EQ(out[0].width, 16);
  EXPECT_EQ(out[1].kind, pqed::scalar_kind::DECIMAL);
}

TEST_F(VariantJsonNumberTest, DecimalPrecisionAndScale)
{
  std::vector<std::string> const inputs{
    "1.5",                                     // precision 2, scale 1
    "0.1",                                     // leading zero is not a significant digit
    "0.0000000001",                            // scale 10
    "-2.25",                                   //
    "1.000000000",                             // trailing zeros count toward precision
    std::string("0.") + std::string(38, '1'),  // scale 38, at the limit
    std::string("0.") + std::string(39, '1'),  // scale 39, past the limit
    std::string(38, '9'),                      // precision 38, at the limit
    std::string(39, '9'),                      // precision 39, past the limit
  };
  auto const out = classify(inputs);

  EXPECT_EQ(out[0].kind, pqed::scalar_kind::DECIMAL);
  EXPECT_EQ(out[0].scale, 1);
  EXPECT_EQ(out[0].width, 4);
  EXPECT_EQ(static_cast<int64_t>(out[0].unscaled), 15);

  EXPECT_EQ(out[1].kind, pqed::scalar_kind::DECIMAL);
  EXPECT_EQ(out[1].scale, 1);
  EXPECT_EQ(static_cast<int64_t>(out[1].unscaled), 1);

  EXPECT_EQ(out[2].kind, pqed::scalar_kind::DECIMAL);
  EXPECT_EQ(out[2].scale, 10);
  EXPECT_EQ(out[2].width, 8);

  EXPECT_EQ(out[3].kind, pqed::scalar_kind::DECIMAL);
  EXPECT_EQ(static_cast<int64_t>(out[3].unscaled), -225);

  EXPECT_EQ(out[4].kind, pqed::scalar_kind::DECIMAL);
  EXPECT_EQ(out[4].scale, 9);

  EXPECT_EQ(out[5].kind, pqed::scalar_kind::DECIMAL);
  EXPECT_EQ(out[5].scale, 38);
  EXPECT_EQ(out[5].width, 16);

  // Past either limit the value becomes a double instead
  EXPECT_EQ(out[6].kind, pqed::scalar_kind::DOUBLE);
  EXPECT_EQ(out[7].kind, pqed::scalar_kind::DECIMAL);
  EXPECT_EQ(out[8].kind, pqed::scalar_kind::DOUBLE);
}

TEST_F(VariantJsonNumberTest, ExponentAlwaysBecomesDouble)
{
  // Spark only produces decimals for numbers written without an exponent
  auto const out = classify({"1e2", "1E2", "1.5e0", "0e0", "100e-2"});
  for (std::size_t i = 0; i < out.size(); ++i) {
    EXPECT_EQ(out[i].kind, pqed::scalar_kind::DOUBLE) << "input " << i;
    EXPECT_EQ(out[i].width, 8);
  }
}

TEST_F(VariantJsonNumberTest, ParseDoubleMatchesStrtod)
{
  // `parse_double` implements Java's `Double.parseDouble`, which is correctly rounded, so the
  // host's `strtod` is an independent oracle for every input that is also valid C.
  std::vector<std::string> inputs{"0e0",
                                  "1e2",
                                  "-1e2",
                                  "1.5e0",
                                  "3.141592653589793",
                                  "2.2250738585072011e-308",  // the PHP rounding bug input
                                  "2.2250738585072014e-308",  // smallest normal
                                  "4.9406564584124654e-324",  // smallest subnormal
                                  "1.7976931348623157e308",   // largest finite
                                  "1.7976931348623159e308",   // overflows to infinity
                                  "1e-400",                   // underflows to zero
                                  "1e400",
                                  "0.1",
                                  "1e22",
                                  "1e23",              // just past the exact fast path
                                  "9007199254740993",  // 2^53 + 1, needs round half to even
                                  "9007199254740995",
                                  "1.0000000000000002",
                                  "123456789012345678901234567890e-29",
                                  "0.3333333333333333333333333333333333333333"};
  // A long run of digits exercises the sticky-bit path past 800 significant digits
  inputs.push_back("1." + std::string(900, '0') + "1");
  inputs.push_back(std::string(850, '9'));

  auto const out = parse_doubles(inputs);
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    auto const expected = std::strtod(inputs[i].c_str(), nullptr);
    EXPECT_EQ(out[i], expected) << inputs[i];
  }
}

TEST_F(VariantJsonNumberTest, ParseDoubleSignedZero)
{
  auto const out = parse_doubles({"-0e0", "0e0", "-0.0", "-1e-400"});
  EXPECT_TRUE(std::signbit(out[0]));
  EXPECT_EQ(out[0], 0.0);
  EXPECT_FALSE(std::signbit(out[1]));
  EXPECT_TRUE(std::signbit(out[2]));
  EXPECT_TRUE(std::signbit(out[3]));  // underflow keeps the sign
}

TEST_F(VariantJsonNumberTest, EncodedSizes)
{
  auto const out = classify({"1", "100000", "1.5", "1e2", "null"});
  EXPECT_EQ(pqed::scalar_encoded_size({out[0].kind, out[0].width, out[0].scale, out[0].unscaled}),
            2);  // header + 1 payload byte
  EXPECT_EQ(pqed::scalar_encoded_size({out[1].kind, out[1].width, out[1].scale, out[1].unscaled}),
            5);
  EXPECT_EQ(pqed::scalar_encoded_size({out[2].kind, out[2].width, out[2].scale, out[2].unscaled}),
            6);  // header + scale + 4 payload bytes
  EXPECT_EQ(pqed::scalar_encoded_size({out[3].kind, out[3].width, out[3].scale, out[3].unscaled}),
            9);
  EXPECT_EQ(pqed::scalar_encoded_size({out[4].kind, out[4].width, out[4].scale, out[4].unscaled}),
            1);
}

CUDF_TEST_PROGRAM_MAIN()
