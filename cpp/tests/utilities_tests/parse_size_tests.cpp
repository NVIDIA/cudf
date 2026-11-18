/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/getenv_or.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

TEST(ParseSizeTest, Bytes)
{
  EXPECT_EQ(cudf::detail::parse_size("1", "size", true), std::uint64_t{1});
  EXPECT_EQ(cudf::detail::parse_size("256", "size", true), std::uint64_t{256});
  EXPECT_EQ(cudf::detail::parse_size("256B", "size", true), std::uint64_t{256});
  EXPECT_EQ(cudf::detail::parse_size("000256", "size", true), std::uint64_t{256});
}

TEST(ParseSizeTest, DecimalUnits)
{
  EXPECT_EQ(cudf::detail::parse_size("1kB", "size", true), std::uint64_t{1'000});
  EXPECT_EQ(cudf::detail::parse_size("1KB", "size", true), std::uint64_t{1'000});
  EXPECT_EQ(cudf::detail::parse_size("256MB", "size", true), std::uint64_t{256'000'000});
  EXPECT_EQ(cudf::detail::parse_size("1GB", "size", true), std::uint64_t{1'000'000'000});
  EXPECT_EQ(cudf::detail::parse_size("1TB", "size", true), std::uint64_t{1'000'000'000'000});
  EXPECT_EQ(cudf::detail::parse_size("1PB", "size", true), std::uint64_t{1'000'000'000'000'000});
  EXPECT_EQ(cudf::detail::parse_size("1EB", "size", true),
            std::uint64_t{1'000'000'000'000'000'000});
}

TEST(ParseSizeTest, BinaryUnits)
{
  EXPECT_EQ(cudf::detail::parse_size("1KiB", "size", true), std::uint64_t{1} << 10);
  EXPECT_EQ(cudf::detail::parse_size("256MiB", "size", true), std::uint64_t{268'435'456});
  EXPECT_EQ(cudf::detail::parse_size("9GiB", "size", true), std::uint64_t{9'663'676'416});
  EXPECT_EQ(cudf::detail::parse_size("1TiB", "size", true), std::uint64_t{1} << 40);
  EXPECT_EQ(cudf::detail::parse_size("1PiB", "size", true), std::uint64_t{1} << 50);
  EXPECT_EQ(cudf::detail::parse_size("1EiB", "size", true), std::uint64_t{1} << 60);
  EXPECT_EQ(cudf::detail::parse_size("000512MiB", "size", true), std::uint64_t{536'870'912});
}

TEST(ParseSizeTest, NonNullTerminatedInput)
{
  constexpr char input[] = {'5', '1', '2', 'M', 'i', 'B', 'x'};
  EXPECT_EQ(cudf::detail::parse_size(std::string_view{input, 6}, "size", true),
            std::uint64_t{536'870'912});
}

TEST(ParseSizeTest, InvalidInput)
{
  std::string_view const invalid_inputs[] = {
    {},        "",          " ",       "\t",           "-1",   "+1",    " 512",
    "512 ",    "512\n",     "512 MiB", "1.5MB",        "1e3",  "0x200", "invalid",
    "MB",      "B",         "512mb",   "512Mb",        "512M", "512m",  "512Mi",
    "512MiBB", "512MBjunk", "1XB",     {"512MiB\0", 7}};
  for (auto const input : invalid_inputs) {
    SCOPED_TRACE(input);
    EXPECT_THROW(cudf::detail::parse_size(input, "size", true), std::invalid_argument);
  }
}

TEST(ParseSizeTest, ZeroSizePolicy)
{
  for (auto const* input : {"0", "000", "0B", "0MB", "0MiB"}) {
    SCOPED_TRACE(input);
    EXPECT_THROW(cudf::detail::parse_size(input, "size", true), std::invalid_argument);
    EXPECT_EQ(cudf::detail::parse_size(input, "size", false), std::uint64_t{0});
  }
}

TEST(ParseSizeTest, IntegerOverflow)
{
  auto const input = std::to_string(std::numeric_limits<std::uint64_t>::max()) + "0";
  EXPECT_THROW(cudf::detail::parse_size(input, "size", true), std::invalid_argument);
  EXPECT_THROW(cudf::detail::parse_size(input + "MiB", "size", false), std::invalid_argument);
}

TEST(ParseSizeTest, LargestRepresentableSize)
{
  constexpr auto max = std::numeric_limits<std::uint64_t>::max();
  EXPECT_EQ(cudf::detail::parse_size(std::to_string(max), "size", true), max);
  EXPECT_EQ(cudf::detail::parse_size(std::to_string(max) + "B", "size", true), max);
  constexpr std::uint64_t bytes_per_mib = 1024 * 1024;
  constexpr auto max_mib                = max / bytes_per_mib;
  EXPECT_EQ(cudf::detail::parse_size(std::to_string(max_mib) + "MiB", "size", true),
            max_mib * bytes_per_mib);
  EXPECT_EQ(cudf::detail::parse_size("18EB", "size", true),
            std::uint64_t{18'000'000'000'000'000'000ULL});
  EXPECT_EQ(cudf::detail::parse_size("15EiB", "size", true), std::uint64_t{15} << 60);
}

TEST(ParseSizeTest, ByteConversionOverflow)
{
  constexpr std::uint64_t bytes_per_mib = 1024 * 1024;
  constexpr auto first_overflowing_mib =
    std::numeric_limits<std::uint64_t>::max() / bytes_per_mib + 1;
  EXPECT_THROW(
    cudf::detail::parse_size(std::to_string(first_overflowing_mib) + "MiB", "size", true),
    std::invalid_argument);
  EXPECT_THROW(cudf::detail::parse_size("19EB", "size", true), std::invalid_argument);
  EXPECT_THROW(cudf::detail::parse_size("16EiB", "size", true), std::invalid_argument);
}

TEST(ParseSizeTest, UnknownUnitInDiagnostics)
{
  try {
    cudf::detail::parse_size("1XB", "size", true);
    FAIL() << "Expected an unknown unit to throw";
  } catch (std::invalid_argument const& error) {
    EXPECT_NE(std::string_view{error.what()}.find("XB"), std::string_view::npos);
  }
}

TEST(ParseSizeTest, IdentifierInDiagnostics)
{
  std::string const invalid_inputs[] = {
    "", "-1", "0", "1XB", "16EiB", std::to_string(std::numeric_limits<std::uint64_t>::max()) + "0"};
  for (auto const& input : invalid_inputs) {
    SCOPED_TRACE(input);
    try {
      cudf::detail::parse_size(input, "memory_limit", true);
      FAIL() << "Expected invalid input to throw";
    } catch (std::invalid_argument const& error) {
      EXPECT_NE(std::string_view{error.what()}.find("memory_limit"), std::string_view::npos);
    }
  }
}
