/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/hashing.hpp>

#include <limits>

class HashTest : public cudf::test::BaseFixture {};

TEST_F(HashTest, MultiValue)
{
  cudf::test::strings_column_wrapper const strings_col(
    {"",
     "The quick brown fox",
     "jumps over the lazy dog.",
     "All work and no play makes Jack a dull boy",
     R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`{|}~)"});

  using limits = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col(
    {0, 100, -100, limits::min(), limits::max()});

  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 1, 1, 0});

  using ts = cudf::timestamp_s;
  cudf::test::fixed_width_column_wrapper<ts, ts::duration> const secs_col(
    {ts::duration::zero(),
     static_cast<ts::duration>(100),
     static_cast<ts::duration>(-100),
     ts::duration::min(),
     ts::duration::max()});

  auto const input1 = cudf::table_view({strings_col, ints_col, bools_col1, secs_col});

  auto const output1 = cudf::hashing::murmurhash3_x86_32(
    input1, cudf::DEFAULT_HASH_SEED, cudf::test::get_default_stream());
}

TEST_F(HashTest, SparkMurmurMultiValue)
{
  // Covers the types that take a Spark specific path, which the shared `MultiValue` case above
  // does not reach: decimal128, floating point, and nested columns.
  using limits = std::numeric_limits<int64_t>;

  cudf::test::strings_column_wrapper const strings_col(
    {"", "The quick brown fox", "jumps over the lazy dog.", "0123456789", "!@#$%^&*()"});
  cudf::test::fixed_width_column_wrapper<int64_t> const longs_col(
    {0L, 100L, -100L, limits::min(), limits::max()});
  cudf::test::fixed_width_column_wrapper<double> const doubles_col(
    {0., -0., 100., -100., std::numeric_limits<double>::quiet_NaN()});
  cudf::test::fixed_point_column_wrapper<__int128_t> const decimal128_col(
    {static_cast<__int128_t>(0),
     static_cast<__int128_t>(100),
     static_cast<__int128_t>(-100),
     static_cast<__int128_t>(1) << 100,
     -(static_cast<__int128_t>(1) << 100)},
    numeric::scale_type{-11});
  cudf::test::lists_column_wrapper<int32_t> const lists_col{{1, 2}, {3}, {}, {4, 5, 6}, {7}};
  cudf::test::fixed_width_column_wrapper<int32_t> struct_a({0, 1, 2, 3, 4});
  cudf::test::fixed_width_column_wrapper<int64_t> struct_b({5L, 6L, 7L, 8L, 9L});
  cudf::test::structs_column_wrapper const structs_col{{struct_a, struct_b}};

  auto const input =
    cudf::table_view({strings_col, longs_col, doubles_col, decimal128_col, lists_col, structs_col});

  auto const output = cudf::hashing::spark_murmurhash3_x86_32(
    input, cudf::DEFAULT_HASH_SEED, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
