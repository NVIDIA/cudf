/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/hashing.hpp>

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

  // Different truth values should be equal
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 1, 2, 255, 0});

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

  auto const output2 = cudf::hashing::spark_murmurhash3_x86_32(
    input1, cudf::DEFAULT_HASH_SEED, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
