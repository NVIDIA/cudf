/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/hashing.hpp>

class XXHash_32_Test : public cudf::test::BaseFixture {};

TEST_F(XXHash_32_Test, TestInteger)
{
  auto col1           = cudf::test::fixed_width_column_wrapper<int32_t>{{0, 42, 825}};
  auto constexpr seed = 0u;
  auto const output   = cudf::hashing::xxhash_32(cudf::table_view({col1}), seed);

  // Expected results were generated with the reference implementation:
  // https://github.com/Cyan4973/xxHash/blob/dev/xxhash.h
  auto expected =
    cudf::test::fixed_width_column_wrapper<uint32_t>({148298089u, 1161967057u, 1066694813u});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(), expected);
}

TEST_F(XXHash_32_Test, TestDouble)
{
  auto col1           = cudf::test::fixed_width_column_wrapper<double>{{-8., 25., 90.}};
  auto constexpr seed = 42u;

  auto const output = cudf::hashing::xxhash_32(cudf::table_view({col1}), seed);

  // Expected results were generated with the reference implementation:
  // https://github.com/Cyan4973/xxHash/blob/dev/xxhash.h
  auto expected =
    cudf::test::fixed_width_column_wrapper<uint32_t>({2276435783u, 3120212431u, 3454197470u});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(), expected);
}

TEST_F(XXHash_32_Test, StringType)
{
  auto col1           = cudf::test::strings_column_wrapper({"I", "am", "AI"});
  auto constexpr seed = 825u;

  auto output = cudf::hashing::xxhash_32(cudf::table_view({col1}), seed);

  // Expected results were generated with the reference implementation:
  // https://github.com/Cyan4973/xxHash/blob/dev/xxhash.h
  auto expected =
    cudf::test::fixed_width_column_wrapper<uint32_t>({320624298u, 1612654309u, 1409499009u});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(), expected);
}

TEST_F(XXHash_32_Test, ZeroColumns)
{
  auto const input  = cudf::table_view{std::vector<cudf::column_view>{}, 5};
  auto const output = cudf::hashing::xxhash_32(input, 42);
  cudf::test::fixed_width_column_wrapper<uint32_t> const expected({42u, 42u, 42u, 42u, 42u});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(), expected);
}
