/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/hashing.hpp>

using NumericTypesNoBools =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

template <typename T>
class XXHash_64_TestTyped : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(XXHash_64_TestTyped, NumericTypesNoBools);

TYPED_TEST(XXHash_64_TestTyped, TestAllNumeric)
{
  using T   = TypeParam;
  auto col1 = cudf::test::fixed_width_column_wrapper<T, int32_t>{
    {-1, -1, 0, 2, 22, 0, 11, 12, 116, 32, 0, 42, 7, 62, 1, -22, 0, 0},
    {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0}};
  auto col2 = cudf::test::fixed_width_column_wrapper<T, int32_t>{
    {-1, -1, 0, 2, 22, 1, 11, 12, 116, 32, 0, 42, 7, 62, 1, -22, 1, -22},
    {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0}};

  auto output1 = cudf::hashing::xxhash_64(cudf::table_view({col1}));
  auto output2 = cudf::hashing::xxhash_64(cudf::table_view({col2}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());

  constexpr uint64_t seed = 7;

  output1 = cudf::hashing::xxhash_64(cudf::table_view({col1}), seed);
  output2 = cudf::hashing::xxhash_64(cudf::table_view({col2}), seed);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

class XXHash_64_Test : public cudf::test::BaseFixture {};

TEST_F(XXHash_64_Test, TestInteger)
{
  auto col1 =
    cudf::test::fixed_width_column_wrapper<int32_t>{{-127,
                                                     -70000,
                                                     0,
                                                     200000,
                                                     128,
                                                     std::numeric_limits<int32_t>::max(),
                                                     std::numeric_limits<int32_t>::min(),
                                                     std::numeric_limits<int32_t>::lowest()}};

  auto const output = cudf::hashing::xxhash_64(cudf::table_view({col1}));

  // these were generated using the CPU compiled version of the cuco xxhash_64 source
  // https://github.com/NVIDIA/cuCollections/blob/dev/include/cuco/detail/hash_functions/xxhash.cuh
  auto expected = cudf::test::fixed_width_column_wrapper<uint64_t>({4827426872506142937ul,
                                                                    13867166853951622683ul,
                                                                    4246796580750024372ul,
                                                                    17339819992360460003ul,
                                                                    7292178400482025765ul,
                                                                    2971168436322821236ul,
                                                                    9380524276503839603ul,
                                                                    9380524276503839603ul});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(), expected);
}

TEST_F(XXHash_64_Test, TestDouble)
{
  auto col1 =
    cudf::test::fixed_width_column_wrapper<double>{{-127.,
                                                    -70000.125,
                                                    0.0,
                                                    200000.5,
                                                    128.5,
                                                    -0.0,
                                                    std::numeric_limits<double>::infinity(),
                                                    std::numeric_limits<double>::quiet_NaN(),
                                                    std::numeric_limits<double>::max(),
                                                    std::numeric_limits<double>::min(),
                                                    std::numeric_limits<double>::lowest()}};

  auto const output = cudf::hashing::xxhash_64(cudf::table_view({col1}));

  // these were generated using the CPU compiled version of the cuco xxhash_64 source
  // https://github.com/NVIDIA/cuCollections/blob/dev/include/cuco/detail/hash_functions/xxhash.cuh
  auto expected = cudf::test::fixed_width_column_wrapper<uint64_t>({16892115221677838993ul,
                                                                    1686446903308179321ul,
                                                                    3803688792395291579ul,
                                                                    18250447068822614389ul,
                                                                    3511911086082166358ul,
                                                                    4558309869707674848ul,
                                                                    18031741628920313605ul,
                                                                    16838308782748609196ul,
                                                                    3127544388062992779ul,
                                                                    1692401401506680154ul,
                                                                    13770442912356326755ul});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(), expected);
}

TEST_F(XXHash_64_Test, StringType)
{
  // clang-format off
  auto col1 = cudf::test::strings_column_wrapper(
    {"The",
     "quick",
     "brown fox",
     "jumps over the lazy dog.",
     "I am Jack's complete lack of null value",
     "A very long (greater than 128 bytes/characters) to test a very long string. "
     "2nd half of the very long string to verify the long string hashing happening.",
     "Some multi-byte characters here: ééé",
     "ééé",
     "ééé ééé",
     "ééé ééé ééé ééé",
     "",
     "!@#$%^&*(())",
     "0123456789",
     "{}|:<>?,./;[]=-"});
  // clang-format on

  auto output = cudf::hashing::xxhash_64(cudf::table_view({col1}));

  // these were generated using the CPU compiled version of the cuco xxhash_64 source
  // https://github.com/NVIDIA/cuCollections/blob/dev/include/cuco/detail/hash_functions/xxhash.cuh
  // Also verified these with https://pypi.org/project/xxhash/
  // using xxhash.xxh64(bytes(s,'utf-8')).intdigest()
  auto expected = cudf::test::fixed_width_column_wrapper<uint64_t>({4686269239494003989ul,
                                                                    6715983472207430822ul,
                                                                    8148134898123095730ul,
                                                                    17291005374665645904ul,
                                                                    2631835514925512071ul,
                                                                    4181420602165187991ul,
                                                                    8749004388517322364ul,
                                                                    17701789113925815768ul,
                                                                    8612485687958712810ul,
                                                                    5148645515269989956ul,
                                                                    17241709254077376921ul,
                                                                    7379359170906687646ul,
                                                                    4566581271137380327ul,
                                                                    17962149534752128981ul});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(), expected);
}

TEST_F(XXHash_64_Test, TestFixedPoint)
{
  auto const col1 = cudf::test::fixed_point_column_wrapper<int32_t>(
    {0, 100, -100, -999999999, 999999999}, numeric::scale_type{-3});
  auto const output = cudf::hashing::xxhash_64(cudf::table_view({col1}));

  // these were generated using the CPU compiled version of the cuco xxhash_64 source
  // https://github.com/NVIDIA/cuCollections/blob/dev/include/cuco/detail/hash_functions/xxhash.cuh
  // and passing the 'value' of each input (without the scale) as the decimal-type
  auto expected = cudf::test::fixed_width_column_wrapper<uint64_t>({4246796580750024372ul,
                                                                    5959467639951725378ul,
                                                                    4122185689695768261ul,
                                                                    3249245648192442585ul,
                                                                    8009575895491381648ul});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(), expected);
}

TEST_F(XXHash_64_Test, ZeroColumns)
{
  auto const input  = cudf::table_view{std::vector<cudf::column_view>{}, 5};
  auto const output = cudf::hashing::xxhash_64(input, 42);
  cudf::test::fixed_width_column_wrapper<uint64_t> const expected({42ul, 42ul, 42ul, 42ul, 42ul});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(), expected);
}
