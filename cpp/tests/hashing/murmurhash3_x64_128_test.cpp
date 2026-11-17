/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/hashing.hpp>

using NumericTypesNoBools =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

template <typename T>
class MurmurHash3_x64_128_TestTyped : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(MurmurHash3_x64_128_TestTyped, NumericTypesNoBools);

TYPED_TEST(MurmurHash3_x64_128_TestTyped, TestNumeric)
{
  using T   = TypeParam;
  auto col1 = cudf::test::fixed_width_column_wrapper<T, int32_t>{
    {-1, -1, 0, 2, 22, 0, 11, 12, 116, 32, 0, 42, 7, 62, 1, -22, 0, 0},
    {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0}};
  auto col2 = cudf::test::fixed_width_column_wrapper<T, int32_t>{
    {-1, -1, 0, 2, 22, 1, 11, 12, 116, 32, 0, 42, 7, 62, 1, -22, 1, -22},
    {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0}};

  auto output1 = cudf::hashing::murmurhash3_x64_128(cudf::table_view({col1}));
  auto output2 = cudf::hashing::murmurhash3_x64_128(cudf::table_view({col2}));
  CUDF_TEST_EXPECT_TABLES_EQUAL(output1->view(), output2->view());

  output1 = cudf::hashing::murmurhash3_x64_128(cudf::table_view({col1}), 7);
  output2 = cudf::hashing::murmurhash3_x64_128(cudf::table_view({col2}), 7);
  CUDF_TEST_EXPECT_TABLES_EQUAL(output1->view(), output2->view());
}

class MurmurHash3_x64_128_Test : public cudf::test::BaseFixture {};

TEST_F(MurmurHash3_x64_128_Test, StringType)
{
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

  auto output = cudf::hashing::murmurhash3_x64_128(cudf::table_view({col1}));
  // these were generated using the CPU compiled
  // https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
  auto expected = cudf::test::fixed_width_column_wrapper<uint64_t>({3481043174314896794ul,
                                                                    1981901315483788749ul,
                                                                    1418748153263580713ul,
                                                                    11224732510765974842ul,
                                                                    10813495276579975748ul,
                                                                    8563282101401420087ul,
                                                                    7289234017606107350ul,
                                                                    225672801045596944ul,
                                                                    14927688838032769435ul,
                                                                    7513581995808204968ul,
                                                                    0ul,
                                                                    14163495587303857889ul,
                                                                    4581940570640870180ul,
                                                                    18164432652839101653ul});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view().column(0), expected);

  auto const seed = uint64_t{7};

  output   = cudf::hashing::murmurhash3_x64_128(cudf::table_view({col1}), seed);
  expected = cudf::test::fixed_width_column_wrapper<uint64_t>({5091211404759866125ul,
                                                               12948345853121693662ul,
                                                               14974420008081159223ul,
                                                               4475830656132398742ul,
                                                               15724398074328467356ul,
                                                               4091324140202743991ul,
                                                               7130403777725115865ul,
                                                               11087585763075301159ul,
                                                               12568262854562899547ul,
                                                               2679775340886828858ul,
                                                               17582832888865278351ul,
                                                               5264478748926531221ul,
                                                               8863578460974333747ul,
                                                               11176802453047055260ul});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view().column(0), expected);
}

TEST_F(MurmurHash3_x64_128_Test, ZeroColumns)
{
  auto const input  = cudf::table_view{std::vector<cudf::column_view>{}, 5};
  auto const output = cudf::hashing::murmurhash3_x64_128(input, 42);

  // With no columns to hash, every row's 128-bit result is the initial {seed, 0}.
  cudf::test::fixed_width_column_wrapper<uint64_t> const expected1({42, 42, 42, 42, 42});
  cudf::test::fixed_width_column_wrapper<uint64_t> const expected2({0, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view().column(0), expected1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view().column(1), expected2);
}
