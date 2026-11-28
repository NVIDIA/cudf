/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/hashing.hpp>

#include <array>
#include <limits>
#include <type_traits>

using NumericTypesNoBools =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

template <typename T>
class MurmurHash3_x64_128_TestTyped : public cudf::test::BaseFixture {};

using NumericParams = cudf::test::CrossProduct<
  NumericTypesNoBools,
  cudf::test::Types<std::integral_constant<cudf::use_jit, cudf::use_jit::NO>,
                    std::integral_constant<cudf::use_jit, cudf::use_jit::YES>>>;

TYPED_TEST_SUITE(MurmurHash3_x64_128_TestTyped, NumericParams);

TYPED_TEST(MurmurHash3_x64_128_TestTyped, TestNumeric)
{
  using T            = cudf::test::GetType<TypeParam, 0>;
  constexpr auto jit = cudf::test::GetType<TypeParam, 1>::value;
  auto col1          = cudf::test::fixed_width_column_wrapper<T, int32_t>{
    {-1, -1, 0, 2, 22, 0, 11, 12, 116, 32, 0, 42, 7, 62, 1, -22, 0, 0},
    {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0}};
  auto col2 = cudf::test::fixed_width_column_wrapper<T, int32_t>{
    {-1, -1, 0, 2, 22, 1, 11, 12, 116, 32, 0, 42, 7, 62, 1, -22, 1, -22},
    {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0}};

  auto output1 =
    cudf::hashing::murmurhash3_x64_128(cudf::table_view({col1}), cudf::DEFAULT_HASH_SEED, jit);
  auto output2 =
    cudf::hashing::murmurhash3_x64_128(cudf::table_view({col2}), cudf::DEFAULT_HASH_SEED, jit);
  CUDF_TEST_EXPECT_TABLES_EQUAL(output1->view(), output2->view());

  output1 = cudf::hashing::murmurhash3_x64_128(cudf::table_view({col1}), 7, jit);
  output2 = cudf::hashing::murmurhash3_x64_128(cudf::table_view({col2}), 7, jit);
  CUDF_TEST_EXPECT_TABLES_EQUAL(output1->view(), output2->view());
}

class MurmurHash3_x64_128_Test : public cudf::test::BaseFixtureWithParam<cudf::use_jit> {};

TEST_P(MurmurHash3_x64_128_Test, StringType)
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

  auto output = cudf::hashing::murmurhash3_x64_128(
    cudf::table_view({col1}), cudf::DEFAULT_HASH_SEED, GetParam());
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

  output   = cudf::hashing::murmurhash3_x64_128(cudf::table_view({col1}), seed, GetParam());
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

  // Cover empty input, partial blocks, and complete 16-byte blocks with a full-width seed.
  auto const boundary_input =
    cudf::test::strings_column_wrapper({"",
                                        "0",
                                        "0123456789abcde",
                                        "0123456789abcdef",
                                        "0123456789abcdef0",
                                        "0123456789abcdef0123456789abcde",
                                        "0123456789abcdef0123456789abcdef",
                                        "0123456789abcdef0123456789abcdef0"});
  auto const boundary_output = cudf::hashing::murmurhash3_x64_128(
    cudf::table_view{{boundary_input}}, uint64_t{0x123456789abcdef0}, GetParam());

  // Reference values from the original cuco MurmurHash3 x64 implementation.
  cudf::test::fixed_width_column_wrapper<uint64_t> const expected_low({12670565569421277103ul,
                                                                       7436438012523435542ul,
                                                                       9182848219183220191ul,
                                                                       1608373816723163038ul,
                                                                       17566612986151629899ul,
                                                                       13352410209864817512ul,
                                                                       1213784217078048059ul,
                                                                       4667731712117700191ul});
  cudf::test::fixed_width_column_wrapper<uint64_t> const expected_high({8552480720497679734ul,
                                                                        15811099216923661374ul,
                                                                        5968932763527278272ul,
                                                                        7539280683575153609ul,
                                                                        16398509401717830041ul,
                                                                        4777729961966234481ul,
                                                                        10427324415301665755ul,
                                                                        11601329378799277982ul});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(boundary_output->view().column(0), expected_low);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(boundary_output->view().column(1), expected_high);
}

TEST_P(MurmurHash3_x64_128_Test, IntegerType)
{
  auto constexpr seed = uint64_t{0x123456789abcdef0};
  auto constexpr min  = std::numeric_limits<int32_t>::min();
  auto constexpr max  = std::numeric_limits<int32_t>::max();

  // Reference values from a CPU implementation of MurmurHash3 x64 128 with a 64-bit seed.
  cudf::test::fixed_width_column_wrapper<int32_t> const int32_input({0, 1, -1, 42, min, max});
  auto output =
    cudf::hashing::murmurhash3_x64_128(cudf::table_view{{int32_input}}, seed, GetParam());
  cudf::test::fixed_width_column_wrapper<uint64_t> const int32_low({16171072276759621859ul,
                                                                    17262603546401455210ul,
                                                                    12711963030754346615ul,
                                                                    18226976503916875536ul,
                                                                    16328679645040193767ul,
                                                                    11179377120613608218ul});
  cudf::test::fixed_width_column_wrapper<uint64_t> const int32_high({16963931180576252815ul,
                                                                     16408157432823198965ul,
                                                                     2074124821410842894ul,
                                                                     4423695892072655596ul,
                                                                     10433797240704751331ul,
                                                                     16165709806370322755ul});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view().column(0), int32_low);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view().column(1), int32_high);

  cudf::test::fixed_width_column_wrapper<int64_t> const int64_input({0, 1, -1, 42, min, max});
  output = cudf::hashing::murmurhash3_x64_128(cudf::table_view{{int64_input}}, seed, GetParam());
  cudf::test::fixed_width_column_wrapper<uint64_t> const int64_low({8446848690776147849ul,
                                                                    2177231992012963529ul,
                                                                    5774400613906242884ul,
                                                                    7282352372194964032ul,
                                                                    1340115803818592068ul,
                                                                    11946284795443978790ul});
  cudf::test::fixed_width_column_wrapper<uint64_t> const int64_high({9845961485014813475ul,
                                                                     7246624320603305217ul,
                                                                     120396094497276611ul,
                                                                     15497694688976670770ul,
                                                                     4390549089724037051ul,
                                                                     7654781420204105039ul});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view().column(0), int64_low);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view().column(1), int64_high);
}

TEST_P(MurmurHash3_x64_128_Test, DictionaryIndices)
{
  for (bool const has_nulls : {false, true}) {
    SCOPED_TRACE(has_nulls);
    auto const validity        = std::array{true, true, true, !has_nulls};
    auto const parent_validity = std::array{true, true, true, true, !has_nulls};
    cudf::test::fixed_width_column_wrapper<int64_t> values{{100, 100, 200, 300, 100},
                                                           parent_validity.begin()};
    cudf::test::fixed_width_column_wrapper<int32_t> indices{{0, 1, 2, 0}, validity.begin()};
    for (auto const type :
         {cudf::type_id::INT8, cudf::type_id::INT16, cudf::type_id::INT32, cudf::type_id::INT64}) {
      SCOPED_TRACE(static_cast<int>(type));
      auto const encoded = cudf::dictionary::encode(values, cudf::data_type{type});
      auto const input   = cudf::slice(encoded->view(), {1, 5}).front();
      for (uint64_t const seed : {uint64_t{0}, uint64_t{0x123456789abcdef0}}) {
        SCOPED_TRACE(seed);
        auto const expected =
          cudf::hashing::murmurhash3_x64_128(cudf::table_view{{indices}}, seed, cudf::use_jit::NO);
        auto const result =
          cudf::hashing::murmurhash3_x64_128(cudf::table_view{{input}}, seed, GetParam());
        CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());
      }
    }
  }
}

TEST_P(MurmurHash3_x64_128_Test, ZeroColumns)
{
  auto const input  = cudf::table_view{std::vector<cudf::column_view>{}, 5};
  auto const output = cudf::hashing::murmurhash3_x64_128(input, 42, GetParam());

  // With no columns to hash, every row's 128-bit result is the initial {seed, 0}.
  cudf::test::fixed_width_column_wrapper<uint64_t> const expected1({42, 42, 42, 42, 42});
  cudf::test::fixed_width_column_wrapper<uint64_t> const expected2({0, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view().column(0), expected1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view().column(1), expected2);
}

INSTANTIATE_TEST_SUITE_P(Execution,
                         MurmurHash3_x64_128_Test,
                         ::testing::Values(cudf::use_jit::NO, cudf::use_jit::YES),
                         [](auto const& info) {
                           return info.param == cudf::use_jit::YES ? "Jit" : "Aot";
                         });
