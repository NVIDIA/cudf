/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/error.hpp>

#include <stdexcept>
#include <string>
#include <vector>

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

class XXHash_64_Bytes_Test : public cudf::test::BaseFixture {};

namespace {

// Lengths cross the 4-, 8- and 32-byte XXH64 block boundaries
std::vector<std::string> const bytes_test_strings{
  "",
  "a",
  "abc",
  "Warsaw",
  "hello world",
  "\xc5\xbc\xc3\xb3\xc5\x82w",                             // "żółw"
  "Pi\xc4\x99kna \xc5\x82\xc4\x85ka pod Warszaw\xc4\x85",  // "Piękna łąka pod Warszawą"
  "abcdefghijklmnopqrstuvwxyz0123456789",
};

// Generated with https://pypi.org/project/xxhash/ using xxhash.xxh64(s.encode()).intdigest()
std::vector<uint64_t> const bytes_test_expected{17241709254077376921ul,
                                                15154266338359012955ul,
                                                4952883123889572249ul,
                                                8590661318172513296ul,
                                                5020219685658847592ul,
                                                11625363514459306632ul,
                                                15948448143701318288ul,
                                                7273945407305660262ul};

// Generated with https://pypi.org/project/xxhash/
// using xxhash.xxh64(s.encode(), seed=42).intdigest()
std::vector<uint64_t> const bytes_test_expected_seed_42{11002672306508523268ul,
                                                        9864288744972464332ul,
                                                        1423657621850124518ul,
                                                        7716771431931769078ul,
                                                        7620854247404556961ul,
                                                        16469419851005331080ul,
                                                        7912833126575397331ul,
                                                        2724237963427677065ul};

// Build a LIST<T> column whose rows are the bytes of `strings`
template <typename T>
std::unique_ptr<cudf::column> make_byte_lists(std::vector<std::string> const& strings)
{
  std::vector<cudf::size_type> offsets{0};
  std::vector<T> bytes;
  for (auto const& s : strings) {
    for (char c : s) {
      bytes.push_back(static_cast<T>(static_cast<unsigned char>(c)));
    }
    offsets.push_back(static_cast<cudf::size_type>(bytes.size()));
  }
  auto offsets_col =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>(offsets.begin(), offsets.end())
      .release();
  auto bytes_col = cudf::test::fixed_width_column_wrapper<T>(bytes.begin(), bytes.end()).release();
  return cudf::make_lists_column(static_cast<cudf::size_type>(strings.size()),
                                 std::move(offsets_col),
                                 std::move(bytes_col),
                                 0,
                                 cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
}

}  // namespace

TEST_F(XXHash_64_Bytes_Test, Strings)
{
  auto const input =
    cudf::test::strings_column_wrapper(bytes_test_strings.begin(), bytes_test_strings.end());
  auto const expected = cudf::test::fixed_width_column_wrapper<uint64_t>(
    bytes_test_expected.begin(), bytes_test_expected.end());
  auto const expected_42 = cudf::test::fixed_width_column_wrapper<uint64_t>(
    bytes_test_expected_seed_42.begin(), bytes_test_expected_seed_42.end());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::hashing::xxhash_64_bytes(input)->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::hashing::xxhash_64_bytes(input, 42)->view(), expected_42);

  // xxhash_64 of a single non-null STRING column is the plain XXH64 of its bytes as well
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::hashing::xxhash_64(cudf::table_view({input}))->view(),
                                 expected);
}

TEST_F(XXHash_64_Bytes_Test, Lists)
{
  auto const expected = cudf::test::fixed_width_column_wrapper<uint64_t>(
    bytes_test_expected.begin(), bytes_test_expected.end());
  auto const expected_42 = cudf::test::fixed_width_column_wrapper<uint64_t>(
    bytes_test_expected_seed_42.begin(), bytes_test_expected_seed_42.end());

  auto const int8_lists  = make_byte_lists<int8_t>(bytes_test_strings);
  auto const uint8_lists = make_byte_lists<uint8_t>(bytes_test_strings);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::hashing::xxhash_64_bytes(*int8_lists)->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::hashing::xxhash_64_bytes(*uint8_lists)->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::hashing::xxhash_64_bytes(*int8_lists, 42)->view(),
                                 expected_42);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::hashing::xxhash_64_bytes(*uint8_lists, 42)->view(),
                                 expected_42);
}

TEST_F(XXHash_64_Bytes_Test, NegativeBytes)
{
  // The hash depends on the bits, not on the signedness of the element type
  auto const int8_lists = cudf::test::lists_column_wrapper<int8_t>{{-128}, {-1}, {-1, -2, -3, -4}};
  auto const uint8_lists =
    cudf::test::lists_column_wrapper<uint8_t>{{0x80}, {0xff}, {0xff, 0xfe, 0xfd, 0xfc}};
  // xxhash.xxh64(bytes.fromhex(h)).intdigest() for "80", "ff", "fffefdfc"
  auto const expected = cudf::test::fixed_width_column_wrapper<uint64_t>(
    {9516710917713200287ul, 10764519495013463364ul, 1589102993886860747ul});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::hashing::xxhash_64_bytes(int8_lists)->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::hashing::xxhash_64_bytes(uint8_lists)->view(), expected);
}

TEST_F(XXHash_64_Bytes_Test, Nulls)
{
  auto const valids   = std::vector<bool>{true, false, true, false, true};
  auto const expected = cudf::test::fixed_width_column_wrapper<uint64_t>(
    {15154266338359012955ul, 0ul, 17241709254077376921ul, 0ul, 8590661318172513296ul},
    valids.begin());

  auto const strings =
    cudf::test::strings_column_wrapper({"a", "", "", "", "Warsaw"}, valids.begin());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::hashing::xxhash_64_bytes(strings)->view(), expected);

  auto const lists = cudf::test::lists_column_wrapper<int8_t>(
    {{'a'}, {}, {}, {}, {'W', 'a', 'r', 's', 'a', 'w'}}, valids.begin());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::hashing::xxhash_64_bytes(lists)->view(), expected);
}

TEST_F(XXHash_64_Bytes_Test, AllNulls)
{
  auto const valids  = std::vector<bool>{false, false};
  auto const strings = cudf::test::strings_column_wrapper({"", ""}, valids.begin());
  auto const lists   = cudf::test::lists_column_wrapper<int8_t>({{}, {}}, valids.begin());
  auto const expected =
    cudf::test::fixed_width_column_wrapper<uint64_t>({0ul, 0ul}, valids.begin());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::hashing::xxhash_64_bytes(strings)->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::hashing::xxhash_64_bytes(lists)->view(), expected);
}

TEST_F(XXHash_64_Bytes_Test, Empty)
{
  auto const expected = cudf::test::fixed_width_column_wrapper<uint64_t>{};
  auto const strings  = cudf::test::strings_column_wrapper{};
  auto const lists    = make_byte_lists<int8_t>({});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::hashing::xxhash_64_bytes(strings)->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::hashing::xxhash_64_bytes(*lists)->view(), expected);
}

TEST_F(XXHash_64_Bytes_Test, Sliced)
{
  auto const strings =
    cudf::test::strings_column_wrapper(bytes_test_strings.begin(), bytes_test_strings.end());
  auto const lists    = make_byte_lists<int8_t>(bytes_test_strings);
  auto const expected = cudf::test::fixed_width_column_wrapper<uint64_t>(
    bytes_test_expected.begin() + 2, bytes_test_expected.begin() + 6);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    cudf::hashing::xxhash_64_bytes(cudf::slice(strings, {2, 6})[0])->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    cudf::hashing::xxhash_64_bytes(cudf::slice(*lists, {2, 6})[0])->view(), expected);
}

TEST_F(XXHash_64_Bytes_Test, UnsupportedTypes)
{
  auto const ints = cudf::test::fixed_width_column_wrapper<int64_t>{1, 2};
  EXPECT_THROW(cudf::hashing::xxhash_64_bytes(ints), cudf::data_type_error);
  auto const int_lists = cudf::test::lists_column_wrapper<int32_t>{{1}, {2}};
  EXPECT_THROW(cudf::hashing::xxhash_64_bytes(int_lists), cudf::data_type_error);
  auto const nested_lists = cudf::test::lists_column_wrapper<int8_t>{{{1}}, {{2}}};
  EXPECT_THROW(cudf::hashing::xxhash_64_bytes(nested_lists), cudf::data_type_error);
  auto const string_lists = cudf::test::lists_column_wrapper<cudf::string_view>{{"a"}, {"b"}};
  EXPECT_THROW(cudf::hashing::xxhash_64_bytes(string_lists), cudf::data_type_error);
  auto child         = cudf::test::fixed_width_column_wrapper<int8_t>{1, 2};
  auto const structs = cudf::test::structs_column_wrapper{{child}};
  EXPECT_THROW(cudf::hashing::xxhash_64_bytes(structs), cudf::data_type_error);
}

TEST_F(XXHash_64_Bytes_Test, NullListElements)
{
  auto const valids = std::vector<bool>{true, false};
  auto const lists  = cudf::test::lists_column_wrapper<int8_t>{{{1, 2}, valids.begin()}};
  EXPECT_THROW(cudf::hashing::xxhash_64_bytes(lists), std::invalid_argument);
}

TEST_F(XXHash_64_Bytes_Test, NullListElementsOutsideSlice)
{
  // Rows [[null], ['a'], [null]]: a null element in the first and last rows, outside the slice
  auto const valids = std::vector<bool>{false};
  auto const lists =
    cudf::test::lists_column_wrapper<int8_t>{{{0}, valids.begin()}, {'a'}, {{0}, valids.begin()}};
  auto const sliced = cudf::slice(lists, {1, 2})[0];
  // xxhash.xxh64(b"a").intdigest()
  auto const expected = cudf::test::fixed_width_column_wrapper<uint64_t>({15154266338359012955ul});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::hashing::xxhash_64_bytes(sliced)->view(), expected);
}
