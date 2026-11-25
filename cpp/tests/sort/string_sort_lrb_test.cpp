/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/memory_resource_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda/stream>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string bytes(std::initializer_list<unsigned int> values)
{
  std::string result;
  result.reserve(values.size());
  for (auto const value : values) {
    result.push_back(static_cast<char>(value));
  }
  return result;
}

std::vector<std::string> edge_case_strings()
{
  return {"",
          "abcdefghZ",
          "abcdefghA",
          "abc",
          std::string{"abc\0", 4},
          std::string{"abc\0x", 5},
          "abd",
          std::string{"\0", 1},
          "abcdefghA",
          "ignored-null",
          std::string{"abc\0", 4},
          // DEL, U+0080, U+00E9, and U+1F600 exercise unsigned high-bit byte ordering.
          bytes({0x7f}),
          bytes({0xc2, 0x80}),
          bytes({0xc3, 0xa9}),
          bytes({0xf0, 0x9f, 0x98, 0x80}),
          bytes({0xc2, 0x80, 0x00})};  // U+0080 followed by embedded NUL
}

std::vector<bool> edge_case_validity()
{
  return {true,
          true,
          true,
          true,
          true,
          true,
          true,
          true,
          true,
          false,
          true,
          true,
          true,
          true,
          true,
          true};
}

bool bytewise_less(std::string const& lhs, std::string const& rhs)
{
  return std::lexicographical_compare(
    lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), [](char left, char right) {
      return static_cast<uint8_t>(left) < static_cast<uint8_t>(right);
    });
}

// Opt this test binary into the radix LRB path before its first sort reads the setting.
[[maybe_unused]] auto const radix_lrb_enabled = setenv("LIBCUDF_STRING_SORT_RADIX_LRB", "ON", 1);

}  // namespace

struct StringRadixLrbSort : public cudf::test::BaseFixture {};

TEST_F(StringRadixLrbSort, EmptySingletonAndAllNull)
{
  auto const empty       = cudf::make_empty_column(cudf::type_id::STRING);
  auto const empty_order = cudf::stable_sorted_order(cudf::table_view{{empty->view()}});
  EXPECT_EQ(empty_order->size(), 0);

  auto const singleton          = cudf::test::strings_column_wrapper{"only"};
  auto const singleton_order    = cudf::stable_sorted_order(cudf::table_view{{singleton}});
  auto const expected_singleton = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_singleton, singleton_order->view());

  auto const all_null       = cudf::test::strings_column_wrapper{{"x", "y", "z"}, {0, 0, 0}};
  auto const all_null_order = cudf::stable_sorted_order(
    cudf::table_view{{all_null}}, {cudf::order::DESCENDING}, {cudf::null_order::AFTER});
  auto const expected_all_null = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_all_null, all_null_order->view());

  auto const unstable_order = cudf::sorted_order(
    cudf::table_view{{all_null}}, {cudf::order::ASCENDING}, {cudf::null_order::BEFORE});
  EXPECT_EQ(unstable_order->size(), 3);

  auto const all_empty       = cudf::test::strings_column_wrapper{"", "", ""};
  auto const all_empty_order = cudf::stable_sorted_order(
    cudf::table_view{{all_empty}}, {cudf::order::DESCENDING}, {cudf::null_order::AFTER});
  auto const expected_all_empty = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_all_empty, all_empty_order->view());
}

TEST_F(StringRadixLrbSort, HalfNullBothOrders)
{
  auto const input = cudf::test::strings_column_wrapper{
    {"z", "ignored", "a", "ignored", "m", "ignored"}, {1, 0, 1, 0, 1, 0}};

  auto const ascending = cudf::stable_sorted_order(
    cudf::table_view{{input}}, {cudf::order::ASCENDING}, {cudf::null_order::BEFORE});
  auto const expected_ascending =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 3, 5, 2, 4, 0};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_ascending, ascending->view());

  auto const descending = cudf::stable_sorted_order(
    cudf::table_view{{input}}, {cudf::order::DESCENDING}, {cudf::null_order::AFTER});
  auto const expected_descending =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 3, 5, 0, 4, 2};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_descending, descending->view());
}

TEST_F(StringRadixLrbSort, UnstableAscendingEdgeCases)
{
  auto const input_strings = edge_case_strings();
  auto const validity      = edge_case_validity();
  auto const input         = cudf::test::strings_column_wrapper{
    input_strings.begin(), input_strings.end(), validity.begin()};

  auto const order = cudf::sorted_order(
    cudf::table_view{{input}}, {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
  auto const actual = cudf::gather(cudf::table_view{{input}}, order->view());

  std::vector<std::string> const expected_strings{"",
                                                  std::string{"\0", 1},
                                                  "abc",
                                                  std::string{"abc\0", 4},
                                                  std::string{"abc\0", 4},
                                                  std::string{"abc\0x", 5},
                                                  "abcdefghA",
                                                  "abcdefghA",
                                                  "abcdefghZ",
                                                  "abd",
                                                  bytes({0x7f}),
                                                  bytes({0xc2, 0x80}),
                                                  bytes({0xc2, 0x80, 0x00}),
                                                  bytes({0xc3, 0xa9}),
                                                  bytes({0xf0, 0x9f, 0x98, 0x80}),
                                                  ""};
  std::vector<bool> const expected_validity{true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            false};
  auto const expected = cudf::test::strings_column_wrapper{
    expected_strings.begin(), expected_strings.end(), expected_validity.begin()};
  CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view{{expected}}, actual->view());
}

TEST_F(StringRadixLrbSort, StableDuplicatesAndDescendingNulls)
{
  auto const input_strings = edge_case_strings();
  auto const validity      = edge_case_validity();
  auto const input         = cudf::test::strings_column_wrapper{
    input_strings.begin(), input_strings.end(), validity.begin()};

  auto const ascending = cudf::stable_sorted_order(
    cudf::table_view{{input}}, {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
  auto const expected_ascending = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    0, 7, 3, 4, 10, 5, 2, 8, 1, 6, 11, 12, 15, 13, 14, 9};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_ascending, ascending->view());

  auto const descending = cudf::stable_sorted_order(
    cudf::table_view{{input}}, {cudf::order::DESCENDING}, {cudf::null_order::BEFORE});
  auto const expected_descending = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    14, 13, 15, 12, 11, 6, 1, 2, 8, 5, 4, 10, 3, 7, 0, 9};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_descending, descending->view());
}

TEST_F(StringRadixLrbSort, PrefixBoundaryAndZeroPaddedTies)
{
  std::vector<std::string> const strings{std::string{"abcdefgh\0A", 10},
                                         "abcdefgh",
                                         std::string{"abcdefgh\0", 9},
                                         "abcdefg",
                                         "abcdefghA",
                                         std::string{"abcdefg\0", 8},
                                         std::string{"abcdefgh\0\0", 10}};
  auto const input = cudf::test::strings_column_wrapper{strings.begin(), strings.end()};

  auto const result = cudf::stable_sorted_order(cudf::table_view{{input}});
  auto const expected =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{3, 5, 1, 2, 6, 0, 4};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(StringRadixLrbSort, UnalignedPrefixesAndExactWidthTies)
{
  std::vector<std::string> strings;
  for (int length = 0; length <= 17; ++length) {
    strings.emplace_back(length, 'a');
    strings.emplace_back(length, '\0');
  }
  strings.insert(strings.end(), {"abcdefgh", std::string{"abcdefgh\0", 9}, "abcdefgh"});
  auto const input = cudf::test::strings_column_wrapper{strings.begin(), strings.end()};

  for (auto const direction : {cudf::order::ASCENDING, cudf::order::DESCENDING}) {
    std::vector<cudf::size_type> expected_indices(strings.size());
    std::iota(expected_indices.begin(), expected_indices.end(), 0);
    std::stable_sort(expected_indices.begin(), expected_indices.end(), [&](auto lhs, auto rhs) {
      return direction == cudf::order::ASCENDING ? bytewise_less(strings[lhs], strings[rhs])
                                                 : bytewise_less(strings[rhs], strings[lhs]);
    });
    auto const expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
      expected_indices.begin(), expected_indices.end());
    auto const stable = cudf::stable_sorted_order(cudf::table_view{{input}}, {direction});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, stable->view());

    auto const unstable        = cudf::sorted_order(cudf::table_view{{input}}, {direction});
    auto const actual_values   = cudf::gather(cudf::table_view{{input}}, unstable->view());
    auto const expected_values = cudf::gather(cudf::table_view{{input}}, expected);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_values->view(), actual_values->view());
  }
}

TEST_F(StringRadixLrbSort, FourBytePrefixBoundaryAndZeroPaddedTies)
{
  std::vector<std::string> const strings{std::string{"abcd\0A", 6},
                                         "abcd",
                                         std::string{"abcd\0", 5},
                                         "abc",
                                         "abcdA",
                                         std::string{"abc\0", 4},
                                         std::string{"abcd\0\0", 6}};
  auto const input = cudf::test::strings_column_wrapper{strings.begin(), strings.end()};

  auto const result = cudf::stable_sorted_order(cudf::table_view{{input}});
  auto const expected =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{3, 5, 1, 2, 6, 0, 4};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(StringRadixLrbSort, SlicedColumnUsesSliceRelativeIndices)
{
  std::vector<std::string> const strings{
    "outside-left", "prefixZZ", "", "prefixAA", "pre", "prefixAA", "outside-right"};
  std::vector<bool> const validity{true, true, false, true, true, true, true};
  auto const parent =
    cudf::test::strings_column_wrapper{strings.begin(), strings.end(), validity.begin()};
  auto const input = cudf::slice(parent, {1, 6}).front();

  auto const result = cudf::stable_sorted_order(
    cudf::table_view{{input}}, {cudf::order::ASCENDING}, {cudf::null_order::BEFORE});
  auto const expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 3, 2, 4, 0};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(StringRadixLrbSort, StableMultiBlockInput)
{
  constexpr cudf::size_type count = 4097;
  std::vector<std::string> const utf8_components{"A",
                                                 bytes({0xc2, 0x80}),
                                                 bytes({0xc3, 0xa9}),
                                                 bytes({0xe2, 0x82, 0xac}),
                                                 bytes({0xf0, 0x9f, 0x98, 0x80})};
  constexpr char ascii_digits[] = "0123456789abcdef";
  std::vector<std::string> strings;
  strings.reserve(count);
  for (cudf::size_type index = 0; index < count; ++index) {
    if (index % 17 == 0) {
      strings.emplace_back("abcdefgh-duplicate");
    } else {
      auto value = std::string{"abcdefgh"};
      value += utf8_components[index % utf8_components.size()];
      value.push_back('-');
      value.push_back(ascii_digits[(index >> 12) & 0x0f]);
      value.push_back(ascii_digits[(index >> 8) & 0x0f]);
      value.push_back(ascii_digits[(index >> 4) & 0x0f]);
      value.push_back(ascii_digits[index & 0x0f]);
      strings.push_back(std::move(value));
    }
  }

  std::vector<cudf::size_type> expected_indices(count);
  std::iota(expected_indices.begin(), expected_indices.end(), 0);
  std::stable_sort(expected_indices.begin(), expected_indices.end(), [&](auto lhs, auto rhs) {
    return bytewise_less(strings[lhs], strings[rhs]);
  });

  auto const input    = cudf::test::strings_column_wrapper{strings.begin(), strings.end()};
  auto const actual   = cudf::stable_sorted_order(cudf::table_view{{input}});
  auto const expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    expected_indices.begin(), expected_indices.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view());
}

TEST_F(StringRadixLrbSort, VariableLengthStrings)
{
  constexpr cudf::size_type count = 513;
  std::vector<std::string> strings;
  strings.reserve(count);
  for (cudf::size_type index = 0; index < count; ++index) {
    std::string value(8, '\0');
    auto encoded = static_cast<std::uint64_t>(index * 2654435761U);
    for (int byte = 7; byte >= 0; --byte) {
      value[byte] = static_cast<char>(encoded & 0xff);
      encoded >>= 8;
    }
    value.append(static_cast<std::size_t>((index * 37) % 121), static_cast<char>('a' + index % 26));
    strings.push_back(std::move(value));
  }

  std::vector<cudf::size_type> ascending_indices(count);
  std::iota(ascending_indices.begin(), ascending_indices.end(), 0);
  auto const less = [&](auto lhs, auto rhs) { return bytewise_less(strings[lhs], strings[rhs]); };
  std::stable_sort(ascending_indices.begin(), ascending_indices.end(), less);

  auto const input     = cudf::test::strings_column_wrapper{strings.begin(), strings.end()};
  auto const ascending = cudf::stable_sorted_order(cudf::table_view{{input}});
  auto const expected_ascending = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    ascending_indices.begin(), ascending_indices.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_ascending, ascending->view());

  auto descending_indices = ascending_indices;
  std::stable_sort(descending_indices.begin(), descending_indices.end(), [&](auto lhs, auto rhs) {
    return less(rhs, lhs);
  });
  auto const descending =
    cudf::stable_sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
  auto const expected_descending = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    descending_indices.begin(), descending_indices.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_descending, descending->view());
}

TEST_F(StringRadixLrbSort, RandomizedPrefixBoundaryDifferential)
{
  // Strings around the 8- and 12-byte cache boundaries with embedded zeros, duplicates, and nulls.
  constexpr cudf::size_type count = 20000;
  std::vector<std::string> strings;
  std::vector<bool> validity;
  strings.reserve(count);
  uint64_t state = 12345;
  auto next      = [&]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(state >> 33);
  };
  char const alphabet[] = {'\0', 'a', 'b'};
  for (cudf::size_type index = 0; index < count; ++index) {
    std::string value = (next() % 2) ? std::string{"abcdefgh"} : std::string{};
    if (next() % 3 == 0) { value += std::string{"ijk\0", 4}; }
    auto const extra = next() % 9;
    for (uint32_t i = 0; i < extra; ++i) {
      value.push_back(alphabet[next() % 3]);
    }
    strings.push_back(std::move(value));
    validity.push_back(next() % 7 != 0);
  }
  auto const input =
    cudf::test::strings_column_wrapper{strings.begin(), strings.end(), validity.begin()};

  for (auto const column_order : {cudf::order::ASCENDING, cudf::order::DESCENDING}) {
    for (auto const nulls : {cudf::null_order::BEFORE, cudf::null_order::AFTER}) {
      auto const ascending   = column_order == cudf::order::ASCENDING;
      auto const nulls_first = (nulls == cudf::null_order::BEFORE) == ascending;
      std::vector<cudf::size_type> expected_indices(count);
      std::iota(expected_indices.begin(), expected_indices.end(), 0);
      std::stable_sort(expected_indices.begin(), expected_indices.end(), [&](auto lhs, auto rhs) {
        if (!validity[lhs] || !validity[rhs]) {
          if (validity[lhs] == validity[rhs]) { return false; }
          return nulls_first ? !validity[lhs] : !validity[rhs];
        }
        return ascending ? bytewise_less(strings[lhs], strings[rhs])
                         : bytewise_less(strings[rhs], strings[lhs]);
      });
      auto const expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
        expected_indices.begin(), expected_indices.end());

      auto const stable =
        cudf::stable_sorted_order(cudf::table_view{{input}}, {column_order}, {nulls});
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, stable->view());

      auto const unstable = cudf::sorted_order(cudf::table_view{{input}}, {column_order}, {nulls});
      CUDF_TEST_EXPECT_TABLES_EQUAL(
        cudf::gather(cudf::table_view{{input}}, cudf::column_view{expected})->view(),
        cudf::gather(cudf::table_view{{input}}, unstable->view())->view());
    }
  }
}

TEST_F(StringRadixLrbSort, NonDefaultStreamAndCurrentMemoryResource)
{
  auto const input = cudf::test::strings_column_wrapper{
    "abcdefghZ", "abcdefghA", "short", "abcdefghA", "long-common-prefix"};
  auto const expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 3, 0, 4, 2};

  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  cuda::stream stream{cuda::device_ref{device}};
  auto const upstream = cudf::get_current_device_resource_ref();
  auto output_mr      = rmm::mr::statistics_resource_adaptor{upstream};
  auto temporary_mr   = rmm::mr::statistics_resource_adaptor{upstream};

  std::unique_ptr<cudf::column> result;
  {
    auto current_scope = cudf::test::scoped_current_device_resource{temporary_mr};
    result = cudf::stable_sorted_order(cudf::table_view{{input}}, {}, {}, stream, output_mr);
    stream.sync();
  }

  EXPECT_GT(output_mr.get_bytes_counter().total, 0);
  EXPECT_GT(temporary_mr.get_bytes_counter().total, 0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

// Mixed segment sizes straddle all refinement tile boundaries and require multiple merge levels.
TEST_F(StringRadixLrbSort, MixedPrefixSegmentsAndLongSuffixes)
{
  std::vector<std::string> strings;
  std::vector<bool> validity;
  std::vector<int> lengths{1, 2, 127, 128, 255, 256, 257, 1023, 1024, 1025, 4097, 8193};
  for (int group = 0; group < static_cast<int>(lengths.size()); ++group) {
    for (int i = 0; i < lengths[group]; ++i) {
      auto str     = std::string(64, static_cast<char>('a' + group));
      auto const n = std::vector<int>{0, 1, 7, 8, 9, 15, 16, 17, 33}[i % 9];
      for (int j = 0; j < n; ++j)
        str.push_back(static_cast<char>((i * 13 + j * 7) % 127));
      strings.push_back(str);
      validity.push_back(i % 17 != 0);
    }
  }
  std::vector<cudf::size_type> permutation(strings.size());
  std::iota(permutation.begin(), permutation.end(), 0);
  std::mt19937 generator(93017);
  std::shuffle(permutation.begin(), permutation.end(), generator);
  std::vector<std::string> shuffled;
  std::vector<bool> valid;
  for (auto row : permutation) {
    shuffled.push_back(strings[row]);
    valid.push_back(validity[row]);
  }
  for (bool nullable : {false, true}) {
    auto validity_now = nullable ? valid : std::vector<bool>(valid.size(), true);
    auto input =
      cudf::test::strings_column_wrapper{shuffled.begin(), shuffled.end(), validity_now.begin()};
    for (auto order : {cudf::order::ASCENDING, cudf::order::DESCENDING}) {
      for (auto null_order : {cudf::null_order::BEFORE, cudf::null_order::AFTER}) {
        auto expected_rows = permutation;
        std::iota(expected_rows.begin(), expected_rows.end(), 0);
        bool const ascending = order == cudf::order::ASCENDING;
        std::stable_sort(expected_rows.begin(), expected_rows.end(), [&](auto lhs, auto rhs) {
          if (!validity_now[lhs] || !validity_now[rhs]) {
            if (validity_now[lhs] == validity_now[rhs]) return false;
            bool const null_first = ascending == (null_order == cudf::null_order::BEFORE);
            return !validity_now[lhs] == null_first;
          }
          return ascending ? bytewise_less(shuffled[lhs], shuffled[rhs])
                           : bytewise_less(shuffled[rhs], shuffled[lhs]);
        });
        auto expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
          expected_rows.begin(), expected_rows.end());
        auto actual = cudf::stable_sorted_order(cudf::table_view{{input}}, {order}, {null_order});
        CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view());
        auto unstable       = cudf::sorted_order(cudf::table_view{{input}}, {order}, {null_order});
        auto expected_table = cudf::gather(cudf::table_view{{input}}, expected);
        auto actual_table   = cudf::gather(cudf::table_view{{input}}, unstable->view());
        CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table->view(), actual_table->view());
      }
    }
  }
}

TEST_F(StringRadixLrbSort, LargeOffsetsRepresentationAndSlices)
{
  auto const strings = edge_case_strings();
  for (bool nullable : {false, true}) {
    auto validity = edge_case_validity();
    if (!nullable) std::fill(validity.begin(), validity.end(), true);
    auto source =
      cudf::test::strings_column_wrapper{strings.begin(), strings.end(), validity.begin()}
        .release();
    auto const size       = source->size();
    auto const null_count = source->null_count();
    auto offsets  = cudf::cast(source->view().child(0), cudf::data_type{cudf::type_id::INT64});
    auto contents = source->release();
    contents.children[0] = std::move(offsets);
    auto input           = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                                size,
                                                std::move(*contents.data),
                                                std::move(*contents.null_mask),
                                                null_count,
                                                std::move(contents.children));
    for (bool sliced : {false, true}) {
      auto const start = sliced ? 1 : 0;
      auto const end   = sliced ? size - 1 : size;
      auto view        = sliced ? cudf::slice(input->view(), {start, end}).front() : input->view();
      for (auto order : {cudf::order::ASCENDING, cudf::order::DESCENDING}) {
        for (auto null_order : {cudf::null_order::BEFORE, cudf::null_order::AFTER}) {
          bool const ascending = order == cudf::order::ASCENDING;
          std::vector<cudf::size_type> expected_rows(end - start);
          std::iota(expected_rows.begin(), expected_rows.end(), 0);
          std::stable_sort(expected_rows.begin(), expected_rows.end(), [&](auto lhs, auto rhs) {
            auto const a = lhs + start, b = rhs + start;
            if (!validity[a] || !validity[b]) {
              if (validity[a] == validity[b]) return false;
              bool const null_first = ascending == (null_order == cudf::null_order::BEFORE);
              return !validity[a] == null_first;
            }
            return ascending ? bytewise_less(strings[a], strings[b])
                             : bytewise_less(strings[b], strings[a]);
          });
          auto expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
            expected_rows.begin(), expected_rows.end());
          auto actual = cudf::stable_sorted_order(cudf::table_view{{view}}, {order}, {null_order});
          CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view());
          auto unstable       = cudf::sorted_order(cudf::table_view{{view}}, {order}, {null_order});
          auto expected_table = cudf::gather(cudf::table_view{{view}}, expected);
          auto actual_table   = cudf::gather(cudf::table_view{{view}}, unstable->view());
          CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table->view(), actual_table->view());
        }
      }
    }
  }
}

TEST_F(StringRadixLrbSort, FunnelLoadAlignmentBoundariesAndSuffixMismatches)
{
  std::vector<std::string> values;
  std::size_t chars   = 0;
  auto append_aligned = [&](std::string value, std::size_t alignment) {
    auto const padding = (alignment + 4 - chars % 4) % 4;
    if (padding != 0) {
      values.emplace_back(padding, 'p');
      chars += padding;
    }
    chars += value.size();
    values.push_back(std::move(value));
  };
  for (std::size_t alignment = 0; alignment < 4; ++alignment) {
    for (std::size_t length = 0; length <= 65; ++length) {
      std::string value(length, 'a');
      for (std::size_t byte = 12; byte < length; ++byte) {
        value[byte] = static_cast<char>((byte * 17 + length * 7) % 128);
      }
      if (length > 3) value[3] = '\0';
      if (length > 8) value[8] = '\0';
      append_aligned(value, alignment);
      append_aligned(value, (alignment + 1) % 4);  // Equal values with different alignments.
    }
    for (std::size_t mismatch = 12; mismatch < 44; ++mismatch) {
      std::string value(44, 'q');
      append_aligned(value, alignment);
      value[mismatch] = '\0';
      append_aligned(value, alignment);
      value[mismatch] = static_cast<char>(0xff);
      append_aligned(value, alignment);
    }
  }
  auto const input = cudf::test::strings_column_wrapper{values.begin(), values.end()};
  auto const views = cudf::slice(input,
                                 {0,
                                  static_cast<cudf::size_type>(values.size()),
                                  1,
                                  static_cast<cudf::size_type>(values.size() - 1)});
  for (std::size_t slice = 0; slice < views.size(); ++slice) {
    auto const begin = slice == 0 ? 0 : 1;
    for (auto direction : {cudf::order::ASCENDING, cudf::order::DESCENDING}) {
      std::vector<cudf::size_type> rows(views[slice].size());
      std::iota(rows.begin(), rows.end(), 0);
      std::stable_sort(rows.begin(), rows.end(), [&](auto lhs, auto rhs) {
        return direction == cudf::order::ASCENDING
                 ? bytewise_less(values[begin + lhs], values[begin + rhs])
                 : bytewise_less(values[begin + rhs], values[begin + lhs]);
      });
      auto const expected =
        cudf::test::fixed_width_column_wrapper<cudf::size_type>(rows.begin(), rows.end());
      auto const order = cudf::stable_sorted_order(cudf::table_view{{views[slice]}}, {direction});
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, order->view());
      auto const unstable = cudf::sorted_order(cudf::table_view{{views[slice]}}, {direction});
      auto const expected_values = cudf::gather(cudf::table_view{{views[slice]}}, expected);
      auto const actual_values   = cudf::gather(cudf::table_view{{views[slice]}}, unstable->view());
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected_values->view(), actual_values->view());
    }
  }
}

TEST_F(StringRadixLrbSort, LogarithmicBinsAndIndependentMergeParity)
{
  std::vector<std::string> strings;
  int group = 0;
  for (auto length :
       {1,    2,    3,    4,    5,    7,    8,    9,    15,   16,   17,   31,   32,
        33,   63,   64,   65,   127,  128,  129,  255,  256,  257,  511,  512,  513,
        1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 8193, 32769}) {
    auto prefix = std::to_string(group++);
    prefix.insert(0, 8 - prefix.size(), '0');
    for (int i = 0; i < length; ++i) {
      auto suffix = std::to_string((i * 97) % 113);  // Equal values exercise stability.
      suffix.insert(0, 8 - suffix.size(), '0');
      strings.push_back(prefix + std::string((group % 3) * 16 + 8, 'x') + suffix);
    }
  }
  std::mt19937 rng{581923};
  std::shuffle(strings.begin(), strings.end(), rng);
  for (bool nullable : {false, true}) {
    std::vector<bool> validity(strings.size(), true);
    if (nullable) {
      for (std::size_t i = 0; i < strings.size(); i += 17)
        validity[i] = false;
    }
    auto const input =
      cudf::test::strings_column_wrapper{strings.begin(), strings.end(), validity.begin()};
    auto const size = static_cast<cudf::size_type>(strings.size());
    for (bool sliced : {false, true}) {
      auto const start = sliced ? 3 : 0;
      auto const end   = sliced ? size - 5 : size;
      auto const view =
        sliced ? cudf::slice(input, {start, end}).front() : cudf::column_view{input};
      for (auto order : {cudf::order::ASCENDING, cudf::order::DESCENDING}) {
        for (auto null_order : {cudf::null_order::BEFORE, cudf::null_order::AFTER}) {
          bool const ascending = order == cudf::order::ASCENDING;
          std::vector<cudf::size_type> rows(end - start);
          std::iota(rows.begin(), rows.end(), 0);
          std::stable_sort(rows.begin(), rows.end(), [&](auto lhs, auto rhs) {
            auto const a = lhs + start, b = rhs + start;
            if (!validity[a] || !validity[b]) {
              if (validity[a] == validity[b]) return false;
              bool const null_first = ascending == (null_order == cudf::null_order::BEFORE);
              return !validity[a] == null_first;
            }
            return ascending ? bytewise_less(strings[a], strings[b])
                             : bytewise_less(strings[b], strings[a]);
          });
          auto const expected =
            cudf::test::fixed_width_column_wrapper<cudf::size_type>(rows.begin(), rows.end());
          auto const stable =
            cudf::stable_sorted_order(cudf::table_view{{view}}, {order}, {null_order});
          CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, stable->view());
          auto const unstable = cudf::sorted_order(cudf::table_view{{view}}, {order}, {null_order});
          auto const expected_values = cudf::gather(cudf::table_view{{view}}, expected);
          auto const actual_values   = cudf::gather(cudf::table_view{{view}}, unstable->view());
          CUDF_TEST_EXPECT_TABLES_EQUAL(expected_values->view(), actual_values->view());
          auto const [host_rows, mask] = cudf::test::to_host<cudf::size_type>(unstable->view());
          auto permutation             = host_rows;
          std::sort(permutation.begin(), permutation.end());
          std::vector<cudf::size_type> identity(end - start);
          std::iota(identity.begin(), identity.end(), 0);
          EXPECT_EQ(identity, permutation);
        }
      }
    }
  }
}

// The large-run proof must cover every row, including differences missed by its samples.
TEST_F(StringRadixLrbSort, LargeRunProofAndSuffixRadix)
{
  constexpr cudf::size_type count = 32769;
  for (int scenario = 0; scenario < 8; ++scenario) {
    SCOPED_TRACE(scenario);
    std::vector<std::string> strings;
    for (cudf::size_type i = 0; i < count; ++i) {
      std::string str(72, 'a');
      if (scenario == 0 || scenario == 1 || scenario == 7) {
        // Equal length, an unaligned common prefix, high bytes and embedded NULs.
        str.resize(scenario == 1 ? 11 : 77);
        auto const rank     = static_cast<uint32_t>((count - i) % 257);
        str[str.size() - 2] = static_cast<char>(rank >> 8);
        str.back()          = static_cast<char>(rank);
      } else if (scenario == 2) {
        // Sampled duplicates with one unsampled difference.
        if (i == 173) str[51] = 'b';
      } else if (scenario == 3) {
        str.append(i % 3, '\0');  // Equal numeric suffix keys do not prove equal lengths.
      } else if (scenario == 4) {
        str.resize(i % 2 ? 6 : 8);
        std::fill(str.begin(), str.end(), '\0');  // A zero-padded R8 collision below eight bytes.
      } else if (scenario == 5) {
        str.assign(79, 'q');  // A truly identical run preserves the original row order.
      } else if (scenario == 6) {
        str.back() = static_cast<char>(i % 19);
      }
      strings.push_back(std::move(str));
    }
    if (scenario == 6) {
      auto second = strings;
      for (auto& str : second)
        str[0] = 'z';
      strings.insert(strings.end(), second.begin(), second.end());
    }
    if (scenario == 7) strings.push_back("outside");  // One large run is not necessarily full span.
    for (auto const direction : {cudf::order::ASCENDING, cudf::order::DESCENDING}) {
      std::vector<cudf::size_type> expected_indices(strings.size());
      std::iota(expected_indices.begin(), expected_indices.end(), 0);
      std::stable_sort(expected_indices.begin(), expected_indices.end(), [&](auto a, auto b) {
        return direction == cudf::order::ASCENDING ? bytewise_less(strings[a], strings[b])
                                                   : bytewise_less(strings[b], strings[a]);
      });
      auto const expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
        expected_indices.begin(), expected_indices.end());
      auto const input  = cudf::test::strings_column_wrapper(strings.begin(), strings.end());
      auto const result = cudf::stable_sorted_order(cudf::table_view{{input}}, {direction});
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
    }
  }
}

CUDF_TEST_PROGRAM_MAIN()
