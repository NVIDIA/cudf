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

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda/stream>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <numeric>
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

}  // namespace

struct StringPrefixSort : public cudf::test::BaseFixture {};

TEST_F(StringPrefixSort, EmptySingletonAndAllNull)
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

TEST_F(StringPrefixSort, HalfNullBothOrders)
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

TEST_F(StringPrefixSort, UnstableAscendingEdgeCases)
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

TEST_F(StringPrefixSort, StableDuplicatesAndDescendingNulls)
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

TEST_F(StringPrefixSort, PrefixBoundaryAndZeroPaddedTies)
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

TEST_F(StringPrefixSort, SlicedColumnUsesSliceRelativeIndices)
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

TEST_F(StringPrefixSort, StableMultiBlockInput)
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

TEST_F(StringPrefixSort, VariableLengthStrings)
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

TEST_F(StringPrefixSort, NonDefaultStreamAndCurrentMemoryResource)
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
