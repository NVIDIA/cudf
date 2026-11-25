/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../src/sort/string_sort_config.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/memory_resource_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
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

struct StringSort : public cudf::test::BaseFixture {};

TEST_F(StringSort, EmptySingletonAndAllNull)
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

TEST_F(StringSort, HalfNullBothOrders)
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

TEST_F(StringSort, UnstableAscendingEdgeCases)
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

TEST_F(StringSort, StableDuplicatesAndDescendingNulls)
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

TEST_F(StringSort, PrefixBoundaryAndZeroPaddedTies)
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

TEST_F(StringSort, UnalignedPrefixesAndExactWidthTies)
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

TEST_F(StringSort, SlicedColumnUsesSliceRelativeIndices)
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

TEST_F(StringSort, StableMultiBlockInput)
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

TEST_F(StringSort, VariableLengthStrings)
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

TEST_F(StringSort, NonDefaultStreamAndCurrentMemoryResource)
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

TEST_F(StringSort, AlgorithmSelectorParsing)
{
  using cudf::detail::parse_string_sort_algorithm;
  using cudf::detail::string_sort_algorithm;

  EXPECT_EQ(parse_string_sort_algorithm(nullptr), string_sort_algorithm::PREFIX);
  EXPECT_EQ(parse_string_sort_algorithm(""), string_sort_algorithm::PREFIX);
  EXPECT_EQ(parse_string_sort_algorithm("0"), string_sort_algorithm::PREFIX);
  EXPECT_EQ(parse_string_sort_algorithm("1"), string_sort_algorithm::SEGMENTED);
  EXPECT_EQ(parse_string_sort_algorithm("2"), string_sort_algorithm::SEGMENTED_RLE);
  EXPECT_EQ(parse_string_sort_algorithm("3"), string_sort_algorithm::RADIX_LRB);
  EXPECT_EQ(parse_string_sort_algorithm("4"), string_sort_algorithm::PREFIX);
  EXPECT_EQ(parse_string_sort_algorithm("-1"), string_sort_algorithm::PREFIX);
  EXPECT_EQ(parse_string_sort_algorithm("invalid"), string_sort_algorithm::PREFIX);
}

TEST_F(StringSort, RadixLrbScheduleSelectorParsing)
{
  using cudf::detail::parse_radix_lrb_schedule;
  EXPECT_EQ(parse_radix_lrb_schedule(nullptr), 0);
  EXPECT_EQ(parse_radix_lrb_schedule("0"), 0);
  EXPECT_EQ(parse_radix_lrb_schedule("2"), 2);
  EXPECT_EQ(parse_radix_lrb_schedule("1"), 0);
  EXPECT_EQ(parse_radix_lrb_schedule("invalid"), 0);
}

TEST_F(StringSort, SegmentedTuningSelectorParsing)
{
  using cudf::detail::parse_segmented_string_sort_config;
  using cudf::detail::string_sort_algorithm;

  auto const defaults =
    parse_segmented_string_sort_config(string_sort_algorithm::SEGMENTED, nullptr, nullptr, nullptr);
  EXPECT_EQ(defaults.lexic_precision, 1);
  EXPECT_EQ(defaults.radix_run_min, 512);
  EXPECT_FALSE(defaults.eliminate_exact_duplicates);

  auto const tuned =
    parse_segmented_string_sort_config(string_sort_algorithm::SEGMENTED, "8", "128", "1");
  EXPECT_EQ(tuned.lexic_precision, 8);
  EXPECT_EQ(tuned.radix_run_min, 128);
  EXPECT_FALSE(tuned.eliminate_exact_duplicates);
  EXPECT_TRUE(tuned.trace);

  auto const compatibility =
    parse_segmented_string_sort_config(string_sort_algorithm::SEGMENTED_RLE, "0", "1048577", "2");
  EXPECT_EQ(compatibility.lexic_precision, 1);
  EXPECT_EQ(compatibility.radix_run_min, 512);
  EXPECT_TRUE(compatibility.eliminate_exact_duplicates);
  EXPECT_FALSE(compatibility.trace);
}

TEST_F(StringSort, IterativeSegmentedRefinementAndArbitraryBytes)
{
  // Forty values exceed the comparison threshold. Their common prefix survives all four radix
  // passes, forcing the comparison finish for modes 1 and 2.
  std::vector<std::string> strings;
  strings.reserve(40);
  for (int i = 0; i < 40; ++i) {
    auto value = std::string(28, 'p');
    value.push_back('\0');
    value.push_back(static_cast<char>(39 - i));
    strings.push_back(std::move(value));
  }
  auto const input = cudf::test::strings_column_wrapper(strings.begin(), strings.end());

  std::vector<cudf::size_type> expected_data;
  expected_data.reserve(strings.size());
  for (cudf::size_type i = 40; i > 0; --i) {
    expected_data.push_back(i - 1);
  }
  auto const expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    expected_data.begin(), expected_data.end());
  auto const result = cudf::sorted_order(cudf::table_view{{input}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(StringSort, KnownPrefixRunsFinalizeAfterDifferentPasses)
{
  std::vector<std::string> strings;

  // This run is small enough to finalize after its first six-byte key. Its first unknown byte
  // determines the order, so skipping beyond the recorded prefix would fail the comparison.
  for (int suffix = 3; suffix >= 0; --suffix) {
    strings.push_back(std::string{"Aaaaa"} + static_cast<char>('a' + suffix));
  }

  // Each remaining family survives until the requested pass, then splits into pairs. The pair
  // shares every radix byte processed so far and differs immediately afterward. Together these
  // exercise known-prefix offsets of 12, 18, and 24 bytes.
  for (int final_pass = 2; final_pass <= 4; ++final_pass) {
    auto const family = static_cast<char>('A' + final_pass);
    for (int subgroup = 16; subgroup >= 0; --subgroup) {
      auto prefix = std::string(static_cast<std::size_t>((final_pass - 1) * 6), family);
      prefix.append(5, static_cast<char>('k' + final_pass));
      prefix.push_back(static_cast<char>('a' + subgroup));
      strings.push_back(prefix + 'y');
      strings.push_back(prefix + 'x');
    }
  }

  auto expected_indices = std::vector<cudf::size_type>(strings.size());
  std::iota(expected_indices.begin(), expected_indices.end(), cudf::size_type{0});
  std::stable_sort(expected_indices.begin(), expected_indices.end(), [&](auto lhs, auto rhs) {
    return bytewise_less(strings[lhs], strings[rhs]);
  });

  auto const input    = cudf::test::strings_column_wrapper(strings.begin(), strings.end());
  auto const expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    expected_indices.begin(), expected_indices.end());
  auto const result = cudf::stable_sorted_order(cudf::table_view{{input}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(StringSort, RadixBoundariesAndZeroPaddedCollisions)
{
  std::vector<std::string> strings;
  for (auto const length : {7, 8, 9, 15, 16, 17, 31, 32, 33}) {
    for (int suffix = 3; suffix >= 0; --suffix) {
      auto value   = std::string(static_cast<std::size_t>(length), 'a');
      value.back() = static_cast<char>('a' + suffix);
      strings.push_back(std::move(value));
    }
  }
  strings.emplace_back("aaaaaaaa");
  strings.emplace_back(std::string{"aaaaaaaa\0", 9});
  strings.emplace_back(std::string{"aaaaaaaa\0x", 10});
  strings.emplace_back("aaaaaaaa");

  auto expected_indices = std::vector<cudf::size_type>(strings.size());
  std::iota(expected_indices.begin(), expected_indices.end(), cudf::size_type{0});
  std::stable_sort(expected_indices.begin(), expected_indices.end(), [&](auto lhs, auto rhs) {
    return bytewise_less(strings[lhs], strings[rhs]);
  });
  auto const input    = cudf::test::strings_column_wrapper(strings.begin(), strings.end());
  auto const expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    expected_indices.begin(), expected_indices.end());
  auto const ascending = cudf::stable_sorted_order(cudf::table_view{{input}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, ascending->view());

  std::stable_sort(expected_indices.begin(), expected_indices.end(), [&](auto lhs, auto rhs) {
    return bytewise_less(strings[rhs], strings[lhs]);
  });
  auto const expected_descending = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    expected_indices.begin(), expected_indices.end());
  auto const descending =
    cudf::stable_sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_descending, descending->view());
}

TEST_F(StringSort, LaterPassZeroPaddedCollision)
{
  constexpr cudf::size_type copies = 513;
  auto const short_value           = std::string{"qqqqqqqqa"};
  auto const long_value            = short_value + std::string(7, '\0');
  std::vector<std::string> strings;
  strings.reserve(2 * copies);
  for (auto i = 0; i < copies; ++i) {
    // Keeping the longer value first ensures stable equal radix keys cannot accidentally put the
    // shorter value in lexical order before comparison finishing.
    strings.push_back(long_value);
    strings.push_back(short_value);
  }
  auto const input = cudf::test::strings_column_wrapper(strings.begin(), strings.end());

  std::vector<cudf::size_type> ascending_data;
  ascending_data.reserve(strings.size());
  for (cudf::size_type row = 1; row < static_cast<cudf::size_type>(strings.size()); row += 2) {
    ascending_data.push_back(row);
  }
  for (cudf::size_type row = 0; row < static_cast<cudf::size_type>(strings.size()); row += 2) {
    ascending_data.push_back(row);
  }
  auto const expected_ascending = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    ascending_data.begin(), ascending_data.end());
  auto const ascending = cudf::stable_sorted_order(cudf::table_view{{input}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_ascending, ascending->view());

  auto descending_data = ascending_data;
  std::rotate(descending_data.begin(), descending_data.begin() + copies, descending_data.end());
  auto const expected_descending = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    descending_data.begin(), descending_data.end());
  auto const descending =
    cudf::stable_sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_descending, descending->view());
}

TEST_F(StringSort, NullableSlicedLaterPassCollision)
{
  constexpr cudf::size_type copies = 513;
  auto const short_value           = std::string{"qqqqqqqqa"};
  auto const long_value            = short_value + std::string(7, '\0');
  std::vector<std::string> strings{"excluded-before"};
  std::vector<bool> validity{true};
  for (cudf::size_type row = 0; row < copies; ++row) {
    // Interspersed nulls make dense radix positions differ from the byte-count sidecar's row IDs.
    strings.insert(strings.end(), {long_value, short_value, "ignored-null"});
    validity.insert(validity.end(), {true, true, false});
  }
  strings.emplace_back("excluded-after");
  validity.push_back(true);
  auto const input =
    cudf::test::strings_column_wrapper(strings.begin(), strings.end(), validity.begin());
  auto const slice = cudf::slice(input, {1, 1 + 3 * copies}).front();
  for (auto direction : {cudf::order::ASCENDING, cudf::order::DESCENDING}) {
    for (auto nulls : {cudf::null_order::BEFORE, cudf::null_order::AFTER}) {
      auto const nulls_first =
        (direction == cudf::order::ASCENDING) == (nulls == cudf::null_order::BEFORE);
      std::vector<cudf::size_type> expected_rows(slice.size());
      std::iota(expected_rows.begin(), expected_rows.end(), cudf::size_type{0});
      std::stable_sort(expected_rows.begin(), expected_rows.end(), [&](auto lhs, auto rhs) {
        auto const left_valid  = validity[lhs + 1];
        auto const right_valid = validity[rhs + 1];
        if (left_valid != right_valid) { return nulls_first ? !left_valid : left_valid; }
        if (!left_valid) { return false; }
        return direction == cudf::order::ASCENDING
                 ? bytewise_less(strings[lhs + 1], strings[rhs + 1])
                 : bytewise_less(strings[rhs + 1], strings[lhs + 1]);
      });
      auto const expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
        expected_rows.begin(), expected_rows.end());
      auto const stable =
        cudf::stable_sorted_order(cudf::table_view{{slice}}, {direction}, {nulls});
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, stable->view());
      auto const unstable = cudf::sorted_order(cudf::table_view{{slice}}, {direction}, {nulls});
      auto const expected_values = cudf::gather(cudf::table_view{{slice}}, expected);
      auto const actual_values   = cudf::gather(cudf::table_view{{slice}}, unstable->view());
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected_values->view(), actual_values->view());
    }
  }
}

TEST_F(StringSort, LongExactDuplicateRun)
{
  constexpr cudf::size_type num_duplicates = 1025;
  std::vector<std::string> strings;
  strings.reserve(num_duplicates + 2);
  strings.emplace_back("z");
  for (cudf::size_type i = 0; i < num_duplicates; ++i) {
    strings.emplace_back(256, 'm');
  }
  strings.emplace_back("a");
  auto const input = cudf::test::strings_column_wrapper(strings.begin(), strings.end());

  std::vector<cudf::size_type> ascending_data;
  ascending_data.reserve(strings.size());
  ascending_data.push_back(num_duplicates + 1);
  for (cudf::size_type i = 0; i < num_duplicates; ++i) {
    ascending_data.push_back(i + 1);
  }
  ascending_data.push_back(0);
  auto const expected_ascending = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    ascending_data.begin(), ascending_data.end());
  auto const ascending = cudf::stable_sorted_order(cudf::table_view{{input}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_ascending, ascending->view());

  auto descending_data = ascending_data;
  std::reverse(descending_data.begin(), descending_data.end());
  // Reverse the value groups while preserving the duplicate run's input order.
  std::reverse(descending_data.begin() + 1, descending_data.end() - 1);
  auto const expected_descending = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    descending_data.begin(), descending_data.end());
  auto const descending =
    cudf::stable_sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_descending, descending->view());
}

TEST_F(StringSort, SegmentedFinishThreshold32And33)
{
  std::vector<std::string> strings;
  strings.reserve(65);
  for (int i = 0; i < 32; ++i) {
    auto value = std::string{"aaaaaaaa"};
    value.push_back(static_cast<char>(31 - i));
    strings.push_back(std::move(value));
  }
  for (int i = 0; i < 33; ++i) {
    auto value = std::string{"bbbbbbbb"};
    value.push_back(static_cast<char>(32 - i));
    strings.push_back(std::move(value));
  }
  auto const input = cudf::test::strings_column_wrapper(strings.begin(), strings.end());

  std::vector<cudf::size_type> expected_data;
  expected_data.reserve(strings.size());
  for (cudf::size_type i = 32; i > 0; --i) {
    expected_data.push_back(i - 1);
  }
  for (cudf::size_type i = 65; i > 32; --i) {
    expected_data.push_back(i - 1);
  }
  auto const expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    expected_data.begin(), expected_data.end());
  auto const result = cudf::stable_sorted_order(cudf::table_view{{input}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(StringSort, SegmentedRadixRunCutoffBoundaries)
{
  std::vector<std::string> strings;
  for (auto const run_size : {511, 512, 513}) {
    auto const run_prefix = std::string(8, static_cast<char>('a' + run_size - 511));
    for (auto suffix = run_size; suffix > 0; --suffix) {
      auto value = run_prefix;
      value.push_back(static_cast<char>((suffix >> 8) & 0xff));
      value.push_back(static_cast<char>(suffix & 0xff));
      strings.push_back(std::move(value));
    }
  }

  auto expected_indices = std::vector<cudf::size_type>(strings.size());
  std::iota(expected_indices.begin(), expected_indices.end(), cudf::size_type{0});
  std::stable_sort(expected_indices.begin(), expected_indices.end(), [&](auto lhs, auto rhs) {
    return bytewise_less(strings[lhs], strings[rhs]);
  });
  auto const input    = cudf::test::strings_column_wrapper(strings.begin(), strings.end());
  auto const expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    expected_indices.begin(), expected_indices.end());
  auto const result = cudf::stable_sorted_order(cudf::table_view{{input}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(StringSort, MultipleLargeEqualPrefixSegments)
{
  std::vector<std::string> strings;
  strings.reserve(70);
  for (int i = 0; i < 35; ++i) {
    auto a = std::string{"a-common-prefix-that-exceeds-24-bytes-"};
    a.push_back(static_cast<char>(34 - i));
    strings.push_back(std::move(a));
    auto b = std::string{"b-common-prefix-that-exceeds-24-bytes-"};
    b.push_back(static_cast<char>(34 - i));
    strings.push_back(std::move(b));
  }
  auto const input = cudf::test::strings_column_wrapper(strings.begin(), strings.end());

  std::vector<cudf::size_type> expected_data;
  expected_data.reserve(strings.size());
  for (cudf::size_type i = 35; i > 0; --i) {
    expected_data.push_back(2 * (i - 1));
  }
  for (cudf::size_type i = 35; i > 0; --i) {
    expected_data.push_back(2 * (i - 1) + 1);
  }
  auto const expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    expected_data.begin(), expected_data.end());
  auto const result = cudf::sorted_order(cudf::table_view{{input}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(StringSort, StableParallelMergeAcrossBlockBoundaries)
{
  constexpr cudf::size_type num_rows = 777;
  std::vector<std::string> strings;
  strings.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    auto value        = std::string(30, 'q');
    auto const suffix = (num_rows - 1 - i) / 2;
    value.push_back(static_cast<char>((suffix >> 8) & 0xff));
    value.push_back(static_cast<char>(suffix & 0xff));
    strings.push_back(std::move(value));
  }
  auto const input = cudf::test::strings_column_wrapper(strings.begin(), strings.end());

  std::vector<cudf::size_type> expected_data;
  expected_data.reserve(num_rows);
  for (cudf::size_type suffix = 0; suffix < 388; ++suffix) {
    expected_data.push_back(775 - 2 * suffix);
    expected_data.push_back(776 - 2 * suffix);
  }
  expected_data.push_back(0);
  auto const expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    expected_data.begin(), expected_data.end());
  auto const result = cudf::stable_sorted_order(cudf::table_view{{input}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(StringSort, ExactRadixTerminationAndChunkBoundaries)
{
  std::vector<std::string> strings;
  for (auto const length : {7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65}) {
    auto const value = std::string(length, static_cast<char>('a' + length % 20));
    for (int row = 0; row < 513; ++row) {
      strings.push_back(value);
    }
    strings.push_back(value + std::string(8, '\0'));
    strings.push_back(value + "z");
  }
  std::reverse(strings.begin(), strings.end());
  auto const input = cudf::test::strings_column_wrapper(strings.begin(), strings.end());
  for (auto const direction : {cudf::order::ASCENDING, cudf::order::DESCENDING}) {
    std::vector<cudf::size_type> expected_rows(strings.size());
    std::iota(expected_rows.begin(), expected_rows.end(), cudf::size_type{0});
    std::stable_sort(expected_rows.begin(), expected_rows.end(), [&](auto lhs, auto rhs) {
      return direction == cudf::order::ASCENDING ? bytewise_less(strings[lhs], strings[rhs])
                                                 : bytewise_less(strings[rhs], strings[lhs]);
    });
    auto const expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
      expected_rows.begin(), expected_rows.end());
    auto const stable = cudf::stable_sorted_order(cudf::table_view{{input}}, {direction});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, stable->view());
    auto const unstable        = cudf::sorted_order(cudf::table_view{{input}}, {direction});
    auto const gathered        = cudf::gather(cudf::table_view{{input}}, unstable->view());
    auto const stable_gathered = cudf::gather(cudf::table_view{{input}}, stable->view());
    CUDF_TEST_EXPECT_TABLES_EQUAL(stable_gathered->view(), gathered->view());
  }
}

TEST_F(StringSort, ExplicitLargeOffsetsAndSlice)
{
  auto strings = edge_case_strings();
  strings.insert(strings.end(), 513, std::string(40, 'q'));
  std::vector<std::int64_t> offsets{0};
  std::vector<char> chars;
  for (auto const& value : strings) {
    chars.insert(chars.end(), value.begin(), value.end());
    offsets.push_back(static_cast<std::int64_t>(chars.size()));
  }
  auto offsets_column =
    cudf::test::fixed_width_column_wrapper<std::int64_t>(offsets.begin(), offsets.end()).release();
  auto const stream = cudf::get_default_stream();
  auto input =
    cudf::make_strings_column(static_cast<cudf::size_type>(strings.size()),
                              std::move(offsets_column),
                              rmm::device_buffer(chars.data(), chars.size(), stream),
                              0,
                              cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream));
  ASSERT_EQ(input->view().child(0).type().id(), cudf::type_id::INT64);
  auto const slice =
    cudf::slice(input->view(), {1, static_cast<cudf::size_type>(strings.size() - 1)})[0];
  for (auto const direction : {cudf::order::ASCENDING, cudf::order::DESCENDING}) {
    std::vector<cudf::size_type> expected_rows(slice.size());
    std::iota(expected_rows.begin(), expected_rows.end(), cudf::size_type{0});
    std::stable_sort(expected_rows.begin(), expected_rows.end(), [&](auto lhs, auto rhs) {
      return direction == cudf::order::ASCENDING
               ? bytewise_less(strings[lhs + 1], strings[rhs + 1])
               : bytewise_less(strings[rhs + 1], strings[lhs + 1]);
    });
    auto const expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
      expected_rows.begin(), expected_rows.end());
    auto const result = cudf::stable_sorted_order(cudf::table_view{{slice}}, {direction});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}

TEST_F(StringSort, SliceOffsetsBeyondInt32)
{
  auto const first_length = static_cast<std::int64_t>(std::numeric_limits<cudf::size_type>::max());
  auto const slice_begin  = first_length + 18;
  constexpr cudf::size_type rows = 513;
  constexpr std::int64_t width   = 10;
  auto values                    = std::string(1, '\0');
  std::vector<std::int64_t> host_offsets{0, first_length, slice_begin};
  for (cudf::size_type row = 0; row < rows; ++row) {
    auto const suffix = rows - row - 1;
    values += "abcdefgh";
    values += static_cast<char>(suffix >> 8);
    values += static_cast<char>(suffix & 0xff);
    host_offsets.push_back(slice_begin + (row + 1) * width);
  }
  auto const chars_size = static_cast<std::size_t>(slice_begin) + values.size() - 1;
  std::size_t free_bytes{}, total_bytes{};
  CUDF_CUDA_TRY(cudaMemGetInfo(&free_bytes, &total_bytes));
  if (free_bytes < chars_size + (std::size_t{512} << 20)) {
    GTEST_SKIP() << "Requires enough device memory to exercise offsets beyond INT32_MAX";
  }

  // The two excluded rows make the slice's actual byte addresses exceed INT32_MAX.
  // An INT64 offsets column containing only small numbers cannot expose narrowing mistakes.
  // More than 512 equal eight-byte prefixes also force subsequent radix extraction when allowed.
  auto offsets =
    cudf::test::fixed_width_column_wrapper<std::int64_t>(host_offsets.begin(), host_offsets.end());
  auto const stream = cudf::get_default_stream();
  auto chars        = rmm::device_buffer(chars_size, stream);
  // Include the alignment byte before the slice so the aligned extractor never reads
  // uninitialized padding, even though it must not use that byte in a prefix key.
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(
    static_cast<char*>(chars.data()) + slice_begin - 1, values.data(), values.size(), stream));
  auto const input =
    cudf::make_strings_column(rows + 2,
                              offsets.release(),
                              std::move(chars),
                              0,
                              cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream));
  auto const slice = cudf::slice(input->view(), {2, rows + 2})[0];
  for (auto const direction : {cudf::order::ASCENDING, cudf::order::DESCENDING}) {
    std::vector<cudf::size_type> expected_rows(rows);
    std::iota(expected_rows.begin(), expected_rows.end(), 0);
    if (direction == cudf::order::ASCENDING) {
      std::reverse(expected_rows.begin(), expected_rows.end());
    }
    auto const expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
      expected_rows.begin(), expected_rows.end());
    auto const stable = cudf::stable_sorted_order(cudf::table_view{{slice}}, {direction});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, stable->view());
    auto const unstable = cudf::sorted_order(cudf::table_view{{slice}}, {direction});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, unstable->view());
  }
}

// The large-run proof must cover every row, including differences missed by its samples.
TEST_F(StringSort, RadixLrbLargeRunProofAndSuffixRadix)
{
  if (cudf::detail::configured_string_sort_algorithm() !=
      cudf::detail::string_sort_algorithm::RADIX_LRB) {
    GTEST_SKIP() << "Tests the opt-in R8 proof and suffix radix path";
  }
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
