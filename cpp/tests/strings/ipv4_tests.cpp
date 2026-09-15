/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/convert/convert_ipv4.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cuda/iterator>

#include <vector>

struct StringsConvertTest : public cudf::test::BaseFixture {};

TEST_F(StringsConvertTest, IPv4ToIntegers)
{
  std::vector<char const*> h_strings{
    nullptr, "", "hello", "41.168.0.1", "127.0.0.1", "41.197.0.1", "192.168.0.1"};
  cudf::test::strings_column_wrapper strings(
    h_strings.cbegin(),
    h_strings.cend(),
    cuda::transform_iterator(h_strings.begin(), [](auto const str) { return str != nullptr; }));

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::ipv4_to_integers(strings_view);

  std::vector<uint32_t> h_expected{0, 0, 0, 698875905, 2130706433, 700776449, 3232235521};
  cudf::test::fixed_width_column_wrapper<uint32_t> expected(
    h_expected.cbegin(),
    h_expected.cend(),
    cuda::transform_iterator(h_strings.begin(), [](auto const str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsConvertTest, IntegersToIPv4)
{
  std::vector<char const*> h_strings{
    "192.168.0.1", "10.0.0.1", nullptr, "0.0.0.0", "41.186.0.1", "41.197.0.1"};
  cudf::test::strings_column_wrapper strings(
    h_strings.cbegin(),
    h_strings.cend(),
    cuda::transform_iterator(h_strings.begin(), [](auto const str) { return str != nullptr; }));

  std::vector<uint32_t> h_column{3232235521, 167772161, 0, 0, 700055553, 700776449};
  cudf::test::fixed_width_column_wrapper<uint32_t> column(
    h_column.cbegin(),
    h_column.cend(),
    cuda::transform_iterator(h_strings.begin(), [](auto const str) { return str != nullptr; }));

  auto results = cudf::strings::integers_to_ipv4(column);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, strings);
}

TEST_F(StringsConvertTest, ZeroSizeStringsColumnIPV4)
{
  auto const zero_size_column = cudf::make_empty_column(cudf::type_id::INT64)->view();

  auto results = cudf::strings::integers_to_ipv4(zero_size_column);
  cudf::test::expect_column_empty(results->view());
  results = cudf::strings::ipv4_to_integers(results->view());
  EXPECT_EQ(0, results->size());
}

TEST_F(StringsConvertTest, IPv4Error)
{
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, 100);
  EXPECT_THROW(cudf::strings::integers_to_ipv4(column->view()), cudf::logic_error);
}

TEST_F(StringsConvertTest, IsIPv4)
{
  std::vector<char const*> h_strings{"",
                                     "123.456.789.10",
                                     nullptr,
                                     "0.0.0.0",
                                     ".111.211.113",
                                     "127:0:0:1",
                                     "255.255.255.255",
                                     "192.168.0.",
                                     "1...1",
                                     "127.0.A.1",
                                     "9.1.2.3.4",
                                     "8.9"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  cudf::test::fixed_width_column_wrapper<bool> expected(
    {0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0},
    {true, true, false, true, true, true, true, true, true, true, true, true});
  auto results = cudf::strings::is_ipv4(cudf::strings_column_view(strings));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}
