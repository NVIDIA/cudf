/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cuda/iterator>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <string>
#include <vector>

struct StringsSliceTest : public cudf::test::BaseFixture {};

TEST_F(StringsSliceTest, Substring)
{
  std::vector<char const*> h_strings{"Héllo", "thesé", nullptr, "ARE THE", "tést strings", ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  std::vector<char const*> h_expected({"llo", "esé", nullptr, "E T", "st ", ""});
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

  auto strings_column = static_cast<cudf::strings_column_view>(strings);
  auto results        = cudf::strings::slice_strings(
    strings_column, std::optional<cudf::size_type>(2), std::optional<cudf::size_type>(5));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

class Parameters : public StringsSliceTest, public testing::WithParamInterface<cudf::size_type> {};

TEST_P(Parameters, Substring)
{
  std::vector<std::string> h_strings{"basic strings", "that can", "be used", "with STL"};
  auto start = std::optional<cudf::size_type>(GetParam());

  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  auto strings_column = cudf::strings_column_view(strings);
  auto results        = cudf::strings::slice_strings(strings_column, start);

  std::vector<std::string> h_expected;
  for (auto& h_string : h_strings)
    h_expected.push_back(h_string.substr(start.value()));

  cudf::test::strings_column_wrapper expected(h_expected.begin(), h_expected.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_P(Parameters, Substring_From)
{
  std::vector<std::string> h_strings{"basic strings", "that can", "be used", "with STL"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  auto strings_column = cudf::strings_column_view(strings);

  auto param_index = GetParam();
  thrust::host_vector<int32_t> starts(h_strings.size());
  thrust::sequence(starts.begin(), starts.end(), param_index);
  cudf::test::fixed_width_column_wrapper<int32_t> starts_column(starts.begin(), starts.end());
  thrust::host_vector<int32_t> stops(h_strings.size());
  thrust::sequence(stops.begin(), stops.end(), param_index + 2);
  cudf::test::fixed_width_column_wrapper<int32_t> stops_column(stops.begin(), stops.end());

  auto results = cudf::strings::slice_strings(strings_column, starts_column, stops_column);

  std::vector<std::string> h_expected;
  for (size_t idx = 0; idx < h_strings.size(); ++idx)
    h_expected.push_back(h_strings[idx].substr(starts[idx], stops[idx] - starts[idx]));

  cudf::test::strings_column_wrapper expected(h_expected.begin(), h_expected.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_P(Parameters, SubstringStopZero)
{
  auto start = GetParam();
  cudf::test::strings_column_wrapper input({"abc", "défgh", "", "XYZ"});
  auto view = cudf::strings_column_view(input);

  auto results = cudf::strings::slice_strings(
    view, std::optional<cudf::size_type>(start), std::optional<cudf::size_type>(0));
  cudf::test::strings_column_wrapper expected({"", "", "", ""});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto starts =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>({start, start, start, start});
  auto stops = cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 0, 0, 0});

  results = cudf::strings::slice_strings(view, starts, stops);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_P(Parameters, AllEmpty)
{
  std::vector<std::string> h_strings{"", "", "", ""};
  auto start = std::optional<cudf::size_type>(GetParam());

  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  auto strings_column = cudf::strings_column_view(strings);
  auto results        = cudf::strings::slice_strings(strings_column, start);

  std::vector<std::string> h_expected(h_strings);
  cudf::test::strings_column_wrapper expected(h_expected.begin(), h_expected.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  thrust::host_vector<int32_t> starts(h_strings.size(), 1);
  cudf::test::fixed_width_column_wrapper<int32_t> starts_column(starts.begin(), starts.end());
  thrust::host_vector<int32_t> stops(h_strings.size(), 2);
  cudf::test::fixed_width_column_wrapper<int32_t> stops_column(stops.begin(), stops.end());

  results = cudf::strings::slice_strings(strings_column, starts_column, stops_column);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_P(Parameters, AllNulls)
{
  std::vector<char const*> h_strings{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  auto start = std::optional<cudf::size_type>(GetParam());

  auto strings_column = cudf::strings_column_view(strings);
  auto results        = cudf::strings::slice_strings(strings_column, start);

  std::vector<char const*> h_expected(h_strings);
  cudf::test::strings_column_wrapper expected(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_expected.begin(), [](auto str) {
      return str != nullptr;
    }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  thrust::host_vector<int32_t> starts(h_strings.size(), 1);
  cudf::test::fixed_width_column_wrapper<int32_t> starts_column(starts.begin(), starts.end());
  thrust::host_vector<int32_t> stops(h_strings.size(), 2);
  cudf::test::fixed_width_column_wrapper<int32_t> stops_column(stops.begin(), stops.end());

  results = cudf::strings::slice_strings(strings_column, starts_column, stops_column);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

INSTANTIATE_TEST_CASE_P(StringsSliceTest,
                        Parameters,
                        testing::ValuesIn(std::array<cudf::size_type, 3>{1, 2, 3}));

TEST_F(StringsSliceTest, NegativePositions)
{
  cudf::test::strings_column_wrapper strings{
    "a", "bc", "def", "ghij", "klmno", "pqrstu", "vwxyz", ""};
  auto strings_column = cudf::strings_column_view(strings);
  {
    auto results = cudf::strings::slice_strings(strings_column, std::optional<cudf::size_type>(-1));
    cudf::test::strings_column_wrapper expected{"a", "c", "f", "j", "o", "u", "z", ""};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(
      strings_column, std::optional<cudf::size_type>(0), std::optional<cudf::size_type>(-1));
    cudf::test::strings_column_wrapper expected{"", "b", "de", "ghi", "klmn", "pqrst", "vwxy", ""};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(strings_column,
                                                std::optional<cudf::size_type>(7),
                                                std::optional<cudf::size_type>(-2),
                                                std::optional<cudf::size_type>(-1));
    cudf::test::strings_column_wrapper expected{"a", "c", "f", "j", "o", "u", "z", ""};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(strings_column,
                                                std::optional<cudf::size_type>(7),
                                                std::optional<cudf::size_type>(-7),
                                                std::optional<cudf::size_type>(-1));
    cudf::test::strings_column_wrapper expected{
      "a", "cb", "fed", "jihg", "onmlk", "utsrqp", "zyxwv", ""};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(
      strings_column, std::optional<cudf::size_type>(-3), std::optional<cudf::size_type>(-1));
    cudf::test::strings_column_wrapper expected{"", "b", "de", "hi", "mn", "st", "xy", ""};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsSliceTest, NullPositions)
{
  cudf::test::strings_column_wrapper strings{"a", "bc", "def", "ghij", "klmno", "pqrstu", "vwxyz"};
  auto strings_column = cudf::strings_column_view(strings);
  {
    auto results = cudf::strings::slice_strings(
      strings_column, std::nullopt, std::nullopt, std::optional<cudf::size_type>(-1));
    cudf::test::strings_column_wrapper expected{
      "a", "cb", "fed", "jihg", "onmlk", "utsrqp", "zyxwv"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(
      strings_column, std::nullopt, std::nullopt, std::optional<cudf::size_type>(2));
    cudf::test::strings_column_wrapper expected{"a", "b", "df", "gi", "kmo", "prt", "vxz"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(strings_column,
                                                std::optional<cudf::size_type>(0),
                                                std::nullopt,
                                                std::optional<cudf::size_type>(-1));
    cudf::test::strings_column_wrapper expected{"a", "b", "d", "g", "k", "p", "v"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(strings_column,
                                                std::nullopt,
                                                std::optional<cudf::size_type>(-2),
                                                std::optional<cudf::size_type>(-1));
    cudf::test::strings_column_wrapper expected{"a", "c", "f", "j", "o", "u", "z"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(strings_column,
                                                std::nullopt,
                                                std::optional<cudf::size_type>(-1),
                                                std::optional<cudf::size_type>(2));
    cudf::test::strings_column_wrapper expected{"", "b", "d", "gi", "km", "prt", "vx"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsSliceTest, MaxPositions)
{
  cudf::test::strings_column_wrapper strings{"a", "bc", "def", "ghij", "klmno", "pqrstu", "vwxyz"};
  auto strings_column = cudf::strings_column_view(strings);
  cudf::test::strings_column_wrapper expected{"", "", "", "", "", "", ""};

  auto results = cudf::strings::slice_strings(strings_column, std::optional<cudf::size_type>(10));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::slice_strings(
    strings_column, std::optional<cudf::size_type>(0), std::optional<cudf::size_type>(-10));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results =
    cudf::strings::slice_strings(strings_column, std::nullopt, std::optional<cudf::size_type>(-10));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::slice_strings(
    strings_column, std::optional<cudf::size_type>(10), std::optional<cudf::size_type>(19));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::slice_strings(strings_column,
                                         std::optional<cudf::size_type>(10),
                                         std::optional<cudf::size_type>(19),
                                         std::optional<cudf::size_type>(9));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::slice_strings(
    strings_column, std::optional<cudf::size_type>(-10), std::optional<cudf::size_type>(-19));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::slice_strings(strings_column,
                                         std::optional<cudf::size_type>(-10),
                                         std::optional<cudf::size_type>(-19),
                                         std::optional<cudf::size_type>(-1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsSliceTest, MixedTypePositions)
{
  auto input =
    cudf::test::strings_column_wrapper({"a", "bc", "def", "ghij", "klmno", "pqrstu", "éuvwxyz"});
  auto sv       = cudf::strings_column_view(input);
  auto starts   = cudf::test::fixed_width_column_wrapper<int16_t>({0, 1, 2, 3, 4, 5, 6});
  auto stops    = cudf::test::fixed_width_column_wrapper<int64_t>({1, 2, 3, 4, 5, 6, 7});
  auto expected = cudf::test::strings_column_wrapper({"a", "c", "f", "j", "o", "u", "z"});
  auto results  = cudf::strings::slice_strings(sv, starts, stops);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
}

TEST_F(StringsSliceTest, MultiByteChars)
{
  auto input = cudf::test::strings_column_wrapper({
    // clang-format off
    "quick brown fox jumped over the lazy brown dog; the fat cats jump in place without moving "
    "the following code snippet demonstrates how to use search for values in an ordered range  "
            // this placement tests proper multi-byte chars handling  ------vvvvv
    "it returns the last position where value could be inserted without the ééééé ordering ",
    "algorithms execution is parallelized as determined by an execution policy; this is a 12345"
    "continuation of previous row to make sure string boundaries are honored 012345678901234567"
           //   v--- this one also
    "01234567890é34567890012345678901234567890"
    // clang-format on
  });

  auto results = cudf::strings::slice_strings(cudf::strings_column_view(input),
                                              std::optional<cudf::size_type>(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, input);
}

TEST_F(StringsSliceTest, Error)
{
  cudf::test::strings_column_wrapper strings{"this string intentionally left blank"};
  auto strings_view = cudf::strings_column_view(strings);
  auto zero         = std::optional<cudf::size_type>(0);
  EXPECT_THROW(cudf::strings::slice_strings(strings_view, zero, zero, zero), cudf::logic_error);

  auto indexes = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2});
  EXPECT_THROW(cudf::strings::slice_strings(strings_view, indexes, indexes), cudf::logic_error);

  auto indexes_null = cudf::test::fixed_width_column_wrapper<int32_t>({1}, {false});
  EXPECT_THROW(cudf::strings::slice_strings(strings_view, indexes_null, indexes_null),
               cudf::logic_error);

  auto indexes_bad = cudf::test::fixed_width_column_wrapper<float>({1});
  EXPECT_THROW(cudf::strings::slice_strings(strings_view, indexes_bad, indexes_bad),
               cudf::logic_error);
}

TEST_F(StringsSliceTest, ZeroSizeStringsColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();
  auto strings_view                   = cudf::strings_column_view(zero_size_strings_column);

  auto results = cudf::strings::slice_strings(
    strings_view, std::optional<cudf::size_type>(1), std::optional<cudf::size_type>(2));
  cudf::test::expect_column_empty(results->view());

  auto const starts_column = cudf::make_empty_column(cudf::type_id::INT32)->view();
  auto const stops_column  = cudf::make_empty_column(cudf::type_id::INT32)->view();

  results = cudf::strings::slice_strings(strings_view, starts_column, stops_column);
  cudf::test::expect_column_empty(results->view());
}

TEST_F(StringsSliceTest, AllEmpty)
{
  auto strings_col  = cudf::test::strings_column_wrapper({"", "", "", "", ""});
  auto strings_view = cudf::strings_column_view(strings_col);
  auto exp_results  = cudf::column_view(strings_col);

  auto results = cudf::strings::slice_strings(
    strings_view, std::optional<cudf::size_type>(0), std::optional<cudf::size_type>(-1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
  results = cudf::strings::slice_strings(
    strings_view, std::optional<cudf::size_type>(0), std::optional<cudf::size_type>(-1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
}
