/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/strings/utilities.hpp>
#include <cudf/table/table_view.hpp>

#include <cuda/iterator>

#include <vector>

struct StringsColumnTest : public cudf::test::BaseFixture {};

TEST_F(StringsColumnTest, Sort)
{
  // cannot initialize std::string with a nullptr so use "<null>" as a place-holder
  cudf::test::strings_column_wrapper h_strings({"eee", "bb", "<null>", "", "aa", "bbb", "ééé"},
                                               {1, 1, 0, 1, 1, 1, 1});
  cudf::test::strings_column_wrapper h_expected({"<null>", "", "aa", "bb", "bbb", "eee", "ééé"},
                                                {0, 1, 1, 1, 1, 1, 1});

  auto results =
    cudf::sort(cudf::table_view({h_strings}), {cudf::order::ASCENDING}, {cudf::null_order::BEFORE});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), h_expected);
}

TEST_F(StringsColumnTest, SortZeroSizeStringsColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();

  auto results = cudf::sort(cudf::table_view({zero_size_strings_column}));
  cudf::test::expect_column_empty(results->view().column(0));
}

class SliceParmsTest : public StringsColumnTest,
                       public testing::WithParamInterface<cudf::size_type> {};

TEST_P(SliceParmsTest, Slice)
{
  std::vector<char const*> h_strings{"eee", "bb", nullptr, "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper input(
    h_strings.begin(), h_strings.end(), cudf::test::iterators::nulls_from_nullptrs(h_strings));

  cudf::size_type start = 3;
  cudf::size_type end   = GetParam();

  auto scol    = cudf::slice(input, {start, end});
  auto results = std::make_unique<cudf::column>(scol.front());

  cudf::test::strings_column_wrapper expected(
    h_strings.begin() + start,
    h_strings.begin() + end,
    cuda::transform_iterator(h_strings.begin() + start, [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_P(SliceParmsTest, SliceAllNulls)
{
  std::vector<char const*> h_strings{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  cudf::test::strings_column_wrapper input(
    h_strings.begin(), h_strings.end(), cudf::test::iterators::nulls_from_nullptrs(h_strings));

  cudf::size_type start = 3;
  cudf::size_type end   = GetParam();

  auto scol    = cudf::slice(input, {start, end});
  auto results = std::make_unique<cudf::column>(scol.front());

  cudf::test::strings_column_wrapper expected(
    h_strings.begin() + start,
    h_strings.begin() + end,
    cuda::transform_iterator(h_strings.begin() + start, [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_P(SliceParmsTest, SliceAllEmpty)
{
  std::vector<char const*> h_strings{"", "", "", "", "", "", ""};
  cudf::test::strings_column_wrapper input(h_strings.begin(), h_strings.end());

  cudf::size_type start = 3;
  cudf::size_type end   = GetParam();

  auto scol    = cudf::slice(input, {start, end});
  auto results = std::make_unique<cudf::column>(scol.front());

  cudf::test::strings_column_wrapper expected(h_strings.begin() + start, h_strings.begin() + end);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

INSTANTIATE_TEST_CASE_P(StringsColumnTest,
                        SliceParmsTest,
                        testing::ValuesIn(std::array<cudf::size_type, 3>{5, 6, 7}));

TEST_F(StringsColumnTest, SliceZeroSizeStringsColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();

  auto scol    = cudf::slice(zero_size_strings_column, {0, 0});
  auto results = std::make_unique<cudf::column>(scol.front());
  cudf::test::expect_column_empty(results->view());
}

TEST_F(StringsColumnTest, Gather)
{
  std::vector<char const*> h_strings{"eee", "bb", nullptr, "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cudf::test::iterators::nulls_from_nullptrs(h_strings));

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map{{4, 1}};
  auto results = cudf::gather(cudf::table_view{{strings}}, gather_map)->release();

  std::vector<char const*> h_expected{"aa", "bb"};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(), h_expected.end(), cudf::test::iterators::nulls_from_nullptrs(h_expected));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results.front()->view(), expected);
}

TEST_F(StringsColumnTest, GatherZeroSizeStringsColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();

  cudf::column_view map_view(cudf::data_type{cudf::type_id::INT32}, 0, nullptr, nullptr, 0);
  auto results = cudf::gather(cudf::table_view{{zero_size_strings_column}}, map_view)->release();
  cudf::test::expect_column_empty(results.front()->view());
}

TEST_F(StringsColumnTest, GatherTooBig)
{
  if (cudf::strings::is_large_strings_enabled()) { return; }

  std::vector<int8_t> h_chars(3000000);
  cudf::test::fixed_width_column_wrapper<int8_t> chars(h_chars.begin(), h_chars.end());
  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets({0, 3000000});
  auto input = cudf::column_view(cudf::data_type{cudf::type_id::STRING},
                                 1,
                                 cudf::column_view(chars).begin<int8_t>(),
                                 nullptr,
                                 0,
                                 0,
                                 {offsets});
  auto map   = cuda::constant_iterator<int8_t>(0);
  cudf::test::fixed_width_column_wrapper<int8_t> gather_map(map, map + 1000);
  EXPECT_THROW(cudf::gather(cudf::table_view{{input}}, gather_map), std::overflow_error);
}

TEST_F(StringsColumnTest, Scatter)
{
  cudf::test::strings_column_wrapper target({"eee", "bb", "", "", "aa", "bbb", "ééé"},
                                            {1, 1, 0, 1, 1, 1, 1});
  cudf::test::strings_column_wrapper source({"1", "22"});

  cudf::test::fixed_width_column_wrapper<int32_t> scatter_map({4, 1});

  auto results = cudf::scatter(cudf::table_view({source}), scatter_map, cudf::table_view({target}));

  cudf::test::strings_column_wrapper expected({"eee", "22", "", "", "1", "bbb", "ééé"},
                                              {1, 1, 0, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), expected);
}

TEST_F(StringsColumnTest, ScatterScalar)
{
  cudf::test::strings_column_wrapper target({"eee", "bb", "", "", "aa", "bbb", "ééé"},
                                            {1, 1, 0, 1, 1, 1, 1});

  cudf::test::fixed_width_column_wrapper<int32_t> scatter_map({0, 5});

  cudf::string_scalar scalar("__");
  auto source  = std::vector<std::reference_wrapper<cudf::scalar const>>({scalar});
  auto results = cudf::scatter(source, scatter_map, cudf::table_view({target}));

  cudf::test::strings_column_wrapper expected({"__", "bb", "", "", "aa", "__", "ééé"},
                                              {1, 1, 0, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), expected);
}

TEST_F(StringsColumnTest, ScatterZeroSizeStringsColumn)
{
  auto const source      = cudf::make_empty_column(cudf::type_id::STRING)->view();
  auto const target      = cudf::make_empty_column(cudf::type_id::STRING)->view();
  auto const scatter_map = cudf::make_empty_column(cudf::type_id::INT8)->view();

  auto results = cudf::scatter(cudf::table_view({source}), scatter_map, cudf::table_view({target}));
  cudf::test::expect_column_empty(results->view().column(0));

  cudf::string_scalar scalar("");
  auto scalar_source = std::vector<std::reference_wrapper<cudf::scalar const>>({scalar});
  results            = cudf::scatter(scalar_source, scatter_map, cudf::table_view({target}));
  cudf::test::expect_column_empty(results->view().column(0));
}

CUDF_TEST_PROGRAM_MAIN()
