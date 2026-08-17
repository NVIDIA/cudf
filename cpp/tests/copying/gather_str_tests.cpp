/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <random>

class GatherTestStr : public cudf::test::BaseFixtureWithHarness {};

TEST_F(GatherTestStr, StringColumn)
{
  auto const st = this->stream();
  auto const mr = this->resources();

  cudf::test::fixed_width_column_wrapper<int16_t> col1{
    {1, 2, 3, 4, 5, 6}, {true, true, false, true, false, true}, st, mr};
  cudf::test::strings_column_wrapper col2{
    {"This", "is", "not", "a", "string", "type"}, {true, true, true, true, true, false}, st, mr};
  cudf::table_view source_table{{col1, col2}};

  cudf::test::fixed_width_column_wrapper<int16_t> gather_map{{0, 1, 3, 4}, st, mr};

  cudf::test::fixed_width_column_wrapper<int16_t> exp_col1{
    {1, 2, 4, 5}, {true, true, true, false}, st, mr};
  cudf::test::strings_column_wrapper exp_col2{
    {"This", "is", "a", "string"}, {true, true, true, true}, st, mr};
  cudf::table_view expected{{exp_col1, exp_col2}};

  std::unique_ptr<cudf::table> got;
  {
    auto fail_on_current = this->_harness.fail_on_current_device_resource_use();
    got = cudf::gather(source_table, gather_map, cudf::out_of_bounds_policy::DONT_CHECK, st, mr);
    this->_harness.synchronize(st);
  }

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view(), st, mr);
}

TEST_F(GatherTestStr, GatherSlicedStringsColumn)
{
  // TODO: enable fail_on_current after cudf::slice accepts memory_resources
  auto const st = this->stream();
  auto const mr = this->resources();

  cudf::test::strings_column_wrapper strings{
    {"This", "is", "not", "a", "string", "type"}, {true, true, true, true, true, false}, st, mr};
  std::vector<cudf::size_type> slice_indices{0, 2, 2, 3, 3, 6};
  auto sliced_strings = cudf::slice(strings, slice_indices);
  {
    cudf::test::fixed_width_column_wrapper<int16_t> gather_map{{1, 0, 1}, st, mr};
    cudf::test::strings_column_wrapper expected_strings{
      {"is", "This", "is"}, {true, true, true}, st, mr};
    cudf::table_view expected{{expected_strings}};
    auto result = cudf::gather(cudf::table_view{{sliced_strings[0]}},
                               gather_map,
                               cudf::out_of_bounds_policy::DONT_CHECK,
                               st,
                               mr);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result->view(), st, mr);
  }
  {
    cudf::test::fixed_width_column_wrapper<int16_t> gather_map{{0, 0, 0}, st, mr};
    cudf::test::strings_column_wrapper expected_strings{
      {"not", "not", "not"}, {true, true, true}, st, mr};
    cudf::table_view expected{{expected_strings}};
    auto result = cudf::gather(cudf::table_view{{sliced_strings[1]}},
                               gather_map,
                               cudf::out_of_bounds_policy::DONT_CHECK,
                               st,
                               mr);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result->view(), st, mr);
  }
  {
    cudf::test::fixed_width_column_wrapper<int16_t> gather_map{{2, 1, 0}, st, mr};
    cudf::test::strings_column_wrapper expected_strings{
      {"", "string", "a"}, {false, true, true}, st, mr};
    cudf::table_view expected{{expected_strings}};
    auto result = cudf::gather(cudf::table_view{{sliced_strings[2]}},
                               gather_map,
                               cudf::out_of_bounds_policy::DONT_CHECK,
                               st,
                               mr);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result->view(), st, mr);
  }
}

TEST_F(GatherTestStr, Gather)
{
  auto const st = this->stream();
  auto const mr = this->resources();

  std::vector<char const*> h_strings{"eee", "bb", "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end(), st, mr);
  cudf::table_view source_table({strings});

  std::vector<int32_t> h_map{4, 1, 5, 2, 7};
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(h_map.begin(), h_map.end(), st, mr);
  std::unique_ptr<cudf::table> results;
  {
    auto fail_on_current = this->_harness.fail_on_current_device_resource_use();
    results = cudf::gather(source_table,
                              gather_map,
                              cudf::out_of_bounds_policy::NULLIFY,
                              cudf::negative_index_policy::NOT_ALLOWED,
                              st,
                              mr);
    this->_harness.synchronize(st);
  }

  std::vector<char const*> h_expected;
  std::vector<int32_t> expected_validity;
  for (int index : h_map) {
    if ((0 <= index) && (index < static_cast<decltype(index)>(h_strings.size()))) {
      h_expected.push_back(h_strings[index]);
      expected_validity.push_back(1);
    } else {
      h_expected.push_back("");
      expected_validity.push_back(0);
    }
  }
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(), h_expected.end(), expected_validity.begin(), st, mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    results->view().column(0), expected, cudf::test::debug_output_level::FIRST_ERROR, st, mr);
}

TEST_F(GatherTestStr, GatherDontCheckOutOfBounds)
{
  auto const st = this->stream();
  auto const mr = this->resources();

  std::vector<char const*> h_strings{"eee", "bb", "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end(), st, mr);
  cudf::table_view source_table({strings});

  std::vector<int32_t> h_map{3, 4, 0, 0};
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(h_map.begin(), h_map.end(), st, mr);
  std::unique_ptr<cudf::table> results;
  {
    auto fail_on_current = this->_harness.fail_on_current_device_resource_use();
    results = cudf::gather(source_table,
                              gather_map,
                              cudf::out_of_bounds_policy::DONT_CHECK,
                              cudf::negative_index_policy::NOT_ALLOWED,
                              st,
                              mr);
    this->_harness.synchronize(st);
  }

  std::vector<char const*> h_expected;
  for (int itr : h_map) {
    h_expected.push_back(h_strings[itr]);
  }
  cudf::test::strings_column_wrapper expected(h_expected.begin(), h_expected.end(), st, mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    results->view().column(0), expected, cudf::test::debug_output_level::FIRST_ERROR, st, mr);
}

TEST_F(GatherTestStr, GatherEmptyMapStringsColumn)
{
  auto const st = this->stream();
  auto const mr = this->resources();

  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{};
  std::unique_ptr<cudf::table> results;
  {
    auto fail_on_current = this->_harness.fail_on_current_device_resource_use();
    results = cudf::gather(cudf::table_view({zero_size_strings_column->view()}),
                              gather_map,
                              cudf::out_of_bounds_policy::NULLIFY,
                              cudf::negative_index_policy::NOT_ALLOWED,
                              st,
                              mr);
    this->_harness.synchronize(st);
  }
  cudf::test::expect_column_empty(results->get_column(0).view());
}

TEST_F(GatherTestStr, GatherZeroSizeStringsColumn)
{
  auto const st = this->stream();
  auto const mr = this->resources();

  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING);
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map{{0}, st, mr};
  cudf::test::strings_column_wrapper expected{{std::pair<std::string, bool>{"", false}}, st, mr};
  std::unique_ptr<cudf::table> results;
  {
    auto fail_on_current = this->_harness.fail_on_current_device_resource_use();
    results = cudf::gather(cudf::table_view({zero_size_strings_column->view()}),
                              gather_map,
                              cudf::out_of_bounds_policy::NULLIFY,
                              cudf::negative_index_policy::NOT_ALLOWED,
                              st,
                              mr);
    this->_harness.synchronize(st);
  }
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    expected, results->get_column(0).view(), cudf::test::debug_output_level::FIRST_ERROR, st, mr);
}

TEST_F(GatherTestStr, GatherRandomStringsColumn)
{
  auto const st = this->stream();
  auto const mr = this->resources();

  constexpr int num_total_strings    = 512;
  constexpr int num_gathered_strings = 128;

  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> len_dist(0, 20);
  std::uniform_int_distribution<int> ch_dist(97, 122);  // 'a'..'z'

  // Generate random strings
  std::vector<std::string> host_strings;
  host_strings.reserve(num_total_strings);
  for (int i = 0; i < num_total_strings; ++i) {
    int len = len_dist(rng);
    std::string s;
    s.reserve(len);
    for (int j = 0; j < len; ++j) {
      s.push_back(static_cast<char>(ch_dist(rng)));
    }
    host_strings.push_back(std::move(s));
  }

  std::vector<char const*> h_ptrs;
  h_ptrs.reserve(num_total_strings);
  for (auto& s : host_strings) {
    h_ptrs.push_back(s.c_str());
  }

  cudf::test::strings_column_wrapper strings(h_ptrs.begin(), h_ptrs.end(), st, mr);
  cudf::table_view source_table({strings});

  // Generate random string indices to gather
  std::uniform_int_distribution<int> idx_dist(0, num_total_strings - 1);
  std::vector<int32_t> h_map;
  h_map.reserve(num_gathered_strings);
  for (int i = 0; i < num_gathered_strings; ++i) {
    h_map.push_back(static_cast<int32_t>(idx_dist(rng)));
  }

  // Gather strings
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(h_map.begin(), h_map.end(), st, mr);
  std::unique_ptr<cudf::table> result;
  {
    auto fail_on_current = this->_harness.fail_on_current_device_resource_use();
    result = cudf::gather(source_table, gather_map, cudf::out_of_bounds_policy::DONT_CHECK, st, mr);
    this->_harness.synchronize(st);
  }

  std::vector<char const*> h_expected;
  h_expected.reserve(num_gathered_strings);
  for (auto idx : h_map) {
    h_expected.push_back(h_ptrs[static_cast<size_t>(idx)]);
  }
  cudf::test::strings_column_wrapper expected(h_expected.begin(), h_expected.end(), st, mr);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    result->view().column(0), expected, cudf::test::debug_output_level::FIRST_ERROR, st, mr);
}
