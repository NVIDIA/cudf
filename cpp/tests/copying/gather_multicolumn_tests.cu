/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/copying.hpp>
#include <cudf/strings/detail/gather.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/iterator>
#include <cuda/std/iterator>

#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

struct StringsGatherPermutationTest : cudf::test::BaseFixtureWithHarness {};

TEST_F(StringsGatherPermutationTest, SlicedMixedLengths)
{
  constexpr cudf::size_type rows = 130, skip = 5;
  std::vector<int> lengths{0, 1, 4, 8, 9, 15, 16, 17, 32, 127, 128, 129, 1024, 4096};
  std::vector<std::string> values(rows + skip), ids(rows + skip), empty(rows + skip);
  for (int i = 0; i < rows + skip; ++i) {
    values[i].assign(lengths[i % lengths.size()], 'a' + i % 26);
    if (values[i].size() > 1) { values[i][1] = '\0'; }
    ids[i] = "é" + std::to_string(i);
  }
  cudf::test::strings_column_wrapper strings(
    values.begin(), values.end(), cuda::make_constant_iterator(true));
  cudf::test::strings_column_wrapper keys(ids.begin(), ids.end());
  cudf::test::strings_column_wrapper blanks(empty.begin(), empty.end());
  auto sliced = cudf::slice(
    cudf::table_view{{strings, keys, blanks, strings}}, {skip, rows + skip}, stream())[0];
  std::vector<cudf::size_type> indices(rows);
  std::vector<std::string> expected_values(rows), expected_ids(rows), expected_empty(rows);
  for (int i = 0; i < rows; ++i) {
    indices[i]         = (i * 37) % rows;
    expected_values[i] = values[skip + indices[i]];
    expected_ids[i]    = ids[skip + indices[i]];
  }
  cudf::test::fixed_width_column_wrapper<cudf::size_type> map(indices.begin(), indices.end());
  auto map_view = cudf::column_view{map};
  cudf::device_span<cudf::size_type const> permutation{map_view.data<cudf::size_type>(),
                                                       indices.size()};
  cudf::test::strings_column_wrapper expected_strings(expected_values.begin(),
                                                      expected_values.end());
  cudf::test::strings_column_wrapper expected_keys(expected_ids.begin(), expected_ids.end());
  cudf::test::strings_column_wrapper expected_blanks(expected_empty.begin(), expected_empty.end());
  for (auto count : {1, 4}) {
    SCOPED_TRACE(count);
    auto input = count == 1 ? cudf::table_view{{sliced.column(0)}} : sliced;
    auto expected =
      count == 1
        ? cudf::table_view{{expected_strings}}
        : cudf::table_view{{expected_strings, expected_keys, expected_blanks, expected_strings}};
    auto output = cudf::strings::detail::gather_non_null_permutation(
      input, permutation.begin(), permutation.end(), stream(), resources());
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, output->view());
  }
}

TEST_F(StringsGatherPermutationTest, AllEmptyStrings)
{
  cudf::test::strings_column_wrapper strings{"", "", ""};
  auto const input = cudf::table_view{{strings, strings}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> map{2, 0, 1};
  auto const map_view = cudf::column_view{map};
  auto output         = cudf::strings::detail::gather_non_null_permutation(
    input,
    map_view.data<cudf::size_type>(),
    map_view.data<cudf::size_type>() + map_view.size(),
    stream(),
    resources());
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, output->view());
}

TEST_F(StringsGatherPermutationTest, EmptyRows)
{
  auto strings                 = cudf::make_empty_column(cudf::type_id::STRING);
  auto input                   = cudf::table_view{{strings->view(), strings->view()}};
  cudf::size_type const* empty = nullptr;
  auto output =
    cudf::strings::detail::gather_non_null_permutation(input, empty, empty, stream(), resources());
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, output->view());
}

TEST_F(StringsGatherPermutationTest, ZeroColumns)
{
  cudf::table_view input{std::vector<cudf::column_view>{}, 3};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> map{2, 0, 1};
  auto map_view = cudf::column_view{map};
  auto output =
    cudf::strings::detail::gather_non_null_permutation(input,
                                                       map_view.data<cudf::size_type>(),
                                                       map_view.data<cudf::size_type>() + 3,
                                                       stream(),
                                                       resources());
  EXPECT_EQ(output->num_rows(), 3);
  EXPECT_EQ(output->num_columns(), 0);
}

TEST_F(StringsGatherPermutationTest, WrongMapSize)
{
  cudf::test::strings_column_wrapper strings{"a"};
  auto input                   = cudf::table_view{{strings}};
  cudf::size_type const* empty = nullptr;
  EXPECT_THROW(
    cudf::strings::detail::gather_non_null_permutation(input, empty, empty, stream(), resources()),
    cudf::logic_error);
}

TEST_F(StringsGatherPermutationTest, OffsetThreshold)
{
  struct restore_environment {
    char const* name;
    std::optional<std::string> previous;

    explicit restore_environment(char const* variable) : name(variable)
    {
      if (auto const value = std::getenv(name)) { previous = value; }
    }

    ~restore_environment()
    {
      if (previous) {
        setenv(name, previous->c_str(), 1);
      } else {
        unsetenv(name);
      }
    }
  };
  restore_environment threshold_restore{"LIBCUDF_LARGE_STRINGS_THRESHOLD"};
  restore_environment enabled_restore{"LIBCUDF_LARGE_STRINGS_ENABLED"};
  ASSERT_EQ(setenv("LIBCUDF_LARGE_STRINGS_THRESHOLD", "4096", 1), 0);
  ASSERT_EQ(setenv("LIBCUDF_LARGE_STRINGS_ENABLED", "1", 1), 0);

  constexpr cudf::size_type rows = 1'024;
  std::vector<std::string> values(rows), keys(rows);
  for (cudf::size_type i = 0; i < rows; ++i) {
    values[i] = i % 2 ? "é" : "ab";
    keys[i].assign(1, 'a' + i % 26);
  }
  cudf::test::strings_column_wrapper strings(values.begin(), values.end());
  cudf::test::strings_column_wrapper ids(keys.begin(), keys.end());
  cudf::test::strings_column_wrapper expected_strings(values.rbegin(), values.rend());
  cudf::test::strings_column_wrapper expected_ids(keys.rbegin(), keys.rend());
  auto const input    = cudf::table_view{{strings, ids}};
  auto const expected = cudf::table_view{{expected_strings, expected_ids}};
  auto const map = cuda::std::make_reverse_iterator(cuda::counting_iterator<cudf::size_type>{rows});
  ASSERT_EQ(cudf::strings_column_view(input.column(0)).offsets().type().id(), cudf::type_id::INT32);

  // Keep input offsets INT32 and exercise both copy paths in the same batch.
  for (auto const threshold : {2'049, 2'048, 2'047}) {
    ASSERT_EQ(setenv("LIBCUDF_LARGE_STRINGS_THRESHOLD", std::to_string(threshold).c_str(), 1), 0);
    for (auto const enabled : {true, false}) {
      SCOPED_TRACE(::testing::Message() << threshold << "/" << enabled);
      ASSERT_EQ(setenv("LIBCUDF_LARGE_STRINGS_ENABLED", enabled ? "1" : "0", 1), 0);
      if (!enabled && 2 * rows >= threshold) {
        EXPECT_THROW(cudf::strings::detail::gather_non_null_permutation(
                       input, map, map + rows, stream(), resources()),
                     std::overflow_error);
      } else {
        auto output = cudf::strings::detail::gather_non_null_permutation(
          input, map, map + rows, stream(), resources());
        auto const expected_type =
          2 * rows >= threshold ? cudf::type_id::INT64 : cudf::type_id::INT32;
        EXPECT_EQ(cudf::strings_column_view(output->view().column(0)).offsets().type().id(),
                  expected_type);
        EXPECT_EQ(cudf::strings_column_view(output->view().column(1)).offsets().type().id(),
                  cudf::type_id::INT32);
        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected, output->view());
      }
    }
  }
}
