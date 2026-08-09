/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "large_strings_fixture.hpp"

#include <cudf_test/column_utilities.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <limits>
#include <vector>

struct ReplaceTest : public cudf::test::StringsLargeTest {};

TEST_F(ReplaceTest, ReplaceLong)
{
  auto const expected = this->very_long_column();
  auto const view     = cudf::column_view(expected);
  // force addressing (rows > max_offset/sizeof(int64)) in a 64-bit offsets column.
  // The threshold is a property of the offsets, which are 32 bits wide until they are not,
  // whatever size_type happens to be.
  int constexpr max_offset = std::numeric_limits<int32_t>::max();
  // minimum number of duplicates to achieve large strings (64-bit offsets)
  int const min_size_multiplier =
    (max_offset / cudf::strings_column_view(view).chars_size(cudf::get_default_stream())) + 1;
  // minimum row multiplier to create max_offset/sizeof(int64) = 268,435,455 rows
  int const min_row_multiplier = ((max_offset / sizeof(int64_t)) / view.size()) + 1;
  int const multiplier         = std::max(min_size_multiplier, min_row_multiplier);

  std::vector<cudf::column_view> input_cols(multiplier, view);
  std::vector<cudf::size_type> splits;
  std::generate_n(std::back_inserter(splits), multiplier - 1, [view, n = 1]() mutable {
    return view.size() * (n++);
  });

  auto large_input = cudf::concatenate(input_cols);  // 480 million rows
  auto const sv    = cudf::strings_column_view(large_input->view());
  EXPECT_EQ(sv.size(), view.size() * multiplier);
  EXPECT_EQ(sv.offsets().type(), cudf::data_type{cudf::type_id::INT64});

  // Using replace tests reading large strings as well as creating large strings
  auto const target = cudf::string_scalar("3");  // fake the actual replace;
  auto const repl   = cudf::string_scalar("3");  // logic still builds the output
  auto result       = cudf::strings::replace(sv, target, repl);

  // verify results in sections
  auto sliced = cudf::split(result->view(), splits);
  for (auto c : sliced) {
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(c, expected);
  }
}

TEST_F(ReplaceTest, ReplaceWide)
{
  auto const expected   = this->long_column();
  auto const view       = cudf::column_view(expected);
  auto const multiplier = 10;
  auto const separator  = cudf::string_scalar("|");
  auto const input      = cudf::strings::concatenate(
    cudf::table_view(std::vector<cudf::column_view>(multiplier, view)), separator);

  auto const input_view = cudf::strings_column_view(input->view());
  auto const target     = cudf::string_scalar("3");  // fake the actual replace;
  auto const repl       = cudf::string_scalar("3");  // logic still builds the output
  auto result           = cudf::strings::replace(input_view, target, repl);

  auto sv = cudf::strings_column_view(result->view());
  EXPECT_EQ(sv.offsets().type(), cudf::data_type{cudf::type_id::INT64});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input->view(), result->view());
}
