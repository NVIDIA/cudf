/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "large_strings_fixture.hpp"

#include <cudf_test/column_utilities.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <vector>

struct ConcatenateTest : public cudf::test::StringsLargeTest {};

TEST_F(ConcatenateTest, ConcatenateVertical)
{
  auto input = this->long_column();
  auto view  = cudf::column_view(input);
  std::vector<cudf::column_view> input_cols;
  std::vector<cudf::size_type> splits;
  int const multiplier = 10;
  for (int i = 0; i < multiplier; ++i) {  // 2500MB > 2GB
    input_cols.push_back(view);
    splits.push_back(view.size() * (i + 1));
  }
  splits.pop_back();  // remove last entry
  auto result = cudf::concatenate(input_cols);
  auto sv     = cudf::strings_column_view(result->view());
  EXPECT_EQ(sv.size(), view.size() * multiplier);
  EXPECT_EQ(sv.offsets().type(), cudf::data_type{cudf::type_id::INT64});

  // verify results in sections
  auto sliced = cudf::split(result->view(), splits);
  for (auto c : sliced) {
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(c, input);
  }

  // also test with large strings column as input
  input_cols.clear();
  input_cols.push_back(input);           // regular column
  input_cols.push_back(result->view());  // large column
  result = cudf::concatenate(input_cols);
  sv     = cudf::strings_column_view(result->view());
  EXPECT_EQ(sv.size(), view.size() * (multiplier + 1));
  EXPECT_EQ(sv.offsets().type(), cudf::data_type{cudf::type_id::INT64});
  splits.push_back(view.size() * multiplier);
  sliced = cudf::split(result->view(), splits);
  for (auto c : sliced) {
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(c, input);
  }
}

TEST_F(ConcatenateTest, ManyColumns)
{
  auto input           = this->wide_column();
  auto view            = cudf::column_view(input);
  int const multiplier = 1200000;
  std::vector<cudf::column_view> input_cols(multiplier, view);  // 2500MB > 2GB
  // this tests a unique path through the code
  auto result = cudf::concatenate(input_cols);
  auto sv     = cudf::strings_column_view(result->view());
  EXPECT_EQ(sv.size(), view.size() * multiplier);
  EXPECT_EQ(sv.offsets().type(), cudf::data_type{cudf::type_id::INT64});

  // verify a sampling of the results: first, middle, and last segments
  for (auto idx : {0, multiplier / 2, multiplier - 1}) {
    auto const begin = idx * view.size();
    auto sliced      = cudf::slice(result->view(), {begin, begin + view.size()});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sliced.front(), view);
  }
}
