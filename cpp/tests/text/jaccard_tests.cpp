/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/jaccard.hpp>

#include <random>
#include <set>
#include <string>
#include <vector>

struct JaccardTest : public cudf::test::BaseFixture {};

TEST_F(JaccardTest, Basic)
{
  auto input1 =
    cudf::test::strings_column_wrapper({"the quick brown fox", "jumped over the lazy dog."});
  auto input2 =
    cudf::test::strings_column_wrapper({"the slowest brown cat", "crawled under the jumping fox"});

  auto view1 = cudf::strings_column_view(input1);
  auto view2 = cudf::strings_column_view(input2);

  auto results = nvtext::jaccard_index(view1, view2, 5);

  auto expected = cudf::test::fixed_width_column_wrapper<float>({0.103448279f, 0.0697674453f});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  expected = cudf::test::fixed_width_column_wrapper<float>({1.0f, 1.0f});
  results  = nvtext::jaccard_index(view1, view1, 5);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  results = nvtext::jaccard_index(view2, view2, 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(JaccardTest, WithNulls)
{
  auto input1 = cudf::test::strings_column_wrapper({"brown fox", "jumps over dog", "", ""},
                                                   {true, true, false, true});
  auto input2 = cudf::test::strings_column_wrapper({"brown cat", "jumps on fox", "", ""},
                                                   {true, true, true, false});

  auto view1 = cudf::strings_column_view(input1);
  auto view2 = cudf::strings_column_view(input2);

  auto results = nvtext::jaccard_index(view1, view2, 5);

  auto expected = cudf::test::fixed_width_column_wrapper<float>({0.25f, 0.200000003f, 0.f, 0.f},
                                                                {true, true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  expected = cudf::test::fixed_width_column_wrapper<float>({1.0f, 1.0f, 0.f, 0.f},
                                                           {true, true, false, true});
  results  = nvtext::jaccard_index(view1, view1, 7);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(JaccardTest, Errors)
{
  auto input = cudf::test::strings_column_wrapper({"1", "2", "3"});
  auto view  = cudf::strings_column_view(input);
  // invalid parameter value
  EXPECT_THROW(nvtext::jaccard_index(view, view, 1), std::invalid_argument);
  // invalid size
  auto input2 = cudf::test::strings_column_wrapper({"1", "2"});
  auto view2  = cudf::strings_column_view(input2);
  EXPECT_THROW(nvtext::jaccard_index(view, view2, 5), std::invalid_argument);
}

namespace {
// host reference for ASCII-only strings
float jaccard_reference(std::string const& str1, std::string const& str2, int width)
{
  auto substrings = [width](std::string const& str) {
    std::set<std::string> result;
    if (str.empty()) { return result; }
    if (static_cast<int>(str.size()) < width) {
      result.insert(str);
      return result;
    }
    for (std::size_t i = 0; i + width <= str.size(); ++i) {
      result.insert(str.substr(i, width));
    }
    return result;
  };
  auto const set1 = substrings(str1);
  auto const set2 = substrings(str2);
  int intersects  = 0;
  for (auto const& s : set1) {
    intersects += static_cast<int>(set2.count(s));
  }
  auto const unions = static_cast<int>(set1.size() + set2.size()) - intersects;
  return unions ? static_cast<float>(intersects) / static_cast<float>(unions) : 0.f;
}
}  // namespace

TEST_F(JaccardTest, LongRows)
{
  // A small alphabet produces many duplicate substrings within each row.
  // Mixing short and long rows exercises both the per-row and the partitioned code paths.
  std::mt19937 gen(42);
  std::uniform_int_distribution<int> chr_dist('a', 'd');
  auto random_string = [&](int length) {
    std::string result(length, ' ');
    for (auto& ch : result) {
      ch = static_cast<char>(chr_dist(gen));
    }
    return result;
  };

  std::vector<std::string> h_input1;
  std::vector<std::string> h_input2;
  for (auto max_length : {0, 3, 50, 500, 3000, 20000, 100000}) {
    std::uniform_int_distribution<int> len_dist(max_length / 2, max_length);
    for (int i = 0; i < 20; ++i) {
      auto const length = len_dist(gen);
      // the rows share a common part so the intersections are not empty
      auto const common = random_string(length / 2);
      h_input1.push_back(common + random_string(length / 2));
      h_input2.push_back(random_string(length / 3) + common);
    }
  }

  auto input1 = cudf::test::strings_column_wrapper(h_input1.begin(), h_input1.end());
  auto input2 = cudf::test::strings_column_wrapper(h_input2.begin(), h_input2.end());
  auto view1  = cudf::strings_column_view(input1);
  auto view2  = cudf::strings_column_view(input2);

  for (auto width : {2, 5}) {
    std::vector<float> h_expected;
    for (std::size_t i = 0; i < h_input1.size(); ++i) {
      h_expected.push_back(jaccard_reference(h_input1[i], h_input2[i], width));
    }
    auto expected =
      cudf::test::fixed_width_column_wrapper<float>(h_expected.begin(), h_expected.end());
    auto results = nvtext::jaccard_index(view1, view2, width);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
}
