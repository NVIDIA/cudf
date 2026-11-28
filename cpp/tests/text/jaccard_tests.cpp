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
// split a UTF-8 string into its characters
std::vector<std::string> to_characters(std::string const& str)
{
  std::vector<std::string> result;
  for (std::size_t i = 0; i < str.size(); ++i) {
    // continuation bytes are appended to the current character
    if ((static_cast<unsigned char>(str[i]) & 0xC0) == 0x80) {
      result.back().push_back(str[i]);
    } else {
      result.emplace_back(1, str[i]);
    }
  }
  return result;
}

// host reference for UTF-8 strings
float jaccard_reference(std::string const& str1, std::string const& str2, int width)
{
  auto substrings = [width](std::string const& str) {
    std::set<std::string> result;
    auto const chars = to_characters(str);
    if (chars.empty()) { return result; }
    if (static_cast<int>(chars.size()) < width) {
      result.insert(str);
      return result;
    }
    for (std::size_t i = 0; i + width <= chars.size(); ++i) {
      std::string sub;
      for (std::size_t j = i; j < i + width; ++j) {
        sub += chars[j];
      }
      result.insert(sub);
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

// compare jaccard_index against the reference for random rows built from the given alphabet
void test_random_rows(std::vector<std::string> const& alphabet, std::vector<int> const& max_lengths)
{
  std::mt19937 gen(42);
  std::uniform_int_distribution<std::size_t> chr_dist(0, alphabet.size() - 1);
  auto random_string = [&](int length) {
    std::string result;
    for (int i = 0; i < length; ++i) {
      result += alphabet[chr_dist(gen)];
    }
    return result;
  };

  std::vector<std::string> h_input1;
  std::vector<std::string> h_input2;
  for (auto max_length : max_lengths) {
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
}  // namespace

TEST_F(JaccardTest, LongRows)
{
  // A small alphabet produces many duplicate substrings within each row.
  // Mixing short and long rows exercises both the per-row and the partitioned code paths.
  test_random_rows({"a", "b", "c", "d"}, {0, 3, 50, 500, 3000, 20000, 100000});
}

TEST_F(JaccardTest, LongRowsUTF8)
{
  // characters of 1, 2, 3, and 4 bytes
  test_random_rows({"a", "é", "中", "😀"}, {0, 3, 50, 500, 3000, 20000});
}
