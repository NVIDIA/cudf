/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <vector>

struct StringsFindTest : public cudf::test::BaseFixture {};

TEST_F(StringsFindTest, Find)
{
  cudf::test::strings_column_wrapper strings({"Héllo", "thesé", "", "lest", "tést strings", ""},
                                             {true, true, false, true, true, true});
  auto strings_view = cudf::strings_column_view(strings);

  {
    auto const target = cudf::string_scalar("é");
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(
      {1, 4, -1, -1, 1, -1}, {true, true, false, true, true, true});
    auto results = cudf::strings::find(strings_view, target);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    results = cudf::strings::rfind(strings_view, target);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(
      {3, -1, -1, 0, -1, -1}, {true, true, false, true, true, true});
    auto results = cudf::strings::rfind(strings_view, cudf::string_scalar("l"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const target = cudf::string_scalar("es");
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(
      {-1, 2, -1, 1, -1, -1}, {true, true, false, true, true, true});
    auto results = cudf::strings::find(strings_view, target);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    results = cudf::strings::rfind(strings_view, target);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(
      {0, 0, 0, 0, 0, 0}, {true, true, false, true, true, true});
    auto results = cudf::strings::find(strings_view, cudf::string_scalar(""));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(
      {5, 5, 0, 4, 12, 0}, {true, true, false, true, true, true});
    auto results = cudf::strings::rfind(strings_view, cudf::string_scalar(""));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const targets = cudf::test::strings_column_wrapper({"l", "t", "", "x", "é", "o"});
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(
      {2, 0, 0, -1, 1, -1}, {true, true, false, true, true, true});
    auto results = cudf::strings::find(strings_view, cudf::strings_column_view(targets));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(
      {0, 0, 0, 0, 0, 0}, {true, true, false, true, true, true});
    auto results = cudf::strings::find(strings_view, strings_view);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsFindTest, Count)
{
  auto validty = cudf::test::iterators::null_at(2);
  auto input   = cudf::test::strings_column_wrapper(
    {"Héllo there", "thesé are some strings: ééé", "", "ababababababa", "tést strings", ""},
    validty);
  auto sv = cudf::strings_column_view(input);

  auto results = cudf::strings::count(sv, cudf::string_scalar("e"));
  auto expected =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>({2, 3, 0, 0, 0, 0}, validty);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results  = cudf::strings::count(sv, cudf::string_scalar("é"));
  expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>({1, 4, 0, 0, 1, 0}, validty);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results  = cudf::strings::count(sv, cudf::string_scalar("the"));
  expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>({1, 1, 0, 0, 0, 0}, validty);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results  = cudf::strings::count(sv, cudf::string_scalar("aba"));
  expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 0, 0, 3, 0, 0}, validty);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsFindTest, CountLongStrings)
{
  auto input =
    cudf::test::strings_column_wrapper({"Héllo there. This is a long string to test the count "
                                        "function. It should be more than 32 bytes.",
                                        "ababababababababababababababababababababababa"});
  auto sv = cudf::strings_column_view(input);

  auto results  = cudf::strings::count(sv, cudf::string_scalar("e"));
  auto expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>({7, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results  = cudf::strings::count(sv, cudf::string_scalar("aba"));
  expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 11});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsFindTest, FindWithNullTargets)
{
  cudf::test::strings_column_wrapper input({"hello hello", "thesé help", "", "helicopter", "", "x"},
                                           {true, true, false, true, true, true});
  auto strings_view = cudf::strings_column_view(input);

  auto const targets = cudf::test::strings_column_wrapper(
    {"lo he", "", "hhh", "cop", "help", "xyz"}, {true, false, true, true, true, true});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(
    {3, -1, -1, 4, -1, -1}, {true, false, false, true, true, true});
  auto results = cudf::strings::find(strings_view, cudf::strings_column_view(targets));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsFindTest, FindLongStrings)
{
  cudf::test::strings_column_wrapper input(
    {"Héllo, there world and goodbye",
     "quick brown fox jumped over the lazy brown dog; the fat cats jump in place without moving",
     "the following code snippet demonstrates how to use search for values in an ordered range",
     "it returns the last position where value could be inserted without violating the ordering",
     "algorithms execution is parallelized as determined by an execution policy. t",
     "he this is a continuation of previous row to make sure string boundaries are honored",
     ""});
  auto view    = cudf::strings_column_view(input);
  auto results = cudf::strings::find(view, cudf::string_scalar("the"));
  auto expected =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>({7, 28, 0, 11, -1, -1, -1});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  auto targets =
    cudf::test::strings_column_wrapper({"the", "the", "the", "the", "the", "the", "the"});
  results = cudf::strings::find(view, cudf::strings_column_view(targets));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  results  = cudf::strings::rfind(view, cudf::string_scalar("the"));
  expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>({7, 48, 0, 77, -1, -1, -1});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  targets  = cudf::test::strings_column_wrapper({"there", "cat", "the", "", "the", "are", "dog"});
  results  = cudf::strings::find(view, cudf::strings_column_view(targets));
  expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>({7, 56, 0, 0, -1, 73, -1});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  results  = cudf::strings::find(view, cudf::string_scalar("ing"));
  expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>({-1, 86, 10, 73, -1, 58, -1});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  results  = cudf::strings::rfind(view, cudf::string_scalar("ing"));
  expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>({-1, 86, 10, 86, -1, 58, -1});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsFindTest, Contains)
{
  cudf::test::strings_column_wrapper strings(
    {"Héllo", "thesé", "", "lease", "tést strings", "", "eé", "éte"},
    {true, true, false, true, true, true, true, true});
  auto strings_view = cudf::strings_column_view(strings);
  {
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {0, 1, 0, 1, 0, 0, 1, 1}, {true, true, false, true, true, true, true, true});
    auto results = cudf::strings::contains(strings_view, cudf::string_scalar("e"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {1, 1, 0, 0, 1, 0, 1, 1}, {true, true, false, true, true, true, true, true});
    auto results = cudf::strings::contains(strings_view, cudf::string_scalar("é"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::strings_column_wrapper targets({"Hello", "é", "e", "x", "", "", "n", "t"},
                                               {true, true, true, true, true, false, true, true});
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {0, 1, 0, 0, 1, 0, 0, 1}, {true, true, false, true, true, true, true, true});
    auto results = cudf::strings::contains(strings_view, cudf::strings_column_view(targets));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsFindTest, ContainsLongStrings)
{
  cudf::test::strings_column_wrapper strings(
    {"Héllo, there world and goodbye",
     "quick brown fox jumped over the lazy brown dog; the fat cats jump in place without moving",
     "the following code snippet demonstrates how to use search for values in an ordered range",
     "it returns the last position where value could be inserted without violating the ordering",
     "algorithms execution is parallelized as determined by an execution policy. t",
     "he this is a continuation of previous row to make sure string boundaries are honored",
     "abcdefghijklmnopqrstuvwxyz 0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ !@#$%^&*()~",
     ""});
  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::contains(strings_view, cudf::string_scalar("e"));
  auto expected     = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1, 1, 1, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  results  = cudf::strings::contains(strings_view, cudf::string_scalar(" the "));
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 1, 0, 1, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  results  = cudf::strings::contains(strings_view, cudf::string_scalar("a"));
  expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1, 1, 1, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  results  = cudf::strings::contains(strings_view, cudf::string_scalar("~"));
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 0, 0, 0, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsFindTest, ContainsAllLengthTiers)
{
  // Rows spanning every LRB bin: 0 bytes up to > 8 KiB (block tier), with the target placed at
  // the start, the end, the middle, or absent; plus nulls and rows shorter than the target.
  auto const target = std::string("0987 5W43");
  std::mt19937 rng(42);
  std::vector<std::string> data;
  std::vector<bool> valid;
  std::vector<bool> expected;
  auto add = [&](std::string row, bool is_valid) {
    expected.push_back(is_valid && row.find(target) != std::string::npos);
    data.push_back(std::move(row));
    valid.push_back(is_valid);
  };
  for (int len : {0, 1, 5, 8, 9, 10, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 255,
                  256, 257, 511, 512, 1000, 1024, 2047, 2048, 4096, 8191, 8192, 8193, 12000,
                  16384, 40000, 70000}) {
    std::string filler(len, 'x');
    add(filler, true);
    if (len >= static_cast<int>(target.size())) {
      auto at_start = filler;
      at_start.replace(0, target.size(), target);
      add(at_start, true);
      auto at_end = filler;
      at_end.replace(len - target.size(), target.size(), target);
      add(at_end, true);
      auto mid = filler;
      mid.replace((len - target.size()) / 2, target.size(), target);
      add(mid, true);
      // near miss: all but the last byte of the target
      auto near = filler;
      near.replace(0, target.size() - 1, target.substr(0, target.size() - 1));
      add(near, true);
    }
    add(filler, false);  // null
  }
  // shuffle so bins are interleaved in row order
  std::vector<size_t> perm(data.size());
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);
  std::vector<std::string> sdata;
  std::vector<bool> svalid, sexp;
  for (auto i : perm) {
    sdata.push_back(data[i]);
    svalid.push_back(valid[i]);
    sexp.push_back(expected[i]);
  }

  cudf::test::strings_column_wrapper strings(sdata.begin(), sdata.end(), svalid.begin());
  auto const view = cudf::strings_column_view(strings);
  auto results    = cudf::strings::contains(view, cudf::string_scalar(target));
  cudf::test::fixed_width_column_wrapper<bool> exp(sexp.begin(), sexp.end(), svalid.begin());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp);

  // sliced view (non-zero offset) must dispatch and index correctly
  auto const n      = static_cast<cudf::size_type>(sdata.size());
  auto const sliced = cudf::slice(strings, {7, n - 5}).front();
  auto sres         = cudf::strings::contains(cudf::strings_column_view(sliced), cudf::string_scalar(target));
  std::vector<bool> ssexp(sexp.begin() + 7, sexp.end() - 5);
  std::vector<bool> ssval(svalid.begin() + 7, svalid.end() - 5);
  cudf::test::fixed_width_column_wrapper<bool> sexp_col(ssexp.begin(), ssexp.end(), ssval.begin());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*sres, sexp_col);

  // empty target: every valid row contains it
  auto eres = cudf::strings::contains(view, cudf::string_scalar(""));
  std::vector<bool> ones(sdata.size(), true);
  cudf::test::fixed_width_column_wrapper<bool> eexp(ones.begin(), ones.end(), svalid.begin());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*eres, eexp);
}

TEST_F(StringsFindTest, ContainsManyRowsMixed)
{
  // Enough rows that every scheduling pass spans many blocks; bins mixed row by row.
  auto const target = std::string("Zq");
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> len_d(0, 600);
  std::uniform_int_distribution<int> hit_d(0, 3);
  std::vector<std::string> data;
  std::vector<bool> expected;
  for (int i = 0; i < 20000; ++i) {
    auto len = len_d(rng);
    if (i % 997 == 0) len = 9000 + (i % 3000);  // a few block-tier rows
    std::string row(len, 'a' + (i % 26));
    auto const hit = hit_d(rng) == 0 && len >= 2;
    if (hit) {
      std::uniform_int_distribution<int> pos_d(0, len - 2);
      row.replace(pos_d(rng), 2, target);
    }
    expected.push_back(hit);
    data.push_back(std::move(row));
  }
  cudf::test::strings_column_wrapper strings(data.begin(), data.end());
  auto results = cudf::strings::contains(cudf::strings_column_view(strings), cudf::string_scalar(target));
  cudf::test::fixed_width_column_wrapper<bool> exp(expected.begin(), expected.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp);
}

TEST_F(StringsFindTest, StartsWith)
{
  cudf::test::strings_column_wrapper strings({"Héllo", "thesé", "", "lease", "tést strings", ""},
                                             {true, true, false, true, true, true});
  auto strings_view = cudf::strings_column_view(strings);
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 1, 0},
                                                          {true, true, false, true, true, true});
    auto results = cudf::strings::starts_with(strings_view, cudf::string_scalar("t"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    std::vector<char const*> h_targets{"éa", "th", "e", "ll", "tést strings", ""};
    cudf::test::strings_column_wrapper targets(h_targets.begin(), h_targets.end());

    auto targets_view = cudf::strings_column_view(targets);
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 1, 1},
                                                          {true, true, false, true, true, true});
    auto results = cudf::strings::starts_with(strings_view, targets_view);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 0, 0},
                                                          {true, true, false, true, true, true});
    auto results = cudf::strings::starts_with(strings_view, cudf::string_scalar("thesé"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    std::vector<char const*> h_targets{"éa", "th", "e", "ll", nullptr, ""};
    cudf::test::strings_column_wrapper targets(
      h_targets.begin(), h_targets.end(), cuda::transform_iterator(h_targets.begin(), [](auto str) {
        return str != nullptr;
      }));

    auto targets_view = cudf::strings_column_view(targets);
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 0, 1},
                                                          {true, true, false, true, true, true});
    auto results = cudf::strings::starts_with(strings_view, targets_view);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsFindTest, EndsWith)
{
  cudf::test::strings_column_wrapper strings({"Héllo", "thesé", "", "lease", "tést strings", ""},
                                             {true, true, false, true, true, true});
  auto strings_view = cudf::strings_column_view(strings);
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 0, 0, 1, 0, 0},
                                                          {true, true, false, true, true, true});
    auto results = cudf::strings::ends_with(strings_view, cudf::string_scalar("se"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    std::vector<char const*> h_targets{"éa", "sé", "th", "ll", "tést strings", ""};
    cudf::test::strings_column_wrapper targets(h_targets.begin(), h_targets.end());

    auto targets_view = cudf::strings_column_view(targets);
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 1, 1},
                                                          {true, true, false, true, true, true});
    auto results = cudf::strings::ends_with(strings_view, targets_view);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 0, 0},
                                                          {true, true, false, true, true, true});
    auto results = cudf::strings::ends_with(strings_view, cudf::string_scalar("thesé"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    std::vector<char const*> h_targets{"éa", "sé", "th", nullptr, "tést strings", ""};
    cudf::test::strings_column_wrapper targets(
      h_targets.begin(), h_targets.end(), cuda::transform_iterator(h_targets.begin(), [](auto str) {
        return str != nullptr;
      }));

    auto targets_view = cudf::strings_column_view(targets);
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 1, 1},
                                                          {true, true, false, true, true, true});
    auto results = cudf::strings::ends_with(strings_view, targets_view);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsFindTest, ZeroSizeStringsColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();
  auto strings_view                   = cudf::strings_column_view(zero_size_strings_column);
  auto results                        = cudf::strings::find(strings_view, cudf::string_scalar("é"));
  EXPECT_EQ(results->size(), 0);
  results = cudf::strings::rfind(strings_view, cudf::string_scalar("é"));
  EXPECT_EQ(results->size(), 0);
  results = cudf::strings::contains(strings_view, cudf::string_scalar("é"));
  EXPECT_EQ(results->size(), 0);
  results = cudf::strings::starts_with(strings_view, cudf::string_scalar("é"));
  EXPECT_EQ(results->size(), 0);
  results = cudf::strings::ends_with(strings_view, cudf::string_scalar("é"));
  EXPECT_EQ(results->size(), 0);
  results = cudf::strings::starts_with(strings_view, strings_view);
  EXPECT_EQ(results->size(), 0);
  results = cudf::strings::ends_with(strings_view, strings_view);
  EXPECT_EQ(results->size(), 0);
  results = cudf::strings::count(strings_view, cudf::string_scalar("é"));
  EXPECT_EQ(results->size(), 0);
}

TEST_F(StringsFindTest, EmptyTarget)
{
  cudf::test::strings_column_wrapper strings({"Héllo", "thesé", "", "lease", "tést strings", ""},
                                             {true, true, false, true, true, true});
  auto strings_view = cudf::strings_column_view(strings);

  cudf::test::fixed_width_column_wrapper<bool> expected({1, 1, 1, 1, 1, 1},
                                                        {true, true, false, true, true, true});
  auto results = cudf::strings::contains(strings_view, cudf::string_scalar(""));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = cudf::strings::starts_with(strings_view, cudf::string_scalar(""));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = cudf::strings::ends_with(strings_view, cudf::string_scalar(""));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_find(
    {0, 0, 0, 0, 0, 0}, {true, true, false, true, true, true});
  results = cudf::strings::find(strings_view, cudf::string_scalar(""));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_find);

  auto expected_rfind = cudf::strings::count_characters(strings_view);
  results             = cudf::strings::rfind(strings_view, cudf::string_scalar(""));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, *expected_rfind);

  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_count(
    {0, 0, 0, 0, 0, 0}, {true, true, false, true, true, true});
  results = cudf::strings::count(strings_view, cudf::string_scalar(""));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_count);
}

TEST_F(StringsFindTest, AllEmpty)
{
  std::vector<std::string> h_strings{"", "", "", "", ""};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());

  std::vector<cudf::size_type> h_expected32(h_strings.size(), -1);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected32(h_expected32.begin(),
                                                                     h_expected32.end());

  std::vector<bool> h_expected8(h_strings.size(), false);
  cudf::test::fixed_width_column_wrapper<bool> expected8(h_expected8.begin(), h_expected8.end());

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::find(strings_view, cudf::string_scalar("e"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected32);
  results = cudf::strings::rfind(strings_view, cudf::string_scalar("e"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected32);
  results = cudf::strings::contains(strings_view, cudf::string_scalar("e"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected8);
  results = cudf::strings::starts_with(strings_view, cudf::string_scalar("e"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected8);
  results = cudf::strings::ends_with(strings_view, cudf::string_scalar("e"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected8);
  std::vector<std::string> h_targets{"abc", "e", "fdg", "g", "p"};
  cudf::test::strings_column_wrapper targets(h_targets.begin(), h_targets.end());
  auto targets_view = cudf::strings_column_view(targets);
  results           = cudf::strings::starts_with(strings_view, targets_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected8);
  results = cudf::strings::ends_with(strings_view, targets_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected8);
  results = cudf::strings::find_instance(strings_view, cudf::string_scalar("e"), 0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected32);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_count({0, 0, 0, 0, 0});
  results = cudf::strings::count(strings_view, cudf::string_scalar(""));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_count);
}

TEST_F(StringsFindTest, AllNull)
{
  cudf::test::strings_column_wrapper strings({"", "", "", ""}, cudf::test::iterators::all_nulls());
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_st(
    {0, 0, 0, 0}, cudf::test::iterators::all_nulls());
  cudf::test::fixed_width_column_wrapper<bool> expected_bool({0, 0, 0, 0},
                                                             cudf::test::iterators::all_nulls());

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::find(strings_view, cudf::string_scalar("e"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_st);
  results = cudf::strings::rfind(strings_view, cudf::string_scalar("e"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_st);
  results = cudf::strings::contains(strings_view, cudf::string_scalar("e"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_bool);
  results = cudf::strings::starts_with(strings_view, cudf::string_scalar("e"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_bool);
  results = cudf::strings::ends_with(strings_view, cudf::string_scalar("e"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_bool);
  cudf::test::strings_column_wrapper targets({"abc", "e", "fdg", "p"});
  auto targets_view = cudf::strings_column_view(targets);
  results           = cudf::strings::starts_with(strings_view, targets_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_bool);
  results = cudf::strings::ends_with(strings_view, targets_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_bool);
  results = cudf::strings::find_instance(strings_view, cudf::string_scalar("e"), 0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_st);
  results = cudf::strings::count(strings_view, cudf::string_scalar(""));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_st);
}

TEST_F(StringsFindTest, ErrorCheck)
{
  cudf::test::strings_column_wrapper strings({"1", "2", "3", "4", "5", "6"});
  auto strings_view = cudf::strings_column_view(strings);
  cudf::test::strings_column_wrapper targets({"1", "2", "3", "4", "5"});
  auto targets_view = cudf::strings_column_view(targets);

  EXPECT_THROW(cudf::strings::contains(strings_view, targets_view), cudf::logic_error);
  EXPECT_THROW(cudf::strings::starts_with(strings_view, targets_view), cudf::logic_error);
  EXPECT_THROW(cudf::strings::ends_with(strings_view, targets_view), cudf::logic_error);

  EXPECT_THROW(cudf::strings::find(strings_view, cudf::string_scalar(""), 2, 1), cudf::logic_error);
  EXPECT_THROW(cudf::strings::rfind(strings_view, cudf::string_scalar(""), 2, 1),
               cudf::logic_error);
  EXPECT_THROW(cudf::strings::find(strings_view, targets_view), cudf::logic_error);
  EXPECT_THROW(cudf::strings::find(strings_view, strings_view, -1), cudf::logic_error);

  auto invalid_str = cudf::string_scalar("", false);
  auto valid_str   = cudf::string_scalar("1");
  EXPECT_THROW(cudf::strings::find_instance(strings_view, invalid_str, 0), std::invalid_argument);
  EXPECT_THROW(cudf::strings::find_instance(strings_view, valid_str, -1), std::invalid_argument);
  EXPECT_THROW(cudf::strings::count(strings_view, invalid_str), std::invalid_argument);
}

class FindParmsTest : public StringsFindTest,
                      public testing::WithParamInterface<cudf::size_type> {};

TEST_P(FindParmsTest, Find)
{
  std::vector<std::string> h_strings{"hello", "", "these", "are stl", "safe"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  cudf::size_type position = GetParam();

  auto strings_view = cudf::strings_column_view(strings);
  {
    auto results = cudf::strings::find(strings_view, cudf::string_scalar("e"), position);
    std::vector<cudf::size_type> h_expected;
    for (auto& h_string : h_strings)
      h_expected.push_back(static_cast<cudf::size_type>(h_string.find("e", position)));
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(h_expected.begin(),
                                                                     h_expected.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::rfind(strings_view, cudf::string_scalar("e"), 0, position + 1);
    std::vector<cudf::size_type> h_expected;
    for (auto& h_string : h_strings)
      h_expected.push_back(static_cast<cudf::size_type>(h_string.rfind("e", position)));
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(h_expected.begin(),
                                                                     h_expected.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto begin   = static_cast<cudf::size_type>(position);
    auto results = cudf::strings::find(strings_view, cudf::string_scalar(""), begin);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(
      {begin, (begin > 0 ? -1 : 0), begin, begin, begin});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto end = static_cast<cudf::size_type>(position + 1);
    results  = cudf::strings::rfind(strings_view, cudf::string_scalar(""), 0, end);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> rexpected({end, 0, end, end, end});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, rexpected);
  }
  {
    std::vector<std::string> h_targets({"l", "", "", "l", "s"});
    std::vector<cudf::size_type> h_expected;
    for (std::size_t i = 0; i < h_strings.size(); ++i)
      h_expected.push_back(static_cast<cudf::size_type>(h_strings[i].find(h_targets[i], position)));
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(h_expected.begin(),
                                                                     h_expected.end());
    cudf::test::strings_column_wrapper targets(h_targets.begin(), h_targets.end());
    auto results = cudf::strings::find(strings_view, cudf::strings_column_view(targets), position);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

INSTANTIATE_TEST_CASE_P(StringsFindTest,
                        FindParmsTest,
                        testing::ValuesIn(std::array<cudf::size_type, 4>{0, 1, 2, 3}));

TEST_F(StringsFindTest, FindInstance)
{
  auto validity = cudf::test::iterators::null_at(4);
  auto input    = cudf::test::strings_column_wrapper(
    {"thésé", "yellellellellellellellellellellellellellello", "eeeee", "", "", "ééééé"}, validity);
  auto sv = cudf::strings_column_view(input);

  using find_col = cudf::test::fixed_width_column_wrapper<cudf::size_type>;
  auto none      = find_col({-1, -1, -1, -1, -1, -1}, validity);

  auto just_e     = cudf::string_scalar("e");
  auto expect_col = cudf::strings::find(sv, just_e);
  auto results    = cudf::strings::find_instance(sv, just_e, 0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, *expect_col);
  auto expected = find_col({-1, 4, 1, -1, -1, -1}, validity);
  results       = cudf::strings::find_instance(sv, just_e, 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  expected = find_col({-1, 7, 2, -1, -1, -1}, validity);
  results  = cudf::strings::find_instance(sv, just_e, 2);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  expected = find_col({-1, 10, 3, -1, -1, -1}, validity);
  results  = cudf::strings::find_instance(sv, just_e, 3);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  expected = find_col({-1, 13, 4, -1, -1, -1}, validity);
  results  = cudf::strings::find_instance(sv, just_e, 4);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto fancy_e = cudf::string_scalar("é");
  expect_col   = cudf::strings::find(sv, fancy_e);
  results      = cudf::strings::find_instance(sv, fancy_e, 0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, *expect_col);
  expected = find_col({4, -1, -1, -1, -1, 1}, validity);
  results  = cudf::strings::find_instance(sv, fancy_e, 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  expected = find_col({-1, -1, -1, -1, -1, 2}, validity);
  results  = cudf::strings::find_instance(sv, fancy_e, 2);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  expected = find_col({-1, -1, -1, -1, -1, 4}, validity);
  results  = cudf::strings::find_instance(sv, fancy_e, 4);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = cudf::strings::find_instance(sv, fancy_e, 5);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, none);

  auto target = cudf::string_scalar("elle");
  expect_col  = cudf::strings::find(sv, target);
  results     = cudf::strings::find_instance(sv, target, 0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, *expect_col);
  expected = find_col({-1, 4, -1, -1, -1, -1}, validity);
  results  = cudf::strings::find_instance(sv, target, 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  expected = find_col({-1, 7, -1, -1, -1, -1}, validity);
  results  = cudf::strings::find_instance(sv, target, 2);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  expected = find_col({-1, 31, -1, -1, -1, -1}, validity);
  results  = cudf::strings::find_instance(sv, target, 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  expected = find_col({-1, 34, -1, -1, -1, -1}, validity);
  results  = cudf::strings::find_instance(sv, target, 11);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = cudf::strings::find_instance(sv, target, 14);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, none);

  auto fancy_es = cudf::string_scalar("éé");
  expect_col    = cudf::strings::find(sv, fancy_es);
  results       = cudf::strings::find_instance(sv, fancy_es, 0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, *expect_col);
  expected = find_col({-1, -1, -1, -1, -1, 1}, validity);
  results  = cudf::strings::find_instance(sv, fancy_es, 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  expected = find_col({-1, -1, -1, -1, -1, 2}, validity);
  results  = cudf::strings::find_instance(sv, fancy_es, 2);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  expected = find_col({-1, -1, -1, -1, -1, 3}, validity);
  results  = cudf::strings::find_instance(sv, fancy_es, 3);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = cudf::strings::find_instance(sv, fancy_es, 4);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, none);
}
