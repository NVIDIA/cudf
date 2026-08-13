/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/null_mask.hpp>

#include <cuda/std/limits>

#include <algorithm>
#include <array>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace cudf::test::iterators;

template <typename T>
using LCW = cudf::test::lists_column_wrapper<T, int32_t>;

auto generate_sorted_lists(cudf::lists_column_view const& input,
                           cudf::order column_order,
                           cudf::null_order null_precedence)
{
  return std::pair{cudf::lists::sort_lists(input, column_order, null_precedence),
                   cudf::lists::stable_sort_lists(input, column_order, null_precedence)};
}

template <typename T>
struct SortLists : public cudf::test::BaseFixture {};

using TypesForTest = cudf::test::Concat<cudf::test::NumericTypes, cudf::test::FixedPointTypes>;
TYPED_TEST_SUITE(SortLists, TypesForTest);

TYPED_TEST(SortLists, NoNull)
{
  using T = TypeParam;

  // List<T>
  LCW<T> list{{3, 2, 1, 4}, {5}, {10, 8, 9}, {6, 7}};

  // Ascending
  // LCW<int>  order{{2, 1, 0, 3}, {0}, {1, 2, 0},  {0, 1}};
  LCW<T> expected{{1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}};
  {
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::ASCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
  {
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::ASCENDING, cudf::null_order::BEFORE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  // Descending
  // LCW<int>  order{{3, 0, 1, 2}, {0}, {0, 1, 2},  {1, 0}};
  LCW<T> expected2{{4, 3, 2, 1}, {5}, {10, 9, 8}, {7, 6}};
  {
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::DESCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected2);
  }
  {
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::DESCENDING, cudf::null_order::BEFORE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected2);
  }
}

TYPED_TEST(SortLists, Null)
{
  using T = TypeParam;
  if (std::is_same_v<T, bool>) return;
  std::vector<bool> valids_o{true, true, false, true};
  std::vector<bool> valids_a{true, true, true, false};
  std::vector<bool> valids_b{false, true, true, true};

  // List<T>
  LCW<T> list{{{3, 2, 4, 1}, valids_o.begin()}, {5}, {10, 8, 9}, {6, 7}};
  // LCW<int>  order{{2, 1, 3, 0}, {0}, {1, 2, 0},  {0, 1}};

  {
    LCW<T> expected{{{1, 2, 3, 4}, valids_a.begin()}, {5}, {8, 9, 10}, {6, 7}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::ASCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    LCW<T> expected{{{4, 1, 2, 3}, valids_b.begin()}, {5}, {8, 9, 10}, {6, 7}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::ASCENDING, cudf::null_order::BEFORE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  // Descending
  // LCW<int>  order{{3, 0, 1, 2}, {0}, {0, 1, 2},  {1, 0}};
  {
    LCW<T> expected{{{4, 3, 2, 1}, valids_b.begin()}, {5}, {10, 9, 8}, {7, 6}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::DESCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    LCW<T> expected{{{3, 2, 1, 4}, valids_a.begin()}, {5}, {10, 9, 8}, {7, 6}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::DESCENDING, cudf::null_order::BEFORE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
}

using SortListsInt = SortLists<int>;

TEST_F(SortListsInt, Empty)
{
  using T = int;

  {
    LCW<T> l{};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
  {
    LCW<T> l{LCW<T>{}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
  {
    LCW<T> l{LCW<T>{}, LCW<T>{}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
}

TEST_F(SortListsInt, Single)
{
  using T = int;

  {
    LCW<T> l{1};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
  {
    LCW<T> l{{1, 2, 3}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
}

TEST_F(SortListsInt, NullRows)
{
  using T = int;
  std::vector<int> valids{0, 1, 0};
  LCW<T> l{{{1, 2, 3}, {4, 5, 6}, {7}}, valids.begin()};  // offset 0, 0, 3, 3

  auto const [sorted_lists, stable_sorted_lists] =
    generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
}

// A nullable but null-free input must come back non-nullable: the sort cannot add nulls, so
// neither the returned parent nor its child may keep an allocated all-valid mask.
TEST_F(SortListsInt, NullableWithZeroNullsDropsMasks)
{
  using T = int;
  cudf::test::fixed_width_column_wrapper<T> child({3, 1, 2, 5, 4}, no_nulls());
  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, 3, 5};
  auto const input =
    cudf::make_lists_column(2,
                            offsets.release(),
                            child.release(),
                            0,
                            cudf::create_null_mask(2, cudf::mask_state::ALL_VALID));
  ASSERT_TRUE(input->view().nullable());
  ASSERT_TRUE(cudf::lists_column_view{input->view()}.child().nullable());

  LCW<T> expected{{1, 2, 3}, {4, 5}};
  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input->view()}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  EXPECT_FALSE(sorted_lists->view().nullable());
  EXPECT_FALSE(cudf::lists_column_view{sorted_lists->view()}.child().nullable());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  EXPECT_FALSE(stable_sorted_lists->view().nullable());
  EXPECT_FALSE(cudf::lists_column_view{stable_sorted_lists->view()}.child().nullable());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

using SortListsString = SortLists<cudf::string_view>;

// Default order: element nulls sort last within a row; a null list row passes through unchanged.
TEST_F(SortListsString, Strings)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  std::vector<bool> const valids{true, true, false, true};
  StrLCW input{{StrLCW{{"pear", "apple", "fig", "kiwi"}, null_at(2)},
                StrLCW{"banana"},
                StrLCW{"unused"},
                StrLCW{"melon", "cherry"}},
               valids.begin()};

  // Null slots keep arbitrary placeholder values ("fig" here); they are never compared.
  StrLCW expected{{StrLCW{{"apple", "kiwi", "pear", "fig"}, null_at(3)},
                   StrLCW{"banana"},
                   StrLCW{"unused"},
                   StrLCW{"cherry", "melon"}},
                  valids.begin()};

  auto const [sorted_lists, stable_sorted_lists] =
    generate_sorted_lists(cudf::lists_column_view{input}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// The Prefix* tests target the string fast path: a packed leading-bytes key, iterative byte-window
// tie-breaks, then a final comparison cleanup. Each asserts the fast path (sort_lists) and the
// comparison path (stable_sort_lists) against the same expected column.

// Identical first eight bytes force the full-string tie-break; the proper prefix sorts first.
TEST_F(SortListsString, PrefixEqualFirstEightBytes)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{{"abcdefgh1", "abcdefgh0", "abcdefgh"}};
  StrLCW expected{{"abcdefgh", "abcdefgh0", "abcdefgh1"}};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// Under eight bytes, zero-padding lets the packed key alone order a proper prefix first.
TEST_F(SortListsString, PrefixSubstringShorterFirst)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{{"aaa", "aa", "a"}};
  StrLCW expected{{"a", "aa", "aaa"}};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// Empty strings pack an all-zero key and must sort before any non-empty string.
TEST_F(SortListsString, PrefixEmptyStringsFirst)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{{"banana", "", "apple", "", "fig"}};
  StrLCW expected{{"", "", "apple", "banana", "fig"}};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// Null slots' placeholder values are never compared.
TEST_F(SortListsString, PrefixMultipleNullsLast)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{StrLCW{{"pear", "apple", "fig", "kiwi", "lime"}, nulls_at({1, 3})}};
  StrLCW expected{StrLCW{{"fig", "lime", "pear", "apple", "kiwi"}, nulls_at({3, 4})}};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

TEST_F(SortListsString, PrefixAllNullListRow)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  std::vector<bool> const valids{true, false, true};
  StrLCW input{{StrLCW{"cherry", "apple", "banana"}, StrLCW{"unused"}, StrLCW{"date", "egg"}},
               valids.begin()};
  StrLCW expected{{StrLCW{"apple", "banana", "cherry"}, StrLCW{"unused"}, StrLCW{"date", "egg"}},
                  valids.begin()};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// One shared eight-byte prefix makes the whole row a single tie run resolved wholly by full-string
// comparison -- the tie-break's worst case.
TEST_F(SortListsString, PrefixWholeRowShared)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{{"prefix__zebra", "prefix__apple", "prefix__mango", "prefix__"}};
  StrLCW expected{{"prefix__", "prefix__apple", "prefix__mango", "prefix__zebra"}};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// Raw unsigned-byte order: the two-/three-/four-byte UTF-8 lead bytes (0xC3 < 0xE2 < 0xF0) sort
// after ASCII and by lead-byte class among themselves.
TEST_F(SortListsString, PrefixUtf8Multibyte)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{
    {"\xE2\x82\xAC\x75\x72\x6F", "zoo", "\xF0\x9F\x98\x80", "\xC3\xA9\x70\xC3\xA9\x65", "apple"}};
  StrLCW expected{
    {"apple", "zoo", "\xC3\xA9\x70\xC3\xA9\x65", "\xE2\x82\xAC\x75\x72\x6F", "\xF0\x9F\x98\x80"}};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// The fast path folds every explicit (order, null_order) into its keys; these pin the two combos
// the ascending/nulls-after tests never touch.
TEST_F(SortListsString, DescendingAndNullsBeforeFastPath)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  {  // DESCENDING, no nulls
    StrLCW input{{"banana", "apple", "cherry", "date"}};
    StrLCW expected{{"date", "cherry", "banana", "apple"}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{input}, cudf::order::DESCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
  {  // ASCENDING, nulls BEFORE: the null moves to the front
    StrLCW input{StrLCW{{"banana", "x", "apple"}, null_at(1)}};
    StrLCW expected{StrLCW{{"x", "apple", "banana"}, null_at(0)}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::BEFORE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
}

// Per-row prefix ties must resolve without crossing row boundaries (the segment_id key field); the
// empty row checks dense segment-ordinal labeling skips a zero-length segment without misaligning
// its neighbors.
TEST_F(SortListsString, PrefixMultipleSegments)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{StrLCW{"commonpx_b", "commonpx_a", "commonpx_c"},
               StrLCW{},
               StrLCW{"melon", "apple", "cherry"},
               StrLCW{"sharedpx2", "sharedpx0", "sharedpx1"}};
  StrLCW expected{StrLCW{"commonpx_a", "commonpx_b", "commonpx_c"},
                  StrLCW{},
                  StrLCW{"apple", "cherry", "melon"},
                  StrLCW{"sharedpx0", "sharedpx1", "sharedpx2"}};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

namespace {
// Asserts `sort_lists` (fast path) and `stable_sort_lists` (comparison path) both equal
// `expected`, so every expectation is also re-checked against the comparison semantics.
void expect_both_sort_paths_match(cudf::lists_column_view const& input,
                                  cudf::column_view const& expected,
                                  cudf::order column_order         = cudf::order::ASCENDING,
                                  cudf::null_order null_precedence = cudf::null_order::AFTER)
{
  auto const sorted = cudf::lists::sort_lists(input, column_order, null_precedence);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->view(), expected);
  auto const stable_sorted = cudf::lists::stable_sort_lists(input, column_order, null_precedence);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted->view(), expected);
}

// EQUIVALENT variant for floating point: -0.0/+0.0 and distinct NaN bit patterns are
// interchangeable under the unstable contract, so per-slot bytes may differ within such groups.
void expect_both_sort_paths_equivalent(cudf::lists_column_view const& input,
                                       cudf::column_view const& expected,
                                       cudf::order column_order         = cudf::order::ASCENDING,
                                       cudf::null_order null_precedence = cudf::null_order::AFTER)
{
  auto const sorted = cudf::lists::sort_lists(input, column_order, null_precedence);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted->view(), expected);
  auto const stable_sorted = cudf::lists::stable_sort_lists(input, column_order, null_precedence);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(stable_sorted->view(), expected);
}

// The three explicit combos beyond the shipped ascending / nulls-after configuration.
constexpr std::array<std::pair<cudf::order, cudf::null_order>, 3> non_default_combos{
  {{cudf::order::DESCENDING, cudf::null_order::AFTER},
   {cudf::order::ASCENDING, cudf::null_order::BEFORE},
   {cudf::order::DESCENDING, cudf::null_order::BEFORE}}};

// Host oracle for one list row. cudf's null_order is comparator-level and DESCENDING swaps the
// comparison operands, inverting null placement too: nulls land first iff (BEFORE) != (DESCENDING).
template <typename T>
std::pair<std::vector<T>, std::vector<bool>> host_sorted_row(std::vector<T> const& vals,
                                                             std::vector<bool> const& valids,
                                                             cudf::order column_order,
                                                             cudf::null_order null_precedence)
{
  std::vector<T> nn;
  for (std::size_t i = 0; i < vals.size(); ++i) {
    if (valids[i]) { nn.push_back(vals[i]); }
  }
  std::sort(nn.begin(), nn.end());
  if (column_order == cudf::order::DESCENDING) { std::reverse(nn.begin(), nn.end()); }
  auto const num_nulls = vals.size() - nn.size();
  auto const nulls_first =
    (null_precedence == cudf::null_order::BEFORE) != (column_order == cudf::order::DESCENDING);
  std::vector<T> out;
  std::vector<bool> out_valids;
  auto const push_nulls = [&] {
    for (std::size_t k = 0; k < num_nulls; ++k) {
      out.push_back(T{});
      out_valids.push_back(false);
    }
  };
  if (nulls_first) { push_nulls(); }
  for (auto const& v : nn) {
    out.push_back(v);
    out_valids.push_back(true);
  }
  if (not nulls_first) { push_nulls(); }
  return {std::move(out), std::move(out_valids)};
}

// Wraps a leaf column as a one-row LIST, so the sort orders within a single segment.
std::unique_ptr<cudf::column> as_single_row_list(std::unique_ptr<cudf::column> leaf)
{
  auto const n = static_cast<cudf::size_type>(leaf->size());
  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, n};
  return cudf::make_lists_column(1, offsets.release(), std::move(leaf), 0, {});
}

// One column straddling every block-tier band seam, checked against an independent host sort. The
// 400 single-element filler rows pull the average list size to ~10 -- below the packed-radix cutoff
// and above the int64 tiny-average pocket -- so a no-null column still routes to the tiered engine.
template <typename T>
void check_block_tier_boundaries(bool with_nulls, cudf::order column_order)
{
  // Both sides of every seam: 63/64/65 (warp cap), 128/129, 256/257, 512/513 (the graduated block
  // bands), 1024/1025 (block cap -> radix spill).
  std::vector<cudf::size_type> const sizes{63, 64, 65, 128, 129, 256, 257, 512, 513, 1'024, 1'025};
  auto const gen = [](cudf::size_type s, cudf::size_type i) -> T {
    if constexpr (std::is_same_v<T, __int128_t>) {
      __int128_t k = 1;
      for (int j = 0; j < 30; ++j) {
        k *= 10;
      }
      return static_cast<__int128_t>(s / 2 - i) * k;  // mixed sign, into the high value word
    } else if constexpr (std::is_same_v<T, int64_t>) {
      return static_cast<int64_t>(s - i) * int64_t{5'000'000'011};  // beyond the 32-bit range
    } else if constexpr (std::is_floating_point_v<T>) {
      return static_cast<T>(s / 2 - i) * T{1.5};  // mixed sign, exactly representable
    } else {
      return static_cast<T>(s - i);
    }
  };
  bool const descending = column_order == cudf::order::DESCENDING;
  std::vector<T> in_vals;
  std::vector<bool> in_valids;
  std::vector<T> ex_vals;
  std::vector<bool> ex_valids;
  std::vector<cudf::size_type> offsets{0};
  auto const push_row = [&](std::vector<T> const& rv, std::vector<bool> const& rok) {
    std::vector<T> nn;
    for (std::size_t i = 0; i < rv.size(); ++i) {
      in_vals.push_back(rv[i]);
      in_valids.push_back(rok[i]);
      if (rok[i]) { nn.push_back(rv[i]); }
    }
    // Skipping the trivial call sidesteps gcc-14's -O3 -Warray-bounds false positive on the
    // inlined insertion sort's memmove for the single-element filler rows.
    if (nn.size() > 1) {
      std::sort(nn.begin(), nn.end());
      if (descending) { std::reverse(nn.begin(), nn.end()); }
    }
    auto const num_nulls  = rv.size() - nn.size();
    auto const push_nulls = [&] {
      for (std::size_t k = 0; k < num_nulls; ++k) {
        ex_vals.push_back(T{0});
        ex_valids.push_back(false);
      }
    };
    if (descending) { push_nulls(); }
    for (auto const v : nn) {
      ex_vals.push_back(v);
      ex_valids.push_back(true);
    }
    if (not descending) { push_nulls(); }
    offsets.push_back(static_cast<cudf::size_type>(in_vals.size()));
  };
  for (auto const s : sizes) {
    std::vector<T> rv(s);
    std::vector<bool> rok(s, true);
    for (cudf::size_type i = 0; i < s; ++i) {
      rv[i] = gen(s, i);
      if (with_nulls) { rok[i] = (i % 7 != 3); }  // scattered element nulls
    }
    // cuda::std::numeric_limits covers __int128_t in every language mode (std:: only in gnu mode).
    rv[0] = cuda::std::numeric_limits<T>::max();
    rv[1] = cuda::std::numeric_limits<T>::lowest();
    rv[2] = rv[3];  // a duplicate value
    push_row(rv, rok);
  }
  for (int f = 0; f < 400; ++f) {
    push_row(std::vector<T>{gen(1, 0)}, std::vector<bool>{true});
  }
  auto const num_rows  = static_cast<cudf::size_type>(offsets.size() - 1);
  auto const make_list = [&](std::vector<T> const& vals, std::vector<bool> const& valids) {
    cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
    auto leaf = [&]() -> std::unique_ptr<cudf::column> {
      if constexpr (std::is_same_v<T, __int128_t>) {
        auto constexpr scale = numeric::scale_type{0};
        if (with_nulls) {
          return cudf::test::fixed_point_column_wrapper<__int128_t>(
                   vals.begin(), vals.end(), valids.begin(), scale)
            .release();
        }
        return cudf::test::fixed_point_column_wrapper<__int128_t>(vals.begin(), vals.end(), scale)
          .release();
      } else {
        if (with_nulls) {
          return cudf::test::fixed_width_column_wrapper<T>(vals.begin(), vals.end(), valids.begin())
            .release();
        }
        return cudf::test::fixed_width_column_wrapper<T>(vals.begin(), vals.end()).release();
      }
    }();
    return cudf::make_lists_column(num_rows, off.release(), std::move(leaf), 0, {});
  };
  auto const input    = make_list(in_vals, in_valids);
  auto const expected = make_list(ex_vals, ex_valids);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()},
                               expected->view(),
                               column_order,
                               cudf::null_order::AFTER);
}
}  // namespace

// All shorter than the packed key, so the key alone orders them -- including the second-byte
// distinction "aaa" < "ab".
TEST_F(SortListsString, PrefixShortStrings)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{{"aaa", "aa", "a", "ab", "b"}};
  StrLCW expected{{"a", "aa", "aaa", "ab", "b"}};
  expect_both_sort_paths_match(cudf::lists_column_view{input}, expected);
}

// Shared leading bytes make the packed key uninformative, so the tie-break does the real ordering.
TEST_F(SortListsString, PrefixSharedPrefixWithNullAndDuplicate)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{
    StrLCW{{"AAAAAAAAzebra", "AAAAAAAAapple", "AAAAAAAAmango", "unused", "AAAAAAAAapple"},
           null_at(3)},
    StrLCW{"food_pear", "food_kiwi", "food_pear"}};
  StrLCW expected{
    StrLCW{{"AAAAAAAAapple", "AAAAAAAAapple", "AAAAAAAAmango", "AAAAAAAAzebra", "unused"},
           null_at(4)},
    StrLCW{"food_kiwi", "food_pear", "food_pear"}};
  expect_both_sort_paths_match(cudf::lists_column_view{input}, expected);
}

// A naive all-0xFF null sentinel would collide with genuine all-0xFF strings; the packed key's
// dedicated is_null bit must separate them. The 72- and 73-byte 0xFF strings stay tied through
// every byte window and reach the comparison cleanup as one run.
TEST_F(SortListsString, PrefixNullSentinelCollisionFF)
{
  using StrCW = cudf::test::strings_column_wrapper;

  std::string const ff72(72, '\xff');
  std::string const ff73(73, '\xff');

  std::vector<std::string> const in_strings{"apple", ff73, "mango", ff72, ""};
  std::vector<bool> const in_valids{true, true, true, true, false};

  std::vector<std::string> const ex_strings{"apple", "mango", ff72, ff73, ""};
  std::vector<bool> const ex_valids{true, true, true, true, false};

  auto const make_single_row_list = [](std::vector<std::string> const& strs,
                                       std::vector<bool> const& valids) {
    StrCW leaf{strs.begin(), strs.end(), valids.begin()};
    auto const n = static_cast<cudf::size_type>(strs.size());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, n};
    return cudf::make_lists_column(1, offsets.release(), leaf.release(), 0, {});
  };

  auto const input    = make_single_row_list(in_strings, in_valids);
  auto const expected = make_single_row_list(ex_strings, ex_valids);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// The shared prefix outlasts every iterative window, so the whole run reaches the tie-break,
// exceeds TIE_HEAPSORT_THRESHOLD, and takes the heapsort branch; > 64 elements keep the column on
// the prefix path. Adds interleaved nulls to the plain PrefixTrueHeapsortRun shape.
TEST_F(SortListsString, PrefixSharedPrefixHeapsortRun)
{
  std::string const prefix(80, 'A');  // 80 > 7 + 8*8 = 71, so ties survive every iterative window.
  std::vector<std::string> non_null;
  for (int i = 0; i < 70; ++i) {
    non_null.push_back(prefix + std::to_string(i / 10) + std::to_string(i % 10));
  }
  non_null.push_back(prefix + "05");  // exact duplicates
  non_null.push_back(prefix + "17");

  auto shuffled = non_null;
  std::shuffle(shuffled.begin(), shuffled.end(), std::mt19937{0xC0'FFEE});

  auto const null_pos_a = shuffled.size() / 3;
  auto const null_pos_b = (shuffled.size() * 2) / 3 + 1;
  std::vector<std::string> in_strings;
  std::vector<bool> in_valids;
  for (std::size_t i = 0; i <= shuffled.size(); ++i) {
    if (i == null_pos_a || i == null_pos_b) {
      in_strings.emplace_back("");  // null placeholder; never compared
      in_valids.push_back(false);
    }
    if (i < shuffled.size()) {
      in_strings.push_back(shuffled[i]);
      in_valids.push_back(true);
    }
  }

  auto sorted = non_null;
  std::sort(sorted.begin(), sorted.end());
  std::vector<std::string> ex_strings = sorted;
  std::vector<bool> ex_valids(sorted.size(), true);
  ex_strings.emplace_back("");
  ex_valids.push_back(false);
  ex_strings.emplace_back("");
  ex_valids.push_back(false);

  using StrCW                     = cudf::test::strings_column_wrapper;
  auto const make_single_row_list = [](std::vector<std::string> const& strs,
                                       std::vector<bool> const& valids) {
    StrCW leaf{strs.begin(), strs.end(), valids.begin()};
    auto const n = static_cast<cudf::size_type>(strs.size());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, n};
    return cudf::make_lists_column(1, offsets.release(), leaf.release(), 0, {});
  };

  auto const input    = make_single_row_list(in_strings, in_valids);
  auto const expected = make_single_row_list(ex_strings, ex_valids);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// Deep shared prefixes force several windows past the packed key. Row 0 adds a pure length tie (a
// string vs itself plus a trailing NUL) that only the cleanup's shorter-is-less rule separates;
// row 1's thousands of shared-prefix strings exercise the pass cap and the parallel re-sort.
TEST_F(SortListsString, PrefixIterativeDeepLongSharedPrefix)
{
  using StrCW = cudf::test::strings_column_wrapper;

  std::string const prefix0(20, 'A');  // Twenty shared bytes: > 2x the 8-byte key, 5x the 4-byte.
  std::vector<std::string> row0_non_null{
    prefix0 + "zebra",
    prefix0 + "ant",  // shorter tail -> mixed run lengths within the shared run
    prefix0 + "mango",
    prefix0 + "ant",                 // exact duplicate
    prefix0,                         // the shared prefix with no tail
    prefix0 + std::string(1, '\0'),  // `prefix0` zero-extended: a pure length tie
    prefix0 + "mango",               // exact duplicate
    ""};                             // all-zero key, must sort first in the row

  std::string const prefix1(24, 'Q');  // Twenty-four shared bytes -> three windows past an 8B key.
  std::vector<std::string> row1;
  row1.reserve(5'000);
  for (int i = 0; i < 5'000; ++i) {
    auto tail = std::to_string(i);
    tail.insert(tail.begin(), 4 - tail.size(), '0');
    row1.push_back(prefix1 + tail);
  }
  std::shuffle(row1.begin(), row1.end(), std::mt19937{0x5EED});

  std::vector<std::string> in_strings;
  std::vector<bool> in_valids;
  for (auto const& s : row0_non_null) {
    in_strings.push_back(s);
    in_valids.push_back(true);
  }
  in_strings.emplace_back("");  // null placeholder; never compared
  in_valids.push_back(false);
  auto const row0_len = static_cast<cudf::size_type>(in_strings.size());
  for (auto const& s : row1) {
    in_strings.push_back(s);
    in_valids.push_back(true);
  }
  auto const total_len = static_cast<cudf::size_type>(in_strings.size());

  // std::string's raw-byte order and length tie-break match cudf's string order.
  auto row0_sorted = row0_non_null;
  std::sort(row0_sorted.begin(), row0_sorted.end());
  auto row1_sorted = row1;
  std::sort(row1_sorted.begin(), row1_sorted.end());

  std::vector<std::string> ex_strings;
  std::vector<bool> ex_valids;
  for (auto const& s : row0_sorted) {
    ex_strings.push_back(s);
    ex_valids.push_back(true);
  }
  ex_strings.emplace_back("");
  ex_valids.push_back(false);
  for (auto const& s : row1_sorted) {
    ex_strings.push_back(s);
    ex_valids.push_back(true);
  }

  auto const make_two_row_list = [&](std::vector<std::string> const& strs,
                                     std::vector<bool> const& valids) {
    StrCW leaf{strs.begin(), strs.end(), valids.begin()};
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, row0_len, total_len};
    return cudf::make_lists_column(2, offsets.release(), leaf.release(), 0, {});
  };

  auto const input    = make_two_row_list(in_strings, in_valids);
  auto const expected = make_two_row_list(ex_strings, ex_valids);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// A run of byte-identical strings becomes length-uniform and exhausted once the windows cover its
// length, so the drop freezes it early instead of dragging it to the pass cap and cleanup.
TEST_F(SortListsString, PrefixExhaustedIdenticalRunDropped)
{
  using StrCW = cudf::test::strings_column_wrapper;
  std::vector<std::string> in;
  for (int i = 0; i < 40; ++i) {
    in.emplace_back("abcdefghijkl");  // a byte-identical run
  }
  in.emplace_back("abcdefghijkZ");  // singleton: differs only in the last byte
  in.emplace_back("abcdefgh");      // shorter proper prefix: its own run, sorts first

  auto expected = in;
  std::sort(expected.begin(), expected.end());

  auto const input        = as_single_row_list(StrCW{in.begin(), in.end()}.release());
  auto const expected_col = as_single_row_list(StrCW{expected.begin(), expected.end()}.release());
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected_col->view());
}

// A string and its zero-extension collide in every window, so the stable radix keeps arrival
// order. Placing the LONGER first makes that order wrong lexicographically; an exhaustion-only
// drop would freeze it, so the length-uniform guard must leave the run for the shorter-first
// cleanup.
TEST_F(SortListsString, PrefixZeroExtensionLongerFirstNotDropped)
{
  using StrCW = cudf::test::strings_column_wrapper;
  std::string const base(20, 'A');
  std::string const shorter = base;
  std::string const longer  = base + std::string(1, '\0');

  // base+"B" / base+"C" keep the run non-trivial; they resolve as singletons.
  std::vector<std::string> in{longer, shorter, base + "B", base + "C"};
  auto expected = in;
  std::sort(expected.begin(), expected.end());  // shorter < longer < base+"B" < base+"C".

  auto const input        = as_single_row_list(StrCW{in.begin(), in.end()}.release());
  auto const expected_col = as_single_row_list(StrCW{expected.begin(), expected.end()}.release());
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected_col->view());
}

// The singleton compaction must extract only the deep row's still-tied elements, re-sort just
// those, and scatter them back without disturbing the first-pass-resolved rows around it.
TEST_F(SortListsString, PrefixCompactionMixedSingletonAndDeepRows)
{
  using StrCW = cudf::test::strings_column_wrapper;

  std::vector<std::vector<std::string>> rows;
  // Every element differs in its first byte, so these rows resolve in the first pass.
  for (int r = 0; r < 12; ++r) {
    rows.push_back({std::string(1, static_cast<char>('a' + r)) + "_zzz",
                    std::string(1, static_cast<char>('A' + r)) + "_aaa",
                    std::string(1, static_cast<char>('0' + r)) + "_mmm"});
  }
  // The only row the compaction should re-sort: a 32-byte shared prefix, four windows past the key.
  std::string const deep_prefix(32, 'D');
  std::vector<std::string> deep;
  deep.reserve(1'500);
  for (int i = 0; i < 1'500; ++i) {
    auto tail = std::to_string(i);
    tail.insert(tail.begin(), 4 - tail.size(), '0');
    deep.push_back(deep_prefix + tail);
  }
  std::shuffle(deep.begin(), deep.end(), std::mt19937{0xD00D});
  // Place the deep row in the middle so resolved rows sit on both sides of the compacted block.
  rows.insert(rows.begin() + 6, deep);

  std::vector<std::string> in_strings;
  std::vector<std::string> ex_strings;
  std::vector<cudf::size_type> offsets{0};
  for (auto const& row : rows) {
    for (auto const& s : row) {
      in_strings.push_back(s);
    }
    auto sorted = row;
    std::sort(sorted.begin(), sorted.end());
    for (auto const& s : sorted) {
      ex_strings.push_back(s);
    }
    offsets.push_back(static_cast<cudf::size_type>(in_strings.size()));
  }

  auto const num_rows  = static_cast<cudf::size_type>(rows.size());
  auto const make_list = [&](std::vector<std::string> const& strs) {
    StrCW leaf{strs.begin(), strs.end()};
    cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
    return cudf::make_lists_column(num_rows, off.release(), leaf.release(), 0, {});
  };

  auto const input    = make_list(in_strings);
  auto const expected = make_list(ex_strings);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// With one segment (S = bit_width(1) = 1) and nulls, the packed key holds P = 64 - 1 - 1 = 62
// prefix bits -- not a byte multiple -- so it proves 7 whole bytes plus byte 7's top six bits.
// The probe strings differ only in byte 7's low two bits and therefore tie on the key; the
// tie-break must resume its windows at byte 7 (a stale fixed eight-byte offset would skip past
// both eight-byte strings and leave them arbitrarily ordered).
TEST_F(SortListsString, PrefixPackedKeyPartialByteTieOffset)
{
  using StrCW = cudf::test::strings_column_wrapper;

  std::string const probe_lo = std::string("AAAAAA") + '\0' + '\x01';  // 8 bytes, byte 7 = 0x01.
  std::string const probe_hi = std::string("AAAAAA") + '\0' + '\x02';  // 8 bytes, byte 7 = 0x02.

  // The "B" fillers resolve in the first pass and never join the probe pair's run; the two nulls
  // force the key's null bit (P = 62, not 63).
  std::vector<std::string> in_strings;
  std::vector<bool> in_valids;
  in_strings.push_back(probe_hi);  // out of order, so the tie-break must reorder the pair
  in_valids.push_back(true);
  in_strings.push_back(probe_lo);
  in_valids.push_back(true);
  in_strings.emplace_back("");  // null placeholder; never compared
  in_valids.push_back(false);
  for (int i = 0; in_strings.size() < 199; ++i) {
    auto tail = std::to_string(i);
    tail.insert(tail.begin(), 4 - tail.size(), '0');
    in_strings.push_back("B" + tail);
    in_valids.push_back(true);
  }
  in_strings.emplace_back("");  // second null placeholder
  in_valids.push_back(false);
  auto const num_elements = static_cast<cudf::size_type>(in_strings.size());

  std::vector<std::string> non_null;
  for (std::size_t i = 0; i < in_strings.size(); ++i) {
    if (in_valids[i]) { non_null.push_back(in_strings[i]); }
  }
  std::sort(non_null.begin(), non_null.end());
  std::vector<std::string> ex_strings = non_null;
  std::vector<bool> ex_valids(non_null.size(), true);
  ex_strings.emplace_back("");
  ex_valids.push_back(false);
  ex_strings.emplace_back("");
  ex_valids.push_back(false);

  auto const make_single_row_list = [](std::vector<std::string> const& strs,
                                       std::vector<bool> const& valids) {
    StrCW leaf{strs.begin(), strs.end(), valids.begin()};
    auto const n = static_cast<cudf::size_type>(strs.size());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, n};
    return cudf::make_lists_column(1, offsets.release(), leaf.release(), 0, {});
  };

  auto const input    = make_single_row_list(in_strings, in_valids);
  auto const expected = make_single_row_list(ex_strings, ex_valids);
  // The count only keeps the probe pair amid a bulk of first-pass singletons; one row fixes S = 1.
  EXPECT_EQ(num_elements, 200);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// Nulls are position-final after the first pass and excluded from the tie-break set; adjacent
// nulls beside genuine tie runs verify the exclusion never misplaces them.
TEST_F(SortListsString, PrefixNullRunsNeverTied)
{
  using StrCW = cudf::test::strings_column_wrapper;

  std::vector<std::string> const in_strings{
    "AAAAAAAA3", "", "", "", "AAAAAAAA1", "AAAAAAAA2", "zebra", "apple", "apple"};
  std::vector<bool> const in_valids{true, false, false, false, true, true, true, true, true};

  // 'A' (0x41) < 'a' (0x61) < 'z' (0x7A) by raw byte; the three nulls collect last.
  std::vector<std::string> const ex_strings{
    "AAAAAAAA1", "AAAAAAAA2", "AAAAAAAA3", "apple", "apple", "zebra", "", "", ""};
  std::vector<bool> const ex_valids{true, true, true, true, true, true, false, false, false};

  auto const make_single_row_list = [](std::vector<std::string> const& strs,
                                       std::vector<bool> const& valids) {
    StrCW leaf{strs.begin(), strs.end(), valids.begin()};
    auto const n = static_cast<cudf::size_type>(strs.size());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, n};
    return cudf::make_lists_column(1, offsets.release(), leaf.release(), 0, {});
  };

  auto const input    = make_single_row_list(in_strings, in_valids);
  auto const expected = make_single_row_list(ex_strings, ex_valids);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// A whole segment of exact duplicates collapses to one segment-spanning tie run beside ordinary
// rows.
TEST_F(SortListsString, PrefixWholeSegmentDuplicate)
{
  using StrCW = cudf::test::strings_column_wrapper;

  std::vector<std::string> const dup_row(10, "duplicate");
  std::vector<std::vector<std::string>> const rows{
    {"cherry", "apple", "banana"}, dup_row, {"melon", "date", "fig"}};

  std::vector<std::string> in_strings;
  std::vector<std::string> ex_strings;
  std::vector<cudf::size_type> offsets{0};
  for (auto const& row : rows) {
    for (auto const& s : row) {
      in_strings.push_back(s);
    }
    auto sorted = row;
    std::sort(sorted.begin(), sorted.end());
    for (auto const& s : sorted) {
      ex_strings.push_back(s);
    }
    offsets.push_back(static_cast<cudf::size_type>(in_strings.size()));
  }

  auto const num_rows  = static_cast<cudf::size_type>(rows.size());
  auto const make_list = [&](std::vector<std::string> const& strs) {
    StrCW leaf{strs.begin(), strs.end()};
    cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
    return cudf::make_lists_column(num_rows, off.release(), leaf.release(), 0, {});
  };

  auto const input    = make_list(in_strings);
  auto const expected = make_list(ex_strings);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// The comparison cleanup's heapsort branch: the first pass proves 7 whole bytes (P = 63) and the
// 8 windows of 8 bytes reach byte 7 + 64 = 71, so an 80-byte shared prefix survives every window
// and the run (>= TIE_HEAPSORT_THRESHOLD) is resolved only by the cleanup.
TEST_F(SortListsString, PrefixTrueHeapsortRun)
{
  using StrCW = cudf::test::strings_column_wrapper;

  std::string const prefix(80, 'Z');  // 80 > 7 + 8*8 = 71, so ties survive every iterative window.
  std::vector<std::string> non_null;
  // > 64 elements rejects the graduated-warp path, keeping the column on the prefix path.
  for (int i = 0; i < 70; ++i) {
    non_null.push_back(prefix + std::to_string(i / 10) + std::to_string(i % 10));
  }
  non_null.push_back(prefix + "05");  // exact duplicates
  non_null.push_back(prefix + "17");

  auto shuffled = non_null;
  std::shuffle(shuffled.begin(), shuffled.end(), std::mt19937{0x8EA7});

  auto sorted = non_null;
  std::sort(sorted.begin(), sorted.end());

  auto const make_single_row_list = [](std::vector<std::string> const& strs) {
    StrCW leaf{strs.begin(), strs.end()};
    auto const n = static_cast<cudf::size_type>(strs.size());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, n};
    return cudf::make_lists_column(1, offsets.release(), leaf.release(), 0, {});
  };

  auto const input    = make_single_row_list(shuffled);
  auto const expected = make_single_row_list(sorted);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

namespace {
// Builds a LIST<STRING> column from per-row (string, validity) data. Embedded NUL bytes ride
// through unchanged, so callers pass explicit-length `std::string`s. The leaf takes a mask only
// when a null exists: the sort returns null-free columns non-nullable, and expected columns built
// here must match that contract.
std::unique_ptr<cudf::column> make_string_lists(std::vector<std::vector<std::string>> const& rows,
                                                std::vector<std::vector<bool>> const& valids)
{
  std::vector<std::string> flat;
  std::vector<bool> flat_v;
  std::vector<cudf::size_type> offsets{0};
  for (std::size_t r = 0; r < rows.size(); ++r) {
    flat.insert(flat.end(), rows[r].begin(), rows[r].end());
    flat_v.insert(flat_v.end(), valids[r].begin(), valids[r].end());
    offsets.push_back(static_cast<cudf::size_type>(flat.size()));
  }
  auto const has_null_element = std::find(flat_v.cbegin(), flat_v.cend(), false) != flat_v.cend();
  auto leaf =
    has_null_element
      ? cudf::test::strings_column_wrapper(flat.begin(), flat.end(), flat_v.begin()).release()
      : cudf::test::strings_column_wrapper(flat.begin(), flat.end()).release();
  cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
  return cudf::make_lists_column(
    static_cast<cudf::size_type>(rows.size()), off.release(), std::move(leaf), 0, {});
}

// Drives the rows through all four (order, null_order) combos against a per-row host oracle;
// `std::string`'s unsigned-byte order and length tie-break match cudf's string order.
void expect_string_polarity_matrix(std::vector<std::vector<std::string>> const& rows,
                                   std::vector<std::vector<bool>> const& valids)
{
  constexpr std::array<std::pair<cudf::order, cudf::null_order>, 4> combos{
    {{cudf::order::ASCENDING, cudf::null_order::AFTER},
     {cudf::order::DESCENDING, cudf::null_order::AFTER},
     {cudf::order::ASCENDING, cudf::null_order::BEFORE},
     {cudf::order::DESCENDING, cudf::null_order::BEFORE}}};
  auto const input = make_string_lists(rows, valids);
  for (auto const& [ord, np] : combos) {
    std::vector<std::vector<std::string>> ex_rows(rows.size());
    std::vector<std::vector<bool>> ex_valids(rows.size());
    for (std::size_t r = 0; r < rows.size(); ++r) {
      auto sorted_row = host_sorted_row(rows[r], valids[r], ord, np);
      ex_rows[r]      = std::move(sorted_row.first);
      ex_valids[r]    = std::move(sorted_row.second);
    }
    auto const expected = make_string_lists(ex_rows, ex_valids);
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view(), ord, np);
  }
}
}  // namespace

// ============= full (order, null_order) polarity matrix over the string prefix path =============
// Each test drives one facet of the key packing through all four combos.

// Lengths 0,1,7,8,9,15,16,17 straddle the packed-key prefix boundary (~7 whole bytes) and the
// window boundaries at bytes 8 and 16; descending exercises the byte complement across all three.
TEST_F(SortListsString, PrefixPolarityMatrixWindowBoundaries)
{
  std::vector<std::string> const row{"",
                                     "z",
                                     "abcdefg",             // 7: exhausts at the first window
                                     "abcdefgh",            // 8: byte-7 divergence vs "abcdefg"
                                     "abcdefgX",            // 8: byte-7 divergence ('X' < 'h')
                                     "PQRSTUVW1",           // 9: shares 8 bytes, diverges at byte 8
                                     "PQRSTUVW2",           // 9
                                     "0123456789ABCDE",     // 15
                                     "0123456789ABCDEF",    // 16
                                     "0123456789ABCDEFg",   // 17: byte-16 divergence, 2nd window
                                     "0123456789ABCDEFh"};  // 17
  expect_string_polarity_matrix({row}, {std::vector<bool>(row.size(), true)});
}

// Exhausted-window zero-fill under the descending complement: an exhausted window packs to zero,
// which descending sends to the maximum, so proper prefixes sort last; the "a" / "a\0"
// zero-extension pair flips with direction only via the cleanup's length tie-break.
TEST_F(SortListsString, PrefixPolarityMatrixExhaustedAndEmbeddedNul)
{
  std::vector<std::string> const row{"",
                                     "a",
                                     std::string("a\0", 2),
                                     std::string("a\0b", 3),
                                     "ab",
                                     "abc",
                                     "abcd",
                                     "b",
                                     "a"};  // exact duplicate of an earlier element
  expect_string_polarity_matrix({row}, {std::vector<bool>(row.size(), true)});
}

// Null-bearing rows exercise the class-bit placement in both the first-pass and window keys and
// the null-run guards in the tie-break; row 1's ties resolve only past the first window.
TEST_F(SortListsString, PrefixPolarityMatrixNullsBothPrecedences)
{
  std::vector<std::vector<std::string>> const rows{
    {"apple", "x", "apricot", "apple", "y", "banana"},
    {"z", "AAAAAAAAA", "AAAAAAAAb", "AAAAAAAAa"}};
  std::vector<std::vector<bool>> const valids{{true, false, true, true, false, true},
                                              {false, true, true, true}};
  expect_string_polarity_matrix(rows, valids);
}

// The 80-byte shared prefix outlasts every window, so the run reaches the tie-break's heapsort
// branch under each combo; the empty row checks dense segment labeling skips it cleanly.
TEST_F(SortListsString, PrefixPolarityMatrixMultiSegmentHeapsortRun)
{
  std::string const shared(80, 'Q');
  std::vector<std::string> big;
  for (int i = 0; i < 70; ++i) {
    big.push_back(shared + std::to_string(i / 10) + std::to_string(i % 10));
  }
  big.push_back(shared + "07");  // exact duplicate within the run
  auto shuffled = big;
  std::shuffle(shuffled.begin(), shuffled.end(), std::mt19937{0x5A17});
  shuffled.insert(shuffled.begin() + 3, "");
  shuffled.push_back("zzz");
  std::vector<bool> row0_valids(shuffled.size(), true);
  row0_valids[1]                   = false;  // two scattered nulls
  row0_valids[shuffled.size() - 2] = false;

  std::vector<std::vector<std::string>> const rows{
    shuffled, {}, {"melon", "apple", "cherry", "apple"}};
  std::vector<std::vector<bool>> const valids{row0_valids, {}, std::vector<bool>(4, true)};
  expect_string_polarity_matrix(rows, valids);
}

// Mirrors PrefixSlicedWithNulls (ascending), pinning offset + null handling in both directions.
TEST_F(SortListsString, PrefixDescendingSliced)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  StrLCW l{StrLCW{"zz", "aa"},
           StrLCW{{"banana", "apple", "x", "cherry"}, null_at(2)},  // element 2 ("x") is null
           StrLCW{"b", "a"}};
  auto const sliced = cudf::slice(l, {1, 3})[0];  // drops row 0 -> nonzero child offset
  {  // DESCENDING / AFTER: the comparator swap places the null first, then values descending
    StrLCW expected{StrLCW{{"z", "cherry", "banana", "apple"}, null_at(0)}, StrLCW{"b", "a"}};
    expect_both_sort_paths_match(
      cudf::lists_column_view{sliced}, expected, cudf::order::DESCENDING, cudf::null_order::AFTER);
  }
  {  // DESCENDING / BEFORE: values descending, then the null last
    StrLCW expected{StrLCW{{"cherry", "banana", "apple", "z"}, null_at(3)}, StrLCW{"b", "a"}};
    expect_both_sort_paths_match(
      cudf::lists_column_view{sliced}, expected, cudf::order::DESCENDING, cudf::null_order::BEFORE);
  }
}

// ===== graduated-warp string path: per-band polarity coverage (segments within the 64-element cap)
// The default string fast path once every segment fits a warp tile; its bands are (0,8] one item
// per lane, then (8,16] W8, W16, and W32.

// Sizes at and across every band boundary (8/9, 15/16/17, 31/32/33, 63/64) plus an empty row, side
// by side in one column; the 8/16/32/64 rows fill their tiles exactly (zero pads).
TEST_F(SortListsString, GradPolarityMatrixBandBoundarySizes)
{
  std::vector<cudf::size_type> const sizes{1, 2, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 0};
  std::mt19937 rng{0xBEEF};
  std::vector<std::vector<std::string>> rows;
  std::vector<std::vector<bool>> valids;
  for (auto const n : sizes) {
    std::vector<std::string> row(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      // Distinct within a row: 97 is coprime to 1000, so the numeric part never repeats for i < 64.
      row[i] = "k" + std::to_string(i * 97 % 1'000) + static_cast<char>('a' + i % 26);
    }
    std::shuffle(row.begin(), row.end(), rng);
    valids.emplace_back(row.size(), true);
    rows.push_back(std::move(row));
  }
  expect_string_polarity_matrix(rows, valids);
}

// The nulls-first combos put the valid class at ordinal 1: a comparator hardcoding it to 0 would
// collapse valid-vs-valid pairs to "equal" and keep the shuffled order. Row 1 (2 valid / 15 null /
// 15 pad) stresses the pad class ranking above tier_null; row 5 is all null.
TEST_F(SortListsString, GradPolarityMatrixNullsAcrossBands)
{
  std::vector<cudf::size_type> const sizes{12, 17, 20, 40, 64, 16, 5, 8};
  std::mt19937 rng{0xD00D};
  std::vector<std::vector<std::string>> rows;
  std::vector<std::vector<bool>> valids;
  for (std::size_t r = 0; r < sizes.size(); ++r) {
    auto const n = sizes[r];
    std::vector<std::pair<std::string, bool>> cells(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      // Row 1: only elements 3 and 11 valid (2 valid / 15 null). Row 5: all null. Others: every
      // (i % 5 == 2) element is null.
      bool const v = (r == 1) ? (i == 3 || i == 11) : (r == 5 ? false : (i % 5 != 2));
      cells[i]     = {"v" + std::to_string(i * 97 % 1'000), v};
    }
    if (n > 1 && cells[0].second && cells[1].second) {
      cells[1].first = cells[0].first;  // A duplicate among the valids.
    }
    std::shuffle(cells.begin(), cells.end(), rng);  // Interleave nulls; don't pre-group them.
    std::vector<std::string> row(n);
    std::vector<bool> valid(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      row[i]   = cells[i].first;
      valid[i] = cells[i].second;
    }
    rows.push_back(std::move(row));
    valids.push_back(std::move(valid));
  }
  expect_string_polarity_matrix(rows, valids);
}

// Zero-extension pairs (`S` vs `S + "\0"`) collide on the zero-filled window and resolve only by
// the length tie-break, whose side the descending complement flips; the shared eight-byte prefixes
// force every pair onto the suffix fallback in each band. Row 3 is pure duplicates.
TEST_F(SortListsString, GradPolarityMatrixTiesAndZeroExtension)
{
  using namespace std::string_literals;
  std::mt19937 rng{0xACE5};
  std::vector<std::string> row0{""s, "a"s, "ab"s, "ab"s, "ab\0"s, "ab\0c"s, "abc"s, "abcd"s, "b"s};
  std::vector<std::string> row1;
  for (int i = 0; i < 16; ++i) {
    row1.push_back("PPPPPPPP" + std::to_string(i));
  }
  row1.push_back("PPPPPPPP"s);
  row1.push_back("PPPPPPPP\0"s);
  row1.push_back("PPPPPPPP3"s);  // duplicate
  row1.push_back("PPPPPPPP7"s);  // duplicate
  std::vector<std::string> row2;
  for (int i = 0; i < 38; ++i) {
    row2.push_back("SAMEPREF" + std::string(1, static_cast<char>('0' + (i % 10))) +
                   std::to_string(i));
  }
  row2.push_back("SAMEPREF"s);
  row2.push_back("SAMEPREF\0"s);
  std::vector<std::string> row3(10, "dup"s);

  std::vector<std::vector<std::string>> rows;
  std::vector<std::vector<bool>> valids;
  for (auto* row : {&row0, &row1, &row2, &row3}) {
    std::shuffle(row->begin(), row->end(), rng);
    valids.emplace_back(row->size(), true);
    rows.push_back(*row);
  }
  expect_string_polarity_matrix(rows, valids);
}

// Raw unsigned-byte order across the 2-/3-/4-byte UTF-8 lead classes; row 1 shares a multibyte
// prefix so the prekey ties past the lead bytes.
TEST_F(SortListsString, GradPolarityMatrixUtf8Multibyte)
{
  std::mt19937 rng{0xFACE};
  std::vector<std::string> row0{
    "\xE2\x82\xAC\x75\x72\x6F", "zoo", "\xF0\x9F\x98\x80", "\xC3\xA9\x70\xC3\xA9\x65", "apple"};
  std::vector<std::string> row1;
  for (int i = 0; i < 36; ++i) {
    row1.push_back("\xC3\xA9" + std::to_string(i * 97 % 1'000));
  }
  std::vector<std::vector<std::string>> rows;
  std::vector<std::vector<bool>> valids;
  for (auto* row : {&row0, &row1}) {
    std::shuffle(row->begin(), row->end(), rng);
    valids.emplace_back(row->size(), true);
    rows.push_back(*row);
  }
  expect_string_polarity_matrix(rows, valids);
}

// A few hundred segments cycling every band size in (0, 64] drive the string instantiations of
// the shared warp-band kernel across block boundaries -- hundreds of virtual warps over multiple
// blocks per band -- where the small tests above stay within two blocks. Scattered nulls and a
// sprinkle of empty rows ride along, all against the host oracle under every combo.
TEST_F(SortListsString, GradManySegmentsAcrossBands)
{
  std::vector<std::vector<std::string>> rows;
  std::vector<std::vector<bool>> valids;
  for (int seg = 0; seg < 300; ++seg) {
    if (seg % 50 == 49) {  // Interleave empty rows; no band owns them.
      rows.emplace_back();
      valids.emplace_back();
      continue;
    }
    auto const size = seg % 64 + 1;
    std::vector<std::string> row(size);
    std::vector<bool> valid(size);
    for (int i = 0; i < size; ++i) {
      row[i]   = "s" + std::to_string((seg * 31 + i * 7) % 97) + std::string(i % 9, 'x');
      valid[i] = (seg * 13 + i) % 11 != 0;
    }
    rows.push_back(std::move(row));
    valids.push_back(std::move(valid));
  }
  expect_string_polarity_matrix(rows, valids);
}

// Slicing off row 0 gives the leaf strings a genuine nonzero offset, which the graduated key
// builders and comparators must honor under every combo.
TEST_F(SortListsString, GradSlicedWithNulls)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  StrLCW l{StrLCW{"zz", "aa"},
           StrLCW{{"banana", "apple", "x", "cherry"}, null_at(2)},  // element 2 ("x") is null
           StrLCW{"b", "a"},
           StrLCW{"x"}};
  auto const sliced = cudf::slice(l, {1, 4})[0];  // drops row 0 -> nonzero child offset
  {                                               // ASC / AFTER: values ascending, null last
    StrLCW expected{
      StrLCW{{"apple", "banana", "cherry", "x"}, null_at(3)}, StrLCW{"a", "b"}, StrLCW{"x"}};
    expect_both_sort_paths_match(cudf::lists_column_view{sliced}, expected);
  }
  {  // ASC / BEFORE: null first, then values ascending
    StrLCW expected{
      StrLCW{{"x", "apple", "banana", "cherry"}, null_at(0)}, StrLCW{"a", "b"}, StrLCW{"x"}};
    expect_both_sort_paths_match(
      cudf::lists_column_view{sliced}, expected, cudf::order::ASCENDING, cudf::null_order::BEFORE);
  }
  {  // DESC / AFTER: the comparator swap places the null first, then values descending
    StrLCW expected{
      StrLCW{{"x", "cherry", "banana", "apple"}, null_at(0)}, StrLCW{"b", "a"}, StrLCW{"x"}};
    expect_both_sort_paths_match(
      cudf::lists_column_view{sliced}, expected, cudf::order::DESCENDING, cudf::null_order::AFTER);
  }
  {  // DESC / BEFORE: values descending, then the null last
    StrLCW expected{
      StrLCW{{"cherry", "banana", "apple", "x"}, null_at(3)}, StrLCW{"b", "a"}, StrLCW{"x"}};
    expect_both_sort_paths_match(
      cudf::lists_column_view{sliced}, expected, cudf::order::DESCENDING, cudf::null_order::BEFORE);
  }
}

// One 65-element segment exceeds the warp cap and disqualifies the whole column, so the prefix
// path must produce the result under every combo.
TEST_F(SortListsString, GradOversizedSegmentFallsThrough)
{
  std::mt19937 rng{0xF00D};
  std::vector<std::string> big(65);
  for (int i = 0; i < 65; ++i) {
    big[i] = "q" + std::to_string(i * 97 % 1'000);
  }
  std::shuffle(big.begin(), big.end(), rng);
  std::vector<std::vector<std::string>> const rows{big, {"b", "a", "c"}};
  std::vector<std::vector<bool>> const valids{std::vector<bool>(big.size(), true),
                                              {true, true, true}};
  expect_string_polarity_matrix(rows, valids);
}

// ===== prefix string path: high-value scenarios forced past the 64-element graduated cap =====
// Each probe pairs its data with a >64-element row: the oversized row fails the graduated
// admission gate, so the whole column -- probe included -- genuinely rides the prefix path.

// Pins the prefix path's descending byte complement and nulls-BEFORE class-bit placement, which
// the grad-routed small-segment polarity tests never reach.
TEST_F(SortListsString, PrefixPolarityMatrixOversizedRun)
{
  std::string const shared(80, 'P');
  std::vector<std::pair<std::string, bool>> elems;
  for (int i = 0; i < 66; ++i) {
    elems.emplace_back(shared + std::to_string(i / 10) + std::to_string(i % 10), i % 13 != 0);
  }
  std::shuffle(elems.begin(), elems.end(), std::mt19937{0x9A1E});
  std::vector<std::string> row;
  std::vector<bool> valids;
  for (auto const& [str, valid] : elems) {
    row.push_back(str);
    valids.push_back(valid);
  }
  expect_string_polarity_matrix({row}, {valids});
}

// The null / all-0xFF sentinel collision under all four combos; the ASCII filler tail pushes the
// segment past the graduated cap.
TEST_F(SortListsString, PrefixNullSentinelCollisionFFOversized)
{
  std::string const ff72(72, '\xff');
  std::string const ff73(73, '\xff');
  std::vector<std::string> row{"apple", ff73, "mango", ff72, ""};
  std::vector<bool> valids{true, true, true, true, false};
  for (int i = 0; i < 65; ++i) {
    row.push_back("filler" + std::to_string(100 + i));
    valids.push_back(true);
  }
  expect_string_polarity_matrix({row}, {valids});
}

// UTF-8 lead-byte ordering (0xC3 / 0xE2 / 0xF0 after ASCII) on the prefix path's packed key and
// window compare.
TEST_F(SortListsString, PrefixUtf8MultibyteOversized)
{
  std::vector<std::string> row{
    "\xE2\x82\xAC\x75\x72\x6F", "zoo", "\xF0\x9F\x98\x80", "\xC3\xA9\x70\xC3\xA9\x65", "apple"};
  std::vector<bool> valids(row.size(), true);
  for (int i = 0; i < 65; ++i) {
    row.push_back("fill" + std::to_string(100 + i));
    valids.push_back(true);
  }
  expect_string_polarity_matrix({row}, {valids});
}

// Nonzero child offset on the prefix path: the packed-key builder and window compare must honor
// it under every combo.
TEST_F(SortListsString, PrefixSlicedOversized)
{
  std::string const shared(80, 'M');
  std::vector<std::pair<std::string, bool>> elems;
  for (int i = 0; i < 66; ++i) {
    elems.emplace_back(shared + std::to_string(i / 10) + std::to_string(i % 10), i % 11 != 0);
  }
  std::shuffle(elems.begin(), elems.end(), std::mt19937{0x5CED});
  std::vector<std::string> big;
  std::vector<bool> big_v;
  for (auto const& [str, valid] : elems) {
    big.push_back(str);
    big_v.push_back(valid);
  }
  std::vector<std::vector<std::string>> const rows{{"drop"}, big, {"b", "a"}};
  std::vector<std::vector<bool>> const valids{{true}, big_v, {true, true}};
  auto const full   = make_string_lists(rows, valids);
  auto const sliced = cudf::slice(full->view(), {1, 3})[0];  // drop row 0 -> nonzero child offset

  constexpr std::array<std::pair<cudf::order, cudf::null_order>, 4> combos{
    {{cudf::order::ASCENDING, cudf::null_order::AFTER},
     {cudf::order::DESCENDING, cudf::null_order::AFTER},
     {cudf::order::ASCENDING, cudf::null_order::BEFORE},
     {cudf::order::DESCENDING, cudf::null_order::BEFORE}}};
  for (auto const& [ord, np] : combos) {
    auto const s1 = host_sorted_row(big, big_v, ord, np);
    auto const s2 =
      host_sorted_row(std::vector<std::string>{"b", "a"}, std::vector<bool>{true, true}, ord, np);
    auto const expected = make_string_lists({s1.first, s2.first}, {s1.second, s2.second});
    expect_both_sort_paths_match(cudf::lists_column_view{sliced}, expected->view(), ord, np);
  }
}

// Sign-flip correctness at INT_MIN / INT_MAX for both tiered key widths (int32 -> uint64,
// int64 -> unsigned __int128); the nulls route the rows to the tiered kernel.
TEST_F(SortListsInt, NumericPackedNegativesAndBounds)
{
  {  // int32: the <= 4-byte uint64 tiered key.
    auto constexpr lo = std::numeric_limits<int32_t>::min();
    auto constexpr hi = std::numeric_limits<int32_t>::max();
    std::vector<bool> const in_valids{true, true, false, true, true, true, false};
    LCW<int32_t> input{{{hi, -1, 0, lo, 0, 1, 0}, in_valids.begin()}};
    std::vector<bool> const ex_valids{true, true, true, true, true, false, false};
    LCW<int32_t> expected{{{lo, -1, 0, 1, hi, 0, 0}, ex_valids.begin()}};
    expect_both_sort_paths_match(cudf::lists_column_view{input}, expected);
  }
  {  // int64: the 8-byte unsigned __int128 key. The file-wide `LCW` sources from `int32_t`, which
     // cannot hold the 64-bit bounds, hence the direct `int64_t` wrapper.
    using LCW64       = cudf::test::lists_column_wrapper<int64_t>;
    auto constexpr lo = std::numeric_limits<int64_t>::min();
    auto constexpr hi = std::numeric_limits<int64_t>::max();
    std::vector<bool> const in_valids{true, true, false, true, true, true, false};
    LCW64 input{{{hi, -1, 0, lo, 0, 1, 0}, in_valids.begin()}};
    std::vector<bool> const ex_valids{true, true, true, true, true, false, false};
    LCW64 expected{{{lo, -1, 0, 1, hi, 0, 0}, ex_valids.begin()}};
    expect_both_sort_paths_match(cudf::lists_column_view{input}, expected);
  }
}

// Degenerate shapes (empty, single-element, all-duplicate rows) through the tiered network tier;
// the null element is what routes the column there.
TEST_F(SortListsInt, NumericPackedEmptyAndSingleRows)
{
  std::vector<bool> const in_valids{true, false, true};
  LCW<int32_t> input{LCW<int32_t>{},
                     LCW<int32_t>{5},
                     LCW<int32_t>{-3, -3},
                     LCW<int32_t>{{2, 7, -1}, in_valids.begin()}};
  std::vector<bool> const ex_valids{true, true, false};
  LCW<int32_t> expected{LCW<int32_t>{},
                        LCW<int32_t>{5},
                        LCW<int32_t>{-3, -3},
                        LCW<int32_t>{{-1, 2, 7}, ex_valids.begin()}};
  expect_both_sort_paths_match(cudf::lists_column_view{input}, expected);
}

// Pre-/post-epoch timestamps for both rep widths guard the is_timestamp rep-extraction branch;
// the signed flip must sort pre-epoch (negative rep) values first.
TEST_F(SortListsInt, NumericPackedTimestamps)
{
  {  // TIMESTAMP_DAYS: int32 rep, the <= 4-byte uint64 tiered key.
    std::vector<int32_t> const in{100, -50, 0, 0, 25};
    std::vector<int32_t> const ex{-50, 0, 25, 100, 0};
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<cudf::timestamp_D>(in.begin(), in.end(), null_at(2))
        .release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<cudf::timestamp_D>(ex.begin(), ex.end(), null_at(4))
        .release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
  {  // TIMESTAMP_MILLISECONDS: int64 rep exceeding int32, the 8-byte unsigned __int128 tiered key.
    auto constexpr big = int64_t{5} * 1'000 * 1'000 * 1'000;  // 5e9 > INT32_MAX, post-epoch
    std::vector<int64_t> const in{big, -big, 0, 1, -1};
    std::vector<int64_t> const ex{-big, -1, 1, big, 0};
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms>(in.begin(), in.end(), null_at(2))
        .release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms>(ex.begin(), ex.end(), null_at(4))
        .release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
}

// DECIMAL128 extremes on the tiered kernel (nulls route both rows there): +/-10^37 (~2^123)
// reaches the high key words, and the sign flip (XOR bit 127) must order them across zero.
TEST_F(SortListsInt, NumericPackedDecimal128Bounds)
{
  auto const pow10_37 = [] {
    __int128_t v = 1;
    for (int i = 0; i < 37; ++i) {
      v *= 10;
    }
    return v;
  }();
  auto constexpr scale = numeric::scale_type{0};

  {  // A longer row.
    std::vector<__int128_t> ascending{-pow10_37, -pow10_37};
    for (int i = -5; i <= 25; ++i) {
      ascending.push_back(i);
    }
    ascending.push_back(pow10_37);
    auto const null_pos = static_cast<cudf::size_type>(ascending.size());

    // Values arrive reversed; the null-slot value is ignored.
    std::vector<__int128_t> in_vals;
    in_vals.push_back(0);
    in_vals.insert(in_vals.end(), ascending.rbegin(), ascending.rend());
    std::vector<__int128_t> ex_vals(ascending.begin(), ascending.end());
    ex_vals.push_back(0);

    auto input    = as_single_row_list(cudf::test::fixed_point_column_wrapper<__int128_t>(
                                      in_vals.begin(), in_vals.end(), null_at(0), scale)
                                      .release());
    auto expected = as_single_row_list(cudf::test::fixed_point_column_wrapper<__int128_t>(
                                         ex_vals.begin(), ex_vals.end(), null_at(null_pos), scale)
                                         .release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
  {  // The shorter-segment shape.
    std::vector<__int128_t> const in{pow10_37, -pow10_37, -1, 0, 1, -pow10_37, 0};
    std::vector<__int128_t> const ex{-pow10_37, -pow10_37, -1, 0, 1, pow10_37, 0};
    auto input = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(in.begin(), in.end(), null_at(3), scale)
        .release());
    auto expected = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(ex.begin(), ex.end(), null_at(6), scale)
        .release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
}

// BOOL8 is packed-radix-supported but not tiered, so a null routes it to the packed radix at any
// size -- the only test reaching the bool `radix_encode_u32` branch with the key's null bit.
TEST_F(SortListsInt, NumericPackedBoolWithNulls)
{
  std::vector<bool> const in_valids{true, true, false, true, true, false, true};
  auto input =
    as_single_row_list(cudf::test::fixed_width_column_wrapper<bool>(
                         {true, false, true, true, false, false, false}, in_valids.begin())
                         .release());
  std::vector<bool> const ex_valids{true, true, true, true, true, false, false};
  auto expected =
    as_single_row_list(cudf::test::fixed_width_column_wrapper<bool>(
                         {false, false, false, true, true, false, false}, ex_valids.begin())
                         .release());
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// All three tiers in one column -- network (<= 8), warp (9..64), radix (> 64), with both exact
// caps and their first over-cap sizes -- must merge into one gather map with no cross-row
// contamination; the nulls are what route the column to the tiered kernel.
TEST_F(SortListsInt, NumericTieredThreeSizeClasses)
{
  std::vector<cudf::size_type> const sizes{0, 1, 3, 8, 9, 33, 64, 65, 600, 3'000};
  std::vector<int32_t> in_vals;
  std::vector<bool> in_valids;
  std::vector<int32_t> ex_vals;
  std::vector<bool> ex_valids;
  std::vector<cudf::size_type> offsets{0};
  for (auto const s : sizes) {
    std::vector<int32_t> rv(s);
    std::vector<bool> rok(s);
    for (cudf::size_type i = 0; i < s; ++i) {
      rv[i]  = static_cast<int32_t>(s - i);  // distinct, descending -> the sort must fully reorder
      rok[i] = (i % 7 != 3);                 // scattered element nulls
    }
    if (s >= 2) { rv[1] = rv[0]; }  // a duplicate value among the non-nulls
    std::vector<int32_t> nn;
    for (cudf::size_type i = 0; i < s; ++i) {
      in_vals.push_back(rv[i]);
      in_valids.push_back(rok[i]);
      if (rok[i]) { nn.push_back(rv[i]); }
    }
    std::sort(nn.begin(), nn.end());
    for (auto const v : nn) {
      ex_vals.push_back(v);
      ex_valids.push_back(true);
    }
    for (cudf::size_type k = static_cast<cudf::size_type>(nn.size()); k < s; ++k) {
      ex_vals.push_back(0);
      ex_valids.push_back(false);
    }
    offsets.push_back(static_cast<cudf::size_type>(in_vals.size()));
  }
  auto const num_rows  = static_cast<cudf::size_type>(sizes.size());
  auto const make_list = [&](std::vector<int32_t> const& vals, std::vector<bool> const& valids) {
    cudf::test::fixed_width_column_wrapper<int32_t> leaf(vals.begin(), vals.end(), valids.begin());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
    return cudf::make_lists_column(num_rows, off.release(), leaf.release(), 0, {});
  };
  auto const input    = make_list(in_vals, in_valids);
  auto const expected = make_list(ex_vals, ex_valids);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// 200 network-tier and 10 warp-tier segments make both kernels' grids span >= 2 blocks (128
// threads = 4 virtual warps per block), which no other tiered test reaches.
TEST_F(SortListsInt, NumericTieredMultiBlockGrid)
{
  std::vector<cudf::size_type> sizes;
  for (int i = 0; i < 200; ++i) {
    sizes.push_back(1 + (i % 8));
  }  // 200 network-tier segments
  for (int i = 0; i < 10; ++i) {
    sizes.push_back(16 + (i % 40));
  }  // 10 warp-tier segments
  std::vector<int32_t> in_vals;
  std::vector<int32_t> ex_vals;
  std::vector<cudf::size_type> offsets{0};
  for (auto const s : sizes) {
    std::vector<int32_t> rv(s);
    for (cudf::size_type i = 0; i < s; ++i) {
      rv[i] = static_cast<int32_t>(s - i);
    }
    for (auto const v : rv) {
      in_vals.push_back(v);
    }
    std::sort(rv.begin(), rv.end());
    for (auto const v : rv) {
      ex_vals.push_back(v);
    }
    offsets.push_back(static_cast<cudf::size_type>(in_vals.size()));
  }
  auto const num_rows  = static_cast<cudf::size_type>(sizes.size());
  auto const make_list = [&](std::vector<int32_t> const& vals) {
    cudf::test::fixed_width_column_wrapper<int32_t> leaf(vals.begin(), vals.end());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
    return cudf::make_lists_column(num_rows, off.release(), leaf.release(), 0, {});
  };
  auto const input    = make_list(in_vals);
  auto const expected = make_list(ex_vals);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// Zero-one principle: a comparison network that sorts every 0/1 input sorts every input, so
// running all patterns at sizes 1..8 permanently pins the hand-unrolled 19-comparator network.
TEST_F(SortListsInt, NumericTieredNetworkZeroOneExhaustive)
{
  std::vector<int32_t> vals;
  std::vector<int32_t> ex_vals;
  std::vector<cudf::size_type> offsets{0};
  for (int n = 1; n <= 8; ++n) {
    for (int pattern = 0; pattern < (1 << n); ++pattern) {
      int ones = 0;
      for (int b = 0; b < n; ++b) {
        int const bit = (pattern >> b) & 1;
        vals.push_back(bit);
        ones += bit;
      }
      for (int b = 0; b < n; ++b) {
        ex_vals.push_back(b < n - ones ? 0 : 1);
      }
      offsets.push_back(static_cast<cudf::size_type>(vals.size()));
    }
  }
  auto const num_lists = static_cast<cudf::size_type>(offsets.size() - 1);
  auto const make_list = [&](std::vector<int32_t> const& v) {
    cudf::test::fixed_width_column_wrapper<int32_t> leaf(v.begin(), v.end());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
    return cudf::make_lists_column(num_lists, off.release(), leaf.release(), 0, {});
  };
  auto const input    = make_list(vals);
  auto const expected = make_list(ex_vals);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// Nonzero child offset plus a retained null: the tiered engine must read validity through the
// offset. Sliced covers the offset alone.
TEST_F(SortListsInt, SlicedWithNulls)
{
  using T = int;
  std::vector<bool> const valids{true, false, true};  // row 1's middle element (5) is null
  LCW<T> l{{3, 2, 1, 4}, {{7, 5, 6}, valids.begin()}, {8, 9}, {10}};

  auto const sliced_list = cudf::slice(l, {1, 4})[0];  // drops row 0 -> nonzero child offset
  std::vector<bool> const ex_valids{true, true, false};
  auto const expected = LCW<T>{{{6, 7, 0}, ex_valids.begin()}, {8, 9}, {10}};
  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{sliced_list}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// Mirrors SlicedWithNulls for the string engine: the packed-key builder, window tie-break, and
// null handling must all honor the nonzero leaf offset.
TEST_F(SortListsString, PrefixSlicedWithNulls)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  StrLCW l{StrLCW{"zz", "aa"},
           StrLCW{{"banana", "apple", "x", "cherry"}, null_at(2)},
           StrLCW{"b", "a"},
           StrLCW{"x"}};
  auto const sliced_list = cudf::slice(l, {1, 4})[0];  // drops row 0 -> nonzero child offset
  StrLCW expected{
    StrLCW{{"apple", "banana", "cherry", "x"}, null_at(3)}, StrLCW{"a", "b"}, StrLCW{"x"}};
  expect_both_sort_paths_match(cudf::lists_column_view{sliced_list}, expected);
}

// The warp tier in isolation (no segment above the 64 cap, so the radix fallback never runs):
// the int64 and double rows fill their warp tiles exactly (zero pads), the DECIMAL128 row leaves
// pad slots.
TEST_F(SortListsInt, NumericTieredWarpSegments)
{
  auto constexpr scale = numeric::scale_type{0};
  {  // int64 warp tier.
    cudf::size_type const n = 64;
    std::vector<int64_t> in(n);
    std::vector<bool> in_v(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] =
        static_cast<int64_t>(n - i) * 1'000'000'007LL;  // distinct, descending, beyond 32 bits
      in_v[i] = (i % 5 != 2);                           // scattered nulls
    }
    in[1] = in[0];  // a duplicate value among the non-nulls
    std::vector<int64_t> nn;
    for (cudf::size_type i = 0; i < n; ++i) {
      if (in_v[i]) { nn.push_back(in[i]); }
    }
    std::sort(nn.begin(), nn.end());
    std::vector<int64_t> ex = nn;
    std::vector<bool> ex_v(nn.size(), true);
    ex.resize(n, int64_t{0});
    ex_v.resize(n, false);
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int64_t>(in.begin(), in.end(), in_v.begin())
        .release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int64_t>(ex.begin(), ex.end(), ex_v.begin())
        .release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
  {  // decimal128 warp tier (24-byte key).
    cudf::size_type const n = 50;
    std::vector<__int128_t> in(n);
    std::vector<bool> in_v(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i]   = static_cast<__int128_t>(n - i) * (i % 2 == 0 ? 1 : -1);  // mixed sign, distinct
      in_v[i] = (i % 6 != 4);
    }
    in[3] = in[2];  // a duplicate value
    std::vector<__int128_t> nn;
    for (cudf::size_type i = 0; i < n; ++i) {
      if (in_v[i]) { nn.push_back(in[i]); }
    }
    std::sort(nn.begin(), nn.end());
    std::vector<__int128_t> ex = nn;
    std::vector<bool> ex_v(nn.size(), true);
    ex.resize(n, __int128_t{0});
    ex_v.resize(n, false);
    auto input = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(in.begin(), in.end(), in_v.begin(), scale)
        .release());
    auto expected = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(ex.begin(), ex.end(), ex_v.begin(), scale)
        .release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
  {  // double at exactly TIERED_WARP_CAP: full-tile occupancy of the generic warp kernel (the
     // path float/double/chrono/decimal128 share).
    cudf::size_type const n = 64;
    std::vector<double> in(n);
    std::vector<bool> in_v(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i]   = static_cast<double>(n - i);  // distinct, descending
      in_v[i] = (i % 5 != 2);                // scattered nulls
    }
    in[1] = in[0];  // a duplicate value among the non-nulls
    std::vector<double> nn;
    for (cudf::size_type i = 0; i < n; ++i) {
      if (in_v[i]) { nn.push_back(in[i]); }
    }
    std::sort(nn.begin(), nn.end());
    std::vector<double> ex = nn;
    std::vector<bool> ex_v(nn.size(), true);
    ex.resize(n, 0.0);
    ex_v.resize(n, false);
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<double>(in.begin(), in.end(), in_v.begin()).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<double>(ex.begin(), ex.end(), ex_v.begin()).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
}

// No-null int64 single-segment shapes this small sit in the eight-byte tiny-CUB pocket (average
// at or below the network cap, few offsets), so both sizes route to `comparison` and resolve to
// the pre-existing CUB `DeviceSegmentedSort` fast path rather than the tiered network/warp
// kernels; the cells pin the pocket's fast-vs-comparison agreement.
TEST_F(SortListsInt, NumericMidBandInt64Tiered)
{
  auto constexpr big          = int64_t{5} * 1'000 * 1'000 * 1'000;  // > INT32_MAX
  auto const check_single_row = [big](cudf::size_type n) {
    std::vector<int64_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = (static_cast<int64_t>(n / 2) - i) * big;
    }
    std::vector<int64_t> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int64_t>(in.begin(), in.end()).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int64_t>(ex.begin(), ex.end()).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  };
  check_single_row(8);   // one segment of eight -> tiny-CUB pocket (comparison -> CUB)
  check_single_row(16);  // one segment of sixteen -> tiny-CUB pocket (comparison -> CUB)
}

// `dispatch_storage_type` does not reduce chrono to its integer rep, so routing an 8-byte no-null
// chrono to CUB throws CUDF_FAIL; the mid band must stay on the tiered kernel.
TEST_F(SortListsInt, NumericMidBandChronoNoNull)
{
  std::vector<int64_t> in(16);
  for (cudf::size_type i = 0; i < 16; ++i) {
    in[i] = (8 - i) * int64_t{1'000'000'000};
  }
  std::vector<int64_t> ex(in);
  std::sort(ex.begin(), ex.end());
  {  // TIMESTAMP_MILLISECONDS (int64 rep)
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms>(in.begin(), in.end()).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms>(ex.begin(), ex.end()).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
  {  // DURATION_MICROSECONDS (int64 rep)
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<cudf::duration_us>(in.begin(), in.end()).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<cudf::duration_us>(ex.begin(), ex.end()).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
}

// DECIMAL128 no-null routing trio: tiered below the CUB band, the lifted CUB `DeviceSegmentedSort`
// (over the __int128 rep) inside it, tiered again above it below the packed-radix cutoff.
TEST_F(SortListsInt, NumericMidBandDecimal128Cub)
{
  auto constexpr scale = numeric::scale_type{0};
  auto const k         = [] {
    __int128_t v = 1;
    for (int i = 0; i < 30; ++i) {
      v *= 10;
    }
    return v;
  }();
  auto const check_single_row = [&](cudf::size_type n) {
    std::vector<__int128_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = static_cast<__int128_t>(n / 2 - i) * k;
    }
    std::vector<__int128_t> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(in.begin(), in.end(), scale).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(ex.begin(), ex.end(), scale).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  };
  check_single_row(8);   // average four: tiered kernel
  check_single_row(16);  // average eight: DECIMAL128 mid band -> lifted CUB DeviceSegmentedSort
  check_single_row(40);  // average twenty: above the CUB band, below packed radix -> tiered kernel
  // The exact gate cells: average five is the first cell past the tiered tiny gate (avg > 4),
  // average sixteen the last in-band CUB cell, average seventeen the first above-band tiered cell.
  check_single_row(10);
  check_single_row(32);
  check_single_row(34);
}

// Pins both outcomes of the shape gate's "oversized segments are sparse" ratio
// (`oversized * 32 <= num_segments`) at one mid-band average: ratio holds -> CUB, fails -> tiered.
TEST_F(SortListsInt, NumericMidBandDecimal128CubShapeGate)
{
  auto constexpr scale = numeric::scale_type{0};
  auto const check =
    [&](int num_small, cudf::size_type small_sz, int num_large, cudf::size_type large_sz) {
      std::vector<__int128_t> in_vals;
      std::vector<__int128_t> ex_vals;
      std::vector<cudf::size_type> offsets{0};
      auto const add_row = [&](cudf::size_type sz) {
        std::vector<__int128_t> row(sz);
        for (cudf::size_type i = 0; i < sz; ++i) {
          row[i] = static_cast<__int128_t>(sz) - i;
        }
        for (auto const v : row) {
          in_vals.push_back(v);
        }
        std::sort(row.begin(), row.end());
        for (auto const v : row) {
          ex_vals.push_back(v);
        }
        offsets.push_back(static_cast<cudf::size_type>(in_vals.size()));
      };
      for (int r = 0; r < num_small; ++r) {
        add_row(small_sz);
      }
      for (int r = 0; r < num_large; ++r) {
        add_row(large_sz);
      }
      auto const num_rows  = static_cast<cudf::size_type>(num_small + num_large);
      auto const make_list = [&](std::vector<__int128_t> const& vals) {
        cudf::test::fixed_point_column_wrapper<__int128_t> leaf(vals.begin(), vals.end(), scale);
        cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
        return cudf::make_lists_column(num_rows, off.release(), leaf.release(), 0, {});
      };
      auto const input    = make_list(in_vals);
      auto const expected = make_list(ex_vals);
      expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
    };
  check(200, 10, 2, 40);  // avg ~10, oversized=2: 2*32=64 <= 202 -> shape OK  -> lifted CUB
  check(10, 10, 2, 40);   // avg ~15, oversized=2: 2*32=64 >  12  -> shape bad -> tiered
  check(62, 10, 2, 40);   // exact ratio cell: 2*32=64 == 64 segments -> shape still OK (<=) -> CUB
}

// Pins the packed-radix gate's avg >= 100 boundary exactly: average 100 is the first packed-radix
// shape, average 99 the last tiered one.
TEST_F(SortListsInt, NumericPackedRadixAvgGateBoundary)
{
  auto const check_single_row = [](cudf::size_type n) {
    std::vector<int32_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = n / 2 - i;
    }
    std::vector<int32_t> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int32_t>(in.begin(), in.end()).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int32_t>(ex.begin(), ex.end()).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  };
  check_single_row(200);  // average exactly 100 -> the first packed-radix cell
  check_single_row(198);  // average 99 -> the last tiered cell
}

// The long-list band (avg at/above the packed-radix cutoff) belongs to the one-shot packed radix;
// all three key widths span the sign boundary.
TEST_F(SortListsInt, NumericPackedRadixLongLists)
{
  cudf::size_type const n = 220;  // single row, average 110 -> packed radix
  {                               // int32
    std::vector<int32_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = n / 2 - i;
    }
    std::vector<int32_t> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int32_t>(in.begin(), in.end()).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int32_t>(ex.begin(), ex.end()).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
  {  // int64 beyond the 32-bit range
    auto constexpr big = int64_t{5} * 1'000 * 1'000 * 1'000;
    std::vector<int64_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = (static_cast<int64_t>(n / 2) - i) * big;
    }
    std::vector<int64_t> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int64_t>(in.begin(), in.end()).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int64_t>(ex.begin(), ex.end()).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
  {  // decimal128 into the high value words
    auto constexpr scale = numeric::scale_type{0};
    auto const k         = [] {
      __int128_t v = 1;
      for (int i = 0; i < 30; ++i) {
        v *= 10;
      }
      return v;
    }();
    std::vector<__int128_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = static_cast<__int128_t>(n / 2 - i) * k;
    }
    std::vector<__int128_t> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(in.begin(), in.end(), scale).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(ex.begin(), ex.end(), scale).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
}

// No-null non-tiered types reach the packed radix only when num_rows >= 1<<18 AND avg >= 100 -- a
// route no other test takes; the no-null key drops the null-class bit for the full value budget.
TEST_F(SortListsInt, NumericPackedRadixNoNullNarrowTypes)
{
  cudf::size_type const n = 1 << 18;
  auto const check        = [&](auto tag) {
    using T = decltype(tag);
    std::vector<T> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = static_cast<T>((static_cast<int64_t>(n) - i) % 60'000);
    }
    std::vector<T> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input =
      as_single_row_list(cudf::test::fixed_width_column_wrapper<T>(in.begin(), in.end()).release());
    auto expected =
      as_single_row_list(cudf::test::fixed_width_column_wrapper<T>(ex.begin(), ex.end()).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  };
  check(uint16_t{});
  check(int16_t{});
  check(uint64_t{});

  // DECIMAL32/64 (non-tiered) take the same no-null packed radix over their int32/int64 rep.
  auto const check_decimal = [&](auto rep_tag) {
    using Rep            = decltype(rep_tag);
    auto constexpr scale = numeric::scale_type{0};
    std::vector<Rep> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = static_cast<Rep>((static_cast<int64_t>(n) - i) % 60'000);
    }
    std::vector<Rep> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<Rep>(in.begin(), in.end(), scale).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<Rep>(ex.begin(), ex.end(), scale).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  };
  check_decimal(int32_t{});  // DECIMAL32.
  check_decimal(int64_t{});  // DECIMAL64.
}

// The CUB-preference gate is a disjunction: a high average alone must not force the packed radix
// while the total row count stays under `MAX_LIST_SIZE_FOR_FAST_SORT`. This shape (avg 149,
// 150'000 rows) holds only through the row-count disjunct, exercising the CUB fast path at a
// narrow-type cell no other test reaches; fast and comparison paths must agree.
TEST_F(SortListsInt, NumericPackedRadixNarrowTypeHighAvgLowRowCount)
{
  cudf::size_type const num_segments = 1'000;
  cudf::size_type const seg_size     = 150;  // avg = 150'000 / 1'001 = 149 >= 100
  std::vector<uint16_t> in_vals(static_cast<std::size_t>(num_segments) * seg_size);
  for (std::size_t i = 0; i < in_vals.size(); ++i) {
    in_vals[i] = static_cast<uint16_t>((i * 31) % 60'000);
  }
  std::vector<cudf::size_type> offsets(num_segments + 1);
  for (cudf::size_type i = 0; i <= num_segments; ++i) {
    offsets[i] = i * seg_size;
  }
  std::vector<uint16_t> ex_vals(in_vals);
  for (cudf::size_type s = 0; s < num_segments; ++s) {
    std::sort(ex_vals.begin() + offsets[s], ex_vals.begin() + offsets[s + 1]);
  }
  auto const make_list = [&](std::vector<uint16_t> const& vals) {
    return cudf::make_lists_column(
      num_segments,
      cudf::test::fixed_width_column_wrapper<cudf::size_type>(offsets.begin(), offsets.end())
        .release(),
      cudf::test::fixed_width_column_wrapper<uint16_t>(vals.begin(), vals.end()).release(),
      0,
      {});
  };
  auto const input    = make_list(in_vals);
  auto const expected = make_list(ex_vals);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// The no-null int32 register-bitonic bands with the 8/9, 32/33, and 64/65 boundaries in one
// column. INT32_MAX radix-encodes to the raw-key pad sentinel, so the pad tie-break must keep it
// inside the segment.
TEST_F(SortListsInt, NumericTieredBitonicNoNullInt32)
{
  std::vector<cudf::size_type> const sizes{8, 9, 16, 17, 32, 33, 48, 64, 65};
  auto constexpr lo = std::numeric_limits<int32_t>::min();
  auto constexpr hi = std::numeric_limits<int32_t>::max();
  std::vector<int32_t> in_vals;
  std::vector<int32_t> ex_vals;
  std::vector<cudf::size_type> offsets{0};
  for (auto const s : sizes) {
    std::vector<int32_t> rv(s);
    for (cudf::size_type i = 0; i < s; ++i) {
      rv[i] = static_cast<int32_t>(s - i);
    }
    rv[0] = hi;                     // a real INT32_MAX (radix-encodes to the raw-key pad sentinel)
    if (s >= 2) { rv[1] = lo; }     // INT32_MIN
    if (s >= 4) { rv[2] = rv[3]; }  // a duplicate value
    for (auto const v : rv) {
      in_vals.push_back(v);
    }
    std::sort(rv.begin(), rv.end());
    for (auto const v : rv) {
      ex_vals.push_back(v);
    }
    offsets.push_back(static_cast<cudf::size_type>(in_vals.size()));
  }
  auto const num_rows  = static_cast<cudf::size_type>(sizes.size());
  auto const make_list = [&](std::vector<int32_t> const& vals) {
    cudf::test::fixed_width_column_wrapper<int32_t> leaf(vals.begin(), vals.end());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
    return cudf::make_lists_column(num_rows, off.release(), leaf.release(), 0, {});
  };
  auto const input    = make_list(in_vals);
  auto const expected = make_list(ex_vals);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// The no-null int64 bitonic/narrow split: register bitonic up to 32, raw-key `WarpMergeSort` for
// 33..64, pinned at the 32/33 and 64/65 boundaries. INT64_MAX encodes to exactly the raw-key pad
// sentinel, so pads must never displace the real element.
TEST_F(SortListsInt, NumericTieredBitonicNarrowNoNullInt64)
{
  std::vector<cudf::size_type> const sizes{9, 32, 33, 40, 48, 64, 65};
  auto constexpr lo  = std::numeric_limits<int64_t>::min();
  auto constexpr hi  = std::numeric_limits<int64_t>::max();
  auto constexpr big = int64_t{5} * 1'000 * 1'000 * 1'000;  // > INT32_MAX
  std::vector<int64_t> in_vals;
  std::vector<int64_t> ex_vals;
  std::vector<cudf::size_type> offsets{0};
  for (auto const s : sizes) {
    std::vector<int64_t> rv(s);
    for (cudf::size_type i = 0; i < s; ++i) {
      rv[i] = (static_cast<int64_t>(s) - i) * big;
    }
    rv[0] = hi;                     // a real INT64_MAX (encodes to the raw-key pad sentinel)
    if (s >= 2) { rv[1] = lo; }     // INT64_MIN
    if (s >= 4) { rv[2] = rv[3]; }  // a duplicate value
    for (auto const v : rv) {
      in_vals.push_back(v);
    }
    std::sort(rv.begin(), rv.end());
    for (auto const v : rv) {
      ex_vals.push_back(v);
    }
    offsets.push_back(static_cast<cudf::size_type>(in_vals.size()));
  }
  auto const num_rows  = static_cast<cudf::size_type>(sizes.size());
  auto const make_list = [&](std::vector<int64_t> const& vals) {
    cudf::test::fixed_width_column_wrapper<int64_t> leaf(vals.begin(), vals.end());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
    return cudf::make_lists_column(num_rows, off.release(), leaf.release(), 0, {});
  };
  auto const input    = make_list(in_vals);
  auto const expected = make_list(ex_vals);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// Null-bearing int32/int64 warp tier: the packed-key `WarpMergeSort` (one item per lane to 32,
// two to 64) across the 8/9, 32/33, and 64/65 boundaries.
TEST_F(SortListsInt, NumericTieredW32x1Nulls)
{
  std::vector<cudf::size_type> const sizes{8, 9, 32, 33, 64, 65};
  auto const run = [&](auto tag) {
    using T           = decltype(tag);
    auto constexpr lo = std::numeric_limits<T>::min();
    auto constexpr hi = std::numeric_limits<T>::max();
    std::vector<T> in_vals;
    std::vector<bool> in_valids;
    std::vector<T> ex_vals;
    std::vector<bool> ex_valids;
    std::vector<cudf::size_type> offsets{0};
    for (auto const s : sizes) {
      std::vector<T> rv(s);
      std::vector<bool> rok(s);
      for (cudf::size_type i = 0; i < s; ++i) {
        rv[i]  = static_cast<T>(s - i);
        rok[i] = (i % 5 != 2);  // scattered element nulls
      }
      rv[0] = hi;
      if (s >= 2) { rv[1] = lo; }
      if (s >= 4) { rv[2] = rv[3]; }  // a duplicate among the non-nulls
      std::vector<T> nn;
      for (cudf::size_type i = 0; i < s; ++i) {
        in_vals.push_back(rv[i]);
        in_valids.push_back(rok[i]);
        if (rok[i]) { nn.push_back(rv[i]); }
      }
      std::sort(nn.begin(), nn.end());
      for (auto const v : nn) {
        ex_vals.push_back(v);
        ex_valids.push_back(true);
      }
      for (cudf::size_type k = static_cast<cudf::size_type>(nn.size()); k < s; ++k) {
        ex_vals.push_back(T{0});
        ex_valids.push_back(false);
      }
      offsets.push_back(static_cast<cudf::size_type>(in_vals.size()));
    }
    auto const num_rows  = static_cast<cudf::size_type>(sizes.size());
    auto const make_list = [&](std::vector<T> const& vals, std::vector<bool> const& valids) {
      cudf::test::fixed_width_column_wrapper<T> leaf(vals.begin(), vals.end(), valids.begin());
      cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
      return cudf::make_lists_column(num_rows, off.release(), leaf.release(), 0, {});
    };
    auto const input    = make_list(in_vals, in_valids);
    auto const expected = make_list(ex_vals, ex_valids);
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  };
  run(int32_t{});
  run(int64_t{});
}

// Site A (the full-column packed radix) picks the narrowest lossless key from the value range:
// < 2^32 -> min-biased uint64, fits int64 -> prefix_key96, else the two-phase hi64/lo64 sort.
TEST_F(SortListsInt, NumericDecimal128ComposedSiteA)
{
  auto constexpr scale    = numeric::scale_type{0};
  cudf::size_type const n = 250;  // single row, avg 125 >= 100 -> full-column packed radix (Site A)
  auto const check        = [&](std::vector<__int128_t> const& in) {
    std::vector<__int128_t> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(in.begin(), in.end(), scale).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(ex.begin(), ex.end(), scale).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  };
  {  // small: value range < 2^32 -> min-biased uint64 key
    std::vector<__int128_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = static_cast<__int128_t>(n / 2 - i);
    }
    check(in);
  }
  {  // fits int64 (range > 2^32) -> prefix_key96 key
    auto const step = static_cast<__int128_t>(int64_t{1} << 40);  // 2^40 > 2^32
    std::vector<__int128_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = static_cast<__int128_t>(n / 2 - i) * step;
    }
    check(in);
  }
  {  // wide, distinct high words -> two-phase, phase one resolves (no tie pass)
    auto const step = static_cast<__int128_t>(1) << 90;  // > int64, distinct hi64
    std::vector<__int128_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = static_cast<__int128_t>(n / 2 - i) * step;
    }
    check(in);
  }
  {  // wide with shared high words -> two-phase second pass fires (two sign clusters, distinct
     // lo64)
    auto const base = static_cast<__int128_t>(1) << 100;
    std::vector<__int128_t> in;
    for (int i = 0; i < 125; ++i) {
      in.push_back(-base + i);
    }
    for (int i = 0; i < 125; ++i) {
      in.push_back(base + i);
    }  // 250 total -> avg 125 -> Site A
    check(in);
  }
}

// Site B: with nulls, a segment above the warp cap becomes a tiered radix outlier on the
// compact-large-segment path, which reuses the range-gated key selection; the all-null case
// degenerates to the cheapest key.
TEST_F(SortListsInt, NumericDecimal128ComposedSiteB)
{
  auto constexpr scale    = numeric::scale_type{0};
  cudf::size_type const n = 200;  // one segment > 64 -> radix-tier outlier -> compact path (Site B)
  auto const check        = [&](std::vector<__int128_t> const& in, std::vector<bool> const& valid) {
    std::vector<__int128_t> nn;
    for (cudf::size_type i = 0; i < n; ++i) {
      if (valid[i]) { nn.push_back(in[i]); }
    }
    std::sort(nn.begin(), nn.end());
    std::vector<__int128_t> ex = nn;
    std::vector<bool> ex_v(nn.size(), true);
    ex.resize(n, __int128_t{0});
    ex_v.resize(n, false);
    auto input = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(in.begin(), in.end(), valid.begin(), scale)
        .release());
    auto expected = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(ex.begin(), ex.end(), ex_v.begin(), scale)
        .release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  };
  {  // null-mixed wide with shared high words: two-phase second pass fires, nulls placed last
    auto const base = static_cast<__int128_t>(1) << 100;
    std::vector<__int128_t> in;
    std::vector<bool> valid;
    for (int i = 0; i < 100; ++i) {
      in.push_back(-base + i);
      valid.push_back(i % 7 != 3);
    }
    for (int i = 0; i < 100; ++i) {
      in.push_back(base + i);
      valid.push_back(i % 5 != 2);
    }
    check(in, valid);
  }
  {  // all-null: degenerate range -> cheapest key; every element sorts as a position-final null
    std::vector<__int128_t> const in(n, __int128_t{0});
    std::vector<bool> const valid(n, false);
    check(in, valid);
  }
}

// The Site A range gate's 2^32 boundary: a span of exactly 2^32 - 1 still fits the min-biased
// uint64 key, a span of 2^32 must take the wider prefix_key96.
TEST_F(SortListsInt, NumericDecimal128RangeGateBoundary)
{
  auto constexpr scale    = numeric::scale_type{0};
  cudf::size_type const n = 250;  // single row, avg 125 >= 100 -> Site A range gate
  auto const check        = [&](__int128_t span) {
    std::vector<__int128_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = (span / (n - 1)) * i;
    }
    in[n - 1] = span;  // exact max, so the range is [0, span]
    std::vector<__int128_t> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(in.begin(), in.end(), scale).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(ex.begin(), ex.end(), scale).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  };
  check((static_cast<__int128_t>(1) << 32) - 1);  // span 2^32 - 1 -> min-biased uint64 (lossless)
  check(static_cast<__int128_t>(1) << 32);        // span 2^32     -> prefix_key96
}

// The key96 gate requires BOTH range bounds inside int64; an asymmetric range -- one bound far
// outside, the other within -- must fall through to the two-phase key. A loosened gate
// (`and` -> `or`) would narrow the out-of-range bound through `dec128_int64_key_builder`'s
// `static_cast<int64_t>` and corrupt the order, which the host oracle below detects.
TEST_F(SortListsInt, NumericDecimal128AsymmetricRangeGate)
{
  auto constexpr scale    = numeric::scale_type{0};
  cudf::size_type const n = 250;  // single row, avg 125 >= 100 -> packed radix, Site A range gate
  auto const check        = [&](__int128_t outlier) {
    std::vector<__int128_t> in(n);
    for (cudf::size_type i = 0; i < n - 1; ++i) {
      in[i] = i - 100;  // small values well inside int64
    }
    in[n - 1] = outlier;  // one bound far outside int64
    std::vector<__int128_t> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(in.begin(), in.end(), scale).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(ex.begin(), ex.end(), scale).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  };
  check(-(static_cast<__int128_t>(1) << 100));  // min far below INT64_MIN, max within int64
  check(static_cast<__int128_t>(1) << 100);     // max far above INT64_MAX, min within int64
}

// The two-phase worst tie shape: every value sits in one aligned 2^64 band, so phase one's
// encoded hi64 is a single constant and every element ties (num_tied == num_elements) -- phase
// two must carry the order entirely in lo64. Row 1 mixes two-element runs with untied singletons,
// so run detection must separate runs inside one segment and the tie scatter must leave untied
// slots alone.
TEST_F(SortListsInt, NumericDecimal128TwoPhaseAllTied)
{
  auto constexpr scale = numeric::scale_type{0};
  auto const band      = static_cast<__int128_t>(1) << 63;

  // Span >= 2^32 within the band and max > int64 force the two-phase key.
  std::vector<__int128_t> row0(250);
  for (cudf::size_type i = 0; i < 250; ++i) {
    row0[i] = band + (static_cast<__int128_t>(249 - i) << 32) + (i * 13) % 251;
  }
  row0[20] = row0[10];  // exact duplicates inside the run
  row0[21] = row0[11];

  // List 1: 50 interleaved (pair, singleton, pair-partner) triples, descending across groups.
  std::vector<__int128_t> row1;
  for (int j = 49; j >= 0; --j) {
    auto const pair_hi      = static_cast<__int128_t>(j + 1) << 70;
    auto const singleton_hi = static_cast<__int128_t>(j + 200) << 70;
    row1.push_back(pair_hi + 2 * j + 1);
    row1.push_back(singleton_hi + j);
    row1.push_back(pair_hi + 2 * j);
  }

  auto const make_lists = [&](std::vector<__int128_t> const& a, std::vector<__int128_t> const& b) {
    std::vector<__int128_t> leaf(a);
    leaf.insert(leaf.end(), b.begin(), b.end());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{
      0, static_cast<cudf::size_type>(a.size()), static_cast<cudf::size_type>(leaf.size())};
    return cudf::make_lists_column(
      2,
      offsets.release(),
      cudf::test::fixed_point_column_wrapper<__int128_t>(leaf.begin(), leaf.end(), scale).release(),
      0,
      {});
  };

  std::vector<__int128_t> ex0(row0);
  std::vector<__int128_t> ex1(row1);
  std::sort(ex0.begin(), ex0.end());
  std::sort(ex1.begin(), ex1.end());
  auto const input    = make_lists(row0, row1);
  auto const expected = make_lists(ex0, ex1);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// The same all-tied shape under DESCENDING: the complement keeps equal encoded hi64 words equal,
// so the run still ties and phase two's complemented lo64 must reverse the refined order.
TEST_F(SortListsInt, NumericDecimal128TwoPhaseAllTiedDescending)
{
  auto constexpr scale = numeric::scale_type{0};
  auto const band      = static_cast<__int128_t>(1) << 63;

  std::vector<__int128_t> row0(250);
  for (cudf::size_type i = 0; i < 250; ++i) {
    row0[i] = band + (static_cast<__int128_t>(249 - i) << 32) + (i * 13) % 251;
  }
  row0[20] = row0[10];  // exact duplicates inside the run
  row0[21] = row0[11];

  std::vector<__int128_t> row1;
  for (int j = 49; j >= 0; --j) {
    auto const pair_hi      = static_cast<__int128_t>(j + 1) << 70;
    auto const singleton_hi = static_cast<__int128_t>(j + 200) << 70;
    row1.push_back(pair_hi + 2 * j + 1);
    row1.push_back(singleton_hi + j);
    row1.push_back(pair_hi + 2 * j);
  }

  auto const make_lists = [&](std::vector<__int128_t> const& a, std::vector<__int128_t> const& b) {
    std::vector<__int128_t> leaf(a);
    leaf.insert(leaf.end(), b.begin(), b.end());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{
      0, static_cast<cudf::size_type>(a.size()), static_cast<cudf::size_type>(leaf.size())};
    return cudf::make_lists_column(
      2,
      offsets.release(),
      cudf::test::fixed_point_column_wrapper<__int128_t>(leaf.begin(), leaf.end(), scale).release(),
      0,
      {});
  };

  std::vector<__int128_t> ex0(row0);
  std::vector<__int128_t> ex1(row1);
  std::sort(ex0.begin(), ex0.end());
  std::reverse(ex0.begin(), ex0.end());
  std::sort(ex1.begin(), ex1.end());
  std::reverse(ex1.begin(), ex1.end());
  auto const input    = make_lists(row0, row1);
  auto const expected = make_lists(ex0, ex1);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()},
                               expected->view(),
                               cudf::order::DESCENDING,
                               cudf::null_order::AFTER);
}

// ============ full (order, null_order) polarity matrix over the fixed-width fast paths ==========
// Each test drives one engine family through the three non-shipped combos; under DESCENDING the
// comparator operands swap, so nulls-AFTER leads with nulls and nulls-BEFORE trails.

// Batcher-network tier (segments <= 8): negatives, duplicates, and nulls make each combo's
// expectation unique.
TEST_F(SortListsInt, NumericTieredNetworkPolarityMatrix)
{
  std::vector<bool> const in3{true, false, true};
  std::vector<bool> const in8{true, true, false, true, true, false, true, true};
  LCW<int32_t> input{LCW<int32_t>{},
                     LCW<int32_t>{5},
                     LCW<int32_t>{{2, 0, -7}, in3.begin()},
                     LCW<int32_t>{{8, -3, 0, 8, 0, 0, -3, 1}, in8.begin()}};

  std::vector<bool> const nulls_lead3{false, true, true};
  std::vector<bool> const nulls_tail3{true, true, false};
  std::vector<bool> const nulls_lead8{false, false, true, true, true, true, true, true};
  std::vector<bool> const nulls_tail8{true, true, true, true, true, true, false, false};
  {  // DESCENDING / AFTER: nulls first, then values descending
    LCW<int32_t> expected{LCW<int32_t>{},
                          LCW<int32_t>{5},
                          LCW<int32_t>{{0, 2, -7}, nulls_lead3.begin()},
                          LCW<int32_t>{{0, 0, 8, 8, 1, 0, -3, -3}, nulls_lead8.begin()}};
    expect_both_sort_paths_match(
      cudf::lists_column_view{input}, expected, cudf::order::DESCENDING, cudf::null_order::AFTER);
  }
  {  // ASCENDING / BEFORE: nulls first, then values ascending
    LCW<int32_t> expected{LCW<int32_t>{},
                          LCW<int32_t>{5},
                          LCW<int32_t>{{0, -7, 2}, nulls_lead3.begin()},
                          LCW<int32_t>{{0, 0, -3, -3, 0, 1, 8, 8}, nulls_lead8.begin()}};
    expect_both_sort_paths_match(
      cudf::lists_column_view{input}, expected, cudf::order::ASCENDING, cudf::null_order::BEFORE);
  }
  {  // DESCENDING / BEFORE: values descending, then nulls last
    LCW<int32_t> expected{LCW<int32_t>{},
                          LCW<int32_t>{5},
                          LCW<int32_t>{{2, -7, 0}, nulls_tail3.begin()},
                          LCW<int32_t>{{8, 8, 1, 0, -3, -3, 0, 0}, nulls_tail8.begin()}};
    expect_both_sort_paths_match(
      cudf::lists_column_view{input}, expected, cudf::order::DESCENDING, cudf::null_order::BEFORE);
  }
}

// Null-bearing int32/int64 across the packed-key warp bands (w32x1 for 9..32, w32x2 for 33..64)
// and the radix outlier, pinning the 32/33 and 64/65 boundaries under every non-shipped combo.
TEST_F(SortListsInt, NumericTieredWarpBandsPolarityMatrix)
{
  std::vector<cudf::size_type> const sizes{9, 32, 33, 64, 65};
  auto const run = [&](auto tag, auto spread) {
    using T = decltype(tag);
    for (auto const& [ord, np] : non_default_combos) {
      std::vector<T> in_vals;
      std::vector<bool> in_valids;
      std::vector<T> ex_vals;
      std::vector<bool> ex_valids;
      std::vector<cudf::size_type> offsets{0};
      for (auto const s : sizes) {
        std::vector<T> rv(s);
        std::vector<bool> rok(s);
        for (cudf::size_type i = 0; i < s; ++i) {
          rv[i]  = static_cast<T>((s - i) * spread);
          rok[i] = (i % 5 != 2);  // scattered element nulls
        }
        rv[0]                = std::numeric_limits<T>::max();
        rv[1]                = std::numeric_limits<T>::min();
        rv[3]                = rv[4];  // a duplicate value among the non-nulls
        auto const [ev, eok] = host_sorted_row(rv, rok, ord, np);
        in_vals.insert(in_vals.end(), rv.begin(), rv.end());
        in_valids.insert(in_valids.end(), rok.begin(), rok.end());
        ex_vals.insert(ex_vals.end(), ev.begin(), ev.end());
        ex_valids.insert(ex_valids.end(), eok.begin(), eok.end());
        offsets.push_back(static_cast<cudf::size_type>(in_vals.size()));
      }
      auto const num_rows  = static_cast<cudf::size_type>(sizes.size());
      auto const make_list = [&](std::vector<T> const& vals, std::vector<bool> const& valids) {
        cudf::test::fixed_width_column_wrapper<T> leaf(vals.begin(), vals.end(), valids.begin());
        cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
        return cudf::make_lists_column(num_rows, off.release(), leaf.release(), 0, {});
      };
      auto const input    = make_list(in_vals, in_valids);
      auto const expected = make_list(ex_vals, ex_valids);
      expect_both_sort_paths_match(
        cudf::lists_column_view{input->view()}, expected->view(), ord, np);
    }
  };
  run(int32_t{}, int32_t{1});
  run(int64_t{}, int64_t{5'000'000'000});  // distinct values beyond 32 bits
}

// Descending encodes the MINIMUM element to the all-ones raw-key pad sentinel -- the mirror of
// the ascending INT*_MAX collision -- so `rv[1] = min` in every row proves the pad tie-breaks
// keep it inside the segment across the 8/9, 16/17, 32/33, and 64/65 boundaries.
TEST_F(SortListsInt, NumericTieredBitonicDescendingPolarityMatrix)
{
  std::vector<cudf::size_type> const sizes{8, 9, 16, 17, 32, 33, 48, 64, 65};
  auto const run = [&](auto tag, auto spread) {
    using T = decltype(tag);
    for (auto const& [ord, np] : non_default_combos) {
      std::vector<T> in_vals;
      std::vector<T> ex_vals;
      std::vector<cudf::size_type> offsets{0};
      for (auto const s : sizes) {
        std::vector<T> rv(s);
        for (cudf::size_type i = 0; i < s; ++i) {
          rv[i] = static_cast<T>((s - i) * spread);
        }
        rv[0] = std::numeric_limits<T>::max();
        rv[1] = std::numeric_limits<T>::min();  // descending: encodes to the raw-key pad sentinel
        rv[2] = rv[3];                          // a duplicate value
        std::vector<bool> const rok(s, true);
        auto const ev = host_sorted_row(rv, rok, ord, np).first;
        in_vals.insert(in_vals.end(), rv.begin(), rv.end());
        ex_vals.insert(ex_vals.end(), ev.begin(), ev.end());
        offsets.push_back(static_cast<cudf::size_type>(in_vals.size()));
      }
      auto const num_rows  = static_cast<cudf::size_type>(sizes.size());
      auto const make_list = [&](std::vector<T> const& vals) {
        cudf::test::fixed_width_column_wrapper<T> leaf(vals.begin(), vals.end());
        cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
        return cudf::make_lists_column(num_rows, off.release(), leaf.release(), 0, {});
      };
      auto const input    = make_list(in_vals);
      auto const expected = make_list(ex_vals);
      expect_both_sort_paths_match(
        cudf::lists_column_view{input->view()}, expected->view(), ord, np);
    }
  };
  run(int32_t{}, int32_t{1});
  run(int64_t{}, int64_t{5'000'000'000});  // distinct values beyond 32 bits
}

// Packed-radix long lists under the non-shipped combos, across all three key widths and every
// Site A range gate; the shared-high-word two-phase tie pass must resolve complemented low words.
TEST_F(SortListsInt, NumericPackedRadixLongListsPolarityMatrix)
{
  cudf::size_type const n = 220;  // single row, average 110 -> packed radix
  auto const check_fixed  = [&](auto tag, auto spread) {
    using T = decltype(tag);
    std::vector<T> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = static_cast<T>((n / 2 - i) * spread);
    }
    std::vector<bool> const ok(n, true);
    auto const input =
      as_single_row_list(cudf::test::fixed_width_column_wrapper<T>(in.begin(), in.end()).release());
    for (auto const& [ord, np] : non_default_combos) {
      auto const ev       = host_sorted_row(in, ok, ord, np).first;
      auto const expected = as_single_row_list(
        cudf::test::fixed_width_column_wrapper<T>(ev.begin(), ev.end()).release());
      expect_both_sort_paths_match(
        cudf::lists_column_view{input->view()}, expected->view(), ord, np);
    }
  };
  check_fixed(int32_t{}, int32_t{1});
  check_fixed(int64_t{}, int64_t{5'000'000'000});

  auto constexpr scale     = numeric::scale_type{0};
  auto const check_decimal = [&](std::vector<__int128_t> const& in) {
    std::vector<bool> const ok(in.size(), true);
    auto const input = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(in.begin(), in.end(), scale).release());
    for (auto const& [ord, np] : non_default_combos) {
      auto const ev       = host_sorted_row(in, ok, ord, np).first;
      auto const expected = as_single_row_list(
        cudf::test::fixed_point_column_wrapper<__int128_t>(ev.begin(), ev.end(), scale).release());
      expect_both_sort_paths_match(
        cudf::lists_column_view{input->view()}, expected->view(), ord, np);
    }
  };
  {  // range < 2^32 -> min-biased uint64 key (the biased value is complemented in-field)
    std::vector<__int128_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = static_cast<__int128_t>(n / 2 - i);
    }
    check_decimal(in);
  }
  {  // fits int64 (range > 2^32) -> prefix_key96 key
    auto const step = static_cast<__int128_t>(int64_t{1} << 40);
    std::vector<__int128_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = static_cast<__int128_t>(n / 2 - i) * step;
    }
    check_decimal(in);
  }
  {  // wide, distinct high words -> two-phase, phase one resolves
    auto const step = static_cast<__int128_t>(1) << 90;
    std::vector<__int128_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = static_cast<__int128_t>(n / 2 - i) * step;
    }
    check_decimal(in);
  }
  {  // wide with shared high words -> the two-phase tie pass orders complemented low words
    auto const base = static_cast<__int128_t>(1) << 100;
    std::vector<__int128_t> in;
    for (int i = 0; i < 110; ++i) {
      in.push_back(-base + i);
    }
    for (int i = 0; i < 110; ++i) {
      in.push_back(base + i);
    }
    check_decimal(in);
  }
}

// The Site B two-phase tie pass under every combo: the nulls-first combos flip the key's class
// bit, so the tie detector's parameterized null flag must still exclude exactly the nulls.
TEST_F(SortListsInt, NumericDecimal128SiteBPolarityMatrix)
{
  auto constexpr scale    = numeric::scale_type{0};
  cudf::size_type const n = 200;
  auto const base         = static_cast<__int128_t>(1) << 100;
  std::vector<__int128_t> in;
  std::vector<bool> ok;
  for (int i = 0; i < 100; ++i) {
    in.push_back(-base + i);
    ok.push_back(i % 7 != 3);
  }
  for (int i = 0; i < 100; ++i) {
    in.push_back(base + i);
    ok.push_back(i % 5 != 2);
  }
  ASSERT_EQ(static_cast<cudf::size_type>(in.size()), n);
  auto const input = as_single_row_list(
    cudf::test::fixed_point_column_wrapper<__int128_t>(in.begin(), in.end(), ok.begin(), scale)
      .release());
  for (auto const& [ord, np] : non_default_combos) {
    auto const [ev, eok] = host_sorted_row(in, ok, ord, np);
    auto const expected  = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(ev.begin(), ev.end(), eok.begin(), scale)
        .release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view(), ord, np);
  }
}

// The router averages a single row of n as n / 2: check(6) scores 3 (tiny -> tiered network) and
// check(50) scores 25 (past the CUB band -> tiered warp), so both route tiered regardless of
// nulls; the nulls exercise tiered_key128's null-class assignment under the descending complement.
TEST_F(SortListsInt, NumericDecimal128TieredNetworkAndWarpPolarityMatrix)
{
  auto constexpr scale = numeric::scale_type{0};
  auto const check     = [&](cudf::size_type n) {
    std::vector<__int128_t> in(n);
    std::vector<bool> ok(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = static_cast<__int128_t>(n / 2 - i) * (i % 2 == 0 ? 1 : -1);
      ok[i] = i % 4 != 1;  // scattered element nulls
    }
    auto const input = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(in.begin(), in.end(), ok.begin(), scale)
        .release());
    for (auto const& [ord, np] : non_default_combos) {
      auto const [ev, eok] = host_sorted_row(in, ok, ord, np);
      auto const expected  = as_single_row_list(
        cudf::test::fixed_point_column_wrapper<__int128_t>(ev.begin(), ev.end(), eok.begin(), scale)
          .release());
      expect_both_sort_paths_match(
        cudf::lists_column_view{input->view()}, expected->view(), ord, np);
    }
  };
  check(6);   // Network tier (<= TIERED_NETWORK_CAP == 8).
  check(50);  // Warp tier (<= TIERED_WARP_CAP == 64).
}

// Descending routes the lifted CUB band to its `SortPairsDescending` variant; the BEFORE combos
// pin the zero-null precedence relaxation on that band.
TEST_F(SortListsInt, NumericMidBandDecimal128CubPolarityMatrix)
{
  auto constexpr scale = numeric::scale_type{0};
  auto const k         = [] {
    __int128_t v = 1;
    for (int i = 0; i < 30; ++i) {
      v *= 10;
    }
    return v;
  }();
  cudf::size_type const n = 16;
  std::vector<__int128_t> in(n);
  for (cudf::size_type i = 0; i < n; ++i) {
    in[i] = static_cast<__int128_t>(n / 2 - i) * k;
  }
  std::vector<bool> const ok(n, true);
  auto const input = as_single_row_list(
    cudf::test::fixed_point_column_wrapper<__int128_t>(in.begin(), in.end(), scale).release());
  for (auto const& [ord, np] : non_default_combos) {
    auto const ev       = host_sorted_row(in, ok, ord, np).first;
    auto const expected = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(ev.begin(), ev.end(), scale).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view(), ord, np);
  }
}

// BOOL8/int8/uint32 are packed-supported but not tiered, so their nulls force the packed radix at
// any size; uint32 additionally pins the unsigned encode (no sign flip) under the complement.
TEST_F(SortListsInt, NumericPackedNarrowAndBoolPolarityMatrix)
{
  {  // BOOL8 with nulls: values {t, f, X, t, f, X, f}
    std::vector<bool> const in_valids{true, true, false, true, true, false, true};
    auto const input =
      as_single_row_list(cudf::test::fixed_width_column_wrapper<bool>(
                           {true, false, true, true, false, false, false}, in_valids.begin())
                           .release());
    auto const check = [&](std::vector<bool> const& ex_vals,
                           std::vector<bool> const& ex_valids,
                           cudf::order ord,
                           cudf::null_order np) {
      auto expected = as_single_row_list(cudf::test::fixed_width_column_wrapper<bool>(
                                           ex_vals.begin(), ex_vals.end(), ex_valids.begin())
                                           .release());
      expect_both_sort_paths_match(
        cudf::lists_column_view{input->view()}, expected->view(), ord, np);
    };
    check({false, false, true, true, false, false, false},
          {false, false, true, true, true, true, true},
          cudf::order::DESCENDING,
          cudf::null_order::AFTER);
    check({false, false, false, false, false, true, true},
          {false, false, true, true, true, true, true},
          cudf::order::ASCENDING,
          cudf::null_order::BEFORE);
    check({true, true, false, false, false, false, false},
          {true, true, true, true, true, false, false},
          cudf::order::DESCENDING,
          cudf::null_order::BEFORE);
  }
  {  // int8 with nulls, covering both signed extremes
    std::vector<int8_t> const in{5, -8, 0, 127, -128, 0, 5};
    std::vector<bool> const ok{true, true, false, true, true, false, true};
    for (auto const& [ord, np] : non_default_combos) {
      auto const [ev, eok] = host_sorted_row(in, ok, ord, np);
      auto input           = as_single_row_list(
        cudf::test::fixed_width_column_wrapper<int8_t>(in.begin(), in.end(), ok.begin()).release());
      auto expected = as_single_row_list(
        cudf::test::fixed_width_column_wrapper<int8_t>(ev.begin(), ev.end(), eok.begin())
          .release());
      expect_both_sort_paths_match(
        cudf::lists_column_view{input->view()}, expected->view(), ord, np);
    }
  }
  {  // uint32 with a null: unsigned encode, values past INT32_MAX
    std::vector<uint32_t> const in{4'000'000'000u, 7u, 0u, 0u, 4'000'000'000u};
    std::vector<bool> const ok{true, true, false, true, true};
    for (auto const& [ord, np] : non_default_combos) {
      auto const [ev, eok] = host_sorted_row(in, ok, ord, np);
      auto input           = as_single_row_list(
        cudf::test::fixed_width_column_wrapper<uint32_t>(in.begin(), in.end(), ok.begin())
          .release());
      auto expected = as_single_row_list(
        cudf::test::fixed_width_column_wrapper<uint32_t>(ev.begin(), ev.end(), eok.begin())
          .release());
      expect_both_sort_paths_match(
        cudf::lists_column_view{input->view()}, expected->view(), ord, np);
    }
  }
}

// Timestamps under the non-shipped combos for both rep widths: the signed rep flip composed with
// the descending complement must order pre-epoch values last.
TEST_F(SortListsInt, NumericTieredTimestampsPolarityMatrix)
{
  auto const run = [&](auto rep_tag, auto wrapper_tag, std::vector<decltype(rep_tag)> const& in) {
    using Rep     = decltype(rep_tag);
    using Wrapper = decltype(wrapper_tag);
    std::vector<bool> const ok{true, true, false, true, true};
    for (auto const& [ord, np] : non_default_combos) {
      auto const [ev, eok] = host_sorted_row(in, ok, ord, np);
      auto input           = as_single_row_list(
        cudf::test::fixed_width_column_wrapper<Wrapper, Rep>(in.begin(), in.end(), ok.begin())
          .release());
      auto expected = as_single_row_list(
        cudf::test::fixed_width_column_wrapper<Wrapper, Rep>(ev.begin(), ev.end(), eok.begin())
          .release());
      expect_both_sort_paths_match(
        cudf::lists_column_view{input->view()}, expected->view(), ord, np);
    }
  };
  run(int32_t{}, cudf::timestamp_D{}, std::vector<int32_t>{100, -50, 0, 0, 25});
  auto constexpr big = int64_t{5} * 1'000 * 1'000 * 1'000;  // 5e9 > INT32_MAX
  run(int64_t{}, cudf::timestamp_ms{}, std::vector<int64_t>{big, -big, 0, 1, -1});
  // The is_duration branch of radix_encode is textually distinct, so cover one duration per width.
  run(int32_t{}, cudf::duration_D{}, std::vector<int32_t>{100, -50, 0, 0, 25});
  run(int64_t{}, cudf::duration_us{}, std::vector<int64_t>{big, -big, 0, 1, -1});
}

// The block tier's band seams (63/64/65, 128/129, 256/257, 512/513, 1024/1025) per key width;
// DECIMAL128 routes tiered here because the CUB lift declines this column's oversized segments.
TEST_F(SortListsInt, NumericTieredBlockTierBoundaries)
{
  check_block_tier_boundaries<int32_t>(false, cudf::order::ASCENDING);
  check_block_tier_boundaries<int64_t>(false, cudf::order::ASCENDING);
  check_block_tier_boundaries<float>(false, cudf::order::ASCENDING);
  check_block_tier_boundaries<__int128_t>(false, cudf::order::ASCENDING);
}

// The same seams with nulls: every block band must place nulls purely through the packed key's
// class flag.
TEST_F(SortListsInt, NumericTieredBlockTierBoundariesWithNulls)
{
  check_block_tier_boundaries<int32_t>(true, cudf::order::ASCENDING);
  check_block_tier_boundaries<int64_t>(true, cudf::order::ASCENDING);
  check_block_tier_boundaries<float>(true, cudf::order::ASCENDING);
  check_block_tier_boundaries<__int128_t>(true, cudf::order::ASCENDING);
}

// The same seams under DESCENDING, with and without nulls (nulls lead under DESCENDING + AFTER).
TEST_F(SortListsInt, NumericTieredBlockTierBoundariesDescending)
{
  check_block_tier_boundaries<int32_t>(false, cudf::order::DESCENDING);
  check_block_tier_boundaries<int64_t>(false, cudf::order::DESCENDING);
  check_block_tier_boundaries<float>(false, cudf::order::DESCENDING);
  check_block_tier_boundaries<__int128_t>(false, cudf::order::DESCENDING);
  check_block_tier_boundaries<int32_t>(true, cudf::order::DESCENDING);
  check_block_tier_boundaries<int64_t>(true, cudf::order::DESCENDING);
  check_block_tier_boundaries<float>(true, cudf::order::DESCENDING);
  check_block_tier_boundaries<__int128_t>(true, cudf::order::DESCENDING);
}

TEST_F(SortListsInt, NestedListElement)
{
  using T = int;
  // Column of LIST<LIST<int>>: each row's inner lists are reordered as whole elements. The third
  // row's inner lists tie on their first element, so ordering falls through to the second.
  LCW<T> input{LCW<T>{{3, 1}, {2, 0}}, LCW<T>{{5, 5}, {4, 9}}, LCW<T>{{1, 3}, {1, 2}}};
  {
    // Ascending.
    LCW<T> expected{LCW<T>{{2, 0}, {3, 1}}, LCW<T>{{4, 9}, {5, 5}}, LCW<T>{{1, 2}, {1, 3}}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
  {
    // Descending reverses each row's ascending order.
    LCW<T> expected{LCW<T>{{3, 1}, {2, 0}}, LCW<T>{{5, 5}, {4, 9}}, LCW<T>{{1, 3}, {1, 2}}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{input}, cudf::order::DESCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
}

// E = LIST<STRUCT<int, int>>: a list-of-struct element type sorts; struct ranks are computed
// internally before the lexicographic comparison.
TEST_F(SortListsInt, ListOfStructElement)
{
  // One row with two elements [{3, 30}] and [{1, 10}]; ascending reorders to [{1, 10}], [{3, 30}].
  cudf::test::fixed_width_column_wrapper<int> in_f0{3, 1};
  cudf::test::fixed_width_column_wrapper<int> in_f1{30, 10};
  cudf::test::structs_column_wrapper in_structs{{in_f0, in_f1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> in_inner_off{0, 1, 2};
  auto in_inner = cudf::make_lists_column(2, in_inner_off.release(), in_structs.release(), 0, {});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> in_outer_off{0, 2};
  auto in_outer = cudf::make_lists_column(1, in_outer_off.release(), std::move(in_inner), 0, {});

  cudf::test::fixed_width_column_wrapper<int> ex_f0{1, 3};
  cudf::test::fixed_width_column_wrapper<int> ex_f1{10, 30};
  cudf::test::structs_column_wrapper ex_structs{{ex_f0, ex_f1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> ex_inner_off{0, 1, 2};
  auto ex_inner = cudf::make_lists_column(2, ex_inner_off.release(), ex_structs.release(), 0, {});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> ex_outer_off{0, 2};
  auto ex_outer = cudf::make_lists_column(1, ex_outer_off.release(), std::move(ex_inner), 0, {});

  auto const [sorted_lists, stable_sorted_lists] =
    generate_sorted_lists(cudf::lists_column_view{in_outer->view()}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), ex_outer->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), ex_outer->view());
}

// E = STRUCT<int, LIST<int>>: a struct-with-list-field element type sorts.
TEST_F(SortListsInt, StructOfListElement)
{
  // One row with two struct elements {2, [9, 0]} and {1, [8, 7]}; ascending reorders them to
  // {1, [8, 7]}, {2, [9, 0]}.
  cudf::test::fixed_width_column_wrapper<int> in_f0{2, 1};
  cudf::test::lists_column_wrapper<int, int32_t> in_f1{{9, 0}, {8, 7}};
  cudf::test::structs_column_wrapper in_structs{{in_f0, in_f1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> in_off{0, 2};
  auto in_list = cudf::make_lists_column(1, in_off.release(), in_structs.release(), 0, {});

  cudf::test::fixed_width_column_wrapper<int> ex_f0{1, 2};
  cudf::test::lists_column_wrapper<int, int32_t> ex_f1{{8, 7}, {9, 0}};
  cudf::test::structs_column_wrapper ex_structs{{ex_f0, ex_f1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> ex_off{0, 2};
  auto ex_list = cudf::make_lists_column(1, ex_off.release(), ex_structs.release(), 0, {});

  auto const [sorted_lists, stable_sorted_lists] =
    generate_sorted_lists(cudf::lists_column_view{in_list->view()}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), ex_list->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), ex_list->view());
}

TEST_F(SortListsInt, Sliced)
{
  using T = int;
  LCW<T> l{{3, 2, 1, 4}, {7, 5, 6}, {8, 9}, {10}};

  {
    auto const sliced_list = cudf::slice(l, {0, 4})[0];
    auto const expected    = LCW<T>{{1, 2, 3, 4}, {5, 6, 7}, {8, 9}, {10}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{sliced_list}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    auto const sliced_list = cudf::slice(l, {1, 4})[0];
    auto const expected    = LCW<T>{{5, 6, 7}, {8, 9}, {10}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{sliced_list}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    auto const sliced_list = cudf::slice(l, {1, 2})[0];
    auto const expected    = LCW<T>{{5, 6, 7}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{sliced_list}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    auto const sliced_list = cudf::slice(l, {0, 2})[0];
    auto const expected    = LCW<T>{{1, 2, 3, 4}, {5, 6, 7}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{sliced_list}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
}

using SortListsDouble = SortLists<double>;
TEST_F(SortListsDouble, InfinityAndNaN)
{
  auto constexpr NaN = std::numeric_limits<double>::quiet_NaN();
  auto constexpr Inf = std::numeric_limits<double>::infinity();

  using LCW = cudf::test::lists_column_wrapper<double>;
  {
    LCW input{-0.0, -NaN, -NaN, NaN, Inf, -Inf, 7, 5, 6, NaN, Inf, -Inf, -NaN, -NaN, -0.0};
    auto [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{input}, {}, {});
    LCW expected{-Inf, -Inf, -0, -0, 5, 6, 7, Inf, Inf, -NaN, -NaN, NaN, NaN, -NaN, -NaN};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(stable_sorted_lists->view(), expected);
  }
  // Over 200 no-null elements route the unstable `sort_lists` through the packed-radix engine;
  // the EQUIVALENT assertions tolerate either engine's ordering.
  {
    // clang-format off
    LCW input{0.0, -0.0, -NaN, -NaN, NaN, Inf, -Inf,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
              NaN, Inf, -Inf, -NaN, -NaN, -0.0, 0.0};
    LCW expected{-Inf, -Inf, 0.0, -0.0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0, 0,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
               3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
               4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
               5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
               6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
               7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
               8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
               9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
              Inf, Inf, -NaN, -NaN, NaN, NaN, -NaN, -NaN};
    // clang-format on
    auto [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{input}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(stable_sorted_lists->view(), expected);
  }
}

// cudf orders -Inf < finite < +Inf < NaN (all payloads equal, last), nulls after NaN. Every NaN
// canonicalizes to the all-ones key, so an un-canonicalized -NaN cannot sort near -Inf; the two
// rows land in the warp tier, covering both key widths.
TEST_F(SortListsDouble, NumericPackedFloatNaNInfinity)
{
  {  // FLOAT64: the 8-byte unsigned __int128 tiered key; thirteen elements land in the warp tier.
    auto constexpr NaN    = std::numeric_limits<double>::quiet_NaN();
    auto constexpr Inf    = std::numeric_limits<double>::infinity();
    auto constexpr denorm = std::numeric_limits<double>::denorm_min();
    auto constexpr Max    = std::numeric_limits<double>::max();  // DBL_MAX boundary vs. Inf
    using LCWd            = cudf::test::lists_column_wrapper<double>;
    LCWd input{
      {{NaN, -Inf, -0.0, 3.5, -NaN, Inf, 0.0, denorm, -2.0, 0.0, Inf, Max, -Max}, null_at(9)}};
    LCWd expected{
      {{-Inf, -Max, -2.0, -0.0, 0.0, denorm, 3.5, Max, Inf, Inf, NaN, -NaN, 0.0}, null_at(12)}};
    expect_both_sort_paths_equivalent(cudf::lists_column_view{input}, expected);
  }
  {  // FLOAT32: the <= 4-byte uint64 tiered key; ten elements land in the warp tier.
    auto constexpr NaN    = std::numeric_limits<float>::quiet_NaN();
    auto constexpr Inf    = std::numeric_limits<float>::infinity();
    auto constexpr denorm = std::numeric_limits<float>::denorm_min();
    auto constexpr Max    = std::numeric_limits<float>::max();  // FLT_MAX boundary vs. Inf
    using LCWf            = cudf::test::lists_column_wrapper<float>;
    LCWf input{{{-NaN, Inf, -0.0f, 2.5f, -Inf, denorm, 0.0f, NaN, Max, -Max}, null_at(6)}};
    LCWf expected{{{-Inf, -Max, -0.0f, denorm, 2.5f, Max, Inf, NaN, -NaN, 0.0f}, null_at(9)}};
    expect_both_sort_paths_equivalent(cudf::lists_column_view{input}, expected);
  }
}

// NaN ranks above +Inf, so a descending output leads with the NaN block (after any leading
// nulls); the descending complement maps the canonical all-ones NaN key to the smallest valid key.
TEST_F(SortListsDouble, NumericPackedFloatNaNInfinityPolarityMatrix)
{
  {  // FLOAT64: thirteen elements (one null) land in the warp tier.
    auto constexpr NaN    = std::numeric_limits<double>::quiet_NaN();
    auto constexpr Inf    = std::numeric_limits<double>::infinity();
    auto constexpr denorm = std::numeric_limits<double>::denorm_min();
    auto constexpr Max    = std::numeric_limits<double>::max();
    using LCWd            = cudf::test::lists_column_wrapper<double>;
    LCWd input{
      {{NaN, -Inf, -0.0, 3.5, -NaN, Inf, 0.0, denorm, -2.0, 0.0, Inf, Max, -Max}, null_at(9)}};
    {  // DESCENDING / AFTER: null first, then NaNs, then values descending
      LCWd expected{
        {{0.0, NaN, -NaN, Inf, Inf, Max, 3.5, denorm, 0.0, -0.0, -2.0, -Max, -Inf}, null_at(0)}};
      expect_both_sort_paths_equivalent(
        cudf::lists_column_view{input}, expected, cudf::order::DESCENDING, cudf::null_order::AFTER);
    }
    {  // ASCENDING / BEFORE: null first, then values ascending, NaNs last
      LCWd expected{
        {{0.0, -Inf, -Max, -2.0, -0.0, 0.0, denorm, 3.5, Max, Inf, Inf, NaN, -NaN}, null_at(0)}};
      expect_both_sort_paths_equivalent(
        cudf::lists_column_view{input}, expected, cudf::order::ASCENDING, cudf::null_order::BEFORE);
    }
    {  // DESCENDING / BEFORE: NaNs first, values descending, null last
      LCWd expected{
        {{NaN, -NaN, Inf, Inf, Max, 3.5, denorm, 0.0, -0.0, -2.0, -Max, -Inf, 0.0}, null_at(12)}};
      expect_both_sort_paths_equivalent(cudf::lists_column_view{input},
                                        expected,
                                        cudf::order::DESCENDING,
                                        cudf::null_order::BEFORE);
    }
  }
  {  // FLOAT32: ten elements (one null) land in the warp tier.
    auto constexpr NaN    = std::numeric_limits<float>::quiet_NaN();
    auto constexpr Inf    = std::numeric_limits<float>::infinity();
    auto constexpr denorm = std::numeric_limits<float>::denorm_min();
    auto constexpr Max    = std::numeric_limits<float>::max();
    using LCWf            = cudf::test::lists_column_wrapper<float>;
    LCWf input{{{-NaN, Inf, -0.0f, 2.5f, -Inf, denorm, 0.0f, NaN, Max, -Max}, null_at(6)}};
    {  // DESCENDING / AFTER
      LCWf expected{{{0.0f, NaN, -NaN, Inf, Max, 2.5f, denorm, -0.0f, -Max, -Inf}, null_at(0)}};
      expect_both_sort_paths_equivalent(
        cudf::lists_column_view{input}, expected, cudf::order::DESCENDING, cudf::null_order::AFTER);
    }
    {  // ASCENDING / BEFORE
      LCWf expected{{{0.0f, -Inf, -Max, -0.0f, denorm, 2.5f, Max, Inf, NaN, -NaN}, null_at(0)}};
      expect_both_sort_paths_equivalent(
        cudf::lists_column_view{input}, expected, cudf::order::ASCENDING, cudf::null_order::BEFORE);
    }
    {  // DESCENDING / BEFORE
      LCWf expected{{{NaN, -NaN, Inf, Max, 2.5f, denorm, -0.0f, -Max, -Inf, 0.0f}, null_at(9)}};
      expect_both_sort_paths_equivalent(cudf::lists_column_view{input},
                                        expected,
                                        cudf::order::DESCENDING,
                                        cudf::null_order::BEFORE);
    }
  }
}
