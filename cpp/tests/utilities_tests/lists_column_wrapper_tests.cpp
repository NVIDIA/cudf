/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/iterator>

struct ListColumnWrapperTest : public cudf::test::BaseFixture {};
template <typename T>
struct ListColumnWrapperTestTyped : public cudf::test::BaseFixture {
  ListColumnWrapperTestTyped() {}

  auto data_type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::DurationTypes,
                                                  cudf::test::TimestampTypes>;
TYPED_TEST_SUITE(ListColumnWrapperTestTyped, FixedWidthTypesNotBool);

TYPED_TEST(ListColumnWrapperTestTyped, List)
{
  using T = TypeParam;

  // List<T>, 1 row
  //
  // List<T>:
  // Length : 1
  // Offsets : 0, 2
  // Children :
  //    2, 3
  //
  {
    cudf::test::lists_column_wrapper<T, int32_t> list{2, 3};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 2);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 2});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto data = lcv.child();
    EXPECT_EQ(data.size(), 2);
    cudf::test::fixed_width_column_wrapper<T, int32_t> e_data({2, 3});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_data, data);
  }

  // List<T>, 1 row
  //
  // List<T>:
  // Length : 1
  // Offsets : 0, 2
  // Children :
  //    2, 3
  //
  {
    cudf::test::lists_column_wrapper<T, int32_t> list{{2, 3}};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 2);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 2});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto data = lcv.child();
    EXPECT_EQ(data.size(), 2);
    cudf::test::fixed_width_column_wrapper<T, int32_t> e_data({2, 3});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_data, data);
  }
}

TYPED_TEST(ListColumnWrapperTestTyped, ListWithValidity)
{
  using T = TypeParam;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  // List<T>, 1 row
  //
  // List<T>:
  // Length : 1
  // Offsets : 0, 2
  // Children :
  //    2, NULL
  //
  {
    cudf::test::lists_column_wrapper<T, int32_t> list{{{2, 3}, valids}};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 2);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 2});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto data = lcv.child();
    EXPECT_EQ(data.size(), 2);
    cudf::test::fixed_width_column_wrapper<T, int32_t> e_data({2, 3}, valids);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_data, data);
  }

  // List<T>, 3 rows
  //
  // List<T>:
  // Length : 3
  // Offsets : 0, 2, 4, 7
  // Children :
  //    2, NULL, 4, NULL, 6, NULL, 8
  {
    cudf::test::lists_column_wrapper<T, int32_t> list{
      {{2, 3}, valids}, {{4, 5}, valids}, {{6, 7, 8}, valids}};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 2, 4, 7});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto data = lcv.child();
    EXPECT_EQ(data.size(), 7);
    cudf::test::fixed_width_column_wrapper<T, int32_t> e_data({2, 3, 4, 5, 6, 7, 8}, valids);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_data, data);
  }
}

TYPED_TEST(ListColumnWrapperTestTyped, ListFromIterator)
{
  using T = TypeParam;

  // List<T>, 1 row
  //
  // List<T>:
  // Length : 1
  // Offsets : 0, 5
  // Children :
  //    0, 1, 2, 3, 4
  //
  auto sequence = cuda::counting_iterator{0};

  cudf::test::lists_column_wrapper<T, typename decltype(sequence)::value_type> list{sequence,
                                                                                    sequence + 5};

  cudf::lists_column_view lcv(list);
  EXPECT_EQ(lcv.size(), 1);

  auto offsets = lcv.offsets();
  EXPECT_EQ(offsets.size(), 2);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 5});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

  auto data = lcv.child();
  EXPECT_EQ(data.size(), 5);
  cudf::test::fixed_width_column_wrapper<T, int32_t> e_data({0, 1, 2, 3, 4});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_data, data);
}

TYPED_TEST(ListColumnWrapperTestTyped, ListFromIteratorWithValidity)
{
  using T = TypeParam;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  // List<int>, 1 row
  //
  // List<int32_t>:
  // Length : 1
  // Offsets : 0, 5
  // Children :
  //    0, NULL, 2, NULL, 4
  //
  auto sequence = cuda::counting_iterator{0};

  cudf::test::lists_column_wrapper<T, typename decltype(sequence)::value_type> list{
    sequence, sequence + 5, valids};

  cudf::lists_column_view lcv(list);
  EXPECT_EQ(lcv.size(), 1);

  auto offsets = lcv.offsets();
  EXPECT_EQ(offsets.size(), 2);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 5});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

  auto data = lcv.child();
  EXPECT_EQ(data.size(), 5);
  cudf::test::fixed_width_column_wrapper<T, int32_t> e_data({0, 0, 2, 0, 4}, valids);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_data, data);
}

TYPED_TEST(ListColumnWrapperTestTyped, ListOfLists)
{
  using T = TypeParam;

  // List<List<T>>, 1 row
  //
  // List<List<T>>:
  // Length : 1
  // Offsets : 0, 2
  // Children :
  //    List<T>:
  //    Length : 2
  //    Offsets : 0, 2, 4
  //    Children :
  //      2, 3, 4, 5
  {
    cudf::test::lists_column_wrapper<T, int32_t> list{{{2, 3}, {4, 5}}};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 2);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 2});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto child = lcv.child();
    cudf::lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 2);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 3);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_offsets({0, 2, 4});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_offsets, child_offsets);

    auto child_data = childv.child();
    EXPECT_EQ(child_data.size(), 4);
    cudf::test::fixed_width_column_wrapper<T, int32_t> e_child_data({2, 3, 4, 5});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_data, child_data);
  }

  // List<List<T>> 3 rows
  //
  // List<List<T>>:
  // Length : 3
  // Offsets : 0, 2, 5, 6
  // Children :
  //    List<T>:
  //    Length : 6
  //    Offsets : 0, 2, 4, 7, 8, 9, 11
  //    Children :
  //      1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10
  {
    cudf::test::lists_column_wrapper<T, int32_t> list{
      {{1, 2}, {3, 4}}, {{5, 6, 7}, {0}, {8}}, {{9, 10}}};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 2, 5, 6});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto child = lcv.child();
    cudf::lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 6);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 7);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_offsets({0, 2, 4, 7, 8, 9, 11});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_offsets, child_offsets);

    auto child_data = childv.child();
    EXPECT_EQ(child_data.size(), 11);
    cudf::test::fixed_width_column_wrapper<T, int32_t> e_child_data(
      {1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_data, child_data);
  }
}

TYPED_TEST(ListColumnWrapperTestTyped, ListOfListsWithValidity)
{
  using T = TypeParam;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  // List<List<T>>, 1 row
  //
  // List<List<T>>:
  // Length : 1
  // Offsets : 0, 2
  // Children :
  //    List<T>:
  //    Length : 2
  //    Offsets : 0, 2, 4
  //    Children :
  //      2, NULL, 4, NULL
  {
    // equivalent to { {2, NULL}, {4, NULL} }
    cudf::test::lists_column_wrapper<T, int32_t> list{{{{2, 3}, valids}, {{4, 5}, valids}}};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 2);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 2});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto child = lcv.child();
    cudf::lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 2);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 3);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_offsets({0, 2, 4});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_offsets, child_offsets);

    auto child_data = childv.child();
    EXPECT_EQ(child_data.size(), 4);
    cudf::test::fixed_width_column_wrapper<T, int32_t> e_child_data({2, 3, 4, 5}, valids);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_data, child_data);
  }

  // List<List<T>> 3 rows
  //
  // List<List<T>>:
  // Length : 3
  // Offsets : 0, 2, 5, 6
  // Children :
  //    List<T>:
  //    Length : 6
  //    Offsets : 0, 2, 2, 5, 5, 6, 8
  //    Null count: 2
  //    110101
  //    Children :
  //      1, 2, 5, 6, 7, 8, 9, 10
  {
    // equivalent to  { {{1, 2}, NULL}, {{5, 6, 7}, NULL, {8}}, {{9, 10}} }
    cudf::test::lists_column_wrapper<T, int32_t> list{
      {{{1, 2}, {3, 4}}, valids}, {{{5, 6, 7}, {0}, {8}}, valids}, {{{9, 10}}, valids}};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 2, 5, 6});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto child = lcv.child();
    cudf::lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 6);
    EXPECT_EQ(childv.null_count(), 2);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 7);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_offsets({0, 2, 2, 5, 5, 6, 8});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_offsets, child_offsets);

    auto child_data = childv.child();
    EXPECT_EQ(child_data.size(), 8);
    cudf::test::fixed_width_column_wrapper<T, int32_t> e_child_data({1, 2, 5, 6, 7, 8, 9, 10});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_data, child_data);
  }
}

TYPED_TEST(ListColumnWrapperTestTyped, ListOfListOfListsWithValidity)
{
  using T = TypeParam;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  // List<List<List<T>>>, 2 rows
  //
  // List<List<List<T>>>:
  // Length : 2
  // Offsets : 0, 2, 4
  // Children :
  //    List<List<T>>:
  //    Length : 4
  //    Offsets : 0, 2, 2, 4, 6
  //    Null count: 1
  //    1101
  //    Children :
  //      List<T>:
  //      Length : 6
  //      Offsets : 0, 2, 4, 6, 8, 11, 12
  //      Children :
  //        1, 2, 3, 4, 10, 20, 30, 40, 50, 60, 70, 0
  {
    // equivalent to  { {{{1, 2}, {3, 4}}, NULL}, {{{10, 20}, {30, 40}}, {{50, 60, 70}, {0}}} }
    cudf::test::lists_column_wrapper<T, int32_t> list{
      {{{{1, 2}, {3, 4}}, {{5, 6, 7}, {0}}}, valids}, {{{10, 20}, {30, 40}}, {{50, 60, 70}, {0}}}};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 2);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 3);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 2, 4});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto child = lcv.child();
    cudf::lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 4);
    EXPECT_EQ(childv.null_count(), 1);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 5);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_offsets({0, 2, 2, 4, 6});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_offsets, child_offsets);

    auto child_child = childv.child();
    cudf::lists_column_view child_childv(child_child);
    EXPECT_EQ(child_childv.size(), 6);

    auto child_child_offsets = child_childv.offsets();
    EXPECT_EQ(child_child_offsets.size(), 7);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_child_offsets(
      {0, 2, 4, 6, 8, 11, 12});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_child_offsets, child_child_offsets);

    auto child_child_data = child_childv.child();
    EXPECT_EQ(child_child_data.size(), 12);
    cudf::test::fixed_width_column_wrapper<T, int32_t> e_child_child_data(
      {1, 2, 3, 4, 10, 20, 30, 40, 50, 60, 70, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_child_data, child_child_data);
  }
}

TYPED_TEST(ListColumnWrapperTestTyped, EmptyLists)
{
  using T = TypeParam;

  // to disambiguate between {} == 0 and {} == List{0}
  // Also, see note about compiler issues when declaring nested
  // empty lists in lists_column_wrapper documentation
  using LCW = cudf::test::lists_column_wrapper<T, int32_t>;

  // List<T>, empty
  //
  // List<T>:
  // Length : 0
  // Offsets :
  // Children :
  {
    // equivalent to  {}
    cudf::test::lists_column_wrapper<T> list{};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 0);
  }

  // List<T>, 1 row
  //
  // List<T>:
  // Length : 1
  // Offsets : 0, 0
  // Children :
  {
    // equivalent to  {}
    cudf::test::lists_column_wrapper<T, int32_t> list{LCW{}};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 2);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);
  }

  // List<T>, 2 rows
  //
  // List<T>:
  // Length : 2
  // Offsets : 0, 0, 0
  // Children :
  {
    // equivalent to  {}
    cudf::test::lists_column_wrapper<T, int32_t> list{LCW{}, LCW{}};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 2);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 3);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);
  }

  // List<L>, mixed
  //
  // List<T>:
  // Length : 3
  // Offsets : 0, 2, 2, 4
  // Children :
  //   1, 2, 3, 4
  {
    // equivalent to  {{1, 2}, {}, {3, 4}}

    cudf::test::lists_column_wrapper<T, int32_t> list{{1, 2}, LCW{}, {3, 4}};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 2, 2, 4});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto child_data = lcv.child();
    EXPECT_EQ(child_data.size(), 4);
    cudf::test::fixed_width_column_wrapper<T, int32_t> e_child_data({1, 2, 3, 4});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_data, child_data);
  }

  // List<List<T>>, mixed
  //
  // List<List<T>>:
  // Length : 3
  // Offsets : 0, 1, 4, 7
  // Children :
  //   List<int32_t>:
  //   Length : 7
  //   Offsets : 0, 0, 2, 2, 4, 4, 8, 8
  //   Children :
  //     1, 2, 3, 4, 5, 6, 7, 8
  {
    // equivalent to  { {{}}, {{1, 2}, {}, {3, 4}}, {{}, {5, 6, 7, 8}, {}} }
    cudf::test::lists_column_wrapper<T, int32_t> list{
      {LCW{}}, {{1, 2}, LCW{}, {3, 4}}, {LCW{}, {5, 6, 7, 8}, LCW{}}};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 1, 4, 7});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto child = lcv.child();
    cudf::lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 7);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 8);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_offsets(
      {0, 0, 2, 2, 4, 4, 8, 8});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_offsets, child_offsets);

    auto child_data = childv.child();
    EXPECT_EQ(child_data.size(), 8);
    cudf::test::fixed_width_column_wrapper<T, int32_t> e_child_data({1, 2, 3, 4, 5, 6, 7, 8});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_data, child_data);
  }
}

TYPED_TEST(ListColumnWrapperTestTyped, EmptyListsWithValidity)
{
  using T = TypeParam;

  // to disambiguate between {} == 0 and {} == List{0}
  // Also, see note about compiler issues when declaring nested
  // empty lists in lists_column_wrapper documentation
  using LCW = cudf::test::lists_column_wrapper<T, int32_t>;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  // List<T>, 2 rows
  //
  // List<T>:
  // Length : 2
  // Offsets : 0, 0, 0
  // Null count: 1
  // 01
  // Children :
  {
    // equivalent to  {{}, NULL}
    cudf::test::lists_column_wrapper<T, int32_t> list{{LCW{}, LCW{}}, valids};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 2);
    EXPECT_EQ(lcv.null_count(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 3);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);
  }

  // List<T>, 3 rows
  //
  // List<T>:
  // Length : 3
  // Offsets : 0, 0, 0, 0
  // Null count: 1
  // 101
  // Children :
  {
    // equivalent to  {{}, NULL, {}}
    cudf::test::lists_column_wrapper<T, int32_t> list{{LCW{}, {1, 2, 3}, LCW{}}, valids};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);
    EXPECT_EQ(lcv.null_count(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 0, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);
  }

  // List<T>, 3 rows
  //
  // List<T>:
  // Length : 3
  // Offsets : 0, 0, 0, 3
  // Null count: 1
  // 101
  // Children :
  //   1, 2, 3
  {
    // equivalent to  {{}, NULL, {1, 2, 3}}
    cudf::test::lists_column_wrapper<T, int32_t> list{{LCW{}, LCW{}, {1, 2, 3}}, valids};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);
    EXPECT_EQ(lcv.null_count(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 0, 0, 3});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);
  }

  // List<List<T>>, mixed
  //
  // List<List<T>>:
  // Length : 3
  // Offsets : 0, 1, 1, 4
  // Null count: 1
  // 101
  // Children :
  //   List<T>:
  //   Length : 4
  //   Offsets : 0, 0, 0, 4, 4
  //   Children :
  //     5, 6, 7, 8
  {
    // equivalent to  { {{}}, NULL, {{}, {5, 6, 7, 8}, {}} }
    cudf::test::lists_column_wrapper<T, int32_t> list{
      {{LCW{}}, {{1, 2}, LCW{}, {3, 4}}, {LCW{}, {5, 6, 7, 8}, LCW{}}}, valids};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 1, 1, 4});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto child = lcv.child();
    cudf::lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 4);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 5);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_offsets({0, 0, 0, 4, 4});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_offsets, child_offsets);

    auto child_data = childv.child();
    EXPECT_EQ(child_data.size(), 4);
    cudf::test::fixed_width_column_wrapper<T, int32_t> e_child_data({5, 6, 7, 8});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_data, child_data);
  }
}

TYPED_TEST(ListColumnWrapperTestTyped, IncompleteHierarchies)
{
  using T = TypeParam;

  // to disambiguate between {} == 0 and {} == List{0}
  // Also, see note about compiler issues when declaring nested
  // empty lists in lists_column_wrapper documentation
  using LCW = cudf::test::lists_column_wrapper<T, int32_t>;

  // List<List<List<T>>>:
  // Length : 3
  // Offsets : 0, 1, 2, 2
  // Children :
  //  List<List<T>>:
  //  Length : 2
  //  Offsets : 0, 1, 1
  //  Children :
  //      List<T>:
  //      Length : 1
  //      Offsets : 0, 0
  //      Children :
  {
    cudf::test::lists_column_wrapper<T, int32_t> list{{{LCW{}}}, {LCW{}}, LCW{}};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 1, 2, 2});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto child = lcv.child();
    cudf::lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 2);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 3);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_offsets({0, 1, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_offsets, child_offsets);

    auto child_child = childv.child();
    cudf::lists_column_view child_childv(child_child);
    EXPECT_EQ(child_childv.size(), 1);

    auto child_child_offsets = child_childv.offsets();
    EXPECT_EQ(child_child_offsets.size(), 2);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_child_offsets({0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_child_offsets, child_child_offsets);

    auto child_child_data = child_childv.child();
    EXPECT_EQ(child_child_data.size(), 0);
  }

  // List<List<List<T>>>:
  // Length : 3
  // Offsets : 0, 0, 1, 2
  // Children :
  //   List<List<T>>:
  //   Length : 2
  //   Offsets : 0, 0, 1
  //   Children :
  //     List<T>:
  //       Length : 1
  //       Offsets : 0, 0
  //       Children :
  {
    cudf::test::lists_column_wrapper<T, int32_t> list{LCW{}, {LCW{}}, {{LCW{}}}};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 0, 1, 2});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto child = lcv.child();
    cudf::lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 2);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 3);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_offsets({0, 0, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_offsets, child_offsets);

    auto child_child = childv.child();
    cudf::lists_column_view child_childv(child_child);
    EXPECT_EQ(child_childv.size(), 1);

    auto child_child_offsets = child_childv.offsets();
    EXPECT_EQ(child_child_offsets.size(), 2);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_child_offsets({0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_child_offsets, child_child_offsets);

    auto child_child_data = child_childv.child();
    EXPECT_EQ(child_child_data.size(), 0);
  }

  // List<List<List<T>>>:
  // Length : 3
  // Offsets : 0, 0, 1, 2
  // Children :
  //   List<List<T>>:
  //   Length : 2
  //   Offsets : 0, 1, 1
  //   Children :
  //       List<T>:
  //       Length : 1
  //       Offsets : 0, 3
  //       Children :
  //         1, 2, 3
  {
    // { {}, {{{1,2,3}}}, {{}} }
    cudf::test::lists_column_wrapper<T, int32_t> list{LCW{}, {{{1, 2, 3}}}, {LCW{}}};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 0, 1, 2});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto child = lcv.child();
    cudf::lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 2);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 3);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_offsets({0, 1, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_offsets, child_offsets);

    auto child_child = childv.child();
    cudf::lists_column_view child_childv(child_child);
    EXPECT_EQ(child_childv.size(), 1);

    auto child_child_offsets = child_childv.offsets();
    EXPECT_EQ(child_child_offsets.size(), 2);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_child_offsets({0, 3});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_child_offsets, child_child_offsets);

    auto child_child_data = child_childv.child();
    EXPECT_EQ(child_child_data.size(), 3);
    cudf::test::fixed_width_column_wrapper<T, int32_t> e_child_child_data({1, 2, 3});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_child_data, child_child_data);
  }

  // List<List<List<T>>>:
  // Length : 3
  // Offsets : 0, 1, 2, 2
  // Null count: 1
  // 011
  // Children :
  // List<List<T>>:
  // Length : 2
  // Offsets : 0, 1, 1
  // Children :
  //   List<T>:
  //   Length : 1
  //   Offsets : 0, 0
  //   Children :
  {
    // { {{{}}}, {{}}, null }
    std::vector<bool> valids{true, true, false};
    cudf::test::lists_column_wrapper<T, int32_t> list{{{{LCW{}}}, {LCW{}}, LCW{}}, valids.begin()};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);
    EXPECT_EQ(lcv.null_count(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 1, 2, 2});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto child = lcv.child();
    cudf::lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 2);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 3);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_offsets({0, 1, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_offsets, child_offsets);

    auto child_child = childv.child();
    cudf::lists_column_view child_childv(child_child);
    EXPECT_EQ(child_childv.size(), 1);

    auto child_child_offsets = child_childv.offsets();
    EXPECT_EQ(child_child_offsets.size(), 2);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_child_offsets({0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_child_offsets, child_child_offsets);
  }

  // List<List<List<T>>>:
  // Length : 3
  // Offsets : 0, 1, 1, 2
  // Null count: 1
  // 101
  // Children :
  // List<List<T>>:
  // Length : 1
  // Offsets : 0, 1
  // Children :
  //   List<T>:
  //   Length : 1
  //   Offsets : 0, 0
  //   Children :
  {
    // { {{{}}}, null, {} }
    std::vector<bool> valids{true, false, true};
    cudf::test::lists_column_wrapper<T, int32_t> list{{{{LCW{}}}, {LCW{}}, LCW{}}, valids.begin()};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);
    EXPECT_EQ(lcv.null_count(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 1, 1, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto child = lcv.child();
    cudf::lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 1);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 2);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_offsets({0, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_offsets, child_offsets);

    auto child_child = childv.child();
    cudf::lists_column_view child_childv(child_child);
    EXPECT_EQ(child_childv.size(), 1);

    auto child_child_offsets = child_childv.offsets();
    EXPECT_EQ(child_child_offsets.size(), 2);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_child_offsets({0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_child_offsets, child_child_offsets);
  }

  // List<List<T>>:
  // Length : 3
  // Offsets : 0, 0, 1, 1
  // Null count: 1
  // 110
  // Children :
  //   List<T>:
  //   Length : 1
  //   Offsets : 0, 0
  //   Children :
  {
    // { null, {{}}, {} }
    std::vector<bool> valids{false, true, true};
    cudf::test::lists_column_wrapper<T, int32_t> list{{{{LCW{}}}, {LCW{}}, LCW{}}, valids.begin()};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);
    EXPECT_EQ(lcv.null_count(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 0, 1, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto child = lcv.child();
    cudf::lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 1);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 2);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_offsets({0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_offsets, child_offsets);
  }

  // List<List<>>:
  // Length : 3
  // Offsets : 0, 0, 0, 0
  // Null count: 3
  // 000
  // Children :
  //   List<>:
  //   Length : 0
  //   Offsets :
  //   Children :
  {
    // { null, null, null }
    std::vector<bool> valids{false, false, false};
    cudf::test::lists_column_wrapper<T, int32_t> list{{{{LCW{}}}, {LCW{}}, LCW{}}, valids.begin()};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);
    EXPECT_EQ(lcv.null_count(), 3);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 0, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto child = lcv.child();
    cudf::lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 0);
  }

  // List<List<>>:
  // Length : 3
  // Offsets : 0, 0, 0, 0
  // Null count: 3
  // 000
  // Children :
  //   List<>:
  //   Length : 0
  //   Offsets :
  //   Children :
  {
    // { null, null, null }
    std::vector<bool> valids{false, false, false};
    cudf::test::lists_column_wrapper<T, int32_t> list{{LCW{}, {{LCW{}}}, {LCW{}}}, valids.begin()};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);
    EXPECT_EQ(lcv.null_count(), 3);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 0, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto child = lcv.child();
    cudf::lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 0);
  }

  // List<List<List<>>>:
  // Length : 3
  // Offsets : 0, 1, 2, 2
  // Children :
  //   List<List<>>:
  //   Length : 2
  //   Offsets : 0, 0, 0
  //   Null count: 1
  //   10
  //   Children :
  //      List<>:
  //      Length : 0
  //      Offsets :
  //      Children :
  {
    // { {null}, {{}}, {} }
    std::vector<bool> valids{false};
    cudf::test::lists_column_wrapper<T, int32_t> list{{{{LCW{}}}, valids.begin()}, {LCW{}}, LCW{}};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 1, 2, 2});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto child = lcv.child();
    cudf::lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 2);
    EXPECT_EQ(childv.null_count(), 1);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 3);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_offsets({0, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_offsets, child_offsets);

    auto child_child = childv.child();
    cudf::lists_column_view child_childv(child_child);
    EXPECT_EQ(child_childv.size(), 0);
  }

  // big mix of truncated stuff
  // List<List<List<T>
  // Length : 4
  // Offsets : 0, 1, 3, 3, 5
  // Children :
  //  List<List<T>>:
  //  Length : 5
  //  Offsets : 0, 2, 2, 3, 3, 3
  //  Children :
  //      List<T>:
  //      Length : 3
  //      Offsets : 0, 3, 5, 5
  //      Children :
  //        1, 2, 3, 4, 5

  {
    // { {{{1, 2, 3}, {4, 5}}}, {{}, {{}}}, {}, {{}, {}} }
    cudf::test::lists_column_wrapper<T, int32_t> list{
      {{{1, 2, 3}, {4, 5}}}, {LCW{}, {LCW{}}}, LCW{}, {LCW{}, LCW{}}};
    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 4);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 5);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 1, 3, 3, 5});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto child = lcv.child();
    cudf::lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 5);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 6);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_offsets({0, 2, 2, 3, 3, 3});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_offsets, child_offsets);

    auto child_child = childv.child();
    cudf::lists_column_view child_childv(child_child);
    EXPECT_EQ(child_childv.size(), 3);

    auto child_child_offsets = child_childv.offsets();
    EXPECT_EQ(child_child_offsets.size(), 4);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_child_offsets({0, 3, 5, 5});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_child_offsets, child_child_offsets);

    auto child_child_data = child_childv.child();
    EXPECT_EQ(child_child_data.size(), 5);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_child_data({1, 2, 3, 4, 5});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_child_data, e_child_child_data);
  }
}

TEST_F(ListColumnWrapperTest, ListOfStrings)
{
  // List<string>, 2 rows
  //
  // List<cudf::string_view>:
  // Length : 2
  // Offsets : 0, 2, 5
  // Children :
  //    one, two, three, four, five
  {
    cudf::test::lists_column_wrapper<cudf::string_view> list{{"one", "two"},
                                                             {"three", "four", "five"}};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 2);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 3);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 2, 5});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto data = lcv.child();
    EXPECT_EQ(data.size(), 5);
    cudf::test::strings_column_wrapper e_data({"one", "two", "three", "four", "five"});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_data, data);
  }
}

TEST_F(ListColumnWrapperTest, ListOfListOfStrings)
{
  // List<List<string>>, 2 rows
  //
  // List<List<cudf::string_view>>:
  // Length : 2
  // Offsets : 0, 2, 4
  // Children :
  //    List<cudf::string_view>:
  //    Length : 4
  //    Offsets : 0, 2, 5, 6, 8
  //    Children :
  //      one, two, three, four, five, eight, nine, ten
  {
    cudf::test::lists_column_wrapper<cudf::string_view> list{
      {{"one", "two"}, {"three", "four", "five"}}, {{"eight"}, {"nine", "ten"}}};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 2);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 3);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 2, 4});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto child = lcv.child();
    cudf::lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 4);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 5);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_offsets({0, 2, 5, 6, 8});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_offsets, child_offsets);

    auto child_data = childv.child();
    EXPECT_EQ(child_data.size(), 8);
    cudf::test::strings_column_wrapper e_child_data(
      {"one", "two", "three", "four", "five", "eight", "nine", "ten"});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_data, child_data);
  }
}

TEST_F(ListColumnWrapperTest, ListOfBools)
{
  // List<bool>, 1 row
  //
  // List<bool>:
  // Length : 1
  // Offsets : 0, 2
  // Children :
  //   1, 0
  //
  {
    cudf::test::lists_column_wrapper<bool> list{true, false};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 2);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 2});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto data = lcv.child();
    EXPECT_EQ(data.size(), 2);
    cudf::test::fixed_width_column_wrapper<bool> e_data({true, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_data, data);
  }

  // List<bool>, 1 row
  //
  // List<bool>:
  // Length : 1
  // Offsets : 0, 3
  // Children :
  //   1, 0, 0
  //
  {
    cudf::test::lists_column_wrapper<bool> list{{true, false, false}};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 2);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 3});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto data = lcv.child();
    EXPECT_EQ(data.size(), 3);
    cudf::test::fixed_width_column_wrapper<bool> e_data({true, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_data, data);
  }
}

TEST_F(ListColumnWrapperTest, ListOfBoolsWithValidity)
{
  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  // List<bool>, 3 rows
  //
  // List<bool>:
  // Length : 3
  // Offsets : 0, 2, 4, 7
  // Children :
  //   1, NULL, 0, NULL, 0, NULL, 0
  {
    cudf::test::lists_column_wrapper<bool> list{
      {{true, true}, valids}, {{false, true}, valids}, {{false, true, false}, valids}};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 2, 4, 7});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto data = lcv.child();
    EXPECT_EQ(data.size(), 7);
    cudf::test::fixed_width_column_wrapper<bool> e_data(
      {true, true, false, true, false, true, false}, valids);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_data, data);
  }
}

TEST_F(ListColumnWrapperTest, ListOfListOfBools)
{
  // List<List<bool>> 3 rows
  //
  // List<List<bool>>:
  // Length : 3
  // Offsets : 0, 2, 5, 6
  // Children :
  //    List<bool>:
  //    Length : 6
  //    Offsets : 0, 2, 4, 7, 8, 9, 11
  //    Children :
  //      0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1
  {
    cudf::test::lists_column_wrapper<bool> list{
      {{false, true}, {true, true}}, {{true, false, true}, {true}, {true}}, {{false, true}}};

    cudf::lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 2, 5, 6});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_offsets, offsets);

    auto child = lcv.child();
    cudf::lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 6);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 7);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> e_child_offsets({0, 2, 4, 7, 8, 9, 11});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_offsets, child_offsets);

    auto child_child_data = childv.child();
    EXPECT_EQ(child_child_data.size(), 11);
    cudf::test::fixed_width_column_wrapper<bool> e_child_child_data(
      {false, true, true, true, true, false, true, true, true, false, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e_child_child_data, child_child_data);
  }
}

TEST_F(ListColumnWrapperTest, MismatchedHierarchies)
{
  using T = int;

  // to disambiguate between {} == 0 and {} == List{0}
  // Also, see note about compiler issues when declaring nested
  // empty lists in lists_column_wrapper documentation
  using LCW = cudf::test::lists_column_wrapper<T>;

  // trying to build a column out of a List<List<int>> column, and a List<int> column
  // is not valid if the leaf lists are not empty.
  {
    auto expect_failure = []() { LCW list{{{1, 2, 3}}, {4, 5}}; };
    EXPECT_THROW(expect_failure(), cudf::logic_error);
  }
}

TYPED_TEST(ListColumnWrapperTestTyped, ListsOfStructs)
{
  using T = TypeParam;

  auto num_struct_rows = 8;
  auto numeric_column  = cudf::test::fixed_width_column_wrapper<T, int32_t>{1, 2, 3, 4, 5, 6, 7, 8};
  auto bool_column     = cudf::test::fixed_width_column_wrapper<bool>{1, 1, 1, 1, 0, 0, 0, 0};
  auto struct_column = cudf::test::structs_column_wrapper{{numeric_column, bool_column}}.release();
  EXPECT_EQ(struct_column->size(), num_struct_rows);
  EXPECT_TRUE(!struct_column->nullable());

  auto lists_column_offsets = cudf::test::fixed_width_column_wrapper<int32_t>{0, 2, 4, 8}.release();
  auto num_lists            = lists_column_offsets->size() - 1;
  auto lists_column =
    make_lists_column(num_lists, std::move(lists_column_offsets), std::move(struct_column), 0, {});

  // Check if child column is unchanged.

  auto expected_numeric_column =
    cudf::test::fixed_width_column_wrapper<T, int32_t>{1, 2, 3, 4, 5, 6, 7, 8};
  auto expected_bool_column = cudf::test::fixed_width_column_wrapper<bool>{1, 1, 1, 1, 0, 0, 0, 0};
  auto expected_struct_column =
    cudf::test::structs_column_wrapper{{expected_numeric_column, expected_bool_column}}.release();

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_struct_column,
                                 cudf::lists_column_view(*lists_column).child());
}

TYPED_TEST(ListColumnWrapperTestTyped, ListsOfStructsWithValidity)
{
  using T = TypeParam;

  auto num_struct_rows = 8;
  auto numeric_column  = cudf::test::fixed_width_column_wrapper<T, int32_t>{
    {1, 2, 3, 4, 5, 6, 7, 8}, {1, 1, 1, 1, 0, 0, 0, 0}};
  auto bool_column   = cudf::test::fixed_width_column_wrapper<bool>{1, 1, 1, 1, 0, 0, 0, 0};
  auto struct_column = cudf::test::structs_column_wrapper{{numeric_column, bool_column}}.release();
  EXPECT_EQ(struct_column->size(), num_struct_rows);
  EXPECT_TRUE(!struct_column->nullable());

  auto lists_column_offsets = cudf::test::fixed_width_column_wrapper<int32_t>{0, 2, 4, 8}.release();
  auto list_null_mask       = {1, 1, 0};
  auto num_lists            = lists_column_offsets->size() - 1;
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(list_null_mask.begin(), list_null_mask.end());
  auto lists_column = [&] {
    auto tmp = cudf::make_lists_column(num_lists,
                                       std::move(lists_column_offsets),
                                       std::move(struct_column),
                                       null_count,
                                       std::move(null_mask));
    return cudf::purge_nonempty_nulls(tmp->view());
  }();

  // Check if child column is unchanged.

  auto expected_numeric_column =
    cudf::test::fixed_width_column_wrapper<T, int32_t>{{1, 2, 3, 4}, {1, 1, 1, 1}};
  auto expected_bool_column = cudf::test::fixed_width_column_wrapper<bool>{1, 1, 1, 1};
  auto expected_struct_column =
    cudf::test::structs_column_wrapper{{expected_numeric_column, expected_bool_column}}.release();

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_struct_column,
                                 cudf::lists_column_view(*lists_column).child());
}

TYPED_TEST(ListColumnWrapperTestTyped, ListsOfListsOfStructs)
{
  using T = TypeParam;

  auto num_struct_rows = 8;
  auto numeric_column  = cudf::test::fixed_width_column_wrapper<T, int32_t>{1, 2, 3, 4, 5, 6, 7, 8};
  auto bool_column     = cudf::test::fixed_width_column_wrapper<bool>{1, 1, 1, 1, 0, 0, 0, 0};
  auto struct_column = cudf::test::structs_column_wrapper{{numeric_column, bool_column}}.release();
  EXPECT_EQ(struct_column->size(), num_struct_rows);
  EXPECT_TRUE(!struct_column->nullable());

  auto lists_column_offsets = cudf::test::fixed_width_column_wrapper<int32_t>{0, 2, 4, 8}.release();
  auto num_lists            = lists_column_offsets->size() - 1;
  auto lists_column =
    make_lists_column(num_lists, std::move(lists_column_offsets), std::move(struct_column), 0, {});

  auto lists_of_lists_column_offsets =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 2, 3}.release();
  auto num_lists_of_lists               = lists_of_lists_column_offsets->size() - 1;
  auto lists_of_lists_of_structs_column = make_lists_column(
    num_lists_of_lists, std::move(lists_of_lists_column_offsets), std::move(lists_column), 0, {});

  // Check if child column is unchanged.

  auto expected_numeric_column =
    cudf::test::fixed_width_column_wrapper<T, int32_t>{1, 2, 3, 4, 5, 6, 7, 8};
  auto expected_bool_column = cudf::test::fixed_width_column_wrapper<bool>{1, 1, 1, 1, 0, 0, 0, 0};
  auto expected_struct_column =
    cudf::test::structs_column_wrapper{{expected_numeric_column, expected_bool_column}}.release();

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *expected_struct_column,
    cudf::lists_column_view{cudf::lists_column_view{*lists_of_lists_of_structs_column}.child()}
      .child());
}

TYPED_TEST(ListColumnWrapperTestTyped, ListsOfListsOfStructsWithValidity)
{
  using T = TypeParam;

  auto num_struct_rows = 8;
  auto numeric_column  = cudf::test::fixed_width_column_wrapper<T, int32_t>{
    {1, 2, 3, 4, 5, 6, 7, 8}, {1, 1, 1, 1, 0, 0, 0, 0}};
  auto bool_column   = cudf::test::fixed_width_column_wrapper<bool>{1, 1, 1, 1, 0, 0, 0, 0};
  auto struct_column = cudf::test::structs_column_wrapper{{numeric_column, bool_column}}.release();
  EXPECT_EQ(struct_column->size(), num_struct_rows);
  EXPECT_TRUE(!struct_column->nullable());

  auto lists_column_offsets = cudf::test::fixed_width_column_wrapper<int32_t>{0, 2, 4, 8}.release();
  auto num_lists            = lists_column_offsets->size() - 1;
  auto list_null_mask       = {1, 1, 0};
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(list_null_mask.begin(), list_null_mask.end());
  auto lists_column = [&] {
    auto tmp = cudf::make_lists_column(num_lists,
                                       std::move(lists_column_offsets),
                                       std::move(struct_column),
                                       null_count,
                                       std::move(null_mask));
    return cudf::purge_nonempty_nulls(tmp->view());
  }();

  auto lists_of_lists_column_offsets =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 2, 3}.release();
  auto num_lists_of_lists      = lists_of_lists_column_offsets->size() - 1;
  auto list_of_lists_null_mask = {1, 0};

  std::tie(null_mask, null_count) = cudf::test::detail::make_null_mask(
    list_of_lists_null_mask.begin(), list_of_lists_null_mask.end());
  auto lists_of_lists_of_structs_column = [&] {
    auto tmp = cudf::make_lists_column(num_lists_of_lists,
                                       std::move(lists_of_lists_column_offsets),
                                       std::move(lists_column),
                                       null_count,
                                       std::move(null_mask));
    return cudf::purge_nonempty_nulls(tmp->view());
  }();

  // Check if child column is unchanged.

  auto expected_numeric_column =
    cudf::test::fixed_width_column_wrapper<T, int32_t>{{1, 2, 3, 4}, {1, 1, 1, 1}};
  auto expected_bool_column = cudf::test::fixed_width_column_wrapper<bool>{1, 1, 1, 1};
  auto expected_struct_column =
    cudf::test::structs_column_wrapper{{expected_numeric_column, expected_bool_column}}.release();

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *expected_struct_column,
    cudf::lists_column_view{cudf::lists_column_view{*lists_of_lists_of_structs_column}.child()}
      .child());
}

TYPED_TEST(ListColumnWrapperTestTyped, LargeListsOfStructsWithValidity)
{
  using T = TypeParam;

  auto num_struct_rows = 10000;

  // Creating Struct<Numeric, Bool>.
  auto numeric_column = cudf::test::fixed_width_column_wrapper<T, int32_t>{
    cuda::counting_iterator<int32_t>{0},
    cuda::counting_iterator{num_struct_rows},
    cudf::test::iterators::nulls_at_multiples_of(2)};

  auto bool_iterator =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 3 == 0; });
  auto bool_column =
    cudf::test::fixed_width_column_wrapper<bool>(bool_iterator, bool_iterator + num_struct_rows);

  auto struct_validity_iterator = cudf::test::iterators::valids_at_multiples_of(5);
  auto struct_column =
    cudf::test::structs_column_wrapper{
      {numeric_column, bool_column},
      std::vector<bool>(struct_validity_iterator, struct_validity_iterator + num_struct_rows)}
      .release();

  EXPECT_EQ(struct_column->size(), num_struct_rows);

  // Now, use struct_column to create a list column.
  // Each list has 50 elements.
  auto num_list_rows = num_struct_rows / 50;
  auto list_offset_iterator =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 50; });
  auto list_offset_column = cudf::test::fixed_width_column_wrapper<int32_t>(
                              list_offset_iterator, list_offset_iterator + num_list_rows + 1)
                              .release();
  auto lists_column = make_lists_column(
    num_list_rows, std::move(list_offset_column), std::move(struct_column), 0, {});

  // List construction succeeded.
  // Verify that the child is unchanged.

  auto expected_numeric_column = cudf::test::fixed_width_column_wrapper<T, int32_t>{
    cuda::counting_iterator<int32_t>{0},
    cuda::counting_iterator{num_struct_rows},
    cudf::test::iterators::nulls_at_multiples_of(2)};

  auto expected_bool_column =
    cudf::test::fixed_width_column_wrapper<bool>(bool_iterator, bool_iterator + num_struct_rows);

  auto expected_struct_column =
    cudf::test::structs_column_wrapper{
      {expected_numeric_column, expected_bool_column},
      std::vector<bool>(struct_validity_iterator, struct_validity_iterator + num_struct_rows)}
      .release();

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_struct_column,
                                 cudf::lists_column_view(*lists_column).child());
}

// Harness-scoped construction via lists_column_initializer.
// Multi-row nested Init still routes through cudf::concatenate temporaries on the current
// resource; keep fail_on_current scopes on leaf / single-child paths until concatenate is ported.
struct ListsColumnInitializerHarnessTest : public cudf::test::BaseFixtureWithHarness {};

TEST_F(ListsColumnInitializerHarnessTest, LeafInitUsesExplicitResources)
{
  using Init = cudf::test::lists_column_initializer<int32_t>;
  using LCW  = cudf::test::lists_column_wrapper<int32_t, int32_t>;

  auto const st = this->stream();
  auto const mr = this->resources();

  std::unique_ptr<cudf::column> built;
  {
    auto fail_on_current = this->fail_on_current_device_resource_use();
    LCW list{Init{{1, 2, 3, 4}}, st, mr};
    this->_harness.synchronize(st);
    built = list.release();
  }

  LCW expected{Init{{1, 2, 3, 4}}, st, mr};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *built, expected, cudf::test::debug_output_level::FIRST_ERROR, st, mr);
}

TEST_F(ListsColumnInitializerHarnessTest, LeafNullableInitUsesExplicitResources)
{
  using Init = cudf::test::lists_column_initializer<int32_t>;
  using LCW  = cudf::test::lists_column_wrapper<int32_t, int32_t>;
  using cudf::test::iterators::null_at;

  auto const st = this->stream();
  auto const mr = this->resources();

  std::unique_ptr<cudf::column> built;
  {
    auto fail_on_current = this->fail_on_current_device_resource_use();
    LCW list{Init{{1, 2, 3, 4}, null_at(1)}, st, mr};
    this->_harness.synchronize(st);
    built = list.release();
  }

  LCW expected{Init{{1, 2, 3, 4}, null_at(1)}, st, mr};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *built, expected, cudf::test::debug_output_level::FIRST_ERROR, st, mr);
}

TEST_F(ListsColumnInitializerHarnessTest, SingleChildNestedInitUsesExplicitResources)
{
  using Init = cudf::test::lists_column_initializer<int32_t>;
  using LCW  = cudf::test::lists_column_wrapper<int32_t, int32_t>;

  auto const st = this->stream();
  auto const mr = this->resources();

  // One outer row whose only child is a leaf list: both levels avoid concatenate.
  std::unique_ptr<cudf::column> built;
  {
    auto fail_on_current = this->fail_on_current_device_resource_use();
    LCW list{Init{{{{1, 2, 3}}}}, st, mr};
    this->_harness.synchronize(st);
    built = list.release();
  }

  LCW expected{Init{{{{1, 2, 3}}}}, st, mr};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *built, expected, cudf::test::debug_output_level::FIRST_ERROR, st, mr);
}
