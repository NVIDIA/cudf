/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <vector>

namespace testdata {
// ----- most numerics

template <typename T>
cudf::test::fixed_width_column_wrapper<T> ascending()
  requires(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>)
{
  return std::is_signed_v<T>
           ? cudf::test::fixed_width_column_wrapper<T>({std::numeric_limits<T>::lowest(),
                                                        T(-100),
                                                        T(-10),
                                                        T(-1),
                                                        T(0),
                                                        T(1),
                                                        T(10),
                                                        T(100),
                                                        std::numeric_limits<T>::max()})
           : cudf::test::fixed_width_column_wrapper<T>({std::numeric_limits<T>::lowest(),
                                                        T(0),
                                                        T(1),
                                                        T(10),
                                                        T(100),
                                                        std::numeric_limits<T>::max()});
}

template <typename T>
cudf::test::fixed_width_column_wrapper<T> descending()
  requires(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>)
{
  return std::is_signed_v<T>
           ? cudf::test::fixed_width_column_wrapper<T>({std::numeric_limits<T>::max(),
                                                        T(100),
                                                        T(10),
                                                        T(1),
                                                        T(0),
                                                        T(-1),
                                                        T(-10),
                                                        T(-100),
                                                        std::numeric_limits<T>::lowest()})
           : cudf::test::fixed_width_column_wrapper<T>({std::numeric_limits<T>::max(),
                                                        T(100),
                                                        T(10),
                                                        T(1),
                                                        T(0),
                                                        std::numeric_limits<T>::lowest()});
}

template <typename T>
auto empty()
{ return cudf::test::fixed_width_column_wrapper<T>(); }

template <typename T>
auto nulls_after()
{ return cudf::test::fixed_width_column_wrapper<T, int32_t>({0, 0}, {1, 0}); }

template <typename T>
auto nulls_before()
{ return cudf::test::fixed_width_column_wrapper<T, int32_t>({0, 0}, {0, 1}); }

// ----- bool

template <typename T>
cudf::test::fixed_width_column_wrapper<bool> ascending()
  requires(std::is_same_v<T, bool>)
{ return cudf::test::fixed_width_column_wrapper<bool>({false, false, true, true}); }

template <typename T>
cudf::test::fixed_width_column_wrapper<bool> descending()
  requires(std::is_same_v<T, bool>)
{ return cudf::test::fixed_width_column_wrapper<bool>({true, true, false, false}); }

// ----- chrono types

template <typename T>
cudf::test::fixed_width_column_wrapper<T> ascending()
  requires(cudf::is_chrono<T>())
{ return cudf::test::fixed_width_column_wrapper<T>({T::min(), T::max()}); }

template <typename T>
cudf::test::fixed_width_column_wrapper<T> descending()
  requires(cudf::is_chrono<T>())
{ return cudf::test::fixed_width_column_wrapper<T>({T::max(), T::min()}); }

// ----- string_view

template <typename T>
cudf::test::strings_column_wrapper ascending()
  requires(std::is_same_v<T, cudf::string_view>)
{ return cudf::test::strings_column_wrapper({"A", "B"}); }

template <typename T>
cudf::test::strings_column_wrapper descending()
  requires(std::is_same_v<T, cudf::string_view>)
{ return cudf::test::strings_column_wrapper({"B", "A"}); }

template <>
auto empty<cudf::string_view>()
{ return cudf::test::strings_column_wrapper(); }

template <>
auto nulls_after<cudf::string_view>()
{ return cudf::test::strings_column_wrapper({"identical", ""}, {true, false}); }

template <>
auto nulls_before<cudf::string_view>()
{ return cudf::test::strings_column_wrapper({"", "identical"}, {false, true}); }

// ----- struct_view {"nestedInt" : {"Int" : 0 }, "float" : 1}

template <typename T>
cudf::test::structs_column_wrapper ascending()
  requires(std::is_same_v<T, cudf::struct_view>)
{
  using T1     = int32_t;
  auto int_col = cudf::test::fixed_width_column_wrapper<int32_t>({std::numeric_limits<T1>::lowest(),
                                                                  T1(-100),
                                                                  T1(-10),
                                                                  T1(-10),
                                                                  T1(0),
                                                                  T1(10),
                                                                  T1(10),
                                                                  T1(100),
                                                                  std::numeric_limits<T1>::max()});
  auto nestedInt_col = cudf::test::structs_column_wrapper{{int_col}};
  auto float_col     = ascending<float>();
  return cudf::test::structs_column_wrapper{{nestedInt_col, float_col}};
}

template <typename T>
cudf::test::structs_column_wrapper descending()
  requires(std::is_same_v<T, cudf::struct_view>)
{
  using T1 = int32_t;
  auto int_col =
    cudf::test::fixed_width_column_wrapper<int32_t>({std::numeric_limits<T1>::max(),
                                                     T1(100),
                                                     T1(10),
                                                     T1(10),
                                                     T1(0),
                                                     T1(-10),
                                                     T1(-10),
                                                     T1(-100),
                                                     std::numeric_limits<T1>::lowest()});
  auto nestedInt_col = cudf::test::structs_column_wrapper{{int_col}};
  auto float_col     = descending<float>();
  return cudf::test::structs_column_wrapper{{nestedInt_col, float_col}};
}

template <>
auto empty<cudf::struct_view>()
{
  auto int_col = cudf::test::fixed_width_column_wrapper<int32_t>();
  auto col1    = cudf::test::structs_column_wrapper{{int_col}};
  auto col2    = cudf::test::fixed_width_column_wrapper<float>();
  return cudf::test::structs_column_wrapper{{col1, col2}};
}

template <>
auto nulls_after<cudf::struct_view>()
{
  auto int_col = cudf::test::fixed_width_column_wrapper<int32_t>({1, 1});
  auto col1    = cudf::test::structs_column_wrapper{{int_col}};
  auto col2    = cudf::test::fixed_width_column_wrapper<float>({1, 1});
  return cudf::test::structs_column_wrapper{{col1, col2}, {true, false}};
}

template <>
auto nulls_before<cudf::struct_view>()
{
  auto int_col = cudf::test::fixed_width_column_wrapper<int32_t>({1, 1});
  auto col1    = cudf::test::structs_column_wrapper{{int_col}};
  auto col2    = cudf::test::fixed_width_column_wrapper<float>({1, 1});
  return cudf::test::structs_column_wrapper{{col1, col2}, {false, true}};
}

using lcw = cudf::test::lists_column_wrapper<int32_t>;
using cudf::test::iterators::null_at;
/*
List<List<List<int>
[
  [[[0]], [[0]], [[0]]],                        0
  [[[0], [0], [0]]],                            1
  [[[0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0]], [[0]]] 2
  [[[0, 0, 0]]],                                3
  [[[0, 0, 0]], [[0]], [[0]]],                  4
]
*/

template <typename T>
lcw ascending()
  requires(std::is_same_v<T, cudf::list_view>)
{
  return lcw{lcw{lcw{lcw{0}}, lcw{lcw{0}}, lcw{lcw{0}}},
             lcw{lcw{lcw{0}, lcw{0}, lcw{0}}},
             lcw{lcw{lcw{0, 0}}, lcw{lcw{0, 0, 0, 0, 0, 0, 0, 0}}, lcw{lcw{0}}},
             lcw{lcw{lcw{0, 0, 0}}},
             lcw{lcw{lcw{0, 0, 0}}, lcw{lcw{0}}, lcw{lcw{0}}}};
}

template <typename T>
lcw descending()
  requires(std::is_same_v<T, cudf::list_view>)
{
  return lcw{lcw{lcw{lcw{0, 0, 0}}, lcw{lcw{0}}, lcw{lcw{0}}},
             lcw{lcw{lcw{0, 0, 0}}},
             lcw{lcw{lcw{0, 0}}, lcw{lcw{0, 0, 0, 0, 0, 0, 0, 0}}, lcw{lcw{0}}},

             lcw{lcw{lcw{0}, lcw{0}, lcw{0}}},
             lcw{lcw{lcw{0}}, lcw{lcw{0}}, lcw{lcw{0}}}};
}

template <>
auto empty<cudf::list_view>()
{ return lcw{}; }

template <>
auto nulls_after<cudf::list_view>()
{ return lcw{{{1}, {2, 2}, {0}}, null_at(2)}; }

template <>
auto nulls_before<cudf::list_view>()
{ return lcw{{{0}, {1}, {2, 2}}, null_at(0)}; }

}  // namespace testdata

// =============================================================================
// ---- tests ------------------------------------------------------------------

template <typename T>
struct IsSortedTest : public cudf::test::BaseFixture {};

using SupportedTypes = cudf::test::
  Concat<cudf::test::ComparableTypes, cudf::test::Types<cudf::struct_view>, cudf::test::ListTypes>;
TYPED_TEST_SUITE(IsSortedTest, SupportedTypes);

TYPED_TEST(IsSortedTest, NoColumns)
{
  cudf::table_view in{std::vector<cudf::table_view>{}};
  std::vector<cudf::order> order{};
  std::vector<cudf::null_order> null_precedence{};

  auto actual = cudf::is_sorted(in, order, null_precedence);

  EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedTest, NoRows)
{
  using T = TypeParam;

  if (std::is_same_v<T, cudf::string_view>) {
    // cudf::test::strings_column_wrapper does not yet support empty columns.
    return;
  } else {
    auto col1 = testdata::empty<T>();
    auto col2 = testdata::empty<T>();

    cudf::table_view in{{col1, col2}};
    std::vector<cudf::order> order{cudf::order::ASCENDING, cudf::order::DESCENDING};
    std::vector<cudf::null_order> null_precedence{};

    auto actual = cudf::is_sorted(in, order, null_precedence);

    EXPECT_EQ(true, actual);
  }
}

TYPED_TEST(IsSortedTest, Ascending)
{
  using T = TypeParam;

  auto col1 = testdata::ascending<T>();
  cudf::table_view in{{col1}};
  std::vector<cudf::order> order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{};

  auto actual = cudf::is_sorted(in, order, null_precedence);

  EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedTest, AscendingFalse)
{
  using T = TypeParam;

  auto col1 = testdata::descending<T>();
  cudf::table_view in{{col1}};
  std::vector<cudf::order> order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{};

  auto actual = cudf::is_sorted(in, order, {});

  EXPECT_EQ(false, actual);
}

TYPED_TEST(IsSortedTest, Descending)
{
  using T = TypeParam;

  auto col1 = testdata::descending<T>();

  cudf::table_view in{{col1}};
  std::vector<cudf::order> order{cudf::order::DESCENDING};
  std::vector<cudf::null_order> null_precedence{};

  auto actual = cudf::is_sorted(in, order, null_precedence);

  EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedTest, DescendingFalse)
{
  using T = TypeParam;

  auto col1 = testdata::ascending<T>();

  cudf::table_view in{{col1}};
  std::vector<cudf::order> order{cudf::order::DESCENDING};
  std::vector<cudf::null_order> null_precedence{};

  auto actual = cudf::is_sorted(in, order, null_precedence);

  EXPECT_EQ(false, actual);
}

TYPED_TEST(IsSortedTest, NullsAfter)
{
  using T = TypeParam;

  auto col1 = testdata::nulls_after<T>();

  cudf::table_view in{{col1}};
  std::vector<cudf::order> order{};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::AFTER};

  auto actual = cudf::is_sorted(in, order, null_precedence);

  EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedTest, NullsAfterFalse)
{
  using T = TypeParam;

  auto col1 = testdata::nulls_before<T>();

  cudf::table_view in{{col1}};
  std::vector<cudf::order> order{};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::AFTER};

  auto actual = cudf::is_sorted(in, order, null_precedence);

  EXPECT_EQ(false, actual);
}

TYPED_TEST(IsSortedTest, NullsBefore)
{
  using T = TypeParam;

  auto col1 = testdata::nulls_before<T>();

  cudf::table_view in{{col1}};
  std::vector<cudf::order> order{};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::BEFORE};

  auto actual = cudf::is_sorted(in, order, null_precedence);

  EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedTest, NullsBeforeFalse)
{
  using T = TypeParam;

  auto col1 = testdata::nulls_after<T>();

  cudf::table_view in{{col1}};
  std::vector<cudf::order> order{};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::BEFORE};

  auto actual = cudf::is_sorted(in, order, null_precedence);

  EXPECT_EQ(false, actual);
}

TYPED_TEST(IsSortedTest, OrderArgsTooFew)
{
  using T = TypeParam;

  auto col1 = testdata::ascending<T>();
  auto col2 = testdata::ascending<T>();

  cudf::table_view in{{col1, col2}};
  std::vector<cudf::order> order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{};

  EXPECT_THROW(cudf::is_sorted(in, order, null_precedence), cudf::logic_error);
}

TYPED_TEST(IsSortedTest, OrderArgsTooMany)
{
  using T = TypeParam;

  auto col1 = testdata::ascending<T>();

  cudf::table_view in{{col1}};
  std::vector<cudf::order> order{cudf::order::ASCENDING, cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{};

  EXPECT_THROW(cudf::is_sorted(in, order, null_precedence), cudf::logic_error);
}

TYPED_TEST(IsSortedTest, NullOrderArgsTooFew)
{
  using T = TypeParam;

  auto col1 = testdata::nulls_before<T>();
  auto col2 = testdata::nulls_before<T>();

  cudf::table_view in{{col1, col2}};
  std::vector<cudf::order> order{};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::BEFORE};

  EXPECT_THROW(cudf::is_sorted(in, order, null_precedence), cudf::logic_error);
}

TYPED_TEST(IsSortedTest, NullOrderArgsTooMany)
{
  using T = TypeParam;

  auto col1 = testdata::nulls_before<T>();

  cudf::table_view in{{col1}};
  std::vector<cudf::order> order{};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::BEFORE, cudf::null_order::BEFORE};

  EXPECT_THROW(cudf::is_sorted(in, order, null_precedence), cudf::logic_error);
}

template <typename T>
struct IsSortedFixedWidthOnly : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(IsSortedFixedWidthOnly, cudf::test::FixedWidthTypes);

CUDF_TEST_PROGRAM_MAIN()
