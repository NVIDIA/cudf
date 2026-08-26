/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};
constexpr int32_t null{0};  // Mark for null child elements
constexpr int32_t XXX{0};   // Mark for null struct elements

using structs_col = cudf::test::structs_column_wrapper;

template <typename T>
struct TypedStructScalarScatterTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(TypedStructScalarScatterTest, cudf::test::FixedWidthTypes);

cudf::column scatter_single_scalar(cudf::scalar const& slr,
                                   cudf::column_view scatter_map,
                                   cudf::column_view target)
{
  auto result = cudf::scatter({slr}, scatter_map, cudf::table_view{{target}});
  return result->get_column(0);
}

TYPED_TEST(TypedStructScalarScatterTest, Basic)
{
  using fixed_width_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  // Source scalar
  fixed_width_wrapper slr_f0{777};
  cudf::test::strings_column_wrapper slr_f1{"hello"};
  auto slr = cudf::make_struct_scalar(cudf::table_view{{slr_f0, slr_f1}});

  // Scatter map
  cudf::test::fixed_width_column_wrapper<cudf::size_type> scatter_map{2};

  // Target column
  fixed_width_wrapper field0{11, 11, 22, 22, 33};
  cudf::test::strings_column_wrapper field1{"aa", "aa", "bb", "bb", "cc"};
  structs_col target{field0, field1};

  // Expect column
  fixed_width_wrapper ef0{11, 11, 777, 22, 33};
  cudf::test::strings_column_wrapper ef1{"aa", "aa", "hello", "bb", "cc"};
  structs_col expected{ef0, ef1};

  auto got = scatter_single_scalar(*slr, scatter_map, target);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, got, verbosity);
}

TYPED_TEST(TypedStructScalarScatterTest, FillNulls)
{
  using fixed_width_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  // Source scalar
  fixed_width_wrapper slr_f0{777};
  auto slr = cudf::make_struct_scalar(cudf::table_view{{slr_f0}});

  // Scatter map
  cudf::test::fixed_width_column_wrapper<cudf::size_type> scatter_map{3, 4};

  // Target column
  fixed_width_wrapper field0({11, 11, 22, null, XXX}, cudf::test::iterators::null_at(3));
  structs_col target({field0}, cudf::test::iterators::null_at(4));

  // Expect column
  fixed_width_wrapper ef0{11, 11, 22, 777, 777};
  structs_col expected{ef0};

  auto got = scatter_single_scalar(*slr, scatter_map, target);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, got, verbosity);
}

TYPED_TEST(TypedStructScalarScatterTest, ScatterNullElements)
{
  using fixed_width_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  // Source scalar
  fixed_width_wrapper slr_f0{777};
  std::vector<cudf::column_view> source_fields{slr_f0};
  auto slr = std::make_unique<cudf::struct_scalar>(source_fields, false);

  // Scatter map
  cudf::test::fixed_width_column_wrapper<cudf::size_type> scatter_map{0, 3, 4};

  // Target column
  fixed_width_wrapper field0({11, 11, 22, null, XXX}, cudf::test::iterators::null_at(3));
  structs_col target({field0}, cudf::test::iterators::null_at(4));

  // Expect column
  fixed_width_wrapper ef0{XXX, 11, 22, XXX, XXX};
  structs_col expected({ef0}, {false, true, true, false, false});

  auto got = scatter_single_scalar(*slr, scatter_map, target);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got, verbosity);
}

TYPED_TEST(TypedStructScalarScatterTest, ScatterNullFields)
{
  using fixed_width_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  // Source scalar
  fixed_width_wrapper slr_f0({null}, {false});
  auto slr = cudf::make_struct_scalar(cudf::table_view{{slr_f0}});

  // Scatter map
  cudf::test::fixed_width_column_wrapper<cudf::size_type> scatter_map{0, 2};

  // Target column
  fixed_width_wrapper field0({11, 11, 22, null, XXX}, cudf::test::iterators::null_at(3));
  structs_col target({field0}, cudf::test::iterators::null_at(4));

  // Expect column
  fixed_width_wrapper ef0({null, 11, null, null, XXX}, {false, true, false, false, true});
  structs_col expected({ef0}, cudf::test::iterators::null_at(4));

  auto got = scatter_single_scalar(*slr, scatter_map, target);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got, verbosity);
}

TYPED_TEST(TypedStructScalarScatterTest, NegativeIndices)
{
  using fixed_width_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  // Source scalar
  fixed_width_wrapper slr_f0{777};
  auto slr = cudf::make_struct_scalar(cudf::table_view{{slr_f0}});

  // Scatter map
  cudf::test::fixed_width_column_wrapper<cudf::size_type> scatter_map{-1, -5};

  // Target column
  fixed_width_wrapper field0({11, 11, 22, null, XXX}, cudf::test::iterators::null_at(3));
  structs_col target({field0}, cudf::test::iterators::null_at(4));

  // Expect column
  fixed_width_wrapper ef0({777, 11, 22, null, 777}, cudf::test::iterators::null_at(3));
  structs_col expected{ef0};

  auto got = scatter_single_scalar(*slr, scatter_map, target);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, got, verbosity);
}

TYPED_TEST(TypedStructScalarScatterTest, EmptyInputTest)
{
  using fixed_width_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  // Source scalar
  fixed_width_wrapper slr_f0{777};
  auto slr = cudf::make_struct_scalar(cudf::table_view{{slr_f0}});

  // Scatter map
  cudf::test::fixed_width_column_wrapper<cudf::size_type> scatter_map{};

  // Target column
  fixed_width_wrapper field0{};
  structs_col target{field0};

  // Expect column
  fixed_width_wrapper ef0{};
  structs_col expected{ef0};

  auto got = scatter_single_scalar(*slr, scatter_map, target);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, got, verbosity);
}

TYPED_TEST(TypedStructScalarScatterTest, EmptyScatterMapTest)
{
  using fixed_width_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  // Source scalar
  fixed_width_wrapper slr_f0{777};
  auto slr = cudf::make_struct_scalar(cudf::table_view{{slr_f0}});

  // Scatter map
  cudf::test::fixed_width_column_wrapper<cudf::size_type> scatter_map{};

  // Target column
  fixed_width_wrapper field0({11, 11, 22, null, XXX}, cudf::test::iterators::null_at(3));
  structs_col target({field0}, cudf::test::iterators::null_at(4));

  auto got = scatter_single_scalar(*slr, scatter_map, target);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(target, got, verbosity);
}

TYPED_TEST(TypedStructScalarScatterTest, FixedWidthStringTypes)
{
  using fixed_width_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  // Source scalar
  fixed_width_wrapper slr_f0{777};
  cudf::test::strings_column_wrapper slr_f1{"hello"};
  auto slr = cudf::make_struct_scalar(cudf::table_view{{slr_f0, slr_f1}});

  // Scatter map
  cudf::test::fixed_width_column_wrapper<cudf::size_type> scatter_map{0, 2, 4};

  // Target column
  fixed_width_wrapper field0({11, 11, 22, null, XXX}, cudf::test::iterators::null_at(3));
  cudf::test::strings_column_wrapper field1({"aa", "null", "ccc", "null", "XXX"},
                                            {true, false, true, false, true});
  structs_col target({field0, field1}, cudf::test::iterators::null_at(4));

  // Expect column
  fixed_width_wrapper ef0({777, 11, 777, null, 777}, cudf::test::iterators::null_at(3));
  cudf::test::strings_column_wrapper ef1({"hello", "null", "hello", "null", "hello"},
                                         {true, false, true, false, true});
  structs_col expected{ef0, ef1};

  auto got = scatter_single_scalar(*slr, scatter_map, target);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, got, verbosity);
}

TYPED_TEST(TypedStructScalarScatterTest, StructOfLists)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // Source scalar
  LCW slr_f0{777};
  auto slr = cudf::make_struct_scalar(cudf::table_view{{slr_f0}});

  // Scatter map
  cudf::test::fixed_width_column_wrapper<cudf::size_type> scatter_map{0, 1, 4};

  // Target column
  LCW field0({LCW{XXX}, LCW{22}, LCW{33, 44}, LCW{null}, LCW{55}},
             cudf::test::iterators::null_at(3));
  structs_col target({field0}, cudf::test::iterators::null_at(0));

  // Expect column
  LCW ef0({LCW{777}, LCW{777}, LCW{33, 44}, LCW{null}, LCW{777}},
          cudf::test::iterators::null_at(3));
  structs_col expected{ef0};

  auto got = scatter_single_scalar(*slr, scatter_map, target);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, got, verbosity);
}
