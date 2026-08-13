/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>

using mask_vector = std::vector<cudf::valid_type>;
using size_column = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

class ScatterListScalarTests : public cudf::test::BaseFixture {};

std::unique_ptr<cudf::column> single_scalar_scatter(cudf::column_view const& target,
                                                    cudf::scalar const& slr,
                                                    cudf::column_view const& scatter_map)
{
  std::vector<std::reference_wrapper<cudf::scalar const>> slrs{slr};
  cudf::table_view targets{{target}};
  auto result = cudf::scatter(slrs, scatter_map, targets);
  return std::move(result->release()[0]);
}

template <typename T>
class ScatterListOfFixedWidthScalarTest : public ScatterListScalarTests {};

TYPED_TEST_SUITE(ScatterListOfFixedWidthScalarTest, cudf::test::FixedWidthTypesWithoutFixedPoint);

// Test grid
// Dim1 : {Fixed width, strings, lists, structs}
// Dim2 : {Null scalar, Non-null empty scalar, Non-null non-empty scalar}
// Dim3 : {Nullable target, non-nullable target row}

TYPED_TEST(ScatterListOfFixedWidthScalarTest, Basic)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;
  using FCW = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto slr = std::make_unique<cudf::list_scalar>(FCW({2, 2, 2}, {1, 0, 1}), true);
  LCW col{LCW{1, 1, 1}, LCW{8, 8}, LCW{10, 10, 10, 10}, LCW{5}};
  size_column scatter_map{3, 1, 0};

  LCW expected{LCW({2, 2, 2}, mask_vector{1, 0, 1}.begin()),
               LCW({2, 2, 2}, mask_vector{1, 0, 1}.begin()),
               LCW{10, 10, 10, 10},
               LCW({2, 2, 2}, mask_vector{1, 0, 1}.begin())};
  auto result = single_scalar_scatter(col, *slr, scatter_map);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
}

TYPED_TEST(ScatterListOfFixedWidthScalarTest, EmptyValidScalar)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;
  using FCW = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto slr = std::make_unique<cudf::list_scalar>(FCW{}, true);
  LCW col{LCW{1, 1, 1},
          LCW{8, 8},
          LCW({10, 10, 10, 10}, mask_vector{1, 0, 1, 0}.begin()),
          LCW{5},
          LCW{42, 42}};
  size_column scatter_map{1, 0};

  LCW expected{
    LCW{}, LCW{}, LCW({10, 10, 10, 10}, mask_vector{1, 0, 1, 0}.begin()), LCW{5}, LCW{42, 42}};
  auto result = single_scalar_scatter(col, *slr, scatter_map);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
}

TYPED_TEST(ScatterListOfFixedWidthScalarTest, NullScalar)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;
  using FCW = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto slr = std::make_unique<cudf::list_scalar>(FCW{}, false);
  LCW col{LCW({1, 1, 1}, mask_vector{0, 0, 1}.begin()), LCW{8, 8}, LCW{10, 10, 10, 10}, LCW{5}};
  size_column scatter_map{3, 1};

  LCW expected({LCW({1, 1, 1}, mask_vector{0, 0, 1}.begin()), LCW{}, LCW{10, 10, 10, 10}, LCW{}},
               mask_vector{1, 0, 1, 0}.begin());
  auto result = single_scalar_scatter(col, *slr, scatter_map);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
}

TYPED_TEST(ScatterListOfFixedWidthScalarTest, NullableTargetRow)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;
  using FCW = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto slr = std::make_unique<cudf::list_scalar>(FCW{9, 9}, true);
  LCW col({LCW{4, 4}, LCW{}, LCW{8, 8, 8}, LCW{}, LCW{9, 9, 9}},
          mask_vector{1, 0, 1, 0, 1}.begin());
  size_column scatter_map{0, 1};

  LCW expected({LCW{9, 9}, LCW{9, 9}, LCW{8, 8, 8}, LCW{}, LCW{9, 9, 9}},
               mask_vector{1, 1, 1, 0, 1}.begin());
  auto result = single_scalar_scatter(col, *slr, scatter_map);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
}

class ScatterListOfStringScalarTest : public ScatterListScalarTests {};

TEST_F(ScatterListOfStringScalarTest, Basic)
{
  using LCW      = cudf::test::lists_column_wrapper<cudf::string_view, int32_t>;
  using StringCW = cudf::test::strings_column_wrapper;

  auto slr = std::make_unique<cudf::list_scalar>(
    StringCW({"Hello!", "", "你好！", "صباح الخير!", "", "こんにちは！"},
             {true, false, true, true, false, true}),
    true);
  LCW col{LCW({"xx", "yy"}, mask_vector{0, 1}.begin()), LCW{""}, LCW{"a", "bab", "bacab"}};

  size_column scatter_map{2, 1};

  LCW expected{LCW({"xx", "yy"}, mask_vector{0, 1}.begin()),
               LCW({"Hello!", "", "你好！", "صباح الخير!", "", "こんにちは！"},
                   mask_vector{1, 0, 1, 1, 0, 1}.begin()),
               LCW({"Hello!", "", "你好！", "صباح الخير!", "", "こんにちは！"},
                   mask_vector{1, 0, 1, 1, 0, 1}.begin())};

  auto result = single_scalar_scatter(col, *slr, scatter_map);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
}

TEST_F(ScatterListOfStringScalarTest, EmptyValidScalar)
{
  using LCW      = cudf::test::lists_column_wrapper<cudf::string_view, int32_t>;
  using StringCW = cudf::test::strings_column_wrapper;

  auto slr = std::make_unique<cudf::list_scalar>(StringCW{}, true);

  LCW col{LCW({"xx", "yy"}, mask_vector{0, 1}.begin()),
          LCW{""},
          LCW{"a", "bab", "bacab"},
          LCW{"888", "777"}};

  size_column scatter_map{0, 3};

  LCW expected{LCW{}, LCW{""}, LCW{"a", "bab", "bacab"}, LCW{}};

  auto result = single_scalar_scatter(col, *slr, scatter_map);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
}

TEST_F(ScatterListOfStringScalarTest, NullScalar)
{
  using LCW      = cudf::test::lists_column_wrapper<cudf::string_view, int32_t>;
  using StringCW = cudf::test::strings_column_wrapper;

  auto slr = std::make_unique<cudf::list_scalar>(StringCW{}, false);
  LCW col{LCW{"xx", "yy"},
          LCW({""}, mask_vector{0}.begin()),
          LCW{"a", "bab", "bacab"},
          LCW{"888", "777"}};

  size_column scatter_map{1, 2};

  LCW expected({LCW{"xx", "yy"}, LCW{}, LCW{}, LCW{"888", "777"}}, mask_vector{1, 0, 0, 1}.begin());

  auto result = single_scalar_scatter(col, *slr, scatter_map);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
}

TEST_F(ScatterListOfStringScalarTest, NullableTargetRow)
{
  using LCW      = cudf::test::lists_column_wrapper<cudf::string_view, int32_t>;
  using StringCW = cudf::test::strings_column_wrapper;

  auto slr = std::make_unique<cudf::list_scalar>(
    StringCW({"Hello!", "", "こんにちは！"}, {true, false, true}), true);
  LCW col({LCW{"xx", "yy"}, LCW({""}, mask_vector{0}.begin()), LCW{}, LCW{"888", "777"}},
          mask_vector{1, 1, 0, 1}.begin());

  size_column scatter_map{3, 2};

  LCW expected({LCW{"xx", "yy"},
                LCW({""}, mask_vector{0}.begin()),
                LCW({"Hello!", "", "こんにちは！"}, mask_vector{1, 0, 1}.begin()),
                LCW({"Hello!", "", "こんにちは！"}, mask_vector{1, 0, 1}.begin())},
               mask_vector{1, 1, 1, 1}.begin());

  auto result = single_scalar_scatter(col, *slr, scatter_map);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
}

template <typename T>
class ScatterListOfListScalarTest : public ScatterListScalarTests {};

TYPED_TEST_SUITE(ScatterListOfListScalarTest, cudf::test::FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(ScatterListOfListScalarTest, Basic)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto slr = std::make_unique<cudf::list_scalar>(
    LCW({LCW{1, 2, 3}, LCW{4}, LCW{}, LCW{5, 6}}, mask_vector{1, 1, 0, 1}.begin()), true);
  LCW col({LCW({LCW{88, 88}, LCW{}, LCW{9, 9, 9}}, mask_vector{1, 0, 1}.begin()),
           LCW{LCW{66}, LCW{}, LCW({77, 77, 77, 77}, mask_vector{1, 0, 0, 1}.begin())},
           LCW{LCW{55, 55}, LCW{}, LCW{10, 10, 10}},
           LCW{LCW{44, 44}}});

  size_column scatter_map{1, 2, 3};

  LCW expected({LCW({LCW{88, 88}, LCW{}, LCW{9, 9, 9}}, mask_vector{1, 0, 1}.begin()),
                LCW({LCW{1, 2, 3}, LCW{4}, LCW{}, LCW{5, 6}}, mask_vector{1, 1, 0, 1}.begin()),
                LCW({LCW{1, 2, 3}, LCW{4}, LCW{}, LCW{5, 6}}, mask_vector{1, 1, 0, 1}.begin()),
                LCW({LCW{1, 2, 3}, LCW{4}, LCW{}, LCW{5, 6}}, mask_vector{1, 1, 0, 1}.begin())});

  auto result = single_scalar_scatter(col, *slr, scatter_map);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
}

TYPED_TEST(ScatterListOfListScalarTest, EmptyValidScalar)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto slr = std::make_unique<cudf::list_scalar>(LCW{}, true);
  LCW col({LCW({LCW{88, 88}, LCW{}, LCW{9, 9, 9}}, mask_vector{1, 0, 1}.begin()),
           LCW{LCW{66}, LCW{}, LCW({77, 77, 77, 77}, mask_vector{1, 0, 0, 1}.begin())},
           LCW{LCW{55, 55}, LCW{}, LCW{10, 10, 10}},
           LCW{LCW{44, 44}}});

  size_column scatter_map{3, 0};

  LCW expected({LCW{},
                LCW{LCW{66}, LCW{}, LCW({77, 77, 77, 77}, mask_vector{1, 0, 0, 1}.begin())},
                LCW{LCW{55, 55}, LCW{}, LCW{10, 10, 10}},
                LCW{}});

  auto result = single_scalar_scatter(col, *slr, scatter_map);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
}

TYPED_TEST(ScatterListOfListScalarTest, NullScalar)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto slr = std::make_unique<cudf::list_scalar>(LCW{}, false);
  LCW col({LCW({LCW{88, 88}, LCW{}, LCW{9, 9, 9}}, mask_vector{1, 0, 1}.begin()),
           LCW{LCW{66}, LCW{}, LCW({77, 77, 77, 77}, mask_vector{1, 0, 0, 1}.begin())},
           LCW{LCW{44, 44}}});

  size_column scatter_map{1, 0};

  LCW expected({LCW{}, LCW{}, LCW{LCW{44, 44}}}, mask_vector{0, 0, 1}.begin());

  auto result = single_scalar_scatter(col, *slr, scatter_map);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
}

TYPED_TEST(ScatterListOfListScalarTest, NullableTargetRows)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto slr = std::make_unique<cudf::list_scalar>(
    LCW({LCW{1, 1, 1}, LCW{3, 3}, LCW{}, LCW{4}}, mask_vector{1, 1, 0, 1}.begin()), true);

  LCW col({LCW({LCW{88, 88}, LCW{}, LCW{9, 9, 9}}, mask_vector{1, 0, 1}.begin()),
           LCW{LCW{66}, LCW{}, LCW({77, 77, 77, 77}, mask_vector{1, 0, 0, 1}.begin())},
           LCW{LCW{44, 44}}},
          mask_vector{1, 0, 1}.begin());

  size_column scatter_map{1};

  LCW expected({LCW({LCW{88, 88}, LCW{}, LCW{9, 9, 9}}, mask_vector{1, 0, 1}.begin()),
                LCW({LCW{1, 1, 1}, LCW{3, 3}, LCW{}, LCW{4}}, mask_vector{1, 1, 0, 1}.begin()),
                LCW{LCW{44, 44}}},
               mask_vector{1, 1, 1}.begin());

  auto result = single_scalar_scatter(col, *slr, scatter_map);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
}

template <typename T>
class ScatterListOfStructScalarTest : public ScatterListScalarTests {
 protected:
  cudf::test::structs_column_wrapper make_test_structs(
    cudf::test::fixed_width_column_wrapper<T> field0,
    cudf::test::strings_column_wrapper field1,
    cudf::test::lists_column_wrapper<T, int32_t> field2,
    std::vector<cudf::valid_type> mask)
  {
    return cudf::test::structs_column_wrapper({field0, field1, field2}, mask.begin());
  }
};

TYPED_TEST_SUITE(ScatterListOfStructScalarTest, cudf::test::FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(ScatterListOfStructScalarTest, Basic)
{
  using LCW      = cudf::test::lists_column_wrapper<TypeParam, int32_t>;
  using offset_t = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  auto data =
    this->make_test_structs({{42, 42, 42}, {1, 0, 1}},
                            {{"hello", "你好！", "bonjour!"}, {false, true, true}},
                            LCW({LCW{88}, LCW{}, LCW{99, 99}}, mask_vector{1, 0, 1}.begin()),
                            {1, 1, 0});
  auto slr = std::make_unique<cudf::list_scalar>(data, true);

  auto child = this->make_test_structs(
    {{1, 1, 2, 3, 3, 3}, {0, 1, 1, 1, 0, 0}},
    {{"x", "x", "yy", "", "zzz", "zzz"}, {true, true, true, false, true, true}},
    LCW({LCW{10, 10}, LCW{}, LCW{10}, LCW{20, 20}, LCW{}, LCW{30, 30}},
        mask_vector{1, 0, 1, 1, 0, 1}.begin()),
    {1, 1, 0, 0, 1, 1});
  offset_t offsets{0, 2, 2, 3, 6};
  auto col =
    cudf::make_lists_column(4, offsets.release(), child.release(), 0, rmm::device_buffer{});

  size_column scatter_map{1, 3};

  auto ex_child = this->make_test_structs(
    {{1, 1, 42, 42, 42, 2, 42, 42, 42}, {0, 1, 1, 0, 1, 1, 1, 0, 1}},
    {{"x", "x", "hello", "你好！", "bonjour!", "yy", "hello", "你好！", "bonjour!"},
     {true, true, false, true, true, true, false, true, true}},
    LCW({LCW{10, 10}, LCW{}, LCW{88}, LCW{}, LCW{99, 99}, LCW{10}, LCW{88}, LCW{}, LCW{99, 99}},
        mask_vector{1, 0, 1, 0, 1, 1, 1, 0, 1}.begin()),
    {1, 1, 1, 1, 0, 0, 1, 1, 0});
  offset_t ex_offsets{0, 2, 5, 6, 9};
  auto expected =
    cudf::make_lists_column(4, ex_offsets.release(), ex_child.release(), 0, rmm::device_buffer{});

  auto result = single_scalar_scatter(*col, *slr, scatter_map);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
}

TYPED_TEST(ScatterListOfStructScalarTest, EmptyValidScalar)
{
  using LCW      = cudf::test::lists_column_wrapper<TypeParam, int32_t>;
  using offset_t = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  auto data = this->make_test_structs({}, {}, LCW{}, {});
  auto slr  = std::make_unique<cudf::list_scalar>(data, true);

  auto child = this->make_test_structs(
    {{1, 1, 2, 3, 3, 3}, {0, 1, 1, 1, 0, 0}},
    {{"x", "x", "yy", "", "zzz", "zzz"}, {true, true, true, false, true, true}},
    LCW({LCW{10, 10}, LCW{}, LCW{10}, LCW{20, 20}, LCW{}, LCW{30, 30}},
        mask_vector{1, 0, 1, 1, 0, 1}.begin()),
    {1, 1, 0, 0, 1, 1});
  offset_t offsets{0, 2, 2, 3, 6};
  auto col =
    cudf::make_lists_column(4, offsets.release(), child.release(), 0, rmm::device_buffer{});

  size_column scatter_map{0, 2};

  auto ex_child =
    this->make_test_structs({{3, 3, 3}, {1, 0, 0}},
                            {{"", "zzz", "zzz"}, {false, true, true}},
                            LCW({LCW{20, 20}, LCW{}, LCW{30, 30}}, mask_vector{1, 0, 1}.begin()),
                            {0, 1, 1});
  offset_t ex_offsets{0, 0, 0, 0, 3};
  auto expected =
    cudf::make_lists_column(4, ex_offsets.release(), ex_child.release(), 0, rmm::device_buffer{});

  auto result = single_scalar_scatter(*col, *slr, scatter_map);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
}

TYPED_TEST(ScatterListOfStructScalarTest, NullScalar)
{
  using LCW      = cudf::test::lists_column_wrapper<TypeParam, int32_t>;
  using offset_t = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  auto data = this->make_test_structs({}, {}, {}, {});
  auto slr  = std::make_unique<cudf::list_scalar>(data, false);

  auto child = this->make_test_structs(
    {{1, 1, 2, 3, 3, 3}, {0, 1, 1, 1, 0, 0}},
    {{"x", "x", "yy", "", "zzz", "zzz"}, {true, true, true, false, true, true}},
    LCW({LCW{10, 10}, LCW{}, LCW{10}, LCW{20, 20}, LCW{}, LCW{30, 30}},
        mask_vector{1, 0, 1, 1, 0, 1}.begin()),
    {1, 1, 1, 0, 1, 1});
  offset_t offsets{0, 2, 2, 3, 6};
  auto col =
    cudf::make_lists_column(4, offsets.release(), child.release(), 0, rmm::device_buffer{});

  size_column scatter_map{3, 1, 0};

  auto ex_child = this->make_test_structs({2}, {"yy"}, LCW({10}, mask_vector{1}.begin()), {1});
  offset_t ex_offsets{0, 0, 0, 1, 1};

  auto null_mask = cudf::create_null_mask(4, cudf::mask_state::ALL_NULL);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(null_mask.data()), 2, 3, true);
  auto expected =
    cudf::make_lists_column(4, ex_offsets.release(), ex_child.release(), 3, std::move(null_mask));

  auto result = single_scalar_scatter(*col, *slr, scatter_map);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
}

TYPED_TEST(ScatterListOfStructScalarTest, NullableTargetRow)
{
  using LCW      = cudf::test::lists_column_wrapper<TypeParam, int32_t>;
  using offset_t = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  auto data =
    this->make_test_structs({{42, 42, 42}, {1, 0, 1}},
                            {{"hello", "你好！", "bonjour!"}, {false, true, true}},
                            LCW({LCW{88}, LCW{}, LCW{99, 99}}, mask_vector{1, 0, 1}.begin()),
                            {1, 1, 0});
  auto slr = std::make_unique<cudf::list_scalar>(data, true);

  auto child = this->make_test_structs(
    {{1, 1, 2, 3, 3, 3}, {0, 1, 1, 1, 0, 0}},
    {{"x", "x", "yy", "", "zzz", "zzz"}, {true, true, true, false, true, true}},
    LCW({LCW{10, 10}, LCW{}, LCW{10}, LCW{20, 20}, LCW{}, LCW{30, 30}},
        mask_vector{1, 0, 1, 1, 0, 1}.begin()),
    {1, 1, 1, 0, 1, 1});
  offset_t offsets{0, 2, 2, 3, 6};
  auto null_mask = cudf::create_null_mask(4, cudf::mask_state::ALL_VALID);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(null_mask.data()), 1, 3, false);
  auto col =
    cudf::make_lists_column(4, offsets.release(), child.release(), 2, std::move(null_mask));

  size_column scatter_map{3, 2};

  auto ex_child = this->make_test_structs(
    {{1, 1, 42, 42, 42, 42, 42, 42}, {0, 1, 1, 0, 1, 1, 0, 1}},
    {{"x", "x", "hello", "你好！", "bonjour!", "hello", "你好！", "bonjour!"},
     {true, true, false, true, true, false, true, true}},
    LCW({LCW{10, 10}, LCW{}, LCW{88}, LCW{}, LCW{99, 99}, LCW{88}, LCW{}, LCW{99, 99}},
        mask_vector{1, 0, 1, 0, 1, 1, 0, 1}.begin()),
    {1, 1, 1, 1, 0, 1, 1, 0});
  offset_t ex_offsets{0, 2, 2, 5, 8};

  auto ex_null_mask = cudf::create_null_mask(4, cudf::mask_state::ALL_VALID);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(ex_null_mask.data()), 1, 2, false);
  auto expected = cudf::make_lists_column(
    4, ex_offsets.release(), ex_child.release(), 1, std::move(ex_null_mask));

  auto result = single_scalar_scatter(*col, *slr, scatter_map);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
}
