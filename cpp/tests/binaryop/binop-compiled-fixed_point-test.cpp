/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <cuda/iterator>

template <typename T>
struct FixedPointCompiledTest : public cudf::test::BaseFixture {};

template <typename T>
using wrapper = cudf::test::fixed_width_column_wrapper<T>;
TYPED_TEST_SUITE(FixedPointCompiledTest, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpAdd)
{
  using namespace numeric;
  using decimalXX = TypeParam;

  auto const sz = std::size_t{1000};

  auto begin = cudf::detail::make_counting_transform_iterator(
    1, [](auto i) { return decimalXX{i, scale_type{0}}; });
  auto const vec1 = std::vector<decimalXX>(begin, begin + sz);
  auto const vec2 = std::vector<decimalXX>(sz, decimalXX{2, scale_type{0}});
  auto expected   = std::vector<decimalXX>(sz);

  std::transform(std::cbegin(vec1),
                 std::cend(vec1),
                 std::cbegin(vec2),
                 std::begin(expected),
                 std::plus<decimalXX>());

  auto const lhs          = wrapper<decimalXX>(vec1.begin(), vec1.end());
  auto const rhs          = wrapper<decimalXX>(vec2.begin(), vec2.end());
  auto const expected_col = wrapper<decimalXX>(expected.begin(), expected.end());

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::ADD,
                                                   static_cast<cudf::column_view>(lhs).type(),
                                                   static_cast<cudf::column_view>(rhs).type());
  auto const result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpMultiply)
{
  using namespace numeric;
  using decimalXX = TypeParam;

  auto const sz = std::size_t{1000};

  auto begin = cudf::detail::make_counting_transform_iterator(
    1, [](auto i) { return decimalXX{i, scale_type{0}}; });
  auto const vec1 = std::vector<decimalXX>(begin, begin + sz);
  auto const vec2 = std::vector<decimalXX>(sz, decimalXX{2, scale_type{0}});
  auto expected   = std::vector<decimalXX>(sz);

  std::transform(std::cbegin(vec1),
                 std::cend(vec1),
                 std::cbegin(vec2),
                 std::begin(expected),
                 std::multiplies<decimalXX>());

  auto const lhs          = wrapper<decimalXX>(vec1.begin(), vec1.end());
  auto const rhs          = wrapper<decimalXX>(vec2.begin(), vec2.end());
  auto const expected_col = wrapper<decimalXX>(expected.begin(), expected.end());

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::MUL,
                                                   static_cast<cudf::column_view>(lhs).type(),
                                                   static_cast<cudf::column_view>(rhs).type());
  auto const result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::MUL, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col, result->view());
}

template <typename T>
using fp_wrapper = cudf::test::fixed_point_column_wrapper<T>;

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpMultiply2)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{11, 22, 33, 44, 55}, scale_type{-1}};
  auto const rhs      = fp_wrapper<RepType>{{10, 10, 10, 10, 10}, scale_type{0}};
  auto const expected = fp_wrapper<RepType>{{110, 220, 330, 440, 550}, scale_type{-1}};

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::MUL,
                                                   static_cast<cudf::column_view>(lhs).type(),
                                                   static_cast<cudf::column_view>(rhs).type());
  auto const result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::MUL, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpDiv)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{10, 30, 50, 70}, scale_type{-1}};
  auto const rhs      = fp_wrapper<RepType>{{4, 4, 4, 4}, scale_type{0}};
  auto const expected = fp_wrapper<RepType>{{2, 7, 12, 17}, scale_type{-1}};

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::DIV,
                                                   static_cast<cudf::column_view>(lhs).type(),
                                                   static_cast<cudf::column_view>(rhs).type());
  auto const result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpDiv2)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{10, 30, 50, 70}, scale_type{-1}};
  auto const rhs      = fp_wrapper<RepType>{{4, 4, 4, 4}, scale_type{-2}};
  auto const expected = fp_wrapper<RepType>{{2, 7, 12, 17}, scale_type{1}};

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::DIV,
                                                   static_cast<cudf::column_view>(lhs).type(),
                                                   static_cast<cudf::column_view>(rhs).type());
  auto const result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpDiv3)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{10, 30, 50, 70}, scale_type{-1}};
  auto const rhs      = cudf::make_fixed_point_scalar<decimalXX>(12, scale_type{-1});
  auto const expected = fp_wrapper<RepType>{{0, 2, 4, 5}, scale_type{0}};

  auto const type = cudf::binary_operation_fixed_point_output_type(
    cudf::binary_operator::DIV, static_cast<cudf::column_view>(lhs).type(), rhs->type());
  auto const result = cudf::binary_operation(lhs, *rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpDiv4)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto begin = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 11; });
  auto result_begin =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i * 11) / 12; });
  auto const lhs      = fp_wrapper<RepType>(begin, begin + 1000, scale_type{-1});
  auto const rhs      = cudf::make_fixed_point_scalar<decimalXX>(12, scale_type{-1});
  auto const expected = fp_wrapper<RepType>(result_begin, result_begin + 1000, scale_type{0});

  auto const type = cudf::binary_operation_fixed_point_output_type(
    cudf::binary_operator::DIV, static_cast<cudf::column_view>(lhs).type(), rhs->type());
  auto const result = cudf::binary_operation(lhs, *rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpAdd2)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{11, 22, 33, 44, 55}, scale_type{-1}};
  auto const rhs      = fp_wrapper<RepType>{{100, 200, 300, 400, 500}, scale_type{-2}};
  auto const expected = fp_wrapper<RepType>{{210, 420, 630, 840, 1050}, scale_type{-2}};

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::ADD,
                                                   static_cast<cudf::column_view>(lhs).type(),
                                                   static_cast<cudf::column_view>(rhs).type());
  auto const result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpAdd3)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{1100, 2200, 3300, 4400, 5500}, scale_type{-3}};
  auto const rhs      = fp_wrapper<RepType>{{100, 200, 300, 400, 500}, scale_type{-2}};
  auto const expected = fp_wrapper<RepType>{{2100, 4200, 6300, 8400, 10500}, scale_type{-3}};

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::ADD,
                                                   static_cast<cudf::column_view>(lhs).type(),
                                                   static_cast<cudf::column_view>(rhs).type());
  auto const result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpAdd4)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{11, 22, 33, 44, 55}, scale_type{-1}};
  auto const rhs      = cudf::make_fixed_point_scalar<decimalXX>(100, scale_type{-2});
  auto const expected = fp_wrapper<RepType>{{210, 320, 430, 540, 650}, scale_type{-2}};

  auto const type = cudf::binary_operation_fixed_point_output_type(
    cudf::binary_operator::ADD, static_cast<cudf::column_view>(lhs).type(), rhs->type());
  auto const result = cudf::binary_operation(lhs, *rhs, cudf::binary_operator::ADD, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpAdd5)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs      = cudf::make_fixed_point_scalar<decimalXX>(100, scale_type{-2});
  auto const rhs      = fp_wrapper<RepType>{{11, 22, 33, 44, 55}, scale_type{-1}};
  auto const expected = fp_wrapper<RepType>{{210, 320, 430, 540, 650}, scale_type{-2}};

  auto const type = cudf::binary_operation_fixed_point_output_type(
    cudf::binary_operator::ADD, lhs->type(), static_cast<cudf::column_view>(rhs).type());
  auto const result = cudf::binary_operation(*lhs, rhs, cudf::binary_operator::ADD, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpAdd6)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const col = fp_wrapper<RepType>{{30, 4, 5, 6, 7, 8}, scale_type{0}};

  auto const expected1 = fp_wrapper<RepType>{{60, 8, 10, 12, 14, 16}, scale_type{0}};
  auto const expected2 = fp_wrapper<RepType>{{6, 0, 1, 1, 1, 1}, scale_type{1}};
  auto const type1     = cudf::data_type{cudf::type_to_id<decimalXX>(), 0};
  auto const type2     = cudf::data_type{cudf::type_to_id<decimalXX>(), 1};
  auto const result1   = cudf::binary_operation(col, col, cudf::binary_operator::ADD, type1);
  auto const result2   = cudf::binary_operation(col, col, cudf::binary_operator::ADD, type2);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, result2->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, result1->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointCast)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const col      = fp_wrapper<RepType>{{6, 8, 10, 12, 14, 16}, scale_type{0}};
  auto const expected = fp_wrapper<RepType>{{0, 0, 1, 1, 1, 1}, scale_type{1}};
  auto const type     = cudf::data_type{cudf::type_to_id<decimalXX>(), 1};
  auto const result   = cudf::cast(col, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpMultiplyScalar)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{11, 22, 33, 44, 55}, scale_type{-1}};
  auto const rhs      = cudf::make_fixed_point_scalar<decimalXX>(100, scale_type{-1});
  auto const expected = fp_wrapper<RepType>{{1100, 2200, 3300, 4400, 5500}, scale_type{-2}};

  auto const type = cudf::binary_operation_fixed_point_output_type(
    cudf::binary_operator::MUL, static_cast<cudf::column_view>(lhs).type(), rhs->type());
  auto const result = cudf::binary_operation(lhs, *rhs, cudf::binary_operator::MUL, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpSimplePlus)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{150, 200}, scale_type{-2}};
  auto const rhs      = fp_wrapper<RepType>{{2250, 1005}, scale_type{-3}};
  auto const expected = fp_wrapper<RepType>{{3750, 3005}, scale_type{-3}};

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::ADD,
                                                   static_cast<cudf::column_view>(lhs).type(),
                                                   static_cast<cudf::column_view>(rhs).type());
  auto const result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpEqualSimple)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const trues    = std::vector<bool>(4, true);
  auto const col1     = fp_wrapper<RepType>{{1, 2, 3, 4}, scale_type{0}};
  auto const col2     = fp_wrapper<RepType>{{100, 200, 300, 400}, scale_type{-2}};
  auto const expected = wrapper<bool>(trues.begin(), trues.end());

  auto const result = cudf::binary_operation(
    col1, col2, cudf::binary_operator::EQUAL, cudf::data_type{cudf::type_id::BOOL8});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpEqualSimpleScale0)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const trues    = std::vector<bool>(4, true);
  auto const col      = fp_wrapper<RepType>{{1, 2, 3, 4}, scale_type{0}};
  auto const expected = wrapper<bool>(trues.begin(), trues.end());

  auto const result = cudf::binary_operation(
    col, col, cudf::binary_operator::EQUAL, cudf::data_type{cudf::type_id::BOOL8});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpEqualSimpleScale0Null)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const col1     = fp_wrapper<RepType>{{1, 2, 3, 4}, {1, 1, 1, 1}, scale_type{0}};
  auto const col2     = fp_wrapper<RepType>{{1, 2, 3, 4}, {0, 0, 0, 0}, scale_type{0}};
  auto const expected = wrapper<bool>{{0, 1, 0, 1}, {false, false, false, false}};

  auto const result = cudf::binary_operation(
    col1, col2, cudf::binary_operator::EQUAL, cudf::data_type{cudf::type_id::BOOL8});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpEqualSimpleScale2Null)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const col1     = fp_wrapper<RepType>{{1, 2, 3, 4}, {1, 1, 1, 1}, scale_type{-2}};
  auto const col2     = fp_wrapper<RepType>{{1, 2, 3, 4}, {0, 0, 0, 0}, scale_type{0}};
  auto const expected = wrapper<bool>{{0, 1, 0, 1}, {false, false, false, false}};

  auto const result = cudf::binary_operation(
    col1, col2, cudf::binary_operator::EQUAL, cudf::data_type{cudf::type_id::BOOL8});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpEqualLessGreater)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const sz = std::size_t{1000};

  // TESTING binary op ADD

  auto begin = cudf::detail::make_counting_transform_iterator(1, [](auto e) { return e * 1000; });
  auto const vec1 = std::vector<RepType>(begin, begin + sz);
  auto const vec2 = std::vector<RepType>(sz, 0);

  auto const iota_3  = fp_wrapper<RepType>(vec1.begin(), vec1.end(), scale_type{-3});
  auto const zeros_3 = fp_wrapper<RepType>(vec2.begin(), vec2.end(), scale_type{-1});

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::ADD,
                                                   static_cast<cudf::column_view>(iota_3).type(),
                                                   static_cast<cudf::column_view>(zeros_3).type());
  auto const iota_3_after_add =
    cudf::binary_operation(zeros_3, iota_3, cudf::binary_operator::ADD, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(iota_3, iota_3_after_add->view());

  // TESTING binary op EQUAL, LESS, GREATER

  auto const trues    = std::vector<bool>(sz, true);
  auto const true_col = wrapper<bool>(trues.begin(), trues.end());

  auto const btype = cudf::data_type{cudf::type_id::BOOL8};
  auto const equal_result =
    cudf::binary_operation(iota_3, iota_3_after_add->view(), cudf::binary_operator::EQUAL, btype);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(true_col, equal_result->view());

  auto const less_result =
    cudf::binary_operation(zeros_3, iota_3_after_add->view(), cudf::binary_operator::LESS, btype);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(true_col, less_result->view());

  auto const greater_result = cudf::binary_operation(
    iota_3_after_add->view(), zeros_3, cudf::binary_operator::GREATER, btype);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(true_col, greater_result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpNullMaxSimple)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const col1     = fp_wrapper<RepType>{{40, 30, 20, 10, 0}, {1, 0, 1, 1, 0}, scale_type{-2}};
  auto const col2     = fp_wrapper<RepType>{{10, 20, 30, 40, 0}, {1, 1, 1, 0, 0}, scale_type{-2}};
  auto const expected = fp_wrapper<RepType>{{40, 20, 30, 10, 0}, {1, 1, 1, 1, 0}, scale_type{-2}};

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::NULL_MAX,
                                                   static_cast<cudf::column_view>(col1).type(),
                                                   static_cast<cudf::column_view>(col2).type());
  auto const result = cudf::binary_operation(col1, col2, cudf::binary_operator::NULL_MAX, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpNullMinSimple)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const col1     = fp_wrapper<RepType>{{40, 30, 20, 10, 0}, {1, 1, 1, 0, 0}, scale_type{-1}};
  auto const col2     = fp_wrapper<RepType>{{10, 20, 30, 40, 0}, {1, 0, 1, 1, 0}, scale_type{-1}};
  auto const expected = fp_wrapper<RepType>{{10, 30, 20, 40, 0}, {1, 1, 1, 1, 0}, scale_type{-1}};

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::NULL_MIN,
                                                   static_cast<cudf::column_view>(col1).type(),
                                                   static_cast<cudf::column_view>(col2).type());
  auto const result = cudf::binary_operation(col1, col2, cudf::binary_operator::NULL_MIN, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpNullEqualsSimple)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const col1     = fp_wrapper<RepType>{{400, 300, 300, 100}, {1, 1, 1, 0}, scale_type{-2}};
  auto const col2     = fp_wrapper<RepType>{{40, 200, 20, 400}, {1, 0, 1, 0}, scale_type{-1}};
  auto const expected = wrapper<bool>{{1, 0, 0, 1}, {true, true, true, true}};

  auto const result = cudf::binary_operation(
    col1, col2, cudf::binary_operator::NULL_EQUALS, cudf::data_type{cudf::type_id::BOOL8});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpNullMaxColumnScalar)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  // DECIMAL scale -2: column {1.00, 5.00, 3.00}, scalar 5.00 -> greatest is 5.00.
  // NULL_MAX always produces a nullable output, so the expected column is nullable (all valid).
  auto const col      = fp_wrapper<RepType>{{100, 500, 300}, scale_type{-2}};
  auto const scalar   = cudf::make_fixed_point_scalar<decimalXX>(500, scale_type{-2});
  auto const expected = fp_wrapper<RepType>{{500, 500, 500}, {1, 1, 1}, scale_type{-2}};

  auto const type = cudf::binary_operation_fixed_point_output_type(
    cudf::binary_operator::NULL_MAX, static_cast<cudf::column_view>(col).type(), scalar->type());
  auto const result = cudf::binary_operation(col, *scalar, cudf::binary_operator::NULL_MAX, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpNullEqualsNullColumnScalar)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const col = fp_wrapper<RepType>{{100, 500, 300}, {1, 0, 1}, scale_type{-2}};

  // A null scalar compared with NULL_EQUALS yields true only where the column value is also null.
  auto scalar = cudf::make_fixed_point_scalar<decimalXX>(500, scale_type{-2});
  scalar->set_valid_async(false);
  auto const expected = wrapper<bool>{{0, 1, 0}, {true, true, true}};

  auto const result = cudf::binary_operation(
    col, *scalar, cudf::binary_operator::NULL_EQUALS, cudf::data_type{cudf::type_id::BOOL8});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOp_Div)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{100, 300, 500, 700}, scale_type{-2}};
  auto const rhs      = fp_wrapper<RepType>{{4, 4, 4, 4}, scale_type{0}};
  auto const expected = fp_wrapper<RepType>{{25, 75, 125, 175}, scale_type{-2}};

  auto const type   = cudf::data_type{cudf::type_to_id<decimalXX>(), -2};
  auto const result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOp_Div2)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{100000, 300000, 500000, 700000}, scale_type{-3}};
  auto const rhs      = fp_wrapper<RepType>{{20, 20, 20, 20}, scale_type{-1}};
  auto const expected = fp_wrapper<RepType>{{5000, 15000, 25000, 35000}, scale_type{-2}};

  auto const type   = cudf::data_type{cudf::type_to_id<decimalXX>(), -2};
  auto const result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOp_Div3)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{10000, 30000, 50000, 70000}, scale_type{-2}};
  auto const rhs      = fp_wrapper<RepType>{{3, 9, 3, 3}, scale_type{0}};
  auto const expected = fp_wrapper<RepType>{{3333, 3333, 16666, 23333}, scale_type{-2}};

  auto const type   = cudf::data_type{cudf::type_to_id<decimalXX>(), -2};
  auto const result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOp_Div4)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{10, 30, 50, 70}, scale_type{1}};
  auto const rhs      = cudf::make_fixed_point_scalar<decimalXX>(3, scale_type{0});
  auto const expected = fp_wrapper<RepType>{{3, 10, 16, 23}, scale_type{1}};

  auto const type   = cudf::data_type{cudf::type_to_id<decimalXX>(), 1};
  auto const result = cudf::binary_operation(lhs, *rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOp_Div6)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs = cudf::make_fixed_point_scalar<decimalXX>(3000, scale_type{-3});
  auto const rhs = fp_wrapper<RepType>{{10, 30, 50, 70}, scale_type{-1}};

  auto const expected = fp_wrapper<RepType>{{300, 100, 60, 42}, scale_type{-2}};

  auto const type   = cudf::data_type{cudf::type_to_id<decimalXX>(), -2};
  auto const result = cudf::binary_operation(*lhs, rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOp_Div7)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs = cudf::make_fixed_point_scalar<decimalXX>(1200, scale_type{0});
  auto const rhs = fp_wrapper<RepType>{{100, 200, 300, 500, 600, 800, 1200, 1300}, scale_type{-2}};

  auto const expected = fp_wrapper<RepType>{{12, 6, 4, 2, 2, 1, 1, 0}, scale_type{2}};

  auto const type   = cudf::data_type{cudf::type_to_id<decimalXX>(), 2};
  auto const result = cudf::binary_operation(*lhs, rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOp_Div8)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{4000, 6000, 80000}, scale_type{-1}};
  auto const rhs      = cudf::make_fixed_point_scalar<decimalXX>(5000, scale_type{-3});
  auto const expected = fp_wrapper<RepType>{{0, 1, 16}, scale_type{2}};

  auto const type   = cudf::data_type{cudf::type_to_id<decimalXX>(), 2};
  auto const result = cudf::binary_operation(lhs, *rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOp_Div9)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{10, 20, 30}, scale_type{2}};
  auto const rhs      = cudf::make_fixed_point_scalar<decimalXX>(7, scale_type{1});
  auto const expected = fp_wrapper<RepType>{{1, 2, 4}, scale_type{1}};

  auto const type   = cudf::data_type{cudf::type_to_id<decimalXX>(), 1};
  auto const result = cudf::binary_operation(lhs, *rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOp_Div10)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{100, 200, 300}, scale_type{1}};
  auto const rhs      = cudf::make_fixed_point_scalar<decimalXX>(7, scale_type{0});
  auto const expected = fp_wrapper<RepType>{{14, 28, 42}, scale_type{1}};

  auto const type   = cudf::data_type{cudf::type_to_id<decimalXX>(), 1};
  auto const result = cudf::binary_operation(lhs, *rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOp_Div11)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{1000, 2000, 3000}, scale_type{1}};
  auto const rhs      = fp_wrapper<RepType>{{7, 7, 7}, scale_type{0}};
  auto const expected = fp_wrapper<RepType>{{142, 285, 428}, scale_type{1}};

  auto const type   = cudf::data_type{cudf::type_to_id<decimalXX>(), 1};
  auto const result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpThrows)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const col           = fp_wrapper<RepType>{{100, 300, 500, 700}, scale_type{-2}};
  auto const non_bool_type = cudf::data_type{cudf::type_to_id<decimalXX>(), -2};
  EXPECT_THROW(cudf::binary_operation(col, col, cudf::binary_operator::LESS, non_bool_type),
               cudf::data_type_error);
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpModSimple)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{-33, -22, -11, 11, 22, 33, 44, 55}, scale_type{-1}};
  auto const rhs      = fp_wrapper<RepType>{{10, 10, 10, 10, 10, 10, 10, 10}, scale_type{-1}};
  auto const expected = fp_wrapper<RepType>{{-3, -2, -1, 1, 2, 3, 4, 5}, scale_type{-1}};

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::MOD,
                                                   static_cast<cudf::column_view>(lhs).type(),
                                                   static_cast<cudf::column_view>(rhs).type());
  auto const result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::MOD, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpPModSimple)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{-33, -22, -11, 11, 22, 33, 44, 55}, scale_type{-1}};
  auto const rhs      = fp_wrapper<RepType>{{10, 10, 10, 10, 10, 10, 10, 10}, scale_type{-1}};
  auto const expected = fp_wrapper<RepType>{{7, 8, 9, 1, 2, 3, 4, 5}, scale_type{-1}};

  for (auto const op : {cudf::binary_operator::PMOD, cudf::binary_operator::PYMOD}) {
    auto const type = cudf::binary_operation_fixed_point_output_type(
      op, static_cast<cudf::column_view>(lhs).type(), static_cast<cudf::column_view>(rhs).type());
    auto const result = cudf::binary_operation(lhs, rhs, op, type);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpModSimple2)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{-33, -22, -11, 11, 22, 33, 44, 55}, scale_type{-1}};
  auto const rhs      = cudf::make_fixed_point_scalar<decimalXX>(10, scale_type{-1});
  auto const expected = fp_wrapper<RepType>{{-3, -2, -1, 1, 2, 3, 4, 5}, scale_type{-1}};

  auto const type = cudf::binary_operation_fixed_point_output_type(
    cudf::binary_operator::MOD, static_cast<cudf::column_view>(lhs).type(), rhs->type());
  auto const result = cudf::binary_operation(lhs, *rhs, cudf::binary_operator::MOD, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpPModAndPyModSimple2)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{-33, -22, -11, 11, 22, 33, 44, 55}, scale_type{-1}};
  auto const rhs      = cudf::make_fixed_point_scalar<decimalXX>(10, scale_type{-1});
  auto const expected = fp_wrapper<RepType>{{7, 8, 9, 1, 2, 3, 4, 5}, scale_type{-1}};

  for (auto const op : {cudf::binary_operator::PMOD, cudf::binary_operator::PYMOD}) {
    auto const type = cudf::binary_operation_fixed_point_output_type(
      op, static_cast<cudf::column_view>(lhs).type(), rhs->type());
    auto const result = cudf::binary_operation(lhs, *rhs, op, type);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpMod)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  auto constexpr N = 1000;

  for (auto scale : {-1, -2, -3}) {
    auto const iota = cuda::counting_iterator{-500};
    auto const lhs  = fp_wrapper<RepType>{iota, iota + N, scale_type{-1}};
    auto const rhs  = cudf::make_fixed_point_scalar<decimalXX>(7, scale_type{scale});

    auto const factor   = static_cast<int>(std::pow(10, -1 - scale));
    auto const f        = [factor](auto i) { return (i * factor) % 7; };
    auto const exp_iter = cudf::detail::make_counting_transform_iterator(-500, f);
    auto const expected = fp_wrapper<RepType>{exp_iter, exp_iter + N, scale_type{scale}};

    auto const type = cudf::binary_operation_fixed_point_output_type(
      cudf::binary_operator::MOD, static_cast<cudf::column_view>(lhs).type(), rhs->type());
    auto const result = cudf::binary_operation(lhs, *rhs, cudf::binary_operator::MOD, type);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}

TYPED_TEST(FixedPointCompiledTest, FixedPointBinaryOpPModAndPyMod)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  auto constexpr N = 1000;

  for (auto const scale : {-1, -2, -3}) {
    auto const iota = cuda::counting_iterator{-500};
    auto const lhs  = fp_wrapper<RepType>{iota, iota + N, scale_type{-1}};
    auto const rhs  = cudf::make_fixed_point_scalar<decimalXX>(7, scale_type{scale});

    auto const factor   = static_cast<int>(std::pow(10, -1 - scale));
    auto const f        = [factor](auto i) { return (((i * factor) % 7) + 7) % 7; };
    auto const exp_iter = cudf::detail::make_counting_transform_iterator(-500, f);
    auto const expected = fp_wrapper<RepType>{exp_iter, exp_iter + N, scale_type{scale}};

    for (auto const op : {cudf::binary_operator::PMOD, cudf::binary_operator::PYMOD}) {
      auto const type = cudf::binary_operation_fixed_point_output_type(
        op, static_cast<cudf::column_view>(lhs).type(), rhs->type());
      auto const result = cudf::binary_operation(lhs, *rhs, op, type);

      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
    }
  }
}

template <typename T>
struct FixedPointTest_64_128_Reps : public cudf::test::BaseFixture {};

using Decimal64And128Types = cudf::test::Types<numeric::decimal64, numeric::decimal128>;
TYPED_TEST_SUITE(FixedPointTest_64_128_Reps, Decimal64And128Types);

TYPED_TEST(FixedPointTest_64_128_Reps, FixedPoint_64_128_ComparisonTests)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  for (auto const rhs_value : {10000000000000000, 100000000000000000}) {
    auto const lhs       = fp_wrapper<RepType>{{33041, 97290, 36438, 25379, 48473}, scale_type{2}};
    auto const rhs       = cudf::make_fixed_point_scalar<decimalXX>(rhs_value, scale_type{0});
    auto const trues     = wrapper<bool>{{1, 1, 1, 1, 1}};
    auto const falses    = wrapper<bool>{{0, 0, 0, 0, 0}};
    auto const bool_type = cudf::data_type{cudf::type_id::BOOL8};

    auto const a = cudf::binary_operation(lhs, *rhs, cudf::binary_operator::LESS, bool_type);
    auto const b = cudf::binary_operation(lhs, *rhs, cudf::binary_operator::LESS_EQUAL, bool_type);
    auto const c = cudf::binary_operation(lhs, *rhs, cudf::binary_operator::GREATER, bool_type);
    auto const d =
      cudf::binary_operation(lhs, *rhs, cudf::binary_operator::GREATER_EQUAL, bool_type);
    auto const e = cudf::binary_operation(*rhs, lhs, cudf::binary_operator::GREATER, bool_type);
    auto const f =
      cudf::binary_operation(*rhs, lhs, cudf::binary_operator::GREATER_EQUAL, bool_type);
    auto const g = cudf::binary_operation(*rhs, lhs, cudf::binary_operator::LESS, bool_type);
    auto const h = cudf::binary_operation(*rhs, lhs, cudf::binary_operator::LESS_EQUAL, bool_type);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(a->view(), trues);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(b->view(), trues);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(c->view(), falses);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(d->view(), falses);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(e->view(), trues);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(f->view(), trues);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(g->view(), falses);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(h->view(), falses);
  }
}

template <typename ResultType>
void test_fixed_floating(cudf::binary_operator op,
                         double floating_value,
                         int decimal_value,
                         int decimal_scale,
                         ResultType expected)
{
  auto const scale       = numeric::scale_type{decimal_scale};
  auto const result_type = cudf::data_type(cudf::type_to_id<ResultType>());
  auto const nullable =
    (op == cudf::binary_operator::NULL_EQUALS || op == cudf::binary_operator::NULL_NOT_EQUALS ||
     op == cudf::binary_operator::NULL_MIN || op == cudf::binary_operator::NULL_MAX);

  cudf::test::fixed_width_column_wrapper<double> floating_col({floating_value});
  cudf::test::fixed_point_column_wrapper<int> decimal_col({decimal_value}, scale);

  auto result = binary_operation(floating_col, decimal_col, op, result_type);

  if constexpr (cudf::is_fixed_point<ResultType>()) {
    using wrapper_type      = cudf::test::fixed_point_column_wrapper<typename ResultType::rep>;
    auto const expected_col = nullable ? wrapper_type({expected.value()}, {true}, expected.scale())
                                       : wrapper_type({expected.value()}, expected.scale());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col, *result.get());
  } else {
    using wrapper_type = cudf::test::fixed_width_column_wrapper<ResultType>;
    auto const expected_col =
      nullable ? wrapper_type({expected}, {true}) : wrapper_type({expected});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col, *result.get());
  }
}

TYPED_TEST(FixedPointCompiledTest, FixedPointWithFloating)
{
  using namespace numeric;

  // BOOLEAN
  test_fixed_floating(cudf::binary_operator::EQUAL, 1.0, 10, -1, true);
  test_fixed_floating(cudf::binary_operator::NOT_EQUAL, 1.0, 10, -1, false);
  test_fixed_floating(cudf::binary_operator::LESS, 2.0, 10, -1, false);
  test_fixed_floating(cudf::binary_operator::GREATER, 2.0, 10, -1, true);
  test_fixed_floating(cudf::binary_operator::LESS_EQUAL, 2.0, 20, -1, true);
  test_fixed_floating(cudf::binary_operator::GREATER_EQUAL, 2.0, 30, -1, false);
  test_fixed_floating(cudf::binary_operator::NULL_EQUALS, 1.0, 10, -1, true);
  test_fixed_floating(cudf::binary_operator::NULL_NOT_EQUALS, 1.0, 10, -1, false);

  // PRIMARY ARITHMETIC
  auto const decimal_result = numeric::decimal32(4, numeric::scale_type{0});
  test_fixed_floating(cudf::binary_operator::ADD, 1.0, 30, -1, decimal_result);
  test_fixed_floating(cudf::binary_operator::SUB, 6.0, 20, -1, decimal_result);
  test_fixed_floating(cudf::binary_operator::MUL, 2.0, 20, -1, decimal_result);
  test_fixed_floating(cudf::binary_operator::DIV, 8.0, 2, 0, decimal_result);
  test_fixed_floating(cudf::binary_operator::MOD, 9.0, 50, -1, decimal_result);

  // OTHER ARITHMETIC
  test_fixed_floating(cudf::binary_operator::NULL_MAX, 4.0, 20, -1, decimal_result);
  test_fixed_floating(cudf::binary_operator::NULL_MIN, 4.0, 200, -1, decimal_result);
}
