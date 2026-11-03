/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/binaryop/assert-binops.h>
#include <tests/binaryop/binop-fixture.hpp>
#include <tests/binaryop/util/operation.h>

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/reduction.hpp>
#include <cudf/types.hpp>

#include <cuda/iterator>

#include <limits>
#include <type_traits>

template <typename T>
auto lhs_random_column(cudf::size_type size)
{
  return BinaryOperationTest::make_random_wrapped_column<T>(size);
}

template <>
auto lhs_random_column<std::string>(cudf::size_type size)
{
  return cudf::test::strings_column_wrapper({"eee", "bb", "<null>", "", "aa", "bbb", "ééé"},
                                            {1, 1, 0, 1, 1, 1, 1});
}
template <typename T>
auto rhs_random_column(cudf::size_type size)
{
  return BinaryOperationTest::make_random_wrapped_column<T>(size);
}
template <>
auto rhs_random_column<std::string>(cudf::size_type size)
{
  return cudf::test::strings_column_wrapper({"ééé", "bbb", "aa", "", "<null>", "bb", "eee"},
                                            {1, 1, 1, 1, 0, 1, 1});
}

// combinations to test
//     n  t   d
// n n.n n.t n.d
// t t.n t.t t.d
// d d.n d.t d.d

constexpr cudf::size_type col_size = 10000;
template <typename T>
struct BinaryOperationCompiledTest : public BinaryOperationTest {
  using TypeOut = cudf::test::GetType<T, 0>;
  using TypeLhs = cudf::test::GetType<T, 1>;
  using TypeRhs = cudf::test::GetType<T, 2>;

  template <template <typename... Ty> class FunctorOP>
  void test(cudf::binary_operator op)
  {
    using OPERATOR = FunctorOP<TypeOut, TypeLhs, TypeRhs>;

    auto lhs = lhs_random_column<TypeLhs>(col_size);
    auto rhs = rhs_random_column<TypeRhs>(col_size);

    auto out = cudf::binary_operation(lhs, rhs, op, cudf::data_type(cudf::type_to_id<TypeOut>()));
    ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, OPERATOR());

    auto s_lhs = this->template make_random_wrapped_scalar<TypeLhs>();
    auto s_rhs = this->template make_random_wrapped_scalar<TypeRhs>();
    s_lhs.set_valid_async(true);
    s_rhs.set_valid_async(true);

    out = cudf::binary_operation(lhs, s_rhs, op, cudf::data_type(cudf::type_to_id<TypeOut>()));
    ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, s_rhs, OPERATOR());
    out = cudf::binary_operation(s_lhs, rhs, op, cudf::data_type(cudf::type_to_id<TypeOut>()));
    ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, s_lhs, rhs, OPERATOR());

    s_lhs.set_valid_async(false);
    s_rhs.set_valid_async(false);
    out = cudf::binary_operation(lhs, s_rhs, op, cudf::data_type(cudf::type_to_id<TypeOut>()));
    ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, s_rhs, OPERATOR());
    out = cudf::binary_operation(s_lhs, rhs, op, cudf::data_type(cudf::type_to_id<TypeOut>()));
    ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, s_lhs, rhs, OPERATOR());
  }
};

// ADD
//     n      t     d
// n n + n
// t      	     	t + d
// d      	d + t	d + d

using namespace numeric;

using Add_types =
  cudf::test::Types<cudf::test::Types<bool, bool, float>,
                    cudf::test::Types<int16_t, double, uint8_t>,
                    cudf::test::Types<cudf::timestamp_s, cudf::timestamp_s, cudf::duration_s>,
                    cudf::test::Types<cudf::timestamp_ns, cudf::duration_ms, cudf::timestamp_us>,
                    cudf::test::Types<cudf::duration_us, cudf::duration_us, cudf::duration_D>,
                    // cudf::test::Types<duration_s, int16_t, int64_t>, //valid
                    cudf::test::Types<decimal32, decimal32, decimal32>,
                    cudf::test::Types<decimal64, decimal64, decimal64>,
                    cudf::test::Types<decimal128, decimal128, decimal128>,
                    cudf::test::Types<int, decimal32, decimal32>,
                    cudf::test::Types<int, decimal64, decimal64>,
                    cudf::test::Types<int, decimal128, decimal128>,
                    // Extras
                    cudf::test::Types<cudf::duration_D, cudf::duration_D, cudf::duration_D>,
                    cudf::test::Types<cudf::timestamp_D, cudf::timestamp_D, cudf::duration_D>,
                    cudf::test::Types<cudf::timestamp_s, cudf::timestamp_D, cudf::duration_s>,
                    cudf::test::Types<cudf::timestamp_ms, cudf::timestamp_ms, cudf::duration_s>,
                    cudf::test::Types<cudf::timestamp_ns, cudf::timestamp_ms, cudf::duration_ns>>;

template <typename T>
struct BinaryOperationCompiledTest_Add : public BinaryOperationCompiledTest<T> {};
TYPED_TEST_SUITE(BinaryOperationCompiledTest_Add, Add_types);

TYPED_TEST(BinaryOperationCompiledTest_Add, Vector_Vector)
{
  this->template test<cudf::library::operation::Add>(cudf::binary_operator::ADD);
}

// SUB
//     n      t     d
// n n - n
// t      	t - t	t - d
// d      	     	d - d

using Sub_types = cudf::test::Types<
  cudf::test::Types<int32_t, bool, float>,                                    // n - n
  cudf::test::Types<cudf::duration_D, cudf::timestamp_D, cudf::timestamp_D>,  // t - t
  cudf::test::Types<cudf::timestamp_s, cudf::timestamp_D, cudf::duration_s>,  // t - d
  cudf::test::Types<cudf::duration_ns, cudf::duration_us, cudf::duration_s>,  // d - d
  cudf::test::Types<cudf::duration_us, cudf::duration_us, cudf::duration_s>,  // d - d
  cudf::test::Types<decimal32, decimal32, decimal32>,
  cudf::test::Types<decimal64, decimal64, decimal64>,
  cudf::test::Types<decimal128, decimal128, decimal128>,
  cudf::test::Types<int, decimal32, decimal32>,
  cudf::test::Types<int, decimal64, decimal64>,
  cudf::test::Types<int, decimal128, decimal128>>;

template <typename T>
struct BinaryOperationCompiledTest_Sub : public BinaryOperationCompiledTest<T> {};
TYPED_TEST_SUITE(BinaryOperationCompiledTest_Sub, Sub_types);

TYPED_TEST(BinaryOperationCompiledTest_Sub, Vector_Vector)
{
  this->template test<cudf::library::operation::Sub>(cudf::binary_operator::SUB);
}

// MUL
//     n      t     d
// n n * n	     	n * d
// t
// d d * n
using Mul_types =
  cudf::test::Types<cudf::test::Types<int32_t, u_int64_t, float>,
                    cudf::test::Types<cudf::duration_s, u_int64_t, cudf::duration_s>,
                    cudf::test::Types<cudf::duration_ms, cudf::duration_D, int16_t>,
                    cudf::test::Types<cudf::duration_ns, cudf::duration_us, uint8_t>,
                    cudf::test::Types<decimal32, decimal32, decimal32>,
                    cudf::test::Types<decimal64, decimal64, decimal64>,
                    cudf::test::Types<decimal128, decimal128, decimal128>,
                    cudf::test::Types<int, decimal32, decimal32>,
                    cudf::test::Types<int, decimal64, decimal64>,
                    cudf::test::Types<int, decimal128, decimal128>,
                    cudf::test::Types<decimal32, int, int>,
                    cudf::test::Types<decimal64, int, int>,
                    cudf::test::Types<decimal128, int, int>>;

template <typename T>
struct BinaryOperationCompiledTest_Mul : public BinaryOperationCompiledTest<T> {};
TYPED_TEST_SUITE(BinaryOperationCompiledTest_Mul, Mul_types);

TYPED_TEST(BinaryOperationCompiledTest_Mul, Vector_Vector)
{
  this->template test<cudf::library::operation::Mul>(cudf::binary_operator::MUL);
}

// DIV
//     n      t     d
// n n / n
// t
// d d / n	     	d / d
using Div_types =
  cudf::test::Types<cudf::test::Types<int16_t, u_int64_t, u_int64_t>,
                    cudf::test::Types<double, int8_t, int64_t>,
                    cudf::test::Types<cudf::duration_ms, cudf::duration_s, u_int32_t>,
                    cudf::test::Types<cudf::duration_ns, cudf::duration_D, int16_t>,
                    cudf::test::Types<double, cudf::duration_D, cudf::duration_ns>,
                    cudf::test::Types<float, cudf::duration_ms, cudf::duration_ns>,
                    cudf::test::Types<u_int64_t, cudf::duration_us, cudf::duration_ns>,
                    cudf::test::Types<decimal32, decimal32, decimal32>,
                    cudf::test::Types<decimal64, decimal64, decimal64>,
                    cudf::test::Types<decimal128, decimal128, decimal128>,
                    cudf::test::Types<int, decimal32, decimal32>,
                    cudf::test::Types<int, decimal64, decimal64>,
                    cudf::test::Types<int, decimal128, decimal128>>;

template <typename T>
struct BinaryOperationCompiledTest_Div : public BinaryOperationCompiledTest<T> {};
TYPED_TEST_SUITE(BinaryOperationCompiledTest_Div, Div_types);

TYPED_TEST(BinaryOperationCompiledTest_Div, Vector_Vector)
{
  this->template test<cudf::library::operation::Div>(cudf::binary_operator::DIV);
}

// TRUE-DIV
//     n      t     d
// n n / n
// t
// d
using TrueDiv_types = cudf::test::Types<cudf::test::Types<int16_t, u_int64_t, u_int64_t>,
                                        cudf::test::Types<double, int8_t, int64_t>,
                                        cudf::test::Types<int8_t, bool, u_int32_t>,
                                        cudf::test::Types<u_int64_t, float, int16_t>>;

template <typename T>
struct BinaryOperationCompiledTest_TrueDiv : public BinaryOperationCompiledTest<T> {};
TYPED_TEST_SUITE(BinaryOperationCompiledTest_TrueDiv, TrueDiv_types);

TYPED_TEST(BinaryOperationCompiledTest_TrueDiv, Vector_Vector)
{
  this->template test<cudf::library::operation::TrueDiv>(cudf::binary_operator::TRUE_DIV);
}
// FLOOR_DIV
//     n      t     d
// n n / n
// t
// d
TYPED_TEST(BinaryOperationCompiledTest_TrueDiv, FloorDiv_Vector_Vector)
{
  this->template test<cudf::library::operation::FloorDiv>(cudf::binary_operator::FLOOR_DIV);
}

// MOD
//     n      t     d
// n n % n
// t
// d d % n	     	d % d
using Mod_types =
  cudf::test::Types<cudf::test::Types<int16_t, u_int64_t, u_int64_t>,
                    cudf::test::Types<double, int8_t, int64_t>,
                    cudf::test::Types<cudf::duration_ms, cudf::duration_s, u_int32_t>,
                    cudf::test::Types<cudf::duration_D, cudf::duration_D, int16_t>,
                    cudf::test::Types<cudf::duration_ns, cudf::duration_D, int16_t>,
                    cudf::test::Types<cudf::duration_ns, cudf::duration_us, cudf::duration_ns>>;
template <typename T>
struct BinaryOperationCompiledTest_Mod : public BinaryOperationCompiledTest<T> {};
TYPED_TEST_SUITE(BinaryOperationCompiledTest_Mod, Mod_types);

TYPED_TEST(BinaryOperationCompiledTest_Mod, Vector_Vector)
{
  this->template test<cudf::library::operation::Mod>(cudf::binary_operator::MOD);
}

// PYMOD
//     n      t     d
// n n % n
// t
// d      	     	d % d
using PyMod_types =
  cudf::test::Types<cudf::test::Types<int16_t, u_int64_t, u_int64_t>,
                    cudf::test::Types<double, int8_t, int64_t>,
                    cudf::test::Types<double, double, double>,
                    cudf::test::Types<cudf::duration_ns, cudf::duration_us, cudf::duration_ns>>;
template <typename T>
struct BinaryOperationCompiledTest_PyMod : public BinaryOperationCompiledTest<T> {};
TYPED_TEST_SUITE(BinaryOperationCompiledTest_PyMod, PyMod_types);
TYPED_TEST(BinaryOperationCompiledTest_PyMod, Vector_Vector)
{
  this->template test<cudf::library::operation::PyMod>(cudf::binary_operator::PYMOD);
}

// POW
//     n      t     d
// n n ^ n
// t
// d

using Pow_types = cudf::test::Types<cudf::test::Types<double, int64_t, int64_t>,
                                    cudf::test::Types<float, float, float>,
                                    cudf::test::Types<int, int32_t, float>,
                                    cudf::test::Types<float, int, int>,
                                    cudf::test::Types<double, int64_t, int32_t>,
                                    cudf::test::Types<double, double, double>,
                                    cudf::test::Types<double, float, double>,
                                    cudf::test::Types<double, int32_t, int64_t>>;

template <typename T>
struct BinaryOperationCompiledTest_FloatOps : public BinaryOperationCompiledTest<T> {};
TYPED_TEST_SUITE(BinaryOperationCompiledTest_FloatOps, Pow_types);

TYPED_TEST(BinaryOperationCompiledTest_FloatOps, Pow_Vector_Vector)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using POW = cudf::library::operation::Pow<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = []() {
    // resulting value can not be represented by the target type => behavior is undefined
    // -2147483648 in host, 2147483647 in device
    if constexpr (std::is_same_v<TypeOut, int>) {
      auto elements =
        cudf::detail::make_counting_transform_iterator(1, [](auto i) { return i % 5; });
      return cudf::test::fixed_width_column_wrapper<TypeLhs>(elements, elements + 100);
    }
    return lhs_random_column<TypeLhs>(100);
  }();
  auto rhs = rhs_random_column<TypeRhs>(100);

  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::POW, cudf::data_type(cudf::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, POW(), NearEqualComparator<TypeOut>{2});
}

// LOG_BASE
//     n      t     d
// n log(n, n)
// t
// d
TYPED_TEST(BinaryOperationCompiledTest_FloatOps, LogBase_Vector_Vector)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using LOG_BASE = cudf::library::operation::LogBase<TypeOut, TypeLhs, TypeRhs>;

  // Make sure there are no zeros
  auto elements = cudf::detail::make_counting_transform_iterator(
    1, [](auto i) { return sizeof(TypeLhs) > 4 ? std::pow(2, i) : i + 30; });
  cudf::test::fixed_width_column_wrapper<TypeLhs> lhs(elements, elements + 50);

  // Find log to the base 7
  auto rhs_elements = cudf::detail::make_counting_transform_iterator(0, [](auto) { return 7; });
  cudf::test::fixed_width_column_wrapper<TypeRhs> rhs(rhs_elements, rhs_elements + 50);

  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::LOG_BASE, cudf::data_type(cudf::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, LOG_BASE());
}

// ATAN2
//     n      t     d
// n ATan2(n, n)
// t
// d
TYPED_TEST(BinaryOperationCompiledTest_FloatOps, ATan2_Vector_Vector)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using ATAN2 = cudf::library::operation::ATan2<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = lhs_random_column<TypeLhs>(col_size);
  auto rhs = rhs_random_column<TypeRhs>(col_size);

  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::ATAN2, cudf::data_type(cudf::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ATAN2(), NearEqualComparator<TypeOut>{2});
}

TYPED_TEST(BinaryOperationCompiledTest_FloatOps, PMod_Vector_Vector)
{
  this->template test<cudf::library::operation::PMod>(cudf::binary_operator::PMOD);
}

using IntPow_types = cudf::test::Types<cudf::test::Types<int32_t, int32_t, int32_t>,
                                       cudf::test::Types<int64_t, int64_t, int64_t>>;
template <typename T>
struct BinaryOperationCompiledTest_IntPow : public BinaryOperationCompiledTest<T> {};
TYPED_TEST_SUITE(BinaryOperationCompiledTest_IntPow, IntPow_types);

TYPED_TEST(BinaryOperationCompiledTest_IntPow, IntPow_SpecialCases)
{
  // This tests special values for which integer powers are required. Casting
  // to double and casting the result back to int results in floating point
  // losses, like 3**1 == 2.
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  auto lhs      = cudf::test::fixed_width_column_wrapper<TypeLhs>({3, -3, 8, -8});
  auto rhs      = cudf::test::fixed_width_column_wrapper<TypeRhs>({1, 1, 7, 7});
  auto expected = cudf::test::fixed_width_column_wrapper<TypeOut>({3, -3, 2097152, -2097152});

  auto result = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::INT_POW, cudf::data_type(cudf::type_to_id<TypeOut>()));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST(BinaryOperationCompiledTestFloorDivInt64, FloorDivInt64Positive)
{
  // This tests special values for which integer floor division is
  // incorrect if round-tripped through casting to double precision.
  // Double precision floating point does not have enough resolution
  // to represent these integers distinctly, so if we were to cast to
  // double, we would get three identical results (all wrong!).
  auto lhs =
    cudf::test::fixed_width_column_wrapper<int64_t>({std::numeric_limits<int64_t>::max(),
                                                     std::numeric_limits<int64_t>::max() - 10,
                                                     std::numeric_limits<int64_t>::max() - 100});
  auto rhs      = cudf::test::fixed_width_column_wrapper<int64_t>({10, 10, 10});
  auto expected = cudf::test::fixed_width_column_wrapper<int64_t>(
    {std::numeric_limits<int64_t>::max() / 10,
     (std::numeric_limits<int64_t>::max() - 10) / 10,
     (std::numeric_limits<int64_t>::max() - 100) / 10});

  auto result = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::FLOOR_DIV, cudf::data_type(cudf::type_to_id<int64_t>()));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST(BinaryOperationCompiledTestFloorDivInt64, FloorDivInt64RoundNegativeInf)
{
  // Floor division should round towards negative infinity. Which is
  // distinct from default integral division in C++ which rounds
  // towards zero (truncation)
  auto lhs =
    cudf::test::fixed_width_column_wrapper<int64_t>({std::numeric_limits<int64_t>::min(),
                                                     std::numeric_limits<int64_t>::min() + 10,
                                                     std::numeric_limits<int64_t>::min() + 100});
  auto rhs = cudf::test::fixed_width_column_wrapper<int64_t>({10, 10, 10});
  // int64_t::min() is not divisible by 10, so there is a non-zero
  // remainder which should be rounded down.
  auto expected = cudf::test::fixed_width_column_wrapper<int64_t>(
    {std::numeric_limits<int64_t>::min() / 10 - 1,
     (std::numeric_limits<int64_t>::min() + 10) / 10 - 1,
     (std::numeric_limits<int64_t>::min() + 100) / 10 - 1});

  auto result = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::FLOOR_DIV, cudf::data_type(cudf::type_to_id<int64_t>()));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

// Bit Operations
//     n      t     d
// n n . n
// t
// d

// i.i, i.u, u.i, u.u -> i
// i.i, i.u, u.i, u.u -> u
using Bit_types = cudf::test::Types<cudf::test::Types<int16_t, int8_t, int16_t>,
                                    cudf::test::Types<int64_t, int32_t, uint16_t>,
                                    cudf::test::Types<int64_t, uint64_t, int64_t>,
                                    cudf::test::Types<int16_t, uint32_t, uint8_t>,
                                    // cudf::test::Types<bool, int8_t, uint8_t>, // valid
                                    cudf::test::Types<uint16_t, int8_t, int16_t>,
                                    cudf::test::Types<uint64_t, int32_t, uint16_t>,
                                    cudf::test::Types<uint64_t, uint64_t, int64_t>,
                                    cudf::test::Types<uint16_t, uint8_t, uint32_t>>;
template <typename T>
struct BinaryOperationCompiledTest_Bit : public BinaryOperationCompiledTest<T> {};
TYPED_TEST_SUITE(BinaryOperationCompiledTest_Bit, Bit_types);

TYPED_TEST(BinaryOperationCompiledTest_Bit, BitwiseAnd_Vector_Vector)
{
  this->template test<cudf::library::operation::BitwiseAnd>(cudf::binary_operator::BITWISE_AND);
}

TYPED_TEST(BinaryOperationCompiledTest_Bit, BitwiseOr_Vector_Vector)
{
  this->template test<cudf::library::operation::BitwiseOr>(cudf::binary_operator::BITWISE_OR);
}

TYPED_TEST(BinaryOperationCompiledTest_Bit, BitwiseXor_Vector_Vector)
{
  this->template test<cudf::library::operation::BitwiseXor>(cudf::binary_operator::BITWISE_XOR);
}

TYPED_TEST(BinaryOperationCompiledTest_Bit, ShiftLeft_Vector_Vector)
{
  this->template test<cudf::library::operation::ShiftLeft>(cudf::binary_operator::SHIFT_LEFT);
}

TYPED_TEST(BinaryOperationCompiledTest_Bit, ShiftRight_Vector_Vector)
{
  this->template test<cudf::library::operation::ShiftRight>(cudf::binary_operator::SHIFT_RIGHT);
}

TYPED_TEST(BinaryOperationCompiledTest_Bit, ShiftRightUnsigned_Vector_Vector)
{
  this->template test<cudf::library::operation::ShiftRightUnsigned>(
    cudf::binary_operator::SHIFT_RIGHT_UNSIGNED);
}

// Logical Operations
//     n      t     d
// n n . n
// t
// d
using Logical_types = cudf::test::Types<cudf::test::Types<bool, int8_t, int16_t>,
                                        cudf::test::Types<bool, int32_t, uint16_t>,
                                        cudf::test::Types<bool, uint64_t, double>,
                                        cudf::test::Types<bool, int8_t, int16_t>,
                                        cudf::test::Types<bool, float, uint16_t>,
                                        cudf::test::Types<bool, uint64_t, int64_t>,
                                        cudf::test::Types<bool, uint8_t, uint32_t>,
                                        cudf::test::Types<bool, uint64_t, int64_t>>;
template <typename T>
struct BinaryOperationCompiledTest_Logical : public BinaryOperationCompiledTest<T> {};
TYPED_TEST_SUITE(BinaryOperationCompiledTest_Logical, Logical_types);

TYPED_TEST(BinaryOperationCompiledTest_Logical, LogicalAnd_Vector_Vector)
{
  this->template test<cudf::library::operation::LogicalAnd>(cudf::binary_operator::LOGICAL_AND);
}

TYPED_TEST(BinaryOperationCompiledTest_Logical, LogicalOr_Vector_Vector)
{
  this->template test<cudf::library::operation::LogicalOr>(cudf::binary_operator::LOGICAL_OR);
}

template <typename T>
using column_wrapper = std::conditional_t<std::is_same_v<T, std::string>,
                                          cudf::test::strings_column_wrapper,
                                          cudf::test::fixed_width_column_wrapper<T>>;

template <typename TypeOut, typename TypeLhs, typename TypeRhs, class OP>
auto NullOp_Result(cudf::column_view lhs, cudf::column_view rhs)
{
  auto [lhs_data, lhs_mask] = cudf::test::to_host<TypeLhs>(lhs);
  auto [rhs_data, rhs_mask] = cudf::test::to_host<TypeRhs>(rhs);
  std::vector<TypeOut> result(lhs.size());
  std::vector<bool> result_mask;
  std::transform(cuda::counting_iterator<cudf::size_type>{0},
                 cuda::counting_iterator{lhs.size()},
                 result.begin(),
                 [&lhs_data    = lhs_data,
                  &lhs_mask    = lhs_mask,
                  &rhs_data    = rhs_data,
                  &rhs_mask    = rhs_mask,
                  &result_mask = result_mask](auto i) -> TypeOut {
                   auto lhs_valid    = lhs_mask.data() and cudf::bit_is_set(lhs_mask.data(), i);
                   auto rhs_valid    = rhs_mask.data() and cudf::bit_is_set(rhs_mask.data(), i);
                   bool output_valid = lhs_valid or rhs_valid;
                   auto result = OP{}(lhs_data[i], rhs_data[i], lhs_valid, rhs_valid, output_valid);
                   result_mask.push_back(output_valid);
                   return result;
                 });
  return column_wrapper<TypeOut>(result.cbegin(), result.cend(), result_mask.cbegin());
}

TYPED_TEST(BinaryOperationCompiledTest_Logical, NullLogicalAnd_Vector_Vector)
{
  using TypeOut  = bool;
  using TypeLhs  = typename TestFixture::TypeLhs;
  using TypeRhs  = typename TestFixture::TypeRhs;
  using NULL_AND = cudf::library::operation::NullLogicalAnd<TypeOut, TypeLhs, TypeRhs>;

  auto lhs            = lhs_random_column<TypeLhs>(col_size);
  auto rhs            = rhs_random_column<TypeRhs>(col_size);
  auto const expected = NullOp_Result<TypeOut, TypeLhs, TypeRhs, NULL_AND>(lhs, rhs);

  auto const result = cudf::binary_operation(lhs,
                                             rhs,
                                             cudf::binary_operator::NULL_LOGICAL_AND,
                                             cudf::data_type(cudf::type_to_id<TypeOut>()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(BinaryOperationCompiledTest_Logical, NullLogicalOr_Vector_Vector)
{
  using TypeOut = bool;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;
  using NULL_OR = cudf::library::operation::NullLogicalOr<TypeOut, TypeLhs, TypeRhs>;

  auto lhs            = lhs_random_column<TypeLhs>(col_size);
  auto rhs            = rhs_random_column<TypeRhs>(col_size);
  auto const expected = NullOp_Result<TypeOut, TypeLhs, TypeRhs, NULL_OR>(lhs, rhs);

  auto const result = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::NULL_LOGICAL_OR, cudf::data_type(cudf::type_to_id<TypeOut>()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

// Comparison Operations ==, !=, <, >, <=, >=
// n<!=>n, t<!=>t, d<!=>d, s<!=>s, dc<!=>dc
using Comparison_types =
  cudf::test::Types<cudf::test::Types<bool, int8_t, int16_t>,
                    cudf::test::Types<bool, uint32_t, uint16_t>,
                    cudf::test::Types<bool, uint64_t, double>,
                    cudf::test::Types<bool, cudf::timestamp_D, cudf::timestamp_s>,
                    cudf::test::Types<bool, cudf::timestamp_ns, cudf::timestamp_us>,
                    cudf::test::Types<bool, cudf::duration_ns, cudf::duration_ns>,
                    cudf::test::Types<bool, cudf::duration_us, cudf::duration_s>,
                    cudf::test::Types<bool, std::string, std::string>,
                    cudf::test::Types<bool, decimal32, decimal32>,
                    cudf::test::Types<bool, decimal64, decimal64>,
                    cudf::test::Types<bool, decimal128, decimal128>>;

template <typename T>
struct BinaryOperationCompiledTest_Comparison : public BinaryOperationCompiledTest<T> {};
TYPED_TEST_SUITE(BinaryOperationCompiledTest_Comparison, Comparison_types);

TYPED_TEST(BinaryOperationCompiledTest_Comparison, Equal_Vector_Vector)
{
  this->template test<cudf::library::operation::Equal>(cudf::binary_operator::EQUAL);
}

TYPED_TEST(BinaryOperationCompiledTest_Comparison, NotEqual_Vector_Vector)
{
  this->template test<cudf::library::operation::NotEqual>(cudf::binary_operator::NOT_EQUAL);
}

TYPED_TEST(BinaryOperationCompiledTest_Comparison, Less_Vector_Vector)
{
  this->template test<cudf::library::operation::Less>(cudf::binary_operator::LESS);
}

TYPED_TEST(BinaryOperationCompiledTest_Comparison, Greater_Vector_Vector)
{
  this->template test<cudf::library::operation::Greater>(cudf::binary_operator::GREATER);
}

TYPED_TEST(BinaryOperationCompiledTest_Comparison, LessEqual_Vector_Vector)
{
  this->template test<cudf::library::operation::LessEqual>(cudf::binary_operator::LESS_EQUAL);
}

TYPED_TEST(BinaryOperationCompiledTest_Comparison, GreaterEqual_Vector_Vector)
{
  this->template test<cudf::library::operation::GreaterEqual>(cudf::binary_operator::GREATER_EQUAL);
}

// Null Operations NullMax, NullMin
// Min(n,n) , Min(t,t), Min(d, d), Min(s, s), Min(dc, dc), Min(n,dc), Min(dc, n)
//    n   t   d  s  dc
// n  .             .
// t      .
// d          .
// s             .
// dc .             .
using Null_types =
  cudf::test::Types<cudf::test::Types<int16_t, int8_t, int16_t>,
                    cudf::test::Types<uint16_t, uint32_t, uint16_t>,
                    cudf::test::Types<double, uint64_t, double>,
                    cudf::test::Types<cudf::timestamp_s, cudf::timestamp_D, cudf::timestamp_s>,
                    cudf::test::Types<cudf::duration_ns, cudf::duration_us, cudf::duration_s>,
                    // cudf::test::Types<std::string, std::string, std::string>, // only fixed-width
                    cudf::test::Types<decimal32, decimal32, decimal32>,
                    cudf::test::Types<decimal64, decimal64, decimal64>,
                    cudf::test::Types<decimal128, decimal128, decimal128>,
                    cudf::test::Types<decimal32, uint32_t, decimal32>,
                    cudf::test::Types<decimal64, uint32_t, decimal64>,
                    cudf::test::Types<decimal128, uint32_t, decimal128>,
                    cudf::test::Types<int64_t, decimal32, decimal32>,
                    cudf::test::Types<int64_t, decimal64, decimal64>,
                    cudf::test::Types<int64_t, decimal128, decimal128>>;

template <typename T>
struct BinaryOperationCompiledTest_NullOps : public BinaryOperationCompiledTest<T> {};
TYPED_TEST_SUITE(BinaryOperationCompiledTest_NullOps, Null_types);

TYPED_TEST(BinaryOperationCompiledTest_NullOps, NullEquals_Vector_Vector)
{
  using TypeOut     = bool;
  using TypeLhs     = typename TestFixture::TypeLhs;
  using TypeRhs     = typename TestFixture::TypeRhs;
  using NULL_EQUALS = cudf::library::operation::NullEquals<TypeOut, TypeLhs, TypeRhs>;

  auto lhs            = lhs_random_column<TypeLhs>(col_size);
  auto rhs            = rhs_random_column<TypeRhs>(col_size);
  auto const expected = NullOp_Result<TypeOut, TypeLhs, TypeRhs, NULL_EQUALS>(lhs, rhs);

  auto const result = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::NULL_EQUALS, cudf::data_type(cudf::type_to_id<TypeOut>()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

using BinaryOperationCompiledTest_NullOpsString =
  BinaryOperationCompiledTest_NullOps<cudf::test::Types<std::string, std::string, std::string>>;
TEST_F(BinaryOperationCompiledTest_NullOpsString, NullEquals_Vector_Vector)
{
  using TypeOut         = bool;
  using TypeLhs         = std::string;
  using TypeRhs         = std::string;
  using NULL_NOT_EQUALS = cudf::library::operation::NullNotEquals<TypeOut, TypeLhs, TypeRhs>;

  auto lhs            = lhs_random_column<TypeLhs>(col_size);
  auto rhs            = rhs_random_column<TypeRhs>(col_size);
  auto const expected = NullOp_Result<TypeOut, TypeLhs, TypeRhs, NULL_NOT_EQUALS>(lhs, rhs);

  auto const result = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::NULL_NOT_EQUALS, cudf::data_type(cudf::type_to_id<TypeOut>()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(BinaryOperationCompiledTest_NullOps, NullNotEquals_Vector_Vector)
{
  using TypeOut         = bool;
  using TypeLhs         = typename TestFixture::TypeLhs;
  using TypeRhs         = typename TestFixture::TypeRhs;
  using NULL_NOT_EQUALS = cudf::library::operation::NullNotEquals<TypeOut, TypeLhs, TypeRhs>;

  auto lhs            = lhs_random_column<TypeLhs>(col_size);
  auto rhs            = rhs_random_column<TypeRhs>(col_size);
  auto const expected = NullOp_Result<TypeOut, TypeLhs, TypeRhs, NULL_NOT_EQUALS>(lhs, rhs);

  auto const result = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::NULL_NOT_EQUALS, cudf::data_type(cudf::type_to_id<TypeOut>()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

using BinaryOperationCompiledTest_NullOpsString =
  BinaryOperationCompiledTest_NullOps<cudf::test::Types<std::string, std::string, std::string>>;
TEST_F(BinaryOperationCompiledTest_NullOpsString, NullNotEquals_Vector_Vector)
{
  using TypeOut     = bool;
  using TypeLhs     = std::string;
  using TypeRhs     = std::string;
  using NULL_EQUALS = cudf::library::operation::NullEquals<TypeOut, TypeLhs, TypeRhs>;

  auto lhs            = lhs_random_column<TypeLhs>(col_size);
  auto rhs            = rhs_random_column<TypeRhs>(col_size);
  auto const expected = NullOp_Result<TypeOut, TypeLhs, TypeRhs, NULL_EQUALS>(lhs, rhs);

  auto const result = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::NULL_EQUALS, cudf::data_type(cudf::type_to_id<TypeOut>()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(BinaryOperationCompiledTest_NullOps, NullMax_Vector_Vector)
{
  using TypeOut  = typename TestFixture::TypeOut;
  using TypeLhs  = typename TestFixture::TypeLhs;
  using TypeRhs  = typename TestFixture::TypeRhs;
  using NULL_MAX = cudf::library::operation::NullMax<TypeOut, TypeLhs, TypeRhs>;

  auto lhs            = lhs_random_column<TypeLhs>(col_size);
  auto rhs            = rhs_random_column<TypeRhs>(col_size);
  auto const expected = NullOp_Result<TypeOut, TypeLhs, TypeRhs, NULL_MAX>(lhs, rhs);

  auto const result = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::NULL_MAX, cudf::data_type(cudf::type_to_id<TypeOut>()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(BinaryOperationCompiledTest_NullOps, NullMin_Vector_Vector)
{
  using TypeOut  = typename TestFixture::TypeOut;
  using TypeLhs  = typename TestFixture::TypeLhs;
  using TypeRhs  = typename TestFixture::TypeRhs;
  using NULL_MIN = cudf::library::operation::NullMin<TypeOut, TypeLhs, TypeRhs>;

  auto lhs            = lhs_random_column<TypeLhs>(col_size);
  auto rhs            = rhs_random_column<TypeRhs>(col_size);
  auto const expected = NullOp_Result<TypeOut, TypeLhs, TypeRhs, NULL_MIN>(lhs, rhs);

  auto const result = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::NULL_MIN, cudf::data_type(cudf::type_to_id<TypeOut>()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(BinaryOperationCompiledTest_NullOpsString, NullMax_Vector_Vector)
{
  using TypeOut  = std::string;
  using TypeLhs  = std::string;
  using TypeRhs  = std::string;
  using NULL_MAX = cudf::library::operation::NullMax<TypeOut, TypeLhs, TypeRhs>;

  auto lhs            = lhs_random_column<TypeLhs>(col_size);
  auto rhs            = rhs_random_column<TypeRhs>(col_size);
  auto const expected = NullOp_Result<TypeOut, TypeLhs, TypeRhs, NULL_MAX>(lhs, rhs);

  auto const result =
    cudf::binary_operation(lhs,
                           rhs,
                           cudf::binary_operator::NULL_MAX,
                           cudf::data_type(cudf::type_to_id<cudf::string_view>()));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
}

TEST_F(BinaryOperationCompiledTest_NullOpsString, NullMin_Vector_Vector)
{
  using TypeOut  = std::string;
  using TypeLhs  = std::string;
  using TypeRhs  = std::string;
  using NULL_MIN = cudf::library::operation::NullMin<TypeOut, TypeLhs, TypeRhs>;

  auto lhs            = lhs_random_column<TypeLhs>(col_size);
  auto rhs            = rhs_random_column<TypeRhs>(col_size);
  auto const expected = NullOp_Result<TypeOut, TypeLhs, TypeRhs, NULL_MIN>(lhs, rhs);

  auto const result =
    cudf::binary_operation(lhs,
                           rhs,
                           cudf::binary_operator::NULL_MIN,
                           cudf::data_type(cudf::type_to_id<cudf::string_view>()));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
}

TEST(BinaryOperationCompiledTest, LargeColumnNoOverflow)
{
  cudf::size_type num_rows{1'799'989'091};
  auto big = cudf::make_column_from_scalar(
    cudf::numeric_scalar<cudf::id_to_type<cudf::type_id::INT8>>{10, true}, num_rows);
  auto small = cudf::make_column_from_scalar(
    cudf::numeric_scalar<cudf::id_to_type<cudf::type_id::INT8>>{1, true}, num_rows);

  auto mask = cudf::binary_operation(big->view(),
                                     small->view(),
                                     cudf::binary_operator::GREATER,
                                     cudf::data_type{cudf::type_id::BOOL8});

  auto agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  auto result =
    cudf::reduce(mask->view(), *agg, cudf::data_type{cudf::type_to_id<cudf::size_type>()});
  auto got = static_cast<cudf::numeric_scalar<cudf::size_type>*>(result.get())->value();
  EXPECT_EQ(num_rows, got);
}

CUDF_TEST_PROGRAM_MAIN()
