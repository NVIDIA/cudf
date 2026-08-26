/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/unary.hpp>

#include <bitset>
#include <numeric>
#include <vector>

using TypesToNegate = cudf::test::Types<int8_t,
                                        int16_t,
                                        int32_t,
                                        int64_t,
                                        float,
                                        double,
                                        cudf::duration_D,
                                        cudf::duration_s,
                                        cudf::duration_ms,
                                        cudf::duration_us,
                                        cudf::duration_ns>;

template <typename T>
struct UnaryNegateTests : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(UnaryNegateTests, TypesToNegate);

TYPED_TEST(UnaryNegateTests, SimpleNEGATE)
{
  using T = TypeParam;
  cudf::test::fixed_width_column_wrapper<T> input{{0, 1, 2, 3}};
  auto const v = cudf::test::make_type_param_vector<T>({0, -1, -2, -3});
  cudf::test::fixed_width_column_wrapper<T> expected(v.begin(), v.end());
  auto output = cudf::unary_operation(input, cudf::unary_operator::NEGATE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

using TypesNotToNegate = cudf::test::Types<uint8_t,
                                           uint16_t,
                                           uint32_t,
                                           uint64_t,
                                           cudf::timestamp_D,
                                           cudf::timestamp_s,
                                           cudf::timestamp_ms,
                                           cudf::timestamp_us,
                                           cudf::timestamp_ns>;

template <typename T>
struct UnaryNegateErrorTests : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(UnaryNegateErrorTests, TypesNotToNegate);

TYPED_TEST(UnaryNegateErrorTests, UnsupportedTypesFail)
{
  using T = TypeParam;
  cudf::test::fixed_width_column_wrapper<T> input({1, 2, 3, 4});
  EXPECT_THROW(cudf::unary_operation(input, cudf::unary_operator::NEGATE), cudf::logic_error);
}

struct UnaryNegateComplexTypesErrorTests : public cudf::test::BaseFixture {};

TEST_F(UnaryNegateComplexTypesErrorTests, NegateStringColumnFail)
{
  cudf::test::strings_column_wrapper input({"foo", "bar"});
  EXPECT_THROW(cudf::unary_operation(input, cudf::unary_operator::NEGATE), cudf::logic_error);
}

TEST_F(UnaryNegateComplexTypesErrorTests, NegateListsColumnFail)
{
  cudf::test::lists_column_wrapper<int32_t> input{{1, 2}, {3, 4}};
  EXPECT_THROW(cudf::unary_operation(input, cudf::unary_operator::NEGATE), cudf::logic_error);
}

struct UnaryBitwiseOpsBoolTest : public cudf::test::BaseFixture {};

template <typename T>
struct UnaryBitwiseOpsTypedTest : public cudf::test::BaseFixture {};
TYPED_TEST_SUITE(UnaryBitwiseOpsTypedTest, cudf::test::IntegralTypesNotBool);

TEST_F(UnaryBitwiseOpsBoolTest, BitCountBool)
{
  using T          = bool;
  auto const data  = std::vector<T>{true, false, true, true, false, true, false, false};
  auto const input = cudf::test::fixed_width_column_wrapper<T>(data.begin(), data.end());

  std::vector<int32_t> expected_data(data.size());
  std::transform(data.begin(), data.end(), expected_data.begin(), [](T val) {
    return static_cast<int32_t>(val);
  });
  auto const expected =
    cudf::test::fixed_width_column_wrapper<int32_t>(expected_data.begin(), expected_data.end());
  auto const output = cudf::unary_operation(input, cudf::unary_operator::BIT_COUNT);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryBitwiseOpsTypedTest, BitCount)
{
  using T         = TypeParam;
  auto const data = [] {
    std::vector<T> data(15);
    std::iota(data.begin(), data.end(), 1);
    return data;
  }();
  auto const input = cudf::test::fixed_width_column_wrapper<T>(data.begin(), data.end());

  std::vector<int32_t> expected_data(data.size());
  std::transform(data.begin(), data.end(), expected_data.begin(), [](T val) {
    using UnsignedT      = std::conditional_t<std::is_same_v<T, bool>, T, std::make_unsigned_t<T>>;
    auto constexpr nbits = CHAR_BIT * sizeof(T);
    auto const b         = std::bitset<nbits>(static_cast<UnsignedT>(val));
    return b.count();
  });
  auto const expected =
    cudf::test::fixed_width_column_wrapper<int32_t>(expected_data.begin(), expected_data.end());
  auto const output = cudf::unary_operation(input, cudf::unary_operator::BIT_COUNT);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryBitwiseOpsTypedTest, BitCountWithNulls)
{
  using T         = TypeParam;
  auto const data = [] {
    std::vector<T> data(15);
    std::iota(data.begin(), data.end(), 1);
    return data;
  }();
  auto const validity = cudf::test::iterators::nulls_at({2, 5, 9, 12});
  auto const input =
    cudf::test::fixed_width_column_wrapper<TypeParam>(data.begin(), data.end(), validity);

  std::vector<int32_t> expected_data(data.size());
  std::transform(data.begin(), data.end(), expected_data.begin(), [](T val) {
    using UnsignedT      = std::conditional_t<std::is_same_v<T, bool>, T, std::make_unsigned_t<T>>;
    auto constexpr nbits = CHAR_BIT * sizeof(T);
    auto const b         = std::bitset<nbits>(static_cast<UnsignedT>(val));
    return b.count();
  });
  auto const expected = cudf::test::fixed_width_column_wrapper<int32_t>(
    expected_data.begin(), expected_data.end(), validity);
  auto const output = cudf::unary_operation(input, cudf::unary_operator::BIT_COUNT);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

template <typename T>
struct UnaryLogicalOpsTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(UnaryLogicalOpsTest, cudf::test::NumericTypes);

TYPED_TEST(UnaryLogicalOpsTest, LogicalNot)
{
  cudf::size_type colSize = 1000;
  std::vector<TypeParam> h_input_v(colSize, false);
  std::vector<bool> h_expect_v(colSize);

  std::transform(
    std::cbegin(h_input_v), std::cend(h_input_v), std::begin(h_expect_v), [](TypeParam e) -> bool {
      return static_cast<bool>(!e);
    });

  cudf::test::fixed_width_column_wrapper<TypeParam> input(std::cbegin(h_input_v),
                                                          std::cend(h_input_v));
  cudf::test::fixed_width_column_wrapper<bool> expected(std::cbegin(h_expect_v),
                                                        std::cend(h_expect_v));

  auto output = cudf::unary_operation(input, cudf::unary_operator::NOT);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryLogicalOpsTest, SimpleLogicalNot)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{true, true, true, true}};
  cudf::test::fixed_width_column_wrapper<bool> expected{{false, false, false, false}};
  auto output = cudf::unary_operation(input, cudf::unary_operator::NOT);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
  auto encoded = cudf::dictionary::encode(input);
  output       = cudf::unary_operation(encoded->view(), cudf::unary_operator::NOT);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryLogicalOpsTest, SimpleLogicalNotWithNullMask)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{true, true, true, true}, {1, 0, 1, 1}};
  cudf::test::fixed_width_column_wrapper<bool> expected{{false, true, false, false},
                                                        {true, false, true, true}};
  auto output = cudf::unary_operation(input, cudf::unary_operator::NOT);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
  auto encoded = cudf::dictionary::encode(input);
  output       = cudf::unary_operation(encoded->view(), cudf::unary_operator::NOT);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryLogicalOpsTest, EmptyLogicalNot)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{};
  cudf::test::fixed_width_column_wrapper<bool> expected{};
  auto output = cudf::unary_operation(input, cudf::unary_operator::NOT);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

template <typename T>
struct UnaryMathOpsTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(UnaryMathOpsTest, cudf::test::NumericTypes);

TYPED_TEST(UnaryMathOpsTest, ABS)
{
  using T = TypeParam;

  cudf::size_type const colSize = 100;
  std::vector<T> h_input_v(colSize);
  std::vector<T> h_expect_v(colSize);

  std::iota(
    std::begin(h_input_v), std::end(h_input_v), std::is_unsigned_v<T> ? colSize : -1 * colSize);

  std::transform(std::cbegin(h_input_v), std::cend(h_input_v), std::begin(h_expect_v), [](auto e) {
    return cudf::util::absolute_value(e);
  });

  cudf::test::fixed_width_column_wrapper<T> const input(std::cbegin(h_input_v),
                                                        std::cend(h_input_v));
  cudf::test::fixed_width_column_wrapper<T> const expected(std::cbegin(h_expect_v),
                                                           std::cend(h_expect_v));

  auto const output = cudf::unary_operation(input, cudf::unary_operator::ABS);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryMathOpsTest, SQRT)
{
  using T = TypeParam;

  cudf::size_type const colSize = 1000;
  std::vector<T> h_input_v(colSize);
  std::vector<T> h_expect_v(colSize);

  std::generate(std::begin(h_input_v), std::end(h_input_v), [i = 0]() mutable {
    ++i;
    return i * i;
  });

  std::transform(std::cbegin(h_input_v), std::cend(h_input_v), std::begin(h_expect_v), [](auto e) {
    return std::sqrt(static_cast<float>(e));
  });

  cudf::test::fixed_width_column_wrapper<T> const input(std::cbegin(h_input_v),
                                                        std::cend(h_input_v));
  cudf::test::fixed_width_column_wrapper<T> const expected(std::cbegin(h_expect_v),
                                                           std::cend(h_expect_v));

  auto const output = cudf::unary_operation(input, cudf::unary_operator::SQRT);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryMathOpsTest, SimpleABS)
{
  auto const v = cudf::test::make_type_param_vector<TypeParam>({-2, -1, 1, 2});
  cudf::test::fixed_width_column_wrapper<TypeParam> input(v.begin(), v.end());
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{{2, 1, 1, 2}};
  auto output = cudf::unary_operation(input, cudf::unary_operator::ABS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryMathOpsTest, SimpleSQRT)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{1, 4, 9, 16}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{{1, 2, 3, 4}};
  auto output = cudf::unary_operation(input, cudf::unary_operator::SQRT);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryMathOpsTest, SimpleCBRT)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{1, 27, 125}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{{1, 3, 5}};
  auto output = cudf::unary_operation(input, cudf::unary_operator::CBRT);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryMathOpsTest, SimpleSQRTWithNullMask)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{1, 4, 9, 16}, {1, 1, 0, 1}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{{1, 2, 9, 4}, {1, 1, 0, 1}};
  auto output = cudf::unary_operation(input, cudf::unary_operator::SQRT);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryMathOpsTest, SimpleCBRTWithNullMask)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{1, 27, 125}, {1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{{1, 3, 125}, {1, 1, 0}};
  auto output = cudf::unary_operation(input, cudf::unary_operator::CBRT);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryMathOpsTest, EmptyABS)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{};
  auto output = cudf::unary_operation(input, cudf::unary_operator::ABS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryMathOpsTest, EmptySQRT)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{};
  auto output = cudf::unary_operation(input, cudf::unary_operator::SQRT);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryMathOpsTest, DictionaryABS)
{
  auto const v = cudf::test::make_type_param_vector<TypeParam>({-2, -1, 1, 2, -1, 2, 0});
  cudf::test::fixed_width_column_wrapper<TypeParam> input_w(v.begin(), v.end());
  auto input = cudf::dictionary::encode(input_w);
  cudf::test::fixed_width_column_wrapper<TypeParam> expected_w{{2, 1, 1, 2, 1, 2, 0}};
  auto expected = cudf::dictionary::encode(expected_w);
  auto output   = cudf::unary_operation(input->view(), cudf::unary_operator::ABS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), output->view());
}

TYPED_TEST(UnaryMathOpsTest, DictionarySQRT)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input_w{{1, 4, 0, 16}, {1, 1, 0, 1}};
  auto input = cudf::dictionary::encode(input_w);
  cudf::test::fixed_width_column_wrapper<TypeParam> expected_w{{1, 2, 0, 4}, {1, 1, 0, 1}};
  auto expected = cudf::dictionary::encode(expected_w);
  auto output   = cudf::unary_operation(input->view(), cudf::unary_operator::SQRT);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), output->view());
}

TYPED_TEST(UnaryMathOpsTest, DictionaryCBRT)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input_w{{1, 27, 0, 125}, {1, 1, 0, 1}};
  auto input = cudf::dictionary::encode(input_w);
  cudf::test::fixed_width_column_wrapper<TypeParam> expected_w{{1, 3, 0, 5}, {1, 1, 0, 1}};
  auto expected = cudf::dictionary::encode(expected_w);
  auto output   = cudf::unary_operation(input->view(), cudf::unary_operator::CBRT);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), output->view());
}

template <typename T>
struct UnaryMathFloatOpsTest : public cudf::test::BaseFixture {};

using floating_point_type_list = ::testing::Types<float, double>;

TYPED_TEST_SUITE(UnaryMathFloatOpsTest, floating_point_type_list);

TYPED_TEST(UnaryMathFloatOpsTest, SimpleSIN)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{0.0}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{{0.0}};
  auto output = cudf::unary_operation(input, cudf::unary_operator::SIN);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryMathFloatOpsTest, SimpleCOS)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{0.0}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{{1.0}};
  auto output = cudf::unary_operation(input, cudf::unary_operator::COS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryMathFloatOpsTest, SimpleSINH)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{0.0}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{{0.0}};
  auto output = cudf::unary_operation(input, cudf::unary_operator::SINH);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryMathFloatOpsTest, SimpleCOSH)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{0.0}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{{1.0}};
  auto output = cudf::unary_operation(input, cudf::unary_operator::COSH);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryMathFloatOpsTest, SimpleTANH)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{0.0}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{{0.0}};
  auto output = cudf::unary_operation(input, cudf::unary_operator::TANH);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryMathFloatOpsTest, SimpleASINH)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{0.0}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{{0.0}};
  auto output = cudf::unary_operation(input, cudf::unary_operator::ARCSINH);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryMathFloatOpsTest, SimpleACOSH)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{1.0}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{{0.0}};
  auto output = cudf::unary_operation(input, cudf::unary_operator::ARCCOSH);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryMathFloatOpsTest, SimpleATANH)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{0.0}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{{0.0}};
  auto output = cudf::unary_operation(input, cudf::unary_operator::ARCTANH);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryMathFloatOpsTest, SimpleFLOOR)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{1.1, 3.3, 5.5, 7.7}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{{1.0, 3.0, 5.0, 7.0}};
  auto output = cudf::unary_operation(input, cudf::unary_operator::FLOOR);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryMathFloatOpsTest, SimpleCEIL)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{1.1, 3.3, 5.5, 7.7}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{{2.0, 4.0, 6.0, 8.0}};
  auto output = cudf::unary_operation(input, cudf::unary_operator::CEIL);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryMathFloatOpsTest, SimpleRINT)
{
  using T = TypeParam;
  cudf::test::fixed_width_column_wrapper<TypeParam> input{
    T(1.5), T(3.5), T(-1.5), T(-3.5), T(0.0), T(NAN)};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{
    T(2.0), T(4.0), T(-2.0), T(-4.0), T(0.0), T(NAN)};
  auto output = cudf::unary_operation(input, cudf::unary_operator::RINT);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TYPED_TEST(UnaryMathFloatOpsTest, SimpleEXP)
{
  using T = TypeParam;
  cudf::test::fixed_width_column_wrapper<T> input{T(1.5), T(3.5), T(-1.5), T(-3.5), T(0.0), T(NAN)};
  cudf::test::fixed_width_column_wrapper<T> expected{T(std::exp(1.5)),
                                                     T(std::exp(3.5)),
                                                     T(std::exp(-1.5)),
                                                     T(std::exp(-3.5)),
                                                     T(std::exp(0.0)),
                                                     T(NAN)};
  auto output = cudf::unary_operation(input, cudf::unary_operator::EXP);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, output->view());
}

TYPED_TEST(UnaryMathFloatOpsTest, SimpleLOG)
{
  using T = TypeParam;
  cudf::test::fixed_width_column_wrapper<T> input{
    T(1.5), T(3.5), T(1.0), T(INFINITY), T(0.0), T(NAN), T(-1.0)};
  cudf::test::fixed_width_column_wrapper<T> expected{
    T(std::log(1.5)), T(std::log(3.5)), T(+0.0), T(INFINITY), T(-INFINITY), T(NAN), T(NAN)};
  auto output = cudf::unary_operation(input, cudf::unary_operator::LOG);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, output->view());
}

TYPED_TEST(UnaryMathFloatOpsTest, DictionaryFLOOR)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input_w{{1.1, 3.3, 5.5, 7.7}};
  auto input = cudf::dictionary::encode(input_w);
  cudf::test::fixed_width_column_wrapper<TypeParam> expected_w{{1.0, 3.0, 5.0, 7.0}};
  auto expected = cudf::dictionary::encode(expected_w);
  auto output   = cudf::unary_operation(input->view(), cudf::unary_operator::FLOOR);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), output->view());
}

TYPED_TEST(UnaryMathFloatOpsTest, DictionaryCEIL)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input_w{{1.1, 3.3, 5.5, 7.7}};
  auto input = cudf::dictionary::encode(input_w);
  cudf::test::fixed_width_column_wrapper<TypeParam> expected_w{{2.0, 4.0, 6.0, 8.0}};
  auto expected = cudf::dictionary::encode(expected_w);
  auto output   = cudf::unary_operation(input->view(), cudf::unary_operator::CEIL);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), output->view());
}

TYPED_TEST(UnaryMathFloatOpsTest, DictionaryEXP)
{
  using T = TypeParam;
  cudf::test::fixed_width_column_wrapper<T> input_w{
    T(1.5), T(3.5), T(-1.5), T(-3.5), T(0.0), T(NAN)};
  auto input    = cudf::dictionary::encode(input_w);
  auto output   = cudf::unary_operation(input->view(), cudf::unary_operator::EXP);
  auto expect   = cudf::unary_operation(input_w, cudf::unary_operator::EXP);
  auto expected = cudf::dictionary::encode(expect->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected->view(), output->view());
}

TYPED_TEST(UnaryMathFloatOpsTest, DictionaryLOG)
{
  using T = TypeParam;
  cudf::test::fixed_width_column_wrapper<T> input_w{
    T(1.5), T(3.5), T(1.0), T(INFINITY), T(0.0), T(NAN), T(-1.0)};
  auto input    = cudf::dictionary::encode(input_w);
  auto output   = cudf::unary_operation(input->view(), cudf::unary_operator::LOG);
  auto expect   = cudf::unary_operation(input_w, cudf::unary_operator::LOG);
  auto expected = cudf::dictionary::encode(expect->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected->view(), output->view());
}

TYPED_TEST(UnaryMathFloatOpsTest, RINTNonFloatingFail)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input{{1, 2, 3, 4, 5}};
  EXPECT_THROW(cudf::unary_operation(input, cudf::unary_operator::RINT), cudf::logic_error);
}

TYPED_TEST(UnaryMathFloatOpsTest, IntegralTypeFail)
{
  auto const test = [](auto const op_type) {
    cudf::test::fixed_width_column_wrapper<TypeParam> input{1.0};
    EXPECT_THROW(cudf::unary_operation(input, op_type), cudf::logic_error);
    auto d = cudf::dictionary::encode(input);
    EXPECT_THROW(cudf::unary_operation(d->view(), op_type), cudf::logic_error);
  };
  test(cudf::unary_operator::BIT_INVERT);
  test(cudf::unary_operator::BIT_COUNT);
}

TYPED_TEST(UnaryMathFloatOpsTest, SimpleCBRT)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{1, 27, 343, 4913}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{{1, 3, 7, 17}};
  auto output = cudf::unary_operation(input, cudf::unary_operator::CBRT);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

struct UnaryMathOpsErrorTest : public cudf::test::BaseFixture {};

TEST_F(UnaryMathOpsErrorTest, ArithmeticTypeFail)
{
  cudf::test::strings_column_wrapper input{"c"};
  EXPECT_THROW(cudf::unary_operation(input, cudf::unary_operator::SQRT), cudf::logic_error);
  auto d = cudf::dictionary::encode(input);
  EXPECT_THROW(cudf::unary_operation(d->view(), cudf::unary_operator::SQRT), cudf::logic_error);
}

TEST_F(UnaryMathOpsErrorTest, LogicalOpTypeFail)
{
  cudf::test::strings_column_wrapper input{"h"};
  EXPECT_THROW(cudf::unary_operation(input, cudf::unary_operator::NOT), cudf::logic_error);
  auto d = cudf::dictionary::encode(input);
  EXPECT_THROW(cudf::unary_operation(d->view(), cudf::unary_operator::NOT), cudf::logic_error);
}
