/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/operators/arithmetic.cuh>
#include <cudf/detail/operators/checked_arithmetic.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/std/cmath>
#include <thrust/tabulate.h>

#include <limits>

template <typename T>
struct ArithmeticOperatorTest : cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ArithmeticOperatorTest, cudf::test::FloatingPointTypes);

template <typename T>
void test_separate_rounding()
{
  auto const stream = cudf::get_default_stream();
  auto const eps    = std::numeric_limits<T>::epsilon();
  auto input        = cudf::test::fixed_width_column_wrapper<T>{1 + eps, 1 - eps, T{-1}};
  auto const data   = static_cast<cudf::column_view>(input).data<T>();
  auto result       = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<T>()}, 12);

  thrust::tabulate(rmm::exec_policy_nosync(stream),
                   result->mutable_view().begin<T>(),
                   result->mutable_view().end<T>(),
                   [data] __device__(cudf::size_type i) -> T {
                     auto const a = data[0];
                     auto const b = data[1];
                     auto const c = data[2];
                     // Protect each operator independently, even when its neighbor permits
                     // contraction.
                     switch (i) {
                       case 0: return cudf::detail::ops::add(a * b, c);
                       case 1: return cudf::detail::ops::sub(a * b, -c);
                       case 2: return cudf::detail::ops::mul(a, b) + c;
                       case 3: return cudf::detail::ops::add_overflow(a * b, c).value();
                       case 4: return cudf::detail::ops::sub_overflow(a * b, -c).value();
                       case 5: return cudf::detail::ops::mul_overflow(a, b).value() + c;
                       case 6: return cudf::detail::ops::sub(-c, a * b);
                       case 7: return -c - cudf::detail::ops::mul(a, b);
                       case 8: return cudf::detail::ops::sub_overflow(-c, a * b).value();
                       case 9: return -c - cudf::detail::ops::mul_overflow(a, b).value();
                       case 10: return cuda::std::fma(a, b, c);
                       default: return cuda::std::fma(-a, b, -c);
                     }
                   });

  // Separate rounding produces zero; explicit FMA retains the product's low bits.
  auto expected = cudf::test::fixed_width_column_wrapper<T>{
    T{0}, T{0}, T{0}, T{0}, T{0}, T{0}, T{0}, T{0}, T{0}, T{0}, -eps * eps, eps * eps};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(ArithmeticOperatorTest, SeparateRounding) { test_separate_rounding<TypeParam>(); }
