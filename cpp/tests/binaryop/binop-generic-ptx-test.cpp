/*
 * SPDX-FileCopyrightText: Copyright 2018-2019 BlazingDB, Inc.
 * SPDX-FileCopyrightText: Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <tests/binaryop/assert-binops.h>
#include <tests/binaryop/binop-fixture.hpp>
#include <tests/transform/udf.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/udf.hpp>
#include <cudf/utilities/error.hpp>

// Exercise the deprecated binary UDF API until its removal.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

struct BinaryOperationGenericPTXTest : public BinaryOperationTest {};

TEST_F(BinaryOperationGenericPTXTest, CAdd_Vector_Vector_FP32_FP32_FP32)
{
  // c = a*a*a + b

  using TypeOut = float;
  using TypeLhs = float;
  using TypeRhs = float;

  auto CADD = [](TypeLhs a, TypeRhs b) { return a * a * a + b; };

  auto lhs = make_random_wrapped_column<TypeLhs>(500);
  auto rhs = make_random_wrapped_column<TypeRhs>(500);

  auto out = cudf::binary_operation(lhs,
                                    rhs,
                                    cudf::test::get_ptx_transform_udf("add_f32"),
                                    cudf::data_type(cudf::type_to_id<TypeOut>()));

  // Allow for floating-point rounding in the cubic expression.
  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, CADD, NearEqualComparator<TypeOut>{2});
}

TEST_F(BinaryOperationGenericPTXTest, CAdd_Vector_Vector_INT64_INT32_INT32)
{
  // c = a*a*a + b

  using TypeOut = int64_t;
  using TypeLhs = int32_t;
  using TypeRhs = int32_t;

  auto CADD = [](TypeLhs a, TypeRhs b) { return a * a * a + b; };

  auto lhs = make_random_wrapped_column<TypeLhs>(500);
  auto rhs = make_random_wrapped_column<TypeRhs>(500);

  auto out = cudf::binary_operation(lhs,
                                    rhs,
                                    cudf::test::get_ptx_transform_udf("add_i32"),
                                    cudf::data_type(cudf::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, CADD);
}

TEST_F(BinaryOperationGenericPTXTest, CAdd_Vector_Vector_INT64_INT32_INT64)
{
  // c = a*a*a + b*b

  using TypeOut = int64_t;
  using TypeLhs = int32_t;
  using TypeRhs = int64_t;

  auto CADD = [](TypeLhs a, TypeRhs b) { return a * a * a + b * b; };

  auto lhs = make_random_wrapped_column<TypeLhs>(500);
  auto rhs = make_random_wrapped_column<TypeRhs>(500);

  auto out = cudf::binary_operation(lhs,
                                    rhs,
                                    cudf::test::get_ptx_transform_udf("add_i32_i64"),
                                    cudf::data_type(cudf::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, CADD);
}

TEST_F(BinaryOperationGenericPTXTest, PropagatesErrorStatus)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input{1, 2, 3};
  EXPECT_THROW(cudf::binary_operation(input,
                                      input,
                                      cudf::test::get_ptx_transform_udf("division_by_zero"),
                                      cudf::data_type{cudf::type_id::INT32}),
               cudf::evaluation_error);
}

#pragma GCC diagnostic pop
