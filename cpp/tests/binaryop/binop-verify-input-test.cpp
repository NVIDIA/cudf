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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/binaryop.hpp>

struct BinopVerifyInputTest : public cudf::test::BaseFixture {};

TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOutputVectorType)
{
  auto lhs = cudf::scalar_type_t<int64_t>(1);
  auto rhs = cudf::test::fixed_width_column_wrapper<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  EXPECT_THROW(
    cudf::binary_operation(
      lhs, rhs, cudf::binary_operator::ADD, cudf::data_type(cudf::type_id::NUM_TYPE_IDS)),
    cudf::logic_error);
}

TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorSecondOperandVectorZeroSize)
{
  auto lhs = cudf::test::fixed_width_column_wrapper<int64_t>{1};
  auto rhs = cudf::test::fixed_width_column_wrapper<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  EXPECT_THROW(cudf::binary_operation(
                 lhs, rhs, cudf::binary_operator::ADD, cudf::data_type(cudf::type_id::INT64)),
               std::invalid_argument);
}

TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorDictionaryLhs)
{
  auto lhs = cudf::test::dictionary_column_wrapper<int64_t>{0, 1, 2, 3, 4};
  auto rhs = cudf::test::fixed_width_column_wrapper<int64_t>{0, 1, 2, 3, 4};

  EXPECT_THROW(cudf::binary_operation(
                 lhs, rhs, cudf::binary_operator::ADD, cudf::data_type(cudf::type_id::INT64)),
               cudf::data_type_error);
}

TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorDictionaryRhs)
{
  auto lhs = cudf::test::fixed_width_column_wrapper<int64_t>{0, 1, 2, 3, 4};
  auto rhs = cudf::test::dictionary_column_wrapper<int64_t>{0, 1, 2, 3, 4};

  EXPECT_THROW(cudf::binary_operation(
                 lhs, rhs, cudf::binary_operator::ADD, cudf::data_type(cudf::type_id::INT64)),
               cudf::data_type_error);
}

TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorDictionaryLhs)
{
  auto lhs = cudf::test::dictionary_column_wrapper<int64_t>{0, 1, 2, 3, 4};
  auto rhs = cudf::scalar_type_t<int64_t>(1);

  EXPECT_THROW(cudf::binary_operation(
                 lhs, rhs, cudf::binary_operator::ADD, cudf::data_type(cudf::type_id::INT64)),
               cudf::data_type_error);
}

TEST_F(BinopVerifyInputTest, Scalar_Vector_ErrorDictionaryRhs)
{
  auto lhs = cudf::scalar_type_t<int64_t>(1);
  auto rhs = cudf::test::dictionary_column_wrapper<int64_t>{0, 1, 2, 3, 4};

  EXPECT_THROW(cudf::binary_operation(
                 lhs, rhs, cudf::binary_operator::ADD, cudf::data_type(cudf::type_id::INT64)),
               cudf::data_type_error);
}

TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorDictionaryCompare)
{
  auto lhs = cudf::test::dictionary_column_wrapper<int64_t>{5, 4, 3, 2, 1};
  auto rhs = cudf::test::dictionary_column_wrapper<int64_t>{5, 4, 3, 2, 1};

  EXPECT_THROW(cudf::binary_operation(
                 lhs, rhs, cudf::binary_operator::EQUAL, cudf::data_type(cudf::type_id::BOOL8)),
               cudf::data_type_error);
}
