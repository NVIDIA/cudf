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
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <array>
#include <utility>

namespace {
template <std::size_t Lhs, std::size_t Rhs>
constexpr cudf::type_id common_type_id()
{
  if constexpr (Lhs != 0 && Rhs != 0) {
    using L = cudf::id_to_type<static_cast<cudf::type_id>(Lhs)>;
    using R = cudf::id_to_type<static_cast<cudf::type_id>(Rhs)>;
    if constexpr (cudf::binary_op_has_common_type_v<L, R>) {
      return cudf::type_to_id<cudf::binary_op_common_type_t<L, R>>();
    }
  }
  return cudf::type_id::EMPTY;
}

template <std::size_t... I>
constexpr auto common_type_matrix(std::index_sequence<I...>)
{
  constexpr auto n = static_cast<std::size_t>(cudf::type_id::NUM_TYPE_IDS);
  return std::array{common_type_id<I / n, I % n>()...};
}
}  // namespace

struct BinopVerifyInputTest : public cudf::test::BaseFixture {};

TEST_F(BinopVerifyInputTest, FixedPointInputsNeverSelectFloatingCommonType)
{
  // The device accessor omits fixed-to-floating conversions based on this host-side
  // invariant. Include output types because they participate in selection and fallback.
  auto const num_types  = static_cast<int>(cudf::type_id::NUM_TYPE_IDS);
  constexpr auto matrix = common_type_matrix(std::make_index_sequence<num_types * num_types>{});
  int resolved_fixed_point_inputs = 0;
  int fallback_fixed_point_inputs = 0;
  for (int out = 1; out < num_types; ++out) {
    for (int lhs = 1; lhs < num_types; ++lhs) {
      for (int rhs = 1; rhs < num_types; ++rhs) {
        auto const lhs_type = cudf::data_type{static_cast<cudf::type_id>(lhs)};
        auto const rhs_type = cudf::data_type{static_cast<cudf::type_id>(rhs)};
        // Match get_common_type's output-first resolution and input-only fallback.
        auto common = matrix[out * num_types + lhs];
        if (common != cudf::type_id::EMPTY) {
          common = matrix[static_cast<int>(common) * num_types + rhs];
        }
        bool const used_fallback = common == cudf::type_id::EMPTY;
        if (used_fallback) { common = matrix[lhs * num_types + rhs]; }
        if (common != cudf::type_id::EMPTY &&
            (cudf::is_fixed_point(lhs_type) || cudf::is_fixed_point(rhs_type))) {
          ++resolved_fixed_point_inputs;
          fallback_fixed_point_inputs += used_fallback;
        }
        EXPECT_FALSE(common != cudf::type_id::EMPTY &&
                     cudf::is_floating_point(cudf::data_type{common}) &&
                     (cudf::is_fixed_point(lhs_type) || cudf::is_fixed_point(rhs_type)))
          << "output=" << out << ", lhs=" << lhs << ", rhs=" << rhs;
      }
    }
  }
  // A broken common-type table must not make the invariant pass vacuously.
  EXPECT_GT(resolved_fixed_point_inputs, 0);
  EXPECT_GT(fallback_fixed_point_inputs, 0);
}

TEST_F(BinopVerifyInputTest, BooleanOutputSelectsDecimalCommonTypeOnlyForDecimalInputs)
{
  // Logical operators require BOOL8 output. This links their decimal common-type
  // specializations to the input types checked by host fixed-point validation.
  constexpr auto n         = static_cast<int>(cudf::type_id::NUM_TYPE_IDS);
  constexpr auto matrix    = common_type_matrix(std::make_index_sequence<n * n>{});
  int decimal_common_cases = 0;
  for (int lhs = 1; lhs < n; ++lhs) {
    for (int rhs = 1; rhs < n; ++rhs) {
      auto common = matrix[static_cast<int>(cudf::type_id::BOOL8) * n + lhs];
      if (common != cudf::type_id::EMPTY) { common = matrix[static_cast<int>(common) * n + rhs]; }
      if (common == cudf::type_id::EMPTY) { common = matrix[lhs * n + rhs]; }
      if (common != cudf::type_id::EMPTY && cudf::is_fixed_point(cudf::data_type{common})) {
        ++decimal_common_cases;
        EXPECT_TRUE(cudf::is_fixed_point(cudf::data_type{static_cast<cudf::type_id>(lhs)}) ||
                    cudf::is_fixed_point(cudf::data_type{static_cast<cudf::type_id>(rhs)}))
          << "lhs=" << lhs << ", rhs=" << rhs;
      }
    }
  }
  EXPECT_GT(decimal_common_cases, 0);
}

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
