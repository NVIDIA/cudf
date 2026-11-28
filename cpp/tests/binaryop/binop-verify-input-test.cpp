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
#include <cudf/detail/binaryop.hpp>
#include <cudf/utilities/traits.hpp>

#include <optional>

struct BinopVerifyInputTest : public cudf::test::BaseFixture {};

TEST_F(BinopVerifyInputTest, CommonTypeSelectionPreservesPruningInvariants)
{
  // Input-conversion and logical-body pruning rely on these host-selection invariants.
  // Include output types because they participate in selection and fallback.
  using cudf::binops::compiled::detail::get_common_type;
  auto const num_types            = static_cast<int>(cudf::type_id::NUM_TYPE_IDS);
  int resolved_fixed_point_inputs = 0;
  int decimal_common_cases        = 0;
  int decimal_output_cases        = 0;
  for (int out = 1; out < num_types; ++out) {
    auto const out_type = cudf::data_type{static_cast<cudf::type_id>(out)};
    for (int lhs = 1; lhs < num_types; ++lhs) {
      auto const lhs_type = cudf::data_type{static_cast<cudf::type_id>(lhs)};
      for (int rhs = 1; rhs < num_types; ++rhs) {
        auto const rhs_type = cudf::data_type{static_cast<cudf::type_id>(rhs)};
        auto const common   = get_common_type(out_type, lhs_type, rhs_type);
        bool const has_fixed_point_input =
          cudf::is_fixed_point(lhs_type) || cudf::is_fixed_point(rhs_type);
        if (common.has_value() && has_fixed_point_input) { ++resolved_fixed_point_inputs; }
        EXPECT_FALSE(common.has_value() && cudf::is_floating_point(*common) &&
                     has_fixed_point_input)
          << "output=" << out << ", lhs=" << lhs << ", rhs=" << rhs;
        // Logical operators require BOOL8 output; connect their decimal common-type
        // specializations to the inputs rejected by host fixed-point validation.
        if (out_type.id() == cudf::type_id::BOOL8 && common.has_value() &&
            cudf::is_fixed_point(*common)) {
          ++decimal_common_cases;
          EXPECT_TRUE(has_fixed_point_input) << "lhs=" << lhs << ", rhs=" << rhs;
        }
        // Without a decimal operand, TRUE_DIV pruning instead relies on the
        // support query rejecting the decimal output that selected this common type.
        if (common.has_value() && cudf::is_fixed_point(*common) && not has_fixed_point_input) {
          ++decimal_output_cases;
          EXPECT_FALSE(cudf::binops::is_supported_operation(
            out_type, lhs_type, rhs_type, cudf::binary_operator::TRUE_DIV))
            << "output=" << out << ", lhs=" << lhs << ", rhs=" << rhs;
        }
      }
    }
  }
  // A broken common-type table must not make the invariant pass vacuously.
  EXPECT_GT(resolved_fixed_point_inputs, 0);
  EXPECT_GT(decimal_common_cases, 0);
  EXPECT_GT(decimal_output_cases, 0);
  // An unrelated output type must not bypass the input-only fallback.
  EXPECT_EQ(get_common_type(cudf::data_type{cudf::type_id::STRING},
                            cudf::data_type{cudf::type_id::DECIMAL32},
                            cudf::data_type{cudf::type_id::DECIMAL32}),
            std::optional{cudf::data_type{cudf::type_id::DECIMAL32}});
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
