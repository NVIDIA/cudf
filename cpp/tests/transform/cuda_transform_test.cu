/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/transform.cuh>

#include <cuda/std/optional>

namespace {

struct CudaTransformTest : public cudf::test::BaseFixture {};

struct add_scalar {
  __device__ cudf::errc operator()(int32_t* output, int32_t input, int32_t scalar) const
  {
    *output = input + scalar;
    return cudf::errc::SUCCESS;
  }
};

TEST_F(CudaTransformTest, FixedWidthWithScalar)
{
  auto input    = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4};
  auto scalar   = cudf::test::fixed_width_column_wrapper<int32_t>{10};
  auto expected = cudf::test::fixed_width_column_wrapper<int32_t>{11, 12, 13, 14};

  cudf::transform_input inputs[]   = {input, cudf::scalar_column_view{scalar}};
  cudf::transform_output outputs[] = {
    {cudf::data_type{cudf::type_id::INT32}, cudf::output_nullability::ALL_VALID}};

  auto result =
    cudf::binary_transform<cudf::null_aware::NO, int32_t, int32_t, int32_t, false, true>(
      add_scalar{}, inputs[0], inputs[1], outputs[0]);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

struct null_aware_square {
  __device__ void operator()(cuda::std::optional<int32_t>* output,
                             cuda::std::optional<int32_t> input) const
  {
    if (input.has_value()) {
      *output = *input * *input;
    } else {
      *output = cuda::std::nullopt;
    }
  }
};

TEST_F(CudaTransformTest, NullAware)
{
  auto input =
    cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2, 3, 4}, {true, false, true, false}};
  auto expected =
    cudf::test::fixed_width_column_wrapper<int32_t>{{1, 0, 9, 0}, {true, false, true, false}};

  cudf::transform_input inputs[]   = {input};
  cudf::transform_output outputs[] = {{cudf::data_type{cudf::type_id::INT32}}};

  auto result = cudf::unary_transform<cudf::null_aware::YES, int32_t, int32_t>(
    null_aware_square{}, inputs[0], outputs[0]);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

struct copy_string_view {
  __device__ void operator()(cudf::string_view* output, cudf::string_view input) const
  {
    *output = input;
  }
};

TEST_F(CudaTransformTest, StringViewOutput)
{
  auto input    = cudf::test::strings_column_wrapper{"one", "two", "three"};
  auto expected = cudf::test::strings_column_wrapper{"one", "two", "three"};

  cudf::transform_input inputs[]   = {input};
  cudf::transform_output outputs[] = {{cudf::data_type{cudf::type_id::STRING}}};

  auto result = cudf::unary_transform<cudf::null_aware::NO, cudf::string_view, cudf::string_view>(
    copy_string_view{}, inputs[0], outputs[0]);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

struct multiply_add {
  __device__ void operator()(int32_t* output, int32_t first, int32_t second, int32_t third) const
  {
    *output = first * second + third;
  }
};

TEST_F(CudaTransformTest, Ternary)
{
  auto first    = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4};
  auto second   = cudf::test::fixed_width_column_wrapper<int32_t>{2, 3, 4, 5};
  auto third    = cudf::test::fixed_width_column_wrapper<int32_t>{10, 20, 30, 40};
  auto expected = cudf::test::fixed_width_column_wrapper<int32_t>{12, 26, 42, 60};

  cudf::transform_input inputs[] = {first, second, third};
  auto output                    = cudf::transform_output{cudf::data_type{cudf::type_id::INT32},
                                       cudf::output_nullability::ALL_VALID};

  auto result = cudf::ternary_transform<cudf::null_aware::NO, int32_t, int32_t, int32_t, int32_t>(
    multiply_add{}, inputs[0], inputs[1], inputs[2], output);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

}  // namespace
