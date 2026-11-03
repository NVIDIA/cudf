/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/file_utilities.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/errc.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>

#include <cudf_test_fragments.hpp>

#include <fstream>
#include <stdexcept>
#include <type_traits>

struct TransformLTOTest : public cudf::test::BaseFixture {};

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

template <typename T>
using decimal_wrapper = cudf::test::fixed_point_column_wrapper<typename T::rep>;

TEST_F(TransformLTOTest, CompiledFragmentOwnershipAndIncludes)
{
  constexpr char source[] = R"(
#include "scale.cuh"
extern "C" __device__ int transform(float* output, float input)
{
  *output = scale * input;
  return 0;
}
)";
  char const* names[]{"scale.cuh"};
  char const* headers[]{"inline constexpr float scale = 3.0f;"};
  auto fragment = cudf::compile_cuda_to_lto_ir(source, false, names, headers, {});
  auto owner    = fragment;
  EXPECT_EQ(owner.view().data(), fragment.view().data());
  fragment = {};
  EXPECT_TRUE(fragment.view().empty());
  ASSERT_FALSE(owner.view().empty());

  column_wrapper<float> input{{1, 2, 4}};
  cudf::transform_input inputs[]{input};
  cudf::transform_output outputs[]{
    {cudf::data_type{cudf::type_id::FLOAT32}, cudf::output_nullability::ALL_VALID}};
  auto apply = [&](cudf::compiled_fragment const& code) {
    return cudf::transform_lto(code.view(),
                               cudf::lto_binary_type::LTO_IR,
                               cudf::null_aware::NO,
                               std::nullopt,
                               inputs,
                               outputs,
                               {},
                               std::nullopt,
                               cudf::test::get_default_stream());
  };
  auto result = apply(owner);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), (column_wrapper<float>{{3, 6, 12}}));

  temp_directory directory{"cudf_lto_headers"};
  {
    std::ofstream file{directory.path() + "scale.cuh"};
    ASSERT_TRUE(file.good());
    file << "inline constexpr float scale = 5.0f;";
  }
  char const* paths[]{directory.path().c_str()};
  auto filesystem_code = cudf::compile_cuda_to_lto_ir(source, false, {}, {}, paths);
  result               = apply(filesystem_code);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), (column_wrapper<float>{{5, 10, 20}}));
}

TEST_F(TransformLTOTest, CompiledFragmentFailuresAndErrorABI)
{
  static_assert(std::is_same_v<std::underlying_type_t<cudf::errc>, cudf::error_int>);
  static_assert(sizeof(cudf::errc) == sizeof(int));
  EXPECT_THROW(static_cast<void>(cudf::compile_cuda_to_lto_ir(nullptr, false, {}, {}, {})),
               std::invalid_argument)
    << "A null CUDA source pointer is invalid";
  char const* missing[]{"missing.cuh"};
  EXPECT_THROW(static_cast<void>(cudf::compile_cuda_to_lto_ir("", false, missing, {}, {})),
               std::invalid_argument)
    << "Each in-memory header name requires matching contents";

  constexpr char source[] = R"(
#include <cudf/errc.hpp>
extern "C" __device__ cudf::errc transform(float* output, float input)
{
  *output = input;
  return cudf::errc::DIVISION_BY_ZERO;
}
)";
  auto fragment           = cudf::compile_cuda_to_lto_ir(source, true, {}, {}, {});
  column_wrapper<float> input{{1, 2}};
  cudf::transform_input inputs[]{input};
  cudf::transform_output outputs[]{
    {cudf::data_type{cudf::type_id::FLOAT32}, cudf::output_nullability::ALL_VALID}};
  try {
    cudf::transform_lto(fragment.view(),
                        cudf::lto_binary_type::LTO_IR,
                        cudf::null_aware::NO,
                        std::nullopt,
                        inputs,
                        outputs,
                        {},
                        std::nullopt,
                        cudf::test::get_default_stream());
    FAIL() << "The device callback must propagate DIVISION_BY_ZERO";
  } catch (cudf::evaluation_error const& error) {
    EXPECT_EQ(error.error_code(), cudf::errc::DIVISION_BY_ZERO);
  }
}

TEST_F(TransformLTOTest, NullableDictionary)
{
  constexpr char source[] = R"(
#include <cuda/std/optional>
extern "C" __device__ int transform(cuda::std::optional<int>* output,
                                   cuda::std::optional<int> input)
{
  *output = input;
  return 0;
}
)";
  auto fragment           = cudf::compile_cuda_to_lto_ir(source, true, {}, {}, {});
  cudf::test::dictionary_column_wrapper<int32_t> input{{1, 2, 3}, {true, false, true}};
  cudf::transform_input inputs[]{input};
  cudf::transform_output outputs[]{
    {cudf::data_type{cudf::type_id::INT32}, cudf::output_nullability::PRESERVE}};
  auto result = cudf::transform_lto(fragment.view(),
                                    cudf::lto_binary_type::LTO_IR,
                                    cudf::null_aware::YES,
                                    std::nullopt,
                                    inputs,
                                    outputs,
                                    {},
                                    std::nullopt,
                                    cudf::test::get_default_stream());
  column_wrapper<int32_t> expected{{1, 2, 3}, {true, false, true}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), expected);
}

TEST_F(TransformLTOTest, InvSqrt)
{
  column_wrapper<float> input{{1.0f, 4.0f, 9.0f, 16.0f}};

  cudf::transform_input inputs[]   = {input};
  cudf::transform_output outputs[] = {
    {cudf::data_type{cudf::type_id::FLOAT32}, cudf::output_nullability::ALL_VALID}};

  auto const range = cudf_test_fragments::file_ranges[cudf_test_fragments::invsqrt];
  std::span<uint8_t const> udf{cudf_test_fragments::files.subspan(range[0], range[1])};

  auto result = cudf::transform_lto(udf,
                                    cudf::lto_binary_type::FATBIN,
                                    cudf::null_aware::NO,
                                    std::nullopt,
                                    inputs,
                                    outputs,
                                    {},
                                    std::nullopt,
                                    cudf::test::get_default_stream());

  column_wrapper<float> expected{{1.0f, 0.5f, 0.33333334f, 0.25f}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), expected);
}

TEST_F(TransformLTOTest, Distance)
{
  column_wrapper<float> x1{{0, 0, 0, 0}};
  column_wrapper<float> y1{{0.0F, 1.2F, 2.5F, 3.7F}};
  column_wrapper<float> x2{{1.6F, 2.1F, 3.2f, 4.5f}};
  column_wrapper<float> y2{{0, 0, 0, 0}};

  cudf::transform_input inputs[]   = {x1, y1, x2, y2};
  cudf::transform_output outputs[] = {
    {cudf::data_type{cudf::type_id::FLOAT32}, cudf::output_nullability::ALL_VALID}};

  auto const range = cudf_test_fragments::file_ranges[cudf_test_fragments::distance];
  std::span<uint8_t const> udf{cudf_test_fragments::files.subspan(range[0], range[1])};

  auto result = cudf::transform_lto(udf,
                                    cudf::lto_binary_type::FATBIN,
                                    cudf::null_aware::NO,
                                    std::nullopt,
                                    inputs,
                                    outputs,
                                    {},
                                    std::nullopt,
                                    cudf::test::get_default_stream());

  auto distance = [](float x1, float y1, float x2, float y2) {
    return std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
  };

  column_wrapper<float> expected{{
    distance(0, 0, 1.6F, 0),
    distance(0, 1.2F, 2.1F, 0),
    distance(0, 2.5F, 3.2F, 0),
    distance(0, 3.7F, 4.5F, 0),
  }};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), expected);
}

TEST_F(TransformLTOTest, ToUpper)
{
  column_wrapper<uint8_t> input{{65, 66, 97, 98, 48, 49, 32, 33, 127, 255}};

  cudf::transform_input inputs[]   = {input};
  cudf::transform_output outputs[] = {
    {cudf::data_type{cudf::type_id::UINT8}, cudf::output_nullability::ALL_VALID}};

  auto const range = cudf_test_fragments::file_ranges[cudf_test_fragments::to_upper];
  std::span<uint8_t const> udf{cudf_test_fragments::files.subspan(range[0], range[1])};

  auto result = cudf::transform_lto(udf,
                                    cudf::lto_binary_type::FATBIN,
                                    cudf::null_aware::NO,
                                    std::nullopt,
                                    inputs,
                                    outputs,
                                    {},
                                    std::nullopt,
                                    cudf::test::get_default_stream());

  column_wrapper<uint8_t> expected{{65, 66, 65, 66, 48, 49, 32, 33, 127, 255}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), expected);
}

TEST_F(TransformLTOTest, SumOfSquares)
{
  column_wrapper<float> lhs{{1.0f, 4.0f, 9.0f, 16.0f}};
  column_wrapper<float> rhs{{1.0f, 2.0f, 2.0f, 10.0f}};

  cudf::transform_input inputs[]   = {lhs, rhs};
  cudf::transform_output outputs[] = {
    {cudf::data_type{cudf::type_id::FLOAT32}, cudf::output_nullability::ALL_VALID}};

  auto const range = cudf_test_fragments::file_ranges[cudf_test_fragments::sum_of_squares];
  std::span<uint8_t const> udf{cudf_test_fragments::files.subspan(range[0], range[1])};

  auto result = cudf::transform_lto(udf,
                                    cudf::lto_binary_type::FATBIN,
                                    cudf::null_aware::NO,
                                    std::nullopt,
                                    inputs,
                                    outputs,
                                    {},
                                    std::nullopt,
                                    cudf::test::get_default_stream());

  column_wrapper<float> expected{{2.0f, 20.0f, 85.0f, 356.0f}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), expected);
}

TEST_F(TransformLTOTest, FallibleIntegerLehmerMean)
{
  // computes integer lehmer mean: `(a^2 + b^2) / (a + b)` for each row using checked arithmetic and
  // throws if an overflow occurs

  column_wrapper<int32_t> a{{2, 3, 4, 6, 8}};
  column_wrapper<int32_t> b{{1, 1, 2, 3, 4}};
  column_wrapper<int32_t> a_fail{{2, 3, -2, 6, 8}};

  cudf::transform_output outputs[] = {
    {cudf::data_type{cudf::type_id::INT32}, cudf::output_nullability::ALL_VALID}};

  auto const range = cudf_test_fragments::file_ranges[cudf_test_fragments::lehmer_mean];
  std::span<uint8_t const> udf{cudf_test_fragments::files.subspan(range[0], range[1])};

  cudf::transform_input inputs[] = {a, b};

  auto result      = cudf::transform_lto(udf,
                                    cudf::lto_binary_type::FATBIN,
                                    cudf::null_aware::NO,
                                    std::nullopt,
                                    inputs,
                                    outputs,
                                         {},
                                    std::nullopt,
                                    cudf::test::get_default_stream());
  auto lehmer_mean = [](int32_t a, int32_t b) { return (a * a + b * b) / (a + b); };

  column_wrapper<int32_t> expected{{lehmer_mean(2, 1),
                                    lehmer_mean(3, 1),
                                    lehmer_mean(4, 2),
                                    lehmer_mean(6, 3),
                                    lehmer_mean(8, 4)}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), expected);

  cudf::transform_input inputs_fail[] = {a_fail, b};

  EXPECT_THROW(cudf::transform_lto(udf,
                                   cudf::lto_binary_type::FATBIN,
                                   cudf::null_aware::NO,
                                   std::nullopt,
                                   inputs_fail,
                                   outputs,
                                   {},
                                   std::nullopt,
                                   cudf::test::get_default_stream()),
               cudf::evaluation_error);
}

TEST_F(TransformLTOTest, BankersRounding)
{
  using T = numeric::decimal128;

  decimal_wrapper<T> input{{12450, 12550, 12650, 12750, -12450, -12550, -12650, -12750},
                           numeric::scale_type{-2}};

  cudf::transform_output output{cudf::data_type{cudf::type_to_id<T>(), numeric::scale_type{0}},
                                cudf::output_nullability::ALL_VALID};

  auto const range = cudf_test_fragments::file_ranges[cudf_test_fragments::bankers_rounding];
  std::span<uint8_t const> udf{cudf_test_fragments::files.subspan(range[0], range[1])};

  cudf::transform_input inputs[]   = {input};
  cudf::transform_output outputs[] = {output};

  auto result = cudf::transform_lto(udf,
                                    cudf::lto_binary_type::FATBIN,
                                    cudf::null_aware::NO,
                                    std::nullopt,
                                    inputs,
                                    outputs,
                                    {},
                                    std::nullopt,
                                    cudf::test::get_default_stream());

  decimal_wrapper<T> expected{{124, 126, 126, 128, -124, -126, -126, -128}, numeric::scale_type{0}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), expected);
}
