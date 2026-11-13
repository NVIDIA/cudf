/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/transform.hpp>

#include <cudf_test_fragments.hpp>

#include <format>
#include <string>

struct TransformLTOTest : public cudf::test::BaseFixture {};

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

template <typename T>
using decimal_wrapper = cudf::test::fixed_point_column_wrapper<typename T::rep>;

TEST_F(TransformLTOTest, UdfAbiLayouts)
{
  // Canonical physical types and the additional values used in callback signatures.
  constexpr char const* types[]{"uint8_t",
                                "uint16_t",
                                "uint32_t",
                                "uint64_t",
                                "numeric::decimal32",
                                "numeric::decimal64",
                                "numeric::decimal128",
                                "cudf::string_view",
                                "cuda::std::span<char>",
                                "cudf::size_type",
                                "cudf::errc",
                                "cudf::voidptr_t"};
  std::string source = R"CUDA(
#include <cudf/errc.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/user_data.hpp>
#include <cuda/std/optional>
#include <cuda/std/span>
#include <jit/udf_abi.cuh>
#pragma nv_hdrstop
)CUDA";
  for (auto const* type : types) {
    source += std::format(R"CUDA(
static_assert(sizeof({0}) == sizeof({1}), "ABI size: {0}");
static_assert(alignof({0}) == alignof({1}), "ABI alignment: {0}");
static_assert(sizeof(cuda::std::optional<{0}>) ==
              sizeof(cudf_udf_abi::cuda::std::optional<{1}>), "optional ABI size: {0}");
static_assert(alignof(cuda::std::optional<{0}>) ==
              alignof(cudf_udf_abi::cuda::std::optional<{1}>), "optional ABI alignment: {0}");
)CUDA",
                          type,
                          std::format("cudf_udf_abi::{}", type));
  }
  EXPECT_NO_THROW(static_cast<void>(cudf::compile_cuda_to_lto_ir(source.c_str(), {}, {}, {})));

  // optional privately inherits its storage. Expose that storage only in this compilation.
  
  constexpr char optional_offsets[] = R"CUDA(
#include <cuda/std/cstddef>
#define private public
#include <cuda/std/optional>
#undef private
#include <jit/udf_abi.cuh>
using uint128_t = unsigned __int128;
using actual_optional = cuda::std::optional<uint128_t>;
using forwarding_optional = cudf_udf_abi::cuda::std::optional<uint128_t>;
static_assert(offsetof(actual_optional, __storage_.__val_) == 0);
static_assert(offsetof(actual_optional, __engaged_) == sizeof(uint128_t));
static_assert(offsetof(actual_optional, __storage_.__val_) ==
              offsetof(forwarding_optional, value));
static_assert(offsetof(actual_optional, __engaged_) ==
              offsetof(forwarding_optional, engaged));
)CUDA";
  EXPECT_NO_THROW(static_cast<void>(
    cudf::compile_cuda_to_lto_ir(optional_offsets, {}, {}, {}, false)));
}

TEST_F(TransformLTOTest, CompileUdfWithIncludes)
{
  constexpr char source[] = R"(
#include "scale.cuh"
extern "C" __device__ int transform(float* output, float input)
{
  *output = scale * input;
  return 0;
}
)";
  column_wrapper<float> input{{1, 2, 4}};
  cudf::transform_input inputs[]{input};
  cudf::transform_output outputs[]{
    {cudf::data_type{cudf::type_id::FLOAT32}, cudf::output_nullability::ALL_VALID}};
  struct test_case {
    char const* header;
    float factor;
  };
  test_case const cases[]{{"inline constexpr float scale = 3.0f;", 3},
                          {"inline constexpr float scale = 3.0f;", 3},
                          {"inline constexpr float scale = 5.0f;", 5}};
  for (auto const& test : cases) {
    SCOPED_TRACE(test.header);
    char const* names[]{"scale.cuh"};
    char const* headers[]{test.header};
    // Header contents change under the same include name, so disable PCH reuse.
    auto fragment   = cudf::compile_cuda_to_lto_ir(source, names, headers, {}, false);
    auto const view = fragment.view();
    auto result     = cudf::transform_lto(view.data,
                                      view.type,
                                      cudf::null_aware::NO,
                                      std::nullopt,
                                      inputs,
                                      outputs,
                                          {},
                                      std::nullopt,
                                      cudf::test::get_default_stream());
    column_wrapper<float> expected{{test.factor, 2 * test.factor, 4 * test.factor}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), expected);
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
  auto fragment           = cudf::compile_cuda_to_lto_ir(source, {}, {}, {});
  cudf::test::dictionary_column_wrapper<int32_t> input{{1, 2, 3}, {true, false, true}};
  cudf::transform_input inputs[]{input};
  cudf::transform_output outputs[]{
    {cudf::data_type{cudf::type_id::INT32}, cudf::output_nullability::PRESERVE}};
  auto view   = fragment.view();
  auto result = cudf::transform_lto(view.data,
                                    view.type,
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

CUDF_TEST_PROGRAM_MAIN()
