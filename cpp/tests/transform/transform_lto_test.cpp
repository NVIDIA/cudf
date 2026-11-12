/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/transform/udf.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/transform.hpp>
#include <cudf/udf.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <array>
#include <format>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct TransformLTOTest : public cudf::test::BaseFixture {};

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

template <typename T>
using decimal_wrapper = cudf::test::fixed_point_column_wrapper<typename T::rep>;

namespace {

// Compare the lightweight carriers with the device types seen by NVRTC.
auto abi_type_pairs()
{
  std::vector<std::pair<std::string, std::string>> types;
  for (auto id : cudf::test::all_type_ids) {
    auto logical  = cudf::type_to_name(cudf::data_type{id});
    auto physical = logical;
    if (id == cudf::type_id::BOOL8) { physical = "uint8_t"; }
    if (id == cudf::type_id::FLOAT32) { physical = "uint32_t"; }
    if (id == cudf::type_id::FLOAT64) { physical = "uint64_t"; }
    types.emplace_back(logical, "cudf_udf_abi::" + physical);
  }
  for (auto unit : {"h", "m"}) {
    for (auto kind : {"duration", "timestamp"}) {
      auto logical = std::format("cudf::{}_{}", kind, unit);
      types.emplace_back(logical, "cudf_udf_abi::" + logical);
    }
  }
  types.emplace_back("cudf::string_view", "cudf_udf_abi::cudf::string_view");
  types.emplace_back("cuda::std::span<char>", "cudf_udf_abi::cuda::std::span<char>");
  types.emplace_back("cudf::size_type", "cudf_udf_abi::cudf::size_type");
  types.emplace_back("int32_t", "cudf_udf_abi::int32_t");
  types.emplace_back("cudf::errc", "cudf_udf_abi::cudf::errc");
  types.emplace_back("void*", "void*");
  return types;
}

constexpr char abi_test_headers[] = R"CUDA(
#include <cudf/udf_includes.hpp>
#include <jit/udf_abi.cuh>
#include <cuda/std/cstddef>
#pragma nv_hdrstop
)CUDA";

}  // namespace

TEST_F(TransformLTOTest, UdfAbiLayouts)
{
  std::string source = abi_test_headers;
  for (auto const& [logical, physical] : abi_type_pairs()) {
    source += std::format(R"CUDA(
static_assert(sizeof({0}) == sizeof({1}), "ABI size: {0}");
static_assert(alignof({0}) == alignof({1}), "ABI alignment: {0}");
static_assert(sizeof(cuda::std::optional<{0}>) ==
              sizeof(cudf_udf_abi::cuda::std::optional<{1}>), "optional ABI size: {0}");
static_assert(alignof(cuda::std::optional<{0}>) ==
              alignof(cudf_udf_abi::cuda::std::optional<{1}>), "optional ABI alignment: {0}");
)CUDA",
                          logical,
                          physical);
  }
  EXPECT_NO_THROW(std::ignore = cudf::compile_cuda_to_lto_ir(source.c_str()));
}

TEST_F(TransformLTOTest, UdfAbiValueLayouts)
{
  std::string source = abi_test_headers;
  source += R"CUDA(
template <typename Logical, typename Physical>
__device__ bool optional_layout()
{
  using physical_optional = cudf_udf_abi::cuda::std::optional<Physical>;
  cuda::std::optional<Logical> value{cuda::std::in_place};
  physical_optional representation{};
  static_assert(sizeof(value) == sizeof(representation));
  static_assert(alignof(decltype(value)) == alignof(physical_optional));
  auto const offset = reinterpret_cast<char const*>(&*value) -
                      reinterpret_cast<char const*>(&value);
  cuda::std::memcpy(&representation, &value, sizeof(value));
  if (offset != offsetof(physical_optional, value) || !representation.engaged) { return false; }
  value.reset();
  cuda::std::memcpy(&representation, &value, sizeof(value));
  return !representation.engaged;
}

template <typename Logical, typename Physical>
__device__ bool decimal_layout()
{
  using rep = typename Logical::rep;
  auto const number = (rep{1} << (sizeof(rep) * 8 - 3)) + rep{37};
  Logical value{numeric::scaled_integer<rep>{number, numeric::scale_type{-3}}};
  Physical representation{};
  static_assert(sizeof(value) == sizeof(representation));
  static_assert(alignof(Logical) == alignof(Physical));
  cuda::std::memcpy(&representation, &value, sizeof(value));
  return representation.value == number && representation.scale == -3;
}

extern "C" __device__ cudf::errc cudf_transform_udf(uint8_t* output)
{
  char text[] = "\xc3\xa9!";
  cudf::string_view string{text, 3};
  auto const length = string.length();
  cudf_udf_abi::cudf::string_view string_rep{};
  static_assert(sizeof(string) == sizeof(string_rep));
  cuda::std::memcpy(&string_rep, &string, sizeof(string));
  cuda::std::span<char> span{text, 3};
  cudf_udf_abi::cuda::std::span<char> span_rep{};
  static_assert(sizeof(span) == sizeof(span_rep));
  cuda::std::memcpy(&span_rep, &span, sizeof(span));
  bool matches = string_rep.data == text && string_rep.bytes == 3 &&
                 string_rep.length == length && length == 2 &&
                 span_rep.data == text && span_rep.size == 3;
  matches &= decimal_layout<numeric::decimal32, cudf_udf_abi::numeric::decimal32>();
  matches &= decimal_layout<numeric::decimal64, cudf_udf_abi::numeric::decimal64>();
  matches &= decimal_layout<numeric::decimal128, cudf_udf_abi::numeric::decimal128>();
)CUDA";
  for (auto const& [logical, physical] : abi_type_pairs()) {
    source += std::format(R"CUDA(
  matches &= optional_layout<{}, {}>();
)CUDA",
                          logical,
                          physical);
  }
  source += R"CUDA(
  *output = matches;
  return cudf::errc::SUCCESS;
}
)CUDA";
  auto fragment   = cudf::compile_cuda_to_lto_ir(source.c_str());
  auto const view = fragment.view();
  cudf::transform_output outputs[]{{cudf::data_type{cudf::type_id::UINT8}}};
  auto result = cudf::transform(cudf::transform_udf{"cudf_transform_udf", {&view, 1}},
                                {},
                                outputs,
                                {},
                                cudf::size_type{1},
                                cudf::test::get_default_stream());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), column_wrapper<uint8_t>{{1}});
}

TEST_F(TransformLTOTest, CompileUdfWithIncludes)
{
  constexpr char source[] = R"CUDA(
#include <cudf/udf_includes.hpp>
#include "scale.cuh"
extern "C" __device__ cudf::errc transform(float* output, float input)
{
  *output = scale * input;
  return cudf::errc::SUCCESS;
}
)CUDA";
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
    auto result     = cudf::transform(cudf::transform_udf{"transform", {&view, 1}},
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
  constexpr char source[] = R"CUDA(
#include <cudf/udf_includes.hpp>
extern "C" __device__ cudf::errc transform(cuda::std::optional<int>* output,
                                           cuda::std::optional<int> input)
{
  *output = input;
  return cudf::errc::SUCCESS;
}
)CUDA";
  auto fragment           = cudf::compile_cuda_to_lto_ir(source);
  cudf::test::dictionary_column_wrapper<int32_t> input{{1, 2, 3}, {true, false, true}};
  cudf::transform_input inputs[]{input};
  cudf::transform_output outputs[]{
    {cudf::data_type{cudf::type_id::INT32}, cudf::output_nullability::PRESERVE}};
  auto view   = fragment.view();
  auto result = cudf::transform(cudf::transform_udf{"transform", {&view, 1}, cudf::null_aware::YES},
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

  auto result = cudf::transform(cudf::test::get_transform_udf("invsqrt"),
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

  auto result = cudf::transform(cudf::test::get_transform_udf("distance"),
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

  auto result = cudf::transform(cudf::test::get_transform_udf("to_upper"),
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

  auto result = cudf::transform(cudf::test::get_transform_udf("sum_of_squares"),
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

  cudf::transform_input inputs[] = {a, b};

  auto result      = cudf::transform(cudf::test::get_transform_udf("lehmer_mean"),
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

  EXPECT_THROW(cudf::transform(cudf::test::get_transform_udf("lehmer_mean"),
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

  cudf::transform_input inputs[]   = {input};
  cudf::transform_output outputs[] = {output};

  auto result = cudf::transform(cudf::test::get_transform_udf("bankers_round"),
                                inputs,
                                outputs,
                                {},
                                std::nullopt,
                                cudf::test::get_default_stream());

  decimal_wrapper<T> expected{{124, 126, 126, 128, -124, -126, -126, -128}, numeric::scale_type{0}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), expected);
}

CUDF_TEST_PROGRAM_MAIN()
