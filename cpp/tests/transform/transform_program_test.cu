/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>

#include <array>
#include <numeric>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace transform_program_test {

template <typename Input, typename Output = Input, bool Scalar = false, bool Nullable = false>
struct identity_operator {
  using Inputs                    = cudf::type_list<cudf::transform_input_type<Input, Scalar>>;
  using Outputs                   = cudf::type_list<Output>;
  static constexpr bool NullAware = Nullable;
  static constexpr bool UserData  = false;
  struct Function {
    __device__ void operator()(auto* out, auto in) const { *out = in; }
  };
};

template <typename Info, std::size_t N>
auto program_views(std::array<Info, N> const& owners)
{
  std::array<typename Info::view_type, N> views{};
  for (std::size_t i = 0; i < N; ++i) {
    views[i] = owners[i].view();
  }
  return views;
}

std::unique_ptr<cudf::table> run(std::span<cudf::transform_input const> inputs,
                                 std::span<cudf::transform_output const> outputs,
                                 std::span<cudf::transform_program_info::view_type const> kernels,
                                 cudf::null_aware nullable           = cudf::null_aware::NO,
                                 std::optional<void*> user_data      = std::nullopt,
                                 std::optional<cudf::size_type> rows = std::nullopt,
                                 std::vector<std::unique_ptr<cudf::column>> offsets = {},
                                 cuda::stream_ref stream = cudf::test::get_default_stream())
{
  return cudf::transform_precompiled(
    kernels, nullable, user_data, inputs, outputs, std::move(offsets), rows, {}, stream);
}

std::unique_ptr<cudf::table> run_jit(std::span<cudf::transform_input const> inputs,
                                     std::span<cudf::transform_output const> outputs,
                                     std::string const& source,
                                     cudf::null_aware nullable = cudf::null_aware::NO)
{
  return cudf::transform(
    source, cudf::udf_source_type::CUDA, nullable, std::nullopt, inputs, outputs, {}, std::nullopt);
}

struct TransformProgramTest : cudf::test::BaseFixture {};

template <typename T>
struct TransformProgramTypesTest : cudf::test::BaseFixture {};
TYPED_TEST_SUITE(TransformProgramTypesTest, cudf::test::FixedWidthTypes);

TYPED_TEST(TransformProgramTypesTest, IdentityParity)
{
  using T    = TypeParam;
  auto input = [] {
    if constexpr (cudf::is_fixed_point<T>()) {
      return cudf::test::fixed_point_column_wrapper<typename T::rep>{
        {1, 2, 3, 4}, {true, false, true, true}, numeric::scale_type{-2}}
        .release();
    } else {
      return cudf::test::fixed_width_column_wrapper<T, int32_t>{{1, 2, 3, 4},
                                                                {true, false, true, true}}
        .release();
    }
  }();
  std::array<cudf::transform_input, 1> inputs{input->view()};
  std::array outputs{cudf::transform_output{input->type()}};
  std::array plain_owners{cudf::transform_program_info::make<identity_operator<T>>()};
  auto plain = program_views(plain_owners);
  std::array nullable_owners{
    cudf::transform_program_info::make<identity_operator<T, T, false, true>>()};
  auto nullable = program_views(nullable_owners);
  auto type     = cudf::type_to_name(input->type());
  auto source   = "__device__ void identity(" + type + "* out, " + type + " in) { *out = in; }";
  auto nullable_source = "__device__ void identity(cuda::std::optional<" + type +
                         ">* out, cuda::std::optional<" + type + "> in) { *out = in; }";
  auto expected = run_jit(inputs, outputs, source);
  auto result   = run(inputs, outputs, plain);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());
  expected = run_jit(inputs, outputs, nullable_source, cudf::null_aware::YES);
  result   = run(inputs, outputs, nullable, cudf::null_aware::YES);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());
}

static_assert(!std::is_copy_constructible_v<cudf::transform_program_info>);
static_assert(!std::is_copy_assignable_v<cudf::transform_program_info>);
static_assert(std::is_nothrow_move_constructible_v<cudf::transform_program_info>);
static_assert(std::is_nothrow_move_assignable_v<cudf::transform_program_info>);

template <typename T>
constexpr bool has_definition = requires { sizeof(T); };
static_assert(!has_definition<cudf::transform_program_info::impl>);

TEST_F(TransformProgramTest, MoveOnlyOwnerAndBorrowedViews)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input{1, 2, 3};
  std::array<cudf::transform_input, 1> inputs{input};
  std::array outputs{cudf::transform_output{cudf::data_type{cudf::type_id::INT32}}};
  cudf::transform_program_info empty;
  EXPECT_EQ(empty.view(), nullptr);
  auto owner    = cudf::transform_program_info::make<identity_operator<int32_t>>();
  auto borrowed = owner.view();
  ASSERT_NE(borrowed, nullptr);
  auto moved = std::move(owner);
  EXPECT_EQ(owner.view(), nullptr);
  EXPECT_EQ(moved.view(), borrowed);
  empty = std::move(moved);
  EXPECT_EQ(moved.view(), nullptr);
  EXPECT_EQ(empty.view(), borrowed);

  // Null views are ignored when a valid entry is available.
  std::array<cudf::transform_program_info::view_type, 3> views{nullptr, borrowed, nullptr};
  auto result = run(inputs, outputs, views);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, result->get_column(0));
  std::array null_views{owner.view(), moved.view()};
  EXPECT_THROW(run(inputs, outputs, null_views), std::invalid_argument);
}

TEST_F(TransformProgramTest, HeterogeneousListAndMissingKernel)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input{1, 2, 3};
  std::array<cudf::transform_input, 1> inputs{input};
  std::array outputs{cudf::transform_output{cudf::data_type{cudf::type_id::INT32}}};
  auto wrong = cudf::transform_program_info::make<identity_operator<float>>();
  auto right = cudf::transform_program_info::make<identity_operator<int32_t>>();
  std::array mixed{wrong.view(), right.view()};
  auto result = run(inputs, outputs, mixed);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, result->get_column(0));
  std::array output_miss_owners{
    cudf::transform_program_info::make<identity_operator<int32_t, float>>()};
  auto output_miss = program_views(output_miss_owners);
  EXPECT_THROW(run(inputs, outputs, output_miss), std::invalid_argument);
  std::array misses{wrong.view()};
  EXPECT_THROW(run(inputs, outputs, misses), std::invalid_argument);
  EXPECT_THROW(run(inputs, outputs, {}), std::invalid_argument);
  std::array empty_entry_owners{cudf::transform_program_info{}};
  auto empty_entry = program_views(empty_entry_owners);
  EXPECT_THROW(run(inputs, outputs, empty_entry), std::invalid_argument);
  std::array<cudf::transform_input, 1> empty_inputs{cudf::slice(input, {0, 0}).front()};
  EXPECT_THROW(run(empty_inputs, outputs, misses), std::invalid_argument);
  EXPECT_THROW(run(empty_inputs, outputs, {}), std::invalid_argument);
  std::array scalar_miss_owners{
    cudf::transform_program_info::make<identity_operator<int32_t, int32_t, true>>()};
  auto scalar_miss = program_views(scalar_miss_owners);
  EXPECT_THROW(run(inputs, outputs, scalar_miss), std::invalid_argument);
  std::array nullable_miss_owners{
    cudf::transform_program_info::make<identity_operator<int32_t, int32_t, false, true>>()};
  auto nullable_miss = program_views(nullable_miss_owners);
  EXPECT_THROW(run(inputs, outputs, nullable_miss), std::invalid_argument);
}

struct increment_operator : identity_operator<int32_t> {
  struct Function {
    __device__ void operator()(int32_t* out, int32_t in) const { *out = in + 1; }
  };
};

TEST_F(TransformProgramTest, FirstCompatibleEntryWins)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<int32_t> incremented{2, 3, 4};
  std::array<cudf::transform_input, 1> inputs{input};
  std::array outputs{cudf::transform_output{cudf::data_type{cudf::type_id::INT32}}};
  std::array kernels_owners{cudf::transform_program_info::make<identity_operator<int32_t>>(),
                            cudf::transform_program_info::make<increment_operator>()};
  auto kernels = program_views(kernels_owners);
  auto result  = run(inputs, outputs, kernels);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, result->get_column(0));
  std::swap(kernels[0], kernels[1]);
  result = run(inputs, outputs, kernels);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(incremented, result->get_column(0));
}

TEST_F(TransformProgramTest, TagDispatch)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<int32_t> incremented{2, 3, 4};
  std::array<cudf::transform_input, 1> inputs{input};
  std::array outputs{cudf::transform_output{cudf::data_type{cudf::type_id::INT32}}};
  std::string tag = "increment";
  auto increment  = cudf::transform_program_info::make<increment_operator>(tag);
  tag             = "changed";
  std::array kernels_owners{
    cudf::transform_program_info::make<identity_operator<int32_t>>(),
    cudf::transform_program_info::make<identity_operator<int32_t>>("identity"),
    cudf::transform_program_info::make<identity_operator<float>>("increment"),
    std::move(increment)};
  auto kernels = program_views(kernels_owners);
  auto invoke  = [&](std::string_view name) {
    return cudf::transform_precompiled(
      kernels, cudf::null_aware::NO, std::nullopt, inputs, outputs, {}, std::nullopt, name);
  };
  auto result = invoke("increment");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(incremented, result->get_column(0));
  result = invoke("identity");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, result->get_column(0));
  result = run(inputs, outputs, kernels);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, result->get_column(0));
  EXPECT_THROW(invoke("missing"), std::invalid_argument);
  EXPECT_THROW(invoke("Increment"), std::invalid_argument);
  std::array named_only{kernels.back()};
  EXPECT_THROW(run(inputs, outputs, named_only), std::invalid_argument);

  cudf::transform_program program{
    kernels, cudf::null_aware::NO, std::nullopt, inputs, outputs, {}, "increment"};
  result = program.run(inputs, outputs, {}, std::nullopt);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(incremented, result->get_column(0));
  std::array input_specs{cudf::transform_input_spec{.type = cudf::type_id::INT32}};
  std::array output_specs{cudf::transform_output_spec{.type = cudf::type_id::INT32}};
  cudf::transform_program spec_program{
    kernels, cudf::null_aware::NO, std::nullopt, input_specs, output_specs, "increment"};
  result = spec_program.run(inputs, outputs, {}, std::nullopt);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(incremented, result->get_column(0));
  EXPECT_THROW(
    (cudf::transform_program{
      kernels, cudf::null_aware::NO, std::nullopt, input_specs, output_specs, "missing"}),
    std::invalid_argument);
}

struct scalar_add_operator {
  using Inputs  = cudf::type_list<int32_t, cudf::transform_input_type<int32_t, true>>;
  using Outputs = cudf::type_list<int32_t>;
  static constexpr bool NullAware = false;
  static constexpr bool UserData  = false;
  struct Function {
    __device__ void operator()(int32_t* out, int32_t in, int32_t scalar) const
    {
      *out = in + scalar;
    }
  };
};

TEST_F(TransformProgramTest, ScalarBroadcastAndSlicedInputs)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input{{0, 1, 2, 3, 4},
                                                        {true, true, false, true, true}};
  cudf::test::fixed_width_column_wrapper<int32_t> scalar{10};
  auto slice = cudf::slice(input, {1, 4}).front();
  std::array<cudf::transform_input, 2> inputs{slice, cudf::scalar_column_view{scalar}};
  std::array outputs{cudf::transform_output{cudf::data_type{cudf::type_id::INT32}}};
  std::array kernels_owners{cudf::transform_program_info::make<scalar_add_operator>()};
  auto kernels = program_views(kernels_owners);
  std::array<cudf::transform_input, 1> single_input{slice};
  EXPECT_THROW(run(single_input, outputs, kernels), std::invalid_argument);
  std::string source = "__device__ void add(int* out, int in, int scalar) { *out = in + scalar; }";
  auto expected      = run_jit(inputs, outputs, source);
  auto result        = run(inputs, outputs, kernels);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());
  cudf::test::fixed_width_column_wrapper<int32_t> null_scalar{{10}, {false}};
  inputs[1] = cudf::scalar_column_view{null_scalar};
  expected  = run_jit(inputs, outputs, source);
  result    = run(inputs, outputs, kernels);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());
}

struct nullable_multi_operator {
  using Inputs                    = cudf::type_list<int32_t>;
  using Outputs                   = cudf::type_list<int32_t, int32_t>;
  static constexpr bool NullAware = true;
  static constexpr bool UserData  = false;
  struct Function {
    __device__ void operator()(cuda::std::optional<int32_t>* first,
                               cuda::std::optional<int32_t>* second,
                               cuda::std::optional<int32_t> in) const
    {
      *first  = in.value_or(-1);
      *second = in.has_value() ? cuda::std::optional<int32_t>{*in + 1} : cuda::std::nullopt;
    }
  };
};

TEST_F(TransformProgramTest, NullableWarpBoundariesAndOutputPolicies)
{
  std::array kernels_owners{cudf::transform_program_info::make<nullable_multi_operator>()};
  auto kernels       = program_views(kernels_owners);
  std::string source = R"(__device__ void f(cuda::std::optional<int>* first,
    cuda::std::optional<int>* second, cuda::std::optional<int> in) {
      *first = in.value_or(-1);
      *second = in.has_value() ? cuda::std::optional<int>{*in + 1} : cuda::std::nullopt;
    })";
  for (auto rows : {0, 1, 31, 32, 33, 65, 1025}) {
    std::vector<int32_t> values(rows);
    std::iota(values.begin(), values.end(), 0);
    std::vector<bool> validity(rows);
    for (int i = 0; i < rows; ++i) {
      validity[i] = i % 3 != 0;
    }
    cudf::test::fixed_width_column_wrapper<int32_t> input(
      values.begin(), values.end(), validity.begin());
    std::array<cudf::transform_input, 1> inputs{input};
    std::array one_output{cudf::transform_output{cudf::data_type{cudf::type_id::INT32}}};
    EXPECT_THROW(run(inputs, one_output, kernels, cudf::null_aware::YES), std::invalid_argument);
    for (auto policy : {cudf::output_nullability::PRESERVE, cudf::output_nullability::ALL_VALID}) {
      std::array outputs{cudf::transform_output{cudf::data_type{cudf::type_id::INT32}, policy},
                         cudf::transform_output{cudf::data_type{cudf::type_id::INT32}}};
      auto expected = run_jit(inputs, outputs, source, cudf::null_aware::YES);
      auto result   = run(inputs, outputs, kernels, cudf::null_aware::YES);
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());
    }
  }
}

struct constant_operator {
  using Inputs                    = cudf::type_list<>;
  using Outputs                   = cudf::type_list<int32_t>;
  static constexpr bool NullAware = false;
  static constexpr bool UserData  = false;
  struct Function {
    __device__ void operator()(int32_t* out) const { *out = 7; }
  };
};

TEST_F(TransformProgramTest, NoInputsExplicitSize)
{
  std::array<cudf::transform_input, 0> inputs{};
  std::array outputs{cudf::transform_output{cudf::data_type{cudf::type_id::INT32}}};
  std::array kernels_owners{cudf::transform_program_info::make<constant_operator>()};
  auto kernels = program_views(kernels_owners);
  cudf::test::fixed_width_column_wrapper<int32_t> expected{7, 7, 7};
  auto result = run(inputs, outputs, kernels, cudf::null_aware::NO, std::nullopt, 3);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->get_column(0));
  result = run(inputs, outputs, kernels, cudf::null_aware::NO, std::nullopt, 0);
  EXPECT_EQ(result->num_rows(), 0);
}

TEST_F(TransformProgramTest, ScalarOnlyExplicitSizeAndZeroRows)
{
  cudf::test::fixed_width_column_wrapper<int32_t> scalar{7};
  cudf::test::fixed_width_column_wrapper<int32_t> expected{7, 7, 7};
  std::array<cudf::transform_input, 1> inputs{cudf::scalar_column_view{scalar}};
  std::array outputs{cudf::transform_output{cudf::data_type{cudf::type_id::INT32}}};
  std::array kernels_owners{
    cudf::transform_program_info::make<identity_operator<int32_t, int32_t, true>>()};
  auto kernels = program_views(kernels_owners);
  auto result  = run(inputs, outputs, kernels, cudf::null_aware::NO, std::nullopt, 3);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->get_column(0));
  result = run(inputs, outputs, kernels, cudf::null_aware::NO, std::nullopt, 0);
  EXPECT_EQ(result->num_rows(), 0);
  EXPECT_THROW(run(inputs, outputs, kernels), std::invalid_argument);
}

TEST_F(TransformProgramTest, Dictionaries)
{
  cudf::test::fixed_width_column_wrapper<float> numeric{{1.5, 3.0, 1.5, 2.0},
                                                        {true, false, true, true}};
  auto encoded = cudf::dictionary::encode(numeric);
  auto sliced  = cudf::slice(encoded->view(), {1, 4}).front();
  std::array<cudf::transform_input, 1> inputs{sliced};
  std::array outputs{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32}}};
  using dict = cudf::dictionary_element<int32_t, float>;
  std::array kernels_owners{cudf::transform_program_info::make<identity_operator<dict, float>>()};
  auto kernels       = program_views(kernels_owners);
  std::string source = "__device__ void f(float* out, float in) { *out = in; }";
  auto expected      = run_jit(inputs, outputs, source);
  auto result        = run(inputs, outputs, kernels);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());

  using wrong_key = cudf::dictionary_element<int32_t, double>;
  std::array key_miss_owners{
    cudf::transform_program_info::make<identity_operator<wrong_key, float>>()};
  auto key_miss = program_views(key_miss_owners);
  EXPECT_THROW(run(inputs, outputs, key_miss), std::invalid_argument);
  cudf::test::strings_column_wrapper strings{{"a", "ccc", "bb"}, {true, false, true}};
  auto string_encoded = cudf::dictionary::encode(strings);
  std::array<cudf::transform_input, 1> string_inputs{string_encoded->view()};
  std::array string_outputs{cudf::transform_output{cudf::data_type{cudf::type_id::STRING}}};
  using string_dict = cudf::dictionary_element<int32_t, cudf::string_view>;
  std::array string_kernels_owners{
    cudf::transform_program_info::make<identity_operator<string_dict, cudf::string_view>>()};
  auto string_kernels = program_views(string_kernels_owners);
  result              = run(string_inputs, string_outputs, string_kernels);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strings, result->get_column(0));
}

struct copy_string_operator {
  using Inputs                    = cudf::type_list<cudf::string_view>;
  using Outputs                   = cudf::type_list<cuda::std::span<char>>;
  static constexpr bool NullAware = false;
  static constexpr bool UserData  = false;
  struct Function {
    __device__ void operator()(cuda::std::span<char>* out, cudf::string_view in) const
    {
      for (int i = 0; i < in.size_bytes(); ++i) {
        (*out)[i] = in.data()[i];
      }
    }
  };
};

TEST_F(TransformProgramTest, StringRepresentationsAndOffsetWidths)
{
  cudf::test::strings_column_wrapper strings{{"a", "bb", "ccc"}, {true, false, true}};
  std::array<cudf::transform_input, 1> inputs{strings};
  std::array outputs{cudf::transform_output{cudf::data_type{cudf::type_id::STRING}}};
  std::array views_owners{
    cudf::transform_program_info::make<identity_operator<cudf::string_view>>()};
  auto views = program_views(views_owners);
  std::array nullable_views_owners{cudf::transform_program_info::make<
    identity_operator<cudf::string_view, cudf::string_view, false, true>>()};
  auto nullable_views = program_views(nullable_views_owners);
  auto result         = run(inputs, outputs, views);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strings, result->get_column(0));
  result = run(inputs, outputs, nullable_views, cudf::null_aware::YES);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strings, result->get_column(0));

  std::array mutable_strings_owners{cudf::transform_program_info::make<copy_string_operator>()};
  auto mutable_strings = program_views(mutable_strings_owners);
  EXPECT_THROW(run(inputs, outputs, mutable_strings), std::invalid_argument);
  std::array mixed{views[0], mutable_strings[0]};
  for (bool wide : {false, true}) {
    std::vector<std::unique_ptr<cudf::column>> offsets;
    if (wide) {
      offsets.push_back(cudf::test::fixed_width_column_wrapper<int64_t>{0, 1, 1, 4}.release());
    } else {
      offsets.push_back(cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 1, 4}.release());
    }
    result = run(
      inputs, outputs, mixed, cudf::null_aware::NO, std::nullopt, std::nullopt, std::move(offsets));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(strings, result->get_column(0));
  }
}

struct user_data_operator {
  using Inputs                    = cudf::type_list<int32_t>;
  using Outputs                   = cudf::type_list<int32_t>;
  static constexpr bool NullAware = false;
  static constexpr bool UserData  = true;
  struct Function {
    __device__ cudf::errc operator()(void* data,
                                     cudf::size_type row,
                                     int32_t* out,
                                     int32_t in) const
    {
      *out = in + static_cast<int32_t*>(data)[row];
      return cudf::errc::SUCCESS;
    }
  };
};

struct error_operator : identity_operator<int32_t> {
  struct Function {
    __device__ int operator()(int32_t* out, int32_t in) const
    {
      *out = in;
      return static_cast<int>(in == 2 ? cudf::errc::DIVISION_BY_ZERO : cudf::errc::SUCCESS);
    }
  };
};

TEST_F(TransformProgramTest, UserDataStreamAndErrors)
{
  rmm::cuda_stream stream;
  cudf::test::fixed_width_column_wrapper<int32_t> input{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<int32_t> expected{11, 22, 33};
  std::array<cudf::transform_input, 1> inputs{input};
  std::array outputs{cudf::transform_output{cudf::data_type{cudf::type_id::INT32}}};
  std::array<int32_t, 3> host_data{10, 20, 30};
  CUDF_CUDA_TRY(cudaStreamSynchronize(cudf::test::get_default_stream().get()));
  rmm::device_uvector<int32_t> data(host_data.size(), cuda::stream_ref{stream});
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    data.data(), host_data.data(), sizeof(host_data), cudaMemcpyHostToDevice, stream.value()));
  std::array kernels_owners{cudf::transform_program_info::make<user_data_operator>()};
  auto kernels = program_views(kernels_owners);
  auto result  = run(inputs,
                    outputs,
                    kernels,
                    cudf::null_aware::NO,
                    data.data(),
                    std::nullopt,
                     {},
                    cuda::stream_ref{stream});
  stream.synchronize();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->get_column(0));
  EXPECT_THROW(run(inputs, outputs, kernels), std::invalid_argument);
  std::array errors_owners{cudf::transform_program_info::make<error_operator>()};
  auto errors = program_views(errors_owners);
  EXPECT_THROW(run(inputs, outputs, errors), cudf::evaluation_error);
  EXPECT_THROW(run(inputs, outputs, errors, cudf::null_aware::NO, data.data()),
               std::invalid_argument);
}

TEST_F(TransformProgramTest, ProgramRetentionMovesAndValidation)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input{1, 2, 3};
  std::array<cudf::transform_input, 1> inputs{input};
  std::array outputs{cudf::transform_output{cudf::data_type{cudf::type_id::INT32}}};
  auto program = [&] {
    std::array kernels_owners{cudf::transform_program_info::make<identity_operator<int32_t>>()};
    auto kernels = program_views(kernels_owners);
    return cudf::transform_program{
      kernels, cudf::null_aware::NO, std::nullopt, inputs, outputs, {}};
  }();
  auto moved = std::move(program);
  for (int i = 0; i < 2; ++i) {
    auto result = moved.run(inputs, outputs, {}, std::nullopt);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, result->get_column(0));
  }
  std::array input_specs{cudf::transform_input_spec{.type = cudf::type_id::INT32}};
  std::array output_specs{cudf::transform_output_spec{.type = cudf::type_id::INT32}};
  auto spec_program = [&] {
    std::array kernels_owners{cudf::transform_program_info::make<identity_operator<int32_t>>()};
    auto kernels = program_views(kernels_owners);
    return cudf::transform_program{
      kernels, cudf::null_aware::NO, std::nullopt, input_specs, output_specs};
  }();
  moved       = std::move(spec_program);
  auto result = moved.run(inputs, outputs, {}, std::nullopt);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, result->get_column(0));
  std::array<cudf::transform_input, 1> scalar{
    cudf::scalar_column_view{cudf::slice(input, {0, 1}).front()}};
  EXPECT_THROW(moved.run(scalar, outputs, {}, 3), std::invalid_argument);
  std::array wrong_output{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32}}};
  EXPECT_THROW(moved.run(inputs, wrong_output, {}, std::nullopt), std::invalid_argument);
  std::array<cudf::transform_input, 0> no_inputs{};
  EXPECT_THROW(moved.run(no_inputs, outputs, {}, 3), std::invalid_argument);

  std::array misses_owners{cudf::transform_program_info::make<identity_operator<float>>()};
  auto misses = program_views(misses_owners);
  EXPECT_THROW(
    (cudf::transform_program{misses, cudf::null_aware::NO, std::nullopt, inputs, outputs, {}}),
    std::invalid_argument);
  EXPECT_THROW((cudf::transform_program{
                 misses, cudf::null_aware::NO, std::nullopt, input_specs, output_specs}),
               std::invalid_argument);
  EXPECT_THROW((cudf::transform_program{std::span<cudf::transform_program_info::view_type const>{},
                                        cudf::null_aware::NO,
                                        std::nullopt,
                                        input_specs,
                                        output_specs}),
               std::invalid_argument);
}

}  // namespace transform_program_test
