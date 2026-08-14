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

#include "assert_unary.h"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/encode.hpp>
#include <cudf/transform.hpp>

#include <array>
#include <utility>

namespace transformation {

struct RuntimeSupportTest : public cudf::test::BaseFixture {
 protected:
  cudf::test::fixed_width_column_wrapper<float> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  cudf::test::fixed_width_column_wrapper<float> b{
    0.1F, 0.25F, 0.5F, 0.1F, 0.4F, 0.75F, 0.2F, 0.33F, 0.45F, 0.66F};

  cudf::test::fixed_width_column_wrapper<float> b_nulls{
    {0.1F, 0.25F, 0.5F, 0.1F, 0.4F, 0.75F, 0.2F, 0.33F, 0.45F, 0.66F},
    {true, true, true, true, true, true, true, true, true, false}};

  cudf::test::fixed_width_column_wrapper<float> t{0.5f};

  cudf::test::fixed_width_column_wrapper<float> bad_col{1, 2, 3, 4, 5, 6, 7, 8, 9};

  cudf::test::structs_column_wrapper struct_col{a, b};

  static constexpr char const* udf =
    R"***(
    __device__ inline void lerp(float* out, float a, float b, float t) { *out = a - t * a + t * b; }
    )***";
};

struct AssertsTest : public RuntimeSupportTest {};

TEST_F(AssertsTest, TypeSupport)
{
  cudf::transform_input inputs[] = {a, b, cudf::scalar_column_view(t)};

  EXPECT_NO_THROW((void)std::move(
    cudf::transform(udf,
                    cudf::udf_source_type::CUDA,
                    cudf::null_aware::NO,
                    std::nullopt,
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32},
                                                      cudf::output_nullability::PRESERVE}},
                    {},
                    std::nullopt)
      ->release()
      .front()));

  EXPECT_THROW((void)std::move(cudf::transform(udf,
                                               cudf::udf_source_type::CUDA,
                                               cudf::null_aware::NO,
                                               std::nullopt,
                                               inputs,
                                               std::array{cudf::transform_output{
                                                 cudf::data_type{cudf::type_id::STRUCT},
                                                 cudf::output_nullability::PRESERVE}},
                                               {},
                                               std::nullopt)
                                 ->release()
                                 .front()),
               std::invalid_argument);

  cudf::transform_input struct_inputs[] = {struct_col, cudf::scalar_column_view(t)};

  EXPECT_THROW((void)std::move(cudf::transform(udf,
                                               cudf::udf_source_type::CUDA,
                                               cudf::null_aware::NO,
                                               std::nullopt,
                                               struct_inputs,
                                               std::array{cudf::transform_output{
                                                 cudf::data_type{cudf::type_id::FLOAT32},
                                                 cudf::output_nullability::PRESERVE}},
                                               {},
                                               std::nullopt)
                                 ->release()
                                 .front()),
               std::invalid_argument);
}

TEST_F(AssertsTest, UnequalRowCount)
{
  cudf::transform_input inputs[] = {a, b, bad_col};

  EXPECT_THROW((void)std::move(cudf::transform(udf,
                                               cudf::udf_source_type::CUDA,
                                               cudf::null_aware::NO,
                                               std::nullopt,
                                               inputs,
                                               std::array{cudf::transform_output{
                                                 cudf::data_type{cudf::type_id::FLOAT32},
                                                 cudf::output_nullability::PRESERVE}},
                                               {},
                                               std::nullopt)
                                 ->release()
                                 .front()),
               std::invalid_argument);
}

TEST_F(AssertsTest, NullSupport)
{
  cudf::transform_input inputs[] = {a, b_nulls, cudf::scalar_column_view(t)};
  EXPECT_NO_THROW((void)std::move(
    cudf::transform(udf,
                    cudf::udf_source_type::CUDA,
                    cudf::null_aware::NO,
                    std::nullopt,
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32},
                                                      cudf::output_nullability::PRESERVE}},
                    {},
                    std::nullopt)
      ->release()
      .front()));
}

struct UnaryOperationIntegrationTest : public cudf::test::BaseFixture {};

template <class dtype, class Op, class Data>
void test_udf(std::string const& udf,
              Op op,
              Data data_init,
              cudf::size_type size,
              cudf::udf_source_type source_type)
{
  auto all_valid = cudf::test::iterators::no_nulls();
  auto data_iter = cudf::detail::make_counting_transform_iterator(0, data_init);

  cudf::test::fixed_width_column_wrapper<dtype, typename decltype(data_iter)::value_type> in(
    data_iter, data_iter + size, all_valid);

  cudf::transform_input inputs[]    = {in};
  std::unique_ptr<cudf::column> out = std::move(
    cudf::transform(udf,
                    source_type,
                    cudf::null_aware::NO,
                    std::nullopt,
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_to_id<dtype>()),
                                                      cudf::output_nullability::PRESERVE}},
                    {},
                    std::nullopt)
      ->release()
      .front());

  ASSERT_UNARY<dtype, dtype>(out->view(), in, op);
}

TEST_F(UnaryOperationIntegrationTest, Transform_FP32_FP32)
{
  // c = a*a*a*a
  std::string const cuda =
    R"***(
__device__ inline void    fdsf   (
       float* C,
       float a
)
{
  *C = a*a*a*a;
}
)***";

  std::string const ptx =
    R"***(
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-24817639
// Cuda compilation tools, release 10.0, V10.0.130
// Based on LLVM 3.4svn
//

.version 6.3
.target sm_70
.address_size 64

	// .globl	_ZN8__main__7add$241Ef
.common .global .align 8 .u64 _ZN08NumbaEnv8__main__7add$241Ef;
.common .global .align 8 .u64 _ZN08NumbaEnv5numba7targets7numbers14int_power_impl12$3clocals$3e13int_power$242Efx;

.visible .func  (.param .b32 func_retval0) _ZN8__main__7add$241Ef(
	.param .b64 _ZN8__main__7add$241Ef_param_0,
	.param .b32 _ZN8__main__7add$241Ef_param_1
)
{
	.reg .f32 	%f<4>;
	.reg .b32 	%r<2>;
	.reg .b64 	%rd<2>;


	ld.param.u64 	%rd1, [_ZN8__main__7add$241Ef_param_0];
	ld.param.f32 	%f1, [_ZN8__main__7add$241Ef_param_1];
	mul.f32 	%f2, %f1, %f1;
	mul.f32 	%f3, %f2, %f2;
	st.f32 	[%rd1], %f3;
	mov.u32 	%r1, 0;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}
)***";

  using dtype    = float;
  auto op        = [](dtype a) { return a * a * a * a; };
  auto data_init = [](cudf::size_type row) { return row % 3; };

  test_udf<dtype>(cuda, op, data_init, 500, cudf::udf_source_type::CUDA);
  test_udf<dtype>(ptx, op, data_init, 500, cudf::udf_source_type::PTX);

  test_udf<dtype>(cuda, op, data_init, 0, cudf::udf_source_type::CUDA);
  test_udf<dtype>(ptx, op, data_init, 0, cudf::udf_source_type::PTX);
}

TEST_F(UnaryOperationIntegrationTest, Transform_INT32_INT32)
{
  // c = a * a - a
  std::string const cuda =
    "__device__ inline void f(int* output,int input){*output = input*input - input;}";

  // Generated from NUMBA, using:
  //
  // ```py
  //
  // from numba import cuda, int32
  // from numba.cuda import compile_ptx_for_current_device
  //
  // # Define a CUDA device function
  //
  // @cuda.jit(device=True)
  // def op(a):
  //         return a * a - a
  //
  // # Define argument types for the function
  // arg_types = (int32, )
  //
  // # Compile the device function as relocatable
  // ptx, _ = cuda.compile_ptx_for_current_device(op, arg_types, device=True)
  //
  //
  // # Print the PTX code
  // print("Relocatable PTX Code:")
  // print(ptx)
  //
  //
  // ```
  //
  std::string const ptx =
    R"***(
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-37061995
// Cuda compilation tools, release 13.1, V13.1.115
// Based on NVVM 7.0.1
//

.version 9.1
.target sm_86
.address_size 64

	// .globl	_ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEi
.common .global .align 8 .u64 _ZN08NumbaEnv8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEi;

.visible .func  (.param .b32 func_retval0) _ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEi(
	.param .b64 _ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEi_param_0,
	.param .b32 _ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEi_param_1
)
{
	.reg .b32 	%r<3>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [_ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEi_param_0];
	ld.param.u32 	%r1, [_ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEi_param_1];
	cvt.s64.s32 	%rd2, %r1;
	mul.wide.s32 	%rd3, %r1, %r1;
	sub.s64 	%rd4, %rd3, %rd2;
	st.u64 	[%rd1], %rd4;
	mov.u32 	%r2, 0;
	st.param.b32 	[func_retval0+0], %r2;
	ret;

}
)***";

  using dtype    = int;
  auto op        = [](dtype a) { return a * a - a; };
  auto data_init = [](cudf::size_type row) { return row % 78; };

  test_udf<dtype>(cuda, op, data_init, 500, cudf::udf_source_type::CUDA);
  test_udf<dtype>(ptx, op, data_init, 500, cudf::udf_source_type::PTX);
}

TEST_F(UnaryOperationIntegrationTest, Transform_INT8_INT8)
{
  // Capitalize all the lower case letters
  // Assuming ASCII, the PTX code is compiled from the following CUDA code

  std::string const cuda =
    R"***(
__device__ inline void f(
  signed char* output,
  signed char input
){
	if(input > 96 && input < 123){
  	*output = input - 32;
  }else{
  	*output = input;
  }
}
)***";

  // Generated from NUMBA, using:
  //
  // ```py
  //
  // from numba import cuda, int8
  // from numba.cuda import compile_ptx_for_current_device
  //
  // # Define a CUDA device function
  //
  // @cuda.jit(device=True)
  // def op(a):
  //         return (a - 32) if (a > 96 and a < 123) else a
  //
  // # Define argument types for the function
  // arg_types = (int8, )
  //
  // # Compile the device function as relocatable
  // ptx, _ = cuda.compile_ptx_for_current_device(op, arg_types, device=True)
  //
  //
  // # Print the PTX code
  // print("Relocatable PTX Code:")
  // print(ptx)
  //
  //
  // ```
  //
  std::string const ptx =
    R"***(
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-37061995
// Cuda compilation tools, release 13.1, V13.1.115
// Based on NVVM 7.0.1
//

.version 9.1
.target sm_86
.address_size 64

	// .globl	_ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEa
.common .global .align 8 .u64 _ZN08NumbaEnv8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEa;

.visible .func  (.param .b32 func_retval0) _ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEa(
	.param .b64 _ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEa_param_0,
	.param .b32 _ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEa_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .b16 	%rs<4>;
	.reg .b32 	%r<2>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [_ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEa_param_0];
	ld.param.u8 	%rs1, [_ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEa_param_1];
	add.s16 	%rs2, %rs1, -97;
	and.b16  	%rs3, %rs2, 255;
	setp.lt.u16 	%p1, %rs3, 26;
	cvt.u64.u16 	%rd3, %rs1;
	cvt.s64.s8 	%rd1, %rd3;
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;

$L__BB0_2:
	add.s64 	%rd4, %rd1, -32;
	st.u64 	[%rd2], %rd4;
	bra.uni 	$L__BB0_3;

$L__BB0_1:
	st.u64 	[%rd2], %rd1;

$L__BB0_3:
	mov.u32 	%r1, 0;
	st.param.b32 	[func_retval0+0], %r1;
	ret;

}
)***";

  using dtype    = int8_t;
  auto op        = [](dtype a) { return std::toupper(a); };
  auto data_init = [](cudf::size_type row) { return 'a' + (row % 26); };

  test_udf<dtype>(cuda, op, data_init, 500, cudf::udf_source_type::CUDA);
  test_udf<dtype>(ptx, op, data_init, 500, cudf::udf_source_type::PTX);
}

TEST_F(UnaryOperationIntegrationTest, Transform_Datetime)
{
  // Add one day to timestamp in microseconds

  std::string const cuda =
    R"***(
__device__ inline void f(cudf::timestamp_us* output, cudf::timestamp_us input)
{
  using dur = cuda::std::chrono::duration<int32_t, cuda::std::ratio<86400>>;
  *output = static_cast<cudf::timestamp_us>(input + dur{1});
}

)***";

  using dtype = cudf::timestamp_us;
  auto op     = [](dtype a) {
    using dur = cuda::std::chrono::duration<int32_t, cuda::std::ratio<86400>>;
    return static_cast<cudf::timestamp_us>(a + dur{1});
  };
  auto random_eng = cudf::test::UniformRandomGenerator<cudf::timestamp_us::rep>(0, 100000000);
  auto data_init  = [&random_eng](cudf::size_type row) { return random_eng.generate(); };

  test_udf<dtype>(cuda, op, data_init, 500, cudf::udf_source_type::CUDA);
}

TEST_F(UnaryOperationIntegrationTest, TransformDictionaryString)
{
  std::string const cuda =
    R"***(
__device__ inline void decode(cudf::string_view * output, cudf::string_view input){
  *output = input;
})***";

  // non-nullable
  {
    auto a =
      cudf::test::strings_column_wrapper{
        "eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa"}
        .release();

    auto a_encoded = cudf::dictionary::encode(a->view());

    cudf::transform_input inputs[] = {*a_encoded};

    auto out = std::move(
      cudf::transform(cuda,
                      cudf::udf_source_type::CUDA,
                      cudf::null_aware::NO,
                      std::nullopt,
                      inputs,
                      std::array{cudf::transform_output{cudf::data_type{cudf::type_id::STRING},
                                                        cudf::output_nullability::PRESERVE}},
                      {},
                      std::nullopt)
        ->release()
        .front());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), a->view());
  }

  // nullable
  {
    auto a =
      cudf::test::strings_column_wrapper{
        {"eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa"},
        {true, true, true, false, true, true, true, true, true}}
        .release();

    auto a_encoded = cudf::dictionary::encode(a->view());

    cudf::transform_input inputs[] = {*a_encoded};

    auto out = std::move(
      cudf::transform(cuda,
                      cudf::udf_source_type::CUDA,
                      cudf::null_aware::NO,
                      std::nullopt,
                      inputs,
                      std::array{cudf::transform_output{cudf::data_type{cudf::type_id::STRING},
                                                        cudf::output_nullability::PRESERVE}},
                      {},
                      std::nullopt)
        ->release()
        .front());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), a->view());
  }
}

TEST_F(UnaryOperationIntegrationTest, TransformDictionaryFloat)
{
  std::string const cuda =
    R"***(
__device__ inline void decode(float * output, float input){
  *output = input;
})***";

  // Generated from NUMBA, using:
  //
  // ```py
  //
  // from numba import cuda, float32
  // from numba.cuda import compile_ptx_for_current_device
  //
  // # Define a CUDA device function
  //
  // @cuda.jit(device=True)
  // def op(a):
  //         return a
  //
  // # Define argument types for the function
  // arg_types = (float32, )
  //
  // # Compile the device function as relocatable
  // ptx, _ = cuda.compile_ptx_for_current_device(op, arg_types, device=True)
  //
  //
  // # Print the PTX code
  // print("Relocatable PTX Code:")
  // print(ptx)
  //
  //
  // ```
  //
  std::string const ptx = R"***(
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: UNKNOWN
// Cuda compilation tools, release 13.2, V13.2.51
// Based on NVVM 7.0.1
//

.version 9.2
.target sm_86
.address_size 64

	// .globl	_ZN8__main__3opxB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEf
.common .global .align 8 .u64 _ZN08NumbaEnv8__main__3opxB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEf;

.visible .func  (.param .b32 func_retval0) _ZN8__main__3opxB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEf(
	.param .b64 _ZN8__main__3opxB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEf_param_0,
	.param .b32 _ZN8__main__3opxB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEf_param_1
)
{
	.reg .f32 	%f<2>;
	.reg .b32 	%r<2>;
	.reg .b64 	%rd<2>;


	ld.param.u64 	%rd1, [_ZN8__main__3opxB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEf_param_0];
	ld.param.f32 	%f1, [_ZN8__main__3opxB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEf_param_1];
	st.f32 	[%rd1], %f1;
	mov.u32 	%r1, 0;
	st.param.b32 	[func_retval0+0], %r1;
	ret;

}
)***";

  // non-nullable
  {
    auto a = cudf::test::fixed_width_column_wrapper<float>(
               {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 5.0F, 5.0F, 1.0F, 2.0F})
               .release();
    auto a_encoded                 = cudf::dictionary::encode(a->view());
    cudf::transform_input inputs[] = {*a_encoded};

    auto out = std::move(
      cudf::transform(cuda,
                      cudf::udf_source_type::CUDA,
                      cudf::null_aware::NO,
                      std::nullopt,
                      inputs,
                      std::array{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32},
                                                        cudf::output_nullability::PRESERVE}},
                      {},
                      std::nullopt)
        ->release()
        .front());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), a->view());

    auto out_ptx = std::move(
      cudf::transform(ptx,
                      cudf::udf_source_type::PTX,
                      cudf::null_aware::NO,
                      std::nullopt,
                      inputs,
                      std::array{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32},
                                                        cudf::output_nullability::PRESERVE}},
                      {},
                      std::nullopt)
        ->release()
        .front());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(out_ptx->view(), a->view());
  }

  // nullable
  {
    auto a = cudf::test::fixed_width_column_wrapper<float>(
               {{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 5.0F, 5.0F, 1.0F, 2.0F},
                {true, true, true, true, true, false, true, true, true}})
               .release();

    auto a_encoded                 = cudf::dictionary::encode(a->view());
    cudf::transform_input inputs[] = {*a_encoded};

    auto out = std::move(
      cudf::transform(cuda,
                      cudf::udf_source_type::CUDA,
                      cudf::null_aware::NO,
                      std::nullopt,
                      inputs,
                      std::array{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32},
                                                        cudf::output_nullability::PRESERVE}},
                      {},
                      std::nullopt)
        ->release()
        .front());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), a->view());

    auto out_ptx = std::move(
      cudf::transform(ptx,
                      cudf::udf_source_type::PTX,
                      cudf::null_aware::NO,
                      std::nullopt,
                      inputs,
                      std::array{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32},
                                                        cudf::output_nullability::PRESERVE}},
                      {},
                      std::nullopt)
        ->release()
        .front());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(out_ptx->view(), a->view());
  }

  // empty column
  {
    auto a_empty                   = cudf::test::fixed_width_column_wrapper<float>({}).release();
    auto a_encoded                 = cudf::dictionary::encode(a_empty->view());
    cudf::transform_input inputs[] = {*a_encoded};

    auto out = std::move(
      cudf::transform(cuda,
                      cudf::udf_source_type::CUDA,
                      cudf::null_aware::NO,
                      std::nullopt,
                      inputs,
                      std::array{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32},
                                                        cudf::output_nullability::PRESERVE}},
                      {},
                      std::nullopt)
        ->release()
        .front());

    EXPECT_EQ(out->size(), 0);
  }

  // sliced
  {
    auto a = cudf::test::fixed_width_column_wrapper<float>(
               {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 5.0F, 5.0F, 1.0F, 2.0F})
               .release();

    auto a_encoded     = cudf::dictionary::encode(a->view());
    auto sliced_input  = cudf::slice(a_encoded->view(), {2, 7}).front();
    auto sliced_expect = cudf::slice(a->view(), {2, 7}).front();

    cudf::transform_input inputs[] = {sliced_input};

    auto out = std::move(
      cudf::transform(cuda,
                      cudf::udf_source_type::CUDA,
                      cudf::null_aware::NO,
                      std::nullopt,
                      inputs,
                      std::array{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32},
                                                        cudf::output_nullability::PRESERVE}},
                      {},
                      std::nullopt)
        ->release()
        .front());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), sliced_expect);

    auto out_ptx = std::move(
      cudf::transform(ptx,
                      cudf::udf_source_type::PTX,
                      cudf::null_aware::NO,
                      std::nullopt,
                      inputs,
                      std::array{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32},
                                                        cudf::output_nullability::PRESERVE}},
                      {},
                      std::nullopt)
        ->release()
        .front());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(out_ptx->view(), sliced_expect);
  }
}

struct TernaryOperationTest : public cudf::test::BaseFixture {};

TEST_F(TernaryOperationTest, TransformWithScalar)
{
  std::string const cuda =
    R"***(
__device__ inline void transform(
       float* out,
       float a,
       float b,
       float c
)
{
  *out = (a + b) * c;
}
)***";

  // Generated from NUMBA, using:
  //
  // ```py
  //
  // from numba import cuda, float32
  // from numba.cuda import compile_ptx_for_current_device
  //
  // # Define a CUDA device function
  //
  // @cuda.jit(device=True)
  // def op(a, b, c):
  //         return (a + b) * c
  //
  // # Define argument types for the function
  // arg_types = (float32, float32, float32)
  //
  // # Compile the device function as relocatable
  // ptx, _ = cuda.compile_ptx_for_current_device(op, arg_types, device=True)
  //
  //
  // # Print the PTX code
  // print("Relocatable PTX Code:")
  // print(ptx)
  //
  //
  // ```
  //
  std::string const ptx =
    R"***(
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-35404655
// Cuda compilation tools, release 12.8, V12.8.61
// Based on NVVM 7.0.1
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEfff
.common .global .align 8 .u64 _ZN08NumbaEnv8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEfff;

.visible .func  (.param .b32 func_retval0) _ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEfff(
	.param .b64 _ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEfff_param_0,
	.param .b32 _ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEfff_param_1,
	.param .b32 _ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEfff_param_2,
	.param .b32 _ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEfff_param_3
)
{
	.reg .f32 	%f<6>;
	.reg .b32 	%r<2>;
	.reg .b64 	%rd<2>;


	ld.param.u64 	%rd1, [_ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEfff_param_0];
	ld.param.f32 	%f1, [_ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEfff_param_1];
	ld.param.f32 	%f2, [_ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEfff_param_2];
	ld.param.f32 	%f3, [_ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEfff_param_3];
	add.f32 	%f4, %f1, %f2;
	mul.f32 	%f5, %f4, %f3;
	st.f32 	[%rd1], %f5;
	mov.u32 	%r1, 0;
	st.param.b32 	[func_retval0+0], %r1;
	ret;

}
)***";

  using T = float;

  constexpr T A   = 90;
  constexpr T B   = 100;
  constexpr T C   = 5;
  constexpr T OUT = (A + B) * C;

  std::vector<T> a_host(200, A);
  std::vector<T> b_host(200, B);
  std::vector<T> c_host(1, C);
  std::vector<T> expected_host(200, OUT);

  cudf::test::fixed_width_column_wrapper<T> a(a_host.begin(), a_host.end());
  cudf::test::fixed_width_column_wrapper<T> b(b_host.begin(), b_host.end());
  cudf::test::fixed_width_column_wrapper<T> c(c_host.begin(), c_host.end());
  cudf::test::fixed_width_column_wrapper<T> expected(expected_host.begin(), expected_host.end());

  cudf::transform_input cuda_inputs[]       = {a, b, cudf::scalar_column_view(c)};
  std::unique_ptr<cudf::column> cuda_result = std::move(
    cudf::transform(cuda,
                    cudf::udf_source_type::CUDA,
                    cudf::null_aware::NO,
                    std::nullopt,
                    cuda_inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_to_id<T>()),
                                                      cudf::output_nullability::PRESERVE}},
                    {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cuda_result, expected);

  cudf::transform_input ptx_inputs[]       = {a, b, cudf::scalar_column_view(c)};
  std::unique_ptr<cudf::column> ptx_result = std::move(
    cudf::transform(ptx,
                    cudf::udf_source_type::PTX,
                    cudf::null_aware::NO,
                    std::nullopt,
                    ptx_inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_to_id<T>()),
                                                      cudf::output_nullability::PRESERVE}},
                    {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ptx_result, expected);
}

template <typename T>
struct TernaryDecimalOperationTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(TernaryDecimalOperationTest, cudf::test::FixedPointTypes);

TYPED_TEST(TernaryDecimalOperationTest, TransformDecimalsAndScalar)
{
  using T = TypeParam;

  auto type_name = cudf::type_to_name(cudf::data_type(cudf::type_to_id<T>()));

  // clang-format off
  std::string const cuda =
    "__device__ void transform("
    + type_name + "* out, "
    + type_name + " a,"
    + type_name + " b,"
    + type_name + " c) {\n"
    + "*out = ((a + b) * c);"
    + " }";
  // clang-format on

  T const A(10, numeric::scale_type{0});
  T const B(20, numeric::scale_type{-1});
  T const C(5, numeric::scale_type{-2});
  T const RES = ((A + B) * C);

  std::vector<typename T::rep> a_host(200, A.value());
  std::vector<typename T::rep> b_host(200, B.value());
  std::vector<typename T::rep> c_host(1, C.value());
  std::vector<typename T::rep> expected_host(200, RES.value());

  cudf::test::fixed_point_column_wrapper<typename T::rep> a(
    a_host.begin(), a_host.end(), A.scale());
  cudf::test::fixed_point_column_wrapper<typename T::rep> b(
    b_host.begin(), b_host.end(), B.scale());
  cudf::test::fixed_point_column_wrapper<typename T::rep> c(
    c_host.begin(), c_host.end(), C.scale());
  cudf::test::fixed_point_column_wrapper<typename T::rep> expected(
    expected_host.begin(), expected_host.end(), RES.scale());

  cudf::transform_input inputs[]            = {a, b, cudf::scalar_column_view(c)};
  std::unique_ptr<cudf::column> cuda_result = std::move(
    cudf::transform(
      cuda,
      cudf::udf_source_type::CUDA,
      cudf::null_aware::NO,
      std::nullopt,
      inputs,
      std::array{cudf::transform_output{cudf::data_type(cudf::type_to_id<T>(), RES.scale()),
                                        cudf::output_nullability::PRESERVE}},
      {},
      std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cuda_result, expected);
}

struct StringOperationTest : public cudf::test::BaseFixture {};

TEST_F(StringOperationTest, StringComparison)
{
  auto a = cudf::test::strings_column_wrapper{"a", "b", "ccc", "dddd", "eee"};
  auto b = cudf::test::strings_column_wrapper{"a", "b", "d", "dddddd", "e"};
  auto c = cudf::test::strings_column_wrapper{"a", "b", "e", "ddd", "e"};

  std::string cuda = R"***(
  __device__ void compare(bool * out, cudf::string_view a, cudf::string_view b, cudf::string_view c){
    *out = (a == b) || (b == c);
  }
  )***";

  auto expected = cudf::test::fixed_width_column_wrapper<bool>{true, true, false, false, true};
  cudf::transform_input inputs[] = {a, b, c};
  auto result                    = std::move(
    cudf::transform(cuda,
                    cudf::udf_source_type::CUDA,
                    cudf::null_aware::NO,
                    std::nullopt,
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::BOOL8),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(StringOperationTest, StringContains)
{
  auto a = cudf::test::strings_column_wrapper{"a", "ab", "ccc", "dddd", "eee", "123"};
  auto b = cudf::test::strings_column_wrapper{"aa", "b", "d", "dddddd", "e", "321"};

  std::string cuda = R"***(
  __device__ void transform(bool * out, cudf::string_view a, cudf::string_view b){
    *out =  a.find(b) != cudf::string_view::npos;
  }
  )***";

  auto expected =
    cudf::test::fixed_width_column_wrapper<bool>{false, true, false, false, true, false};
  cudf::transform_input inputs[] = {a, b};
  auto result                    = std::move(
    cudf::transform(cuda,
                    cudf::udf_source_type::CUDA,
                    cudf::null_aware::NO,
                    std::nullopt,
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::BOOL8),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(StringOperationTest, StringFind)
{
  auto a = cudf::test::strings_column_wrapper{"a", "b", " h", "123"};
  auto b = cudf::test::strings_column_wrapper{"aa", "about", "oh hi", "012345"};

  std::string cuda = R"***(
  __device__ void transform(int32_t * out, cudf::string_view a, cudf::string_view b){
    *out =  b.find(a);
  }
  )***";

  auto expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2, 1};
  cudf::transform_input inputs[] = {a, b};
  auto result                    = std::move(
    cudf::transform(cuda,
                    cudf::udf_source_type::CUDA,
                    cudf::null_aware::NO,
                    std::nullopt,
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::INT32),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(StringOperationTest, MixedTypes)
{
  auto a = cudf::test::strings_column_wrapper{"a", "b", "ccc", "dddd", "eee", "123"};
  auto b = cudf::test::strings_column_wrapper{"a", "b", "d", "dddddd", "e", "321"};
  auto c = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6};
  auto d = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 5, 6, 7, 8};

  std::string cuda = R"***(
  __device__ void transform(bool * out, cudf::string_view a, cudf::string_view b, int32_t c, int32_t d){
    *out =  (a == b) && (c == d);
  }
  )***";

  auto expected =
    cudf::test::fixed_width_column_wrapper<bool>{true, true, false, false, false, false};
  cudf::transform_input inputs[] = {a, b, c, d};
  auto result                    = std::move(
    cudf::transform(cuda,
                    cudf::udf_source_type::CUDA,
                    cudf::null_aware::NO,
                    std::nullopt,
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::BOOL8),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(StringOperationTest, Output)
{
  auto a = cudf::test::strings_column_wrapper{"this", "b", "c", "d", "e", "f"};
  auto b = cudf::test::strings_column_wrapper{"aa", "is", "dd", "ddd", "e", "fff"};
  auto c = cudf::test::strings_column_wrapper{"a", "b", "the", "dddd", "e", "fff"};
  auto d = cudf::test::strings_column_wrapper{"a", "b", "d", "largest", "lexicographical", "test"};
  auto empty = cudf::test::strings_column_wrapper{};

  std::string cuda = R"***(
    __device__ void transform(cudf::string_view * out, cudf::string_view a, cudf::string_view b, cudf::string_view c, cudf::string_view d){
      *out =  cuda::std::max(cuda::std::max(cuda::std::max(a, b), c), d);
    }
    )***";

  auto expected =
    cudf::test::strings_column_wrapper{"this", "is", "the", "largest", "lexicographical", "test"};
  cudf::transform_input inputs[] = {a, b, c, d};
  auto result                    = std::move(
    cudf::transform(cuda,
                    cudf::udf_source_type::CUDA,
                    cudf::null_aware::NO,
                    std::nullopt,
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::STRING),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());

  auto expected_empty                  = cudf::test::strings_column_wrapper{};
  cudf::transform_input empty_inputs[] = {empty, empty, empty, empty};
  auto result_empty                    = std::move(
    cudf::transform(cuda,
                    cudf::udf_source_type::CUDA,
                    cudf::null_aware::NO,
                    std::nullopt,
                    empty_inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::STRING),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_empty, result_empty->view());
}

TEST_F(StringOperationTest, OutputOffsetted)
{
  auto a = cudf::test::strings_column_wrapper{"x", "xx", "xxx", "xxxx", "xxxxx", "xxxxxx"};
  auto b = cudf::test::strings_column_wrapper{"aa", "bb", "cc", "dd", "ee", "ff"};
  auto c = cudf::test::strings_column_wrapper{"1", "2", "3", "4", "5", "6"};

  std::string cuda = R"***(
    __device__ void concat(cuda::std::span<char> * out, cudf::string_view a, cudf::string_view b, cudf::string_view c){
      auto iter = out->data();
      memcpy(iter, a.data(), a.size_bytes());
      iter += a.size_bytes();
      memcpy(iter, b.data(), b.size_bytes());
      iter += b.size_bytes();
      memcpy(iter, c.data(), c.size_bytes());
    }
    )***";

  auto offsets = cudf::test::fixed_width_column_wrapper<int32_t>{0, 4, 9, 15, 22, 30, 39}.release();
  std::vector<std::unique_ptr<cudf::column>> strings_offsets;
  strings_offsets.push_back(std::move(offsets));
  auto expected = cudf::test::strings_column_wrapper{
    "xaa1", "xxbb2", "xxxcc3", "xxxxdd4", "xxxxxee5", "xxxxxxff6"};
  cudf::transform_input inputs[]   = {a, b, c};
  cudf::transform_output outputs[] = {cudf::data_type(cudf::type_id::STRING)};
  auto result                      = cudf::transform(cuda,
                                cudf::udf_source_type::CUDA,
                                cudf::null_aware::NO,
                                std::nullopt,
                                inputs,
                                outputs,
                                std::move(strings_offsets),
                                std::nullopt);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->get_column(0));
}

TEST_F(StringOperationTest, OutputOffsettedMixed)
{
  auto a = cudf::test::strings_column_wrapper{"x", "xx", "xxx", "xxxx", "xxxxx", "xxxxxx"};
  auto b = cudf::test::strings_column_wrapper{"aa", "bb", "cc", "dd", "ee", "ff"};
  auto c = cudf::test::strings_column_wrapper{"1", "2", "3", "4", "5", "67"};

  std::string cuda = R"***(
    __device__ void concat(int32_t * ab_size, cuda::std::span<char> * out, bool * is_odd, cudf::string_view a, cudf::string_view b, cudf::string_view c){
      auto iter = out->data();
      auto begin = iter;
      memcpy(iter, a.data(), a.size_bytes());
      iter += a.size_bytes();
      memcpy(iter, b.data(), b.size_bytes());
      iter += b.size_bytes();
      memcpy(iter, c.data(), c.size_bytes());
      iter += c.size_bytes();
      auto out_size = static_cast<int32_t>(iter - begin);
      *is_odd = (out_size % 2) == 1;
      *ab_size = static_cast<int32_t>(a.size_bytes() + b.size_bytes());
    }
    )***";

  auto offsets = cudf::test::fixed_width_column_wrapper<int32_t>{0, 4, 9, 15, 22, 30, 40}.release();
  std::vector<std::unique_ptr<cudf::column>> strings_offsets;

  strings_offsets.push_back(nullptr);
  strings_offsets.push_back(std::move(offsets));
  strings_offsets.push_back(nullptr);

  auto expected_is_odd =
    cudf::test::fixed_width_column_wrapper<bool>{false, true, false, true, false, false};
  auto expected_strings = cudf::test::strings_column_wrapper{
    "xaa1", "xxbb2", "xxxcc3", "xxxxdd4", "xxxxxee5", "xxxxxxff67"};
  auto expected_sizes = cudf::test::fixed_width_column_wrapper<int32_t>{3, 4, 5, 6, 7, 8};
  auto expected       = cudf::table_view({expected_sizes, expected_strings, expected_is_odd});
  cudf::transform_input inputs[]   = {a, b, c};
  cudf::transform_output outputs[] = {{cudf::data_type(cudf::type_id::INT32)},
                                      {cudf::data_type(cudf::type_id::STRING)},
                                      {cudf::data_type(cudf::type_id::BOOL8)}};
  auto result                      = cudf::transform(cuda,
                                cudf::udf_source_type::CUDA,
                                cudf::null_aware::NO,
                                std::nullopt,
                                inputs,
                                outputs,
                                std::move(strings_offsets),
                                std::nullopt);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result->view());
}

TEST_F(StringOperationTest, StringConcat)
{
  auto first_name = cudf::test::strings_column_wrapper{
    "John", "Mia", "Abd", "Mendes", "Arya", "John", "François", "José", "Søren", "张"};
  auto last_name = cudf::test::strings_column_wrapper{
    "Doe", "Folk", "Louis", "Xi", "Serenity", "Scott", "Ольга", "Łukasz", "Zoë", "伟"};
  rmm::device_buffer scratch(100 * static_cast<cudf::column_view>(first_name).size(),
                             cudf::get_default_stream());
  auto scratch_sizes = cudf::test::fixed_width_column_wrapper<int32_t>{100};

  std::string cuda = R"***(
__device__ void transform(void* user_data, cudf::size_type row,
                          cudf::string_view* out,
                          cudf::string_view first_name,
                          cudf::string_view last_name,
                          int32_t size)
{
  char* it = static_cast<char*>(user_data) + static_cast<size_t>(row) * static_cast<size_t>(size);
  char const* const begin = it;

  memcpy(it, first_name.data(), first_name.size_bytes());
  it += first_name.size_bytes();

  memcpy(it, " ", 1);
  it += 1;

  memcpy(it, last_name.data(), last_name.size_bytes());
  it += last_name.size_bytes();

  *out = cudf::string_view{begin, static_cast<cudf::size_type>(it - begin)};
}
    )***";

  auto expected                  = cudf::test::strings_column_wrapper{"John Doe",
                                                     "Mia Folk",
                                                     "Abd Louis",
                                                     "Mendes Xi",
                                                     "Arya Serenity",
                                                     "John Scott",
                                                     "François Ольга",
                                                     "José Łukasz",
                                                     "Søren Zoë",
                                                     "张 伟"};
  cudf::transform_input inputs[] = {first_name, last_name, cudf::scalar_column_view(scratch_sizes)};
  auto result                    = std::move(
    cudf::transform(cuda,
                    cudf::udf_source_type::CUDA,
                    cudf::null_aware::NO,
                    scratch.data(),
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::STRING),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(StringOperationTest, EmptyInput)
{
  std::string rtn_bool = R"***(
  __device__ void transform(bool * out, cudf::string_view a, cudf::string_view b){
    *out =  a.find(b) != cudf::string_view::npos;
  }
  )***";

  auto empty                          = cudf::test::strings_column_wrapper();
  cudf::transform_input bool_inputs[] = {empty, empty};
  auto result                         = std::move(
    cudf::transform(rtn_bool,
                    cudf::udf_source_type::CUDA,
                    cudf::null_aware::NO,
                    std::nullopt,
                    bool_inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::BOOL8),
                                                      cudf::output_nullability::PRESERVE}},
                                            {},
                    std::nullopt)
      ->release()
      .front());
  EXPECT_EQ(0, result->size());

  std::string rtn_str                = R"***(
  __device__ void transform(cudf::string_view * out, cudf::string_view a, cudf::string_view b){
    *out =  (a==b ? a : cudf::string_view{});
  }
  )***";
  cudf::transform_input str_inputs[] = {empty, empty};
  result                             = std::move(
    cudf::transform(rtn_str,
                    cudf::udf_source_type::CUDA,
                    cudf::null_aware::NO,
                    std::nullopt,
                    str_inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::STRING),
                                                      cudf::output_nullability::PRESERVE}},
                                                {},
                    std::nullopt)
      ->release()
      .front());
  EXPECT_EQ(0, result->size());
}

TEST_F(StringOperationTest, MultiOutput)
{
  auto a = cudf::test::strings_column_wrapper{
    "415 555 1234", "212 867 5309", "214 947 5689", "712 312 1863", "777 214 9063", "964 123 2934"};
  auto expected_area_code =
    cudf::test::strings_column_wrapper{"415", "212", "214", "712", "777", "964"};
  auto expected_prefix =
    cudf::test::strings_column_wrapper{"555", "867", "947", "312", "214", "123"};
  auto expected_line_number =
    cudf::test::strings_column_wrapper{"1234", "5309", "5689", "1863", "9063", "2934"};

  auto expected = cudf::table_view({expected_area_code, expected_prefix, expected_line_number});

  std::string cuda = R"***(
__device__ void capture_group(cudf::string_view* area_code,
                              cudf::string_view* prefix,
                              cudf::string_view* line_number,
                              cudf::string_view input) {
    auto seek = [] (char const*& iter, char const* end, char c) {
        while (iter != end && *iter != c) {
            iter++;
        }
    };
    auto* iter = input.data();
    auto* end = iter + input.size_bytes();
    auto* area_code_begin = iter;
    seek(iter, end, ' ');
    auto* area_code_end = iter;
    iter++;
    auto* prefix_begin = iter;
    seek(iter, end, ' ');
    auto* prefix_end = iter;
    iter++;
    auto* line_number_begin = iter;
    seek(iter, end, ' ');
    auto* line_number_end = iter;
    *area_code = cudf::string_view{
        area_code_begin,
        static_cast<cudf::size_type>(area_code_end - area_code_begin)};
    *prefix = cudf::string_view{
        prefix_begin, static_cast<cudf::size_type>(prefix_end - prefix_begin)};
    *line_number = cudf::string_view{
        line_number_begin,
        static_cast<cudf::size_type>(line_number_end - line_number_begin)};
}
    )***";

  cudf::transform_input inputs[]   = {a};
  cudf::transform_output outputs[] = {{.type = cudf::data_type(cudf::type_id::STRING)},
                                      {.type = cudf::data_type(cudf::type_id::STRING)},
                                      {.type = cudf::data_type(cudf::type_id::STRING)}};
  auto result                      = cudf::transform(cuda,
                                cudf::udf_source_type::CUDA,
                                cudf::null_aware::NO,
                                std::nullopt,
                                inputs,
                                outputs,
                                                     {},
                                std::nullopt);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result->view());
}

struct NullTest : public cudf::test::BaseFixture {
 protected:
  char const* const cuda =
    "__device__ inline void lerp(float* output, float low, float high, float t){*output = low - t "
    "* low + t * high; }";

  // Generated from NUMBA, using:
  //
  // ```py
  //
  // from numba import cuda, float32
  // from numba.cuda import compile_ptx_for_current_device
  //
  // # Define a CUDA device function
  //
  // @cuda.jit(device=True)
  // def op(low, high, t):
  //         return low - t * low + t * high
  //
  // # Define argument types for the function
  // arg_types = (float32, float32, float32)
  //
  // # Compile the device function as relocatable
  // ptx, _ = cuda.compile_ptx_for_current_device(op, arg_types, device=True)
  //
  //
  // # Print the PTX code
  // print("Relocatable PTX Code:")
  // print(ptx)
  //
  //
  // ```
  //
  char const* const ptx =
    R"***(
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-35813241
// Cuda compilation tools, release 12.9, V12.9.41
// Based on NVVM 7.0.1
//

.version 8.8
.target sm_86
.address_size 64

// .globl	_ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEfff
.common .global .align 8 .u64 _ZN08NumbaEnv8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEfff;

.visible .func  (.param .b32 func_retval0) _ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEfff(
.param .b64 _ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEfff_param_0,
.param .b32 _ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEfff_param_1,
.param .b32 _ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEfff_param_2,
.param .b32 _ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEfff_param_3
)
{
.reg .f32 	%f<7>;
.reg .b32 	%r<2>;
.reg .b64 	%rd<2>;


ld.param.u64 	%rd1, [_ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEfff_param_0];
ld.param.f32 	%f1, [_ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEfff_param_1];
ld.param.f32 	%f2, [_ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEfff_param_2];
ld.param.f32 	%f3, [_ZN8__main__2opB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw09RRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dEfff_param_3];
mul.f32 	%f4, %f1, %f3;
sub.f32 	%f5, %f1, %f4;
fma.rn.f32 	%f6, %f2, %f3, %f5;
st.f32 	[%rd1], %f6;
mov.u32 	%r1, 0;
st.param.b32 	[func_retval0+0], %r1;
ret;
}
)***";

  float const LOW         = 100.0F;
  float const HIGH        = 200.0F;
  float const T           = 0.25F;
  float const RES         = LOW - T * LOW + T * HIGH;
  cudf::size_type const N = 200;

  std::vector<float> low_host      = std::vector<float>(N, LOW);
  std::vector<float> high_host     = std::vector<float>(N, HIGH);
  std::vector<float> t_host        = std::vector<float>(N, T);
  std::vector<float> t_scalar_host = std::vector<float>(1, T);
  std::vector<float> expected_host = std::vector<float>(N, RES);

  auto fourth() { return cudf::test::iterators::valids_at_multiples_of(4); }

  auto fifth() { return cudf::test::iterators::valids_at_multiples_of(5); }
};

TEST_F(NullTest, ColumnNulls)
{
  auto low =
    cudf::test::fixed_width_column_wrapper<float>(low_host.begin(), low_host.end(), fourth())
      .release();
  auto high =
    cudf::test::fixed_width_column_wrapper<float>(high_host.begin(), high_host.end(), fifth())
      .release();
  auto t = cudf::test::fixed_width_column_wrapper<float>(t_host.begin(), t_host.end()).release();
  auto expected =
    cudf::test::fixed_width_column_wrapper<float>(expected_host.begin(), expected_host.end())
      .release();

  auto [expected_mask, expect_null_count] = cudf::bitmask_and(cudf::table_view({*low, *high}));

  expected->set_null_mask(std::move(expected_mask), expect_null_count);

  cudf::transform_input cuda_inputs[] = {*low, *high, *t};
  auto cuda_result                    = std::move(
    cudf::transform(cuda,
                    cudf::udf_source_type::CUDA,
                    cudf::null_aware::NO,
                    std::nullopt,
                    cuda_inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::FLOAT32),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cuda_result, *expected);

  cudf::transform_input ptx_inputs[] = {*low, *high, *t};
  auto ptx_result                    = std::move(
    cudf::transform(ptx,
                    cudf::udf_source_type::PTX,
                    cudf::null_aware::NO,
                    std::nullopt,
                    ptx_inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::FLOAT32),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ptx_result, *expected);
}

TEST_F(NullTest, ColumnNulls_And_Scalar)
{
  auto low =
    cudf::test::fixed_width_column_wrapper<float>(low_host.begin(), low_host.end(), fourth())
      .release();
  auto high =
    cudf::test::fixed_width_column_wrapper<float>(high_host.begin(), high_host.end(), fifth())
      .release();
  auto t_scalar =
    cudf::test::fixed_width_column_wrapper<float>(t_scalar_host.begin(), t_scalar_host.end())
      .release();
  auto expected =
    cudf::test::fixed_width_column_wrapper<float>(expected_host.begin(), expected_host.end())
      .release();

  auto [expected_mask, expect_null_count] = cudf::bitmask_and(cudf::table_view({*low, *high}));

  expected->set_null_mask(std::move(expected_mask), expect_null_count);

  cudf::transform_input cuda_inputs[] = {*low, *high, cudf::scalar_column_view(*t_scalar)};
  auto cuda_result                    = std::move(
    cudf::transform(cuda,
                    cudf::udf_source_type::CUDA,
                    cudf::null_aware::NO,
                    std::nullopt,
                    cuda_inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::FLOAT32),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cuda_result, *expected);

  cudf::transform_input ptx_inputs[] = {*low, *high, cudf::scalar_column_view(*t_scalar)};
  auto ptx_result                    = std::move(
    cudf::transform(ptx,
                    cudf::udf_source_type::PTX,
                    cudf::null_aware::NO,
                    std::nullopt,
                    ptx_inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::FLOAT32),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ptx_result, *expected);
}

TEST_F(NullTest, ColumnNulls_And_ScalarNull)
{
  auto low =
    cudf::test::fixed_width_column_wrapper<float>(low_host.begin(), low_host.end(), fourth())
      .release();
  auto high =
    cudf::test::fixed_width_column_wrapper<float>(high_host.begin(), high_host.end(), fifth())
      .release();
  auto t_scalar = cudf::test::fixed_width_column_wrapper<float>(
                    t_scalar_host.begin(), t_scalar_host.end(), {false})
                    .release();
  auto expected =
    cudf::test::fixed_width_column_wrapper<float>(expected_host.begin(), expected_host.end())
      .release();

  expected->set_null_mask(cudf::create_null_mask(low_host.size(), cudf::mask_state::ALL_NULL),
                          low->size());

  cudf::transform_input cuda_inputs[] = {*low, *high, cudf::scalar_column_view(*t_scalar)};
  auto cuda_result                    = std::move(
    cudf::transform(cuda,
                    cudf::udf_source_type::CUDA,
                    cudf::null_aware::NO,
                    std::nullopt,
                    cuda_inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::FLOAT32),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cuda_result, *expected);

  cudf::transform_input ptx_inputs[] = {*low, *high, cudf::scalar_column_view(*t_scalar)};
  auto ptx_result                    = std::move(
    cudf::transform(ptx,
                    cudf::udf_source_type::PTX,
                    cudf::null_aware::NO,
                    std::nullopt,
                    ptx_inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::FLOAT32),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ptx_result, *expected);
}

TEST_F(NullTest, IsNull)
{
  auto udf = R"***(
  __device__ inline void is_null(cuda::std::optional<bool>* output, cuda::std::optional<float> input)
  {
    *output = !input.has_value();
  }
  )***";

  auto value = cudf::test::fixed_width_column_wrapper<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
                                                             {false, false, true, false, true})
                 .release();

  auto expected = cudf::test::fixed_width_column_wrapper<bool>({true, true, false, true, false});

  cudf::transform_input inputs[] = {*value};
  auto result                    = std::move(
    cudf::transform(udf,
                    cudf::udf_source_type::CUDA,
                    cudf::null_aware::YES,
                    std::nullopt,
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::BOOL8),
                                                      cudf::output_nullability::ALL_VALID}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(NullTest, NullProject)
{
  auto udf = R"***(
__device__ inline void null_lerp(
       cuda::std::optional<float>* output,
       cuda::std::optional<float> low,
       cuda::std::optional<float> high,
       cuda::std::optional<float> t
)
{
auto lerp = [] (auto l, auto h, auto t) {
return l - t * l + t * h;
};
  *output =  low.has_value() && high.has_value() && t.has_value()
    ? lerp(*low, *high, *t)
    : 0.0F;
}
)***";

  auto low =
    cudf::test::fixed_width_column_wrapper<float>(low_host.begin(), low_host.end(), fourth())
      .release();
  auto high =
    cudf::test::fixed_width_column_wrapper<float>(high_host.begin(), high_host.end(), fifth())
      .release();
  auto t = cudf::test::fixed_width_column_wrapper<float>(t_host.begin(), t_host.end()).release();

  auto expected_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](auto i) { return ((i % 5) == 0) && ((i % 4) == 0) ? expected_host[i] : 0.0F; });

  auto expected =
    cudf::test::fixed_width_column_wrapper<float>(expected_iter, expected_iter + low_host.size())
      .release();

  cudf::transform_input inputs[] = {*low, *high, *t};
  auto cuda_result               = std::move(
    cudf::transform(udf,
                    cudf::udf_source_type::CUDA,
                    cudf::null_aware::YES,
                    std::nullopt,
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::FLOAT32),
                                                      cudf::output_nullability::ALL_VALID}},
                                  {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cuda_result, *expected);
}

TEST_F(UnaryOperationIntegrationTest, Transform_ErrorHandling)
{
  auto cuda =
    R"***(
__device__ cudf::errc expression (
       int* c,
       int a,
       int b
)
{
  auto e = cudf::detail::ops::div_overflow(a, b);
  if(!e.has_value()){
    return e.error();
  }
  *c = e.value();
  return cudf::errc::SUCCESS;
}
)***";

  cudf::test::fixed_width_column_wrapper<int32_t> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  cudf::test::fixed_width_column_wrapper<int32_t> b_fail{1, 2, 3, 4, 5, 6, 0, 8, 9, 10};
  cudf::test::fixed_width_column_wrapper<int32_t> b{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  cudf::test::fixed_width_column_wrapper<int32_t> expected{1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  {
    cudf::transform_input inputs[]   = {a, b};
    cudf::transform_output outputs[] = {
      {cudf::data_type(cudf::type_id::INT32), cudf::output_nullability::ALL_VALID}};
    std::unique_ptr<cudf::table> result;
    EXPECT_NO_THROW(result = cudf::transform(cuda,
                                             cudf::udf_source_type::CUDA,
                                             cudf::null_aware::NO,
                                             std::nullopt,
                                             inputs,
                                             outputs,
                                             {},
                                             std::nullopt));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), expected);
  }

  {
    cudf::transform_input inputs[]   = {a, b_fail};
    cudf::transform_output outputs[] = {
      {cudf::data_type(cudf::type_id::INT32), cudf::output_nullability::ALL_VALID}};
    std::unique_ptr<cudf::table> result;
    EXPECT_THROW(result = cudf::transform(cuda,
                                          cudf::udf_source_type::CUDA,
                                          cudf::null_aware::NO,
                                          std::nullopt,
                                          inputs,
                                          outputs,
                                          {},
                                          std::nullopt),
                 cudf::evaluation_error);
  }
}

}  // namespace transformation
