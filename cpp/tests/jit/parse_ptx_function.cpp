/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/parser.hpp"

#include <cudf_test/testing_main.hpp>

#include <algorithm>
#include <cctype>

struct JitParseTest : public ::testing::Test {};

TEST_F(JitParseTest, PTXNoFunction)
{
  std::string raw_ptx = R"(
.visible .entry _ZN3cub17CUB_101702_750_NS11EmptyKernelIvEEvv()
{
  ret;
})";

  EXPECT_THROW(cudf::jit::parse_single_function_ptx(raw_ptx, "GENERIC_OP", {{0, "float *"}}),
               cudf::logic_error);
}

inline bool ptx_equal(std::string input, std::string expected)
{
  // Remove all whitespace and newline characters and compare
  // This allows us to handle things like excess newline characters
  // and trailing whitespace in the 'input'

  auto whitespace_or_newline = [](unsigned char c) { return std::isspace(c) || c == '\n'; };
  input.erase(std::remove_if(input.begin(), input.end(), whitespace_or_newline), input.end());
  expected.erase(std::remove_if(expected.begin(), expected.end(), whitespace_or_newline),
                 expected.end());
  return input == expected;
}

TEST_F(JitParseTest, SimplePTX)
{
  std::string raw_ptx = R"(
.visible .func  (.param .b32 func_retval0) _ZN8__main__7add$241Eff(
  .param .b64 _ZN8__main__7add$241Eff_param_0,
  .param .b32 _ZN8__main__7add$241Eff_param_1,
  .param .b32 _ZN8__main__7add$241Eff_param_2
)
{
  ret;
}
)";

  std::string expected = R"(
__device__ __inline__ void GENERIC_OP(
  float* _ZN8__main__7add_241Eff_param_0,
  int _ZN8__main__7add_241Eff_param_1,
  int _ZN8__main__7add_241Eff_param_2
){
 asm volatile ("{");
 asm volatile ("bra RETTGT;");
 /** ret*/
 asm volatile ("RETTGT:}");}
)";

  std::string cuda_source =
    cudf::jit::parse_single_function_ptx(raw_ptx, "GENERIC_OP", {{0, "float *"}});

  EXPECT_TRUE(ptx_equal(cuda_source, expected));
}

TEST_F(JitParseTest, PTXWithBranchLabel)
{
  std::string raw_ptx = R"(
.visible .func _Z1flPaS_(){
BB0:
  ret;
}
)";

  std::string expected = R"(
__device__ __inline__ void GENERIC_OP(){
 asm volatile ("{");
 asm volatile ("BB0: bra RETTGT;");
 /** BB0: ret*/
 asm volatile ("RETTGT:}");}
)";

  std::string cuda_source = cudf::jit::parse_single_function_ptx(raw_ptx, "GENERIC_OP", {});

  EXPECT_TRUE(ptx_equal(cuda_source, expected));
}

TEST_F(JitParseTest, PTXWithPragma)
{
  std::string raw_ptx = R"(
.visible .func _ZN3cub17CUB_101702_750_NS11EmptyKernelIvEEvv()
{
$L__BB0_151:
  .pragma "nounroll";
  mov.u32 % r1517, % r1516;
  mov.u32 % r1516, % r1515;
  mov.u32 % r1515, % r1505;
  mov.u32 % r1457, 0;
$L__BB0_152:
  .pragma "nounroll";
})";

  std::string expected = R"(
__device__ __inline__ void EmptyKern(){
 asm volatile ("{");  asm volatile (" $L__BB0_151:  .pragma \"nounroll\";");
   /**   $L__BB0_151:
  .pragma "nounroll"  */

  asm volatile ("  mov.u32 _ r1517, _ r1516;");
   /**   mov.u32 % r1517, % r1516  */

  asm volatile ("  mov.u32 _ r1516, _ r1515;");
   /**   mov.u32 % r1516, % r1515  */

  asm volatile ("  mov.u32 _ r1515, _ r1505;");
   /**   mov.u32 % r1515, % r1505  */

  asm volatile ("  mov.u32 _ r1457, 0;");
   /**   mov.u32 % r1457, 0  */

  asm volatile (" $L__BB0_152:  .pragma \"nounroll\";");
   /**   $L__BB0_152:
  .pragma "nounroll"  */

 asm volatile ("RETTGT:}");}
)";

  std::string cuda_source = cudf::jit::parse_single_function_ptx(raw_ptx, "EmptyKern", {});
  EXPECT_TRUE(ptx_equal(cuda_source, expected));
}

TEST_F(JitParseTest, PTXWithPragmaWithSpaces)
{
  std::string raw_ptx = R"(
.visible .func _ZN3cub17CUB_101702_750_NS11EmptyKernelIvEEvv()
{
  $L__BB0_58:
    ld.param.u32 % r1419, [% rd419 + 80];
    setp.ne.s32 % p394, % r1419, 22;
    mov.u32 % r2050, 0;
    mov.u32 % r2048, % r2050;
    @ % p394 bra $L__BB0_380;

    ld.param.u8 % rs1369, [% rd419 + 208];
    setp.eq.s16 % p395, % rs1369, 0;
    selp.b32 % r1422, % r1925, 0, % p395;
    ld.param.u32 % r1423, [% rd419 + 112];
    add.s32 % r427, % r1422, % r1423;
    ld.param.u64 % rd1249, [% rd419 + 120];
    cvta.to.global.u64 % rd1250, % rd1249;
    .pragma "used_bytes_mask 4095";
    ld.global.v4.u32{ % r1424, % r1425, % r1426, % r1427}, [% rd1250];
    ld.global.v2.u64{ % rd1251, % rd1252}, [% rd1250 + 16];
    ld.global.s32 % rd230, [% rd1250 + 32];
    setp.gt.s32 % p396, % r1424, 6;
    @ % p396 bra $L__BB0_376;
}
}
)";

  std::string expected = R"(
__device__ __inline__ void LongKernel(){
 asm volatile ("{");  asm volatile (" $L__BB0_58:  cvt.u32.u32 _  %0, [_ rd419 + 80];": : "r"( *reinterpret_cast<int *>(&r1419) ));
   /**   $L__BB0_58:
    ld.param.u32 % r1419, [% rd419 + 80]  */

  asm volatile ("  setp.ne.s32 _ p394, _ r1419, 22;");
   /**   setp.ne.s32 % p394, % r1419, 22  */

  asm volatile ("  mov.u32 _ r2050, 0;");
   /**   mov.u32 % r2050, 0  */

  asm volatile ("  mov.u32 _ r2048, _ r2050;");
   /**   mov.u32 % r2048, % r2050  */

  asm volatile ("  @ _ p394 bra $L__BB0_380;");
   /**   @ % p394 bra $L__BB0_380  */

  asm volatile ("  cvt.u8.u8 _  %0, [_ rd419 + 208];": : "h"( static_cast<short>( rs1369 ) ));
   /**   ld.param.u8 % rs1369, [% rd419 + 208]  */

  asm volatile ("  setp.eq.s16 _ p395, _ rs1369, 0;");
   /**   setp.eq.s16 % p395, % rs1369, 0  */

  asm volatile ("  selp.b32 _ r1422, _ r1925, 0, _ p395;");
   /**   selp.b32 % r1422, % r1925, 0, % p395  */

  asm volatile ("  cvt.u32.u32 _  %0, [_ rd419 + 112];": : "r"( *reinterpret_cast<int *>(&r1423) ));
   /**   ld.param.u32 % r1423, [% rd419 + 112]  */

  asm volatile ("  add.s32 _ r427, _ r1422, _ r1423;");
   /**   add.s32 % r427, % r1422, % r1423  */

  asm volatile ("  mov.u64 _  %0, [_ rd419 + 120];": : "l"( *reinterpret_cast<long long int *>(&rd1249) ));
   /**   ld.param.u64 % rd1249, [% rd419 + 120]  */

  asm volatile ("  cvta.to.global.u64 _ rd1250, _ rd1249;");
   /**   cvta.to.global.u64 % rd1250, % rd1249  */

  asm volatile ("  .pragma \"used_bytes_mask 4095\";");
   /**   .pragma "used_bytes_mask 4095"  */

  asm volatile ("  ld.global.v4.u32{ _ r1424, _ r1425, _ r1426, _ r1427}, [_ rd1250];");
   /**   ld.global.v4.u32{ % r1424, % r1425, % r1426, % r1427}, [% rd1250]  */

  asm volatile ("  ld.global.v2.u64{ _ rd1251, _ rd1252}, [_ rd1250 + 16];");
   /**   ld.global.v2.u64{ % rd1251, % rd1252}, [% rd1250 + 16]  */

  asm volatile ("  ld.global.s32 _ rd230, [_ rd1250 + 32];");
   /**   ld.global.s32 % rd230, [% rd1250 + 32]  */

  asm volatile ("  setp.gt.s32 _ p396, _ r1424, 6;");
   /**   setp.gt.s32 % p396, % r1424, 6  */

  asm volatile ("  @ _ p396 bra $L__BB0_376;");
   /**   @ % p396 bra $L__BB0_376  */

  asm volatile ("RETTGT:}");}
 )";

  std::string cuda_source = cudf::jit::parse_single_function_ptx(raw_ptx, "LongKernel", {});
  EXPECT_TRUE(ptx_equal(cuda_source, expected));
}

// test that an ld.param instruction that doesn't contain the exact semantic type
// is still parsed correctly. This is important because NVVM IR doesn't always use the exact
// semantic type in the ld.param instruction. For example, it may use `ld.param.u8` to load a `char`
// parameter, which is semantically correct but doesn't match the expected `ld.param.s8`.
TEST_F(JitParseTest, PTXWithUntypedLdParam)
{
  //
  // Generated from NUMBA using:
  //
  // """py
  //
  // from numba import cuda, float64
  // from numba.cuda import compile_ptx_for_current_device
  //
  // @cuda.jit(device=True)
  // def op(a, b, c):
  //         return (a + b) *c
  //
  // ptx, _ = cuda.compile_ptx_for_current_device(op,  (float64, float64, float64), device=True,
  // abi="c")
  //
  // print(ptx)
  //
  // """
  //
  auto raw_ptx = R"***(
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-37061995
// Cuda compilation tools, release 13.1, V13.1.115
// Based on NVVM 7.0.1
//


.visible .func  (.param .b32 func_retval0) mad(
	.param .b64 param_0,
	.param .b64 param_1,
	.param .b64 param_2,
	.param .b64 param_3
)
{
	.reg .b32 	%r<2>;
	.reg .f64 	%fd<6>;
	.reg .b64 	%rd<2>;


	ld.param.u64 	%rd1, [param_0];
	ld.param.f64 	%fd1, [param_1];
	ld.param.f64 	%fd2, [param_2];
	ld.param.f64 	%fd3, [param_3];
	add.f64 	%fd4, %fd1, %fd2;
	mul.f64 	%fd5, %fd4, %fd3;
	st.f64 	[%rd1], %fd5;
	mov.u32 	%r1, 0;
	st.param.b32 	[func_retval0+0], %r1;
	ret;

}
)***";

  std::string expected = R"***(
  __device__ __inline__ void GENERIC_OP(double * param_0,
 double param_1,
 double param_2,
 double param_3){

 asm volatile ("{");  asm volatile ("  .reg .b32 _r<2>;");
   /**   .reg .b32      %r<2>  */

  asm volatile ("  .reg .f64 _fd<6>;");
   /**   .reg .f64      %fd<6>  */

  asm volatile ("  .reg .b64 _rd<2>;");
   /**   .reg .b64      %rd<2>  */

  asm volatile ("  mov.u64 _rd1,  %0;": : "l"( *reinterpret_cast<long long int *>(&param_0) ));
   /**   ld.param.u64   %rd1, [param_0]  */

  asm volatile ("  mov.f64 _fd1,  %0;": : "d"( *reinterpret_cast<double *>(&param_1) ));
   /**   ld.param.f64   %fd1, [param_1]  */

  asm volatile ("  mov.f64 _fd2,  %0;": : "d"( *reinterpret_cast<double *>(&param_2) ));
   /**   ld.param.f64   %fd2, [param_2]  */

  asm volatile ("  mov.f64 _fd3,  %0;": : "d"( *reinterpret_cast<double *>(&param_3) ));
   /**   ld.param.f64   %fd3, [param_3]  */

  asm volatile ("  add.f64 _fd4, _fd1, _fd2;");
   /**   add.f64        %fd4, %fd1, %fd2  */

  asm volatile ("  mul.f64 _fd5, _fd4, _fd3;");
   /**   mul.f64        %fd5, %fd4, %fd3  */

  asm volatile ("  st.f64 [_rd1], _fd5;");
   /**   st.f64         [%rd1], %fd5  */

  asm volatile ("  mov.u32 _r1, 0;");
   /**   mov.u32        %r1, 0  */

  asm volatile (" /** *** The way we parse the CUDA PTX assumes the function returns the return value through the first function parameter. Thus the `st.param.***` instructions are not processed. *** */");
   /**   st.param.b32   [func_retval0+0], %r1  */

  asm volatile ("  bra RETTGT;");
   /**   ret  */



 asm volatile ("RETTGT:}");}
)***";

  std::string cuda_source = cudf::jit::parse_single_function_ptx(raw_ptx,
                                                                 "GENERIC_OP",
                                                                 {
                                                                   {0, "double *"},
                                                                   {1, "double"},
                                                                   {2, "double"},
                                                                   {3, "double"},
                                                                 });
  EXPECT_TRUE(ptx_equal(cuda_source, expected));
}

CUDF_TEST_PROGRAM_MAIN()
