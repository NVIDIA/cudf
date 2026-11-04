/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/parser.hpp"

#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/utilities/error.hpp>

#include <initializer_list>
#include <string>

struct JitParseTest : public ::testing::Test {};

TEST_F(JitParseTest, PTXNoFunction)
{
  EXPECT_THROW(cudf::jit::parse_single_function_ptx(".visible .entry kernel() { ret; }"),
               cudf::logic_error);
}

TEST_F(JitParseTest, PTXRenameOnly)
{
  std::string const source = R"ptx(
.version 8.0
.target sm_80
.address_size 64
// .func fake() {} original$fn
/* .func another_fake() {} */
.visible .func (.param .b32 status) original$fn(
  .param .u64 original$fn_param_0,
  .param .align 16 .b8 value[16])
{
  .reg .b64 %rd;
  .pragma "original$fn";
  ld.param.u64 %rd, [value+8];
  st.param.b32 [status], 0;
  ret;
}
.func helper() {
  call original$fn;
  ret;
}
)ptx";
  auto expected            = source;
  auto name                = expected.find("original$fn(");
  expected.replace(name, std::string{"original$fn"}.size(), "transform");
  name = expected.find("call original$fn;");
  expected.replace(name + 5, std::string{"original$fn"}.size(), "transform");
  EXPECT_EQ(cudf::jit::parse_single_function_ptx(source), expected);
}

TEST_F(JitParseTest, PTXWithoutReturnParameters)
{
  EXPECT_EQ(cudf::jit::parse_single_function_ptx(".visible .func old() { ret; }", "target"),
            ".visible .func target() { ret; }");
}

TEST_F(JitParseTest, PTXAlreadyNamedTransform)
{
  std::string const source = ".visible .func transform() { ret; }";
  EXPECT_EQ(cudf::jit::parse_single_function_ptx(source), source);
}

TEST_F(JitParseTest, PTXMalformedFunction)
{
  for (auto const* source : {".func", ".func (", ".func (.param .b32 status)", ".func ()"}) {
    EXPECT_THROW(cudf::jit::parse_single_function_ptx(source), cudf::logic_error);
  }
  EXPECT_THROW(cudf::jit::parse_single_function_ptx("/* unterminated"), cudf::logic_error);
  EXPECT_THROW(cudf::jit::parse_single_function_ptx(".func \"unterminated"), cudf::logic_error);
}

TEST_F(JitParseTest, PTXForwardDeclaration)
{
  std::string const source =
    ".visible .func original();\n.func helper() { ret; }\n"
    ".visible .func original() { call helper; ret; }";
  std::string const expected =
    ".visible .func transform();\n.func helper() { ret; }\n"
    ".visible .func transform() { call helper; ret; }";
  EXPECT_EQ(cudf::jit::parse_single_function_ptx(source), expected);
}

CUDF_TEST_PROGRAM_MAIN()
