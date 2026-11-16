/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LtoFragmentTest extends CudfTestBase {
  @Test
  void testCompileCudaToLtoIr() {
    String source = String.join("\n",
        "#include <cudf/udf_includes.hpp>",
        "#pragma nv_hdrstop",
        "extern \"C\" __device__ cudf::errc add_one(int32_t* output, int32_t input)",
        "{",
        "  *output = input + 1;",
        "  return cudf::errc::SUCCESS;",
        "}");
    LtoFragment fragment = LtoFragment.compileCudaToLtoIr(source);
    assertEquals(LtoBinaryType.LTO_IR, fragment.getBinaryType());
    assertTrue(fragment.getData().length > 0);
    try (ColumnVector input = ColumnVector.fromBoxedInts(2, null, -3);
         ColumnVector result = input.transform(new TransformUDF("add_one", fragment), DType.INT32);
         ColumnVector expected = ColumnVector.fromBoxedInts(3, null, -2)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testCompileWithHeadersAndOptions(@TempDir Path includeDirectory) throws IOException {
    Files.write(includeDirectory.resolve("increment.cuh"),
        "constexpr int increment = 7;\n".getBytes(StandardCharsets.UTF_8));
    String source = String.join("\n",
        "#include <cudf/errc.hpp>",
        "#pragma nv_hdrstop",
        "#include \"options.cuh\"",
        "static_assert(__cplusplus == 201703L, \"C++17 must be selected\");",
        "extern \"C\" __device__ cudf::errc add_increment(int* output, int input)",
        "{",
        "  *output = input + increment;",
        "  return cudf::errc::SUCCESS;",
        "}");
    LtoFragment fragment = LtoFragment.compileCudaToLtoIr(source,
        new String[]{"options.cuh"}, new String[]{"#include <increment.cuh>\n"},
        new String[]{includeDirectory.toString()}, false, "java_udf_options", "c++17");
    assertEquals(LtoBinaryType.LTO_IR, fragment.getBinaryType());
    try (ColumnVector input = ColumnVector.fromInts(2, -3);
         ColumnVector result = input.transform(new TransformUDF("add_increment", fragment),
             DType.INT32);
         ColumnVector expected = ColumnVector.fromInts(9, 4)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testCompileRejectsInvalidSource() {
    assertThrows(CudfException.class,
        () -> LtoFragment.compileCudaToLtoIr("#error Deliberate compilation failure\n"));
  }

  @Test
  void testCompileRejectsMismatchedHeaders() {
    assertThrows(IllegalArgumentException.class, () -> LtoFragment.compileCudaToLtoIr("",
        new String[]{"missing.cuh"}, new String[0], new String[0],
        false, "java_udf_invalid_headers", "c++20"));
  }
}
