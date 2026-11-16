/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import java.util.Objects;

/**
 * A compiled device module to link into a JIT kernel.
 *
 * The byte array is retained and must not be modified while the fragment is in use.
 */
public final class LtoFragment {
  private final LtoBinaryType binaryType;
  private final byte[] data;

  /**
   * Create a fragment retaining the supplied module bytes.
   * @param binaryType binary format of the fragment
   * @param data precompiled module bytes
   */
  public LtoFragment(LtoBinaryType binaryType, byte[] data) {
    this.binaryType = Objects.requireNonNull(binaryType, "binaryType");
    this.data = Objects.requireNonNull(data, "data");
  }

  /**
   * Compile CUDA device source into an LTO IR fragment.
   * Uses C++20, enables automatic precompiled headers, and uses the PCH key
   * {@code cudf_user_fragment}. No additional headers or include directories are supplied.
   *
   * The returned fragment owns its module bytes and can be shared by multiple UDFs:
   * {@code
   * LtoFragment fragment = LtoFragment.compileCudaToLtoIr(source);
   * TransformUDF udf = new TransformUDF("add_one", fragment);
   * try (ColumnVector result = input.transform(udf, DType.INT32)) {
   *   // Use result.
   * }
   * }
   *
   * @param source CUDA device source; include {@code <cudf/udf_includes.hpp>} for common UDF types
   * @return an owning fragment with binary type {@link LtoBinaryType#LTO_IR}
   * @throws NullPointerException if source is null
   * @throws CudfException if compilation fails
   */
  public static LtoFragment compileCudaToLtoIr(String source) {
    return compileCudaToLtoIr(source, new String[0], new String[0], new String[0],
        true, "cudf_user_fragment", "c++20");
  }

  /**
   * Compile CUDA device source into an LTO IR fragment with explicit compilation options.
   * Compiled fragments are cached by source, in-memory headers, ordered include paths, PCH use,
   * and C++ standard. Filesystem headers and their dependencies must remain unchanged while
   * caches may be reused; use versioned include paths when changing filesystem dependencies.
   *
   * When PCH is enabled, headers used to build it must remain unchanged while it may be reused.
   * Place changing headers after {@code #pragma nv_hdrstop} or disable PCH. Source filenames do
   * not isolate PCH reuse: NVRTC may reuse a compatible PCH across filenames. PCH is recommended
   * whenever possible to improve compilation performance.
   *
   * @param source CUDA device source; include {@code <cudf/udf_includes.hpp>} for common UDF types
   * @param includeNames names of additional in-memory headers
   * @param includes matching header contents, in the same order as includeNames
   * @param includePaths filesystem include directories, searched before bundled header directories
   * @param usePch whether to enable automatic precompiled headers on supported NVRTC versions
   * @param pchKey source filename stem used to identify or create the precompiled header
   * @param std C++ language standard passed to NVRTC, such as {@code c++17} or {@code c++20}
   * @return an owning fragment with binary type {@link LtoBinaryType#LTO_IR}
   * @throws NullPointerException if any string, array, or array element is null
   * @throws IllegalArgumentException if header names and contents have different lengths
   * @throws CudfException if compilation fails
   */
  public static LtoFragment compileCudaToLtoIr(String source, String[] includeNames,
      String[] includes, String[] includePaths, boolean usePch, String pchKey, String std) {
    Objects.requireNonNull(source, "source");
    requireNonNullStrings(includeNames, "includeNames");
    requireNonNullStrings(includes, "includes");
    requireNonNullStrings(includePaths, "includePaths");
    Objects.requireNonNull(pchKey, "pchKey");
    Objects.requireNonNull(std, "std");
    if (includeNames.length != includes.length) {
      throw new IllegalArgumentException("Header names and contents must have the same size");
    }
    NativeDepsLoader.loadNativeDeps();
    return new LtoFragment(LtoBinaryType.LTO_IR,
        compileCudaToLtoIrNative(source, includeNames, includes, includePaths, usePch, pchKey, std));
  }

  private static void requireNonNullStrings(String[] values, String name) {
    Objects.requireNonNull(values, name);
    for (String value : values) {
      Objects.requireNonNull(value, name + " element");
    }
  }

  private static native byte[] compileCudaToLtoIrNative(String source, String[] includeNames,
      String[] includes, String[] includePaths, boolean usePch, String pchKey, String std);

  /**
   * Returns the binary format of the fragment.
   * @return the binary format of the fragment
   */
  public LtoBinaryType getBinaryType() {
    return binaryType;
  }

  /**
   * Returns the retained fragment byte array.
   * @return the retained fragment byte array
   */
  public byte[] getData() {
    return data;
  }
}
