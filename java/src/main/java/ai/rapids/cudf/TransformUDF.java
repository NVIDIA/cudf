/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import java.util.Objects;

/**
 * An explicitly named transform function and the modules defining it.
 *
 * The CUDF ABI is {@code cudf::errc symbol(Indices... indices, Outputs*... outputs, Inputs... inputs)}.
 * The output pointer precedes all inputs.
 * When indexed, a logical row index precedes the output pointer.
 * Indices use {@code cudf::size_type} and are never optional.
 * Null-aware calls wrap column and output values in {@code cuda::std::optional}.
 *
 * Modules may export multiple symbols and reference symbols in other supplied modules.
 * This descriptor retains its fragments and their module byte arrays.
 */
public final class TransformUDF {
  private final String symbol;
  private final LtoFragment[] fragments;
  private final boolean nullAware;
  private final boolean indexed;

  /**
   * Create an unindexed UDF using ordinary input and output values.
   * @param symbol exact exported device symbol, expressible as a CUDA identifier
   * @param fragments compiled modules defining the function and its dependencies
   */
  public TransformUDF(String symbol, LtoFragment... fragments) {
    this(symbol, false, false, fragments);
  }

  /**
   * Create a UDF with explicit null-awareness and indexing.
   * @param symbol exact exported device symbol, expressible as a CUDA identifier
   * @param nullAware whether column and output values use {@code cuda::std::optional}
   * @param indexed whether a logical row index precedes all other arguments
   * @param fragments compiled modules defining the function and its dependencies
   */
  public TransformUDF(String symbol, boolean nullAware, boolean indexed,
                      LtoFragment... fragments) {
    this.symbol = Objects.requireNonNull(symbol, "symbol");
    Objects.requireNonNull(fragments, "fragments");
    this.fragments = fragments.clone();
    for (LtoFragment fragment : this.fragments) {
      Objects.requireNonNull(fragment, "fragment");
    }
    this.nullAware = nullAware;
    this.indexed = indexed;
  }

  /**
   * Returns the device symbol associated with this UDF.
   * @return the device symbol.
   */
  public String getSymbol() {
    return symbol;
  }

  /**
   * Returns a copy of the fragment array in link order.
   * @return the modules defining the function and its dependencies, in link order
   */
  public LtoFragment[] getFragments() {
    return fragments.clone();
  }

  /**
   * Returns whether column and output values use {@code cuda::std::optional}.
   * @return whether column and output values use {@code cuda::std::optional}
   */
  public boolean isNullAware() {
    return nullAware;
  }

  /**
   * Returns whether a logical row index precedes all other arguments.
   * @return whether a logical row index precedes all other arguments
   */
  public boolean isIndexed() {
    return indexed;
  }
}
