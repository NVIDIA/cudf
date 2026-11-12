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
