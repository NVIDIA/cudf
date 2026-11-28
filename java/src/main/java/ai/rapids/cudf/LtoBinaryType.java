/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/**
 * Compiled device module formats, mirroring native {@code cudf::lto_binary_type}.
 */
public enum LtoBinaryType {
  /** LTO-IR Binary. */
  LTO_IR(0),
  /** FATBIN Binary. */
  FATBIN(1),
  /** PTX source. */
  PTX(2);

  private final int nativeId;

  LtoBinaryType(int nativeId) {
    this.nativeId = nativeId;
  }

  /**
   * Return the corresponding native binary type ID.
   * @return the native binary type ID
   */
  public int getNativeId() {
    return nativeId;
  }
}
