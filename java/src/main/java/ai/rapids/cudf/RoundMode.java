/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * Rounding modes supported by decimal division.
 * HALF_UP : Round towards the nearest neighbor. If both neighbors are equidistant, round away
 * from zero.
 * HALF_EVEN : Round towards the nearest neighbor. If both neighbors are equidistant, round
 * towards the even neighbor.
 */
public enum RoundMode {
  HALF_UP(0),
  HALF_EVEN(1);

  final int nativeId;

  RoundMode(int nativeId) { this.nativeId = nativeId; }
}
