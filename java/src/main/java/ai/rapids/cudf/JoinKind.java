/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/**
 * Join semantics to apply when filtering equality-join gather maps of the corresponding kind.
 * See {@link Table#filterJoinGatherMaps} for the input-map contract.
 */
public enum JoinKind {
  /** Retain only row pairs that satisfy the condition. */
  INNER(0),
  /** Retain every left row, using an invalid right index when no pair satisfies the condition. */
  LEFT(1),
  /**
   * Retain passing pairs and every row from both sides, using one invalid opposite-side index
   * for each row with no pair that satisfies the condition.
   */
  FULL(2);

  final int nativeId;

  JoinKind(int nativeId) {
    this.nativeId = nativeId;
  }
}
