/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * The unit to which timestamps are rounded.
 * Used by {@link ColumnView#dateTimeCeil}, {@link ColumnView#dateTimeFloor} and
 * {@link ColumnView#dateTimeRound}.
 */
public enum DateTimeRoundingFrequency {
  DAY(0),
  HOUR(1),
  MINUTE(2),
  SECOND(3),
  MILLISECOND(4),
  MICROSECOND(5),
  NANOSECOND(6);

  final int id;
  DateTimeRoundingFrequency(int id) {
    this.id = id;
  }

  public int getNativeId() {
    return id;
  }
}
