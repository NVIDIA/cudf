/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf.examples.deployment;

import ai.rapids.cudf.ColumnVector;

/**
 * Minimal cuDF Java workload used by the deployment-modes examples.
 *
 * <p>The very first reference to any cuDF JNI class triggers
 * {@code NativeDepsLoader.loadNativeDeps()}, which is exactly what this
 * example wants to time. We deliberately keep the actual GPU work tiny so
 * that the dominant cost in {@code first_call_ms} is the native-library
 * load itself.</p>
 *
 * <p>Output is a single line beginning with {@code Workload Result:} so it
 * is trivially grep-able from the driver script.</p>
 */
public final class SimpleWorkload {
  private SimpleWorkload() {}

  public static void main(String[] args) {
    long t0 = System.currentTimeMillis();
    long rows;
    try (ColumnVector cv = ColumnVector.fromInts(1, 2, 3, 4, 5)) {
      rows = cv.getRowCount();
    }
    long firstCallMs = System.currentTimeMillis() - t0;
    System.out.printf("Workload Result: rows=%d  first_call_ms=%d%n", rows, firstCallMs);
  }
}
