/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf.examples.joincache;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ContiguousTable;
import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.GatherMap;
import ai.rapids.cudf.HashJoin;
import ai.rapids.cudf.JoinMatchContext;
import ai.rapids.cudf.Rmm;
import ai.rapids.cudf.RmmAllocationMode;
import ai.rapids.cudf.Scalar;
import ai.rapids.cudf.Table;

import java.util.Arrays;
import java.util.function.LongSupplier;

/**
 * Joins a large fact table against a smaller dimension table and produces the results one
 * partition at a time, timing two ways of doing it.
 *
 * <p>The uncached path counts the total number of matches first and then joins each partition,
 * so every fact row is looked up in the hash table twice. The cached path builds a
 * {@link JoinMatchContext}, which looks up every fact row once and keeps the matches for each
 * partition to reuse.</p>
 *
 * <p>Dimension keys are unique. Fact keys are skewed so that a few keys are very common, like
 * popular products in a sales table.</p>
 */
public final class JoinMatchContextExample {
  private JoinMatchContextExample() {}

  public static void main(String[] args) {
    if (args.length != 4) {
      System.err.println("usage: JoinMatchContextExample buildRows probeRows partitions iterations");
      System.exit(2);
    }
    int buildRows = Integer.parseInt(args[0]);
    int probeRows = Integer.parseInt(args[1]);
    int partitions = Integer.parseInt(args[2]);
    int iterations = Integer.parseInt(args[3]);
    if (buildRows <= 0 || probeRows <= 0 || partitions <= 0 || iterations <= 0) {
      System.err.println("all arguments must be positive");
      System.exit(2);
    }

    Rmm.initialize(RmmAllocationMode.CUDA_ASYNC, null, Cuda.memGetInfo().free * 9 / 10);

    int[] bounds = new int[partitions + 1];
    for (int i = 0; i <= partitions; i++) {
      bounds[i] = (int) ((long) probeRows * i / partitions);
    }

    try (Scalar zero = Scalar.fromInt(0);
         ColumnVector dimKeys = ColumnVector.sequence(zero, buildRows);
         ColumnVector factKeys = skewedKeys(probeRows, buildRows);
         Table dim = new Table(dimKeys);
         Table fact = new Table(factKeys);
         HashJoin hashJoin = new HashJoin(dim, false)) {
      ContiguousTable[] slices = fact.contiguousSplit(Arrays.copyOfRange(bounds, 1, partitions));
      try {
        Result uncached = measure(iterations, () -> {
          long total = fact.innerJoinRowCount(hashJoin);
          long rows = 0;
          for (ContiguousTable slice : slices) {
            rows += countAndClose(slice.getTable().innerJoinGatherMaps(hashJoin));
          }
          return checkTotal("uncached", rows, total);
        });
        Result cached = measure(iterations, () -> {
          long rows = 0;
          try (JoinMatchContext ctx = fact.innerJoinMatchContext(hashJoin)) {
            for (int p = 0; p < partitions; p++) {
              rows += countAndClose(ctx.partitionGatherMaps(bounds[p], bounds[p + 1]));
            }
            return checkTotal("cached", rows, ctx.getOutputSize());
          }
        });

        System.out.printf("Inner join: dimension=%,d rows (unique keys), "
            + "fact=%,d rows (skewed keys), partitions=%d%n", buildRows, probeRows, partitions);
        System.out.printf("Fact rows matched: %.1f%%%n", 100.0 * cached.rows / probeRows);
        System.out.printf("uncached  median %8.2f ms   min %8.2f ms   rows %,d%n",
            uncached.medianMs, uncached.minMs, uncached.rows);
        System.out.printf("cached    median %8.2f ms   min %8.2f ms   rows %,d%n",
            cached.medianMs, cached.minMs, cached.rows);
        System.out.printf("speedup=%.2fx%n", uncached.medianMs / cached.medianMs);
        if (uncached.rows != cached.rows) {
          System.err.println("row counts differ between the two paths");
          System.exit(1);
        }
      } finally {
        for (ContiguousTable slice : slices) {
          slice.close();
        }
      }
    }
  }

  // Builds fact keys where a few values are very common and about one in five has no match.
  private static ColumnVector skewedKeys(int rows, int dimRows) {
    try (Scalar zero = Scalar.fromLong(0);
         Scalar multiplier = Scalar.fromLong(2654435761L);
         Scalar twoTo32 = Scalar.fromLong(1L << 32);
         Scalar skew = Scalar.fromDouble(3.0);
         Scalar range = Scalar.fromDouble(2.0 * dimRows);
         // 64-bit row numbers so the multiply below cannot overflow
         ColumnVector rowIds = ColumnVector.sequence(zero, rows);
         // Knuth's multiplicative hash scatters the row numbers like random values,
         // but the same on every run
         ColumnVector hashed = rowIds.mul(multiplier);
         ColumnVector wrapped = hashed.mod(twoTo32);
         // Uniform values between 0 and 1
         ColumnVector uniform = wrapped.trueDiv(twoTo32, DType.FLOAT64);
         // Cubing pushes values toward 0, so small keys become far more common
         ColumnVector skewed = uniform.pow(skew);
         // Keys go up to twice the dimension size, and the upper half has no match
         ColumnVector keys = skewed.mul(range)) {
      return keys.castTo(DType.INT32);
    }
  }

  private static long checkTotal(String path, long rows, long expected) {
    if (rows != expected) {
      throw new IllegalStateException(
          path + ": partitions produced " + rows + " rows but the total size is " + expected);
    }
    return rows;
  }

  private static long countAndClose(GatherMap[] maps) {
    long rows = maps[0].getRowCount();
    for (GatherMap map : maps) {
      map.close();
    }
    return rows;
  }

  private static Result measure(int iterations, LongSupplier run) {
    run.getAsLong();
    Cuda.DEFAULT_STREAM.sync();
    double[] ms = new double[iterations];
    long rows = 0;
    for (int i = 0; i < iterations; i++) {
      long start = System.nanoTime();
      rows = run.getAsLong();
      Cuda.DEFAULT_STREAM.sync();
      ms[i] = (System.nanoTime() - start) / 1e6;
    }
    Arrays.sort(ms);
    return new Result(ms[iterations / 2], ms[0], rows);
  }

  private static final class Result {
    final double medianMs;
    final double minMs;
    final long rows;

    Result(double medianMs, double minMs, long rows) {
      this.medianMs = medianMs;
      this.minMs = minMs;
      this.rows = rows;
    }
  }
}
