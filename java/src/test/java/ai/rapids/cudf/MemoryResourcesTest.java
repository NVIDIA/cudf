/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.function.Supplier;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static ai.rapids.cudf.AssertUtils.assertTablesAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks that the overloads taking a stream and {@link MemoryResources} allocate their results
 * from the output resource and their scratch memory from the temporary resource, and that they
 * match the deprecated overloads.
 */
@SuppressWarnings("deprecation")
public class MemoryResourcesTest extends CudfTestBase {
  private RmmTrackingResourceAdaptor<RmmCudaMemoryResource> output;
  private RmmTrackingResourceAdaptor<RmmCudaMemoryResource> temporary;
  private MemoryResources mr;
  private Cuda.Stream stream;

  @BeforeEach
  void setUp() {
    output = new RmmTrackingResourceAdaptor<>(new RmmCudaMemoryResource(), 256);
    temporary = new RmmTrackingResourceAdaptor<>(new RmmCudaMemoryResource(), 256);
    mr = new MemoryResources(output, temporary);
    stream = new Cuda.Stream(true);
  }

  @AfterEach
  void tearDown() {
    stream.close();
    output.close();
    temporary.close();
  }

  private long beforeRun() {
    Cuda.DEFAULT_STREAM.sync();
    return Rmm.getTotalBytesAllocated();
  }

  /**
   * Check that the result of an operation on the test stream is the only memory left allocated,
   * and that it came from the output resource.
   */
  private void afterRun(long currentBefore) {
    stream.sync();
    assertEquals(currentBefore, Rmm.getTotalBytesAllocated(),
        "result was allocated from the current device resource");
    assertTrue(output.getTotalBytesAllocated() > 0, "result was not allocated from output");
    assertEquals(0, temporary.getTotalBytesAllocated(), "temporary memory was not freed");
  }

  private void checkColumn(Supplier<ColumnVector> expected, Supplier<ColumnVector> actual) {
    try (ColumnVector e = expected.get()) {
      long currentBefore = beforeRun();
      try (ColumnVector a = actual.get()) {
        afterRun(currentBefore);
        assertColumnsAreEqual(e, a);
      }
    }
    assertEquals(0, output.getTotalBytesAllocated());
  }

  private void checkTable(Supplier<Table> expected, Supplier<Table> actual) {
    try (Table e = expected.get()) {
      long currentBefore = beforeRun();
      try (Table a = actual.get()) {
        afterRun(currentBefore);
        assertTablesAreEqual(e, a);
      }
    }
    assertEquals(0, output.getTotalBytesAllocated());
  }

  @Test
  void testDefaults() {
    assertNull(MemoryResources.CURRENT.getOutput());
    assertNull(MemoryResources.CURRENT.getTemporary());
    assertEquals(0, MemoryResources.CURRENT.getOutputHandle());
    assertEquals(0, MemoryResources.CURRENT.getTemporaryHandle());
    MemoryResources outputOnly = new MemoryResources(output);
    assertSame(output, outputOnly.getOutput());
    assertNull(outputOnly.getTemporary());
    assertEquals(output.getHandle(), outputOnly.getOutputHandle());
    assertEquals(0, outputOnly.getTemporaryHandle());
    assertSame(temporary, mr.getTemporary());
    assertEquals(temporary.getHandle(), mr.getTemporaryHandle());
  }

  @Test
  void testClosedResource() {
    RmmCudaMemoryResource closed = new RmmCudaMemoryResource();
    closed.close();
    MemoryResources closedMr = new MemoryResources(closed);
    try (ColumnVector cv = ColumnVector.fromInts(1, 2, 3)) {
      assertThrows(IllegalStateException.class, () -> cv.isNull(stream, closedMr));
    }
  }

  @Test
  void testOutputOnly() {
    MemoryResources outputOnly = new MemoryResources(output);
    try (ColumnVector cv = ColumnVector.fromBoxedInts(1, null, 3)) {
      checkColumn(cv::isNull, () -> cv.isNull(stream, outputOnly));
    }
  }

  @Test
  void testNullAndNanPredicates() {
    try (ColumnVector cv = ColumnVector.fromBoxedDoubles(1.0, null, Double.NaN, 4.0)) {
      checkColumn(cv::isNull, () -> cv.isNull(stream, mr));
      checkColumn(cv::isNotNull, () -> cv.isNotNull(stream, mr));
      checkColumn(cv::isNan, () -> cv.isNan(stream, mr));
      checkColumn(cv::isNotNan, () -> cv.isNotNan(stream, mr));
    }
  }

  @Test
  void testReplaceNulls() {
    try (ColumnVector cv = ColumnVector.fromBoxedInts(1, null, 3, null);
         ColumnVector replacements = ColumnVector.fromInts(10, 20, 30, 40);
         Scalar zero = Scalar.fromInt(0)) {
      checkColumn(() -> cv.replaceNulls(zero), () -> cv.replaceNulls(zero, stream, mr));
      checkColumn(() -> cv.replaceNulls(replacements),
          () -> cv.replaceNulls(replacements, stream, mr));
      checkColumn(() -> cv.replaceNulls(ReplacePolicy.PRECEDING),
          () -> cv.replaceNulls(ReplacePolicy.PRECEDING, stream, mr));
    }
  }

  @Test
  void testIfElse() {
    try (ColumnVector pred = ColumnVector.fromBooleans(true, false, true);
         ColumnVector t = ColumnVector.fromInts(1, 2, 3);
         ColumnVector f = ColumnVector.fromInts(4, 5, 6);
         Scalar ts = Scalar.fromInt(7);
         Scalar fs = Scalar.fromInt(8)) {
      checkColumn(() -> pred.ifElse(t, f), () -> pred.ifElse(t, f, stream, mr));
      checkColumn(() -> pred.ifElse(t, fs), () -> pred.ifElse(t, fs, stream, mr));
      checkColumn(() -> pred.ifElse(ts, f), () -> pred.ifElse(ts, f, stream, mr));
      checkColumn(() -> pred.ifElse(ts, fs), () -> pred.ifElse(ts, fs, stream, mr));
    }
  }

  @Test
  void testUnaryAndBinaryOps() {
    try (ColumnVector lhs = ColumnVector.fromInts(1, 2, 3);
         ColumnVector rhs = ColumnVector.fromInts(4, 5, 6);
         Scalar s = Scalar.fromInt(10)) {
      checkColumn(() -> lhs.unaryOp(UnaryOp.ABS), () -> lhs.unaryOp(UnaryOp.ABS, stream, mr));
      checkColumn(() -> lhs.binaryOp(BinaryOp.ADD, rhs, DType.INT64),
          () -> lhs.binaryOp(BinaryOp.ADD, rhs, DType.INT64, stream, mr));
      checkColumn(() -> lhs.binaryOp(BinaryOp.MUL, s, DType.INT32),
          () -> lhs.binaryOp(BinaryOp.MUL, s, DType.INT32, stream, mr));
      checkColumn(() -> s.binaryOp(BinaryOp.SUB, rhs, DType.INT32),
          () -> s.binaryOp(BinaryOp.SUB, rhs, DType.INT32, stream, mr));
    }
  }

  @Test
  void testStructBinaryOps() {
    try (ColumnVector a = ColumnVector.fromInts(1, 2, 3);
         ColumnVector b = ColumnVector.fromInts(3, 2, 1);
         ColumnVector lhs = ColumnVector.makeStruct(a);
         ColumnVector rhs = ColumnVector.makeStruct(b);
         ColumnVector two = ColumnVector.fromInts(2);
         Scalar s = Scalar.structFromColumnViews(two)) {
      checkColumn(() -> lhs.binaryOp(BinaryOp.LESS, rhs, DType.BOOL8),
          () -> lhs.binaryOp(BinaryOp.LESS, rhs, DType.BOOL8, stream, mr));
      checkColumn(() -> lhs.binaryOp(BinaryOp.LESS, s, DType.BOOL8),
          () -> lhs.binaryOp(BinaryOp.LESS, s, DType.BOOL8, stream, mr));
      checkColumn(() -> s.binaryOp(BinaryOp.LESS, rhs, DType.BOOL8),
          () -> s.binaryOp(BinaryOp.LESS, rhs, DType.BOOL8, stream, mr));
    }
  }

  @Test
  void testCastTo() {
    try (ColumnVector ints = ColumnVector.fromInts(1, 2, 3);
         ColumnVector bools = ColumnVector.fromBooleans(true, false)) {
      checkColumn(() -> ints.castTo(DType.INT64), () -> ints.castTo(DType.INT64, stream, mr));
      checkColumn(() -> ints.castTo(DType.STRING), () -> ints.castTo(DType.STRING, stream, mr));
      ColumnView[] views = ints.splitAsViews();
      try (ColumnView view = views[0]) {
        checkColumn(() -> view.castTo(DType.INT32), () -> view.castTo(DType.INT32, stream, mr));
      }
      temporary.resetScopedMaxTotalBytesAllocated(0);
      checkColumn(() -> bools.castTo(DType.STRING), () -> bools.castTo(DType.STRING, stream, mr));
      assertTrue(temporary.getScopedMaxTotalBytesAllocated() > 0,
          "the true/false scalars were not allocated from the temporary resource");
    }
  }

  @Test
  void testTableOps() {
    try (Table table = new Table.TestBuilder()
             .column(3, 1, 2, 1)
             .column("c", "a", "b", "a")
             .build();
         Table other = new Table.TestBuilder()
             .column(5, 4)
             .column("e", "d")
             .build();
         ColumnVector mask = ColumnVector.fromBooleans(true, false, true, false);
         ColumnVector map = ColumnVector.fromInts(3, 0, 1);
         ColumnVector scatterMap = ColumnVector.fromInts(2, 0);
         Scalar i = Scalar.fromInt(9);
         Scalar s = Scalar.fromString("z")) {
      OrderByArg[] order = {OrderByArg.asc(0), OrderByArg.desc(1)};
      checkTable(() -> Table.concatenate(table, other),
          () -> Table.concatenate(new Table[]{table, other}, stream, mr));
      checkColumn(() -> table.sortOrder(order), () -> table.sortOrder(order, stream, mr));
      temporary.resetScopedMaxTotalBytesAllocated(0);
      checkTable(() -> table.orderBy(order), () -> table.orderBy(order, stream, mr));
      assertTrue(temporary.getScopedMaxTotalBytesAllocated() > 0,
          "the sort order was not allocated from the temporary resource");
      checkTable(() -> table.filter(mask), () -> table.filter(mask, stream, mr));
      checkTable(() -> table.gather(map), () -> table.gather(map, stream, mr));
      checkTable(() -> table.gather(map, OutOfBoundsPolicy.DONT_CHECK),
          () -> table.gather(map, OutOfBoundsPolicy.DONT_CHECK, stream, mr));
      checkTable(() -> other.scatter(scatterMap, table),
          () -> other.scatter(scatterMap, table, stream, mr));
      Scalar[] row = {i, s};
      checkTable(() -> Table.scatter(row, scatterMap, table),
          () -> Table.scatter(row, scatterMap, table, stream, mr));
    }
  }
}
