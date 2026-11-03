/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * This class will house only tests that need to explicitly set non-empty nulls
 */
public class ColumnViewNonEmptyNullsTest extends CudfTestBase {

  private static final HostMemoryAllocator hostMemoryAllocator = DefaultHostMemoryAllocator.get();

  @Test
  void testAndNullReconfigureNulls() {
    try (ColumnVector v0 = ColumnVector.fromBoxedInts(0, 100, null, null, Integer.MIN_VALUE, null);
         ColumnVector v1 = ColumnVector.fromBoxedInts(0, 100, 1, 2, Integer.MIN_VALUE, null);
         ColumnVector intResult = v1.mergeAndSetValidity(BinaryOp.BITWISE_AND, v0);
         ColumnVector v2 = ColumnVector.fromStrings("0", "100", "1", "2", "MIN_VALUE", "3");
         ColumnVector v3 = v0.mergeAndSetValidity(BinaryOp.BITWISE_AND, v1, v2);
         ColumnVector stringResult = v2.mergeAndSetValidity(BinaryOp.BITWISE_AND, v0, v1);
         ColumnVector stringExpected = ColumnVector.fromStrings("0", "100", null, null, "MIN_VALUE", null);
         ColumnVector noMaskResult = v2.mergeAndSetValidity(BinaryOp.BITWISE_AND)) {
      assertColumnsAreEqual(v0, intResult);
      assertColumnsAreEqual(v0, v3);
      assertColumnsAreEqual(stringExpected, stringResult);
      assertColumnsAreEqual(v2, noMaskResult);
    }
  }

  @Test
  void testOrNullReconfigureNulls() {
    try (ColumnVector v0 = ColumnVector.fromBoxedInts(0, 100, null, null, Integer.MIN_VALUE, null);
         ColumnVector v1 = ColumnVector.fromBoxedInts(0, 100, 1, 2, Integer.MIN_VALUE, null);
         ColumnVector v2 = ColumnVector.fromBoxedInts(0, 100, 1, 2, Integer.MIN_VALUE, Integer.MAX_VALUE);
         ColumnVector intResultV0 = v1.mergeAndSetValidity(BinaryOp.BITWISE_OR, v0);
         ColumnVector intResultV0V1 = v1.mergeAndSetValidity(BinaryOp.BITWISE_OR, v0, v1);
         ColumnVector intResultMulti = v1.mergeAndSetValidity(BinaryOp.BITWISE_OR, v0, v0, v1, v1, v0, v1, v0);
         ColumnVector intResultv0v1v2 = v2.mergeAndSetValidity(BinaryOp.BITWISE_OR, v0, v1, v2);
         ColumnVector v3 = ColumnVector.fromStrings("0", "100", "1", "2", "MIN_VALUE", "3");
         ColumnVector stringResult = v3.mergeAndSetValidity(BinaryOp.BITWISE_OR, v0, v1);
         ColumnVector stringExpected = ColumnVector.fromStrings("0", "100", "1", "2", "MIN_VALUE", null);
         ColumnVector noMaskResult = v3.mergeAndSetValidity(BinaryOp.BITWISE_OR)) {
      assertColumnsAreEqual(v0, intResultV0);
      assertColumnsAreEqual(v1, intResultV0V1);
      assertColumnsAreEqual(v1, intResultMulti);
      assertColumnsAreEqual(v2, intResultv0v1v2);
      assertColumnsAreEqual(stringExpected, stringResult);
      assertColumnsAreEqual(v3, noMaskResult);
    }
  }

  @Test
  void testMergeWithNoColumnsIsNoOp() {
    try (ColumnVector v0 = ColumnVector.fromBoxedInts(0, 100, null, null, Integer.MIN_VALUE, null);
         ColumnVector result = v0.mergeAndSetValidity(BinaryOp.BITWISE_AND)) {
      assertSame(v0, result);
      assertEquals(3, result.getNullCount());
    }
  }

  /**
   * The caller needs to make sure to close the returned ColumnView
   */
  private ColumnView[] getColumnViewWithNonEmptyNulls() {
    List<Integer> list0 = Arrays.asList(1, 2, 3);
    List<Integer> list1 = Arrays.asList(4, 5, null);
    List<Integer> list2 = Arrays.asList(7, 8, 9);
    List<Integer> list3 = null;
    ColumnVector input = ColumnVectorTest.makeListsColumn(DType.INT32, list0, list1, list2, list3);
    // Modify the validity buffer
    BaseDeviceMemoryBuffer dmb = input.getDeviceBufferFor(BufferType.VALIDITY);
    try (HostMemoryBuffer newValidity = hostMemoryAllocator.allocate(64)) {
      newValidity.copyFromDeviceBuffer(dmb);
      BitVectorHelper.setNullAt(newValidity, 1);
      dmb.copyFromHostBuffer(newValidity);
    }
    try (HostColumnVector hostColumnVector = input.copyToHost()) {
      assert (hostColumnVector.isNull(1));
      assert (hostColumnVector.isNull(3));
    }
    try (ColumnVector expectedOffsetsBeforePurge = ColumnVector.fromInts(0, 3, 6, 9, 9)) {
      ColumnView offsetsCvBeforePurge = input.getListOffsetsView();
      assertColumnsAreEqual(expectedOffsetsBeforePurge, offsetsCvBeforePurge);
    }
    ColumnView colWithNonEmptyNulls = new ColumnView(input.type, input.rows, Optional.of(2L), dmb,
        input.getDeviceBufferFor(BufferType.OFFSET), input.getChildColumnViews());
    assertEquals(2, colWithNonEmptyNulls.nullCount);
    return new ColumnView[]{input, colWithNonEmptyNulls};
  }

  @Test
  void testPurgeNonEmptyNullsList() {
    ColumnView[] values = getColumnViewWithNonEmptyNulls();
    try (ColumnView colWithNonEmptyNulls = values[1];
         ColumnView input = values[0];
         // purge non-empty nulls
         ColumnView colWithEmptyNulls = colWithNonEmptyNulls.purgeNonEmptyNulls();
         ColumnVector expectedOffsetsAfterPurge = ColumnVector.fromInts(0, 3, 3, 6, 6);
         ColumnView offsetsCvAfterPurge = colWithEmptyNulls.getListOffsetsView()) {
      assertTrue(colWithNonEmptyNulls.hasNonEmptyNulls());
      assertColumnsAreEqual(expectedOffsetsAfterPurge, offsetsCvAfterPurge);
      assertFalse(colWithEmptyNulls.hasNonEmptyNulls());
    }
  }

  @Test
  void testPurgeNonEmptyNullsStruct() {
    ColumnView[] values = getColumnViewWithNonEmptyNulls();
    try (ColumnView listCol = values[1];
         ColumnView input = values[0];
         ColumnView stringsCol = ColumnVector.fromStrings("A", "col", "of", "Strings");
         ColumnView structView = ColumnView.makeStructView(stringsCol, listCol);
         ColumnView structWithEmptyNulls = structView.purgeNonEmptyNulls();
         ColumnView newListChild = structWithEmptyNulls.getChildColumnView(1);
         ColumnVector expectedOffsetsAfterPurge = ColumnVector.fromInts(0, 3, 3, 6, 6);
         ColumnView offsetsCvAfterPurge = newListChild.getListOffsetsView()) {
      assertColumnsAreEqual(expectedOffsetsAfterPurge, offsetsCvAfterPurge);
      assertFalse(newListChild.hasNonEmptyNulls());
    }
  }

  @Test
  void testMergeAndSetValidityStructPropagatesToChildren() {
    try (ColumnVector c0 = ColumnVector.fromInts(1, 2, 3, 4, 5);
         ColumnVector c1 = ColumnVector.fromInts(10, 20, 30, 40, 50);
         ColumnVector struct = ColumnVector.makeStruct(c0, c1);
         ColumnVector mask = ColumnVector.fromBoxedInts(0, 0, null, null, 0);
         ColumnVector merged = struct.mergeAndSetValidity(BinaryOp.BITWISE_AND, mask);
         HostColumnVector hostMerged = merged.copyToHost()) {
      assertEquals(2, hostMerged.getNullCount(), "parent null count");
      assertFalse(hostMerged.isNull(0));
      assertFalse(hostMerged.isNull(1));
      assertTrue(hostMerged.isNull(2));
      assertTrue(hostMerged.isNull(3));
      assertFalse(hostMerged.isNull(4));

      // Each child should have the same null mask as the parent.
      assertEquals(2, hostMerged.getNumChildren());
      for (int i = 0; i < hostMerged.getNumChildren(); i++) {
        HostColumnVectorCore child = hostMerged.getChildColumnView(i);
        assertEquals(2, child.getNullCount(), "child " + i + " null count");
        assertFalse(child.isNull(0), "child " + i + " row 0");
        assertFalse(child.isNull(1), "child " + i + " row 1");
        assertTrue(child.isNull(2),  "child " + i + " row 2");
        assertTrue(child.isNull(3),  "child " + i + " row 3");
        assertFalse(child.isNull(4), "child " + i + " row 4");
      }
    }
  }

  @Test
  void testMergeAndSetValidityStructOfListPropagatesAndPurges() {
    HostColumnVector.DataType intType  = new HostColumnVector.BasicType(true, DType.INT32);
    HostColumnVector.DataType listType = new HostColumnVector.ListType(true, intType);
    // Inner INT has 11 elements grouped into 5 lists.
    try (ColumnVector listChild = ColumnVector.fromLists(listType,
             Arrays.asList(1, 2),
             Arrays.asList(3, 4, 5),
             Arrays.asList(6),  // will be masked null.
             Arrays.asList(7, 8, 9, 10),  // will be masked null.
             Arrays.asList(11));
         ColumnVector struct = ColumnVector.makeStruct(listChild);
         ColumnVector mask = ColumnVector.fromBoxedInts(0, 0, null, null, 0);
         ColumnVector merged = struct.mergeAndSetValidity(BinaryOp.BITWISE_AND, mask);
         HostColumnVector hostMerged = merged.copyToHost()) {
      assertEquals(2, hostMerged.getNullCount(), "parent null count");
      assertFalse(hostMerged.isNull(0));
      assertFalse(hostMerged.isNull(1));
      assertTrue(hostMerged.isNull(2));
      assertTrue(hostMerged.isNull(3));
      assertFalse(hostMerged.isNull(4));

      // The LIST child should have the same null mask as the parent.
      assertEquals(1, hostMerged.getNumChildren());
      HostColumnVectorCore listView = hostMerged.getChildColumnView(0);
      assertEquals(2, listView.getNullCount(), "list child null count");
      assertFalse(listView.isNull(0));
      assertFalse(listView.isNull(1));
      assertTrue(listView.isNull(2));
      assertTrue(listView.isNull(3));
      assertFalse(listView.isNull(4));

      // The LIST's offsets should be purged. Rows 2 and 3 collapse
      // so the inner INT should have only 6 elements.
      assertEquals(1, listView.getNumChildren());
      HostColumnVectorCore intGrandchild = listView.getChildColumnView(0);
      assertEquals(6, intGrandchild.getRowCount(), "purged inner row count");
      int[] expectedInner = {1, 2, 3, 4, 5, 11};
      for (int i = 0; i < expectedInner.length; i++) {
        assertFalse(intGrandchild.isNull(i), "inner " + i);
        assertEquals(expectedInner[i], intGrandchild.getInt(i), "inner " + i);
      }
      HostMemoryBuffer listOffsets = listView.getOffsets();
      int[] expectedOffsets = {0, 2, 5, 5, 5, 6};
      for (int i = 0; i < expectedOffsets.length; i++) {
        assertEquals(expectedOffsets[i], listOffsets.getInt(i * 4L), "offset " + i);
      }
    }
  }

  @Test
  void testMergeAndSetValidityStructAndWithAllValidMask() {
    try (ColumnVector child = ColumnVector.fromInts(1, 2, 3, 4, 5);
         ColumnVector base = ColumnVector.makeStruct(child);
         ColumnVector baseMask = ColumnVector.fromBoxedInts(1, 1, null, 1, 1);
         ColumnVector nullableBase = base.mergeAndSetValidity(BinaryOp.BITWISE_AND, baseMask);
         ColumnVector allValidMask0 = ColumnVector.fromInts(1, 1, 1, 1, 1);
         ColumnVector allValidMask1 = ColumnVector.fromInts(2, 2, 2, 2, 2);
         ColumnVector merged = nullableBase.mergeAndSetValidity(
             BinaryOp.BITWISE_AND, allValidMask0, allValidMask1);
         HostColumnVector hostMerged = merged.copyToHost()) {
      assertEquals(1, hostMerged.getNullCount(), "parent null count");
      assertFalse(hostMerged.isNull(0));
      assertFalse(hostMerged.isNull(1));
      assertTrue(hostMerged.isNull(2));
      assertFalse(hostMerged.isNull(3));
      assertFalse(hostMerged.isNull(4));

      assertEquals(1, hostMerged.getNumChildren());
      HostColumnVectorCore childView = hostMerged.getChildColumnView(0);
      assertEquals(1, childView.getNullCount(), "child null count");
      assertFalse(childView.isNull(0));
      assertFalse(childView.isNull(1));
      assertTrue(childView.isNull(2));
      assertFalse(childView.isNull(3));
      assertFalse(childView.isNull(4));
    }
  }

  @Test
  void testMergeAndSetValidityStructOrWithAllValidMask() {
    try (ColumnVector child = ColumnVector.fromInts(1, 2, 3, 4, 5);
         ColumnVector base = ColumnVector.makeStruct(child);
         ColumnVector baseMask = ColumnVector.fromBoxedInts(1, 1, null, 1, 1);
         ColumnVector nullableBase = base.mergeAndSetValidity(BinaryOp.BITWISE_AND, baseMask);
         ColumnVector nullableMask = ColumnVector.fromBoxedInts(1, null, 1, 1, 1);
         ColumnVector allValidMask = ColumnVector.fromInts(1, 1, 1, 1, 1);
         ColumnVector merged = nullableBase.mergeAndSetValidity(
             BinaryOp.BITWISE_OR, nullableMask, allValidMask);
         HostColumnVector hostMerged = merged.copyToHost()) {
      assertEquals(1, hostMerged.getNullCount(), "parent null count");
      assertFalse(hostMerged.isNull(0));
      assertFalse(hostMerged.isNull(1));
      assertTrue(hostMerged.isNull(2));
      assertFalse(hostMerged.isNull(3));
      assertFalse(hostMerged.isNull(4));

      assertEquals(1, hostMerged.getNumChildren());
      HostColumnVectorCore childView = hostMerged.getChildColumnView(0);
      assertEquals(1, childView.getNullCount(), "child null count");
      assertFalse(childView.isNull(0));
      assertFalse(childView.isNull(1));
      assertTrue(childView.isNull(2));
      assertFalse(childView.isNull(3));
      assertFalse(childView.isNull(4));
    }
  }
}
