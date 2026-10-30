/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */


package ai.rapids.cudf;

/**
 * A table along with some metadata about the table. This is typically returned when
 * reading data from an input file where the metadata can be important.
 */
public class TableWithMeta implements AutoCloseable {
  private long handle;
  private NestedChildren children = null;

  public static class NestedChildren {
    private final String[] names;
    private final NestedChildren[] children;

    private NestedChildren(String[] names, NestedChildren[] children) {
      this.names = names;
      this.children = children;
    }

    public String[] getNames() {
      return names;
    }

    public NestedChildren getChild(int i) {
      return children[i];
    }
    public boolean isChildNested(int i) {
      return (getChild(i) != null);
    }

    @Override
    public String toString() {
      StringBuilder sb = new StringBuilder();
      sb.append("{");
      if (names != null) {
        for (int i = 0; i < names.length; i++) {
          if (i != 0) {
            sb.append(", ");
          }
          sb.append(names[i]);
          sb.append(": ");
          if (children != null) {
            sb.append(children[i]);
          }
        }
      }
      sb.append("}");
      return sb.toString();
    }
  }

  TableWithMeta(long handle) {
    this.handle = handle;
  }

  /**
   * Get the table out of this metadata. Note that this can only be called once. Later calls
   * will return a null.
   */
  public Table releaseTable() {
    long[] ptr = releaseTable(handle);
    if (ptr == null || ptr.length == 0) {
      return null;
    } else {
      return Table.fromRowCountAndColumns(ptr);
    }
  }

  private static class ChildAndOffset {
    public NestedChildren child;
    public int newOffset;
  }

  private ChildAndOffset unflatten(int startOffset, String[] flatNames, int[] flatCounts) {
    ChildAndOffset ret = new ChildAndOffset();
    int length = flatCounts[startOffset];
    if (length == 0) {
      ret.newOffset = startOffset + 1;
      return ret;
    } else {
      String[] names = new String[length];
      NestedChildren[] children = new NestedChildren[length];
      int currentOffset = startOffset + 1;
      for (int i = 0; i < length; i++) {
        names[i] = flatNames[currentOffset];
        ChildAndOffset tmp = unflatten(currentOffset, flatNames, flatCounts);
        children[i] = tmp.child;
        currentOffset = tmp.newOffset;
      }
      ret.newOffset = currentOffset;
      ret.child = new NestedChildren(names, children);
      return ret;
    }
  }

  NestedChildren getChildren() {
    if (children == null) {
      int[] flatCount = getFlattenedChildCounts(handle);
      String[] flatNames = getFlattenedColumnNames(handle);
      ChildAndOffset tmp = unflatten(0, flatNames, flatCount);
      children = tmp.child;
      if (children == null) {
        children = new NestedChildren(new String[0], new NestedChildren[0]);
      }
    }
    return children;
  }

  /**
   * Get the names of the top level columns. In the future new APIs can be added to get
   * names of child columns.
   */
  public String[] getColumnNames() {
    return getChildren().getNames();
  }

  public NestedChildren getChild(int i) {
    return getChildren().getChild(i);
  }

  public boolean isChildNested(int i) {
    return getChildren().isChildNested(i);
  }

  @Override
  public void close() {
    if (handle != 0) {
      close(handle);
      handle = 0;
    }
  }

  private static native void close(long handle);

  private static native long[] releaseTable(long handle);

  private static native String[] getFlattenedColumnNames(long handle);

  private static native int[] getFlattenedChildCounts(long handle);
}
