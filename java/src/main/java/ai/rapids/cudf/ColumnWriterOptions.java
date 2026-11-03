/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.util.ArrayList;
import java.util.List;

/**
 * Per column settings for writing Parquet/ORC files.
 *
 * The native also uses the same "column_in_metadata" for both Parquet and ORC.
 */
public class ColumnWriterOptions {
  // `isTimestampTypeInt96` is ignored in ORC
  private boolean isTimestampTypeInt96;
  private int precision;
  private boolean isNullable;
  private boolean isMap = false;
  private boolean isBinary = false;
  private String columnName;
  // only for Parquet
  private boolean hasParquetFieldId;
  private int parquetFieldId;

  private ColumnWriterOptions(AbstractStructBuilder builder) {
    this.columnName = builder.name;
    this.isNullable = builder.isNullable;
    this.hasParquetFieldId = builder.hasParquetFieldId;
    this.parquetFieldId = builder.parquetFieldId;
    this.childColumnOptions =
        (ColumnWriterOptions[]) builder.children.toArray(new ColumnWriterOptions[0]);
  }

  // The sentinel value of unknown precision (default value)
  public static int UNKNOWN_PRECISION = -1;

  /**
   * Constructor used for list
   */
  private ColumnWriterOptions(ListBuilder builder) {
    assert(builder.children.size() == 1) : "Lists can only have one child";
    this.columnName = builder.name;
    this.isNullable = builder.isNullable;
    this.hasParquetFieldId = builder.hasParquetFieldId;
    this.parquetFieldId = builder.parquetFieldId;
    // we are adding the child twice even though lists have one child only because the way the cudf
    // has implemented this it requires two children to be set for the list, but it drops the
    // first one. This is something that is a lower priority and might be fixed in future
    this.childColumnOptions =
        new ColumnWriterOptions[]{DUMMY_CHILD, builder.children.get(0)};
  }

  protected ColumnWriterOptions[] childColumnOptions = {};
  protected abstract static class AbstractStructBuilder<T extends AbstractStructBuilder,
      V extends ColumnWriterOptions> extends NestedBuilder<T, V> {
    /**
     * Builder specific to build a Struct meta
     */
    public AbstractStructBuilder(String name, boolean isNullable) {
      super(name, isNullable);
    }

    public AbstractStructBuilder(String name, boolean isNullable, int parquetFieldId) {
      super(name, isNullable, parquetFieldId);
    }

    protected AbstractStructBuilder() {
      super();
    }
  }

  // This child is needed as the first child of a List column meta due to how cudf has been
  // implemented. Cudf drops the first child from the meta if a column is a LIST. This is done
  // this way due to some complications in the parquet reader. There was change to fix this here:
  // https://github.com/NVIDIA/cudf/pull/7461/commits/5ce33b40abb87cc7b76b5efeb0a3a0215f9ef6fb
  // but it was reverted later on here:
  // https://github.com/NVIDIA/cudf/pull/7461/commits/f248eb7265de995a95f998d46d897fb0ae47f53e
  static ColumnWriterOptions DUMMY_CHILD = new ColumnWriterOptions("DUMMY");

  public static abstract class NestedBuilder<T extends NestedBuilder, V extends ColumnWriterOptions> {
    protected List<ColumnWriterOptions> children = new ArrayList<>();
    protected boolean isNullable = true;
    protected String name = "";
    // Parquet structure needs
    protected boolean hasParquetFieldId;
    protected int parquetFieldId;

    /**
     * Builder specific to build a Struct meta
     */
    protected NestedBuilder(String name, boolean isNullable) {
      this.name = name;
      this.isNullable = isNullable;
    }

    protected NestedBuilder(String name, boolean isNullable, int parquetFieldId) {
      this.name = name;
      this.isNullable = isNullable;
      this.hasParquetFieldId = true;
      this.parquetFieldId = parquetFieldId;
    }

    protected NestedBuilder() {}

    protected ColumnWriterOptions withColumn(String name, boolean isNullable) {
      return new ColumnWriterOptions(name, isNullable);
    }

    protected ColumnWriterOptions withColumn(String name, boolean isNullable, int parquetFieldId) {
      return new ColumnWriterOptions(name, isNullable, parquetFieldId);
    }

    protected ColumnWriterOptions withDecimal(String name, int precision,
                                              boolean isNullable) {
      return new ColumnWriterOptions(name, false, precision, isNullable);
    }

    protected ColumnWriterOptions withDecimal(String name, int precision,
                                              boolean isNullable, int parquetFieldId) {
      return new ColumnWriterOptions(name, false, precision, isNullable, parquetFieldId);
    }

    protected ColumnWriterOptions withTimestamp(String name, boolean isInt96,
                                                boolean isNullable) {
      return new ColumnWriterOptions(name, isInt96, UNKNOWN_PRECISION, isNullable);
    }

    protected ColumnWriterOptions withTimestamp(String name, boolean isInt96,
                                                boolean isNullable, int parquetFieldId) {
      return new ColumnWriterOptions(name, isInt96, UNKNOWN_PRECISION, isNullable, parquetFieldId);
    }

    protected ColumnWriterOptions withBinary(String name, boolean isNullable) {
      ColumnWriterOptions opt = listBuilder(name, isNullable)
          // The name here does not matter. It will not be included in the final file
          // This is just to get the metadata to line up properly for the C++ APIs
          .withColumns(false, "BINARY_DATA")
          .build();
      opt.isBinary = true;
      return opt;
    }

    protected ColumnWriterOptions withBinary(String name, boolean isNullable, int parquetFieldId) {
      // The parquet field id must be attached to the outer list (the column the writer emits as
      // BINARY), not to the inner BINARY_DATA child, because the writer drops the inner child when
      // flattening the list-as-binary into a parquet primitive.
      ColumnWriterOptions opt = listBuilder(name, isNullable, parquetFieldId)
          // The name here does not matter. It will not be included in the final file
          // This is just to get the metadata to line up properly for the C++ APIs
          .withColumns(false, "BINARY_DATA")
          .build();
      opt.isBinary = true;
      return opt;
    }

    /**
     * Set the list column meta.
     * Lists should have only one child in ColumnVector, but the metadata expects a
     * LIST column to have two children and the first child to be the
     * {@link ColumnWriterOptions#DUMMY_CHILD}.
     * This is the current behavior in cudf and will change in future
     * @return this for chaining.
     */
    public T withListColumn(ListColumnWriterOptions child) {
      assert (child.getChildColumnOptions().length == 2) : "Lists can only have two children";
      if (child.getChildColumnOptions()[0] != DUMMY_CHILD) {
        throw new IllegalArgumentException("First child in the list has to be DUMMY_CHILD");
      }
      if (child.getChildColumnOptions()[1].getColumnName().isEmpty()) {
        throw new IllegalArgumentException("Column name can't be empty");
      }
      children.add(child);
      return (T) this;
    }

    /**
     * Set the map column meta.
     * @return this for chaining.
     */
    public T withMapColumn(ColumnWriterOptions child) {
      children.add(child);
      return (T) this;
    }

    /**
     * Set a child struct meta data
     * @return this for chaining.
     */
    public T withStructColumn(StructColumnWriterOptions child) {
      for (ColumnWriterOptions opt: child.getChildColumnOptions()) {
        if (opt.getColumnName().isEmpty()) {
          throw new IllegalArgumentException("Column name can't be empty");
        }
      }
      children.add(child);
      return (T) this;
    }

    /**
     * Set column name
     */
    public T withNonNullableColumns(String... names) {
      withColumns(false, names);
      return (T) this;
    }

    /**
     * Set nullable column meta data
     */
    public T withNullableColumns(String... names) {
      withColumns(true, names);
      return (T) this;
    }

    /**
     * Set a simple child meta data
     * @return this for chaining.
     */
    public T withColumns(boolean nullable, String... names) {
      for (String n : names) {
        children.add(withColumn(n, nullable));
      }
      return (T) this;
    }

    /**
     * Set a simple child meta data
     * @return this for chaining.
     */
    public T withColumn(boolean nullable, String name, int parquetFieldId) {
      children.add(withColumn(name, nullable, parquetFieldId));
      return (T) this;
    }

    /**
     * Set a Decimal child meta data
     * @return this for chaining.
     */
    public T withDecimalColumn(String name, int precision, boolean nullable) {
      children.add(withDecimal(name, precision, nullable));
      return (T) this;
    }

    /**
     * Set a Decimal child meta data
     * @return this for chaining.
     */
    public T withDecimalColumn(String name, int precision, boolean nullable, int parquetFieldId) {
      children.add(withDecimal(name, precision, nullable, parquetFieldId));
      return (T) this;
    }

    /**
     * Set a Decimal child meta data
     * @return this for chaining.
     */
    public T withNullableDecimalColumn(String name, int precision) {
      withDecimalColumn(name, precision, true);
      return (T) this;
    }

    /**
     * Set a Decimal child meta data
     * @return this for chaining.
     */
    public T withDecimalColumn(String name, int precision) {
      withDecimalColumn(name, precision, false);
      return (T) this;
    }

    /**
     * Set a binary child meta data
     * @return this for chaining.
     */
    public T withBinaryColumn(String name, boolean nullable, int parquetFieldId) {
      children.add(withBinary(name, nullable, parquetFieldId));
      return (T) this;
    }

    /**
     * Set a binary child meta data
     * @return this for chaining.
     */
    public T withBinaryColumn(String name, boolean nullable) {
      children.add(withBinary(name, nullable));
      return (T) this;
    }

    /**
     * Set a timestamp child meta data
     * @return this for chaining.
     */
    public T withTimestampColumn(String name, boolean isInt96, boolean nullable, int parquetFieldId) {
      children.add(withTimestamp(name, isInt96, nullable, parquetFieldId));
      return (T) this;
    }

    /**
     * Set a timestamp child meta data
     * @return this for chaining.
     */
    public T withTimestampColumn(String name, boolean isInt96, boolean nullable) {
      children.add(withTimestamp(name, isInt96, nullable));
      return (T) this;
    }

    /**
     * Set a timestamp child meta data
     * @return this for chaining.
     */
    public T withTimestampColumn(String name, boolean isInt96) {
      withTimestampColumn(name, isInt96, false);
      return (T) this;
    }

    /**
     * Set a timestamp child meta data
     * @return this for chaining.
     */
    public T withNullableTimestampColumn(String name, boolean isInt96) {
      withTimestampColumn(name, isInt96, true);
      return (T) this;
    }

    public abstract V build();
  }

  public ColumnWriterOptions(String columnName, boolean isTimestampTypeInt96,
                             int precision, boolean isNullable) {
    this.isTimestampTypeInt96 = isTimestampTypeInt96;
    this.precision = precision;
    this.isNullable = isNullable;
    this.columnName = columnName;
  }

  public ColumnWriterOptions(String columnName, boolean isTimestampTypeInt96,
                             int precision, boolean isNullable, int parquetFieldId) {
    this(columnName, isTimestampTypeInt96, precision, isNullable);
    this.hasParquetFieldId = true;
    this.parquetFieldId = parquetFieldId;
  }

  public ColumnWriterOptions(String columnName, boolean isNullable) {
    this.isTimestampTypeInt96 = false;
    this.precision = UNKNOWN_PRECISION;
    this.isNullable = isNullable;
    this.columnName = columnName;
  }

  public ColumnWriterOptions(String columnName, boolean isNullable, int parquetFieldId) {
    this(columnName, isNullable);
    this.hasParquetFieldId = true;
    this.parquetFieldId = parquetFieldId;
  }

  public ColumnWriterOptions(String columnName) {
    this(columnName, true);
  }

  @FunctionalInterface
  protected interface ByteArrayProducer {
    boolean[] apply(ColumnWriterOptions opt);
  }

  @FunctionalInterface
  protected interface IntArrayProducer {
    int[] apply(ColumnWriterOptions opt);
  }

  boolean[] getFlatIsTimeTypeInt96() {
    boolean[] ret = {isTimestampTypeInt96};
    if (childColumnOptions.length > 0) {
      return getFlatBooleans(ret, (opt) -> opt.getFlatIsTimeTypeInt96());
    } else {
      return ret;
    }
  }

  protected boolean[] getFlatBooleans(boolean[] ret, ByteArrayProducer producer) {
    boolean[][] childResults = new boolean[childColumnOptions.length][];
    int totalChildrenFlatLength = ret.length;
    for (int i = 0 ; i < childColumnOptions.length ; i++) {
      ColumnWriterOptions opt = childColumnOptions[i];
      childResults[i] = producer.apply(opt);
      totalChildrenFlatLength += childResults[i].length;
    }

    boolean[] result = new boolean[totalChildrenFlatLength];
    System.arraycopy(ret, 0, result, 0, ret.length);
    int copiedSoFar = ret.length;
    for (int i = 0 ; i < childColumnOptions.length ; i++) {
      System.arraycopy(childResults[i], 0, result, copiedSoFar, childResults[i].length);
      copiedSoFar += childResults[i].length;
    }
    return result;
  }

  int[] getFlatPrecision() {
    int[] ret = {precision};
    if (childColumnOptions.length > 0) {
      return getFlatInts(ret, (opt) -> opt.getFlatPrecision());
    } else {
      return ret;
    }
  }

  boolean[] getFlatHasParquetFieldId() {
    boolean[] ret = {hasParquetFieldId};
    if (childColumnOptions.length > 0) {
      return getFlatBooleans(ret, (opt) -> opt.getFlatHasParquetFieldId());
    } else {
      return ret;
    }
  }

  int[] getFlatParquetFieldId() {
    int[] ret = {parquetFieldId};
    if (childColumnOptions.length > 0) {
      return getFlatInts(ret, (opt) -> opt.getFlatParquetFieldId());
    } else {
      return ret;
    }
  }

  boolean[] getFlatIsNullable() {
    boolean[] ret = {isNullable};
    if (childColumnOptions.length > 0) {
      return getFlatBooleans(ret, (opt) -> opt.getFlatIsNullable());
    } else {
      return ret;
    }
  }

  boolean[] getFlatIsMap() {
    boolean[] ret = {isMap};
    if (childColumnOptions.length > 0) {
      return getFlatBooleans(ret, (opt) -> opt.getFlatIsMap());
    } else {
      return ret;
    }
  }

  boolean[] getFlatIsBinary() {
    boolean[] ret = {isBinary};
    if (childColumnOptions.length > 0) {
      return getFlatBooleans(ret, (opt) -> opt.getFlatIsBinary());
    } else {
      return ret;
    }
  }

  int[] getFlatNumChildren() {
    int[] ret = {childColumnOptions.length};
    if (childColumnOptions.length > 0) {
      return getFlatInts(ret, (opt) -> opt.getFlatNumChildren());
    } else {
      return ret;
    }
  }

  protected int[] getFlatInts(int[] ret, IntArrayProducer producer) {
    int[][] childResults = new int[childColumnOptions.length][];
    int totalChildrenFlatLength = ret.length;
    for (int i = 0 ; i < childColumnOptions.length ; i++) {
      ColumnWriterOptions opt = childColumnOptions[i];
      childResults[i] = producer.apply(opt);
      totalChildrenFlatLength += childResults[i].length;
    }

    int[] result = new int[totalChildrenFlatLength];
    System.arraycopy(ret, 0, result, 0, ret.length);
    int copiedSoFar = ret.length;
    for (int i = 0 ; i < childColumnOptions.length ; i++) {
      System.arraycopy(childResults[i], 0, result, copiedSoFar, childResults[i].length);
      copiedSoFar += childResults[i].length;
    }
    return result;
  }

  String[] getFlatColumnNames() {
    String[] ret = {columnName};
    if (childColumnOptions.length > 0) {
      return getFlatColumnNames(ret);
    } else {
      return ret;
    }
  }

  protected String[] getFlatColumnNames(String[] ret) {
    String[][] childResults = new String[childColumnOptions.length][];
    int totalChildrenFlatLength = ret.length;
    for (int i = 0 ; i < childColumnOptions.length ; i++) {
      ColumnWriterOptions opt = childColumnOptions[i];
      childResults[i] = opt.getFlatColumnNames();
      totalChildrenFlatLength += childResults[i].length;
    }

    String[] result = new String[totalChildrenFlatLength];
    System.arraycopy(ret, 0, result, 0, ret.length);
    int copiedSoFar = ret.length;
    for (int i = 0 ; i < childColumnOptions.length ; i++) {
      System.arraycopy(childResults[i], 0, result, copiedSoFar, childResults[i].length);
      copiedSoFar += childResults[i].length;
    }
    return result;
  }

  /**
   * Add a Map Column to the schema.
   * <p>
   * Maps are List columns with a Struct named 'key_value' with a child named 'key' and a child
   * named 'value'. The caller of this method doesn't need to worry about this as this method will
   * take care of this without the knowledge of the caller.
   *
   * Note: This method always returns a nullabe column, cannot return non-nullable column.
   * Do not use this, use the next function with the parameter `isNullable`.
   */
  @Deprecated
  public static ColumnWriterOptions mapColumn(String name, ColumnWriterOptions key,
                                              ColumnWriterOptions value) {
    StructColumnWriterOptions struct = structBuilder("key_value").build();
    if (key.isNullable) {
      throw new IllegalArgumentException("key column can not be nullable");
    }
    struct.childColumnOptions = new ColumnWriterOptions[]{key, value};
    ColumnWriterOptions opt = listBuilder(name)
        .withStructColumn(struct)
        .build();
    opt.isMap = true;
    return opt;
  }

  /**
   * Add a Map Column to the schema.
   * <p>
   * Maps are List columns with a Struct named 'key_value' with a child named 'key' and a child
   * named 'value'. The caller of this method doesn't need to worry about this as this method will
   * take care of this without the knowledge of the caller.
   *
   * Note: If this map column is a key of another map, should pass isNullable = false.
   * e.g.: map1(map2(int, int), int) the map2 should be non-nullable.
   *
   * @param isNullable is the returned map nullable.
   */
  public static ColumnWriterOptions mapColumn(String name, ColumnWriterOptions key,
                                              ColumnWriterOptions value, Boolean isNullable) {
    if (key.isNullable) {
      throw new IllegalArgumentException("key column can not be nullable");
    }
    StructColumnWriterOptions struct = structBuilder("key_value").build();
    struct.childColumnOptions = new ColumnWriterOptions[]{key, value};
    ColumnWriterOptions opt = listBuilder(name, isNullable)
        .withStructColumn(struct)
        .build();
    opt.isMap = true;
    return opt;
  }

  /**
   * Add a Map Column to the schema with a Parquet field ID set on the outer list.
   * <p>
   * Maps are List columns with a Struct named 'key_value' with a child named 'key' and a child
   * named 'value'. The caller of this method doesn't need to worry about this as this method will
   * take care of this without the knowledge of the caller.
   *
   * @param isNullable     is the returned map nullable. Primitive {@code boolean} (not
   *                       {@code Boolean}) so a {@code null} can't slip through and trigger an
   *                       unexpected {@link NullPointerException} when forwarded to
   *                       {@link #listBuilder(String, boolean, int)}.
   * @param parquetFieldId the Parquet field ID to attach to the outer (list) column.
   */
  public static ColumnWriterOptions mapColumn(String name, ColumnWriterOptions key,
                                              ColumnWriterOptions value, boolean isNullable,
                                              int parquetFieldId) {
    if (key.isNullable) {
      throw new IllegalArgumentException("key column can not be nullable");
    }
    StructColumnWriterOptions struct = structBuilder("key_value").build();
    struct.childColumnOptions = new ColumnWriterOptions[]{key, value};
    ColumnWriterOptions opt = listBuilder(name, isNullable, parquetFieldId)
        .withStructColumn(struct)
        .build();
    opt.isMap = true;
    return opt;
  }

  /**
   * Creates a ListBuilder for column called 'name'
   */
  public static ListBuilder listBuilder(String name) {
    return new ListBuilder(name, true);
  }

  /**
   * Creates a ListBuilder for column called 'name'
   */
  public static ListBuilder listBuilder(String name, boolean isNullable) {
    return new ListBuilder(name, isNullable);
  }

  /**
   * Creates a ListBuilder for column called 'name' with a Parquet field ID set on the list itself.
   */
  public static ListBuilder listBuilder(String name, boolean isNullable, int parquetFieldId) {
    return new ListBuilder(name, isNullable, parquetFieldId);
  }

  /**
   * Creates a StructBuilder for column called 'name'
   */
  public static StructBuilder structBuilder(String name, boolean isNullable) {
    return new StructBuilder(name, isNullable);
  }

  /**
   * Creates a StructBuilder for column called 'name'
   */
  public static StructBuilder structBuilder(String name, boolean isNullable, int parquetFieldId) {
    return new StructBuilder(name, isNullable, parquetFieldId);
  }

  /**
   * Creates a StructBuilder for column called 'name'
   */
  public static StructBuilder structBuilder(String name) {
    return new StructBuilder(name, true);
  }

  /**
   * Return if the column can have null values
   */
  public String getColumnName() {
    return columnName;
  }

  /**
   * Return if the column can have null values
   */
  public boolean isNullable() {
    return isNullable;
  }

  /**
   * Return the precision for this column
   */
  public int getPrecision() {
    return precision;
  }

  /**
   * Returns true if the writer is expected to write timestamps in INT96
   */
  public boolean isTimestampTypeInt96() {
    return isTimestampTypeInt96;
  }

  /**
   * Return the child columnOptions for this column
   */
  public ColumnWriterOptions[] getChildColumnOptions() {
    return childColumnOptions;
  }

  /**
   * Returns true if this column is a binary (byte array) column in Parquet.
   */
  public boolean isBinary() {
    return isBinary;
  }

  /**
   * Returns true if a Parquet field ID was set on this column.
   */
  public boolean hasParquetFieldId() {
    return hasParquetFieldId;
  }

  /**
   * Return the Parquet field ID for this column. The value is only meaningful when
   * {@link #hasParquetFieldId()} returns true.
   */
  public int getParquetFieldId() {
    return parquetFieldId;
  }

  public static class StructColumnWriterOptions extends ColumnWriterOptions {
    protected StructColumnWriterOptions(AbstractStructBuilder builder) {
      super(builder);
    }
  }

  public static class ListColumnWriterOptions extends ColumnWriterOptions {
    protected ListColumnWriterOptions(ListBuilder builder) {
      super(builder);
    }
  }

  public static class StructBuilder extends AbstractStructBuilder<StructBuilder, StructColumnWriterOptions> {
    public StructBuilder(String name, boolean isNullable) {
      super(name, isNullable);
    }

    public StructBuilder(String name, boolean isNullable, int parquetFieldId) {
      super(name, isNullable, parquetFieldId);
    }

    public StructColumnWriterOptions build() {
      return new StructColumnWriterOptions(this);
    }
  }

  public static class ListBuilder extends NestedBuilder<ListBuilder, ListColumnWriterOptions> {
    public ListBuilder(String name, boolean isNullable) {
      super(name, isNullable);
    }

    public ListBuilder(String name, boolean isNullable, int parquetFieldId) {
      super(name, isNullable, parquetFieldId);
    }

    public ListColumnWriterOptions build() {
      return new ListColumnWriterOptions(this);
    }
  }
}
