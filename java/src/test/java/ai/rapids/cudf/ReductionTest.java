/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */
package ai.rapids.cudf;

import com.google.common.collect.Lists;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.EnumSet;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ReductionTest extends CudfTestBase {
  public static final double DELTAD = 0.00001;
  public static final float DELTAF = 0.001f;

  // reduction operations that produce a floating point value
  private static final EnumSet<Aggregation.Kind> FLOAT_REDUCTIONS = EnumSet.of(
      Aggregation.Kind.MEAN,
      Aggregation.Kind.STD,
      Aggregation.Kind.VARIANCE,
      Aggregation.Kind.QUANTILE);

  // reduction operations that produce a bool value
  private static final EnumSet<Aggregation.Kind> BOOL_REDUCTIONS = EnumSet.of(
      Aggregation.Kind.ANY,
      Aggregation.Kind.ALL);

  private static Scalar buildExpectedScalar(ReductionAggregation op,
      HostColumnVector.DataType dataType, Object expectedObject) {

    if (expectedObject == null) {
      return Scalar.fromNull(dataType.getType());
    }
    if (FLOAT_REDUCTIONS.contains(op.getWrapped().kind)) {
      if (dataType.getType().equals(DType.FLOAT32)) {
        return Scalar.fromFloat((Float) expectedObject);
      }
      return Scalar.fromDouble((Double) expectedObject);
    }
    if (BOOL_REDUCTIONS.contains(op.getWrapped().kind)) {
      return Scalar.fromBool((Boolean) expectedObject);
    }
    switch (dataType.getType().typeId) {
    case BOOL8:
      return Scalar.fromBool((Boolean) expectedObject);
    case INT8:
      return Scalar.fromByte((Byte) expectedObject);
    case INT16:
      return Scalar.fromShort((Short) expectedObject);
    case INT32:
      return Scalar.fromInt((Integer) expectedObject);
    case INT64:
      return Scalar.fromLong((Long) expectedObject);
    case FLOAT32:
      return Scalar.fromFloat((Float) expectedObject);
    case FLOAT64:
      return Scalar.fromDouble((Double) expectedObject);
    case TIMESTAMP_DAYS:
      return Scalar.timestampDaysFromInt((Integer) expectedObject);
    case TIMESTAMP_SECONDS:
    case TIMESTAMP_MILLISECONDS:
    case TIMESTAMP_MICROSECONDS:
    case TIMESTAMP_NANOSECONDS:
      return Scalar.timestampFromLong(dataType.getType(), (Long) expectedObject);
    case STRING:
      return Scalar.fromString((String) expectedObject);
    case LIST:
      HostColumnVector.DataType et = dataType.getChild(0);
      ColumnVector col = null;
      try {
        switch (et.getType().typeId) {
        case BOOL8:
          col = et.isNullable() ? ColumnVector.fromBoxedBooleans((Boolean[]) expectedObject) :
              ColumnVector.fromBooleans((boolean[]) expectedObject);
          return Scalar.listFromColumnView(col);
        case INT8:
          col = et.isNullable() ? ColumnVector.fromBoxedBytes((Byte[]) expectedObject) :
              ColumnVector.fromBytes((byte[]) expectedObject);
          return Scalar.listFromColumnView(col);
        case INT16:
          col = et.isNullable() ? ColumnVector.fromBoxedShorts((Short[]) expectedObject) :
              ColumnVector.fromShorts((short[]) expectedObject);
          return Scalar.listFromColumnView(col);
        case INT32:
          col = et.isNullable() ? ColumnVector.fromBoxedInts((Integer[]) expectedObject) :
              ColumnVector.fromInts((int[]) expectedObject);
          return Scalar.listFromColumnView(col);
        case INT64:
          col = et.isNullable() ? ColumnVector.fromBoxedLongs((Long[]) expectedObject) :
              ColumnVector.fromLongs((long[]) expectedObject);
          return Scalar.listFromColumnView(col);
        case FLOAT32:
          col = et.isNullable() ? ColumnVector.fromBoxedFloats((Float[]) expectedObject) :
              ColumnVector.fromFloats((float[]) expectedObject);
          return Scalar.listFromColumnView(col);
        case FLOAT64:
          col = et.isNullable() ? ColumnVector.fromBoxedDoubles((Double[]) expectedObject) :
              ColumnVector.fromDoubles((double[]) expectedObject);
          return Scalar.listFromColumnView(col);
        case STRING:
          col = ColumnVector.fromStrings((String[]) expectedObject);
          return Scalar.listFromColumnView(col);
        default:
          throw new IllegalArgumentException("Unexpected element type of List: " + et);
        }
      } finally {
        if (col != null) {
          col.close();
        }
      }
    default:
      throw new IllegalArgumentException("Unexpected type: " + dataType);
    }
  }

  private static Stream<Arguments> createBooleanParams() {
    Boolean[] vals = new Boolean[]{true, true, null, false, true, false, null};
    HostColumnVector.DataType bool = new HostColumnVector.BasicType(true, DType.BOOL8);
    HostColumnVector.DataType int32 = new HostColumnVector.BasicType(true, DType.INT32);
    return Stream.of(
        Arguments.of(ReductionAggregation.sum(), new Boolean[0], bool, null, 0.),
        Arguments.of(ReductionAggregation.sum(), new Boolean[]{null, null, null}, bool, null, 0.),
        Arguments.of(ReductionAggregation.sum(), vals, bool, true, 0.),
        Arguments.of(ReductionAggregation.argMin(), vals, int32, 3, 0.),
        Arguments.of(ReductionAggregation.argMax(), vals, int32, 0, 0.),
        Arguments.of(ReductionAggregation.min(), vals, bool, false, 0.),
        Arguments.of(ReductionAggregation.max(), vals, bool, true, 0.),
        Arguments.of(ReductionAggregation.product(), vals, bool, false, 0.),
        Arguments.of(ReductionAggregation.sumOfSquares(), vals, bool, true, 0.),
        Arguments.of(ReductionAggregation.mean(), vals, bool, 0.6, DELTAD),
        Arguments.of(ReductionAggregation.standardDeviation(), vals, bool, 0.5477225575051662, DELTAD),
        Arguments.of(ReductionAggregation.variance(), vals, bool, 0.3, DELTAD),
        Arguments.of(ReductionAggregation.any(), vals, bool, true, 0.),
        Arguments.of(ReductionAggregation.all(), vals, bool, false, 0.)
    );
  }

  private static Stream<Arguments> createByteParams() {
    Byte[] vals = new Byte[]{-1, 7, 123, null, 50, 60, 100};
    HostColumnVector.DataType int8 = new HostColumnVector.BasicType(true, DType.INT8);
    HostColumnVector.DataType int32 = new HostColumnVector.BasicType(true, DType.INT32);
    return Stream.of(
        Arguments.of(ReductionAggregation.sum(), new Byte[0], int8, null, 0.),
        Arguments.of(ReductionAggregation.sum(), new Byte[]{null, null, null}, int8, null, 0.),
        Arguments.of(ReductionAggregation.sum(), vals, int8, (byte) 83, 0.),
        Arguments.of(ReductionAggregation.argMin(), vals, int32, 0, 0.),
        Arguments.of(ReductionAggregation.argMax(), vals, int32, 2, 0.),
        Arguments.of(ReductionAggregation.min(), vals, int8, (byte) -1, 0.),
        Arguments.of(ReductionAggregation.max(), vals, int8, (byte) 123, 0.),
        Arguments.of(ReductionAggregation.product(), vals, int8, (byte) 160, 0.),
        Arguments.of(ReductionAggregation.sumOfSquares(), vals, int8, (byte) 47, 0.),
        Arguments.of(ReductionAggregation.mean(), vals, int8, 56.5, DELTAD),
        Arguments.of(ReductionAggregation.standardDeviation(), vals, int8, 49.24530434467839, DELTAD),
        Arguments.of(ReductionAggregation.variance(), vals, int8, 2425.1, DELTAD),
        Arguments.of(ReductionAggregation.any(), vals, int8, true, 0.),
        Arguments.of(ReductionAggregation.all(), vals, int8, true, 0.)
    );
  }

  private static Stream<Arguments> createShortParams() {
    Short[] vals = new Short[]{-1, 7, 123, null, 50, 60, 100};
    HostColumnVector.DataType int16 = new HostColumnVector.BasicType(true, DType.INT16);
    HostColumnVector.DataType int32 = new HostColumnVector.BasicType(true, DType.INT32);
    return Stream.of(
        Arguments.of(ReductionAggregation.sum(), new Short[0], int16, null, 0.),
        Arguments.of(ReductionAggregation.sum(), new Short[]{null, null, null}, int16, null, 0.),
        Arguments.of(ReductionAggregation.sum(), vals, int16, (short) 339, 0.),
        Arguments.of(ReductionAggregation.argMin(), vals, int32, 0, 0.),
        Arguments.of(ReductionAggregation.argMax(), vals, int32, 2, 0.),
        Arguments.of(ReductionAggregation.min(), vals, int16, (short) -1, 0.),
        Arguments.of(ReductionAggregation.max(), vals, int16, (short) 123, 0.),
        Arguments.of(ReductionAggregation.product(), vals, int16, (short) -22624, 0.),
        Arguments.of(ReductionAggregation.sumOfSquares(), vals, int16, (short) 31279, 0.),
        Arguments.of(ReductionAggregation.mean(), vals, int16, 56.5, DELTAD),
        Arguments.of(ReductionAggregation.standardDeviation(), vals, int16, 49.24530434467839, DELTAD),
        Arguments.of(ReductionAggregation.variance(), vals, int16, 2425.1, DELTAD),
        Arguments.of(ReductionAggregation.any(), vals, int16, true, 0.),
        Arguments.of(ReductionAggregation.all(), vals, int16, true, 0.)
    );
  }

  private static Stream<Arguments> createIntParams() {
    Integer[] vals = new Integer[]{-1, 7, 123, null, 50, 60, 100};
    HostColumnVector.BasicType int32 = new HostColumnVector.BasicType(true, DType.INT32);
    return Stream.of(
        Arguments.of(ReductionAggregation.sum(), new Integer[0], int32, null, 0.),
        Arguments.of(ReductionAggregation.sum(), new Integer[]{null, null, null}, int32, null, 0.),
        Arguments.of(ReductionAggregation.sum(), vals, int32, 339, 0.),
        Arguments.of(ReductionAggregation.argMin(), vals, int32, 0, 0.),
        Arguments.of(ReductionAggregation.argMax(), vals, int32, 2, 0.),
        Arguments.of(ReductionAggregation.min(), vals, int32, -1, 0.),
        Arguments.of(ReductionAggregation.max(), vals, int32, 123, 0.),
        Arguments.of(ReductionAggregation.product(), vals, int32, -258300000, 0.),
        Arguments.of(ReductionAggregation.sumOfSquares(), vals, int32, 31279, 0.),
        Arguments.of(ReductionAggregation.mean(), vals, int32, 56.5, DELTAD),
        Arguments.of(ReductionAggregation.standardDeviation(), vals, int32, 49.24530434467839, DELTAD),
        Arguments.of(ReductionAggregation.variance(), vals, int32, 2425.1, DELTAD),
        Arguments.of(ReductionAggregation.any(), vals, int32, true, 0.),
        Arguments.of(ReductionAggregation.all(), vals, int32, true, 0.)
    );
  }

  private static Stream<Arguments> createLongParams() {
    Long[] vals = new Long[]{-1L, 7L, 123L, null, 50L, 60L, 100L};
    HostColumnVector.DataType int32 = new HostColumnVector.BasicType(true, DType.INT32);
    HostColumnVector.BasicType int64 = new HostColumnVector.BasicType(true, DType.INT64);
    return Stream.of(
        Arguments.of(ReductionAggregation.sum(), new Long[0], int64, null, 0.),
        Arguments.of(ReductionAggregation.sum(), new Long[]{null, null, null}, int64, null, 0.),
        Arguments.of(ReductionAggregation.sum(), vals, int64, 339L, 0.),
        Arguments.of(ReductionAggregation.argMin(), vals, int32, 0, 0.),
        Arguments.of(ReductionAggregation.argMax(), vals, int32, 2, 0.),
        Arguments.of(ReductionAggregation.min(), vals, int64, -1L, 0.),
        Arguments.of(ReductionAggregation.max(), vals, int64, 123L, 0.),
        Arguments.of(ReductionAggregation.product(), vals, int64, -258300000L, 0.),
        Arguments.of(ReductionAggregation.sumOfSquares(), vals, int64, 31279L, 0.),
        Arguments.of(ReductionAggregation.mean(), vals, int64, 56.5, DELTAD),
        Arguments.of(ReductionAggregation.standardDeviation(), vals, int64, 49.24530434467839, DELTAD),
        Arguments.of(ReductionAggregation.variance(), vals, int64, 2425.1, DELTAD),
        Arguments.of(ReductionAggregation.any(), vals, int64, true, 0.),
        Arguments.of(ReductionAggregation.all(), vals, int64, true, 0.),
        Arguments.of(ReductionAggregation.quantile(0.5), vals, int64, 55.0, DELTAD),
        Arguments.of(ReductionAggregation.quantile(0.9), vals, int64, 111.5, DELTAD)
    );
  }

  private static Stream<Arguments> createFloatParams() {
    Float[] vals = new Float[]{-1f, 7f, 123f, null, 50f, 60f, 100f};
    Float[] notNulls = new Float[]{-1f, 7f, 123f, 50f, 60f, 100f};
    Float[] repeats = new Float[]{Float.MIN_VALUE, 7f, 7f, null, null, Float.NaN, Float.NaN, 50f, 50f, 100f};
    HostColumnVector.BasicType fp32 = new HostColumnVector.BasicType(true, DType.FLOAT32);
    HostColumnVector.DataType int32 = new HostColumnVector.BasicType(true, DType.INT32);
    HostColumnVector.DataType listOfFloat = new HostColumnVector.ListType(
        true, new HostColumnVector.BasicType(true, DType.FLOAT32));
    return Stream.of(
        Arguments.of(ReductionAggregation.sum(), new Float[0], fp32, null, 0f),
        Arguments.of(ReductionAggregation.sum(), new Float[]{null, null, null}, fp32, null, 0f),
        Arguments.of(ReductionAggregation.sum(), vals, fp32, 339f, 0f),
        Arguments.of(ReductionAggregation.argMin(), vals, int32, 0, 0.f),
        Arguments.of(ReductionAggregation.argMax(), vals, int32, 2, 0.f),
        Arguments.of(ReductionAggregation.min(), vals, fp32, -1f, 0f),
        Arguments.of(ReductionAggregation.max(), vals, fp32, 123f, 0f),
        Arguments.of(ReductionAggregation.product(), vals, fp32, -258300000f, 0f),
        Arguments.of(ReductionAggregation.sumOfSquares(), vals, fp32, 31279f, 0f),
        Arguments.of(ReductionAggregation.mean(), vals, fp32, 56.5f, DELTAF),
        Arguments.of(ReductionAggregation.standardDeviation(), vals, fp32, 49.24530434467839f, DELTAF),
        Arguments.of(ReductionAggregation.variance(), vals, fp32, 2425.1f, DELTAF),
        Arguments.of(ReductionAggregation.any(), vals, fp32, true, 0f),
        Arguments.of(ReductionAggregation.all(), vals, fp32, true, 0f),
        Arguments.of(ReductionAggregation.collectList(NullPolicy.INCLUDE), vals, listOfFloat, vals, 0f),
        Arguments.of(ReductionAggregation.collectList(), vals, listOfFloat, notNulls, 0f),
        Arguments.of(ReductionAggregation.collectSet(
                NullPolicy.EXCLUDE, NullEquality.EQUAL, NaNEquality.ALL_EQUAL),
            repeats, listOfFloat,
            new Float[]{Float.MIN_VALUE, 7f, 50f, 100f, Float.NaN}, 0f),
        Arguments.of(ReductionAggregation.collectSet(
                NullPolicy.INCLUDE, NullEquality.EQUAL, NaNEquality.ALL_EQUAL),
            repeats, listOfFloat,
            new Float[]{Float.MIN_VALUE, 7f, 50f, 100f, Float.NaN, null}, 0f),
        Arguments.of(ReductionAggregation.collectSet(
                NullPolicy.INCLUDE, NullEquality.UNEQUAL, NaNEquality.ALL_EQUAL),
            repeats, listOfFloat,
            new Float[]{Float.MIN_VALUE, 7f, 50f, 100f, Float.NaN, null, null}, 0f),
        Arguments.of(ReductionAggregation.collectSet(
                NullPolicy.INCLUDE, NullEquality.EQUAL, NaNEquality.UNEQUAL),
            repeats, listOfFloat,
            new Float[]{Float.MIN_VALUE, 7f, 50f, 100f, Float.NaN, Float.NaN, null}, 0f),
        Arguments.of(ReductionAggregation.collectSet(
                NullPolicy.INCLUDE, NullEquality.UNEQUAL, NaNEquality.UNEQUAL),
            repeats, listOfFloat,
            new Float[]{Float.MIN_VALUE, 7f, 50f, 100f, Float.NaN, Float.NaN, null, null}, 0f),
        Arguments.of(ReductionAggregation.collectSet(),
            repeats, listOfFloat,
            new Float[]{Float.MIN_VALUE, 7f, 50f, 100f, Float.NaN, Float.NaN}, 0f)
    );
  }

  private static Stream<Arguments> createDoubleParams() {
    Double[] vals = new Double[]{-1., 7., 123., null, 50., 60., 100.};
    Double[] notNulls = new Double[]{-1., 7., 123., 50., 60., 100.};
    Double[] repeats = new Double[]{Double.MIN_VALUE, 7., 7., null, null, Double.NaN, Double.NaN, 50., 50., 100.};
    HostColumnVector.BasicType fp64 = new HostColumnVector.BasicType(true, DType.FLOAT64);
    HostColumnVector.DataType int32 = new HostColumnVector.BasicType(true, DType.INT32);
    HostColumnVector.DataType listOfDouble = new HostColumnVector.ListType(
        true, new HostColumnVector.BasicType(true, DType.FLOAT64));
    return Stream.of(
        Arguments.of(ReductionAggregation.sum(), new Double[0], fp64, null, 0.),
        Arguments.of(ReductionAggregation.sum(), new Double[]{null, null, null}, fp64, null, 0.),
        Arguments.of(ReductionAggregation.sum(), vals, fp64, 339., 0.),
        Arguments.of(ReductionAggregation.argMin(), vals, int32, 0, 0.),
        Arguments.of(ReductionAggregation.argMax(), vals, int32, 2, 0.),
        Arguments.of(ReductionAggregation.min(), vals, fp64, -1., 0.),
        Arguments.of(ReductionAggregation.max(), vals, fp64, 123., 0.),
        Arguments.of(ReductionAggregation.product(), vals, fp64, -258300000., 0.),
        Arguments.of(ReductionAggregation.sumOfSquares(), vals, fp64, 31279., 0.),
        Arguments.of(ReductionAggregation.mean(), vals, fp64, 56.5, DELTAD),
        Arguments.of(ReductionAggregation.standardDeviation(), vals, fp64, 49.24530434467839, DELTAD),
        Arguments.of(ReductionAggregation.variance(), vals, fp64, 2425.1, DELTAD),
        Arguments.of(ReductionAggregation.any(), vals, fp64, true, 0.),
        Arguments.of(ReductionAggregation.all(), vals, fp64, true, 0.),
        Arguments.of(ReductionAggregation.quantile(0.5), vals, fp64, 55.0, DELTAD),
        Arguments.of(ReductionAggregation.quantile(0.9), vals, fp64, 111.5, DELTAD),
        Arguments.of(ReductionAggregation.collectList(NullPolicy.INCLUDE), vals, listOfDouble, vals, 0.),
        Arguments.of(ReductionAggregation.collectList(NullPolicy.EXCLUDE), vals, listOfDouble, notNulls, 0.),
        Arguments.of(ReductionAggregation.collectSet(
                NullPolicy.EXCLUDE, NullEquality.EQUAL, NaNEquality.ALL_EQUAL),
            repeats, listOfDouble,
            new Double[]{Double.MIN_VALUE, 7., 50., 100., Double.NaN}, 0.),
        Arguments.of(ReductionAggregation.collectSet(
                NullPolicy.INCLUDE, NullEquality.EQUAL, NaNEquality.ALL_EQUAL),
            repeats, listOfDouble,
            new Double[]{Double.MIN_VALUE, 7., 50., 100., Double.NaN, null}, 0.),
        Arguments.of(ReductionAggregation.collectSet(
                NullPolicy.INCLUDE, NullEquality.UNEQUAL, NaNEquality.ALL_EQUAL),
            repeats, listOfDouble,
            new Double[]{Double.MIN_VALUE, 7., 50., 100., Double.NaN, null, null}, 0.),
        Arguments.of(ReductionAggregation.collectSet(
                NullPolicy.INCLUDE, NullEquality.EQUAL, NaNEquality.UNEQUAL),
            repeats, listOfDouble,
            new Double[]{Double.MIN_VALUE, 7., 50., 100., Double.NaN, Double.NaN, null}, 0.),
        Arguments.of(ReductionAggregation.collectSet(
                NullPolicy.INCLUDE, NullEquality.UNEQUAL, NaNEquality.UNEQUAL),
            repeats, listOfDouble,
            new Double[]{Double.MIN_VALUE, 7., 50., 100., Double.NaN, Double.NaN, null, null}, 0.),
        Arguments.of(ReductionAggregation.collectSet(),
            repeats, listOfDouble,
            new Double[]{Double.MIN_VALUE, 7., 50., 100., Double.NaN, Double.NaN}, 0.)
    );
  }

  private static Stream<Arguments> createTimestampDaysParams() {
    Integer[] vals = new Integer[]{-1, 7, 123, null, 50, 60, 100};
    HostColumnVector.BasicType tsDay = new HostColumnVector.BasicType(true, DType.TIMESTAMP_DAYS);
    HostColumnVector.DataType int32 = new HostColumnVector.BasicType(true, DType.INT32);
    return Stream.of(
        Arguments.of(ReductionAggregation.max(), new Integer[0], tsDay, null),
        Arguments.of(ReductionAggregation.max(), new Integer[]{null, null, null}, tsDay, null),
        Arguments.of(ReductionAggregation.max(), vals, tsDay, 123),
        Arguments.of(ReductionAggregation.min(), vals, tsDay, -1),
        Arguments.of(ReductionAggregation.argMin(), vals, int32, 0, 0.),
        Arguments.of(ReductionAggregation.argMax(), vals, int32, 2, 0.)
    );
  }

  private static Stream<Arguments> createTimestampResolutionParams(HostColumnVector.BasicType type) {
    Long[] vals = new Long[]{-1L, 7L, 123L, null, 50L, 60L, 100L};
    HostColumnVector.DataType int32 = new HostColumnVector.BasicType(true, DType.INT32);
    return Stream.of(
        Arguments.of(ReductionAggregation.max(), new Long[0], type, null),
        Arguments.of(ReductionAggregation.max(), new Long[]{null, null, null}, type, null),
        Arguments.of(ReductionAggregation.min(), vals, type, -1L),
        Arguments.of(ReductionAggregation.max(), vals, type, 123L),
        Arguments.of(ReductionAggregation.argMin(), vals, int32, 0, 0.),
        Arguments.of(ReductionAggregation.argMax(), vals, int32, 2, 0.)
    );
  }

  private static Stream<Arguments> createTimestampSecondsParams() {
    return createTimestampResolutionParams(
        new HostColumnVector.BasicType(true, DType.TIMESTAMP_SECONDS));
  }

  private static Stream<Arguments> createTimestampMilliSecondsParams() {
    return createTimestampResolutionParams(
        new HostColumnVector.BasicType(true, DType.TIMESTAMP_MILLISECONDS));
  }

  private static Stream<Arguments> createTimestampMicroSecondsParams() {
    return createTimestampResolutionParams(
        new HostColumnVector.BasicType(true, DType.TIMESTAMP_MICROSECONDS));
  }

  private static Stream<Arguments> createTimestampNanoSecondsParams() {
    return createTimestampResolutionParams(
        new HostColumnVector.BasicType(true, DType.TIMESTAMP_NANOSECONDS));
  }

  private static Stream<Arguments> createFloatArrayParams() {
    List<Float>[] inputs = new List[]{
        Lists.newArrayList(-1f, 7f, null),
        Lists.newArrayList(7f, 50f, 60f, Float.NaN),
        Lists.newArrayList(),
        Lists.newArrayList(60f, 100f, Float.NaN, null)
    };
    HostColumnVector.DataType fpList = new HostColumnVector.ListType(
        true, new HostColumnVector.BasicType(true, DType.FLOAT32));
    return Stream.of(
        Arguments.of(ReductionAggregation.mergeLists(), inputs, fpList,
            new Float[]{-1f, 7f, null,
                7f, 50f, 60f, Float.NaN,
                60f, 100f, Float.NaN, null}, 0f),
        Arguments.of(ReductionAggregation.mergeSets(NullEquality.EQUAL, NaNEquality.ALL_EQUAL),
            inputs, fpList,
            new Float[]{-1f, 7f, 50f, 60f, 100f, Float.NaN, null}, 0f),
        Arguments.of(ReductionAggregation.mergeSets(NullEquality.UNEQUAL, NaNEquality.ALL_EQUAL),
            inputs, fpList,
            new Float[]{-1f, 7f, 50f, 60f, 100f, Float.NaN, null, null}, 0f),
        Arguments.of(ReductionAggregation.mergeSets(NullEquality.EQUAL, NaNEquality.UNEQUAL),
            inputs, fpList,
            new Float[]{-1f, 7f, 50f, 60f, 100f, Float.NaN, Float.NaN, null}, 0f),
        Arguments.of(ReductionAggregation.mergeSets(),
            inputs, fpList,
            new Float[]{-1f, 7f, 50f, 60f, 100f, Float.NaN, Float.NaN, null, null}, 0f)
    );
  }

  private static void assertEqualsDelta(ReductionAggregation op, Scalar expected, Scalar result,
                                        Double percentage) {
    if (FLOAT_REDUCTIONS.contains(op.getWrapped().kind)) {
      assertEqualsWithinPercentage(expected.getDouble(), result.getDouble(), percentage);
    } else if (expected.getType().typeId == DType.DTypeEnum.LIST) {
      try (ColumnVector expectedAsList = ColumnVector.fromScalar(expected, 1);
           ColumnVector resultAsList = ColumnVector.fromScalar(result, 1);
           ColumnVector expectedSorted = expectedAsList.listSortRows(false, false);
           ColumnVector resultSorted = resultAsList.listSortRows(false, false)) {
        AssertUtils.assertColumnsAreEqual(expectedSorted, resultSorted);
      }
    } else {
      assertEquals(expected, result);
    }
  }

  private static void assertEqualsDelta(ReductionAggregation op, Scalar expected, Scalar result,
                                        Float percentage) {
    if (FLOAT_REDUCTIONS.contains(op.getWrapped().kind)) {
      assertEqualsWithinPercentage(expected.getFloat(), result.getFloat(), percentage);
    } else if (expected.getType().typeId == DType.DTypeEnum.LIST) {
      try (ColumnVector expectedAsList = ColumnVector.fromScalar(expected, 1);
           ColumnVector resultAsList = ColumnVector.fromScalar(result, 1);
           ColumnVector expectedSorted = expectedAsList.listSortRows(false, false);
           ColumnVector resultSorted = resultAsList.listSortRows(false, false)) {
        AssertUtils.assertColumnsAreEqual(expectedSorted, resultSorted);
      }
    } else {
      assertEquals(expected, result);
    }
  }

  @Tag("noSanitizer")
  @ParameterizedTest
  @MethodSource("createBooleanParams")
  void testBoolean(ReductionAggregation op, Boolean[] values,
      HostColumnVector.DataType type, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, type, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedBooleans(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @Tag("noSanitizer")
  @ParameterizedTest
  @MethodSource("createByteParams")
  void testByte(ReductionAggregation op, Byte[] values,
      HostColumnVector.DataType type, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, type, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedBytes(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @Tag("noSanitizer")
  @ParameterizedTest
  @MethodSource("createShortParams")
  void testShort(ReductionAggregation op, Short[] values,
      HostColumnVector.DataType type, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, type, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedShorts(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createIntParams")
  void testInt(ReductionAggregation op, Integer[] values,
      HostColumnVector.DataType type, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, type, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedInts(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @Tag("noSanitizer")
  @ParameterizedTest
  @MethodSource("createLongParams")
  void testLong(ReductionAggregation op, Long[] values,
      HostColumnVector.DataType type, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, type, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedLongs(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createFloatParams")
  void testFloat(ReductionAggregation op, Float[] values,
      HostColumnVector.DataType type, Object expectedObject, Float delta) {
    try (Scalar expected = buildExpectedScalar(op, type, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedFloats(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @Tag("noSanitizer")
  @ParameterizedTest
  @MethodSource("createDoubleParams")
  void testDouble(ReductionAggregation op, Double[] values,
      HostColumnVector.DataType type, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, type, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedDoubles(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @Tag("noSanitizer")
  @ParameterizedTest
  @MethodSource("createTimestampDaysParams")
  void testTimestampDays(ReductionAggregation op, Integer[] values,
      HostColumnVector.DataType type, Object expectedObject) {
    try (Scalar expected = buildExpectedScalar(op, type, expectedObject);
         ColumnVector v = ColumnVector.timestampDaysFromBoxedInts(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEquals(expected, result);
    }
  }

  @Tag("noSanitizer")
  @ParameterizedTest
  @MethodSource("createTimestampSecondsParams")
  void testTimestampSeconds(ReductionAggregation op, Long[] values,
      HostColumnVector.DataType type, Object expectedObject) {
    try (Scalar expected = buildExpectedScalar(op, type, expectedObject);
         ColumnVector v = ColumnVector.timestampSecondsFromBoxedLongs(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEquals(expected, result);
    }
  }

  @Tag("noSanitizer")
  @ParameterizedTest
  @MethodSource("createTimestampMilliSecondsParams")
  void testTimestampMilliseconds(ReductionAggregation op, Long[] values,
      HostColumnVector.DataType type, Object expectedObject) {
    try (Scalar expected = buildExpectedScalar(op, type, expectedObject);
         ColumnVector v = ColumnVector.timestampMilliSecondsFromBoxedLongs(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEquals(expected, result);
    }
  }

  @Tag("noSanitizer")
  @ParameterizedTest
  @MethodSource("createTimestampMicroSecondsParams")
  void testTimestampMicroseconds(ReductionAggregation op, Long[] values,
      HostColumnVector.DataType type, Object expectedObject) {
    try (Scalar expected = buildExpectedScalar(op, type, expectedObject);
         ColumnVector v = ColumnVector.timestampMicroSecondsFromBoxedLongs(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEquals(expected, result);
    }
  }

  @Tag("noSanitizer")
  @ParameterizedTest
  @MethodSource("createTimestampNanoSecondsParams")
  void testTimestampNanoseconds(ReductionAggregation op, Long[] values,
      HostColumnVector.DataType type, Object expectedObject) {
    try (Scalar expected = buildExpectedScalar(op, type, expectedObject);
         ColumnVector v = ColumnVector.timestampNanoSecondsFromBoxedLongs(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEquals(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createFloatArrayParams")
  void testFloatArray(ReductionAggregation op, List<Float>[] values,
      HostColumnVector.DataType type, Object expectedObject, Float delta) {
    HostColumnVector.DataType listType = new HostColumnVector.ListType(
        true, new HostColumnVector.BasicType(true, DType.FLOAT32));
    try (Scalar expected = buildExpectedScalar(op, type, expectedObject);
         ColumnVector v = ColumnVector.fromLists(listType, values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @Test
  void testWithSetOutputType() {
    try (Scalar expected = Scalar.fromLong(1 * 2 * 3 * 4L);
         ColumnVector cv = ColumnVector.fromBytes(new byte[]{1, 2, 3, 4});
         Scalar result = cv.product(DType.INT64)) {
      assertEquals(expected, result);
    }

    try (Scalar expected = Scalar.fromLong(1 + 2 + 3 + 4L);
         ColumnVector cv = ColumnVector.fromBytes(new byte[]{1, 2, 3, 4});
         Scalar result = cv.sum(DType.INT64)) {
      assertEquals(expected, result);
    }

    try (Scalar expected = Scalar.fromLong((1 * 1L) + (2 * 2L) + (3 * 3L) + (4 * 4L));
         ColumnVector cv = ColumnVector.fromBytes(new byte[]{1, 2, 3, 4});
         Scalar result = cv.sumOfSquares(DType.INT64)) {
      assertEquals(expected, result);
    }

    try (Scalar expected = Scalar.fromFloat((1 + 2 + 3 + 4f) / 4);
         ColumnVector cv = ColumnVector.fromBytes(new byte[]{1, 2, 3, 4});
         Scalar result = cv.mean(DType.FLOAT32)) {
      assertEquals(expected, result);
    }

    try (Scalar expected = Scalar.fromFloat(1.6666666f);
         ColumnVector cv = ColumnVector.fromBytes(new byte[]{1, 2, 3, 4});
         Scalar result = cv.variance(DType.FLOAT32)) {
      assertEquals(expected, result);
    }

    try (Scalar expected = Scalar.fromFloat(1.2909944f);
         ColumnVector cv = ColumnVector.fromBytes(new byte[]{1, 2, 3, 4});
         Scalar result = cv.standardDeviation(DType.FLOAT32)) {
      assertEquals(expected, result);
    }
  }

  // Decoded struct scalar produced by SUM_OVERFLOW reduction.
  // sumValid=false models the cudf "no valid input" case (null sum child).
  private static final class SumOverflowResult {
    final boolean sumValid;
    final long sumValue;
    final boolean overflow;

    SumOverflowResult(boolean sumValid, long sumValue, boolean overflow) {
      this.sumValid = sumValid;
      this.sumValue = sumValue;
      this.overflow = overflow;
    }
  }

  // SUM_OVERFLOW reduction returns a struct scalar with children
  // {sum: INT64, overflow: BOOL8}. Helper closes the temporary children views.
  private static SumOverflowResult readSumOverflow(Scalar result) {
    assertEquals(DType.STRUCT, result.getType());
    assertTrue(result.isValid());
    ColumnView[] children = result.getChildrenFromStructScalar();
    try {
      assertEquals(2, children.length);
      assertEquals(DType.INT64, children[0].getType());
      assertEquals(DType.BOOL8, children[1].getType());
      try (ColumnVector sumCol = children[0].copyToColumnVector();
           ColumnVector ovfCol = children[1].copyToColumnVector();
           HostColumnVector sumHost = sumCol.copyToHost();
           HostColumnVector ovfHost = ovfCol.copyToHost()) {
        boolean sumValid = !sumHost.isNull(0);
        long sumValue = sumValid ? sumHost.getLong(0) : 0L;
        boolean overflow = ovfHost.getBoolean(0);
        return new SumOverflowResult(sumValid, sumValue, overflow);
      }
    } finally {
      for (ColumnView c : children) c.close();
    }
  }

  @Test
  void testSumOverflowNoOverflow() {
    try (ColumnVector cv = ColumnVector.fromLongs(1L, 2L, 3L, 4L, 5L);
         Scalar result = cv.reduce(ReductionAggregation.sumOverflow(), DType.STRUCT)) {
      SumOverflowResult r = readSumOverflow(result);
      assertTrue(r.sumValid);
      assertEquals(15L, r.sumValue);  // 1+2+3+4+5
      assertFalse(r.overflow);
    }
  }

  @Test
  void testSumOverflowPositiveOverflow() {
    // On overflow the sum value is unspecified; the overflow flag is the source of truth.
    try (ColumnVector cv = ColumnVector.fromLongs(Long.MAX_VALUE, 1L);
         Scalar result = cv.reduce(ReductionAggregation.sumOverflow(), DType.STRUCT)) {
      SumOverflowResult r = readSumOverflow(result);
      assertTrue(r.sumValid);
      assertTrue(r.overflow);
    }
  }

  @Test
  void testSumOverflowNegativeOverflow() {
    // On overflow the sum value is unspecified; the overflow flag is the source of truth.
    try (ColumnVector cv = ColumnVector.fromLongs(Long.MIN_VALUE, -1L);
         Scalar result = cv.reduce(ReductionAggregation.sumOverflow(), DType.STRUCT)) {
      SumOverflowResult r = readSumOverflow(result);
      assertTrue(r.sumValid);
      assertTrue(r.overflow);
    }
  }

  @Test
  void testSumOverflowEmptyColumn() {
    try (ColumnVector cv = ColumnVector.fromLongs();
         Scalar result = cv.reduce(ReductionAggregation.sumOverflow(), DType.STRUCT)) {
      SumOverflowResult r = readSumOverflow(result);
      assertFalse(r.sumValid);  // empty input -> null sum
      assertFalse(r.overflow);
    }
  }

  @Test
  void testSumOverflowAllNullColumn() {
    try (ColumnVector cv = ColumnVector.fromBoxedLongs(null, null, null);
         Scalar result = cv.reduce(ReductionAggregation.sumOverflow(), DType.STRUCT)) {
      SumOverflowResult r = readSumOverflow(result);
      assertFalse(r.sumValid);  // all-null input -> null sum
      assertFalse(r.overflow);
    }
  }

  @Test
  void testSumOverflowRejectsUnsupportedTypes() {
    // SUM_OVERFLOW supports signed integers and decimals; float remains unsupported.
    try (ColumnVector cv = ColumnVector.fromFloats(1.0f, 2.0f, 3.0f)) {
      assertThrows(CudfException.class, () ->
          cv.reduce(ReductionAggregation.sumOverflow(), DType.STRUCT).close());
    }
  }
}
