/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * A value that can be an operand of a binary operation, such as a column or a scalar.
 * Implemented by {@link ColumnView} and {@link Scalar}; the operations are built on
 * {@link #binaryOp(BinaryOp, BinaryOperable, DType)}.
 */
public interface BinaryOperable {
  /**
   * Finds the proper DType for an implicit output.  This follows the typical rules of
   * C++, Java, and most SQL implementations.
   * FLOAT64/double >
   * FLOAT32/float >
   * INT64/long >
   * INT32/int >
   * INT16/short >
   * INT8/byte/char
   * <p>
   * Currently most TIMESTAMPs are treated the same as INT64. TIMESTAMP_DAYS is treated the same
   * as INT32. All time information is stripped from them.  This may change in the future.
   * <p>
   * BOOL8 is treated like an INT8.  Math on boolean operations makes little sense.  If
   * you want to stay as a BOOL8 you will need to explicitly specify the output type.
   * For decimal types, DECIMAL32 and DECIMAL64 takes in another parameter `scale`. DType is created
   * with scale=0 as scale is required. Dtype is discarded for binary operations for decimal
   * types in cudf as a new DType is created for output type with the new scale.
   */
  static DType implicitConversion(BinaryOp op, BinaryOperable lhs, BinaryOperable rhs) {
    DType a = lhs.getType();
    DType b = rhs.getType();
    if (a.equals(DType.FLOAT64) || b.equals(DType.FLOAT64)) {
      return DType.FLOAT64;
    }
    if (a.equals(DType.FLOAT32) || b.equals(DType.FLOAT32)) {
      return DType.FLOAT32;
    }
    if (a.equals(DType.UINT64) || b.equals(DType.UINT64)) {
      return DType.UINT64;
    }
    if (a.equals(DType.INT64) || b.equals(DType.INT64) ||
        a.equals(DType.TIMESTAMP_MILLISECONDS) || b.equals(DType.TIMESTAMP_MILLISECONDS) ||
        a.equals(DType.TIMESTAMP_MICROSECONDS) || b.equals(DType.TIMESTAMP_MICROSECONDS) ||
        a.equals(DType.TIMESTAMP_SECONDS) || b.equals(DType.TIMESTAMP_SECONDS) ||
        a.equals(DType.TIMESTAMP_NANOSECONDS) || b.equals(DType.TIMESTAMP_NANOSECONDS)) {
      return DType.INT64;
    }
    if (a.equals(DType.UINT32) || b.equals(DType.UINT32)) {
      return DType.UINT32;
    }
    if (a.equals(DType.INT32) || b.equals(DType.INT32) ||
        a.equals(DType.TIMESTAMP_DAYS) || b.equals(DType.TIMESTAMP_DAYS)) {
      return DType.INT32;
    }
    if (a.equals(DType.UINT16) || b.equals(DType.UINT16)) {
      return DType.UINT16;
    }
    if (a.equals(DType.INT16) || b.equals(DType.INT16)) {
      return DType.INT16;
    }
    if (a.equals(DType.UINT8) || b.equals(DType.UINT8)) {
      return DType.UINT8;
    }
    if (a.equals(DType.INT8) || b.equals(DType.INT8)) {
      return DType.INT8;
    }
    if (a.equals(DType.BOOL8) || b.equals(DType.BOOL8)) {
      return DType.BOOL8;
    }
    if (a.isDecimalType() && b.isDecimalType()) {
      if (a.typeId != b.typeId) {
        throw new IllegalArgumentException("Both columns must be of the same fixed_point type");
      }
      final int scale = ColumnView.getFixedPointOutputScale(op, lhs.getType(), rhs.getType());
      // The output precision/size should be at least as large as the input.
      // It may be larger if room is needed for it based off of the output scale.
      final DType.DTypeEnum outputEnum;
      if (scale <= DType.DECIMAL32_MAX_PRECISION && a.typeId == DType.DTypeEnum.DECIMAL32) {
        outputEnum = DType.DTypeEnum.DECIMAL32;
      } else if (scale <= DType.DECIMAL64_MAX_PRECISION &&
          (a.typeId == DType.DTypeEnum.DECIMAL32 || a.typeId == DType.DTypeEnum.DECIMAL64)) {
        outputEnum = DType.DTypeEnum.DECIMAL64;
      } else {
        outputEnum = DType.DTypeEnum.DECIMAL128;
      }
      return DType.create(outputEnum, scale);
    }
    throw new IllegalArgumentException("Unsupported types " + a + " and " + b);
  }

  /**
   * Get the type of this data.
   */
  DType getType();

  /**
   * Multiple different binary operations.
   * @param op      the operation to perform
   * @param rhs     the rhs of the operation
   * @param outType the type of output you want.
   * @return the result
   * @deprecated Use
   *     {@link #binaryOp(BinaryOp, BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector binaryOp(BinaryOp op, BinaryOperable rhs, DType outType) {
    return binaryOp(op, rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Multiple different binary operations.
   * @param op      the operation to perform
   * @param rhs     the rhs of the operation
   * @param outType the type of output you want.
   * @param stream  CUDA stream used for device memory operations and kernel launches
   * @param mr      device memory resources for the returned column and for temporary allocations
   * @return the result
   */
  ColumnVector binaryOp(BinaryOp op, BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr);

  /**
   * Add one vector to another with the given output type. this + rhs
   * Output type is ignored for the operations between decimal types and
   * it is always decimal type.
   * @deprecated Use {@link #add(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector add(BinaryOperable rhs, DType outType) {
    return add(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Add one vector to another with the given output type. this + rhs
   * Output type is ignored for the operations between decimal types and
   * it is always decimal type.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector add(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.ADD, rhs, outType, stream, mr);
  }

  /**
   * Add + operator. this + rhs
   * @deprecated Use {@link #add(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector add(BinaryOperable rhs) {
    return add(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Add + operator. this + rhs
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector add(BinaryOperable rhs, Cuda.Stream stream, MemoryResources mr) {
    return add(rhs, implicitConversion(BinaryOp.ADD, this, rhs), stream, mr);
  }

  /**
   * Subtract one vector from another with the given output type. this - rhs
   * Output type is ignored for the operations between decimal types and
   * it is always decimal type.
   * @deprecated Use {@link #sub(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector sub(BinaryOperable rhs, DType outType) {
    return sub(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Subtract one vector from another with the given output type. this - rhs
   * Output type is ignored for the operations between decimal types and
   * it is always decimal type.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector sub(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.SUB, rhs, outType, stream, mr);
  }

  /**
   * Subtract one vector from another. this - rhs
   * @deprecated Use {@link #sub(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector sub(BinaryOperable rhs) {
    return sub(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Subtract one vector from another. this - rhs
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector sub(BinaryOperable rhs, Cuda.Stream stream, MemoryResources mr) {
    return sub(rhs, implicitConversion(BinaryOp.SUB, this, rhs), stream, mr);
  }

  /**
   * Multiply two vectors together with the given output type. this * rhs
   * Output type is ignored for the operations between decimal types and
   * it is always decimal type.
   * @deprecated Use {@link #mul(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector mul(BinaryOperable rhs, DType outType) {
    return mul(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Multiply two vectors together with the given output type. this * rhs
   * Output type is ignored for the operations between decimal types and
   * it is always decimal type.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector mul(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.MUL, rhs, outType, stream, mr);
  }

  /**
   * Multiply two vectors together. this * rhs
   * @deprecated Use {@link #mul(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector mul(BinaryOperable rhs) {
    return mul(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Multiply two vectors together. this * rhs
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector mul(BinaryOperable rhs, Cuda.Stream stream, MemoryResources mr) {
    return mul(rhs, implicitConversion(BinaryOp.MUL, this, rhs), stream, mr);
  }

  /**
   * Divide one vector by another with the given output type. this / rhs
   * Output type is ignored for the operations between decimal types and
   * it is always decimal type.
   * @deprecated Use {@link #div(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector div(BinaryOperable rhs, DType outType) {
    return div(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Divide one vector by another with the given output type. this / rhs
   * Output type is ignored for the operations between decimal types and
   * it is always decimal type.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector div(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.DIV, rhs, outType, stream, mr);
  }

  /**
   * Divide one vector by another. this / rhs
   * @deprecated Use {@link #div(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector div(BinaryOperable rhs) {
    return div(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Divide one vector by another. this / rhs
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector div(BinaryOperable rhs, Cuda.Stream stream, MemoryResources mr) {
    return div(rhs, implicitConversion(BinaryOp.DIV, this, rhs), stream, mr);
  }

  /**
   * Divide one vector by another converting to FLOAT64 in between with the given output type.
   * (double)this / (double)rhs
   * @deprecated Use {@link #trueDiv(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector trueDiv(BinaryOperable rhs, DType outType) {
    return trueDiv(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Divide one vector by another converting to FLOAT64 in between with the given output type.
   * (double)this / (double)rhs
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector trueDiv(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.TRUE_DIV, rhs, outType, stream, mr);
  }

  /**
   * Divide one vector by another converting to FLOAT64 in between.
   * (double)this / (double)rhs
   * @deprecated Use {@link #trueDiv(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector trueDiv(BinaryOperable rhs) {
    return trueDiv(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Divide one vector by another converting to FLOAT64 in between.
   * (double)this / (double)rhs
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector trueDiv(BinaryOperable rhs, Cuda.Stream stream, MemoryResources mr) {
    return trueDiv(rhs, implicitConversion(BinaryOp.TRUE_DIV, this, rhs), stream, mr);
  }

  /**
   * Divide one vector by another and calculate the floor of the result with the given output type.
   * Math.floor(this/rhs)
   * @deprecated Use {@link #floorDiv(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector floorDiv(BinaryOperable rhs, DType outType) {
    return floorDiv(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Divide one vector by another and calculate the floor of the result with the given output type.
   * Math.floor(this/rhs)
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector floorDiv(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.FLOOR_DIV, rhs, outType, stream, mr);
  }

  /**
   * Divide one vector by another and calculate the floor of the result.
   * Math.floor(this/rhs)
   * @deprecated Use {@link #floorDiv(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector floorDiv(BinaryOperable rhs) {
    return floorDiv(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Divide one vector by another and calculate the floor of the result.
   * Math.floor(this/rhs)
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector floorDiv(BinaryOperable rhs, Cuda.Stream stream, MemoryResources mr) {
    return floorDiv(rhs, implicitConversion(BinaryOp.FLOOR_DIV, this, rhs), stream, mr);
  }

  /**
   * Compute the modulus with the given output type.
   * this % rhs
   * @deprecated Use {@link #mod(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector mod(BinaryOperable rhs, DType outType) {
    return mod(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Compute the modulus with the given output type.
   * this % rhs
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector mod(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.MOD, rhs, outType, stream, mr);
  }

  /**
   * Compute the modulus.
   * this % rhs
   * @deprecated Use {@link #mod(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector mod(BinaryOperable rhs) {
    return mod(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Compute the modulus.
   * this % rhs
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector mod(BinaryOperable rhs, Cuda.Stream stream, MemoryResources mr) {
    return mod(rhs, implicitConversion(BinaryOp.MOD, this, rhs), stream, mr);
  }

  /**
   * Compute the power with the given output type.
   * Math.pow(this, rhs)
   * @deprecated Use {@link #pow(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector pow(BinaryOperable rhs, DType outType) {
    return pow(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Compute the power with the given output type.
   * Math.pow(this, rhs)
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector pow(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.POW, rhs, outType, stream, mr);
  }

  /**
   * Compute the power.
   * Math.pow(this, rhs)
   * @deprecated Use {@link #pow(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector pow(BinaryOperable rhs) {
    return pow(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Compute the power.
   * Math.pow(this, rhs)
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector pow(BinaryOperable rhs, Cuda.Stream stream, MemoryResources mr) {
    return pow(rhs, implicitConversion(BinaryOp.POW, this, rhs), stream, mr);
  }

  /**
   * this == rhs 1 is true 0 is false with the output cast to the given type.
   * @deprecated Use {@link #equalTo(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector equalTo(BinaryOperable rhs, DType outType) {
    return equalTo(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * this == rhs 1 is true 0 is false with the output cast to the given type.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector equalTo(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.EQUAL, rhs, outType, stream, mr);
  }

  /**
   * this == rhs 1 is true 0 is false.  The output type is BOOL8.
   * @deprecated Use {@link #equalTo(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector equalTo(BinaryOperable rhs) {
    return equalTo(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * this == rhs 1 is true 0 is false.  The output type is BOOL8.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector equalTo(BinaryOperable rhs, Cuda.Stream stream, MemoryResources mr) {
    return equalTo(rhs, DType.BOOL8, stream, mr);
  }

  /**
   * this != rhs 1 is true 0 is false with the output cast to the given type.
   * @deprecated Use {@link #notEqualTo(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector notEqualTo(BinaryOperable rhs, DType outType) {
    return notEqualTo(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * this != rhs 1 is true 0 is false with the output cast to the given type.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector notEqualTo(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.NOT_EQUAL, rhs, outType, stream, mr);
  }

  /**
   * this != rhs 1 is true 0 is false. The output type is BOOL8.
   * @deprecated Use {@link #notEqualTo(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector notEqualTo(BinaryOperable rhs) {
    return notEqualTo(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * this != rhs 1 is true 0 is false. The output type is BOOL8.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector notEqualTo(BinaryOperable rhs, Cuda.Stream stream, MemoryResources mr) {
    return notEqualTo(rhs, DType.BOOL8, stream, mr);
  }

  /**
   * this < rhs 1 is true 0 is false with the output cast to the given type.
   * @deprecated Use {@link #lessThan(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector lessThan(BinaryOperable rhs, DType outType) {
    return lessThan(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * this < rhs 1 is true 0 is false with the output cast to the given type.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector lessThan(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.LESS, rhs, outType, stream, mr);
  }

  /**
   * this < rhs 1 is true 0 is false.  The output type is BOOL8.
   * @deprecated Use {@link #lessThan(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector lessThan(BinaryOperable rhs) {
    return lessThan(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * this < rhs 1 is true 0 is false.  The output type is BOOL8.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector lessThan(BinaryOperable rhs, Cuda.Stream stream, MemoryResources mr) {
    return lessThan(rhs, DType.BOOL8, stream, mr);
  }

  /**
   * this > rhs 1 is true 0 is false with the output cast to the given type.
   * @deprecated Use {@link #greaterThan(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector greaterThan(BinaryOperable rhs, DType outType) {
    return greaterThan(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * this > rhs 1 is true 0 is false with the output cast to the given type.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector greaterThan(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.GREATER, rhs, outType, stream, mr);
  }

  /**
   * this > rhs 1 is true 0 is false.  The output type is BOOL8.
   * @deprecated Use {@link #greaterThan(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector greaterThan(BinaryOperable rhs) {
    return greaterThan(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * this > rhs 1 is true 0 is false.  The output type is BOOL8.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector greaterThan(BinaryOperable rhs, Cuda.Stream stream, MemoryResources mr) {
    return greaterThan(rhs, DType.BOOL8, stream, mr);
  }

  /**
   * this <= rhs 1 is true 0 is false with the output cast to the given type.
   * @deprecated Use {@link #lessOrEqualTo(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector lessOrEqualTo(BinaryOperable rhs, DType outType) {
    return lessOrEqualTo(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * this <= rhs 1 is true 0 is false with the output cast to the given type.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector lessOrEqualTo(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.LESS_EQUAL, rhs, outType, stream, mr);
  }

  /**
   * this <= rhs 1 is true 0 is false.  The output type is BOOL8.
   * @deprecated Use {@link #lessOrEqualTo(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector lessOrEqualTo(BinaryOperable rhs) {
    return lessOrEqualTo(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * this <= rhs 1 is true 0 is false.  The output type is BOOL8.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector lessOrEqualTo(BinaryOperable rhs, Cuda.Stream stream, MemoryResources mr) {
    return lessOrEqualTo(rhs, DType.BOOL8, stream, mr);
  }

  /**
   * this >= rhs 1 is true 0 is false with the output cast to the given type.
   * @deprecated Use {@link #greaterOrEqualTo(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector greaterOrEqualTo(BinaryOperable rhs, DType outType) {
    return greaterOrEqualTo(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * this >= rhs 1 is true 0 is false with the output cast to the given type.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector greaterOrEqualTo(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.GREATER_EQUAL, rhs, outType, stream, mr);
  }

  /**
   * this >= rhs 1 is true 0 is false.  The output type is BOOL8.
   * @deprecated Use {@link #greaterOrEqualTo(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector greaterOrEqualTo(BinaryOperable rhs) {
    return greaterOrEqualTo(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * this >= rhs 1 is true 0 is false.  The output type is BOOL8.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector greaterOrEqualTo(BinaryOperable rhs, Cuda.Stream stream,
      MemoryResources mr) {
    return greaterOrEqualTo(rhs, DType.BOOL8, stream, mr);
  }

  /**
   * Bit wise and (&) with the given output type. this & rhs
   * @deprecated Use {@link #bitAnd(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector bitAnd(BinaryOperable rhs, DType outType) {
    return bitAnd(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Bit wise and (&) with the given output type. this & rhs
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector bitAnd(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.BITWISE_AND, rhs, outType, stream, mr);
  }

  /**
   * Bit wise and (&). this & rhs
   * @deprecated Use {@link #bitAnd(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector bitAnd(BinaryOperable rhs) {
    return bitAnd(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Bit wise and (&). this & rhs
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector bitAnd(BinaryOperable rhs, Cuda.Stream stream, MemoryResources mr) {
    return bitAnd(rhs, implicitConversion(BinaryOp.BITWISE_AND, this, rhs), stream, mr);
  }

  /**
   * Bit wise or (|) with the given output type. this | rhs
   * @deprecated Use {@link #bitOr(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector bitOr(BinaryOperable rhs, DType outType) {
    return bitOr(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Bit wise or (|) with the given output type. this | rhs
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector bitOr(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.BITWISE_OR, rhs, outType, stream, mr);
  }

  /**
   * Bit wise or (|). this | rhs
   * @deprecated Use {@link #bitOr(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector bitOr(BinaryOperable rhs) {
    return bitOr(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Bit wise or (|). this | rhs
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector bitOr(BinaryOperable rhs, Cuda.Stream stream, MemoryResources mr) {
    return bitOr(rhs, implicitConversion(BinaryOp.BITWISE_OR, this, rhs), stream, mr);
  }

  /**
   * Bit wise xor (^) with the given output type. this ^ rhs
   * @deprecated Use {@link #bitXor(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector bitXor(BinaryOperable rhs, DType outType) {
    return bitXor(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Bit wise xor (^) with the given output type. this ^ rhs
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector bitXor(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.BITWISE_XOR, rhs, outType, stream, mr);
  }

  /**
   * Bit wise xor (^). this ^ rhs
   * @deprecated Use {@link #bitXor(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector bitXor(BinaryOperable rhs) {
    return bitXor(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Bit wise xor (^). this ^ rhs
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector bitXor(BinaryOperable rhs, Cuda.Stream stream, MemoryResources mr) {
    return bitXor(rhs, implicitConversion(BinaryOp.BITWISE_XOR, this, rhs), stream, mr);
  }

  /**
   * Logical and (&&) with the given output type. this && rhs
   * @deprecated Use {@link #and(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector and(BinaryOperable rhs, DType outType) {
    return and(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Logical and (&&) with the given output type. this && rhs
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector and(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.LOGICAL_AND, rhs, outType, stream, mr);
  }

  /**
   * Logical and (&&). this && rhs
   * @deprecated Use {@link #and(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector and(BinaryOperable rhs) {
    return and(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Logical and (&&). this && rhs
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector and(BinaryOperable rhs, Cuda.Stream stream, MemoryResources mr) {
    return and(rhs, implicitConversion(BinaryOp.LOGICAL_AND, this, rhs), stream, mr);
  }

  /**
   * Logical or (||) with the given output type. this || rhs
   * @deprecated Use {@link #or(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector or(BinaryOperable rhs, DType outType) {
    return or(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Logical or (||) with the given output type. this || rhs
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector or(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.LOGICAL_OR, rhs, outType, stream, mr);
  }

  /**
   * Logical or (||). this || rhs
   * @deprecated Use {@link #or(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector or(BinaryOperable rhs) {
    return or(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Logical or (||). this || rhs
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector or(BinaryOperable rhs, Cuda.Stream stream, MemoryResources mr) {
    return or(rhs, implicitConversion(BinaryOp.LOGICAL_OR, this, rhs), stream, mr);
  }

  /**
   * Bitwise left shifts the values of this vector by shiftBy.
   *
   * If "this" and shiftBy are both vectors then, this[i] << shiftBy[i]
   * If "this" is a scalar and shiftBy is a vector then returns a vector of size shiftBy.rows
   *    with the scalar << shiftBy[i]
   * If "this" is a vector and shiftBy is a scalar then returns a vector of size this.rows
   *    with this[i] << shiftBy
   *
   * @deprecated Use {@link #shiftLeft(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector shiftLeft(BinaryOperable shiftBy, DType outType) {
    return shiftLeft(shiftBy, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Bitwise left shifts the values of this vector by shiftBy.
   *
   * If "this" and shiftBy are both vectors then, this[i] << shiftBy[i]
   * If "this" is a scalar and shiftBy is a vector then returns a vector of size shiftBy.rows
   *    with the scalar << shiftBy[i]
   * If "this" is a vector and shiftBy is a scalar then returns a vector of size this.rows
   *    with this[i] << shiftBy
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector shiftLeft(BinaryOperable shiftBy, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.SHIFT_LEFT, shiftBy, outType, stream, mr);
  }

  /**
   * Bitwise left shift the values of this vector by the shiftBy.
   *
   * If "this" and shiftBy are both vectors then, this[i] << shiftBy[i]
   * If "this" is a scalar and shiftBy is a vector then returns a vector of size shiftBy.rows
   *    with the scalar << shiftBy[i]
   * If "this" is a vector and shiftBy is a scalar then returns a vector of size this.rows
   *    with this[i] << shiftBy
   * @deprecated Use {@link #shiftLeft(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector shiftLeft(BinaryOperable shiftBy) {
    return shiftLeft(shiftBy, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Bitwise left shift the values of this vector by the shiftBy.
   *
   * If "this" and shiftBy are both vectors then, this[i] << shiftBy[i]
   * If "this" is a scalar and shiftBy is a vector then returns a vector of size shiftBy.rows
   *    with the scalar << shiftBy[i]
   * If "this" is a vector and shiftBy is a scalar then returns a vector of size this.rows
   *    with this[i] << shiftBy
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector shiftLeft(BinaryOperable shiftBy, Cuda.Stream stream, MemoryResources mr) {
    return shiftLeft(shiftBy, implicitConversion(BinaryOp.SHIFT_LEFT, this, shiftBy), stream, mr);
  }

  /**
   * Bitwise right shift this vector by the shiftBy.
   *
   * If "this" and shiftBy are both vectors then, this[i] >> shiftBy[i]
   * If "this" is a scalar and shiftBy is a vector then returns a vector of size shiftBy.rows
   *    with the scalar >> shiftBy[i]
   * If "this" is a vector and shiftBy is a scalar then returns a vector of size this.rows
   *    with this[i] >> shiftBy
   * @deprecated Use {@link #shiftRight(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector shiftRight(BinaryOperable shiftBy, DType outType) {
    return shiftRight(shiftBy, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Bitwise right shift this vector by the shiftBy.
   *
   * If "this" and shiftBy are both vectors then, this[i] >> shiftBy[i]
   * If "this" is a scalar and shiftBy is a vector then returns a vector of size shiftBy.rows
   *    with the scalar >> shiftBy[i]
   * If "this" is a vector and shiftBy is a scalar then returns a vector of size this.rows
   *    with this[i] >> shiftBy
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector shiftRight(BinaryOperable shiftBy, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.SHIFT_RIGHT, shiftBy, outType, stream, mr);
  }

  /**
   * Bitwise right shift this vector by the shiftBy.
   *
   * If "this" and shiftBy are both vectors then, this[i] >> shiftBy[i]
   * If "this" is a scalar and shiftBy is a vector then returns a vector of size shiftBy.rows
   *    with the scalar >> shiftBy[i]
   * If "this" is a vector and shiftBy is a scalar then returns a vector of size this.rows
   *    with this[i] >> shiftBy
   * @deprecated Use {@link #shiftRight(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector shiftRight(BinaryOperable shiftBy) {
    return shiftRight(shiftBy, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Bitwise right shift this vector by the shiftBy.
   *
   * If "this" and shiftBy are both vectors then, this[i] >> shiftBy[i]
   * If "this" is a scalar and shiftBy is a vector then returns a vector of size shiftBy.rows
   *    with the scalar >> shiftBy[i]
   * If "this" is a vector and shiftBy is a scalar then returns a vector of size this.rows
   *    with this[i] >> shiftBy
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector shiftRight(BinaryOperable shiftBy, Cuda.Stream stream, MemoryResources mr) {
    return shiftRight(shiftBy, implicitConversion(BinaryOp.SHIFT_RIGHT, this, shiftBy), stream, mr);
  }

  /**
   * This method bitwise right shifts the values of this vector by the shiftBy.
   * This method always fills 0 irrespective of the sign of the number.
   *
   * If "this" and shiftBy are both vectors then, this[i] >>> shiftBy[i]
   * If "this" is a scalar and shiftBy is a vector then returns a vector of size shiftBy.rows
   *    with the scalar >>> shiftBy[i]
   * If "this" is a vector and shiftBy is a scalar then returns a vector of size this.rows
   *    with this[i] >>> shiftBy
   * @deprecated Use
   *     {@link #shiftRightUnsigned(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector shiftRightUnsigned(BinaryOperable shiftBy, DType outType) {
    return shiftRightUnsigned(shiftBy, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * This method bitwise right shifts the values of this vector by the shiftBy.
   * This method always fills 0 irrespective of the sign of the number.
   *
   * If "this" and shiftBy are both vectors then, this[i] >>> shiftBy[i]
   * If "this" is a scalar and shiftBy is a vector then returns a vector of size shiftBy.rows
   *    with the scalar >>> shiftBy[i]
   * If "this" is a vector and shiftBy is a scalar then returns a vector of size this.rows
   *    with this[i] >>> shiftBy
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector shiftRightUnsigned(BinaryOperable shiftBy, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.SHIFT_RIGHT_UNSIGNED, shiftBy, outType, stream, mr);
  }

  /**
   * This method bitwise right shifts the values of this vector by the shiftBy.
   * This method always fills 0 irrespective of the sign of the number.
   *
   * If "this" and shiftBy are both vectors then, this[i] >>> shiftBy[i]
   * If "this" is a scalar and shiftBy is a vector then returns a vector of size shiftBy.rows
   *    with the scalar >>> shiftBy[i]
   * If "this" is a vector and shiftBy is a scalar then returns a vector of size this.rows
   *    with this[i] >>> shiftBy
   * @deprecated Use {@link #shiftRightUnsigned(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector shiftRightUnsigned(BinaryOperable shiftBy) {
    return shiftRightUnsigned(shiftBy, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * This method bitwise right shifts the values of this vector by the shiftBy.
   * This method always fills 0 irrespective of the sign of the number.
   *
   * If "this" and shiftBy are both vectors then, this[i] >>> shiftBy[i]
   * If "this" is a scalar and shiftBy is a vector then returns a vector of size shiftBy.rows
   *    with the scalar >>> shiftBy[i]
   * If "this" is a vector and shiftBy is a scalar then returns a vector of size this.rows
   *    with this[i] >>> shiftBy
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector shiftRightUnsigned(BinaryOperable shiftBy, Cuda.Stream stream,
      MemoryResources mr) {
    return shiftRightUnsigned(shiftBy, implicitConversion(BinaryOp.SHIFT_RIGHT_UNSIGNED, this,
        shiftBy), stream, mr);
  }

  /**
   * Calculate the log with the specified base
   * @deprecated Use {@link #log(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector log(BinaryOperable rhs, DType outType) {
    return log(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Calculate the log with the specified base
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector log(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.LOG_BASE, rhs, outType, stream, mr);
  }

  /**
   * Calculate the log with the specified base, output is the same as this.
   * @deprecated Use {@link #log(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector log(BinaryOperable rhs) {
    return log(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Calculate the log with the specified base, output is the same as this.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector log(BinaryOperable rhs, Cuda.Stream stream, MemoryResources mr) {
    return log(rhs, getType(), stream, mr);
  }

  /**
   * The function arctan2(y,x) or atan2(y,x) is defined as the angle in the Euclidean plane, given
   * in radians, between the positive x axis and the ray to the point (x, y) ≠ (0, 0).
   * @deprecated Use {@link #arctan2(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector arctan2(BinaryOperable xCoordinate, DType outType) {
    return arctan2(xCoordinate, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * The function arctan2(y,x) or atan2(y,x) is defined as the angle in the Euclidean plane, given
   * in radians, between the positive x axis and the ray to the point (x, y) ≠ (0, 0).
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector arctan2(BinaryOperable xCoordinate, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.ATAN2, xCoordinate, outType, stream, mr);
  }

  /**
   * The function arctan2(y,x) or atan2(y,x) is defined as the angle in the Euclidean plane, given
   * in radians, between the positive x axis and the ray to the point (x, y) ≠ (0, 0).
   * @deprecated Use {@link #arctan2(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector arctan2(BinaryOperable xCoordinate) {
    return arctan2(xCoordinate, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * The function arctan2(y,x) or atan2(y,x) is defined as the angle in the Euclidean plane, given
   * in radians, between the positive x axis and the ray to the point (x, y) ≠ (0, 0).
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector arctan2(BinaryOperable xCoordinate, Cuda.Stream stream, MemoryResources mr) {
    return arctan2(xCoordinate, implicitConversion(BinaryOp.ATAN2, this, xCoordinate), stream, mr);
  }

  /**
   * Returns the positive value of lhs mod rhs.
   *
   * r = lhs % rhs
   * if r < 0 then (r + rhs) % rhs
   * else r
   *
   * @deprecated Use {@link #pmod(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector pmod(BinaryOperable rhs, DType outputType) {
    return pmod(rhs, outputType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Returns the positive value of lhs mod rhs.
   *
   * r = lhs % rhs
   * if r < 0 then (r + rhs) % rhs
   * else r
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector pmod(BinaryOperable rhs, DType outputType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.PMOD, rhs, outputType, stream, mr);
  }

  /**
   * Returns the positive value of lhs mod rhs.
   *
   * r = lhs % rhs
   * if r < 0 then (r + rhs) % rhs
   * else r
   *
   * @deprecated Use {@link #pmod(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector pmod(BinaryOperable rhs) {
    return pmod(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Returns the positive value of lhs mod rhs.
   *
   * r = lhs % rhs
   * if r < 0 then (r + rhs) % rhs
   * else r
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector pmod(BinaryOperable rhs, Cuda.Stream stream, MemoryResources mr) {
    return pmod(rhs, implicitConversion(BinaryOp.PMOD, this, rhs), stream, mr);
  }

  /**
   * like equalTo but NULL == NULL is TRUE and NULL == not NULL is FALSE
   * @deprecated Use {@link #equalToNullAware(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector equalToNullAware(BinaryOperable rhs, DType outType) {
    return equalToNullAware(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * like equalTo but NULL == NULL is TRUE and NULL == not NULL is FALSE
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector equalToNullAware(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.NULL_EQUALS, rhs, outType, stream, mr);
  }

  /**
   * like equalTo but NULL == NULL is TRUE and NULL == not NULL is FALSE
   * @deprecated Use {@link #equalToNullAware(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector equalToNullAware(BinaryOperable rhs) {
    return equalToNullAware(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * like equalTo but NULL == NULL is TRUE and NULL == not NULL is FALSE
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector equalToNullAware(BinaryOperable rhs, Cuda.Stream stream,
      MemoryResources mr) {
    return equalToNullAware(rhs, DType.BOOL8, stream, mr);
  }

  /**
   * like notEqualTo but NULL != NULL is TRUE and NULL != not NULL is FALSE
   * @deprecated Use
   *     {@link #notEqualToNullAware(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector notEqualToNullAware(BinaryOperable rhs, DType outType) {
    return notEqualToNullAware(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * like notEqualTo but NULL != NULL is TRUE and NULL != not NULL is FALSE
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector notEqualToNullAware(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.NULL_NOT_EQUALS, rhs, outType, stream, mr);
  }

  /**
   * like notEqualTo but NULL != NULL is TRUE and NULL != not NULL is FALSE
   * @deprecated Use {@link #notEqualToNullAware(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector notEqualToNullAware(BinaryOperable rhs) {
    return notEqualToNullAware(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * like notEqualTo but NULL != NULL is TRUE and NULL != not NULL is FALSE
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector notEqualToNullAware(BinaryOperable rhs, Cuda.Stream stream,
      MemoryResources mr) {
    return notEqualToNullAware(rhs, DType.BOOL8, stream, mr);
  }

  /**
   * Returns the max non null value.
   * @deprecated Use {@link #maxNullAware(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector maxNullAware(BinaryOperable rhs, DType outType) {
    return maxNullAware(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Returns the max non null value.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector maxNullAware(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.NULL_MAX, rhs, outType, stream, mr);
  }

  /**
   * Returns the max non null value.
   * @deprecated Use {@link #maxNullAware(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector maxNullAware(BinaryOperable rhs) {
    return maxNullAware(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Returns the max non null value.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector maxNullAware(BinaryOperable rhs, Cuda.Stream stream, MemoryResources mr) {
    return maxNullAware(rhs, implicitConversion(BinaryOp.NULL_MAX, this, rhs), stream, mr);
  }

  /**
   * Returns the min non null value.
   * @deprecated Use {@link #minNullAware(BinaryOperable, DType, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector minNullAware(BinaryOperable rhs, DType outType) {
    return minNullAware(rhs, outType, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Returns the min non null value.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector minNullAware(BinaryOperable rhs, DType outType, Cuda.Stream stream,
      MemoryResources mr) {
    return binaryOp(BinaryOp.NULL_MIN, rhs, outType, stream, mr);
  }

  /**
   * Returns the min non null value.
   * @deprecated Use {@link #minNullAware(BinaryOperable, Cuda.Stream, MemoryResources)}.
   */
  @Deprecated
  default ColumnVector minNullAware(BinaryOperable rhs) {
    return minNullAware(rhs, Cuda.DEFAULT_STREAM, MemoryResources.CURRENT);
  }

  /**
   * Returns the min non null value.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr device memory resources for the returned column and for temporary allocations
   */
  default ColumnVector minNullAware(BinaryOperable rhs, Cuda.Stream stream, MemoryResources mr) {
    return minNullAware(rhs, implicitConversion(BinaryOp.NULL_MIN, this, rhs), stream, mr);
  }

}
