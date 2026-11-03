/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * An aggregation that can be used for a reduce.
 */
public final class GroupByAggregation {
  private final Aggregation wrapped;

  private GroupByAggregation(Aggregation wrapped) {
    this.wrapped = wrapped;
  }

  Aggregation getWrapped() {
    return wrapped;
  }


  /**
   * Add a column to the Aggregation so it can be used on a specific column of data.
   * @param columnIndex the index of the column to operate on.
   */
  public GroupByAggregationOnColumn onColumn(int columnIndex) {
    return new GroupByAggregationOnColumn(this, columnIndex);
  }

  @Override
  public int hashCode() {
    return wrapped.hashCode();
  }

  @Override
  public boolean equals(Object other) {
    if (other == this) {
      return true;
    } else if (other instanceof GroupByAggregation) {
      GroupByAggregation o = (GroupByAggregation) other;
      return wrapped.equals(o.wrapped);
    }
    return false;
  }

  /**
   * Count number of valid, a.k.a. non-null, elements.
   */
  public static GroupByAggregation count() {
    return new GroupByAggregation(Aggregation.count());
  }

  /**
   * Count number of elements.
   * @param nullPolicy INCLUDE if nulls should be counted. EXCLUDE if only non-null values
   *                   should be counted.
   */
  public static GroupByAggregation count(NullPolicy nullPolicy) {
    return new GroupByAggregation(Aggregation.count(nullPolicy));
  }

  /**
   * Sum Aggregation
   */
  public static GroupByAggregation sum() {
    return new GroupByAggregation(Aggregation.sum());
  }

  /**
   * Sum aggregation that also reports per-group overflow. The result column is a
   * STRUCT with children {sum: input-type, overflow: BOOL8}. Supported input
   * types are signed integers (INT8/16/32/64) and fixed-point decimal
   * (DECIMAL32/64/128). The sum-child has the same type AND scale as the input
   * column -- e.g. a DECIMAL64 input at scale -4 produces a DECIMAL64 sum-child
   * at scale -4; cudf does not widen or rescale. On overflow the sum value is
   * unspecified; the boolean flag is the source of truth. Both the hash-based
   * and sort-based groupby paths are supported.
   */
  public static GroupByAggregation sumOverflow() {
    return new GroupByAggregation(Aggregation.sumOverflow());
  }

  /**
   * Product Aggregation.
   */
  public static GroupByAggregation product() {
    return new GroupByAggregation(Aggregation.product());
  }


  /**
   * Index of max element. Please note that when using this aggregation if the
   * data is not already sorted by the grouping keys it may be automatically sorted
   * prior to doing the aggregation. This would result in an index into the sorted data being
   * returned.
   */
  public static GroupByAggregation argMax() {
    return new GroupByAggregation(Aggregation.argMax());
  }

  /**
   * Index of min element. Please note that when using this aggregation if the
   * data is not already sorted by the grouping keys it may be automatically sorted
   * prior to doing the aggregation. This would result in an index into the sorted data being
   * returned.
   */
  public static GroupByAggregation argMin() {
    return new GroupByAggregation(Aggregation.argMin());
  }

  /**
   * Min Aggregation
   */
  public static GroupByAggregation min() {
    return new GroupByAggregation(Aggregation.min());
  }

  /**
   * Max Aggregation
   */
  public static GroupByAggregation max() {
    return new GroupByAggregation(Aggregation.max());
  }

  /**
   * Arithmetic mean reduction.
   */
  public static GroupByAggregation mean() {
    return new GroupByAggregation(Aggregation.mean());
  }

  /**
   * Sum of square of differences from mean.
   */
  public static GroupByAggregation M2() {
    return new GroupByAggregation(Aggregation.M2());
  }

  /**
   * Variance aggregation with 1 as the delta degrees of freedom.
   */
  public static GroupByAggregation variance() {
    return new GroupByAggregation(Aggregation.variance());
  }

  /**
   * Variance aggregation.
   * @param ddof delta degrees of freedom. The divisor used in calculation of variance is
   *             <code>N - ddof</code>, where N is the population size.
   */
  public static GroupByAggregation variance(int ddof) {
    return new GroupByAggregation(Aggregation.variance(ddof));
  }

  /**
   * Standard deviation aggregation with 1 as the delta degrees of freedom.
   */
  public static GroupByAggregation standardDeviation() {
    return new GroupByAggregation(Aggregation.standardDeviation());
  }

  /**
   * Standard deviation aggregation.
   * @param ddof delta degrees of freedom. The divisor used in calculation of std is
   *             <code>N - ddof</code>, where N is the population size.
   */
  public static GroupByAggregation standardDeviation(int ddof) {
    return new GroupByAggregation(Aggregation.standardDeviation(ddof));
  }

  /**
   * Aggregate to compute the specified quantiles. Uses linear interpolation by default.
   */
  public static GroupByAggregation quantile(double ... quantiles) {
    return new GroupByAggregation(Aggregation.quantile(quantiles));
  }

  /**
   * Aggregate to compute various quantiles.
   */
  public static GroupByAggregation quantile(QuantileMethod method, double ... quantiles) {
    return new GroupByAggregation(Aggregation.quantile(method, quantiles));
  }

  /**
   * Median reduction.
   */
  public static GroupByAggregation median() {
    return new GroupByAggregation(Aggregation.median());
  }

  /**
   * Number of unique, non-null, elements.
   */
  public static GroupByAggregation nunique() {
    return new GroupByAggregation(Aggregation.nunique());
  }

  /**
   * Number of unique elements.
   * @param nullPolicy INCLUDE if nulls should be counted else EXCLUDE. If nulls are counted they
   *                   compare as equal so multiple null values in a range would all only
   *                   increase the count by 1.
   */
  public static GroupByAggregation nunique(NullPolicy nullPolicy) {
    return new GroupByAggregation(Aggregation.nunique(nullPolicy));
  }

  /**
   * Get the nth, non-null, element in a group.
   * @param offset the offset to look at. Negative numbers go from the end of the group. Any
   *               value outside of the group range results in a null.
   */
  public static GroupByAggregation nth(int offset) {
    return new GroupByAggregation(Aggregation.nth(offset));
  }

  /**
   * Get the nth element in a group.
   * @param offset the offset to look at. Negative numbers go from the end of the group. Any
   *               value outside of the group range results in a null.
   * @param nullPolicy INCLUDE if nulls should be included in the aggregation or EXCLUDE if they
   *                   should be skipped.
   */
  public static GroupByAggregation nth(int offset, NullPolicy nullPolicy) {
    return new GroupByAggregation(Aggregation.nth(offset, nullPolicy));
  }

  /**
   * Collect the values into a list. Nulls will be skipped.
   */
  public static GroupByAggregation collectList() {
    return new GroupByAggregation(Aggregation.collectList());
  }

  /**
   * Collect the values into a list.
   *
   * @param nullPolicy Indicates whether to include/exclude nulls during collection.
   */
  public static GroupByAggregation collectList(NullPolicy nullPolicy) {
    return new GroupByAggregation(Aggregation.collectList(nullPolicy));
  }

  /**
   * Collect the values into a set. All null values will be excluded, and all NaN values are regarded as
   * unique instances.
   */
  public static GroupByAggregation collectSet() {
    return new GroupByAggregation(Aggregation.collectSet());
  }

  /**
   * Collect the values into a set.
   *
   * @param nullPolicy   Indicates whether to include/exclude nulls during collection.
   * @param nullEquality Flag to specify whether null entries within each list should be considered equal.
   * @param nanEquality  Flag to specify whether NaN values in floating point column should be considered equal.
   */
  public static GroupByAggregation collectSet(NullPolicy nullPolicy, NullEquality nullEquality, NaNEquality nanEquality) {
    return new GroupByAggregation(Aggregation.collectSet(nullPolicy, nullEquality, nanEquality));
  }

  /**
   * Merge the partial lists produced by multiple CollectListAggregations.
   * NOTICE: The partial lists to be merged should NOT include any null list element (but can include null list entries).
   */
  public static GroupByAggregation mergeLists() {
    return new GroupByAggregation(Aggregation.mergeLists());
  }

  /**
   * Merge the partial sets produced by multiple CollectSetAggregations. Each null/NaN value will be regarded as
   * a unique instance.
   */
  public static GroupByAggregation mergeSets() {
    return new GroupByAggregation(Aggregation.mergeSets());
  }

  /**
   * Execute an aggregation using a host-side user-defined function (UDF).
   * @param wrapper The wrapper for the native host UDF instance.
   * @return A new GroupByAggregation instance
   */
  public static GroupByAggregation hostUDF(HostUDFWrapper wrapper) {
    return new GroupByAggregation(Aggregation.hostUDF(wrapper));
  }

  /**
   * Merge the partial sets produced by multiple CollectSetAggregations.
   *
   * @param nullEquality Flag to specify whether null entries within each list should be considered equal.
   * @param nanEquality  Flag to specify whether NaN values in floating point column should be considered equal.
   */
  public static GroupByAggregation mergeSets(NullEquality nullEquality, NaNEquality nanEquality) {
    return new GroupByAggregation(Aggregation.mergeSets(nullEquality, nanEquality));
  }

  /**
   * Merge the partial M2 values produced by multiple instances of M2Aggregation.
   */
  public static GroupByAggregation mergeM2() {
    return new GroupByAggregation(Aggregation.mergeM2());
  }

  /**
   * Compute a t-digest from on a fixed-width numeric input column.
   *
   * @param delta Required accuracy (number of buckets).
   * @return A list of centroids per grouping, where each centroid has a mean value and a
   *         weight. The number of centroids will be <= delta.
   */
  public static GroupByAggregation createTDigest(int delta) {
    return new GroupByAggregation(Aggregation.createTDigest(delta));
  }

  /**
   * Merge t-digests.
   *
   * @param delta Required accuracy (number of buckets).
   * @return A list of centroids per grouping, where each centroid has a mean value and a
   *         weight. The number of centroids will be <= delta.
   */
  public static GroupByAggregation mergeTDigest(int delta) {
    return new GroupByAggregation(Aggregation.mergeTDigest(delta));
  }

  /**
   * Histogram aggregation, computing the frequencies for each unique row.
   *
   * A histogram is given as a lists column, in which the first child stores unique rows from
   * the input values and the second child stores their corresponding frequencies.
   *
   * @return A lists of structs column in which each list contains a histogram corresponding to
   *         an input key.
   */
  public static GroupByAggregation histogram() {
    return new GroupByAggregation(Aggregation.histogram());
  }

  /**
   * MergeHistogram aggregation, to merge multiple histograms.
   *
   * @return A new histogram in which the frequencies of the unique rows are sum up.
   */
  public static GroupByAggregation mergeHistogram() {
    return new GroupByAggregation(Aggregation.mergeHistogram());
  }

  /**
   * Bitwise AND aggregation, computing the bitwise AND of all non-null values in a group.
   */
  public static GroupByAggregation bitAnd() {
    return new GroupByAggregation(Aggregation.bitAnd());
  }

  /**
   * Bitwise OR aggregation, computing the bitwise OR of all non-null values in a group.
   */
  public static GroupByAggregation bitOr() {
    return new GroupByAggregation(Aggregation.bitOr());
  }

  /**
   * Bitwise XOR aggregation, computing the bitwise XOR of all non-null values in a group.
   */
  public static GroupByAggregation bitXor() {
    return new GroupByAggregation(Aggregation.bitXor());
  }
}
