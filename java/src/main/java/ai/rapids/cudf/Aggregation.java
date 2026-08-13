/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.util.Arrays;

/**
 * Represents an aggregation operation.  Please note that not all aggregations work, or even make
 * sense in all types of aggregation operations.
 */
abstract class Aggregation {
    static {
        NativeDepsLoader.loadNativeDeps();
    }

    /*
     * This should be kept in sync with AggregationJni.cpp.  Note that the nativeId here is not the
     * same as the C++ cudf::aggregation::Kind.  They are very closely related, but both are
     * implementation details and generally should be hidden from the end user.
     * Visible for testing.
     */
    enum Kind {
        SUM(0),
        PRODUCT(1),
        MIN(2),
        MAX(3),
        COUNT(4),
        ANY(5),
        ALL(6),
        SUM_OF_SQUARES(7),
        MEAN(8),
        VARIANCE(9), // This can take a delta degrees of freedom
        STD(10), // This can take a delta degrees of freedom
        MEDIAN(11),
        QUANTILE(12),
        ARGMAX(13),
        ARGMIN(14),
        NUNIQUE(15),
        NTH_ELEMENT(16),
        ROW_NUMBER(17),
        COLLECT_LIST(18),
        COLLECT_SET(19),
        MERGE_LISTS(20),
        MERGE_SETS(21),
        LEAD(22),
        LAG(23),
        PTX(24),
        CUDA(25),
        HOST_UDF(26),
        M2(27),
        MERGE_M2(28),
        RANK(29),
        DENSE_RANK(30),
        PERCENT_RANK(31),
        TDIGEST(32), // This can take a delta argument for accuracy level
        MERGE_TDIGEST(33), // This can take a delta argument for accuracy level
        HISTOGRAM(34),
        MERGE_HISTOGRAM(35),
        BITWISE_AGG(36),
        SUM_OVERFLOW(37);

        final int nativeId;

        Kind(int nativeId) {this.nativeId = nativeId;}
    }

    /**
     * An Aggregation that only needs a kind and nothing else.
     */
    private static class NoParamAggregation extends Aggregation {
        public NoParamAggregation(Kind kind) {
            super(kind);
        }

        @Override
        long createNativeInstance() {
            return Aggregation.createNoParamAgg(kind.nativeId);
        }

        @Override
        public int hashCode() {
            return kind.hashCode();
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            } else if (other instanceof NoParamAggregation) {
                NoParamAggregation o = (NoParamAggregation) other;
                return o.kind.equals(this.kind);
            }
            return false;
        }
    }

    static final class NthAggregation extends Aggregation {
        private final int offset;
        private final NullPolicy nullPolicy;

        private NthAggregation(int offset, NullPolicy nullPolicy) {
            super(Kind.NTH_ELEMENT);
            this.offset = offset;
            this.nullPolicy = nullPolicy;
        }

        @Override
        long createNativeInstance() {
            return Aggregation.createNthAgg(offset, nullPolicy.includeNulls);
        }

        @Override
        public int hashCode() {
            return 31 * offset + nullPolicy.hashCode();
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            } else if (other instanceof NthAggregation) {
                NthAggregation o = (NthAggregation) other;
                return o.offset == this.offset && o.nullPolicy == this.nullPolicy;
            }
            return false;
        }
    }

    private static class DdofAggregation extends Aggregation {
        private final int ddof;

        public DdofAggregation(Kind kind, int ddof) {
            super(kind);
            this.ddof = ddof;
        }

        @Override
        long createNativeInstance() {
            return Aggregation.createDdofAgg(kind.nativeId, ddof);
        }

        @Override
        public int hashCode() {
            return 31 * kind.hashCode() + ddof;
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            } else if (other instanceof DdofAggregation) {
                DdofAggregation o = (DdofAggregation) other;
                return o.ddof == this.ddof;
            }
            return false;
        }
    }

    private static class CountLikeAggregation extends Aggregation {
        private final NullPolicy nullPolicy;

        public CountLikeAggregation(Kind kind, NullPolicy nullPolicy) {
            super(kind);
            this.nullPolicy = nullPolicy;
        }

        @Override
        long createNativeInstance() {
            return Aggregation.createCountLikeAgg(kind.nativeId, nullPolicy.includeNulls);
        }

        @Override
        public int hashCode() {
            return 31 * kind.hashCode() + nullPolicy.hashCode();
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            } else if (other instanceof CountLikeAggregation) {
                CountLikeAggregation o = (CountLikeAggregation) other;
                return o.nullPolicy == this.nullPolicy;
            }
            return false;
        }
    }

    private static final class QuantileAggregation extends Aggregation {
        private final QuantileMethod method;
        private final double[] quantiles;

        public QuantileAggregation(QuantileMethod method, double[] quantiles) {
            super(Kind.QUANTILE);
            this.method = method;
            this.quantiles = quantiles;
        }

        @Override
        long createNativeInstance() {
            return Aggregation.createQuantAgg(method.nativeId, quantiles);
        }

        @Override
        public int hashCode() {
            return 31 * (31 * kind.hashCode() + method.hashCode()) + Arrays.hashCode(quantiles);
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            } else if (other instanceof QuantileAggregation) {
                QuantileAggregation o = (QuantileAggregation) other;
                return this.method == o.method && Arrays.equals(this.quantiles, o.quantiles);
            }
            return false;
        }
    }

    private static class LeadLagAggregation extends Aggregation {
        private final int offset;
        private final ColumnVector defaultOutput;

        LeadLagAggregation(Kind kind, int offset, ColumnVector defaultOutput) {
            super(kind);
            this.offset = offset;
            this.defaultOutput = defaultOutput;
        }

        @Override
        long createNativeInstance() {
            // Default output comes from a different path
            return Aggregation.createLeadLagAgg(kind.nativeId, offset);
        }

        @Override
        public int hashCode() {
            int ret = 31 * kind.hashCode() + offset;
            if (defaultOutput != null) {
                ret = 31 * ret + defaultOutput.hashCode();
            }
            return ret;
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            } else if (other instanceof LeadLagAggregation) {
                LeadLagAggregation o = (LeadLagAggregation) other;
                boolean ret = o.kind == this.kind && this.offset == o.offset;
                if (defaultOutput != null) {
                    ret = ret && defaultOutput.equals(o.defaultOutput);
                } else if (o.defaultOutput != null) {
                    // defaultOutput == null and o.defaultOutput != null so they are not equal
                    ret = false;
                } // else they are both null which is the same and a noop.
                return ret;
            }
            return false;
        }

        @Override
        long getDefaultOutput() {
            return defaultOutput == null ? 0 : defaultOutput.getNativeView();
        }
    }

    static final class CollectListAggregation extends Aggregation {
        private final NullPolicy nullPolicy;

        private CollectListAggregation(NullPolicy nullPolicy) {
            super(Kind.COLLECT_LIST);
            this.nullPolicy = nullPolicy;
        }

        @Override
        long createNativeInstance() {
            return Aggregation.createCollectListAgg(nullPolicy.includeNulls);
        }

        @Override
        public int hashCode() {
            return 31 * kind.hashCode() + nullPolicy.hashCode();
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            } else if (other instanceof CollectListAggregation) {
                CollectListAggregation o = (CollectListAggregation) other;
                return o.nullPolicy == this.nullPolicy;
            }
            return false;
        }
    }

    static final class CollectSetAggregation extends Aggregation {
        private final NullPolicy nullPolicy;
        private final NullEquality nullEquality;
        private final NaNEquality nanEquality;

        private CollectSetAggregation(NullPolicy nullPolicy, NullEquality nullEquality, NaNEquality nanEquality) {
            super(Kind.COLLECT_SET);
            this.nullPolicy = nullPolicy;
            this.nullEquality = nullEquality;
            this.nanEquality = nanEquality;
        }

        @Override
        long createNativeInstance() {
            return Aggregation.createCollectSetAgg(nullPolicy.includeNulls,
                nullEquality.nullsEqual,
                nanEquality.nansEqual);
        }

        @Override
        public int hashCode() {
            return 31 * kind.hashCode()
                + Boolean.hashCode(nullPolicy.includeNulls)
                + Boolean.hashCode(nullEquality.nullsEqual)
                + Boolean.hashCode(nanEquality.nansEqual);
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            } else if (other instanceof CollectSetAggregation) {
                CollectSetAggregation o = (CollectSetAggregation) other;
                return o.nullPolicy == this.nullPolicy &&
                    o.nullEquality == this.nullEquality &&
                    o.nanEquality == this.nanEquality;
            }
            return false;
        }
    }

    static final class MergeSetsAggregation extends Aggregation {
        private final NullEquality nullEquality;
        private final NaNEquality nanEquality;

        private MergeSetsAggregation(NullEquality nullEquality, NaNEquality nanEquality) {
            super(Kind.MERGE_SETS);
            this.nullEquality = nullEquality;
            this.nanEquality = nanEquality;
        }

        @Override
        long createNativeInstance() {
            return Aggregation.createMergeSetsAgg(nullEquality.nullsEqual, nanEquality.nansEqual);
        }

        @Override
        public int hashCode() {
            return 31 * kind.hashCode()
                + Boolean.hashCode(nullEquality.nullsEqual)
                + Boolean.hashCode(nanEquality.nansEqual);
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            } else if (other instanceof MergeSetsAggregation) {
                MergeSetsAggregation o = (MergeSetsAggregation) other;
                return o.nullEquality == this.nullEquality && o.nanEquality == this.nanEquality;
            }
            return false;
        }
    }

    static final class HostUDFAggregation extends Aggregation {
        private final HostUDFWrapper wrapper;

        private HostUDFAggregation(HostUDFWrapper wrapper) {
            super(Kind.HOST_UDF);
            this.wrapper = wrapper;
        }

        @Override
        long createNativeInstance() {
            long udf = 0;
            try {
                udf = wrapper.createUDFInstance();
                return Aggregation.createHostUDFAgg(udf);
            } finally {
                // a new UDF is cloned in `createHostUDFAgg`, here should close the UDF instance.
                if (udf != 0) {
                    HostUDFWrapper.closeUDFInstance(udf);
                }
            }
        }

        @Override
        public int hashCode() {
            return 31 * kind.hashCode() + wrapper.hashCode();
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            } else if (other instanceof HostUDFAggregation) {
                return wrapper.equals(((HostUDFAggregation) other).wrapper);
            }
            return false;
        }
    }

    static final class BitAndAggregation extends Aggregation {
        private BitAndAggregation() {
            super(Kind.BITWISE_AGG);
        }

        @Override
        long createNativeInstance() {
            return Aggregation.createBitAndAgg();
        }

        @Override
        public int hashCode() {
            return 31 * kind.hashCode() + ("AND").hashCode();
        }

        @Override
        public boolean equals(Object other) {
            return other instanceof BitAndAggregation;
        }
    }

    static final class BitOrAggregation extends Aggregation {
        private BitOrAggregation() {
            super(Kind.BITWISE_AGG);
        }

        @Override
        long createNativeInstance() {
            return Aggregation.createBitOrAgg();
        }

        @Override
        public int hashCode() {
            return 31 * kind.hashCode() + ("OR").hashCode();
        }

        @Override
        public boolean equals(Object other) {
            return other instanceof BitAndAggregation;
        }
    }

    static final class BitXorAggregation extends Aggregation {
        private BitXorAggregation() {
            super(Kind.BITWISE_AGG);
        }

        @Override
        long createNativeInstance() {
            return Aggregation.createBitXorAgg();
        }

        @Override
        public int hashCode() {
            return 31 * kind.hashCode() + ("XOR").hashCode();
        }

        @Override
        public boolean equals(Object other) {
            return other instanceof BitAndAggregation;
        }
    }

    protected final Kind kind;

    protected Aggregation(Kind kind) {
        this.kind = kind;
    }

    /**
     * Get the native view of a ColumnVector that provides default values to be used for some window
     * aggregations when there is not enough data to do the computation.  This really only happens
     * for a very few number of window aggregations. Also note that the ownership and life cycle of
     * the column is controlled outside of this, so don't try to close it.
     * @return the native view of the column vector or 0.
     */
    long getDefaultOutput() {
        return 0;
    }

    /**
     * returns a <code>cudf::aggregation *</code> cast to a long. We don't want to force users to
     * close an Aggregation. Because of this Aggregation objects are created in pure java, but when
     * it is time to use them this method is called to return a pointer to the c++ aggregation
     * instance. All values returned by this can be used multiple times, and should be closed by
     * calling the static close method. Yes, this creates a lot more JNI calls, but it keeps the
     * user API clean.
     */
    abstract long createNativeInstance();

    @Override
    public abstract int hashCode();

    @Override
    public abstract boolean equals(Object other);

    static void close(long[] ptrs) {
        for (long ptr: ptrs) {
            if (ptr != 0) {
                close(ptr);
            }
        }
    }

    static native void close(long ptr);

    static final class SumAggregation extends NoParamAggregation {
        private SumAggregation() {
            super(Kind.SUM);
        }
    }

    /**
     * Sum reduction.
     */
    static SumAggregation sum() {
        return new SumAggregation();
    }

    static final class SumOverflowAggregation extends NoParamAggregation {
        private SumOverflowAggregation() {
            super(Kind.SUM_OVERFLOW);
        }
    }

    /**
     * Sum aggregation that also reports overflow. The result is a struct with
     * children {sum: same type as input, overflow: BOOL8}. The input may be any
     * signed integer type (INT8/16/32/64) or fixed-point decimal
     * (DECIMAL32/64/128), for both column reductions and hash-based groupby.
     * On overflow the sum value is unspecified; the boolean flag is the source of
     * truth. Sort-based groupby, scan, segmented reduce, and rolling are not
     * supported by cudf.
     */
    static SumOverflowAggregation sumOverflow() {
        return new SumOverflowAggregation();
    }

    static final class ProductAggregation extends NoParamAggregation {
        private ProductAggregation() {
            super(Kind.PRODUCT);
        }
    }

    /**
     * Product reduction.
     */
    static ProductAggregation product() {
        return new ProductAggregation();
    }

    static final class MinAggregation extends NoParamAggregation {
        private MinAggregation() {
            super(Kind.MIN);
        }
    }

    /**
     * Min reduction.
     */
    static MinAggregation min() {
        return new MinAggregation();
    }

    static final class MaxAggregation extends NoParamAggregation {
        private MaxAggregation() {
            super(Kind.MAX);
        }
    }

    /**
     * Max reduction.
     */
    static MaxAggregation max() {
        return new MaxAggregation();
    }

    static final class CountAggregation extends CountLikeAggregation {
        private CountAggregation(NullPolicy nullPolicy) {
            super(Kind.COUNT, nullPolicy);
        }
    }

    /**
     * Count number of valid, a.k.a. non-null, elements.
     */
    static CountAggregation count() {
        return count(NullPolicy.EXCLUDE);
    }

    /**
     * Count number of elements.
     * @param nullPolicy INCLUDE if nulls should be counted. EXCLUDE if only non-null values
     *                   should be counted.
     */
    static CountAggregation count(NullPolicy nullPolicy) {
        return new CountAggregation(nullPolicy);
    }

    static final class AnyAggregation extends NoParamAggregation {
        private AnyAggregation() {
            super(Kind.ANY);
        }
    }

    /**
     * Any reduction. Produces a true or 1, depending on the output type,
     * if any of the elements in the range are true or non-zero, otherwise produces a false or 0.
     * Null values are skipped.
     */
    static AnyAggregation any() {
        return new AnyAggregation();
    }

    static final class AllAggregation extends NoParamAggregation {
        private AllAggregation() {
            super(Kind.ALL);
        }
    }

    /**
     * All reduction. Produces true or 1, depending on the output type, if all of the elements in
     * the range are true or non-zero, otherwise produces a false or 0.
     * Null values are skipped.
     */
    static AllAggregation all() {
        return new AllAggregation();
    }

    static final class SumOfSquaresAggregation extends NoParamAggregation {
        private SumOfSquaresAggregation() {
            super(Kind.SUM_OF_SQUARES);
        }
    }

    /**
     * Sum of squares reduction.
     */
    static SumOfSquaresAggregation sumOfSquares() {
        return new SumOfSquaresAggregation();
    }

    static final class MeanAggregation extends NoParamAggregation {
        private MeanAggregation() {
            super(Kind.MEAN);
        }
    }

    /**
     * Arithmetic mean reduction.
     */
    static MeanAggregation mean() {
        return new MeanAggregation();
    }

    static final class M2Aggregation extends NoParamAggregation {
        private M2Aggregation() {
            super(Kind.M2);
        }
    }

    /**
     * Sum of square of differences from mean.
     */
    static M2Aggregation M2() {
        return new M2Aggregation();
    }

    static final class VarianceAggregation extends DdofAggregation {
        private VarianceAggregation(int ddof) {
            super(Kind.VARIANCE, ddof);
        }
    }

    /**
     * Variance aggregation with 1 as the delta degrees of freedom.
     */
    static VarianceAggregation variance() {
        return variance(1);
    }

    /**
     * Variance aggregation.
     * @param ddof delta degrees of freedom. The divisor used in calculation of variance is
     *             <code>N - ddof</code>, where N is the population size.
     */
    static VarianceAggregation variance(int ddof) {
        return new VarianceAggregation(ddof);
    }


    static final class StandardDeviationAggregation extends DdofAggregation {
        private StandardDeviationAggregation(int ddof) {
            super(Kind.STD, ddof);
        }
    }

    /**
     * Standard deviation aggregation with 1 as the delta degrees of freedom.
     */
    static StandardDeviationAggregation standardDeviation() {
        return standardDeviation(1);
    }

    /**
     * Standard deviation aggregation.
     * @param ddof delta degrees of freedom. The divisor used in calculation of std is
     *             <code>N - ddof</code>, where N is the population size.
     */
    static StandardDeviationAggregation standardDeviation(int ddof) {
        return new StandardDeviationAggregation(ddof);
    }

    static final class MedianAggregation extends NoParamAggregation {
        private MedianAggregation() {
            super(Kind.MEDIAN);
        }
    }

    /**
     * Median reduction.
     */
    static MedianAggregation median() {
        return new MedianAggregation();
    }

    /**
     * Aggregate to compute the specified quantiles. Uses linear interpolation by default.
     */
    static QuantileAggregation quantile(double ... quantiles) {
        return quantile(QuantileMethod.LINEAR, quantiles);
    }

    /**
     * Aggregate to compute various quantiles.
     */
    static QuantileAggregation quantile(QuantileMethod method, double ... quantiles) {
        return new QuantileAggregation(method, quantiles);
    }

    static final class ArgMaxAggregation extends NoParamAggregation {
        private ArgMaxAggregation() {
            super(Kind.ARGMAX);
        }
    }

    /**
     * Index of the max element.
     */
    static ArgMaxAggregation argMax() {
        return new ArgMaxAggregation();
    }

    static final class ArgMinAggregation extends NoParamAggregation {
        private ArgMinAggregation() {
            super(Kind.ARGMIN);
        }
    }

    /**
     * Index of the min element.
     */
    static ArgMinAggregation argMin() {
        return new ArgMinAggregation();
    }

    static final class NuniqueAggregation extends CountLikeAggregation {
        private NuniqueAggregation(NullPolicy nullPolicy) {
            super(Kind.NUNIQUE, nullPolicy);
        }
    }

    /**
     * Number of unique, non-null, elements.
     */
    static NuniqueAggregation nunique() {
        return nunique(NullPolicy.EXCLUDE);
    }

    /**
     * Number of unique elements.
     * @param nullPolicy INCLUDE if nulls should be counted else EXCLUDE. If nulls are counted they
     *                   compare as equal so multiple null values in a range would all only
     *                   increase the count by 1.
     */
    static NuniqueAggregation nunique(NullPolicy nullPolicy) {
        return new NuniqueAggregation(nullPolicy);
    }

    /**
     * Get the nth element in a group.
     * <p>
     * NULL values are included (i.e. a NULL element can be returned). Use
     * {@link #nth(int, NullPolicy)} to control NULL policy.
     *
     * @param offset the offset to look at. Negative numbers go from the end of the group. Any
     *               value outside of the group range results in a null.
     */
    static NthAggregation nth(int offset) {
        return nth(offset, NullPolicy.INCLUDE);
    }

    /**
     * Get the nth element in a group.
     * @param offset the offset to look at. Negative numbers go from the end of the group. Any
     *               value outside of the group range results in a null.
     * @param nullPolicy INCLUDE if nulls should be included in the aggregation or EXCLUDE if they
     *                   should be skipped.
     */
    static NthAggregation nth(int offset, NullPolicy nullPolicy) {
        return new NthAggregation(offset, nullPolicy);
    }

    static final class RowNumberAggregation extends NoParamAggregation {
        private RowNumberAggregation() {
            super(Kind.ROW_NUMBER);
        }
    }

    /**
     * Get the row number, only makes sense for a window operations.
     */
    static RowNumberAggregation rowNumber() {
        return new RowNumberAggregation();
    }

    static final class RankAggregation extends NoParamAggregation {
        private RankAggregation() {
            super(Kind.RANK);
        }
    }

    /**
     * Get the row's ranking.
     */
    static RankAggregation rank() {
        return new RankAggregation();
    }

    static final class DenseRankAggregation extends NoParamAggregation {
        private DenseRankAggregation() {
            super(Kind.DENSE_RANK);
        }
    }

    /**
     * Get the row's dense ranking.
     */
    static DenseRankAggregation denseRank() {
        return new DenseRankAggregation();
    }

    static final class PercentRankAggregation extends NoParamAggregation {
        private PercentRankAggregation() {
            super(Kind.PERCENT_RANK);
        }
    }

    /**
     * Get the row's percent ranking.
     */
    static PercentRankAggregation percentRank() {
        return new PercentRankAggregation();
    }

    /**
     * Collect the values into a list. Nulls will be skipped.
     */
    static CollectListAggregation collectList() {
        return collectList(NullPolicy.EXCLUDE);
    }

    /**
     * Collect the values into a list.
     *
     * @param nullPolicy Indicates whether to include/exclude nulls during collection.
     */
    static CollectListAggregation collectList(NullPolicy nullPolicy) {
        return new CollectListAggregation(nullPolicy);
    }

    /**
     * Collect the values into a set. All null values will be excluded, and all nan values are regarded as
     * unique instances.
     */
    static CollectSetAggregation collectSet() {
        return collectSet(NullPolicy.EXCLUDE, NullEquality.UNEQUAL, NaNEquality.UNEQUAL);
    }

    /**
     * Collect the values into a set.
     *
     * @param nullPolicy   Indicates whether to include/exclude nulls during collection.
     * @param nullEquality Flag to specify whether null entries within each list should be considered equal.
     * @param nanEquality  Flag to specify whether NaN values in floating point column should be considered equal.
     */
    static CollectSetAggregation collectSet(NullPolicy nullPolicy, NullEquality nullEquality, NaNEquality nanEquality) {
        return new CollectSetAggregation(nullPolicy, nullEquality, nanEquality);
    }

    static final class MergeListsAggregation extends NoParamAggregation {
        private MergeListsAggregation() {
            super(Kind.MERGE_LISTS);
        }
    }

    /**
     * Merge the partial lists produced by multiple CollectListAggregations.
     * NOTICE: The partial lists to be merged should NOT include any null list element (but can include null list entries).
     */
    static MergeListsAggregation mergeLists() {
        return new MergeListsAggregation();
    }

    /**
     * Merge the partial sets produced by multiple CollectSetAggregations. Each null/nan value will be regarded as
     * a unique instance.
     */
    static MergeSetsAggregation mergeSets() {
        return mergeSets(NullEquality.UNEQUAL, NaNEquality.UNEQUAL);
    }

    /**
     * Merge the partial sets produced by multiple CollectSetAggregations.
     *
     * @param nullEquality Flag to specify whether null entries within each list should be considered equal.
     * @param nanEquality  Flag to specify whether NaN values in floating point column should be considered equal.
     */
    static MergeSetsAggregation mergeSets(NullEquality nullEquality, NaNEquality nanEquality) {
        return new MergeSetsAggregation(nullEquality, nanEquality);
    }

    /**
     * Host UDF aggregation, to execute a host-side user-defined function (UDF).
     * @param wrapper The wrapper for the native host UDF instance.
     * @return A new HostUDFAggregation instance
     */
    static HostUDFAggregation hostUDF(HostUDFWrapper wrapper) {
        return new HostUDFAggregation(wrapper);
    }

    static final class LeadAggregation extends LeadLagAggregation {
        private LeadAggregation(int offset, ColumnVector defaultOutput) {
            super(Kind.LEAD, offset, defaultOutput);
        }
    }

    /**
     * In a rolling window return the value offset entries ahead or the corresponding value from
     * defaultOutput if it is outside of the window. Note that this does not take any ownership of
     * defaultOutput and the caller mush ensure that defaultOutput remains valid during the life
     * time of this aggregation operation.
     */
    static LeadAggregation lead(int offset, ColumnVector defaultOutput) {
        return new LeadAggregation(offset, defaultOutput);
    }

    static final class LagAggregation extends LeadLagAggregation {
        private LagAggregation(int offset, ColumnVector defaultOutput) {
            super(Kind.LAG, offset, defaultOutput);
        }
    }

    /**
     * In a rolling window return the value offset entries behind or the corresponding value from
     * defaultOutput if it is outside of the window. Note that this does not take any ownership of
     * defaultOutput and the caller mush ensure that defaultOutput remains valid during the life
     * time of this aggregation operation.
     */
    static LagAggregation lag(int offset, ColumnVector defaultOutput) {
        return new LagAggregation(offset, defaultOutput);
    }

    public static final class MergeM2Aggregation extends NoParamAggregation {
        private MergeM2Aggregation() {
            super(Kind.MERGE_M2);
        }
    }

    /**
     * Merge the partial M2 values produced by multiple instances of M2Aggregation.
     */
    static MergeM2Aggregation mergeM2() {
        return new MergeM2Aggregation();
    }

    static class TDigestAggregation extends Aggregation {
        private final int delta;

        public TDigestAggregation(Kind kind, int delta) {
            super(kind);
            this.delta = delta;
        }

        @Override
        long createNativeInstance() {
            return Aggregation.createTDigestAgg(kind.nativeId, delta);
        }

        @Override
        public int hashCode() {
            return 31 * kind.hashCode() + delta;
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            } else if (other instanceof TDigestAggregation) {
                TDigestAggregation o = (TDigestAggregation) other;
                return o.delta == this.delta;
            }
            return false;
        }
    }

    static TDigestAggregation createTDigest(int delta) {
        return new TDigestAggregation(Kind.TDIGEST, delta);
    }

    static TDigestAggregation mergeTDigest(int delta) {
        return new TDigestAggregation(Kind.MERGE_TDIGEST, delta);
    }

    static final class HistogramAggregation extends NoParamAggregation {
        private HistogramAggregation() {
            super(Kind.HISTOGRAM);
        }
    }

    static final class MergeHistogramAggregation extends NoParamAggregation {
        private MergeHistogramAggregation() {
            super(Kind.MERGE_HISTOGRAM);
        }
    }

    static HistogramAggregation histogram() {
        return new HistogramAggregation();
    }

    static MergeHistogramAggregation mergeHistogram() {
        return new MergeHistogramAggregation();
    }

    static BitAndAggregation bitAnd() {
        return new BitAndAggregation();
    }

    static BitOrAggregation bitOr() {
        return new BitOrAggregation();
    }

    static BitXorAggregation bitXor() {
        return new BitXorAggregation();
    }

    /**
     * Create one of the aggregations that only needs a kind, no other parameters. This does not
     * work for all types and for code safety reasons each kind is added separately.
     */
    private static native long createNoParamAgg(int kind);

    /**
     * Create an nth aggregation.
     */
    private static native long createNthAgg(int offset, boolean includeNulls);

    /**
     * Create an aggregation that uses a ddof
     */
    private static native long createDdofAgg(int kind, int ddof);

    /**
     * Create an aggregation that is like count including nulls or not.
     */
    private static native long createCountLikeAgg(int kind, boolean includeNulls);

    /**
     * Create quantile aggregation.
     */
    private static native long createQuantAgg(int method, double[] quantiles);

    /**
     * Create a lead or lag aggregation.
     */
    private static native long createLeadLagAgg(int kind, int offset);

    /**
     * Create a collect list aggregation including nulls or not.
     */
    private static native long createCollectListAgg(boolean includeNulls);

    /**
     * Create a collect set aggregation.
     */
    private static native long createCollectSetAgg(boolean includeNulls, boolean nullsEqual, boolean nansEqual);

    /**
     * Create a merge sets aggregation.
     */
    private static native long createMergeSetsAgg(boolean nullsEqual, boolean nansEqual);

    /**
     * Create a TDigest aggregation.
     */
    private static native long createTDigestAgg(int kind, int delta);

    /**
     * Create a HOST_UDF aggregation.
     */
    private static native long createHostUDFAgg(long udfNativeHandle);

    /**
     * Create a bitwise AND aggregation.
     */
    private static native long createBitAndAgg();

    /**
     * Create a bitwise OR aggregation.
     */
    private static native long createBitOrAgg();

    /**
     * Create a bitwise XOR aggregation.
     */
    private static native long createBitXorAgg();
}
