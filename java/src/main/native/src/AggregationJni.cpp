/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/aggregation/host_udf.hpp>

extern "C" {

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Aggregation_close(JNIEnv* env,
                                                             jclass class_object,
                                                             jlong ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto to_del = reinterpret_cast<cudf::aggregation*>(ptr);
    delete to_del;
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createNoParamAgg(JNIEnv* env,
                                                                         jclass class_object,
                                                                         jint kind)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto ret = [&] {
      // These numbers come from Aggregation.java and must stay in sync
      switch (kind) {
        case 0:  // SUM
          return cudf::make_sum_aggregation();
        case 1:  // PRODUCT
          return cudf::make_product_aggregation();
        case 2:  // MIN
          return cudf::make_min_aggregation();
        case 3:  // MAX
          return cudf::make_max_aggregation();
        // case 4 COUNT
        case 5:  // ANY
          return cudf::make_any_aggregation();
        case 6:  // ALL
          return cudf::make_all_aggregation();
        case 7:  // SUM_OF_SQUARES
          return cudf::make_sum_of_squares_aggregation();
        case 8:  // MEAN
          return cudf::make_mean_aggregation();
        // case 9: VARIANCE
        // case 10: STD
        case 11:  // MEDIAN
          return cudf::make_median_aggregation();
        // case 12: QUANTILE
        case 13:  // ARGMAX
          return cudf::make_argmax_aggregation();
        case 14:  // ARGMIN
          return cudf::make_argmin_aggregation();
        // case 15: NUNIQUE
        // case 16: NTH_ELEMENT
        case 17:  // ROW_NUMBER
          return cudf::make_row_number_aggregation();
        // case 18: COLLECT_LIST
        // case 19: COLLECT_SET
        case 20:  // MERGE_LISTS
          return cudf::make_merge_lists_aggregation();
        // case 21: MERGE_SETS
        // case 22: LEAD
        // case 23: LAG
        // case 24: PTX
        // case 25: CUDA
        // case 26: HOST_UDF
        case 27:  // M2
          return cudf::make_m2_aggregation();
        case 28:  // MERGE_M2
          return cudf::make_merge_m2_aggregation();
        case 29:  // RANK
          return cudf::make_rank_aggregation(
            cudf::rank_method::MIN, {}, cudf::null_policy::INCLUDE);
        case 30:  // DENSE_RANK
          return cudf::make_rank_aggregation(
            cudf::rank_method::DENSE, {}, cudf::null_policy::INCLUDE);
        case 31:  // ANSI SQL PERCENT_RANK
          return cudf::make_rank_aggregation(cudf::rank_method::MIN,
                                             {},
                                             cudf::null_policy::INCLUDE,
                                             {},
                                             cudf::rank_percentage::ONE_NORMALIZED);
        // case 32: TDIGEST
        // case 33: MERGE_TDIGEST
        case 34:  // HISTOGRAM
          return cudf::make_histogram_aggregation();
        case 35:  // MERGE_HISTOGRAM
          return cudf::make_merge_histogram_aggregation();
          // case 36: BITWISE_AGG
        case 37:  // SUM_OVERFLOW
          return cudf::make_sum_overflow_aggregation();

        default: throw std::logic_error("Unsupported No Parameter Aggregation Operation");
      }
    }();

    return reinterpret_cast<jlong>(ret.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createNthAgg(JNIEnv* env,
                                                                     jclass class_object,
                                                                     jint offset,
                                                                     jboolean include_nulls)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    std::unique_ptr<cudf::aggregation> ret = cudf::make_nth_element_aggregation(
      offset, include_nulls ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE);
    return reinterpret_cast<jlong>(ret.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createDdofAgg(JNIEnv* env,
                                                                      jclass class_object,
                                                                      jint kind,
                                                                      jint ddof)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    std::unique_ptr<cudf::aggregation> ret;
    // These numbers come from Aggregation.java and must stay in sync
    switch (kind) {
      case 9:  // VARIANCE
        ret = cudf::make_variance_aggregation(ddof);
        break;
      case 10:  // STD
        ret = cudf::make_std_aggregation(ddof);
        break;
      default: throw std::logic_error("Unsupported DDOF Aggregation Operation");
    }
    return reinterpret_cast<jlong>(ret.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createTDigestAgg(JNIEnv* env,
                                                                         jclass class_object,
                                                                         jint kind,
                                                                         jint delta)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    std::unique_ptr<cudf::aggregation> ret;
    // These numbers come from Aggregation.java and must stay in sync
    switch (kind) {
      case 32:  // TDIGEST
        ret = cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta);
        break;
      case 33:  // MERGE_TDIGEST
        ret = cudf::make_merge_tdigest_aggregation<cudf::groupby_aggregation>(delta);
        break;
      default: throw std::logic_error("Unsupported TDigest Aggregation Operation");
    }
    return reinterpret_cast<jlong>(ret.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createCountLikeAgg(JNIEnv* env,
                                                                           jclass class_object,
                                                                           jint kind,
                                                                           jboolean include_nulls)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    cudf::null_policy policy =
      include_nulls ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE;
    std::unique_ptr<cudf::aggregation> ret;
    // These numbers come from Aggregation.java and must stay in sync
    switch (kind) {
      case 4:  // COUNT
        ret = cudf::make_count_aggregation(policy);
        break;
      case 15:  // NUNIQUE
        ret = cudf::make_nunique_aggregation(policy);
        break;
      default: throw std::logic_error("Unsupported Count Like Aggregation Operation");
    }
    return reinterpret_cast<jlong>(ret.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createQuantAgg(JNIEnv* env,
                                                                       jclass class_object,
                                                                       jint j_method,
                                                                       jdoubleArray j_quantiles)
{
  JNI_NULL_CHECK(env, j_quantiles, "quantiles are null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    cudf::jni::native_jdoubleArray const quantiles(env, j_quantiles);

    std::vector<double> quants(quantiles.data(), quantiles.data() + quantiles.size());
    cudf::interpolation interp = static_cast<cudf::interpolation>(j_method);

    std::unique_ptr<cudf::aggregation> ret = cudf::make_quantile_aggregation(quants, interp);
    return reinterpret_cast<jlong>(ret.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createLeadLagAgg(JNIEnv* env,
                                                                         jclass class_object,
                                                                         jint kind,
                                                                         jint offset)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    std::unique_ptr<cudf::aggregation> ret;
    // These numbers come from Aggregation.java and must stay in sync
    switch (kind) {
      case 22:  // LEAD
        ret = cudf::make_lead_aggregation(offset);
        break;
      case 23:  // LAG
        ret = cudf::make_lag_aggregation(offset);
        break;
      default: throw std::logic_error("Unsupported Lead/Lag Aggregation Operation");
    }
    return reinterpret_cast<jlong>(ret.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createCollectListAgg(JNIEnv* env,
                                                                             jclass class_object,
                                                                             jboolean include_nulls)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::null_policy policy =
      include_nulls ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE;
    std::unique_ptr<cudf::aggregation> ret = cudf::make_collect_list_aggregation(policy);
    return reinterpret_cast<jlong>(ret.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createCollectSetAgg(JNIEnv* env,
                                                                            jclass class_object,
                                                                            jboolean include_nulls,
                                                                            jboolean nulls_equal,
                                                                            jboolean nans_equal)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::null_policy null_policy =
      include_nulls ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE;
    cudf::null_equality null_equality =
      nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
    cudf::nan_equality nan_equality =
      nans_equal ? cudf::nan_equality::ALL_EQUAL : cudf::nan_equality::UNEQUAL;
    std::unique_ptr<cudf::aggregation> ret =
      cudf::make_collect_set_aggregation(null_policy, null_equality, nan_equality);
    return reinterpret_cast<jlong>(ret.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createMergeSetsAgg(JNIEnv* env,
                                                                           jclass class_object,
                                                                           jboolean nulls_equal,
                                                                           jboolean nans_equal)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::null_equality null_equality =
      nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
    cudf::nan_equality nan_equality =
      nans_equal ? cudf::nan_equality::ALL_EQUAL : cudf::nan_equality::UNEQUAL;
    std::unique_ptr<cudf::aggregation> ret =
      cudf::make_merge_sets_aggregation(null_equality, nan_equality);
    return reinterpret_cast<jlong>(ret.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createHostUDFAgg(JNIEnv* env,
                                                                         jclass class_object,
                                                                         jlong udf_native_handle)
{
  JNI_NULL_CHECK(env, udf_native_handle, "udf_native_handle is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const udf_ptr = reinterpret_cast<cudf::host_udf_base const*>(udf_native_handle);
    auto output        = cudf::make_host_udf_aggregation(udf_ptr->clone());
    return reinterpret_cast<jlong>(output.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createBitAndAgg(JNIEnv* env,
                                                                        jclass class_object)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto output = cudf::make_bitwise_aggregation<cudf::aggregation>(cudf::bitwise_op::AND);
    return reinterpret_cast<jlong>(output.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createBitOrAgg(JNIEnv* env,
                                                                       jclass class_object)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto output = cudf::make_bitwise_aggregation<cudf::aggregation>(cudf::bitwise_op::OR);
    return reinterpret_cast<jlong>(output.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createBitXorAgg(JNIEnv* env,
                                                                        jclass class_object)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto output = cudf::make_bitwise_aggregation<cudf::aggregation>(cudf::bitwise_op::XOR);
    return reinterpret_cast<jlong>(output.release());
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
