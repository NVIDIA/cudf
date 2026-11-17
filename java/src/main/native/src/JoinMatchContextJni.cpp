/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"

#include <cudf/column/column.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>
#include <stdexcept>
#include <vector>

namespace {

// JoinKind enum values must match ai.rapids.cudf.JoinKind.nativeId.
constexpr jint JOIN_KIND_INNER = 0;
constexpr jint JOIN_KIND_LEFT  = 1;
constexpr jint JOIN_KIND_FULL  = 2;

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
dispatch_partitioned(cudf::hash_join const& hash,
                     cudf::join_partition_context const& ctx,
                     jint kind)
{
  switch (kind) {
    case JOIN_KIND_INNER: return hash.partitioned_inner_join(ctx);
    case JOIN_KIND_LEFT: return hash.partitioned_left_join(ctx);
    case JOIN_KIND_FULL: return hash.partitioned_full_join(ctx);
    default: throw std::invalid_argument("Unsupported join kind for partitioned retrieval");
  }
}

cudf::join_match_context build_match_context(cudf::hash_join const& hash,
                                             cudf::table_view const& left,
                                             jint kind)
{
  switch (kind) {
    case JOIN_KIND_INNER: return hash.inner_join_match_context(left);
    case JOIN_KIND_LEFT: return hash.left_join_match_context(left);
    case JOIN_KIND_FULL: return hash.full_join_match_context(left);
    default: throw std::invalid_argument("Unsupported join kind for match context");
  }
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_JoinMatchContext_create(
  JNIEnv* env, jclass, jlong j_left_view, jlong j_hash_join, jint j_kind)
{
  JNI_NULL_CHECK(env, j_left_view, "left table is null", 0);
  JNI_NULL_CHECK(env, j_hash_join, "hash join is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const left = reinterpret_cast<cudf::table_view const*>(j_left_view);
    auto const hash = reinterpret_cast<cudf::hash_join const*>(j_hash_join);

    auto match_ctx =
      std::make_unique<cudf::join_match_context>(build_match_context(*hash, *left, j_kind));
    auto part_ctx                = std::make_unique<cudf::join_partition_context>();
    part_ctx->left_table_context = std::move(match_ctx);
    part_ctx->left_start_idx     = 0;
    part_ctx->left_end_idx       = 0;
    return cudf::jni::release_as_jlong(part_ctx);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_JoinMatchContext_destroy(JNIEnv* env,
                                                                    jclass,
                                                                    jlong j_handle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::safe_delete<cudf::join_partition_context>(j_handle);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_JoinMatchContext_outputSize(JNIEnv* env,
                                                                        jclass,
                                                                        jlong j_handle)
{
  JNI_NULL_CHECK(env, j_handle, "join match context is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const ctx = reinterpret_cast<cudf::join_partition_context const*>(j_handle);
    return static_cast<jlong>(ctx->left_table_context->output_size());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_JoinMatchContext_matchCountsColumn(JNIEnv* env,
                                                                               jclass,
                                                                               jlong j_handle)
{
  JNI_NULL_CHECK(env, j_handle, "join match context is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const ctx    = reinterpret_cast<cudf::join_partition_context const*>(j_handle);
    auto const counts = ctx->left_table_context->match_counts();
    auto const stream = cudf::get_default_stream();
    rmm::device_buffer data(counts.data(), counts.size_bytes(), stream);
    auto column =
      std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                     static_cast<cudf::size_type>(counts.size()),
                                     std::move(data),
                                     cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                     0);
    return cudf::jni::release_as_jlong(column);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_JoinMatchContext_partitionGatherMaps(
  JNIEnv* env, jclass, jlong j_handle, jlong j_hash_join, jint j_kind, jint j_start, jint j_end)
{
  JNI_NULL_CHECK(env, j_handle, "join match context is null", NULL);
  JNI_NULL_CHECK(env, j_hash_join, "hash join is null", NULL);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto ctx            = reinterpret_cast<cudf::join_partition_context*>(j_handle);
    auto const hash     = reinterpret_cast<cudf::hash_join const*>(j_hash_join);
    ctx->left_start_idx = static_cast<cudf::size_type>(j_start);
    ctx->left_end_idx   = static_cast<cudf::size_type>(j_end);
    return cudf::jni::gather_maps_to_java(env, dispatch_partitioned(*hash, *ctx, j_kind));
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_JoinMatchContext_finalizePartitionedFullJoin(JNIEnv* env,
                                                                 jclass,
                                                                 jlongArray j_left_addrs,
                                                                 jlongArray j_left_lens,
                                                                 jlongArray j_right_addrs,
                                                                 jlongArray j_right_lens,
                                                                 jint j_left_rows,
                                                                 jint j_right_rows)
{
  JNI_NULL_CHECK(env, j_left_addrs, "left partition addresses are null", NULL);
  JNI_NULL_CHECK(env, j_left_lens, "left partition lengths are null", NULL);
  JNI_NULL_CHECK(env, j_right_addrs, "right partition addresses are null", NULL);
  JNI_NULL_CHECK(env, j_right_lens, "right partition lengths are null", NULL);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jlongArray left_addrs(env, j_left_addrs);
    cudf::jni::native_jlongArray left_lens(env, j_left_lens);
    cudf::jni::native_jlongArray right_addrs(env, j_right_addrs);
    cudf::jni::native_jlongArray right_lens(env, j_right_lens);

    auto const n = left_addrs.size();
    if (left_lens.size() != n || right_addrs.size() != n || right_lens.size() != n) {
      throw std::invalid_argument("partition arrays must have the same length");
    }

    std::vector<cudf::device_span<cudf::size_type const>> left_spans;
    std::vector<cudf::device_span<cudf::size_type const>> right_spans;
    left_spans.reserve(n);
    right_spans.reserve(n);
    for (int i = 0; i < n; ++i) {
      left_spans.emplace_back(reinterpret_cast<cudf::size_type const*>(left_addrs[i]),
                              static_cast<std::size_t>(left_lens[i]));
      right_spans.emplace_back(reinterpret_cast<cudf::size_type const*>(right_addrs[i]),
                               static_cast<std::size_t>(right_lens[i]));
    }

    return cudf::jni::gather_maps_to_java(
      env,
      cudf::hash_join::finalize_partitioned_full_join(left_spans,
                                                      right_spans,
                                                      static_cast<cudf::size_type>(j_left_rows),
                                                      static_cast<cudf::size_type>(j_right_rows)));
  }
  JNI_CATCH(env, NULL);
}

}  // extern "C"
