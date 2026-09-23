/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_jni_internal.hpp"

#include <cudf/utilities/error.hpp>

#include <string>
#include <utility>

namespace cudf {
namespace jni {
namespace hybrid_scan {

cudf::io::parquet_reader_options build_options(JNIEnv* env,
                                               jobjectArray j_column_names,
                                               jint time_unit_type_id)
{
  cudf::io::parquet_reader_options_builder builder;

  cudf::jni::native_jstringArray names(env, j_column_names);
  if (!names.is_null() && names.size() > 0) {
    builder = builder.column_names(names.as_cpp_vector());
  }

  return builder.convert_strings_to_categories(false)
    .timestamp_type(cudf::data_type(static_cast<cudf::type_id>(time_unit_type_id)))
    .ignore_missing_columns(true)
    .build();
}

row_group_span_holder make_row_group_span(JNIEnv* env, jintArray j_row_groups)
{
  row_group_span_holder h;
  cudf::jni::native_jintArray arr(env, j_row_groups);
  h.storage.reserve(arr.size());
  for (int i = 0; i < arr.size(); ++i) {
    h.storage.push_back(static_cast<cudf::size_type>(arr[i]));
  }
  arr.cancel();
  return h;
}

jlongArray ranges_to_jlong_array(JNIEnv* env, std::vector<byte_range_info> const& ranges)
{
  auto result = env->NewLongArray(ranges.size() * 2);
  if (result == nullptr) { return nullptr; }
  if (ranges.empty()) { return result; }
  std::vector<jlong> data;
  data.reserve(ranges.size() * 2);
  for (auto const& r : ranges) {
    data.push_back(static_cast<jlong>(r.offset()));
    data.push_back(static_cast<jlong>(r.size()));
  }
  env->SetLongArrayRegion(result, 0, data.size(), data.data());
  return result;
}

jintArray sizes_to_jint_array(JNIEnv* env, std::vector<cudf::size_type> const& vals)
{
  auto result = env->NewIntArray(vals.size());
  if (result == nullptr) { return nullptr; }
  if (vals.empty()) { return result; }
  // jint is int32_t; size_type is also int32_t. Static-cast just to be explicit.
  std::vector<jint> j(vals.begin(), vals.end());
  env->SetIntArrayRegion(result, 0, j.size(), j.data());
  return result;
}

std::vector<cudf::device_span<uint8_t const>> make_device_spans(JNIEnv* env,
                                                                jlongArray j_addrs,
                                                                jlongArray j_lens)
{
  cudf::jni::native_jlongArray addrs(env, j_addrs);
  cudf::jni::native_jlongArray lens(env, j_lens);
  CUDF_EXPECTS(addrs.size() == lens.size(), "addrs and lens arrays must have the same length");
  std::vector<cudf::device_span<uint8_t const>> out;
  out.reserve(addrs.size());
  for (int i = 0; i < addrs.size(); ++i) {
    out.emplace_back(reinterpret_cast<uint8_t const*>(addrs[i]),
                     checked_size_t(env, lens[i], "byte range length"));
  }
  addrs.cancel();
  lens.cancel();
  return out;
}

std::size_t checked_size_t(JNIEnv* env, jlong value, char const* name)
{
  if (value < 0) {
    auto const msg = std::string(name) + " must be non-negative, got " + std::to_string(value);
    cudf::jni::throw_java_exception(env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, msg.c_str());
  }
  return static_cast<std::size_t>(value);
}

}  // namespace hybrid_scan
}  // namespace jni
}  // namespace cudf
