/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"
#include "jni_utils.hpp"

#include <cudf/io/protobuf.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

std::vector<uint8_t> jni_byte_array_to_vector(JNIEnv* env, jobject obj)
{
  if (obj == nullptr) { return {}; }
  auto byte_arr = static_cast<jbyteArray>(obj);
  jsize len     = env->GetArrayLength(byte_arr);
  jbyte* bytes  = env->GetByteArrayElements(byte_arr, nullptr);
  if (bytes == nullptr) { return {}; }
  auto vec = std::vector<uint8_t>(len);
  std::copy(
    reinterpret_cast<uint8_t*>(bytes), reinterpret_cast<uint8_t*>(bytes) + len, vec.begin());
  env->ReleaseByteArrayElements(byte_arr, bytes, JNI_ABORT);
  return vec;
}

std::vector<int32_t> jni_int_array_to_vector(JNIEnv* env, jobject obj)
{
  if (obj == nullptr) { return {}; }
  auto int_arr = static_cast<jintArray>(obj);
  jsize len    = env->GetArrayLength(int_arr);
  jint* ints   = env->GetIntArrayElements(int_arr, nullptr);
  if (ints == nullptr) { return {}; }
  auto vec = std::vector<int32_t>(len);
  std::copy(ints, ints + len, vec.begin());
  env->ReleaseIntArrayElements(int_arr, ints, JNI_ABORT);
  return vec;
}

template <typename CppT, typename ConvertFn>
std::vector<CppT> jni_object_array_to_vectors(JNIEnv* env,
                                              jobjectArray arr,
                                              int num_elements,
                                              ConvertFn convert)
{
  std::vector<CppT> result;
  result.reserve(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    jobject elem = env->GetObjectArrayElement(arr, i);
    if (env->ExceptionCheck()) { return {}; }
    auto vec = convert(env, elem);
    if (elem != nullptr) { env->DeleteLocalRef(elem); }
    if (env->ExceptionCheck()) { return {}; }
    result.push_back(std::move(vec));
  }
  return result;
}

}  // anonymous namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_ColumnView_decodeProtobuf(JNIEnv* env,
                                              jclass,
                                              jlong j_view_handle,
                                              jintArray field_numbers,
                                              jintArray parent_indices,
                                              jintArray depth_levels,
                                              jintArray wire_types,
                                              jintArray output_type_ids,
                                              jintArray encodings,
                                              jbooleanArray is_repeated,
                                              jbooleanArray is_required,
                                              jbooleanArray has_default_value,
                                              jlongArray default_ints,
                                              jdoubleArray default_floats,
                                              jbooleanArray default_bools,
                                              jobjectArray default_strings,
                                              jobjectArray enum_valid_values,
                                              jobjectArray enum_names,
                                              jboolean fail_on_errors)
{
  JNI_NULL_CHECK(env, j_view_handle, "column view cannot be null", 0);
  JNI_NULL_CHECK(env, field_numbers, "field_numbers cannot be null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const* input = reinterpret_cast<cudf::column_view const*>(j_view_handle);

    cudf::jni::native_jintArray n_field_numbers(env, field_numbers);
    cudf::jni::native_jintArray n_parent_indices(env, parent_indices);
    cudf::jni::native_jintArray n_depth_levels(env, depth_levels);
    cudf::jni::native_jintArray n_wire_types(env, wire_types);
    cudf::jni::native_jintArray n_output_type_ids(env, output_type_ids);
    cudf::jni::native_jintArray n_encodings(env, encodings);
    cudf::jni::native_jbooleanArray n_is_repeated(env, is_repeated);
    cudf::jni::native_jbooleanArray n_is_required(env, is_required);
    cudf::jni::native_jbooleanArray n_has_default(env, has_default_value);
    cudf::jni::native_jlongArray n_default_ints(env, default_ints);
    cudf::jni::native_jdoubleArray n_default_floats(env, default_floats);
    cudf::jni::native_jbooleanArray n_default_bools(env, default_bools);

    int const num_fields = n_field_numbers.size();

    // Build schema descriptors
    std::vector<cudf::io::protobuf::nested_field_descriptor> schema;
    schema.reserve(num_fields);
    for (int i = 0; i < num_fields; ++i) {
      schema.push_back({n_field_numbers[i],
                        n_parent_indices[i],
                        n_depth_levels[i],
                        static_cast<cudf::io::protobuf::proto_wire_type>(n_wire_types[i]),
                        static_cast<cudf::type_id>(n_output_type_ids[i]),
                        static_cast<cudf::io::protobuf::proto_encoding>(n_encodings[i]),
                        n_is_repeated[i] != 0,
                        n_is_required[i] != 0,
                        n_has_default[i] != 0});
    }

    // Convert default values
    std::vector<int64_t> default_int_values(n_default_ints.begin(), n_default_ints.end());
    std::vector<double> default_float_values(n_default_floats.begin(), n_default_floats.end());

    std::vector<uint8_t> default_bool_values;
    default_bool_values.reserve(num_fields);
    for (int i = 0; i < num_fields; ++i) {
      default_bool_values.push_back(n_default_bools[i] != 0 ? 1 : 0);
    }

    // Convert default strings (byte[][])
    auto default_string_values = jni_object_array_to_vectors<std::vector<uint8_t>>(
      env, default_strings, num_fields, jni_byte_array_to_vector);
    if (env->ExceptionCheck()) { return 0; }

    // Convert enum valid values (int[][])
    auto enum_values = jni_object_array_to_vectors<std::vector<int32_t>>(
      env, enum_valid_values, num_fields, jni_int_array_to_vector);
    if (env->ExceptionCheck()) { return 0; }

    // Convert enum names (byte[][][])
    auto enum_name_values = jni_object_array_to_vectors<std::vector<std::vector<uint8_t>>>(
      env, enum_names, num_fields, [](JNIEnv* e, jobject obj) -> std::vector<std::vector<uint8_t>> {
        if (obj == nullptr) { return {}; }
        auto inner_arr = static_cast<jobjectArray>(obj);
        jsize num      = e->GetArrayLength(inner_arr);
        return jni_object_array_to_vectors<std::vector<uint8_t>>(
          e, inner_arr, num, jni_byte_array_to_vector);
      });
    if (env->ExceptionCheck()) { return 0; }

    cudf::io::protobuf::decode_protobuf_options options{std::move(schema),
                                                        std::move(default_int_values),
                                                        std::move(default_float_values),
                                                        std::move(default_bool_values),
                                                        std::move(default_string_values),
                                                        std::move(enum_values),
                                                        std::move(enum_name_values),
                                                        static_cast<bool>(fail_on_errors)};

    auto result = cudf::io::protobuf::decode_protobuf(
      *input, options, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

    return cudf::jni::release_as_jlong(result);
  }
  CATCH_STD(env, 0);
}

}  // extern "C"
