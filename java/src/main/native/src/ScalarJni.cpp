/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"
#include "dtype_utils.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/types.hpp>

using cudf::jni::release_as_jlong;

extern "C" {

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Scalar_closeScalar(JNIEnv* env,
                                                              jclass,
                                                              jlong scalar_handle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::scalar* s = reinterpret_cast<cudf::scalar*>(scalar_handle);
    delete s;
  }
  JNI_CATCH(env, );
}

JNIEXPORT jboolean JNICALL Java_ai_rapids_cudf_Scalar_isScalarValid(JNIEnv* env,
                                                                    jclass,
                                                                    jlong scalar_handle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::scalar* s = reinterpret_cast<cudf::scalar*>(scalar_handle);
    return static_cast<jboolean>(s->is_valid());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jbyte JNICALL Java_ai_rapids_cudf_Scalar_getByte(JNIEnv* env, jclass, jlong scalar_handle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    using ScalarType = cudf::scalar_type_t<int8_t>;
    auto s           = reinterpret_cast<ScalarType*>(scalar_handle);
    return static_cast<jbyte>(s->value());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jshort JNICALL Java_ai_rapids_cudf_Scalar_getShort(JNIEnv* env,
                                                             jclass,
                                                             jlong scalar_handle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    using ScalarType = cudf::scalar_type_t<int16_t>;
    auto s           = reinterpret_cast<ScalarType*>(scalar_handle);
    return static_cast<jshort>(s->value());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_Scalar_getInt(JNIEnv* env, jclass, jlong scalar_handle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    using ScalarType = cudf::scalar_type_t<int32_t>;
    auto s           = reinterpret_cast<ScalarType*>(scalar_handle);
    return static_cast<jint>(s->value());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_getLong(JNIEnv* env, jclass, jlong scalar_handle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    using ScalarType = cudf::scalar_type_t<int64_t>;
    auto s           = reinterpret_cast<ScalarType*>(scalar_handle);
    return static_cast<jlong>(s->value());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jfloat JNICALL Java_ai_rapids_cudf_Scalar_getFloat(JNIEnv* env,
                                                             jclass,
                                                             jlong scalar_handle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    using ScalarType = cudf::scalar_type_t<float>;
    auto s           = reinterpret_cast<ScalarType*>(scalar_handle);
    return static_cast<jfloat>(s->value());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jdouble JNICALL Java_ai_rapids_cudf_Scalar_getDouble(JNIEnv* env,
                                                               jclass,
                                                               jlong scalar_handle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    using ScalarType = cudf::scalar_type_t<double>;
    auto s           = reinterpret_cast<ScalarType*>(scalar_handle);
    return static_cast<jdouble>(s->value());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jbyteArray JNICALL Java_ai_rapids_cudf_Scalar_getBigIntegerBytes(JNIEnv* env,
                                                                           jclass,
                                                                           jlong scalar_handle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    using ScalarType = cudf::scalar_type_t<__int128_t>;
    auto s           = reinterpret_cast<ScalarType*>(scalar_handle);
    auto val         = s->value();
    jbyte const* ptr = reinterpret_cast<jbyte const*>(&val);
    cudf::jni::native_jbyteArray jbytes{env, ptr, sizeof(__int128_t)};
    return jbytes.get_jArray();
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jbyteArray JNICALL Java_ai_rapids_cudf_Scalar_getUTF8(JNIEnv* env,
                                                                jclass,
                                                                jlong scalar_handle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto s = reinterpret_cast<cudf::string_scalar*>(scalar_handle);
    std::string val{s->to_string()};
    if (val.size() > 0x7FFFFFFF) {
      cudf::jni::throw_java_exception(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "string scalar too large");
    }
    cudf::jni::native_jbyteArray jbytes{
      env, reinterpret_cast<jbyte const*>(val.data()), static_cast<int>(val.size())};
    return jbytes.get_jArray();
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_getListAsColumnView(JNIEnv* env,
                                                                       jclass,
                                                                       jlong scalar_handle)
{
  JNI_NULL_CHECK(env, scalar_handle, "scalar handle is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto s = reinterpret_cast<cudf::list_scalar*>(scalar_handle);
    // Creates a column view in heap with the stack one, to let JVM take care of its
    // life cycle.
    return reinterpret_cast<jlong>(new cudf::column_view(s->view()));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_Scalar_getChildrenFromStructScalar(JNIEnv* env, jclass, jlong scalar_handle)
{
  JNI_NULL_CHECK(env, scalar_handle, "scalar handle is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const s                  = reinterpret_cast<cudf::struct_scalar*>(scalar_handle);
    cudf::table_view const& table = s->view();
    cudf::jni::native_jpointerArray<cudf::column_view> column_handles(env, table.num_columns());
    for (int i = 0; i < table.num_columns(); i++) {
      column_handles[i] = new cudf::column_view(table.column(i));
    }
    return column_handles.get_jArray();
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeBool8Scalar(JNIEnv* env,
                                                                   jclass,
                                                                   jboolean value,
                                                                   jboolean is_valid)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    std::unique_ptr<cudf::scalar> s =
      cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::BOOL8));
    s->set_valid_async(is_valid);
    if (is_valid) {
      using ScalarType = cudf::scalar_type_t<int8_t>;
      int8_t val       = value ? 1 : 0;
      static_cast<ScalarType*>(s.get())->set_value(val);
    }
    return reinterpret_cast<jlong>(s.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeInt8Scalar(JNIEnv* env,
                                                                  jclass,
                                                                  jbyte value,
                                                                  jboolean is_valid)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    std::unique_ptr<cudf::scalar> s =
      cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT8));
    s->set_valid_async(is_valid);
    if (is_valid) {
      using ScalarType = cudf::scalar_type_t<int8_t>;
      static_cast<ScalarType*>(s.get())->set_value(static_cast<int8_t>(value));
    }
    return reinterpret_cast<jlong>(s.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeUint8Scalar(JNIEnv* env,
                                                                   jclass,
                                                                   jbyte value,
                                                                   jboolean is_valid)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    std::unique_ptr<cudf::scalar> s =
      cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::UINT8));
    s->set_valid_async(is_valid);
    if (is_valid) {
      using ScalarType = cudf::scalar_type_t<uint8_t>;
      static_cast<ScalarType*>(s.get())->set_value(static_cast<uint8_t>(value));
    }
    return reinterpret_cast<jlong>(s.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeInt16Scalar(JNIEnv* env,
                                                                   jclass,
                                                                   jshort value,
                                                                   jboolean is_valid)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    std::unique_ptr<cudf::scalar> s =
      cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT16));
    s->set_valid_async(is_valid);
    if (is_valid) {
      using ScalarType = cudf::scalar_type_t<int16_t>;
      static_cast<ScalarType*>(s.get())->set_value(static_cast<int16_t>(value));
    }
    return reinterpret_cast<jlong>(s.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeUint16Scalar(JNIEnv* env,
                                                                    jclass,
                                                                    jshort value,
                                                                    jboolean is_valid)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    std::unique_ptr<cudf::scalar> s =
      cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::UINT16));
    s->set_valid_async(is_valid);
    if (is_valid) {
      using ScalarType = cudf::scalar_type_t<uint16_t>;
      static_cast<ScalarType*>(s.get())->set_value(static_cast<uint16_t>(value));
    }
    return reinterpret_cast<jlong>(s.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeDurationDaysScalar(JNIEnv* env,
                                                                          jclass,
                                                                          jint value,
                                                                          jboolean is_valid)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    std::unique_ptr<cudf::scalar> s =
      cudf::make_duration_scalar(cudf::data_type(cudf::type_id::DURATION_DAYS));
    s->set_valid_async(is_valid);
    if (is_valid) {
      using ScalarType = cudf::scalar_type_t<int32_t>;
      static_cast<ScalarType*>(s.get())->set_value(static_cast<int32_t>(value));
    }
    return reinterpret_cast<jlong>(s.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeInt32Scalar(JNIEnv* env,
                                                                   jclass,
                                                                   jint value,
                                                                   jboolean is_valid)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    std::unique_ptr<cudf::scalar> s =
      cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
    s->set_valid_async(is_valid);
    if (is_valid) {
      using ScalarType = cudf::scalar_type_t<int32_t>;
      static_cast<ScalarType*>(s.get())->set_value(static_cast<int32_t>(value));
    }
    return reinterpret_cast<jlong>(s.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeUint32Scalar(JNIEnv* env,
                                                                    jclass,
                                                                    jint value,
                                                                    jboolean is_valid)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    std::unique_ptr<cudf::scalar> s =
      cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::UINT32));
    s->set_valid_async(is_valid);
    if (is_valid) {
      using ScalarType = cudf::scalar_type_t<uint32_t>;
      static_cast<ScalarType*>(s.get())->set_value(static_cast<uint32_t>(value));
    }
    return reinterpret_cast<jlong>(s.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeInt64Scalar(JNIEnv* env,
                                                                   jclass,
                                                                   jlong value,
                                                                   jboolean is_valid)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    std::unique_ptr<cudf::scalar> s =
      cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT64));
    s->set_valid_async(is_valid);
    if (is_valid) {
      using ScalarType = cudf::scalar_type_t<int64_t>;
      static_cast<ScalarType*>(s.get())->set_value(static_cast<int64_t>(value));
    }
    return reinterpret_cast<jlong>(s.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeUint64Scalar(JNIEnv* env,
                                                                    jclass,
                                                                    jlong value,
                                                                    jboolean is_valid)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    std::unique_ptr<cudf::scalar> s =
      cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::UINT64));
    s->set_valid_async(is_valid);
    if (is_valid) {
      using ScalarType = cudf::scalar_type_t<uint64_t>;
      static_cast<ScalarType*>(s.get())->set_value(static_cast<uint64_t>(value));
    }
    return reinterpret_cast<jlong>(s.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeFloat32Scalar(JNIEnv* env,
                                                                     jclass,
                                                                     jfloat value,
                                                                     jboolean is_valid)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    std::unique_ptr<cudf::scalar> s =
      cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::FLOAT32));
    s->set_valid_async(is_valid);
    if (is_valid) {
      using ScalarType = cudf::scalar_type_t<float>;
      static_cast<ScalarType*>(s.get())->set_value(static_cast<float>(value));
    }
    return reinterpret_cast<jlong>(s.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeFloat64Scalar(JNIEnv* env,
                                                                     jclass,
                                                                     jdouble value,
                                                                     jboolean is_valid)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    std::unique_ptr<cudf::scalar> s =
      cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::FLOAT64));
    s->set_valid_async(is_valid);
    if (is_valid) {
      using ScalarType = cudf::scalar_type_t<double>;
      static_cast<ScalarType*>(s.get())->set_value(static_cast<double>(value));
    }
    return reinterpret_cast<jlong>(s.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeStringScalar(JNIEnv* env,
                                                                    jclass,
                                                                    jbyteArray value,
                                                                    jboolean is_valid)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    std::string strval;
    if (is_valid) {
      cudf::jni::native_jbyteArray jbytes{env, value};
      strval.assign(reinterpret_cast<char*>(jbytes.data()), jbytes.size());
    }

    auto s = new cudf::string_scalar{strval, static_cast<bool>(is_valid)};
    return reinterpret_cast<jlong>(s);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeTimestampDaysScalar(JNIEnv* env,
                                                                           jclass,
                                                                           jint value,
                                                                           jboolean is_valid)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    std::unique_ptr<cudf::scalar> s =
      cudf::make_timestamp_scalar(cudf::data_type(cudf::type_id::TIMESTAMP_DAYS));
    s->set_valid_async(is_valid);
    if (is_valid) {
      using ScalarType = cudf::scalar_type_t<int32_t>;
      static_cast<ScalarType*>(s.get())->set_value(static_cast<int32_t>(value));
    }
    return reinterpret_cast<jlong>(s.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeDurationTimeScalar(
  JNIEnv* env, jclass, jint jdtype_id, jlong value, jboolean is_valid)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto dtype_id                   = static_cast<cudf::type_id>(jdtype_id);
    std::unique_ptr<cudf::scalar> s = cudf::make_duration_scalar(cudf::data_type(dtype_id));
    s->set_valid_async(is_valid);
    if (is_valid) {
      using ScalarType = cudf::scalar_type_t<int64_t>;
      static_cast<ScalarType*>(s.get())->set_value(static_cast<int64_t>(value));
    }
    return reinterpret_cast<jlong>(s.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeTimestampTimeScalar(
  JNIEnv* env, jclass, jint jdtype_id, jlong value, jboolean is_valid)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto dtype_id                   = static_cast<cudf::type_id>(jdtype_id);
    std::unique_ptr<cudf::scalar> s = cudf::make_timestamp_scalar(cudf::data_type(dtype_id));
    s->set_valid_async(is_valid);
    if (is_valid) {
      using ScalarType = cudf::scalar_type_t<int64_t>;
      static_cast<ScalarType*>(s.get())->set_value(static_cast<int64_t>(value));
    }
    return reinterpret_cast<jlong>(s.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeDecimal32Scalar(
  JNIEnv* env, jclass, jint value, jint scale, jboolean is_valid)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const value_ = static_cast<int32_t>(value);
    auto const scale_ = numeric::scale_type{static_cast<int32_t>(scale)};
    std::unique_ptr<cudf::scalar> s =
      cudf::make_fixed_point_scalar<numeric::decimal32>(value_, scale_);
    s->set_valid_async(is_valid);
    return reinterpret_cast<jlong>(s.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeDecimal64Scalar(
  JNIEnv* env, jclass, jlong value, jint scale, jboolean is_valid)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const value_ = static_cast<int64_t>(value);
    auto const scale_ = numeric::scale_type{static_cast<int32_t>(scale)};
    std::unique_ptr<cudf::scalar> s =
      cudf::make_fixed_point_scalar<numeric::decimal64>(value_, scale_);
    s->set_valid_async(is_valid);
    return reinterpret_cast<jlong>(s.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeDecimal128Scalar(
  JNIEnv* env, jclass, jbyteArray value, jint scale, jboolean is_valid)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const scale_ = numeric::scale_type{static_cast<int32_t>(scale)};
    cudf::jni::native_jbyteArray jbytes{env, value};
    auto const value_ = reinterpret_cast<__int128_t*>(jbytes.data());
    std::unique_ptr<cudf::scalar> s =
      cudf::make_fixed_point_scalar<numeric::decimal128>(*value_, scale_);
    s->set_valid_async(is_valid);
    return reinterpret_cast<jlong>(s.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_binaryOpSV(
  JNIEnv* env, jclass, jlong lhs_ptr, jlong rhs_view, jint int_op, jint out_dtype, jint scale)
{
  JNI_NULL_CHECK(env, lhs_ptr, "lhs is null", 0);
  JNI_NULL_CHECK(env, rhs_view, "rhs is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::scalar* lhs           = reinterpret_cast<cudf::scalar*>(lhs_ptr);
    auto rhs                    = reinterpret_cast<cudf::column_view*>(rhs_view);
    cudf::data_type n_data_type = cudf::jni::make_data_type(out_dtype, scale);
    cudf::binary_operator op    = static_cast<cudf::binary_operator>(int_op);

    if (lhs->type().id() == cudf::type_id::STRUCT) {
      auto out = make_fixed_width_column(n_data_type, rhs->size(), cudf::mask_state::UNALLOCATED);

      if (op == cudf::binary_operator::NULL_EQUALS) {
        out->set_null_mask(cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED), 0);
      } else {
        auto [new_mask, new_null_count] = cudf::binops::scalar_col_valid_mask_and(*rhs, *lhs);
        out->set_null_mask(std::move(new_mask), new_null_count);
      }

      auto lhs_col  = cudf::make_column_from_scalar(*lhs, 1);
      auto out_view = out->mutable_view();
      cudf::binops::compiled::detail::apply_sorting_struct_binary_op(
        out_view, lhs_col->view(), *rhs, true, false, op, cudf::get_default_stream());
      return release_as_jlong(out);
    }

    return release_as_jlong(cudf::binary_operation(*lhs, *rhs, op, n_data_type));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeListScalar(JNIEnv* env,
                                                                  jclass,
                                                                  jlong view_handle,
                                                                  jboolean is_valid)
{
  JNI_NULL_CHECK(env, view_handle, "Column view should NOT be null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto col_view = reinterpret_cast<cudf::column_view*>(view_handle);

    // Instead of calling the `cudf::empty_like` to create an empty column when `is_valid`
    // is false, always passes the input view to the scalar, to avoid copying the column
    // twice.
    // Let the Java layer make sure the view is empty when `is_valid` is false.
    cudf::scalar* s = new cudf::list_scalar(*col_view);
    s->set_valid_async(is_valid);
    return reinterpret_cast<jlong>(s);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeStructScalar(JNIEnv* env,
                                                                    jclass,
                                                                    jlongArray handles,
                                                                    jboolean is_valid)
{
  JNI_NULL_CHECK(env, handles, "native view handles are null", 0)
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    std::unique_ptr<cudf::column_view> ret;
    cudf::jni::native_jpointerArray<cudf::column_view> column_pointers(env, handles);
    std::vector<cudf::column_view> columns;
    columns.reserve(column_pointers.size());
    std::transform(column_pointers.data(),
                   column_pointers.data() + column_pointers.size(),
                   std::back_inserter(columns),
                   [](auto const& col_ptr) { return *col_ptr; });
    auto s = std::make_unique<cudf::struct_scalar>(
      cudf::host_span<cudf::column_view const>{columns}, is_valid);
    return reinterpret_cast<jlong>(s.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_repeatString(JNIEnv* env,
                                                                jclass,
                                                                jlong handle,
                                                                jint repeat_times)
{
  JNI_NULL_CHECK(env, handle, "scalar handle is null", 0)
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const str = *reinterpret_cast<cudf::string_scalar*>(handle);
    return reinterpret_cast<jlong>(cudf::strings::repeat_string(str, repeat_times).release());
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
