/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/error.hpp>

#include <rmm/error.hpp>

#include <jni.h>

namespace cudf::jni {

// Declare the strings as inline to avoid multiple definitions in different translation units.
#define JNI_EXP_TYPE inline constexpr char const*

// Wrapper for cudf JNI exception classes, which also store native stacktrace.
JNI_EXP_TYPE CUDA_EXCEPTION_CLASS          = "ai/rapids/cudf/CudaException";
JNI_EXP_TYPE CUDA_FATAL_EXCEPTION_CLASS    = "ai/rapids/cudf/CudaFatalException";
JNI_EXP_TYPE CUDF_EXCEPTION_CLASS          = "ai/rapids/cudf/CudfException";
JNI_EXP_TYPE CUDF_OVERFLOW_EXCEPTION_CLASS = "ai/rapids/cudf/CudfColumnSizeOverflowException";
JNI_EXP_TYPE NVCOMP_EXCEPTION_CLASS        = "ai/rapids/cudf/nvcomp/NvcompException";
JNI_EXP_TYPE NVCOMP_CUDA_EXCEPTION_CLASS   = "ai/rapids/cudf/nvcomp/NvcompCudaException";

// Java exceptions classes.
JNI_EXP_TYPE INDEX_OOB_EXCEPTION_CLASS   = "java/lang/ArrayIndexOutOfBoundsException";
JNI_EXP_TYPE ILLEGAL_ARG_EXCEPTION_CLASS = "java/lang/IllegalArgumentException";
JNI_EXP_TYPE NPE_EXCEPTION_CLASS         = "java/lang/NullPointerException";
JNI_EXP_TYPE RUNTIME_EXCEPTION_CLASS     = "java/lang/RuntimeException";
JNI_EXP_TYPE UNSUPPORTED_EXCEPTION_CLASS = "java/lang/UnsupportedOperationException";

// Java error classes.
// An error is a serious problem and the applications should not expect to recover from it.
JNI_EXP_TYPE OOM_ERROR_CLASS = "java/lang/OutOfMemoryError";

/**
 * @brief Exception class indicating that a JNI error of some kind was thrown and the main
 * function should return.
 */
class jni_exception : public std::runtime_error {
 public:
  jni_exception(char const* const message) : std::runtime_error(message) {}
  jni_exception(std::string const& message) : std::runtime_error(message) {}
};

/**
 * @brief Throw a Java exception and a C++ one for flow control.
 */
inline void throw_java_exception(JNIEnv* const env, char const* class_name, char const* message)
{
  jclass ex_class = env->FindClass(class_name);
  if (ex_class != nullptr) { env->ThrowNew(ex_class, message); }
  throw jni_exception(message);
}

/**
 * @brief Check if a Java exceptions have been thrown and if so throw a C++ exception so the flow
 * control stop processing.
 */
inline void check_java_exception(JNIEnv* const env)
{
  if (env->ExceptionCheck()) {
    // Not going to try to get the message out of the Exception, too complex and
    // might fail.
    throw jni_exception("JNI Exception...");
  }
}

/**
 * @brief Create CudaException or CudaFatalException from a given cudaError_t code.
 */
inline jthrowable cuda_exception(JNIEnv* const env, cudaError_t status, jthrowable cause = nullptr)
{
  // Calls cudaGetLastError twice. It is nearly certain that a fatal error occurred if the second
  // call doesn't return with cudaSuccess.
  cudaGetLastError();
  auto const last = cudaGetLastError();
  // Call cudaDeviceSynchronize to ensure `last` did not result from an asynchronous error
  // between two calls.
  auto const ex_class_name = status == last && last == cudaDeviceSynchronize()
                               ? CUDA_FATAL_EXCEPTION_CLASS
                               : CUDA_EXCEPTION_CLASS;

  jclass ex_class = env->FindClass(ex_class_name);
  if (ex_class == nullptr) { return nullptr; }
  jmethodID ctor_id =
    env->GetMethodID(ex_class, "<init>", "(Ljava/lang/String;ILjava/lang/Throwable;)V");
  if (ctor_id == nullptr) { return nullptr; }

  jstring msg = env->NewStringUTF(cudaGetErrorString(status));
  if (msg == nullptr) { return nullptr; }

  jint err_code = static_cast<jint>(status);

  jobject ret = env->NewObject(ex_class, ctor_id, msg, err_code, cause);
  return static_cast<jthrowable>(ret);
}

inline void jni_cuda_check(JNIEnv* const env, cudaError_t cuda_status)
{
  if (cudaSuccess != cuda_status) {
    jthrowable jt = cuda_exception(env, cuda_status);
    if (jt != nullptr) { env->Throw(jt); }
    throw jni_exception(std::string("CUDA ERROR: code ") +
                        std::to_string(static_cast<int>(cuda_status)));
  }
}
}  // namespace cudf::jni

#define JNI_EXCEPTION_OCCURRED_CHECK(env, ret_val)    \
  {                                                   \
    if (env->ExceptionOccurred()) { return ret_val; } \
  }

#define JNI_THROW_NEW(env, class_name, message, ret_val) \
  {                                                      \
    jclass ex_class = env->FindClass(class_name);        \
    if (ex_class == nullptr) { return ret_val; }         \
    env->ThrowNew(ex_class, message);                    \
    return ret_val;                                      \
  }

// Throw a new exception only if one is not pending then always return with the specified value
#define JNI_CHECK_THROW_CUDF_EXCEPTION(env, class_name, message, stacktrace, ret_val)           \
  {                                                                                             \
    JNI_EXCEPTION_OCCURRED_CHECK(env, ret_val);                                                 \
    auto const ex_class = env->FindClass(class_name);                                           \
    if (ex_class == nullptr) { return ret_val; }                                                \
    auto const ctor_id =                                                                        \
      env->GetMethodID(ex_class, "<init>", "(Ljava/lang/String;Ljava/lang/String;)V");          \
    if (ctor_id == nullptr) { return ret_val; }                                                 \
    auto const empty_str = std::string{""};                                                     \
    auto const jmessage  = env->NewStringUTF(message == nullptr ? empty_str.c_str() : message); \
    if (jmessage == nullptr) { return ret_val; }                                                \
    auto const jstacktrace =                                                                    \
      env->NewStringUTF(stacktrace == nullptr ? empty_str.c_str() : stacktrace);                \
    if (jstacktrace == nullptr) { return ret_val; }                                             \
    auto const jobj = env->NewObject(ex_class, ctor_id, jmessage, jstacktrace);                 \
    if (jobj == nullptr) { return ret_val; }                                                    \
    env->Throw(reinterpret_cast<jthrowable>(jobj));                                             \
    return ret_val;                                                                             \
  }

// Throw a new exception only if one is not pending then always return with the specified value
#define JNI_CHECK_THROW_CUDA_EXCEPTION(env, class_name, message, stacktrace, error_code, ret_val)   \
  {                                                                                                 \
    JNI_EXCEPTION_OCCURRED_CHECK(env, ret_val);                                                     \
    auto const ex_class = env->FindClass(class_name);                                               \
    if (ex_class == nullptr) { return ret_val; }                                                    \
    auto const ctor_id =                                                                            \
      env->GetMethodID(ex_class, "<init>", "(Ljava/lang/String;Ljava/lang/String;I)V");             \
    if (ctor_id == nullptr) { return ret_val; }                                                     \
    auto const empty_str = std::string{""};                                                         \
    auto const jmessage  = env->NewStringUTF(message == nullptr ? empty_str.c_str() : message);     \
    if (jmessage == nullptr) { return ret_val; }                                                    \
    auto const jstacktrace =                                                                        \
      env->NewStringUTF(stacktrace == nullptr ? empty_str.c_str() : stacktrace);                    \
    if (jstacktrace == nullptr) { return ret_val; }                                                 \
    auto const jerror_code = static_cast<jint>(error_code);                                         \
    auto const jobj        = env->NewObject(ex_class, ctor_id, jmessage, jstacktrace, jerror_code); \
    if (jobj == nullptr) { return ret_val; }                                                        \
    env->Throw(reinterpret_cast<jthrowable>(jobj));                                                 \
    return ret_val;                                                                                 \
  }

#define JNI_NULL_CHECK(env, obj, error_msg, ret_val)                                            \
  {                                                                                             \
    if ((obj) == 0) { JNI_THROW_NEW(env, cudf::jni::NPE_EXCEPTION_CLASS, error_msg, ret_val); } \
  }

#define JNI_ARG_CHECK(env, obj, error_msg, ret_val)                                   \
  {                                                                                   \
    if (!(obj)) {                                                                     \
      JNI_THROW_NEW(env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, error_msg, ret_val); \
    }                                                                                 \
  }

// Catch the commonly known exceptions.
#define CATCH_SPECIAL_EXCEPTION(env, ret_val)                                                    \
  catch (rmm::out_of_memory const& e)                                                            \
  {                                                                                              \
    JNI_EXCEPTION_OCCURRED_CHECK(env, ret_val);                                                  \
    auto const what =                                                                            \
      std::string("Could not allocate native memory: ") + (e.what() == nullptr ? "" : e.what()); \
    JNI_THROW_NEW(env, cudf::jni::OOM_ERROR_CLASS, what.c_str(), ret_val);                       \
  }                                                                                              \
  catch (cudf::fatal_cuda_error const& e)                                                        \
  {                                                                                              \
    JNI_CHECK_THROW_CUDA_EXCEPTION(                                                              \
      env, cudf::jni::CUDA_FATAL_EXCEPTION_CLASS, e.what(), nullptr, e.error_code(), ret_val);   \
  }                                                                                              \
  catch (cudf::cuda_error const& e)                                                              \
  {                                                                                              \
    JNI_CHECK_THROW_CUDA_EXCEPTION(                                                              \
      env, cudf::jni::CUDA_EXCEPTION_CLASS, e.what(), nullptr, e.error_code(), ret_val);         \
  }                                                                                              \
  catch (cudf::data_type_error const& e)                                                         \
  {                                                                                              \
    JNI_CHECK_THROW_CUDF_EXCEPTION(                                                              \
      env, cudf::jni::CUDF_EXCEPTION_CLASS, e.what(), nullptr, ret_val);                         \
  }                                                                                              \
  catch (std::overflow_error const& e)                                                           \
  {                                                                                              \
    JNI_CHECK_THROW_CUDF_EXCEPTION(                                                              \
      env, cudf::jni::CUDF_OVERFLOW_EXCEPTION_CLASS, e.what(), nullptr, ret_val);                \
  }

// Catch any exceptions derived from std::exception.
#define CATCH_STD_EXCEPTION(env, ret_val)                                                   \
  catch (std::exception const& e)                                                           \
  {                                                                                         \
    /* Double check whether the thrown exception is unrecoverable CUDA error or not. */     \
    /* Like cudf::detail::throw_cuda_error, it is nearly certain that a fatal error  */     \
    /* occurred if the second call doesn't return with cudaSuccess. */                      \
    cudaGetLastError();                                                                     \
    auto const last = cudaFree(0);                                                          \
    if (cudaSuccess != last && last == cudaDeviceSynchronize()) {                           \
      /* Throw CudaFatalException since the thrown exception is unrecoverable CUDA error */ \
      JNI_CHECK_THROW_CUDA_EXCEPTION(                                                       \
        env, cudf::jni::CUDA_FATAL_EXCEPTION_CLASS, e.what(), nullptr, last, ret_val);      \
    }                                                                                       \
    JNI_CHECK_THROW_CUDF_EXCEPTION(                                                         \
      env, cudf::jni::CUDF_EXCEPTION_CLASS, e.what(), nullptr, ret_val);                    \
  }

// Define try-catch macros which can be patched to something else if needed.
// Typically, JNI_TRY/JNI_CATCH should be used in every JNI function.
// The JNI_CATCH macro consists of several smaller catch macros to allow inserting more catch
// blocks if needed. The macro JNI_CATCH_BEGIN must always be called before any catch block,
// and CATCH_STD_EXCEPTION must always be called last.
#define JNI_TRY                       try {
#define JNI_CATCH_BEGIN(env, ret_val) }  // no-op by default
#define JNI_CATCH(env, ret_val)         \
  JNI_CATCH_BEGIN(env, ret_val)         \
  CATCH_SPECIAL_EXCEPTION(env, ret_val) \
  CATCH_STD_EXCEPTION(env, ret_val)

// Deprecated: macros from here will be removed once spark-rapids-jni is updated to use the new
// macros JNI_TRY/JNI_CATCH.
#define CATCH_STD_CLASS(env, class_name, ret_val)                                                \
  catch (const rmm::out_of_memory& e)                                                            \
  {                                                                                              \
    JNI_EXCEPTION_OCCURRED_CHECK(env, ret_val);                                                  \
    auto const what =                                                                            \
      std::string("Could not allocate native memory: ") + (e.what() == nullptr ? "" : e.what()); \
    JNI_THROW_NEW(env, cudf::jni::OOM_ERROR_CLASS, what.c_str(), ret_val);                       \
  }                                                                                              \
  catch (const cudf::fatal_cuda_error& e)                                                        \
  {                                                                                              \
    JNI_CHECK_THROW_CUDA_EXCEPTION(                                                              \
      env, cudf::jni::CUDA_FATAL_EXCEPTION_CLASS, e.what(), nullptr, e.error_code(), ret_val);   \
  }                                                                                              \
  catch (const cudf::cuda_error& e)                                                              \
  {                                                                                              \
    JNI_CHECK_THROW_CUDA_EXCEPTION(                                                              \
      env, cudf::jni::CUDA_EXCEPTION_CLASS, e.what(), nullptr, e.error_code(), ret_val);         \
  }                                                                                              \
  catch (const cudf::data_type_error& e)                                                         \
  {                                                                                              \
    JNI_CHECK_THROW_CUDF_EXCEPTION(                                                              \
      env, cudf::jni::CUDF_EXCEPTION_CLASS, e.what(), nullptr, ret_val);                         \
  }                                                                                              \
  catch (std::overflow_error const& e)                                                           \
  {                                                                                              \
    JNI_CHECK_THROW_CUDF_EXCEPTION(                                                              \
      env, cudf::jni::CUDF_OVERFLOW_EXCEPTION_CLASS, e.what(), nullptr, ret_val);                \
  }                                                                                              \
  catch (const std::exception& e)                                                                \
  {                                                                                              \
    /* Double check whether the thrown exception is unrecoverable CUDA error or not. */          \
    /* Like cudf::detail::throw_cuda_error, it is nearly certain that a fatal error  */          \
    /* occurred if the second call doesn't return with cudaSuccess. */                           \
    cudaGetLastError();                                                                          \
    auto const last = cudaFree(0);                                                               \
    if (cudaSuccess != last && last == cudaDeviceSynchronize()) {                                \
      /* Throw CudaFatalException since the thrown exception is unrecoverable CUDA error */      \
      JNI_CHECK_THROW_CUDA_EXCEPTION(                                                            \
        env, cudf::jni::CUDA_FATAL_EXCEPTION_CLASS, e.what(), nullptr, last, ret_val);           \
    }                                                                                            \
    JNI_CHECK_THROW_CUDF_EXCEPTION(env, class_name, e.what(), nullptr, ret_val);                 \
  }

#define CATCH_STD(env, ret_val) CATCH_STD_CLASS(env, cudf::jni::CUDF_EXCEPTION_CLASS, ret_val)
