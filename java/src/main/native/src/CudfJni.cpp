/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"

#include <cudf/copying.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <sstream>

namespace {

// handles detaching a thread from the JVM when the thread terminates
class jvm_detach_on_destruct {
 public:
  explicit jvm_detach_on_destruct(JavaVM* jvm) : jvm{jvm} {}

  ~jvm_detach_on_destruct() { jvm->DetachCurrentThread(); }

 private:
  JavaVM* jvm;
};

}  // anonymous namespace

namespace cudf {
namespace jni {

#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
constexpr bool is_ptds_enabled{true};
#else
constexpr bool is_ptds_enabled{false};
#endif

static jclass Host_memory_buffer_jclass;
static jfieldID Host_buffer_address;
static jfieldID Host_buffer_length;

#define HOST_MEMORY_BUFFER_CLASS          "ai/rapids/cudf/HostMemoryBuffer"
#define HOST_MEMORY_BUFFER_SIG(param_sig) "(" param_sig ")L" HOST_MEMORY_BUFFER_CLASS ";"

static bool cache_host_memory_buffer_jni(JNIEnv* env)
{
  jclass cls = env->FindClass(HOST_MEMORY_BUFFER_CLASS);
  if (cls == nullptr) { return false; }

  Host_buffer_address = env->GetFieldID(cls, "address", "J");
  if (Host_buffer_address == nullptr) { return false; }

  Host_buffer_length = env->GetFieldID(cls, "length", "J");
  if (Host_buffer_length == nullptr) { return false; }

  // Convert local reference to global so it cannot be garbage collected.
  Host_memory_buffer_jclass = static_cast<jclass>(env->NewGlobalRef(cls));
  if (Host_memory_buffer_jclass == nullptr) { return false; }
  return true;
}

static void release_host_memory_buffer_jni(JNIEnv* env)
{
  Host_memory_buffer_jclass = del_global_ref(env, Host_memory_buffer_jclass);
}

jobject allocate_host_buffer(JNIEnv* env,
                             jlong amount,
                             jboolean prefer_pinned,
                             jobject host_memory_allocator)
{
  auto const host_memory_allocator_class = env->GetObjectClass(host_memory_allocator);
  auto const allocateMethodId =
    env->GetMethodID(host_memory_allocator_class, "allocate", HOST_MEMORY_BUFFER_SIG("JZ"));
  jobject ret =
    env->CallObjectMethod(host_memory_allocator, allocateMethodId, amount, prefer_pinned);

  if (env->ExceptionCheck()) { throw std::runtime_error("allocateHostBuffer threw an exception"); }
  return ret;
}

jlong get_host_buffer_address(JNIEnv* env, jobject buffer)
{
  return env->GetLongField(buffer, Host_buffer_address);
}

jlong get_host_buffer_length(JNIEnv* env, jobject buffer)
{
  return env->GetLongField(buffer, Host_buffer_length);
}

// Get the JNI environment, attaching the current thread to the JVM if necessary. If the thread
// needs to be attached, the thread will automatically detach when the thread terminates.
JNIEnv* get_jni_env(JavaVM* jvm)
{
  JNIEnv* env = nullptr;
  jint rc     = jvm->GetEnv(reinterpret_cast<void**>(&env), MINIMUM_JNI_VERSION);
  if (rc == JNI_OK) { return env; }
  if (rc == JNI_EDETACHED) {
    JavaVMAttachArgs attach_args;
    attach_args.version = MINIMUM_JNI_VERSION;
    attach_args.name    = const_cast<char*>("cudf thread");
    attach_args.group   = NULL;

    if (jvm->AttachCurrentThreadAsDaemon(reinterpret_cast<void**>(&env), &attach_args) == JNI_OK) {
      // use thread_local object to detach the thread from the JVM when thread terminates.
      thread_local jvm_detach_on_destruct detacher(jvm);
    } else {
      throw std::runtime_error("unable to attach to JVM");
    }

    return env;
  }

  throw std::runtime_error("error detecting thread attach state with JVM");
}

}  // namespace jni
}  // namespace cudf

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*)
{
  JNIEnv* env;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), cudf::jni::MINIMUM_JNI_VERSION) != JNI_OK) {
    return JNI_ERR;
  }

  // make sure libcudf and the JNI library are built with the same PTDS mode
  if (cudf::is_ptds_enabled() != cudf::jni::is_ptds_enabled) {
    std::ostringstream ss;
    ss << "Libcudf is_ptds_enabled=" << cudf::is_ptds_enabled()
       << ", which does not match cudf jni is_ptds_enabled=" << cudf::jni::is_ptds_enabled
       << ". They need to be built with the same per-thread default stream flag.";
    env->ThrowNew(env->FindClass(cudf::jni::RUNTIME_EXCEPTION_CLASS), ss.str().c_str());
    return JNI_ERR;
  }

  // cache any class objects and method IDs here
  if (!cudf::jni::cache_contiguous_table_jni(env)) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass(cudf::jni::RUNTIME_EXCEPTION_CLASS),
                    "Unable to locate contiguous table methods needed by JNI");
    }
    return JNI_ERR;
  }

  if (!cudf::jni::cache_contig_split_group_by_result_jni(env)) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass(cudf::jni::RUNTIME_EXCEPTION_CLASS),
                    "Unable to locate group by table result methods needed by JNI");
    }
    return JNI_ERR;
  }

  if (!cudf::jni::cache_host_memory_buffer_jni(env)) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass(cudf::jni::RUNTIME_EXCEPTION_CLASS),
                    "Unable to locate host memory buffer methods needed by JNI");
    }
    return JNI_ERR;
  }

  if (!cudf::jni::cache_data_source_jni(env)) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass(cudf::jni::RUNTIME_EXCEPTION_CLASS),
                    "Unable to locate data source helper methods needed by JNI");
    }
    return JNI_ERR;
  }

  return cudf::jni::MINIMUM_JNI_VERSION;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void*)
{
  JNIEnv* env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), cudf::jni::MINIMUM_JNI_VERSION) != JNI_OK) {
    return;
  }

  // release cached class objects here.
  cudf::jni::release_contiguous_table_jni(env);
  cudf::jni::release_contig_split_group_by_result_jni(env);
  cudf::jni::release_host_memory_buffer_jni(env);
}

JNIEXPORT jboolean JNICALL Java_ai_rapids_cudf_Cuda_isPtdsEnabled(JNIEnv* env, jclass)
{
  return cudf::jni::is_ptds_enabled;
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cudf_setKernelPinnedCopyThreshold(JNIEnv* env,
                                                                             jclass clazz,
                                                                             jlong jthreshold)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto threshold = static_cast<std::size_t>(jthreshold);
    cudf::set_kernel_pinned_copy_threshold(threshold);
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cudf_setPinnedAllocationThreshold(JNIEnv* env,
                                                                             jclass clazz,
                                                                             jlong jthreshold)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto threshold = static_cast<std::size_t>(jthreshold);
    cudf::set_allocate_host_as_pinned_threshold(threshold);
  }
  JNI_CATCH(env, );
}

}  // extern "C"
