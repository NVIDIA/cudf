/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "cudf_jni_apis.hpp"
#include "jni_utils.hpp"

#include <cudf/io/data_sink.hpp>

namespace cudf::jni {

constexpr long MINIMUM_WRITE_BUFFER_SIZE = 10 * 1024 * 1024;  // 10 MB

class jni_writer_data_sink final : public cudf::io::data_sink {
 public:
  explicit jni_writer_data_sink(JNIEnv* env, jobject callback, jobject host_memory_allocator)
  {
    if (env->GetJavaVM(&jvm) < 0) { throw std::runtime_error("GetJavaVM failed"); }

    jclass cls = env->GetObjectClass(callback);
    if (cls == nullptr) { throw cudf::jni::jni_exception("class not found"); }

    handle_buffer_method =
      env->GetMethodID(cls, "handleBuffer", "(Lai/rapids/cudf/HostMemoryBuffer;J)V");
    if (handle_buffer_method == nullptr) { throw cudf::jni::jni_exception("handleBuffer method"); }

    this->callback              = add_global_ref(env, callback);
    this->host_memory_allocator = add_global_ref(env, host_memory_allocator);
  }

  virtual ~jni_writer_data_sink()
  {
    // This should normally be called by a JVM thread. If the JVM environment is missing then this
    // is likely being triggered by the C++ runtime during shutdown. In that case the JVM may
    // already be destroyed and this thread should not try to attach to get an environment.
    JNIEnv* env = nullptr;
    if (jvm->GetEnv(reinterpret_cast<void**>(&env), cudf::jni::MINIMUM_JNI_VERSION) == JNI_OK) {
      callback              = del_global_ref(env, callback);
      current_buffer        = del_global_ref(env, current_buffer);
      host_memory_allocator = del_global_ref(env, host_memory_allocator);
    }
    callback              = nullptr;
    current_buffer        = nullptr;
    host_memory_allocator = nullptr;
  }

  void host_write(void const* data, size_t size) override
  {
    JNIEnv* env           = cudf::jni::get_jni_env(jvm);
    long left_to_copy     = static_cast<long>(size);
    char const* copy_from = static_cast<char const*>(data);
    while (left_to_copy > 0) {
      long buffer_amount_available = current_buffer_len - current_buffer_written;
      if (buffer_amount_available <= 0) {
        // should never be < 0, but just to be safe
        rotate_buffer(env);
        buffer_amount_available = current_buffer_len - current_buffer_written;
      }
      long amount_to_copy =
        left_to_copy < buffer_amount_available ? left_to_copy : buffer_amount_available;
      char* copy_to = current_buffer_data + current_buffer_written;

      std::memcpy(copy_to, copy_from, amount_to_copy);
      copy_from = copy_from + amount_to_copy;
      current_buffer_written += amount_to_copy;
      total_written += amount_to_copy;
      left_to_copy -= amount_to_copy;
    }
  }

  bool supports_device_write() const override { return true; }

  void device_write(void const* gpu_data, size_t size, cuda::stream_ref stream) override
  {
    JNIEnv* env           = cudf::jni::get_jni_env(jvm);
    long left_to_copy     = static_cast<long>(size);
    char const* copy_from = static_cast<char const*>(gpu_data);
    while (left_to_copy > 0) {
      long buffer_amount_available = current_buffer_len - current_buffer_written;
      if (buffer_amount_available <= 0) {
        // should never be < 0, but just to be safe
        stream.sync();
        rotate_buffer(env);
        buffer_amount_available = current_buffer_len - current_buffer_written;
      }
      long amount_to_copy =
        left_to_copy < buffer_amount_available ? left_to_copy : buffer_amount_available;
      char* copy_to = current_buffer_data + current_buffer_written;

      CUDF_CUDA_TRY(
        cudaMemcpyAsync(copy_to, copy_from, amount_to_copy, cudaMemcpyDefault, stream.get()));

      copy_from = copy_from + amount_to_copy;
      current_buffer_written += amount_to_copy;
      total_written += amount_to_copy;
      left_to_copy -= amount_to_copy;
    }
    stream.sync();
  }

  std::future<void> device_write_async(void const* gpu_data,
                                       size_t size,
                                       cuda::stream_ref stream) override
  {
    // Call the sync version until figuring out how to write asynchronously.
    device_write(gpu_data, size, stream);
    return std::async(std::launch::deferred, [] {});
  }

  void flush() override
  {
    if (current_buffer_written > 0) {
      JNIEnv* env = cudf::jni::get_jni_env(jvm);
      handle_buffer(env, current_buffer, current_buffer_written);
      current_buffer         = del_global_ref(env, current_buffer);
      current_buffer_len     = 0;
      current_buffer_data    = nullptr;
      current_buffer_written = 0;
    }
  }

  size_t bytes_written() override { return total_written; }

  void set_alloc_size(long size) { this->alloc_size = size; }

 private:
  void rotate_buffer(JNIEnv* env)
  {
    if (current_buffer != nullptr) { handle_buffer(env, current_buffer, current_buffer_written); }
    current_buffer         = del_global_ref(env, current_buffer);
    jobject tmp_buffer     = allocate_host_buffer(env, alloc_size, true, host_memory_allocator);
    current_buffer         = add_global_ref(env, tmp_buffer);
    current_buffer_len     = get_host_buffer_length(env, current_buffer);
    current_buffer_data    = reinterpret_cast<char*>(get_host_buffer_address(env, current_buffer));
    current_buffer_written = 0;
  }

  void handle_buffer(JNIEnv* env, jobject buffer, jlong len)
  {
    env->CallVoidMethod(callback, handle_buffer_method, buffer, len);
    if (env->ExceptionCheck()) { throw std::runtime_error("handleBuffer threw an exception"); }
  }

  JavaVM* jvm;
  jobject callback;
  jmethodID handle_buffer_method;
  jobject current_buffer      = nullptr;
  char* current_buffer_data   = nullptr;
  long current_buffer_len     = 0;
  long current_buffer_written = 0;
  size_t total_written        = 0;
  long alloc_size             = MINIMUM_WRITE_BUFFER_SIZE;
  jobject host_memory_allocator;
};

}  // namespace cudf::jni
