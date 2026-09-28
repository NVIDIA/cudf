/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"
#include "jni_utils.hpp"

#include <cudf/io/datasource.hpp>

namespace {

#define DATA_SOURCE_CLASS "ai/rapids/cudf/DataSource"

jclass DataSource_jclass;
jmethodID hostRead_method;
jmethodID hostReadBuff_method;
jmethodID onHostBufferDone_method;
jmethodID deviceRead_method;

}  // anonymous namespace

namespace cudf {
namespace jni {
bool cache_data_source_jni(JNIEnv* env)
{
  jclass cls = env->FindClass(DATA_SOURCE_CLASS);
  if (cls == nullptr) { return false; }

  hostRead_method = env->GetMethodID(cls, "hostRead", "(JJJ)J");
  if (hostRead_method == nullptr) { return false; }

  hostReadBuff_method = env->GetMethodID(cls, "hostReadBuff", "(JJ)[J");
  if (hostReadBuff_method == nullptr) { return false; }

  onHostBufferDone_method = env->GetMethodID(cls, "onHostBufferDone", "(J)V");
  if (onHostBufferDone_method == nullptr) { return false; }

  deviceRead_method = env->GetMethodID(cls, "deviceRead", "(JJJJ)J");
  if (deviceRead_method == nullptr) { return false; }

  // Convert local reference to global so it cannot be garbage collected.
  DataSource_jclass = static_cast<jclass>(env->NewGlobalRef(cls));
  if (DataSource_jclass == nullptr) { return false; }
  return true;
}

void release_data_source_jni(JNIEnv* env)
{
  DataSource_jclass = cudf::jni::del_global_ref(env, DataSource_jclass);
}

class host_buffer_done_callback {
 public:
  explicit host_buffer_done_callback(JavaVM* jvm, jobject ds, long id) : jvm(jvm), ds(ds), id(id) {}

  host_buffer_done_callback(host_buffer_done_callback const& other) = delete;
  host_buffer_done_callback(host_buffer_done_callback&& other) noexcept
    : jvm(other.jvm), ds(other.ds), id(other.id)
  {
    other.jvm = nullptr;
    other.ds  = nullptr;
    other.id  = -1;
  }

  host_buffer_done_callback& operator=(host_buffer_done_callback&& other)      = delete;
  host_buffer_done_callback& operator=(host_buffer_done_callback const& other) = delete;

  ~host_buffer_done_callback()
  {
    // because we are in a destructor we cannot throw an exception, so for now we are
    // just going to keep the java exceptions around and have them be thrown when this
    // thread returns to the JVM. It might be kind of confusing, but we will not lose
    // them.
    if (jvm != nullptr) {
      // We cannot throw an exception in the destructor, so this is really best effort
      JNIEnv* env = nullptr;
      if (jvm->GetEnv(reinterpret_cast<void**>(&env), cudf::jni::MINIMUM_JNI_VERSION) == JNI_OK) {
        env->CallVoidMethod(this->ds, onHostBufferDone_method, id);
      }
    }
  }

 private:
  JavaVM* jvm;
  jobject ds;
  long id;
};

class jni_datasource : public cudf::io::datasource {
 public:
  explicit jni_datasource(
    JNIEnv* env, jobject ds, size_t ds_size, bool device_read_supported, size_t device_read_cutoff)
    : ds_size(ds_size),
      device_read_supported(device_read_supported),
      device_read_cutoff(device_read_cutoff)
  {
    if (env->GetJavaVM(&jvm) < 0) { throw std::runtime_error("GetJavaVM failed"); }
    this->ds = add_global_ref(env, ds);
  }

  virtual ~jni_datasource()
  {
    JNIEnv* env = nullptr;
    if (jvm->GetEnv(reinterpret_cast<void**>(&env), cudf::jni::MINIMUM_JNI_VERSION) == JNI_OK) {
      ds = del_global_ref(env, ds);
    }
    ds = nullptr;
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    JNIEnv* env = nullptr;
    if (jvm->GetEnv(reinterpret_cast<void**>(&env), cudf::jni::MINIMUM_JNI_VERSION) != JNI_OK) {
      throw cudf::jni::jni_exception("Could not load JNIEnv");
    }

    jlongArray jbuffer_info =
      static_cast<jlongArray>(env->CallObjectMethod(this->ds, hostReadBuff_method, offset, size));
    if (env->ExceptionOccurred()) { throw cudf::jni::jni_exception("Java exception in hostRead"); }

    cudf::jni::native_jlongArray buffer_info(env, jbuffer_info);
    auto ptr      = reinterpret_cast<uint8_t*>(buffer_info[0]);
    size_t length = buffer_info[1];
    long id       = buffer_info[2];

    cudf::jni::host_buffer_done_callback cb(this->jvm, this->ds, id);
    return std::make_unique<owning_buffer<cudf::jni::host_buffer_done_callback>>(
      std::move(cb), ptr, length);
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    JNIEnv* env = nullptr;
    if (jvm->GetEnv(reinterpret_cast<void**>(&env), cudf::jni::MINIMUM_JNI_VERSION) != JNI_OK) {
      throw cudf::jni::jni_exception("Could not load JNIEnv");
    }

    jlong amount_read =
      env->CallLongMethod(this->ds, hostRead_method, offset, size, reinterpret_cast<jlong>(dst));
    if (env->ExceptionOccurred()) { throw cudf::jni::jni_exception("Java exception in hostRead"); }
    return amount_read;
  }

  size_t size() const override { return ds_size; }

  bool supports_device_read() const override { return device_read_supported; }

  bool is_device_read_preferred(size_t size) const override
  {
    return device_read_supported && size >= device_read_cutoff;
  }

  size_t device_read(size_t offset, size_t size, uint8_t* dst, cuda::stream_ref stream) override
  {
    JNIEnv* env = nullptr;
    if (jvm->GetEnv(reinterpret_cast<void**>(&env), cudf::jni::MINIMUM_JNI_VERSION) != JNI_OK) {
      throw cudf::jni::jni_exception("Could not load JNIEnv");
    }

    jlong amount_read = env->CallLongMethod(this->ds,
                                            deviceRead_method,
                                            offset,
                                            size,
                                            reinterpret_cast<jlong>(dst),
                                            reinterpret_cast<jlong>(stream.get()));
    if (env->ExceptionOccurred()) {
      throw cudf::jni::jni_exception("Java exception in deviceRead");
    }
    return amount_read;
  }

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        cuda::stream_ref stream) override
  {
    auto amount_read = device_read(offset, size, dst, stream);
    // This is a bit ugly, but we don't have a good way or a need to return
    // a future for the read
    std::promise<size_t> ret;
    ret.set_value(amount_read);
    return ret.get_future();
  }

 private:
  size_t ds_size;
  bool device_read_supported;
  size_t device_read_cutoff;
  JavaVM* jvm;
  jobject ds;
};
}  // namespace jni
}  // namespace cudf

extern "C" {

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_DataSourceHelper_createWrapperDataSource(JNIEnv* env,
                                                             jclass,
                                                             jobject ds,
                                                             jlong ds_size,
                                                             jboolean device_read_supported,
                                                             jlong device_read_cutoff)
{
  JNI_NULL_CHECK(env, ds, "Null data source", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto source =
      new cudf::jni::jni_datasource(env, ds, ds_size, device_read_supported, device_read_cutoff);
    return reinterpret_cast<jlong>(source);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_DataSourceHelper_destroyWrapperDataSource(JNIEnv* env,
                                                                                     jclass,
                                                                                     jlong handle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    if (handle != 0) {
      auto source = reinterpret_cast<cudf::jni::jni_datasource*>(handle);
      delete (source);
    }
  }
  JNI_CATCH(env, );
}

}  // extern "C"
