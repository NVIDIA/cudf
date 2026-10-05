/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jni_utils.hpp"

#include <errno.h>
#include <fcntl.h>
#include <jni.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

extern "C" {

JNIEXPORT jobject JNICALL Java_ai_rapids_cudf_HostMemoryBufferNativeUtils_wrapRangeInBuffer(
  JNIEnv* env, jclass, jlong addr, jlong len)
{ return env->NewDirectByteBuffer(reinterpret_cast<void*>(addr), len); }

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_HostMemoryBufferNativeUtils_mmap(
  JNIEnv* env, jclass, jstring jpath, jint mode, jlong offset, jlong length)
{
  JNI_NULL_CHECK(env, jpath, "path is null", 0);
  JNI_ARG_CHECK(env, (mode == 0 || mode == 1), "bad mode value", 0);
  JNI_TRY
  {
    cudf::jni::native_jstring path(env, jpath);

    int fd = open(path.get(), (mode == 0) ? O_RDONLY : O_RDWR);
    if (fd == -1) { cudf::jni::throw_java_exception(env, "java/io/IOException", strerror(errno)); }

    void* address =
      mmap(NULL, length, (mode == 0) ? PROT_READ : PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset);
    if (address == MAP_FAILED) {
      char const* error_msg = strerror(errno);
      close(fd);
      cudf::jni::throw_java_exception(env, "java/io/IOException", error_msg);
    }

    close(fd);
    return reinterpret_cast<jlong>(address);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_HostMemoryBufferNativeUtils_munmap(JNIEnv* env,
                                                                              jclass,
                                                                              jlong address,
                                                                              jlong length)
{
  JNI_NULL_CHECK(env, address, "address is NULL", );
  JNI_TRY
  {
    int rc = munmap(reinterpret_cast<void*>(address), length);
    if (rc == -1) { cudf::jni::throw_java_exception(env, "java/io/IOException", strerror(errno)); }
  }
  JNI_CATCH(env, );
}

}  // extern "C"
