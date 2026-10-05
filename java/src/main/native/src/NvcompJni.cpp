/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "check_nvcomp_output_sizes.hpp"
#include "cudf_jni_apis.hpp"
#include "error.hpp"

#include <rmm/device_uvector.hpp>

#include <nvcomp.h>
#include <nvcomp/lz4.h>
#include <nvcomp/zstd.h>

namespace {

void check_nvcomp_status(JNIEnv* env, nvcompStatus_t status)
{
  switch (status) {
    case nvcompSuccess: break;
    case nvcompErrorInvalidValue:
      cudf::jni::throw_java_exception(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "nvcomp invalid value");
      break;
    case nvcompErrorNotSupported:
      cudf::jni::throw_java_exception(
        env, cudf::jni::UNSUPPORTED_EXCEPTION_CLASS, "nvcomp unsupported");
      break;
    case nvcompErrorCannotDecompress:
      cudf::jni::throw_java_exception(
        env, cudf::jni::NVCOMP_EXCEPTION_CLASS, "nvcomp cannot decompress");
      break;
    case nvcompErrorCudaError:
      cudf::jni::throw_java_exception(
        env, cudf::jni::NVCOMP_CUDA_EXCEPTION_CLASS, "nvcomp CUDA error");
      break;
    case nvcompErrorInternal:
      cudf::jni::throw_java_exception(
        env, cudf::jni::NVCOMP_EXCEPTION_CLASS, "nvcomp internal error");
      break;
    default:
      cudf::jni::throw_java_exception(
        env, cudf::jni::NVCOMP_EXCEPTION_CLASS, "nvcomp unknown error");
      break;
  }
}

}  // anonymous namespace

extern "C" {

// methods for lz4
JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4CompressGetTempSize(
  JNIEnv* env, jclass, jlong j_batch_size, jlong j_max_chunk_size, jlong j_max_total_size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto batch_size       = static_cast<std::size_t>(j_batch_size);
    auto max_chunk_size   = static_cast<std::size_t>(j_max_chunk_size);
    auto total_size       = static_cast<std::size_t>(j_max_total_size);
    std::size_t temp_size = 0;
    auto status           = nvcompBatchedLZ4CompressGetTempSizeAsync(
      batch_size, max_chunk_size, nvcompBatchedLZ4CompressDefaultOpts, &temp_size, total_size);
    check_nvcomp_status(env, status);
    return static_cast<jlong>(temp_size);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4CompressGetMaxOutputChunkSize(JNIEnv* env,
                                                                             jclass,
                                                                             jlong j_max_chunk_size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto max_chunk_size         = static_cast<std::size_t>(j_max_chunk_size);
    std::size_t max_output_size = 0;
    auto status                 = nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
      max_chunk_size, nvcompBatchedLZ4CompressDefaultOpts, &max_output_size);
    check_nvcomp_status(env, status);
    return static_cast<jlong>(max_output_size);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4CompressAsync(JNIEnv* env,
                                                             jclass,
                                                             jlong j_in_ptrs,
                                                             jlong j_in_sizes,
                                                             jlong j_chunk_size,
                                                             jlong j_batch_size,
                                                             jlong j_temp_ptr,
                                                             jlong j_temp_size,
                                                             jlong j_out_ptrs,
                                                             jlong j_compressed_sizes_out_ptr,
                                                             jlong j_stream)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto in_ptrs              = reinterpret_cast<void const* const*>(j_in_ptrs);
    auto in_sizes             = reinterpret_cast<std::size_t const*>(j_in_sizes);
    auto chunk_size           = static_cast<std::size_t>(j_chunk_size);
    auto batch_size           = static_cast<std::size_t>(j_batch_size);
    auto temp_ptr             = reinterpret_cast<void*>(j_temp_ptr);
    auto temp_size            = static_cast<std::size_t>(j_temp_size);
    auto out_ptrs             = reinterpret_cast<void* const*>(j_out_ptrs);
    auto compressed_out_sizes = reinterpret_cast<std::size_t*>(j_compressed_sizes_out_ptr);
    auto stream               = reinterpret_cast<cudaStream_t>(j_stream);
    // FIXME how to use these statuses ? They are not used either in the corresponding
    // decompressor.
    auto comp_statuses = rmm::device_uvector<nvcompStatus_t>(batch_size, stream);
    auto status        = nvcompBatchedLZ4CompressAsync(in_ptrs,
                                                       in_sizes,
                                                       chunk_size,
                                                       batch_size,
                                                       temp_ptr,
                                                       temp_size,
                                                       out_ptrs,
                                                       compressed_out_sizes,
                                                       nvcompBatchedLZ4CompressDefaultOpts,
                                                       comp_statuses.data(),
                                                       stream);
    check_nvcomp_status(env, status);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4DecompressGetTempSize(
  JNIEnv* env, jclass, jlong j_batch_size, jlong j_chunk_size, jlong j_max_total_size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto batch_size       = static_cast<std::size_t>(j_batch_size);
    auto chunk_size       = static_cast<std::size_t>(j_chunk_size);
    auto total_size       = static_cast<std::size_t>(j_max_total_size);
    std::size_t temp_size = 0;
    auto status           = nvcompBatchedLZ4DecompressGetTempSizeAsync(
      batch_size, chunk_size, nvcompBatchedLZ4DecompressDefaultOpts, &temp_size, total_size);
    check_nvcomp_status(env, status);
    return static_cast<jlong>(temp_size);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4DecompressAsync(JNIEnv* env,
                                                               jclass,
                                                               jlong j_in_ptrs,
                                                               jlong j_in_sizes,
                                                               jlong j_out_sizes,
                                                               jlong j_batch_size,
                                                               jlong j_temp_ptr,
                                                               jlong j_temp_size,
                                                               jlong j_out_ptrs,
                                                               jlong j_stream)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto compressed_ptrs           = reinterpret_cast<void const* const*>(j_in_ptrs);
    auto compressed_sizes          = reinterpret_cast<std::size_t const*>(j_in_sizes);
    auto uncompressed_sizes        = reinterpret_cast<std::size_t const*>(j_out_sizes);
    auto batch_size                = static_cast<std::size_t>(j_batch_size);
    auto temp_ptr                  = reinterpret_cast<void*>(j_temp_ptr);
    auto temp_size                 = static_cast<std::size_t>(j_temp_size);
    auto uncompressed_ptrs         = reinterpret_cast<void* const*>(j_out_ptrs);
    auto stream                    = reinterpret_cast<cudaStream_t>(j_stream);
    auto uncompressed_statuses     = rmm::device_uvector<nvcompStatus_t>(batch_size, stream);
    auto actual_uncompressed_sizes = rmm::device_uvector<std::size_t>(batch_size, stream);
    auto status = nvcompBatchedLZ4DecompressAsync(compressed_ptrs,
                                                  compressed_sizes,
                                                  uncompressed_sizes,
                                                  actual_uncompressed_sizes.data(),
                                                  batch_size,
                                                  temp_ptr,
                                                  temp_size,
                                                  uncompressed_ptrs,
                                                  nvcompBatchedLZ4DecompressDefaultOpts,
                                                  uncompressed_statuses.data(),
                                                  stream);
    check_nvcomp_status(env, status);
    if (!cudf::java::check_nvcomp_output_sizes(
          uncompressed_sizes, actual_uncompressed_sizes.data(), batch_size, stream)) {
      cudf::jni::throw_java_exception(
        env, cudf::jni::NVCOMP_EXCEPTION_CLASS, "nvcomp decompress output size mismatch");
    }
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4GetDecompressSizeAsync(JNIEnv* env,
                                                                      jclass,
                                                                      jlong j_in_ptrs,
                                                                      jlong j_in_sizes,
                                                                      jlong j_out_sizes,
                                                                      jlong j_batch_size,
                                                                      jlong j_stream)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto compressed_ptrs    = reinterpret_cast<void const* const*>(j_in_ptrs);
    auto compressed_sizes   = reinterpret_cast<std::size_t const*>(j_in_sizes);
    auto uncompressed_sizes = reinterpret_cast<std::size_t*>(j_out_sizes);
    auto batch_size         = static_cast<std::size_t>(j_batch_size);
    auto stream             = reinterpret_cast<cudaStream_t>(j_stream);
    auto status             = nvcompBatchedLZ4GetDecompressSizeAsync(
      compressed_ptrs, compressed_sizes, uncompressed_sizes, batch_size, stream);
    check_nvcomp_status(env, status);
  }
  JNI_CATCH(env, );
}

// methods for zstd
JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedZstdCompressGetTempSize(
  JNIEnv* env, jclass, jlong j_batch_size, jlong j_max_chunk_size, jlong j_max_total_size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto batch_size       = static_cast<std::size_t>(j_batch_size);
    auto max_chunk_size   = static_cast<std::size_t>(j_max_chunk_size);
    auto total_size       = static_cast<std::size_t>(j_max_total_size);
    std::size_t temp_size = 0;
    auto status           = nvcompBatchedZstdCompressGetTempSizeAsync(
      batch_size, max_chunk_size, nvcompBatchedZstdCompressDefaultOpts, &temp_size, total_size);
    check_nvcomp_status(env, status);
    return static_cast<jlong>(temp_size);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedZstdCompressGetMaxOutputChunkSize(
  JNIEnv* env, jclass, jlong j_max_chunk_size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto max_chunk_size         = static_cast<std::size_t>(j_max_chunk_size);
    std::size_t max_output_size = 0;
    auto status                 = nvcompBatchedZstdCompressGetMaxOutputChunkSize(
      max_chunk_size, nvcompBatchedZstdCompressDefaultOpts, &max_output_size);
    check_nvcomp_status(env, status);
    return static_cast<jlong>(max_output_size);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedZstdCompressAsync(JNIEnv* env,
                                                              jclass,
                                                              jlong j_in_ptrs,
                                                              jlong j_in_sizes,
                                                              jlong j_chunk_size,
                                                              jlong j_batch_size,
                                                              jlong j_temp_ptr,
                                                              jlong j_temp_size,
                                                              jlong j_out_ptrs,
                                                              jlong j_compressed_sizes_out_ptr,
                                                              jlong j_stream)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto in_ptrs              = reinterpret_cast<void const* const*>(j_in_ptrs);
    auto in_sizes             = reinterpret_cast<std::size_t const*>(j_in_sizes);
    auto chunk_size           = static_cast<std::size_t>(j_chunk_size);
    auto batch_size           = static_cast<std::size_t>(j_batch_size);
    auto temp_ptr             = reinterpret_cast<void*>(j_temp_ptr);
    auto temp_size            = static_cast<std::size_t>(j_temp_size);
    auto out_ptrs             = reinterpret_cast<void* const*>(j_out_ptrs);
    auto compressed_out_sizes = reinterpret_cast<std::size_t*>(j_compressed_sizes_out_ptr);
    auto stream               = reinterpret_cast<cudaStream_t>(j_stream);
    // FIXME how to use these statuses ? They are not used either in the corresponding
    // decompressor.
    auto comp_statuses = rmm::device_uvector<nvcompStatus_t>(batch_size, stream);
    auto status        = nvcompBatchedZstdCompressAsync(in_ptrs,
                                                        in_sizes,
                                                        chunk_size,
                                                        batch_size,
                                                        temp_ptr,
                                                        temp_size,
                                                        out_ptrs,
                                                        compressed_out_sizes,
                                                        nvcompBatchedZstdCompressDefaultOpts,
                                                        comp_statuses.data(),
                                                        stream);
    check_nvcomp_status(env, status);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedZstdDecompressGetTempSize(
  JNIEnv* env, jclass, jlong j_batch_size, jlong j_chunk_size, jlong j_max_total_size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto batch_size       = static_cast<std::size_t>(j_batch_size);
    auto chunk_size       = static_cast<std::size_t>(j_chunk_size);
    auto total_size       = static_cast<std::size_t>(j_max_total_size);
    std::size_t temp_size = 0;
    auto status           = nvcompBatchedZstdDecompressGetTempSizeAsync(
      batch_size, chunk_size, nvcompBatchedZstdDecompressDefaultOpts, &temp_size, total_size);
    check_nvcomp_status(env, status);
    return static_cast<jlong>(temp_size);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedZstdDecompressAsync(JNIEnv* env,
                                                                jclass,
                                                                jlong j_in_ptrs,
                                                                jlong j_in_sizes,
                                                                jlong j_out_sizes,
                                                                jlong j_batch_size,
                                                                jlong j_temp_ptr,
                                                                jlong j_temp_size,
                                                                jlong j_out_ptrs,
                                                                jlong j_stream)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto compressed_ptrs           = reinterpret_cast<void const* const*>(j_in_ptrs);
    auto compressed_sizes          = reinterpret_cast<std::size_t const*>(j_in_sizes);
    auto uncompressed_sizes        = reinterpret_cast<std::size_t const*>(j_out_sizes);
    auto batch_size                = static_cast<std::size_t>(j_batch_size);
    auto temp_ptr                  = reinterpret_cast<void*>(j_temp_ptr);
    auto temp_size                 = static_cast<std::size_t>(j_temp_size);
    auto uncompressed_ptrs         = reinterpret_cast<void* const*>(j_out_ptrs);
    auto stream                    = reinterpret_cast<cudaStream_t>(j_stream);
    auto uncompressed_statuses     = rmm::device_uvector<nvcompStatus_t>(batch_size, stream);
    auto actual_uncompressed_sizes = rmm::device_uvector<std::size_t>(batch_size, stream);
    auto status = nvcompBatchedZstdDecompressAsync(compressed_ptrs,
                                                   compressed_sizes,
                                                   uncompressed_sizes,
                                                   actual_uncompressed_sizes.data(),
                                                   batch_size,
                                                   temp_ptr,
                                                   temp_size,
                                                   uncompressed_ptrs,
                                                   nvcompBatchedZstdDecompressDefaultOpts,
                                                   uncompressed_statuses.data(),
                                                   stream);
    check_nvcomp_status(env, status);
    if (!cudf::java::check_nvcomp_output_sizes(
          uncompressed_sizes, actual_uncompressed_sizes.data(), batch_size, stream)) {
      cudf::jni::throw_java_exception(
        env, cudf::jni::NVCOMP_EXCEPTION_CLASS, "nvcomp decompress output size mismatch");
    }
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedZstdGetDecompressSizeAsync(JNIEnv* env,
                                                                       jclass,
                                                                       jlong j_in_ptrs,
                                                                       jlong j_in_sizes,
                                                                       jlong j_out_sizes,
                                                                       jlong j_batch_size,
                                                                       jlong j_stream)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto compressed_ptrs    = reinterpret_cast<void const* const*>(j_in_ptrs);
    auto compressed_sizes   = reinterpret_cast<std::size_t const*>(j_in_sizes);
    auto uncompressed_sizes = reinterpret_cast<std::size_t*>(j_out_sizes);
    auto batch_size         = static_cast<std::size_t>(j_batch_size);
    auto stream             = reinterpret_cast<cudaStream_t>(j_stream);
    auto status             = nvcompBatchedZstdGetDecompressSizeAsync(
      compressed_ptrs, compressed_sizes, uncompressed_sizes, batch_size, stream);
    check_nvcomp_status(env, status);
  }
  JNI_CATCH(env, );
}

}  // extern "C"
