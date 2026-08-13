/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/stream_pool.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <dlfcn.h>
#include <generated_cuda_runtime_api_meta.h>
#include <sanitizer.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

// This file is compiled into a separate library that is dynamically loaded with LD_PRELOAD at
// runtime to libcudf to override some stream-related symbols in libcudf. The goal of such a library
// is to verify if the stream/stream pool is being correctly forwarded between API calls.
//
// We control whether to override cudf::test::get_default_stream or
// cudf::get_default_stream with a compile-time flag. The behaviour of tests
// depend on whether STREAM_MODE_TESTING is defined:
// 1. If STREAM_MODE_TESTING is not defined, cudf::get_default_stream will
//    return a custom stream and stream_is_invalid will return true if any CUDA
//    API is called using any of CUDA's default stream constants
//    (cudaStreamLegacy, cudaStreamDefault, or cudaStreamPerThread). This check
//    is sufficient to ensure that cudf is using cudf::get_default_stream
//    everywhere internally rather than implicitly using stream 0,
//    cudaStreamDefault, cudaStreamLegacy, thrust execution policies, etc. It
//    is not sufficient to guarantee a stream-ordered API because it will not
//    identify places in the code that use cudf::get_default_stream instead of
//    properly forwarding along a user-provided stream.
// 2. If STREAM_MODE_TESTING compiler option is defined, cudf::test::get_default_stream
//    returns a custom stream and stream_is_invalid returns true if any CUDA
//    API is called using any stream other than cudf::test::get_default_stream.
//    This is a necessary and sufficient condition to ensure that libcudf is
//    properly passing streams through all of its (tested) APIs.

namespace cudf {

#ifdef STREAM_MODE_TESTING
namespace test {
#endif

rmm::cuda_stream_view const get_default_stream()
{
  static rmm::cuda_stream stream{};
  return stream;
}

#ifdef STREAM_MODE_TESTING
}  // namespace test
#endif

#ifdef STREAM_MODE_TESTING
namespace detail {

/**
 * @brief Implementation of `cuda_stream_pool` that always returns the
 * `cudf::test::get_default_stream()`
 */
class test_cuda_stream_pool : public cuda_stream_pool {
 public:
  rmm::cuda_stream_view get_stream() override { return cudf::test::get_default_stream(); }
  [[maybe_unused]] rmm::cuda_stream_view get_stream(stream_id_type stream_id) override
  {
    return cudf::test::get_default_stream();
  }

  std::vector<rmm::cuda_stream_view> get_streams(std::size_t count) override
  {
    return std::vector<rmm::cuda_stream_view>(count, cudf::test::get_default_stream());
  }

  [[nodiscard]] std::size_t get_stream_pool_size() const override { return 1UL; }
};

cuda_stream_pool* create_global_cuda_stream_pool() { return new test_cuda_stream_pool(); }

}  // namespace detail
#endif

}  // namespace cudf

bool stream_is_invalid(cudaStream_t stream)
{
#ifdef STREAM_MODE_TESTING
  // In this mode the _only_ valid stream is the one returned by cudf::test::get_default_stream.
  return (stream != cudf::test::get_default_stream().value());
#else
  // We explicitly list the possibilities rather than using
  // `cudf::get_default_stream().value()` because there is no guarantee that
  // `thrust::device` and the default value of
  // `cudf::get_default_stream().value()` are actually the same. At present, the
  // former is `cudaStreamLegacy` while the latter is 0.
  return (stream == cudaStreamDefault) || (stream == cudaStreamLegacy) ||
         (stream == cudaStreamPerThread);
#endif
}

/**
 * @brief Raise an error if stream is invalid.
 */
void check_stream_and_error(cudaStream_t stream)
{
  if (stream_is_invalid(stream)) {
    char const* env_stream_error_mode{std::getenv("GTEST_CUDF_STREAM_ERROR_MODE")};
    if (env_stream_error_mode && !strcmp(env_stream_error_mode, "print")) {
      std::cout << "cudf_identify_stream_usage found unexpected stream!" << std::endl;
    } else {
      throw std::runtime_error("cudf_identify_stream_usage found unexpected stream!");
    }
  }
}

class sanitizer_subscriber {
 public:
  sanitizer_subscriber();
  ~sanitizer_subscriber();

 private:
  Sanitizer_SubscriberHandle handle;

  static void check_result(SanitizerResult result);

  template <typename Args, cudaStream_t Args::* Field>
  static void check_stream_arg(Sanitizer_CallbackData const* cbdata);

  void callback(Sanitizer_CallbackDomain domain, Sanitizer_CallbackId cbid, void const* cbdata);
};

sanitizer_subscriber::sanitizer_subscriber()
{
  auto const cb = [](void* userdata,
                     Sanitizer_CallbackDomain domain,
                     Sanitizer_CallbackId cbid,
                     void const* cbdata) {
    auto* subscriber = static_cast<sanitizer_subscriber*>(userdata);
    subscriber->callback(domain, cbid, cbdata);
  };
  check_result(sanitizerSubscribe(&this->handle, cb, this));

  check_result(sanitizerEnableDomain(1, this->handle, SANITIZER_CB_DOMAIN_RUNTIME_API));
}

sanitizer_subscriber::~sanitizer_subscriber() { check_result(sanitizerUnsubscribe(this->handle)); }

void sanitizer_subscriber::check_result(SanitizerResult result)
{
  if (result != SANITIZER_SUCCESS) {
    char const* str;
    sanitizerGetResultString(result, &str);
    throw std::runtime_error(std::string("Sanitizer error: ") + str);
  }
}

template <typename Args, cudaStream_t Args::* Field>
void sanitizer_subscriber::check_stream_arg(Sanitizer_CallbackData const* cbdata)
{
  auto const* args = static_cast<Args const*>(cbdata->functionParams);
  check_stream_and_error(args->*Field);
}

// `generated_cuda_runtime_api_meta.h` is provided by the CUDA Toolkit/Compute Sanitizer.
// It defines versioned callback parameter structs named like
// `cudaMemcpyAsync_v3020_params`, where the numeric suffix identifies the CUDA runtime
// API version associated with that parameter layout.
#define CHECK_STREAM_ARG(call, version, field)                \
  case SANITIZER_CBID_RUNTIME_API_##call: {                   \
    using args_t = call##_v##version##_params;                \
    check_stream_arg<args_t, &args_t::field>(runtime_cbdata); \
  } break

void sanitizer_subscriber::callback(Sanitizer_CallbackDomain domain,
                                    Sanitizer_CallbackId cbid,
                                    const void* cbdata)
{
  switch (domain) {
    case SANITIZER_CB_DOMAIN_RUNTIME_API: {
      auto const* runtime_cbdata = static_cast<Sanitizer_CallbackData const*>(cbdata);

      if (runtime_cbdata->callbackSite == SANITIZER_API_ENTER) {
        switch (cbid) {
          CHECK_STREAM_ARG(cudaEventRecord, 3020, stream);
          CHECK_STREAM_ARG(cudaEventRecord_ptsz, 7000, stream);
          CHECK_STREAM_ARG(cudaEventRecordWithFlags, 11010, stream);
          CHECK_STREAM_ARG(cudaEventRecordWithFlags_ptsz, 11010, stream);
          CHECK_STREAM_ARG(cudaLaunchKernel, 7000, stream);
          CHECK_STREAM_ARG(cudaLaunchKernel_ptsz, 7000, stream);
          CHECK_STREAM_ARG(cudaLaunchCooperativeKernel, 9000, stream);
          CHECK_STREAM_ARG(cudaLaunchCooperativeKernel_ptsz, 9000, stream);
          CHECK_STREAM_ARG(cudaLaunchHostFunc, 10000, stream);
          CHECK_STREAM_ARG(cudaLaunchHostFunc_ptsz, 10000, stream);
#if CUDART_VERSION >= 13000
          CHECK_STREAM_ARG(cudaMemPrefetchAsync, 12020, stream);
          CHECK_STREAM_ARG(cudaMemPrefetchAsync_ptsz, 12020, stream);
#else
          CHECK_STREAM_ARG(cudaMemPrefetchAsync, 8000, stream);
          CHECK_STREAM_ARG(cudaMemPrefetchAsync_ptsz, 8000, stream);
          CHECK_STREAM_ARG(cudaMemPrefetchAsync_v2, 12020, stream);
          CHECK_STREAM_ARG(cudaMemPrefetchAsync_v2_ptsz, 12020, stream);
#endif
          CHECK_STREAM_ARG(cudaMemcpy2DAsync, 3020, stream);
          CHECK_STREAM_ARG(cudaMemcpy2DAsync_ptsz, 7000, stream);
          CHECK_STREAM_ARG(cudaMemcpy2DFromArrayAsync, 3020, stream);
          CHECK_STREAM_ARG(cudaMemcpy2DFromArrayAsync_ptsz, 7000, stream);
          CHECK_STREAM_ARG(cudaMemcpy2DToArrayAsync, 3020, stream);
          CHECK_STREAM_ARG(cudaMemcpy2DToArrayAsync_ptsz, 7000, stream);
          CHECK_STREAM_ARG(cudaMemcpy3DAsync, 3020, stream);
          CHECK_STREAM_ARG(cudaMemcpy3DAsync_ptsz, 7000, stream);
          CHECK_STREAM_ARG(cudaMemcpy3DPeerAsync, 4000, stream);
          CHECK_STREAM_ARG(cudaMemcpy3DPeerAsync_ptsz, 7000, stream);
          CHECK_STREAM_ARG(cudaMemcpyAsync, 3020, stream);
          CHECK_STREAM_ARG(cudaMemcpyAsync_ptsz, 7000, stream);
          CHECK_STREAM_ARG(cudaMemcpyFromSymbolAsync, 3020, stream);
          CHECK_STREAM_ARG(cudaMemcpyFromSymbolAsync_ptsz, 7000, stream);
          CHECK_STREAM_ARG(cudaMemcpyToSymbolAsync, 3020, stream);
          CHECK_STREAM_ARG(cudaMemcpyToSymbolAsync_ptsz, 7000, stream);
          CHECK_STREAM_ARG(cudaMemset2DAsync, 3020, stream);
          CHECK_STREAM_ARG(cudaMemset2DAsync_ptsz, 7000, stream);
          CHECK_STREAM_ARG(cudaMemset3DAsync, 3020, stream);
          CHECK_STREAM_ARG(cudaMemset3DAsync_ptsz, 7000, stream);
          CHECK_STREAM_ARG(cudaMemsetAsync, 3020, stream);
          CHECK_STREAM_ARG(cudaMemsetAsync_ptsz, 7000, stream);
          CHECK_STREAM_ARG(cudaFreeAsync, 11020, hStream);
          CHECK_STREAM_ARG(cudaFreeAsync_ptsz, 11020, hStream);
          CHECK_STREAM_ARG(cudaMallocAsync, 11020, hStream);
          CHECK_STREAM_ARG(cudaMallocAsync_ptsz, 11020, hStream);
          CHECK_STREAM_ARG(cudaMallocFromPoolAsync, 11020, stream);
          CHECK_STREAM_ARG(cudaMallocFromPoolAsync_ptsz, 11020, stream);
        }
      }
    } break;
    default: break;
  }
}

#undef CHECK_STREAM_ARG

sanitizer_subscriber subscriber;
