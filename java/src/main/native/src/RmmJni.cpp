/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"
#include "jni_cccl_any_resource.hpp"

#include <cudf/logger.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/aligned_resource_adaptor.hpp>
#include <rmm/mr/arena_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/logging_resource_adaptor.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using cudf::jni::delete_jni_resource;
using cudf::jni::get_resource;
using cudf::jni::make_jni_resource;

constexpr char const* RMM_EXCEPTION_CLASS = "ai/rapids/cudf/RmmException";

/**
 * @brief Implementation class for tracking resource adaptor.
 * This class is not copyable due to atomic/mutex members.
 * Owns the upstream resource via any_resource.
 */
class tracking_resource_adaptor_impl {
 public:
  tracking_resource_adaptor_impl(cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
                                 std::size_t size_alignment)
    : upstream_{std::move(upstream)}, size_align{size_alignment}
  {
  }

  std::size_t get_total_allocated() { return total_allocated.load(); }

  std::size_t get_max_total_allocated() { return max_total_allocated; }

  void reset_scoped_max_total_allocated(std::size_t initial_value)
  {
    std::scoped_lock lock(max_total_allocated_mutex);
    scoped_allocated           = initial_value;
    scoped_max_total_allocated = initial_value;
  }

  std::size_t get_scoped_max_total_allocated()
  {
    std::scoped_lock lock(max_total_allocated_mutex);
    return scoped_max_total_allocated;
  }

  void* allocate(cuda::stream_ref stream,
                 std::size_t num_bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    auto const result = upstream_.allocate(stream, num_bytes, size_align);
    if (result) {
      total_allocated += num_bytes;
      scoped_allocated += num_bytes;
      std::scoped_lock lock(max_total_allocated_mutex);
      max_total_allocated        = std::max(total_allocated.load(), max_total_allocated);
      scoped_max_total_allocated = std::max(scoped_allocated.load(), scoped_max_total_allocated);
    }
    return result;
  }

  void deallocate(cuda::stream_ref stream,
                  void* p,
                  std::size_t size,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    upstream_.deallocate(stream, p, size, size_align);
    if (p) {
      total_allocated -= size;
      scoped_allocated -= size;
    }
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
  }

  bool operator==(tracking_resource_adaptor_impl const& other) const noexcept
  {
    return this == &other;
  }

  friend void get_property(tracking_resource_adaptor_impl const&,
                           cuda::mr::device_accessible) noexcept
  {
  }

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
  std::size_t const size_align;
  // sum of what is currently allocated
  std::atomic_size_t total_allocated{0};
  // the maximum total allocated for the lifetime of this class
  std::size_t max_total_allocated{0};
  // the sum of what is currently outstanding from the last
  // `reset_scoped_max_total_allocated` call. This can be negative.
  std::atomic_long scoped_allocated{0};
  // the maximum total allocated relative to the last
  // `reset_scoped_max_total_allocated` call.
  long scoped_max_total_allocated{0};
  std::mutex max_total_allocated_mutex;
};
static_assert(cuda::mr::resource_with<tracking_resource_adaptor_impl, cuda::mr::device_accessible>);

/**
 * @brief Tracking resource adaptor with reference-counted shared ownership.
 * This wrapper holds a shared_ptr to the impl and forwards resource operations.
 * It satisfies the CCCL resource concept and is copyable for use with any_resource.
 */
class tracking_resource_adaptor {
 public:
  tracking_resource_adaptor(cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
                            std::size_t size_alignment)
    : impl_{std::make_shared<tracking_resource_adaptor_impl>(std::move(upstream), size_alignment)}
  {
  }

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return impl_->allocate(stream, bytes, alignment);
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    impl_->deallocate(stream, ptr, bytes, alignment);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return impl_->allocate_sync(bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    impl_->deallocate_sync(ptr, bytes, alignment);
  }

  bool operator==(tracking_resource_adaptor const& other) const noexcept
  {
    return impl_ == other.impl_;
  }

  friend void get_property(tracking_resource_adaptor const&, cuda::mr::device_accessible) noexcept
  {
  }

  std::size_t get_total_allocated() { return impl_->get_total_allocated(); }
  std::size_t get_max_total_allocated() { return impl_->get_max_total_allocated(); }
  void reset_scoped_max_total_allocated(std::size_t initial_value)
  {
    impl_->reset_scoped_max_total_allocated(initial_value);
  }
  std::size_t get_scoped_max_total_allocated() { return impl_->get_scoped_max_total_allocated(); }

 private:
  std::shared_ptr<tracking_resource_adaptor_impl> impl_;
};
static_assert(cuda::mr::resource_with<tracking_resource_adaptor, cuda::mr::device_accessible>);

/**
 * @brief Implementation class for java event handler memory resource.
 * This class holds all the non-copyable JNI state and is wrapped in a shared_ptr.
 */
class java_event_handler_memory_resource_impl {
 public:
  java_event_handler_memory_resource_impl(
    JNIEnv* env,
    jobject jhandler,
    jlongArray jalloc_thresholds,
    jlongArray jdealloc_thresholds,
    cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
    tracking_resource_adaptor tracker)
    : upstream_{std::move(upstream)}, tracker_(std::move(tracker))
  {
    if (env->GetJavaVM(&jvm) < 0) { throw std::runtime_error("GetJavaVM failed"); }

    jclass cls = env->GetObjectClass(jhandler);
    if (cls == nullptr) { throw cudf::jni::jni_exception("class not found"); }
    on_alloc_fail_method = env->GetMethodID(cls, "onAllocFailure", "(JI)Z");
    if (on_alloc_fail_method == nullptr) {
      use_old_alloc_fail_interface = true;
      on_alloc_fail_method         = env->GetMethodID(cls, "onAllocFailure", "(J)Z");
      if (on_alloc_fail_method == nullptr) {
        throw cudf::jni::jni_exception("onAllocFailure method");
      }
    } else {
      use_old_alloc_fail_interface = false;
    }
    on_alloc_threshold_method = env->GetMethodID(cls, "onAllocThreshold", "(J)V");
    if (on_alloc_threshold_method == nullptr) {
      throw cudf::jni::jni_exception("onAllocThreshold method");
    }
    on_dealloc_threshold_method = env->GetMethodID(cls, "onDeallocThreshold", "(J)V");
    if (on_dealloc_threshold_method == nullptr) {
      throw cudf::jni::jni_exception("onDeallocThreshold method");
    }

    update_thresholds(env, alloc_thresholds, jalloc_thresholds);
    update_thresholds(env, dealloc_thresholds, jdealloc_thresholds);

    handler_obj = cudf::jni::add_global_ref(env, jhandler);
  }

  virtual ~java_event_handler_memory_resource_impl()
  {
    // This should normally be called by a JVM thread. If the JVM environment is missing then this
    // is likely being triggered by the C++ runtime during shutdown. In that case the JVM may
    // already be destroyed and this thread should not try to attach to get an environment.
    JNIEnv* env = nullptr;
    if (jvm->GetEnv(reinterpret_cast<void**>(&env), cudf::jni::MINIMUM_JNI_VERSION) == JNI_OK) {
      handler_obj = cudf::jni::del_global_ref(env, handler_obj);
    }
    handler_obj = nullptr;
  }

  virtual void* allocate(cuda::stream_ref stream,
                         std::size_t num_bytes,
                         std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    std::size_t total_before;
    void* result;
    // a non-zero retry_count signifies that the `on_alloc_fail`
    // callback is being invoked while re-attempting an allocation
    // that had previously failed.
    int retry_count = 0;
    while (true) {
      try {
        total_before = tracker_.get_total_allocated();
        result       = upstream_.allocate(stream, num_bytes, alignment);
        break;
      } catch (rmm::out_of_memory const& e) {
        if (!on_alloc_fail(num_bytes, retry_count++)) { throw; }
      }
    }
    auto total_after = tracker_.get_total_allocated();

    try {
      check_for_threshold_callback(total_before,
                                   total_after,
                                   alloc_thresholds,
                                   on_alloc_threshold_method,
                                   "onAllocThreshold",
                                   total_after);
    } catch (std::exception const& e) {
      // Free the allocation as app will think the exception means the memory was not allocated.
      upstream_.deallocate(stream, result, num_bytes, alignment);
      throw;
    }

    return result;
  }

  virtual void deallocate(cuda::stream_ref stream,
                          void* p,
                          std::size_t size,
                          std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    auto total_before = tracker_.get_total_allocated();
    upstream_.deallocate(stream, p, size, alignment);
    auto total_after = tracker_.get_total_allocated();
    check_for_threshold_callback(total_after,
                                 total_before,
                                 dealloc_thresholds,
                                 on_dealloc_threshold_method,
                                 "onDeallocThreshold",
                                 total_after);
  }

 protected:
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
  tracking_resource_adaptor tracker_;
  jmethodID on_alloc_fail_method;
  bool use_old_alloc_fail_interface;
  jmethodID on_alloc_threshold_method;
  jmethodID on_dealloc_threshold_method;
  // sorted memory thresholds to trigger callbacks
  std::vector<std::size_t> alloc_thresholds{};
  std::vector<std::size_t> dealloc_thresholds{};
  JavaVM* jvm;
  jobject handler_obj;

  static void update_thresholds(JNIEnv* env,
                                std::vector<std::size_t>& thresholds,
                                jlongArray from_java)
  {
    thresholds.clear();
    if (from_java != nullptr) {
      cudf::jni::native_jlongArray jvalues(env, from_java);
      thresholds.insert(thresholds.end(), jvalues.data(), jvalues.data() + jvalues.size());
    } else {
      // use a single, maximum-threshold value so we don't have to always check for the corner case.
      thresholds.push_back(std::numeric_limits<std::size_t>::max());
    }
  }

  bool on_alloc_fail(std::size_t num_bytes, int retry_count)
  {
    JNIEnv* env     = cudf::jni::get_jni_env(jvm);
    jboolean result = false;
    if (!use_old_alloc_fail_interface) {
      result = env->CallBooleanMethod(handler_obj,
                                      on_alloc_fail_method,
                                      static_cast<jlong>(num_bytes),
                                      static_cast<jint>(retry_count));
    } else {
      result =
        env->CallBooleanMethod(handler_obj, on_alloc_fail_method, static_cast<jlong>(num_bytes));
    }
    if (env->ExceptionCheck()) {
      throw std::runtime_error("onAllocFailure handler threw an exception");
    }
    return result;
  }

  void check_for_threshold_callback(std::size_t low,
                                    std::size_t high,
                                    std::vector<std::size_t> const& thresholds,
                                    jmethodID callback_method,
                                    char const* callback_name,
                                    std::size_t current_total)
  {
    if (high >= thresholds.front() && low < thresholds.back()) {
      // could use binary search, but assumption is threshold count is very small
      auto it = std::find_if(thresholds.begin(), thresholds.end(), [=](std::size_t t) -> bool {
        return low < t && high >= t;
      });
      if (it != thresholds.end()) {
        JNIEnv* env = cudf::jni::get_jni_env(jvm);
        env->CallVoidMethod(handler_obj, callback_method, current_total);
      }
    }
  }
};

/**
 * @brief Debug implementation that adds allocation/deallocation callbacks.
 */
class java_debug_event_handler_memory_resource_impl final
  : public java_event_handler_memory_resource_impl {
 public:
  java_debug_event_handler_memory_resource_impl(
    JNIEnv* env,
    jobject jhandler,
    jlongArray jalloc_thresholds,
    jlongArray jdealloc_thresholds,
    cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
    tracking_resource_adaptor tracker)
    : java_event_handler_memory_resource_impl(env,
                                              jhandler,
                                              jalloc_thresholds,
                                              jdealloc_thresholds,
                                              std::move(upstream),
                                              std::move(tracker))
  {
    jclass cls = env->GetObjectClass(jhandler);
    if (cls == nullptr) { throw cudf::jni::jni_exception("class not found"); }

    on_allocated_method = env->GetMethodID(cls, "onAllocated", "(J)V");
    if (on_allocated_method == nullptr) { throw cudf::jni::jni_exception("onAllocated method"); }

    on_deallocated_method = env->GetMethodID(cls, "onDeallocated", "(J)V");
    if (on_deallocated_method == nullptr) {
      throw cudf::jni::jni_exception("onDeallocated method");
    }
  }

  void* allocate(cuda::stream_ref stream,
                 std::size_t num_bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) override
  {
    void* result = java_event_handler_memory_resource_impl::allocate(stream, num_bytes, alignment);
    on_allocated_callback(num_bytes);
    return result;
  }

  void deallocate(cuda::stream_ref stream,
                  void* p,
                  std::size_t size,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept override
  {
    java_event_handler_memory_resource_impl::deallocate(stream, p, size, alignment);
    on_deallocated_callback(size);
  }

 private:
  jmethodID on_allocated_method;
  jmethodID on_deallocated_method;

  void on_allocated_callback(std::size_t num_bytes)
  {
    JNIEnv* env = cudf::jni::get_jni_env(jvm);
    env->CallVoidMethod(handler_obj, on_allocated_method, num_bytes);
    if (env->ExceptionCheck()) {
      throw std::runtime_error("onAllocated handler threw an exception");
    }
  }

  void on_deallocated_callback(std::size_t size)
  {
    JNIEnv* env = cudf::jni::get_jni_env(jvm);
    env->CallVoidMethod(handler_obj, on_deallocated_method, size);
  }
};

/**
 * @brief Copyable wrapper for java event handler that holds shared_ptr to impl.
 * Satisfies CCCL resource concept for use with device_async_resource_ref.
 */
class java_event_handler_memory_resource {
 public:
  java_event_handler_memory_resource(JNIEnv* env,
                                     jobject jhandler,
                                     jlongArray jalloc_thresholds,
                                     jlongArray jdealloc_thresholds,
                                     cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
                                     tracking_resource_adaptor tracker,
                                     bool enable_debug)
    : impl_(enable_debug
              ? std::make_shared<java_debug_event_handler_memory_resource_impl>(
                  env, jhandler, jalloc_thresholds, jdealloc_thresholds, upstream, tracker)
              : std::make_shared<java_event_handler_memory_resource_impl>(env,
                                                                          jhandler,
                                                                          jalloc_thresholds,
                                                                          jdealloc_thresholds,
                                                                          std::move(upstream),
                                                                          std::move(tracker)))
  {
  }

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return impl_->allocate(stream, bytes, alignment);
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    impl_->deallocate(stream, ptr, bytes, alignment);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
  }

  bool operator==(java_event_handler_memory_resource const& other) const noexcept
  {
    return impl_ == other.impl_;
  }

  friend void get_property(java_event_handler_memory_resource const&,
                           cuda::mr::device_accessible) noexcept
  {
  }

 private:
  std::shared_ptr<java_event_handler_memory_resource_impl> impl_;
};
static_assert(
  cuda::mr::resource_with<java_event_handler_memory_resource, cuda::mr::device_accessible>);

inline void log_system_error_noexcept(char const* operation,
                                      std::optional<int> error = std::nullopt,
                                      bool warning             = false) noexcept
{
  try {
    auto const* sep           = error.has_value() ? ": " : "";
    std::string const err_msg = error.has_value() ? std::system_category().message(*error) : "";
    if (warning) {
      CUDF_LOG_WARN(
        "%s failed for parallel pinned allocation%s%s", operation, sep, err_msg.c_str());
    } else {
      CUDF_LOG_ERROR(
        "%s failed for parallel pinned allocation%s%s", operation, sep, err_msg.c_str());
    }
  } catch (...) {
    // Logging must not mask the original exception or escape a noexcept cleanup path.
  }
}

inline void log_cuda_error_noexcept(char const* operation, cudaError_t error) noexcept
{
  try {
    CUDF_LOG_ERROR("%s failed for parallel pinned allocation: %s %s",
                   operation,
                   cudaGetErrorName(error),
                   cudaGetErrorString(error));
  } catch (...) {
    // Logging must not escape a noexcept cleanup path.
  }
}

/**
 * @brief Pinned host resource that parallelizes the first touch of its backing pages.
 *
 * In cudaHostAlloc, the CUDA driver faults and pins the whole allocation. This resource instead
 * mmaps anonymous memory, requests huge pages, and then spawns threads to touch the pages to handle
 * page faults concurrently. After the pages are physically backed the allocation is then registered
 * with CUDA. The RMM pool using this upstream resource is otherwise unchanged.
 */
class parallel_init_pinned_host_memory_resource final {
 public:
  explicit parallel_init_pinned_host_memory_resource(std::size_t initialization_threads)
    : initialization_threads_{initialization_threads}
  {
    CUDF_EXPECTS(initialization_threads_ > 0,
                 "parallel initialization thread count must be positive",
                 rmm::logic_error);
  }

  void* allocate([[maybe_unused]] cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate_sync(bytes, alignment);
  }

  void deallocate([[maybe_unused]] cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate_sync(ptr, bytes, alignment);
  }

  [[nodiscard]] void* allocate_sync(std::size_t bytes,
                                    std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    if (bytes == 0) { return nullptr; }
    CUDF_EXPECTS(std::has_single_bit(alignment),
                 "pinned allocation alignment must be a power of two",
                 rmm::bad_alloc);
    CUDF_EXPECTS(alignment <= system_page_size(),
                 "pinned allocation alignment cannot exceed the system page size",
                 rmm::bad_alloc);

    // Round the pool size up so that it is page-aligned.
    auto const mapping_bytes = page_aligned_size(bytes);
    void* allocation =
      ::mmap(nullptr, mapping_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (allocation == MAP_FAILED) {
      auto const error   = errno;
      auto const message = std::string{"mmap failed for parallel pinned allocation: "} +
                           std::system_category().message(error);
      if (error == ENOMEM) { throw rmm::out_of_memory{message}; }
      throw std::system_error(error, std::generic_category(), message);
    }

    try {
      // Mark these pages as DONTFORK so that if the JVM is forked during a DMA
      // the pages are not moved from underneath it due to copy on write semantics.
      if (::madvise(allocation, mapping_bytes, MADV_DONTFORK) != 0) {
        auto const error = errno;
        throw std::system_error(error, std::generic_category(), "madvise(MADV_DONTFORK) failed");
      }
      // Request huge-pages. This is an optimization to reduce page faults by orders of magnitude.
      // It is safe to do without affecting allocator behavior since RMM never gives the pages back
      // to the OS, and suballocates purely in user space. Note that mmap guarantees only base-page
      // alignment so edges may remain backed by base pages.
      if (::madvise(allocation, mapping_bytes, MADV_HUGEPAGE) != 0) {
        log_system_error_noexcept("madvise(MADV_HUGEPAGE)", errno, true);
      }
      // Concurrently pre-touch the pages to back the virtual range with physical memory.
      pretouch_parallel(allocation, mapping_bytes, initialization_threads_);
      // Pin and register the host memory range with CUDA. We do this once for the whole range
      // instead of parallelizing since it contends internally on driver locks.
      RMM_CUDA_TRY_ALLOC(cudaHostRegister(allocation, mapping_bytes, cudaHostRegisterDefault),
                         mapping_bytes);
      return allocation;
    } catch (...) {
      if (::munmap(allocation, mapping_bytes) != 0) {
        log_system_error_noexcept("munmap while rolling back", errno);
      }
      throw;
    }
  }

  void deallocate_sync(
    void* ptr,
    std::size_t bytes,
    [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    if (ptr == nullptr) { return; }
    auto const status = cudaHostUnregister(ptr);
    if (status != cudaSuccess) {
      cudaGetLastError();
      log_cuda_error_noexcept("cudaHostUnregister", status);
    }
    auto const mapping_bytes = page_aligned_size_noexcept(bytes);
    if (!mapping_bytes.has_value()) {
      log_system_error_noexcept("page alignment during deallocation");
    }
    // Fall back to unmapping the unaligned bytes.
    if (::munmap(ptr, mapping_bytes.value_or(bytes)) != 0) {
      log_system_error_noexcept("munmap", errno);
    }
  }

  [[nodiscard]] bool operator==(
    parallel_init_pinned_host_memory_resource const& other) const noexcept
  {
    return this == std::addressof(other);
  }

  /**
   * @brief Enables CCCL's `cuda::mr::device_accessible` property
   *
   * This overload declares that memory allocated by this resource is device accessible.
   */
  friend void get_property(parallel_init_pinned_host_memory_resource const&,
                           cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Enables CCCL's `cuda::mr::host_accessible` property
   *
   * This overload declares that memory allocated by this resource is host accessible.
   */
  friend void get_property(parallel_init_pinned_host_memory_resource const&,
                           cuda::mr::host_accessible) noexcept
  {
  }

 private:
  static std::size_t system_page_size()
  {
    static std::size_t const page_size = [] {
      long const value = ::sysconf(_SC_PAGESIZE);
      if (value <= 0) {
        throw std::system_error(errno, std::generic_category(), "sysconf(_SC_PAGESIZE) failed");
      }
      return static_cast<std::size_t>(value);
    }();
    return page_size;
  }

  static std::size_t page_aligned_size(std::size_t bytes)
  {
    auto const page_size = system_page_size();
    CUDF_EXPECTS(bytes <= std::numeric_limits<std::size_t>::max() - (page_size - 1),
                 "pinned allocation size overflows page alignment",
                 rmm::bad_alloc);
    return ((bytes + page_size - 1) / page_size) * page_size;
  }

  static std::optional<std::size_t> page_aligned_size_noexcept(std::size_t bytes) noexcept
  {
    try {
      return page_aligned_size(bytes);
    } catch (...) {
      return std::nullopt;
    }
  }

  static void pretouch_parallel(void* allocation, std::size_t bytes, std::size_t requested_threads)
  {
    // Partition by system page size. With huge pages this may touch more than necessary,
    // but is low overhead (just TLB hits) and ensures all pages are touched if madvise was ignored.
    auto const page_size    = system_page_size();
    auto const page_count   = bytes / page_size;
    auto const thread_count = [&]() {
      auto const needed_threads   = std::min(requested_threads, page_count);
      auto const hardware_threads = std::thread::hardware_concurrency();
      if (hardware_threads == 0) { return needed_threads; }
      return std::min(needed_threads, static_cast<std::size_t>(hardware_threads));
    }();

    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    std::atomic<bool> start_workers{false};
    // Importantly the bytes are volatile so that the writes are not compiled away.
    auto* const base      = static_cast<std::uint8_t volatile*>(allocation);
    std::size_t next_page = 0;

    // Note that these threads will inherit the caller's affinity/mempolicy. We respect the
    // caller's placement and do not impose any additional constraints. If the calling thread
    // allows multiple NUMA nodes, the resultant pages could span multiple nodes.
    try {
      for (std::size_t worker_index = 0; worker_index < thread_count; ++worker_index) {
        auto const pages_for_worker =
          page_count / thread_count + (worker_index < page_count % thread_count ? 1 : 0);
        auto const first_page = next_page;
        next_page += pages_for_worker;
        workers.emplace_back([base, first_page, pages_for_worker, page_size, &start_workers]() {
          // Wait until every thread starts up before pre-touching. Otherwise, an mmap call
          // to allocate a thread's stack - which needs the mmap_lock in write mode - can block
          // waiting on another thread that is already handling a page fault and holding the
          // mmap_lock for reading. This is especially important with huge pages where page faults
          // are longer.
          start_workers.wait(false);
          for (std::size_t page = 0; page < pages_for_worker; ++page) {
            base[(first_page + page) * page_size] = 0;
          }
        });
      }
      start_workers.store(true);
      start_workers.notify_all();
      for (auto& worker : workers) {
        worker.join();
      }
    } catch (...) {
      auto const exception = std::current_exception();
      start_workers.store(true);
      start_workers.notify_all();
      // Try to join every worker; if a join throws leaving a joinable thread, terminate at the end.
      bool has_unjoined = false;
      for (auto& worker : workers) {
        if (worker.joinable()) {
          try {
            worker.join();
          } catch (...) {
            has_unjoined |= worker.joinable();
          }
        }
      }
      if (has_unjoined) { std::terminate(); }
      std::rethrow_exception(exception);
    }
  }

  std::size_t initialization_threads_;
};

static_assert(cuda::mr::synchronous_resource_with<parallel_init_pinned_host_memory_resource,
                                                  cuda::mr::device_accessible,
                                                  cuda::mr::host_accessible>);
static_assert(cuda::mr::resource_with<parallel_init_pinned_host_memory_resource,
                                      cuda::mr::device_accessible,
                                      cuda::mr::host_accessible>);

inline auto& prior_cudf_pinned_mr()
{
  static cuda::mr::resource_ref<cuda::mr::host_accessible, cuda::mr::device_accessible>
    _prior_cudf_pinned_mr = cudf::get_pinned_memory_resource();
  return _prior_cudf_pinned_mr;
}

/**
 * This is a pinned fallback memory resource that will try to allocate from the provided
 * `pool` resource if the requested size is less than or equal to the pool size, otherwise it
 * will fall back to the prior resource used by cuDF `prior_cudf_pinned_mr`.
 *
 * We detect whether a pointer to free is inside of the pool by checking its address (see
 * constructor).
 *
 * Most of this comes directly from `pinned_host_memory_resource` in RMM.
 */
class pinned_fallback_host_memory_resource {
 public:
  pinned_fallback_host_memory_resource(rmm::mr::pool_memory_resource pool_) : pool{pool_}
  {
    auto pool_size = pool.pool_size();
    pool_begin     = pool.allocate_sync(pool_size);
    pool_end       = static_cast<void*>(static_cast<uint8_t*>(pool_begin) + pool_size);
    pool.deallocate_sync(pool_begin, pool_size);
  }

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    if (bytes <= pool.pool_size()) {
      try {
        return pool.allocate(stream, bytes, alignment);
      } catch (...) {
        // If the pool is exhausted, fall back to the upstream memory resource
      }
    }
    return prior_cudf_pinned_mr().allocate(stream, bytes);
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    if (bytes <= pool.pool_size() && ptr >= pool_begin && ptr < pool_end) {
      pool.deallocate(stream, ptr, bytes, alignment);
    } else {
      prior_cudf_pinned_mr().deallocate(stream, ptr, bytes);
    }
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
  }

  bool operator==(pinned_fallback_host_memory_resource const& other) const noexcept
  {
    return pool == other.pool;
  }

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that a `pinned_host_memory_resource` provides device accessible memory
   */
  friend void get_property(pinned_fallback_host_memory_resource const&,
                           cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Enables the `cuda::mr::host_accessible` property
   *
   * This property declares that a `pinned_host_memory_resource` provides host accessible memory
   */
  friend void get_property(pinned_fallback_host_memory_resource const&,
                           cuda::mr::host_accessible) noexcept
  {
  }

 private:
  rmm::mr::pool_memory_resource pool;
  void* pool_begin;
  void* pool_end;
};

// carryover from RMM pinned_host_memory_resource
static_assert(cuda::mr::resource_with<pinned_fallback_host_memory_resource,
                                      cuda::mr::device_accessible,
                                      cuda::mr::host_accessible>);

// we set this to our fallback resource if we have set it.
std::unique_ptr<pinned_fallback_host_memory_resource> pinned_fallback_mr;

}  // anonymous namespace

extern "C" {

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_initDefaultCudaDevice(JNIEnv* env, jclass clazz)
{
  // make sure the CUDA device is setup in the context
  cudaError_t cuda_status = cudaFree(0);
  cudf::jni::jni_cuda_check(env, cuda_status);
  int device_id;
  cuda_status = cudaGetDevice(&device_id);
  cudf::jni::jni_cuda_check(env, cuda_status);
  // Now that RMM has successfully initialized, setup all threads calling
  // cudf to use the same device RMM is using.
  cudf::jni::set_cudf_device(device_id);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_cleanupDefaultCudaDevice(JNIEnv* env, jclass clazz)
{
  cudf::jni::set_cudf_device(cudaInvalidDeviceId);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_allocInternal(JNIEnv* env,
                                                              jclass clazz,
                                                              jlong size,
                                                              jlong stream)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref();
    auto c_stream                     = cuda::stream_ref(reinterpret_cast<cudaStream_t>(stream));
    void* ret                         = mr.allocate(c_stream, size);
    return reinterpret_cast<jlong>(ret);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_Rmm_free(JNIEnv* env, jclass clazz, jlong ptr, jlong size, jlong stream)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref();
    void* cptr                        = reinterpret_cast<void*>(ptr);
    auto c_stream                     = cuda::stream_ref(reinterpret_cast<cudaStream_t>(stream));
    mr.deallocate(c_stream, cptr, size);
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_freeDeviceBuffer(JNIEnv* env,
                                                                jclass clazz,
                                                                jlong ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    rmm::device_buffer* cptr = reinterpret_cast<rmm::device_buffer*>(ptr);
    delete cptr;
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_allocCudaInternal(JNIEnv* env,
                                                                  jclass clazz,
                                                                  jlong size,
                                                                  jlong stream)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    void* ptr{nullptr};
    RMM_CUDA_TRY_ALLOC(cudaMalloc(&ptr, size), size);
    return reinterpret_cast<jlong>(ptr);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_Rmm_freeCuda(JNIEnv* env, jclass clazz, jlong ptr, jlong size, jlong stream)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    void* cptr = reinterpret_cast<void*>(ptr);
    RMM_ASSERT_CUDA_SUCCESS(cudaFree(cptr));
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_newCudaMemoryResource(JNIEnv* env, jclass clazz)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    return make_jni_resource(rmm::mr::cuda_memory_resource{});
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_releaseCudaMemoryResource(JNIEnv* env,
                                                                         jclass clazz,
                                                                         jlong ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete_jni_resource(ptr);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_newManagedMemoryResource(JNIEnv* env, jclass clazz)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    return make_jni_resource(rmm::mr::managed_memory_resource{});
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_releaseManagedMemoryResource(JNIEnv* env,
                                                                            jclass clazz,
                                                                            jlong ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete_jni_resource(ptr);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_newPoolMemoryResource(
  JNIEnv* env, jclass clazz, jlong child, jlong init, jlong max)
{
  JNI_NULL_CHECK(env, child, "child is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto upstream = get_resource(child);
    return make_jni_resource(rmm::mr::pool_memory_resource{
      upstream, static_cast<std::size_t>(init), static_cast<std::size_t>(max)});
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_releasePoolMemoryResource(JNIEnv* env,
                                                                         jclass clazz,
                                                                         jlong ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete_jni_resource(ptr);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_newArenaMemoryResource(
  JNIEnv* env, jclass clazz, jlong child, jlong init, jboolean dump_on_oom)
{
  JNI_NULL_CHECK(env, child, "child is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto upstream = get_resource(child);
    return make_jni_resource(rmm::mr::arena_memory_resource{
      upstream, static_cast<std::size_t>(init), static_cast<bool>(dump_on_oom)});
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_releaseArenaMemoryResource(JNIEnv* env,
                                                                          jclass clazz,
                                                                          jlong ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete_jni_resource(ptr);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_newCudaAsyncMemoryResource(
  JNIEnv* env, jclass clazz, jlong init, jlong release, jboolean fabric)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto handle_type =
      fabric ? std::optional{rmm::mr::cuda_async_memory_resource::allocation_handle_type::fabric}
             : std::nullopt;

    return make_jni_resource(rmm::mr::cuda_async_memory_resource{init, release, handle_type});
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_releaseCudaAsyncMemoryResource(JNIEnv* env,
                                                                              jclass clazz,
                                                                              jlong ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete_jni_resource(ptr);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_newLimitingResourceAdaptor(
  JNIEnv* env, jclass clazz, jlong child, jlong limit, jlong align)
{
  JNI_NULL_CHECK(env, child, "child is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto upstream = get_resource(child);
    return make_jni_resource(rmm::mr::limiting_resource_adaptor{
      upstream, static_cast<std::size_t>(limit), static_cast<std::size_t>(align)});
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_releaseLimitingResourceAdaptor(JNIEnv* env,
                                                                              jclass clazz,
                                                                              jlong ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete_jni_resource(ptr);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_newLoggingResourceAdaptor(
  JNIEnv* env, jclass clazz, jlong child, jint type, jstring jpath, jboolean auto_flush)
{
  JNI_NULL_CHECK(env, child, "child is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto upstream = get_resource(child);
    switch (type) {
      case 1:  // File
      {
        cudf::jni::native_jstring path(env, jpath);
        return make_jni_resource(
          rmm::mr::logging_resource_adaptor{upstream, path, static_cast<bool>(auto_flush)});
      }
      case 2:  // stdout
        return make_jni_resource(
          rmm::mr::logging_resource_adaptor{upstream, std::cout, static_cast<bool>(auto_flush)});
      case 3:  // stderr
        return make_jni_resource(
          rmm::mr::logging_resource_adaptor{upstream, std::cerr, static_cast<bool>(auto_flush)});
      default: throw std::logic_error("unsupported logging location type");
    }
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_releaseLoggingResourceAdaptor(JNIEnv* env,
                                                                             jclass clazz,
                                                                             jlong ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete_jni_resource(ptr);
  }
  JNI_CATCH(env, );
}

// Map to store tracking adaptors for metrics access.
// tracking_resource_adaptor is copyable (via shared_ptr to impl), so copies share state.
std::mutex tracking_adaptor_map_mutex;
std::unordered_map<jlong, tracking_resource_adaptor> tracking_adaptor_map;

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_newTrackingResourceAdaptor(JNIEnv* env,
                                                                           jclass clazz,
                                                                           jlong child,
                                                                           jlong align)
{
  JNI_NULL_CHECK(env, child, "child is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto upstream = get_resource(child);
    auto adaptor  = tracking_resource_adaptor(upstream, static_cast<std::size_t>(align));
    auto handle   = make_jni_resource(adaptor);
    // Store a copy in map for metrics access (copies share impl via shared_ptr)
    {
      std::lock_guard<std::mutex> lock(tracking_adaptor_map_mutex);
      tracking_adaptor_map.emplace(handle, adaptor);
    }
    return handle;
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_releaseTrackingResourceAdaptor(JNIEnv* env,
                                                                              jclass clazz,
                                                                              jlong ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    {
      std::lock_guard<std::mutex> lock(tracking_adaptor_map_mutex);
      tracking_adaptor_map.erase(ptr);
    }
    delete_jni_resource(ptr);
  }
  JNI_CATCH(env, );
}

// Helper to get tracking adaptor from the map
inline tracking_resource_adaptor& get_tracking_adaptor(jlong handle)
{
  std::lock_guard<std::mutex> lock(tracking_adaptor_map_mutex);
  auto it = tracking_adaptor_map.find(handle);
  if (it == tracking_adaptor_map.end()) {
    throw std::runtime_error("tracking adaptor not found for handle");
  }
  return it->second;
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_nativeGetTotalBytesAllocated(JNIEnv* env,
                                                                             jclass clazz,
                                                                             jlong ptr)
{
  JNI_NULL_CHECK(env, ptr, "adaptor is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto& mr = get_tracking_adaptor(ptr);
    return mr.get_total_allocated();
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_nativeGetMaxTotalBytesAllocated(JNIEnv* env,
                                                                                jclass clazz,
                                                                                jlong ptr)
{
  JNI_NULL_CHECK(env, ptr, "adaptor is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto& mr = get_tracking_adaptor(ptr);
    return mr.get_max_total_allocated();
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_nativeResetScopedMaxTotalBytesAllocated(JNIEnv* env,
                                                                                       jclass clazz,
                                                                                       jlong ptr,
                                                                                       jlong init)
{
  JNI_NULL_CHECK(env, ptr, "adaptor is null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto& mr = get_tracking_adaptor(ptr);
    mr.reset_scoped_max_total_allocated(init);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_nativeGetScopedMaxTotalBytesAllocated(JNIEnv* env,
                                                                                      jclass clazz,
                                                                                      jlong ptr)
{
  JNI_NULL_CHECK(env, ptr, "adaptor is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto& mr = get_tracking_adaptor(ptr);
    return mr.get_scoped_max_total_allocated();
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_Rmm_newEventHandlerResourceAdaptor(JNIEnv* env,
                                                       jclass,
                                                       jlong child,
                                                       jlong tracker,
                                                       jobject handler_obj,
                                                       jlongArray jalloc_thresholds,
                                                       jlongArray jdealloc_thresholds,
                                                       jboolean enable_debug)
{
  JNI_NULL_CHECK(env, child, "child is null", 0);
  JNI_NULL_CHECK(env, tracker, "tracker is null", 0);
  JNI_TRY
  {
    auto upstream = get_resource(child);
    auto t        = get_tracking_adaptor(tracker);
    return make_jni_resource(java_event_handler_memory_resource(
      env, handler_obj, jalloc_thresholds, jdealloc_thresholds, upstream, t, enable_debug));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_releaseEventHandlerResourceAdaptor(
  JNIEnv* env, jclass clazz, jlong ptr, jboolean enable_debug)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete_jni_resource(ptr);
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_setCurrentDeviceResourceInternal(JNIEnv* env,
                                                                                jclass clazz,
                                                                                jlong new_handle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::set_current_device_resource(get_resource(new_handle));
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_newPinnedPoolMemoryResource(JNIEnv* env,
                                                                            jclass clazz,
                                                                            jlong init,
                                                                            jlong max)
{
  JNI_ARG_CHECK(env, init >= 0, "initial pool size must not be negative", 0);
  JNI_ARG_CHECK(env, max >= 0, "maximum pool size must not be negative", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto pool =
      new rmm::mr::pool_memory_resource(rmm::mr::pinned_host_memory_resource{}, init, max);
    return reinterpret_cast<jlong>(pool);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_newParallelPinnedPoolMemoryResource(
  JNIEnv* env, jclass clazz, jlong pool_size, jint initialization_threads)
{
  JNI_ARG_CHECK(env, pool_size >= 0, "pool size must not be negative", 0);
  JNI_ARG_CHECK(env, initialization_threads > 0, "parallel init thread count must be positive", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto pool = new rmm::mr::pool_memory_resource(
      parallel_init_pinned_host_memory_resource{static_cast<std::size_t>(initialization_threads)},
      static_cast<std::size_t>(pool_size),
      static_cast<std::size_t>(pool_size));
    return reinterpret_cast<jlong>(pool);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_setCudfPinnedPoolMemoryResource(JNIEnv* env,
                                                                               jclass clazz,
                                                                               jlong pool_ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto pool = reinterpret_cast<rmm::mr::pool_memory_resource*>(pool_ptr);
    // create a pinned fallback pool that will allocate pinned memory
    // if the regular pinned pool is exhausted
    pinned_fallback_mr.reset(new pinned_fallback_host_memory_resource(*pool));
    prior_cudf_pinned_mr() = cudf::set_pinned_memory_resource(*pinned_fallback_mr);
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_releasePinnedPoolMemoryResource(JNIEnv* env,
                                                                               jclass clazz,
                                                                               jlong pool_ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    // set the cuio host memory resource to what it was before, or the same
    // if we didn't overwrite it with setCudfPinnedPoolMemoryResource
    cudf::set_pinned_memory_resource(prior_cudf_pinned_mr());
    pinned_fallback_mr.reset();
    delete reinterpret_cast<rmm::mr::pool_memory_resource*>(pool_ptr);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_allocFromPinnedPool(JNIEnv* env,
                                                                    jclass clazz,
                                                                    jlong pool_ptr,
                                                                    jlong size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto pool = reinterpret_cast<rmm::mr::pool_memory_resource*>(pool_ptr);
    void* ret = pool->allocate(cudf::get_default_stream(), size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    return reinterpret_cast<jlong>(ret);
  }
  JNI_CATCH_BEGIN(env, 0)
  catch (...) { return -1; }  // Catch and suppress all exceptions.
  // The return value of -1 indicates that the allocation failed.
  // This is different from the return value of 0, which indicates that the allocation succeeded
  // but the returned pointer is null (such cases can be due to allocating 0 bytes).
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_freeFromPinnedPool(
  JNIEnv* env, jclass clazz, jlong pool_ptr, jlong ptr, jlong size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto pool  = reinterpret_cast<rmm::mr::pool_memory_resource*>(pool_ptr);
    void* cptr = reinterpret_cast<void*>(ptr);
    pool->deallocate(cudf::get_default_stream(), cptr, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
  JNI_CATCH(env, );
}

// only for tests
JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_allocFromFallbackPinnedPool(JNIEnv* env,
                                                                            jclass clazz,
                                                                            jlong size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    void* ret = cudf::get_pinned_memory_resource().allocate(cudf::get_default_stream(), size);
    return reinterpret_cast<jlong>(ret);
  }
  JNI_CATCH(env, 0);
}

// only for tests
JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_freeFromFallbackPinnedPool(JNIEnv* env,
                                                                          jclass clazz,
                                                                          jlong ptr,
                                                                          jlong size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    void* cptr = reinterpret_cast<void*>(ptr);
    cudf::get_pinned_memory_resource().deallocate(cudf::get_default_stream(), cptr, size);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jboolean JNICALL Java_ai_rapids_cudf_Rmm_configureDefaultCudfPinnedPoolSizeImpl(
  JNIEnv* env, jclass clazz, jlong size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    return cudf::config_default_pinned_memory_resource(cudf::pinned_mr_options{size});
  }
  JNI_CATCH(env, false);
}
}
