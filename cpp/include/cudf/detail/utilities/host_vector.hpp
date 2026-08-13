/*
 *  SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/aligned.hpp>

#include <thrust/host_vector.h>

#include <cstddef>
#include <limits>
#include <new>  // for bad_alloc
#include <span>

namespace CUDF_EXPORT cudf {
namespace detail {

/*! \p rmm_host_allocator is a CUDA-specific host memory allocator
 *  that employs \c a `cudf::host_async_resource_ref` for allocation.
 *
 *  \see https://en.cppreference.com/cpp/memory/allocator
 */
template <typename T>
class rmm_host_allocator;

/*! \p rmm_host_allocator is a CUDA-specific host memory allocator
 *  that employs \c an `cudf::host_async_resource_ref` for allocation.
 *
 *  \see https://en.cppreference.com/cpp/memory/allocator
 */
template <>
class rmm_host_allocator<void> {
 public:
  using value_type      = void;            ///< The type of the elements in the allocator
  using pointer         = void*;           ///< The type returned by address() / allocate()
  using const_pointer   = void const*;     ///< The type returned by address()
  using size_type       = std::size_t;     ///< The type used for the size of the allocation
  using difference_type = std::ptrdiff_t;  ///< The type of the distance between two pointers

  /**
   * @brief converts a `rmm_host_allocator<void>` to `rmm_host_allocator<U>`
   */
  template <typename U>
  struct rebind {
    using other = rmm_host_allocator<U>;  ///< The rebound type
  };
};

/*! \p rmm_host_allocator is a CUDA-specific host memory allocator
 *  that employs \c `cudf::host_async_resource_ref` for allocation.
 *
 * The \p rmm_host_allocator provides an interface for host memory allocation through the user
 * provided \c `cudf::host_async_resource_ref`. The \p rmm_host_allocator does not take ownership of
 * this reference and therefore it is the user's responsibility to ensure its lifetime for the
 * duration of the lifetime of the \p rmm_host_allocator.
 *
 *  \see https://en.cppreference.com/cpp/memory/allocator
 */
template <typename T>
class rmm_host_allocator {
 public:
  using value_type      = T;               ///< The type of the elements in the allocator
  using pointer         = T*;              ///< The type returned by address() / allocate()
  using const_pointer   = T const*;        ///< The type returned by address()
  using reference       = T&;              ///< The parameter type for address()
  using const_reference = T const&;        ///< The parameter type for address()
  using size_type       = std::size_t;     ///< The type used for the size of the allocation
  using difference_type = std::ptrdiff_t;  ///< The type of the distance between two pointers

  using propagate_on_container_move_assignment = cuda::std::true_type;

  /**
   * @brief converts a `rmm_host_allocator<T>` to `rmm_host_allocator<U>`
   */
  template <typename U>
  struct rebind {
    using other = rmm_host_allocator<U>;  ///< The rebound type
  };

  /**
   * @brief Cannot declare an empty host allocator.
   */
  rmm_host_allocator() = delete;

  template <class... Properties>
  using async_host_resource_ref = cuda::mr::resource_ref<cuda::mr::host_accessible, Properties...>;

  /**
   * @brief Construct from a `cudf::host_async_resource_ref`
   */
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  template <typename ResourceType>
  rmm_host_allocator(ResourceType _mr, rmm::cuda_stream_view _stream)
    : mr(std::move(_mr)),
      stream(std::move(_stream)),
      _is_device_accessible{
        cuda::mr::synchronous_resource_with<ResourceType, cuda::mr::device_accessible>}
  {
  }

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  rmm_host_allocator(rmm_host_allocator const&) = default;

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  rmm_host_allocator(rmm_host_allocator&&) = default;

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  rmm_host_allocator& operator=(rmm_host_allocator const&) = default;

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  rmm_host_allocator& operator=(rmm_host_allocator&&) = default;

  /**
   * @brief This method allocates storage for objects in host memory.
   *
   *  @param cnt The number of objects to allocate.
   *  @return a \c pointer to the newly allocated objects.
   *  @note This method does not invoke \p value_type's constructor.
   *        It is the responsibility of the caller to initialize the
   *        objects at the returned \c pointer.
   */
  inline pointer allocate(size_type cnt)
  {
    if (cnt > this->max_size()) { throw std::bad_alloc(); }  // end if
    auto const result = mr.allocate(stream, cnt * sizeof(value_type), alignof(value_type));
    // Synchronize to ensure the memory is allocated before thrust::host_vector initialization
    // TODO: replace thrust::host_vector with a type that does not require synchronization
    stream.synchronize();
    return static_cast<pointer>(result);
  }

  /**
   * @brief This method deallocates host memory previously allocated
   *  with this \c rmm_host_allocator.
   *
   *  @param p A \c pointer to the previously allocated memory.
   *  @param cnt Number of objects that were previously allocated.
   *  @note This method does not invoke \p value_type's destructor.
   *        It is the responsibility of the caller to destroy
   *        the objects stored at \p p.
   */
  inline void deallocate(pointer p, size_type cnt) noexcept
  {
    mr.deallocate(stream, p, cnt * sizeof(value_type), alignof(value_type));
  }

  /**
   * @brief This method returns the maximum size of the \c cnt parameter
   *  accepted by the \p allocate() method.
   *
   *  @return The maximum number of objects that may be allocated
   *          by a single call to \p allocate().
   */
  [[nodiscard]] constexpr inline size_type max_size() const
  {
    return (std::numeric_limits<size_type>::max)() / sizeof(T);
  }

  /**
   * @brief This method tests this \p rmm_host_allocator for equality to
   *  another.
   *
   *  @param x The other \p rmm_host_allocator of interest.
   *  @return This method always returns \c true.
   */
  inline bool operator==(rmm_host_allocator const& x) const
  {
    return x.mr == mr && x.stream == stream;
  }

  /**
   * @brief This method tests this \p rmm_host_allocator for inequality
   *  to another.
   *
   *  @param x The other \p rmm_host_allocator of interest.
   *  @return This method always returns \c false.
   */
  inline bool operator!=(rmm_host_allocator const& x) const { return !operator==(x); }

  [[nodiscard]] bool is_device_accessible() const { return _is_device_accessible; }

 private:
  rmm::host_async_resource_ref mr;
  rmm::cuda_stream_view stream;
  bool _is_device_accessible;
};

/**
 * @brief A vector class with rmm host memory allocator
 */
template <typename T>
class host_vector : public thrust::host_vector<T, rmm_host_allocator<T>> {
 public:
  using base = thrust::host_vector<T, rmm_host_allocator<T>>;

  host_vector(rmm_host_allocator<T> const& alloc) : base(alloc) {}

  host_vector(size_t size, rmm_host_allocator<T> const& alloc) : base(size, alloc) {}

  [[nodiscard]] operator host_span<T const>() const
  {
    return host_span<T const>{
      base::data(), base::size(), base::get_allocator().is_device_accessible()};
  }

  [[nodiscard]] operator host_span<T>()
  {
    return host_span<T>{base::data(), base::size(), base::get_allocator().is_device_accessible()};
  }

  [[nodiscard]] operator std::span<T const>() const noexcept
  {
    return std::span<T const>(base::data(), base::size());
  }

  [[nodiscard]] operator std::span<T>() noexcept
  {
    return std::span<T>(base::data(), base::size());
  }
};

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
