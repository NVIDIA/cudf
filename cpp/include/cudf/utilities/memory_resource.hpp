/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/cuda_device.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <concepts>
#include <utility>

/**
 * @file
 * @brief APIs for getting and setting the current device memory resource.
 */

namespace cudf {

/**
 * @addtogroup memory_resource
 * @{
 */

/**
 * @brief Get the current device memory resource reference.
 *
 * @return The current device memory resource reference.
 */
inline rmm::device_async_resource_ref get_current_device_resource_ref()
{
  return rmm::mr::get_current_device_resource_ref();
}

/**
 * @brief Non-owning references to the memory resources used by a cuDF operation.
 *
 * The output resource allocates memory returned to the caller. The temporary resource allocates
 * intermediate memory that is released before the operation returns. If allocations are made from
 * a resource ref, callers must construct an owning resource from the resource ref, keep that owning
 * resource alive for the allocation's lifetime, and use it for deallocation.
 */
class memory_resources {
 public:
  /**
   * @brief Construct with an explicit output resource and capture the current resource for
   * temporaries.
   *
   * This constructor is intentionally implicit so an existing resource object or resource ref can
   * be passed directly to an API accepting `memory_resources`.
   *
   * @tparam Resource Type that can construct a device asynchronous resource ref
   * @param output_mr Resource used for returned allocations
   */
  template <typename Resource>
  memory_resources(Resource&& output_mr)
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    requires std::constructible_from<rmm::device_async_resource_ref, Resource&&>
#endif
    : _output_mr{std::forward<Resource>(output_mr)},
      _temporary_mr{cudf::get_current_device_resource_ref()}
  {
  }

  /**
   * @brief Construct with explicit output and temporary resources.
   *
   * This constructor does not query the current device resource.
   *
   * @tparam OutputResource Type that can construct the output resource ref
   * @tparam TemporaryResource Type that can construct the temporary resource ref
   * @param output_mr Resource used for returned allocations
   * @param temporary_mr Resource used for intermediate allocations
   */
  template <typename OutputResource, typename TemporaryResource>
  memory_resources(OutputResource&& output_mr, TemporaryResource&& temporary_mr)
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    requires(std::constructible_from<rmm::device_async_resource_ref, OutputResource &&> and
             std::constructible_from<rmm::device_async_resource_ref, TemporaryResource &&>)
#endif
    : _output_mr{std::forward<OutputResource>(output_mr)},
      _temporary_mr{std::forward<TemporaryResource>(temporary_mr)}
  {
  }

  /**
   * @brief Return the resource used for allocations returned to the caller.
   *
   * @return Output device memory resource reference
   */
  [[nodiscard]] rmm::device_async_resource_ref get_output_mr() const noexcept { return _output_mr; }

  /**
   * @brief Return the resource used for intermediate allocations.
   *
   * @return Temporary device memory resource reference
   */
  [[nodiscard]] rmm::device_async_resource_ref get_temporary_mr() const noexcept
  {
    return _temporary_mr;
  }

 private:
  rmm::device_async_resource_ref _output_mr;
  rmm::device_async_resource_ref _temporary_mr;
};

/**
 * @brief Set the current device memory resource.
 *
 * @param mr The new device memory resource.
 * @return An owning any_resource holding the previous resource.
 */
inline cuda::mr::any_resource<cuda::mr::device_accessible> set_current_device_resource(
  cuda::mr::any_resource<cuda::mr::device_accessible> mr)
{
  return rmm::mr::set_current_device_resource(std::move(mr));
}

/**
 * @brief Set the current device memory resource reference.
 *
 * @deprecated Use `set_current_device_resource` instead.
 *
 * The object referenced by `mr` must outlive the last use of the resource, otherwise behavior is
 * undefined. It is the caller's responsibility to maintain the lifetime of the resource object.
 *
 * @param mr The new device memory resource reference.
 * @return An owning any_resource holding the previous resource.
 */
[[deprecated("Use set_current_device_resource instead.")]]  //
inline cuda::mr::any_resource<cuda::mr::device_accessible>
set_current_device_resource_ref(rmm::device_async_resource_ref mr)
{
  return set_current_device_resource(cuda::mr::any_resource<cuda::mr::device_accessible>{mr});
}

/**
 * @brief Reset the current device memory resource to the initial resource.
 *
 * @return An owning any_resource holding the previous resource.
 */
inline cuda::mr::any_resource<cuda::mr::device_accessible> reset_current_device_resource()
{
  return rmm::mr::reset_current_device_resource();
}

/**
 * @brief Reset the current device memory resource reference to the initial resource.
 *
 * @deprecated Use `reset_current_device_resource` instead.
 *
 * @return An owning any_resource holding the previous resource.
 */
[[deprecated("Use reset_current_device_resource instead.")]]  //
inline cuda::mr::any_resource<cuda::mr::device_accessible>
reset_current_device_resource_ref()
{
  return reset_current_device_resource();
}

/** @} */  // end of group
}  // namespace cudf
