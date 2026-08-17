/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "partitioned_retrieve_kernels.cuh"
#include "ref_types.cuh"

namespace cudf::detail {

template std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                   std::unique_ptr<rmm::device_uvector<size_type>>>
launch_partitioned_retrieve<true, primitive_count_ref_t>(probe_key_type const*,
                                                         thread_index_type,
                                                         size_type const*,
                                                         primitive_count_ref_t,
                                                         size_type,
                                                         cuda::stream_ref,
                                                         rmm::device_async_resource_ref);

template std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                   std::unique_ptr<rmm::device_uvector<size_type>>>
launch_partitioned_retrieve<true, nested_count_ref_t>(probe_key_type const*,
                                                      thread_index_type,
                                                      size_type const*,
                                                      nested_count_ref_t,
                                                      size_type,
                                                      cuda::stream_ref,
                                                      rmm::device_async_resource_ref);

template std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                   std::unique_ptr<rmm::device_uvector<size_type>>>
launch_partitioned_retrieve<true, flat_count_ref_t>(probe_key_type const*,
                                                    thread_index_type,
                                                    size_type const*,
                                                    flat_count_ref_t,
                                                    size_type,
                                                    cuda::stream_ref,
                                                    rmm::device_async_resource_ref);

}  // namespace cudf::detail
