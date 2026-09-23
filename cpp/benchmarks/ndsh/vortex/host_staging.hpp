/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/interop.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/detail/error.hpp>

#include <cuda/stream_ref>

#include <memory>
#include <vector>

namespace ndsh::detail {

// Declare after buffer owners so failures drain work before releasing its inputs.
// Cleanup uses RMM's noexcept policy (Debug assertion, unchecked result in Release), not the
// reader adapter's terminate-on-drain-error policy. Explicit wait() still propagates CUDA errors.
class staging_stream_drain {
 public:
  explicit staging_stream_drain(cuda::stream_ref stream) : stream_{stream} {}
  ~staging_stream_drain()
  {
    if (pending_) { RMM_ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream_.get())); }
  }
  staging_stream_drain(staging_stream_drain const&)            = delete;
  staging_stream_drain& operator=(staging_stream_drain const&) = delete;

  void wait()
  {
    pending_ = false;
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream_.get()));
  }

 private:
  cuda::stream_ref stream_;
  bool pending_{true};
};

inline cudf::unique_device_array_t stage_host_chunk(cudf::table_view chunk,
                                                    cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr)
{
  cudf::unique_device_array_t host{nullptr, nullptr};
  {
    std::vector<std::unique_ptr<cudf::column>> owned_strings;
    std::vector<cudf::column_view> columns{chunk.begin(), chunk.end()};
    owned_strings.reserve(chunk.num_columns());
    staging_stream_drain drain{stream};
    for (auto& column : columns) {
      if (column.type().id() == cudf::type_id::STRING && column.size() != 0) {
        auto const strings = cudf::strings_column_view{column};
        if (strings.offset() != 0 || strings.size() != strings.offsets().size() - 1) {
          // Host export otherwise copies the parent's entire chars buffer.
          owned_strings.push_back(std::make_unique<cudf::column>(column, stream, mr));
          column = owned_strings.back()->view();
        }
      }
    }
    host = cudf::to_arrow_host(cudf::table_view{columns}, stream, mr);
    drain.wait();
  }
  // Include async frees of the staging copies.
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  return host;
}

}  // namespace ndsh::detail
