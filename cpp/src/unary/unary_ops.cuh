/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/stream>
#include <thrust/transform.h>

namespace cudf {
namespace unary {
template <typename T, typename Tout, typename F>
struct launcher {
  static std::unique_ptr<cudf::column> launch(cudf::column_view const& input,
                                              cudf::unary_operator op,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
  {
    std::unique_ptr<cudf::column> output = [&] {
      if (op == cudf::unary_operator::NOT) {
        auto type = cudf::data_type{cudf::type_id::BOOL8};
        auto size = input.size();

        return std::make_unique<column>(type,
                                        size,
                                        rmm::device_buffer{size * cudf::size_of(type), 0, mr},
                                        cudf::detail::copy_bitmask(input, stream, mr),
                                        input.null_count());

      } else {
        return cudf::detail::allocate_like(
          input, input.size(), mask_allocation_policy::NEVER, stream, mr);
      }
    }();

    if (input.is_empty()) return output;

    auto output_view = output->mutable_view();

    CUDF_EXPECTS(input.size() > 0, "Launcher requires input size to be non-zero.");
    CUDF_EXPECTS(input.size() == output_view.size(),
                 "Launcher requires input and output size to be equal.");

    if (input.nullable())
      output->set_null_mask(
        rmm::device_buffer{input.null_mask(), bitmask_allocation_size_bytes(input.size())},
        input.null_count());

    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      input.begin<T>(),
                      input.end<T>(),
                      output_view.begin<Tout>(),
                      F{});

    CUDF_CHECK_CUDA(stream.get());

    return output;
  }
};

}  // namespace unary
}  // namespace cudf
