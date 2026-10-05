/*
 * SPDX-FileCopyrightText: Copyright 2018-2019 BlazingDB, Inc.
 * SPDX-FileCopyrightText: Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
 * SPDX-FileCopyrightText: Copyright 2018 Rommel Quintanilla <rommel@blazingdb.com>
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
 *     Copyright 2018 Rommel Quintanilla <rommel@blazingdb.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/type_traits>

#pragma nv_hdrstop  // The above headers are used by the kernel below and need to be included before
                    // it. Each UDF will have a different operation_udf.cuh generated for it, so we
                    // need to put this pragma before including it to avoid PCH mismatch.

// clang-format off
#include <cudf/detail/kernel_instance.cuh>
#include <cudf/detail/operation_udf.cuh>
// clang-format on

namespace cudf {
namespace binops {
namespace jit {

struct UserDefinedOp {
  template <typename TypeOut, typename TypeLhs, typename TypeRhs>
  static TypeOut operate(TypeLhs x, TypeRhs y)
  {
    TypeOut output;
    using TypeCommon = typename cuda::std::common_type<TypeOut, TypeLhs, TypeRhs>::type;
    GENERIC_BINARY_OP(&output, static_cast<TypeCommon>(x), static_cast<TypeCommon>(y));
    return output;
  }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs, typename TypeOpe>
__device__ void binary_op_kernel(cudf::size_type size,
                                 void* p_out_data,
                                 void* p_lhs_data,
                                 void* p_rhs_data)
{
  auto const start = cudf::detail::grid_1d::global_thread_id();
  auto const step  = cudf::detail::grid_1d::grid_stride();

  auto* out_data = static_cast<TypeOut*>(p_out_data);
  auto* lhs_data = static_cast<TypeLhs*>(p_lhs_data);
  auto* rhs_data = static_cast<TypeRhs*>(p_rhs_data);

  for (auto i = start; i < size; i += step) {
    out_data[i] = TypeOpe::template operate<TypeOut, TypeLhs, TypeRhs>(lhs_data[i], rhs_data[i]);
  }
}

}  // namespace jit
}  // namespace binops
}  // namespace cudf

extern "C" __global__ void cudf_kernel_entry(cudf::size_type size,
                                             void* out_data,
                                             void* lhs_data,
                                             void* rhs_data)
{ CUDF_KERNEL_INSTANCE(size, out_data, lhs_data, rhs_data); }
