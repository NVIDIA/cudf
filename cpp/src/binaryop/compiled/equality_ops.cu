/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "binary_ops.cuh"

#include <cuda/iterator>

namespace cudf::binops::compiled {
void dispatch_equality_op(mutable_column_view& out,
                          column_view const& lhs,
                          column_view const& rhs,
                          bool is_lhs_scalar,
                          bool is_rhs_scalar,
                          binary_operator op,
                          cuda::stream_ref stream)
{
  CUDF_EXPECTS(op == binary_operator::EQUAL || op == binary_operator::NOT_EQUAL,
               "Unsupported operator for these types",
               cudf::data_type_error);
  auto common_dtype = get_common_type(out.type(), lhs.type(), rhs.type());
  auto outd         = mutable_column_device_view::create(out, stream);
  auto lhsd         = column_device_view::create(lhs, stream);
  auto rhsd         = column_device_view::create(rhs, stream);
  if (common_dtype) {
    if (op == binary_operator::EQUAL) {
      thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         cuda::counting_iterator<size_type>{0},
                         out.size(),
                         binary_op_device_dispatcher<ops::Equal>{
                           *common_dtype, *outd, *lhsd, *rhsd, is_lhs_scalar, is_rhs_scalar});
    } else if (op == binary_operator::NOT_EQUAL) {
      thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         cuda::counting_iterator<size_type>{0},
                         out.size(),
                         binary_op_device_dispatcher<ops::NotEqual>{
                           *common_dtype, *outd, *lhsd, *rhsd, is_lhs_scalar, is_rhs_scalar});
    }
  } else {
    if (op == binary_operator::EQUAL) {
      thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         cuda::counting_iterator<size_type>{0},
                         out.size(),
                         binary_op_double_device_dispatcher<ops::Equal>{
                           *outd, *lhsd, *rhsd, is_lhs_scalar, is_rhs_scalar});
    } else if (op == binary_operator::NOT_EQUAL) {
      thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         cuda::counting_iterator<size_type>{0},
                         out.size(),
                         binary_op_double_device_dispatcher<ops::NotEqual>{
                           *outd, *lhsd, *rhsd, is_lhs_scalar, is_rhs_scalar});
    }
  }
}

}  // namespace cudf::binops::compiled
