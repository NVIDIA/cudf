/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/protobuf/kernels.cuh"

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <algorithm>

namespace cudf::io::protobuf::detail {

namespace {

CUDF_KERNEL void set_error_if_unset_kernel(int* error_flag, proto_decode_error error)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) { set_error_once(error_flag, error); }
}

// Stub kernels — replaced with real implementations in follow-up PRs.
CUDF_KERNEL void check_required_fields_kernel(field_location const*,
                                              uint8_t const*,
                                              int,
                                              int,
                                              cudf::bitmask_type const*,
                                              cudf::size_type,
                                              field_location const*,
                                              bool*,
                                              int32_t const*,
                                              int*)
{
}

CUDF_KERNEL void validate_enum_values_kernel(int32_t const*, bool*, bool*, int32_t const*, int, int)
{
}

}  // namespace

void set_error_once_async(int* error_flag, proto_decode_error error, rmm::cuda_stream_view stream)
{
  set_error_if_unset_kernel<<<1, 1, 0, stream.value()>>>(error_flag, error);
  CUDF_CUDA_TRY(cudaPeekAtLastError());
}

void maybe_check_required_fields(field_location const* locations,
                                 std::vector<int> const& field_indices,
                                 std::vector<nested_field_descriptor> const& schema,
                                 int num_rows,
                                 cudf::bitmask_type const* input_null_mask,
                                 cudf::size_type input_offset,
                                 field_location const* parent_locs,
                                 bool* row_force_null,
                                 int32_t const* top_row_indices,
                                 int* error_flag,
                                 rmm::cuda_stream_view stream)
{
  if (num_rows == 0 || field_indices.empty()) { return; }

  bool has_required = false;
  auto h_is_required =
    cudf::detail::make_pinned_vector_async<uint8_t>(field_indices.size(), stream);
  for (size_t i = 0; i < field_indices.size(); ++i) {
    h_is_required[i] = schema[field_indices[i]].is_required ? 1 : 0;
    has_required |= (h_is_required[i] != 0);
  }
  if (!has_required) { return; }

  auto d_is_required = cudf::detail::make_device_uvector_async(
    h_is_required, stream, rmm::mr::get_current_device_resource_ref());

  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  check_required_fields_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    locations,
    d_is_required.data(),
    static_cast<int>(field_indices.size()),
    num_rows,
    input_null_mask,
    input_offset,
    parent_locs,
    row_force_null,
    top_row_indices,
    error_flag);
  CUDF_CUDA_TRY(cudaPeekAtLastError());
}

void propagate_invalid_enum_flags_to_rows(rmm::device_uvector<bool> const& item_invalid,
                                          rmm::device_uvector<bool>& row_invalid,
                                          int num_items,
                                          int32_t const* top_row_indices,
                                          bool propagate_to_rows,
                                          rmm::cuda_stream_view stream)
{
  if (num_items == 0 || row_invalid.size() == 0 || !propagate_to_rows) { return; }

  auto const temp_mr = rmm::mr::get_current_device_resource_ref();
  if (top_row_indices == nullptr) {
    CUDF_EXPECTS(static_cast<size_t>(num_items) <= row_invalid.size(),
                 "enum invalid-row propagation exceeded row buffer");
    thrust::transform(rmm::exec_policy_nosync(stream, temp_mr),
                      row_invalid.begin(),
                      row_invalid.begin() + num_items,
                      item_invalid.begin(),
                      row_invalid.begin(),
                      [] __device__(bool row_is_invalid, bool item_is_invalid) {
                        return row_is_invalid || item_is_invalid;
                      });
    return;
  }

  thrust::for_each(rmm::exec_policy_nosync(stream, temp_mr),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(num_items),
                   [item_invalid = item_invalid.data(),
                    top_row_indices,
                    row_invalid = row_invalid.data()] __device__(int idx) {
                     if (item_invalid[idx]) { row_invalid[top_row_indices[idx]] = true; }
                   });
}

void validate_enum_and_propagate_rows(rmm::device_uvector<int32_t> const& values,
                                      rmm::device_uvector<bool>& valid,
                                      decode_protobuf_options::enum_value_vector const& valid_enums,
                                      rmm::device_uvector<bool>& row_invalid,
                                      int num_items,
                                      int32_t const* top_row_indices,
                                      bool propagate_to_rows,
                                      rmm::cuda_stream_view stream)
{
  if (num_items == 0 || valid_enums.empty()) { return; }

  auto const blocks  = static_cast<int>((num_items + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  auto const temp_mr = rmm::mr::get_current_device_resource_ref();
  auto h_valid_enums = cudf::detail::make_pinned_vector_async<int32_t>(valid_enums.size(), stream);
  std::copy(valid_enums.begin(), valid_enums.end(), h_valid_enums.begin());
  auto d_valid_enums = cudf::detail::make_device_uvector_async(h_valid_enums, stream, temp_mr);

  rmm::device_uvector<bool> item_invalid(num_items, stream, temp_mr);
  thrust::fill(
    rmm::exec_policy_nosync(stream, temp_mr), item_invalid.begin(), item_invalid.end(), false);
  validate_enum_values_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    values.data(),
    valid.data(),
    item_invalid.data(),
    d_valid_enums.data(),
    static_cast<int>(valid_enums.size()),
    num_items);
  CUDF_CUDA_TRY(cudaPeekAtLastError());

  propagate_invalid_enum_flags_to_rows(
    item_invalid, row_invalid, num_items, top_row_indices, propagate_to_rows, stream);
}

}  // namespace cudf::io::protobuf::detail
