/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/transform.h>

namespace cudf {
namespace detail {
namespace {

template <typename F>
struct float_pair {
  size_type s;  // index bias for sorting nan
  F f;          // actual float value to sort
};

template <typename F>
struct float_decomposer {
  __device__ cuda::std::tuple<size_type&, F&> operator()(float_pair<F>& key) const
  {
    return {key.s, key.f};
  }
};

template <typename F>
struct float_to_pair_fn {
  F const* fs;
  __device__ float_pair<F> operator()(cudf::size_type idx) const
  {
    auto const f = fs[idx];
    auto const s = (isnan(f) * (idx + 1));  // multiplier helps keep the sort stable for NaNs
    return float_pair<F>{s, f};
  }
};

/**
 * @brief Sorts fixed-width columns using cub radix sort
 *
 * Should not be used if `input.has_nulls()==true`
 */
struct sort_radix_fn {
  column_view const input;     // input column to sort
  mutable_column_view output;  // output sorted
  bool ascending;              // true if ascending sort
  cuda::stream_ref stream;     // stream for allocations and kernel launches

  template <typename T>
  void sort_radix()
  {
    auto d_in          = input.data<T>();
    auto d_out         = output.data<T>();
    auto const end_bit = sizeof(T) * 8;
    auto const sv      = stream.get();
    auto const n       = input.size();
    // cub radix sort implementation is always stable
    std::size_t tmp_bytes = 0;
    if (ascending) {
      cub::DeviceRadixSort::SortKeys(nullptr, tmp_bytes, d_in, d_out, n, 0, end_bit, sv);
      auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
      cub::DeviceRadixSort::SortKeys(tmp_stg.data(), tmp_bytes, d_in, d_out, n, 0, end_bit, sv);
    } else {
      cub::DeviceRadixSort::SortKeysDescending(nullptr, tmp_bytes, d_in, d_out, n, 0, end_bit, sv);
      auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
      cub::DeviceRadixSort::SortKeysDescending(
        tmp_stg.data(), tmp_bytes, d_in, d_out, n, 0, end_bit, sv);
    }
  }

  template <typename T>
  void operator()()
    requires(cudf::is_floating_point<T>())
  {
    auto pair_in  = rmm::device_uvector<float_pair<T>>(input.size(), stream);
    auto d_in     = pair_in.begin();
    auto pair_out = rmm::device_uvector<float_pair<T>>(input.size(), stream);
    auto d_out    = pair_out.begin();

    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{input.size()},
                      d_in,
                      float_to_pair_fn<T>{input.begin<T>()});

    auto const decomposer = float_decomposer<T>{};
    auto const end_bit    = sizeof(float_pair<T>) * 8;
    auto const sv         = stream.get();
    auto const n          = input.size();
    // cub radix sort implementation is always stable
    std::size_t tmp_bytes = 0;
    if (ascending) {
      cub::DeviceRadixSort::SortKeys(
        nullptr, tmp_bytes, d_in, d_out, n, decomposer, 0, end_bit, sv);
      auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
      cub::DeviceRadixSort::SortKeys(
        tmp_stg.data(), tmp_bytes, d_in, d_out, n, decomposer, 0, end_bit, sv);
    } else {
      cub::DeviceRadixSort::SortKeysDescending(
        nullptr, tmp_bytes, d_in, d_out, n, decomposer, 0, end_bit, sv);
      auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
      cub::DeviceRadixSort::SortKeysDescending(
        tmp_stg.data(), tmp_bytes, d_in, d_out, n, decomposer, 0, end_bit, sv);
    }
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      d_out,
                      d_out + input.size(),
                      output.begin<T>(),
                      [] __device__(float_pair<T> const& p) { return p.f; });
  }

  template <typename T>
  void operator()()
    requires(cudf::is_chrono<T>())
  {
    using rep_type = typename T::rep;
    sort_radix<rep_type>();
  }

  template <typename T>
  void operator()()
    requires(cudf::is_fixed_width<T>() and !cudf::is_chrono<T>() and !cudf::is_floating_point<T>())
  {
    sort_radix<T>();
  }

  template <typename T>
  void operator()()
    requires(not cudf::is_fixed_width<T>())
  {
    CUDF_UNREACHABLE("invalid type for faster sort");
  }
};

}  // namespace

bool is_radix_sortable(column_view const& column)
{
  return !column.has_nulls() && cudf::is_fixed_width(column.type());
}

std::unique_ptr<column> sort_radix(column_view const& input,
                                   bool ascending,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  auto result   = std::make_unique<column>(input, stream, mr);
  auto out_view = result->mutable_view();
  cudf::type_dispatcher<dispatch_storage_type>(input.type(),
                                               sort_radix_fn{input, out_view, ascending, stream});
  return result;
}

}  // namespace detail
}  // namespace cudf
