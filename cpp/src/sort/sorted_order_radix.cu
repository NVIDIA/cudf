/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sort_radix.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cuda/iterator>
#include <thrust/sequence.h>
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
struct float_to_pair_and_seq {
  F const* fs;
  __device__ cuda::std::pair<float_pair<F>, size_type> operator()(cudf::size_type idx) const
  {
    auto const f = fs[idx];
    auto const s = (isnan(f) * (idx + 1));  // multiplier helps keep the sort stable for NaNs
    return {float_pair<F>{s, f}, idx};
  }
};

/**
 * @brief Sorts fixed-width columns using faster thrust sort
 *
 * Should not be called if `input.has_nulls()==true`
 */
struct sorted_order_radix_fn {
  column_view const& input;      // keys to sort
  mutable_column_view& indices;  // output of sort
  bool ascending;                // true for ascending sort
  rmm::cuda_stream_view stream;  // for allocation and kernel launches

  template <typename T>
  void radix_sort()
  {
    auto d_in   = input.begin<T>();
    auto output = rmm::device_uvector<T>(input.size(), stream);
    auto d_out  = output.begin();  // not returned
    auto seqs   = rmm::device_uvector<cudf::size_type>(input.size(), stream);
    thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     seqs.begin(),
                     seqs.end(),
                     0);
    auto dv_in  = seqs.begin();
    auto dv_out = indices.begin<cudf::size_type>();

    auto const n       = input.size();
    auto const sv      = stream.value();
    auto const end_bit = sizeof(T) * 8;

    // cub radix sort implementation is always stable
    std::size_t tmp_bytes = 0;
    if (ascending) {
      cub::DeviceRadixSort::SortPairs(
        nullptr, tmp_bytes, d_in, d_out, dv_in, dv_out, n, 0, end_bit, sv);
      auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
      cub::DeviceRadixSort::SortPairs(
        tmp_stg.data(), tmp_bytes, d_in, d_out, dv_in, dv_out, n, 0, end_bit, sv);
    } else {
      cub::DeviceRadixSort::SortPairsDescending(
        nullptr, tmp_bytes, d_in, d_out, dv_in, dv_out, n, 0, end_bit, sv);
      auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
      cub::DeviceRadixSort::SortPairsDescending(
        tmp_stg.data(), tmp_bytes, d_in, d_out, dv_in, dv_out, n, 0, end_bit, sv);
    }
  }

  template <typename T>
  void operator()()
    requires(cudf::is_floating_point<T>())
  {
    auto pair_in = rmm::device_uvector<float_pair<T>>(input.size(), stream);
    auto d_in    = pair_in.begin();
    // pair_out/d_out is not returned to the caller but used as an intermediate
    auto pair_out = rmm::device_uvector<float_pair<T>>(input.size(), stream);
    auto d_out    = pair_out.begin();
    auto vals     = rmm::device_uvector<size_type>(indices.size(), stream);
    auto dv_in    = vals.begin();
    auto dv_out   = indices.begin<cudf::size_type>();

    auto zip_out = cuda::make_zip_iterator(d_in, dv_in);
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{input.size()},
                      zip_out,
                      float_to_pair_and_seq<T>{input.begin<T>()});

    auto const decomposer = float_decomposer<T>{};
    auto const end_bit    = sizeof(float_pair<T>) * 8;
    auto const sv         = stream.value();
    auto const n          = input.size();
    // cub radix sort implementation is always stable
    std::size_t tmp_bytes = 0;
    if (ascending) {
      cub::DeviceRadixSort::SortPairs(
        nullptr, tmp_bytes, d_in, d_out, dv_in, dv_out, n, decomposer, 0, end_bit, sv);
      auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
      cub::DeviceRadixSort::SortPairs(
        tmp_stg.data(), tmp_bytes, d_in, d_out, dv_in, dv_out, n, decomposer, 0, end_bit, sv);
    } else {
      cub::DeviceRadixSort::SortPairsDescending(
        nullptr, tmp_bytes, d_in, d_out, dv_in, dv_out, n, decomposer, 0, end_bit, sv);
      auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
      cub::DeviceRadixSort::SortPairsDescending(
        tmp_stg.data(), tmp_bytes, d_in, d_out, dv_in, dv_out, n, decomposer, 0, end_bit, sv);
    }
  }

  template <typename T>
  void operator()()
    requires(cudf::is_chrono<T>())
  {
    using rep_type = typename T::rep;
    radix_sort<rep_type>();
  }

  template <typename T>
  void operator()()
    requires(cudf::is_fixed_width<T>() and !cudf::is_chrono<T>() and !cudf::is_floating_point<T>())
  {
    radix_sort<T>();
  }

  template <typename T>
  void operator()()
    requires(not cudf::is_fixed_width<T>())
  {
    CUDF_UNREACHABLE("invalid type for radix sort");
  }
};
}  // namespace

/**
 * @brief Sort indices of a single column.
 *
 * This API offers fast sorting for most primitive types.
 *
 * @param input Column to sort
 * @param indices The result of the sort
 * @param ascending Sort order
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void sorted_order_radix(column_view const& input,
                        mutable_column_view& indices,
                        bool ascending,
                        rmm::cuda_stream_view stream)
{
  cudf::type_dispatcher<dispatch_storage_type>(
    input.type(), sorted_order_radix_fn{input, indices, ascending, stream});
}
}  // namespace detail
}  // namespace cudf
