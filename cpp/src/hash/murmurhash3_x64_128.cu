/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/accumulate.cuh>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/hashing/detail/murmurhash3_x64_128.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/array>
#include <cuda/std/limits>
#include <thrust/for_each.h>

namespace cudf {
namespace hashing {
namespace detail {
namespace {

using hash_value_type = cuda::std::array<uint64_t, 2>;

/**
 * @brief Computes the hash value of a row in the given table.
 *
 * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
 */
template <typename Nullate>
class murmur_device_row_hasher {
 public:
  murmur_device_row_hasher(Nullate nulls,
                           table_device_view const& t,
                           uint64_t seed,
                           uint64_t* d_output1,
                           uint64_t* d_output2)
    : _check_nulls(nulls), _input(t), _seed(seed), _output1(d_output1), _output2(d_output2)
  {
  }

  /**
   * @brief Return the hash value of a row in the given table.
   *
   * @param row_index The row index to compute the hash value of
   * @return The hash value of the row
   */
  __device__ void operator()(size_type row_index) const noexcept
  {
    auto const h = cudf::detail::accumulate(
      _input.begin(),
      _input.end(),
      hash_value_type{_seed, 0},
      [row_index, nulls = this->_check_nulls] __device__(auto hash, auto column) {
        return cudf::type_dispatcher(
          column.type(), element_hasher_adapter{}, column, row_index, nulls, hash);
      });
    _output1[row_index] = h[0];
    _output2[row_index] = h[1];
  }

  /**
   * @brief Computes the hash value of an element in the given column.
   */
  class element_hasher_adapter {
   public:
    template <typename T, CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
    __device__ hash_value_type operator()(column_device_view const& col,
                                          size_type row_index,
                                          Nullate const check_nulls,
                                          hash_value_type const seed) const noexcept
    {
      if (check_nulls && col.is_null(row_index)) {
        return {cuda::std::numeric_limits<uint64_t>::max(),
                cuda::std::numeric_limits<uint64_t>::max()};
      }
      auto const hasher = MurmurHash3_x64_128<T>{seed[0]};
      return hasher(col.element<T>(row_index));
    }

    template <typename T, CUDF_ENABLE_IF(not column_device_view::has_element_accessor<T>())>
    __device__ hash_value_type operator()(column_device_view const&,
                                          size_type,
                                          Nullate const,
                                          hash_value_type const) const noexcept
    {
      CUDF_UNREACHABLE("Unsupported type for MurmurHash3_x64_128");
    }
  };

  Nullate const _check_nulls;
  table_device_view const _input;
  uint64_t const _seed;
  uint64_t* _output1;
  uint64_t* _output2;
};

}  // namespace

std::unique_ptr<table> murmurhash3_x64_128(table_view const& input,
                                           uint64_t seed,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  auto output1 = make_numeric_column(
    data_type(type_id::UINT64), input.num_rows(), mask_state::UNALLOCATED, stream, mr);
  auto output2 = make_numeric_column(
    data_type(type_id::UINT64), input.num_rows(), mask_state::UNALLOCATED, stream, mr);

  if (input.num_rows() != 0) {
    bool const nullable   = has_nulls(input);
    auto const input_view = table_device_view::create(input, stream);
    auto d_output1        = output1->mutable_view().data<uint64_t>();
    auto d_output2        = output2->mutable_view().data<uint64_t>();

    // Compute the hash value for each row
    thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       cuda::counting_iterator<size_type>{0},
                       input.num_rows(),
                       murmur_device_row_hasher(nullable, *input_view, seed, d_output1, d_output2));
  }

  std::vector<std::unique_ptr<column>> out_columns(2);
  out_columns.front() = std::move(output1);
  out_columns.back()  = std::move(output2);
  return std::make_unique<table>(std::move(out_columns));
}

}  // namespace detail

std::unique_ptr<table> murmurhash3_x64_128(table_view const& input,
                                           uint64_t seed,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::murmurhash3_x64_128(input, seed, stream, mr);
}

}  // namespace hashing
}  // namespace cudf
