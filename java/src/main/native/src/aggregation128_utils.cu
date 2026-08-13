/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "aggregation128_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/iterator/transform_iterator.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace {

// Functor to reassemble a 128-bit value from four 64-bit chunks with overflow detection.
class chunk_assembler {
 public:
  chunk_assembler(bool* overflows,
                  uint64_t const* chunks0,
                  uint64_t const* chunks1,
                  uint64_t const* chunks2,
                  int64_t const* chunks3)
    : overflows(overflows), chunks0(chunks0), chunks1(chunks1), chunks2(chunks2), chunks3(chunks3)
  {
  }

  __device__ __int128_t operator()(cudf::size_type i) const
  {
    // Starting with the least significant input and moving to the most significant, propagate the
    // upper 32-bits of the previous column into the next column, i.e.: propagate the "carry" bits
    // of each 64-bit chunk into the next chunk.
    uint64_t const c0      = chunks0[i];
    uint64_t const c1      = chunks1[i] + (c0 >> 32);
    uint64_t const c2      = chunks2[i] + (c1 >> 32);
    int64_t const c3       = chunks3[i] + (c2 >> 32);
    uint64_t const lower64 = (c1 << 32) | static_cast<uint32_t>(c0);
    int64_t const upper64  = (c3 << 32) | static_cast<uint32_t>(c2);

    // check for overflow by ensuring the sign bit matches the top carry bits
    int32_t const replicated_sign_bit = static_cast<int32_t>(c3) >> 31;
    int32_t const top_carry_bits      = static_cast<int32_t>(c3 >> 32);
    overflows[i]                      = (replicated_sign_bit != top_carry_bits);

    return (static_cast<__int128_t>(upper64) << 64) | lower64;
  }

 private:
  // output column for overflow detected
  bool* const overflows;

  // input columns for the four 64-bit values
  uint64_t const* const chunks0;
  uint64_t const* const chunks1;
  uint64_t const* const chunks2;
  int64_t const* const chunks3;
};

}  // anonymous namespace

namespace cudf::jni {

// Extract a 32-bit chunk from a 128-bit value.
std::unique_ptr<cudf::column> extract_chunk32(cudf::column_view const& in_col,
                                              cudf::data_type type,
                                              int chunk_idx,
                                              rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(in_col.type().id() == cudf::type_id::DECIMAL128, "not a 128-bit type");
  CUDF_EXPECTS(chunk_idx >= 0 && chunk_idx < 4, "invalid chunk index");
  CUDF_EXPECTS(type.id() == cudf::type_id::INT32 || type.id() == cudf::type_id::UINT32,
               "not a 32-bit integer type");
  auto const num_rows = in_col.size();
  auto out_col =
    cudf::make_fixed_width_column(type, num_rows, copy_bitmask(in_col), in_col.null_count());
  auto out_view       = out_col->mutable_view();
  auto const in_begin = in_col.begin<int32_t>();

  // Build an iterator for every fourth 32-bit value, i.e.: one "chunk" of a __int128_t value
  thrust::transform_iterator transform_iter{
    cuda::counting_iterator{cudf::size_type{0}},
    cuda::proclaim_return_type<cudf::size_type>([] __device__(auto i) { return i * 4; })};
  cuda::permutation_iterator stride_iter{in_begin + chunk_idx, transform_iter};

  thrust::copy(
    rmm::exec_policy_nosync(stream), stride_iter, stride_iter + num_rows, out_view.data<int32_t>());
  return out_col;
}

// Reassemble a column of 128-bit values from four 64-bit integer columns with overflow detection.
std::unique_ptr<cudf::table> assemble128_from_sum(cudf::table_view const& chunks_table,
                                                  cudf::data_type output_type,
                                                  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(output_type.id() == cudf::type_id::DECIMAL128, "not a 128-bit type");
  CUDF_EXPECTS(chunks_table.num_columns() == 4, "must be 4 column table");
  auto const num_rows = chunks_table.num_rows();
  auto const chunks0  = chunks_table.column(0);
  auto const chunks1  = chunks_table.column(1);
  auto const chunks2  = chunks_table.column(2);
  auto const chunks3  = chunks_table.column(3);
  CUDF_EXPECTS(cudf::size_of(chunks0.type()) == 8 && cudf::size_of(chunks1.type()) == 8 &&
                 cudf::size_of(chunks2.type()) == 8 && chunks3.type().id() == cudf::type_id::INT64,
               "chunks type mismatch");
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::BOOL8}, num_rows, copy_bitmask(chunks0), chunks0.null_count()));
  columns.push_back(cudf::make_fixed_width_column(
    output_type, num_rows, copy_bitmask(chunks0), chunks0.null_count()));
  auto overflows_view = columns[0]->mutable_view();
  auto assembled_view = columns[1]->mutable_view();
  thrust::transform(rmm::exec_policy_nosync(stream),
                    cuda::counting_iterator{cudf::size_type{0}},
                    cuda::counting_iterator{cudf::size_type{num_rows}},
                    assembled_view.begin<__int128_t>(),
                    chunk_assembler(overflows_view.begin<bool>(),
                                    chunks0.begin<uint64_t>(),
                                    chunks1.begin<uint64_t>(),
                                    chunks2.begin<uint64_t>(),
                                    chunks3.begin<int64_t>()));
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace cudf::jni
