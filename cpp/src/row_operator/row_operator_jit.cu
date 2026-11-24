/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/cache.hpp"
#include "reflection.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_child_offsets.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/hashing.hpp>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>

#include <cudf_fragments.hpp>
#include <rtcx/rtcx.hpp>

#include <cstdint>
#include <format>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudf::row_operators {
namespace {

std::span<uint8_t const> get_hash_fragment(algorithm alg)
{
  switch (alg) {
    case algorithm::xxhash_32:
      return cudf_fragments::get(cudf_fragments::row_operator_xxhash_32_lto_ir);
    case algorithm::xxhash_64:
      return cudf_fragments::get(cudf_fragments::row_operator_xxhash_64_lto_ir);
    case algorithm::murmurhash3_x86_32:
      return cudf_fragments::get(cudf_fragments::row_operator_murmurhash3_x86_32_lto_ir);
    case algorithm::murmurhash3_x64_128:
      return cudf_fragments::get(cudf_fragments::row_operator_murmurhash3_x64_128_lto_ir);
    case algorithm::spark_murmurhash3_x86_32:
      return cudf_fragments::get(cudf_fragments::row_operator_spark_murmurhash3_x86_32_lto_ir);
    default:
      CUDF_FAIL(std::format("Unsupported hash algorithm: {}", static_cast<int>(alg)),
                std::invalid_argument);
  }
}

kernel get_kernel(algorithm alg, std::string const& instance)
{
  auto const name   = std::string{to_string(alg)};
  auto const source = std::string{"cudf/cpp/src/row_operator/jit/hash_kernel.cu"};

  auto const header = std::format(R"CUDA(
#pragma once
#define CUDF_KERNEL_INSTANCE {}
)CUDA",
                                  instance);
  char const* include_names[]{"cudf/detail/kernel_instance.cuh"};
  char const* headers[]{header.c_str()};

  auto const kernel_ir = get_kernel_fragment(name, source, include_names, headers, instance);

  rtcx::memory_fragment fragments[] = {
    {.data = kernel_ir->view(), .type = rtcx::binary_type::LTO_IR},
    {.data = get_hash_fragment(alg), .type = rtcx::binary_type::FATBIN},
    {.data = cudf_fragments::get(cudf_fragments::row_operator_hash_combine_lto_ir),
     .type = rtcx::binary_type::FATBIN}};

  return get_lto_linked_kernel(name, {}, fragments);
}

void launch(kernel const& operation,
            column_device_view const* columns,
            size_type num_rows,
            uint64_t seed,
            bool nullable,
            void* low,
            void* high,
            cuda::stream_ref stream)
{
  auto const config = operation.max_occupancy_config(0, 0);

  operation.launch_with({config.min_grid_size},
                        {config.block_size},
                        0,
                        stream,
                        columns,
                        num_rows,
                        seed,
                        nullable,
                        low,
                        high);
}

void check_spark_type(column_view const& column)
{
  using enum type_id;

  switch (column.type().id()) {
    case DICTIONARY32: check_spark_type(column.child(dictionary_keys_column_index)); break;
    case LIST:
      CUDF_EXPECTS(
        column.child(lists_column_view::child_column_index).type().id() != type_id::STRUCT,
        "Cannot compute hash of a table with a LIST of STRUCT columns.");
      check_spark_type(column.child(lists_column_view::child_column_index));
      break;
    case STRUCT:
      for (auto child = column.child_begin(); child != column.child_end(); ++child) {
        check_spark_type(*child);
      }
      break;
    case TIMESTAMP_SECONDS:
    case TIMESTAMP_MILLISECONDS:
    case TIMESTAMP_NANOSECONDS:
    case DURATION_DAYS:
    case DURATION_SECONDS:
    case DURATION_MILLISECONDS:
    case DURATION_NANOSECONDS:
      CUDF_FAIL(
        "Spark MurmurHash3 requires dates in days and timestamps and durations in "
        "microseconds.");
    default: break;
  }
}

std::vector<std::unique_ptr<column>> hash_column(algorithm alg,
                                                 table_view const& input,
                                                 type_id low,
                                                 std::optional<type_id> high,
                                                 uint64_t seed,
                                                 cuda::stream_ref stream,
                                                 cudf::memory_resources mr)
{
  if (alg == algorithm::spark_murmurhash3_x86_32 && input.num_rows() != 0) {
    for (auto const& column : input) {
      check_spark_type(column);
    }
  }

  std::vector<std::unique_ptr<column>> outputs;

  outputs.push_back(
    make_numeric_column(data_type{low}, input.num_rows(), mask_state::UNALLOCATED, stream, mr));
  if (high.has_value()) {
    outputs.push_back(
      make_numeric_column(data_type{*high}, input.num_rows(), mask_state::UNALLOCATED, stream, mr));
  }

  if (input.num_rows() == 0) { return outputs; }

  auto const d_low  = outputs[0]->mutable_view().head();
  auto const d_high = high.has_value() ? outputs[1]->mutable_view().head() : nullptr;

  if (input.num_columns() == 0) {
    auto policy = rmm::exec_policy_nosync(stream, mr.get_temporary_mr());
    // A row with no columns hashes to the seed; its second 128-bit result word is zero.
    if (alg == algorithm::xxhash_64 || alg == algorithm::murmurhash3_x64_128) {
      thrust::fill_n(policy, static_cast<uint64_t*>(d_low), input.num_rows(), seed);
    } else {
      thrust::fill_n(
        policy, static_cast<uint32_t*>(d_low), input.num_rows(), static_cast<uint32_t>(seed));
    }

    if (high.has_value()) {
      CUDF_CUDA_TRY(cudaMemsetAsync(d_high, 0, input.num_rows() * sizeof(uint64_t), stream.get()));
    }

    return outputs;
  }

  auto const instance     = reflect_hash_kernel(alg, input);
  auto const operation    = get_kernel(alg, instance);
  auto [storage, columns] = create_column_device_views<column_device_view>(input, stream, mr);
  auto const nullable =
    alg == algorithm::spark_murmurhash3_x86_32 ? has_nested_nulls(input) : has_nulls(input);

  launch(operation, columns, input.num_rows(), seed, nullable, d_low, d_high, stream);

  return outputs;
}

}  // namespace
}  // namespace cudf::row_operators

namespace cudf::hashing::detail {

std::unique_ptr<column> murmurhash3_x86_32_jit(table_view const& input,
                                               uint32_t seed,
                                               cuda::stream_ref stream,
                                               cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  auto outputs = row_operators::hash_column(row_operators::algorithm::murmurhash3_x86_32,
                                            input,
                                            type_id::UINT32,
                                            std::nullopt,
                                            seed,
                                            stream,
                                            mr);
  return std::move(outputs.front());
}

std::unique_ptr<column> spark_murmurhash3_x86_32_jit(table_view const& input,
                                                     uint32_t seed,
                                                     cuda::stream_ref stream,
                                                     cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  auto outputs = row_operators::hash_column(row_operators::algorithm::spark_murmurhash3_x86_32,
                                            input,
                                            type_id::INT32,
                                            std::nullopt,
                                            seed,
                                            stream,
                                            mr);
  return std::move(outputs.front());
}

std::unique_ptr<table> murmurhash3_x64_128_jit(table_view const& input,
                                               uint64_t seed,
                                               cuda::stream_ref stream,
                                               cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return std::make_unique<table>(
    row_operators::hash_column(row_operators::algorithm::murmurhash3_x64_128,
                               input,
                               type_id::UINT64,
                               type_id::UINT64,
                               seed,
                               stream,
                               mr));
}

std::unique_ptr<column> xxhash_32_jit(table_view const& input,
                                      uint32_t seed,
                                      cuda::stream_ref stream,
                                      cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  auto outputs = row_operators::hash_column(
    row_operators::algorithm::xxhash_32, input, type_id::UINT32, std::nullopt, seed, stream, mr);
  return std::move(outputs.front());
}

std::unique_ptr<column> xxhash_64_jit(table_view const& input,
                                      uint64_t seed,
                                      cuda::stream_ref stream,
                                      cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  auto outputs = row_operators::hash_column(
    row_operators::algorithm::xxhash_64, input, type_id::UINT64, std::nullopt, seed, stream, mr);
  return std::move(outputs.front());
}

}  // namespace cudf::hashing::detail
